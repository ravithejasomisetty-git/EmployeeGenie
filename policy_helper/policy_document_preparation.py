"""Prepare policy JSON for search without calling OpenAI or Pinecone.

Read this module in order: schema classes -> load_documents -> clean_content ->
prepare -> catalog. A document is a full source record; a chunk is a smaller
passage that can be retrieved independently. Metadata describes a passage so
Pinecone can restrict a search to categories such as HR or onboarding.

Python reading guide: a dict stores named values, a list stores ordered values,
and a set stores unique values. Type annotations such as list[str] describe
expected data. Pydantic performs the actual input validation at runtime.
"""
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Literal

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Literal restricts this field to the exact labels already used in the dataset.
Category = Literal['hr', 'compliance', 'product', 'onboarding']
# Increment this when preparation behavior changes so new data gets a new namespace.
PIPELINE_VERSION = '1'


class Metadata(BaseModel):
    """Required descriptive fields on every source document.

    BaseModel is Pydantic's schema class. strict=True rejects incorrect types
    rather than converting them; extra='forbid' catches unexpected field names.
    Empty lists are valid: an absent country must not be mistaken for 'global'.
    """
    model_config = ConfigDict(strict=True, extra='forbid')
    title: str = Field(min_length=1)
    category: Category
    subcategory: str = Field(min_length=1)
    topics: list[str]
    document_type: str
    audience_roles: list[str]
    countries: list[str]
    applicability_scope: str
    onboarding_stage: list[str]


class Document(BaseModel):
    """One JSON record: a stable ID, descriptive metadata, and policy text."""
    model_config = ConfigDict(strict=True, extra='forbid')
    id: str = Field(min_length=1, max_length=350)
    metadata: Metadata
    content: str = Field(min_length=1)

    @field_validator('id', 'content')
    @classmethod
    def nonblank(cls, value):
        """Reject strings made only of whitespace, even if their length is positive.

        The decorators make Pydantic run this rule on both fields automatically.
        Returning the original value preserves the source text exactly.

        Used by:
            Called automatically by Pydantic when Document fields are validated, normally through load_documents().
        """
        if not value.strip():
            raise ValueError('must not be blank')
        return value


def load_documents(path: Path) -> list[Document]:
    """Return validated document objects, or raise one error listing bad records.

    Validation finishes for the entire file before ingestion can write anything.
    utf-8-sig also accepts UTF-8 files with a byte-order marker from Windows tools.

    Used by:
        ingest_policies.py: main(), before preparing or uploading documents.
        Direct test/fixture callers in tests/test_policy_pipeline.py: prepared(), test_invalid_dataset_fails().
    """
    raw = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(raw, list) or not raw:
        raise ValueError('Dataset must be a nonempty JSON array')
    # Track duplicate IDs across records; schema validation checks one record at a time.
    documents, errors, ids = [], [], set()
    for position, record in enumerate(raw):
        try:
            # Convert a plain JSON dictionary into an object with checked fields.
            doc = Document.model_validate(record)
            if doc.id in ids:
                raise ValueError('duplicate document ID')
            ids.add(doc.id)
            documents.append(doc)
        except ValueError as exc:
            identifier = record.get('id', position) if isinstance(record, dict) else position
            errors.append(f'{identifier}: {exc}')
    if errors:
        raise ValueError('Dataset validation failed:\n' + '\n'.join(errors))
    return documents


def clean_content(text: str) -> str:
    """Remove display markup while retaining readable policy wording.

    re.sub means 'replace matches of this regular-expression pattern'. re.S lets
    a match span lines; re.I ignores letter case. This is cleanup for this dataset,
    not a general-purpose HTML parser or a guarantee against prompt injection.
    The original JSON is never modified.

    Used by:
        This module: prepare(), once for each source document.
        Direct test/fixture callers in tests/test_policy_pipeline.py: test_cleaning_preserves_warning().
    """
    # Frontmatter is the introductory --- block containing page publishing fields.
    text = re.sub(r'\A---\s*\n.*?\n---\s*\n', '', text, count=1, flags=re.S)
    # Remove static-site template markers, leaving text between opening/closing markers.
    text = re.sub(r'\{\{[%<].*?[%>]\}\}', '', text, flags=re.S)
    # Scripts/styles are not policy prose; discard their contents as well as their tags.
    text = re.sub(r'<(?:script|style)\b[^>]*>.*?</(?:script|style)>', '', text, flags=re.S | re.I)
    # Preserve visual breaks before removing remaining HTML tags.
    text = re.sub(r'<(?:br\s*/?|/?(?:p|div|summary|details))\b[^>]*>', '\n', text, flags=re.I)
    text = re.sub(r'</?[A-Za-z][^>]*>', '', text)
    # Decode entities such as &amp; and collapse excessive blank lines.
    return re.sub(r'\n{3,}', '\n\n', html.unescape(text)).strip()


def prepare(documents: list[Document], source: str, model: str, size=600, overlap=100, dimensions=512):
    """Return (chunks, namespace, settings) for a validated dataset.

    A token is a model-sized piece of text, not necessarily a whole word. size
    includes the title/heading prefix; overlap repeats nearby context within a
    section so a boundary is less likely to separate a rule from its qualification.
    Sections are kept separate, so short sections can produce small chunks.

    Each chunk has an ASCII ID, filterable metadata, and embedding_text. The
    embedding input includes extra context; metadata['text'] holds the passage
    shown to the answer model. The namespace fingerprints the prepared data and
    settings, allowing identical runs to target the same logical partition.

    Used by:
        ingest_policies.py: main(), in both dry-run and live ingestion modes.
        Direct test/fixture callers in tests/test_policy_pipeline.py: prepared(), test_dimensions_change_namespace(), test_real_dataset_and_chunks().
    """
    if not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < chunk size')
    # This tokenizer is used for our embedding-input budget. Its public vocabulary
    # table may be downloaded and cached on the first run.
    encoder = tiktoken.get_encoding('cl100k_base')
    chunks = []
    for doc in documents:
        cleaned = clean_content(doc.content)
        if not cleaned:
            raise ValueError(f'{doc.id}: no content remains after cleaning')
        # Collect text under each Markdown heading (# through ######).
        sections, heading, lines = [], '', []
        for line in cleaned.splitlines():
            if re.match(r'^#{1,6}\s+', line):
                if lines:
                    sections.append((heading, '\n'.join(lines)))
                heading, lines = re.sub(r'^#+\s+', '', line), [line]
            else:
                lines.append(line)
        # Flush the final section: no later heading exists to trigger its save.
        if lines:
            sections.append((heading, '\n'.join(lines)))
        for heading, section in sections:
            prefix = f'{doc.metadata.title}\n{heading}\n\n'
            # Reserve space for contextual labels before allocating passage tokens.
            budget = size - len(encoder.encode(prefix))
            if budget <= overlap:
                raise ValueError(f'{doc.id}: title/heading too long for chunk budget')
            # Accumulate paragraphs, splitting oversized paragraphs by token windows.
            windows, current = [], ''
            for paragraph in re.split(r'\n\s*\n', section):
                candidate = current + ('\n\n' if current else '') + paragraph
                if len(encoder.encode(candidate)) <= budget:
                    current = candidate
                    continue
                if current:
                    windows.append(current)
                    # [-overlap:] selects the last N tokens; reuse them in the next chunk.
                    tail = encoder.encode(current)[-overlap:] if overlap else []
                    current = encoder.decode(tail)
                tokens = encoder.encode(current + ('\n\n' if current else '') + paragraph)
                while len(tokens) > budget:
                    windows.append(encoder.decode(tokens[:budget]))
                    # Advance by less than a full window to retain the requested overlap.
                    tokens = tokens[budget-overlap:]
                current = encoder.decode(tokens)
            if current.strip():
                windows.append(current)
            for text in windows:
                # Chunk positions restart for each document, not for each section.
                number = sum(c['metadata']['document_id'] == doc.id for c in chunks)
                # model_dump converts the checked schema back to a plain dictionary.
                # Flatten its fields for Pinecone; omit empty optional arrays.
                metadata = {k: v for k, v in doc.metadata.model_dump().items() if v != []}
                metadata.update(document_id=doc.id, chunk_index=number,
                                section_heading=heading, source_file=source, text=text)
                # Hashing makes IDs ASCII-safe and repeatable. :04d pads positions
                # with zeros, for example 0003. This is identity, not encryption.
                identifier = hashlib.sha256(doc.id.encode()).hexdigest()[:24] + f'-{number:04d}'
                chunks.append({'id': identifier, 'metadata': metadata, 'embedding_text': prefix + text})
    # Dimensions belong in the fingerprint: vectors of different lengths must not mix.
    settings = {'pipeline': PIPELINE_VERSION, 'model': model, 'dimensions': dimensions,
                'size': size, 'overlap': overlap}
    # A stable JSON representation produces a stable fingerprint. Source filename
    # is part of chunk metadata, so changing it also changes this namespace.
    payload = json.dumps({'settings': settings, 'chunks': chunks}, sort_keys=True, ensure_ascii=False)
    namespace = 'policies-' + hashlib.sha256(payload.encode()).hexdigest()[:20]
    return chunks, namespace, settings


def catalog(documents):
    """Return compact topic descriptions for routing, without full document bodies.

    The classifier needs to know what can be searched. The answer stage receives
    actual retrieved passages later; catalog titles alone are not answer evidence.

    Used by:
        ingest_policies.py: main(), when building the ingestion report used by the query classifier.
    """
    return [{'category': d.metadata.category, 'subcategory': d.metadata.subcategory,
             'title': d.metadata.title, 'topics': d.metadata.topics} for d in documents]
