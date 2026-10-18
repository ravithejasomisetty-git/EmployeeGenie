"""Standalone command-line entry point for ETL (extract, transform, load).

Extract: read and validate policy JSON. Transform: clean text, split passages,
and generate embeddings through LangChain. Load: PineconeVectorStore.add_texts()
upserts vectors and metadata. PreparedBatchEmbeddings preserves contextual input.
Finally, verify the upload and save a report that the chat application checks.

Run `uv run --locked python ingest_policies.py --dry-run` to validate/chunk without service credentials
or OpenAI/Pinecone calls. First-use tokenization may download a public encoding
table. Run `uv run --locked python ingest_policies.py` for real ingestion, which uses hosted services.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
import time

import tiktoken

from policy_helper.policy_document_preparation import catalog, load_documents, prepare
from policy_helper.ai_service_clients import (
    Settings, clients, embed, retry, vector_store, PreparedBatchEmbeddings,
)


def chunk_statistics(chunks):
    """Measure the actual prepared chunks, rather than estimate tokens from words.

    Embedding input includes the title and heading prefix; passage text does not.
    Report both so the configured ceiling and the readable content are distinct.
    Word counts use whitespace splitting and are descriptive, not sizing rules.
    Totals include repeated overlap and prefixes, not just unique document text.

    Used by:
        This file: main(), while building the preparation summary for both dry runs and live ingestion.
        Direct test/fixture callers in tests/test_policy_pipeline.py: test_chunk_statistics_measures_embedding_prefix_separately().
    """
    encoder = tiktoken.get_encoding('cl100k_base')

    def describe(counts):
        """Summarize one measurement across all chunks, including an empty input.

        Used by:
            Its enclosing chunk_statistics() function, for embedding-token, passage-token, and passage-word summaries.
        """
        return {
            'minimum': min(counts, default=0),
            'median': median(counts) if counts else 0,
            'mean': round(mean(counts), 2) if counts else 0,
            'maximum': max(counts, default=0),
            'total': sum(counts),
        }

    embedding_tokens = [len(encoder.encode(chunk['embedding_text'])) for chunk in chunks]
    passage_tokens = [len(encoder.encode(chunk['metadata']['text'])) for chunk in chunks]
    passage_words = [len(chunk['metadata']['text'].split()) for chunk in chunks]
    return {
        'embedding_input_tokens': describe(embedding_tokens),
        'passage_tokens': describe(passage_tokens),
        'passage_words_whitespace_split': describe(passage_words),
        'small_passages_under_100_tokens': sum(count < 100 for count in passage_tokens),
        'chunks_by_category': dict(Counter(chunk['metadata']['category'] for chunk in chunks)),
    }


def verify(index, chunks, namespace, attempts=12, delay=5):
    """Fetch every expected ID and compare its metadata with the prepared record.

    Pinecone writes can take time to become visible. Recheck with a bounded
    delay instead of declaring success as soon as the upload request returns.
    This verifies stored text/metadata, not answer quality or vector accuracy.

    Used by:
        This file: ingest(), after all vector batches have been uploaded.
        Direct test/fixture callers in tests/test_policy_pipeline.py: test_upload_verification_failure().
    """
    for attempt in range(attempts):
        missing = []
        # range(start, stop, step) visits batch offsets 0, 100, 200, ... .
        for offset in range(0, len(chunks), 100):
            # A slice [start:end] includes start but excludes end.
            batch = chunks[offset:offset+100]
            response = retry(lambda: index.fetch(ids=[c['id'] for c in batch], namespace=namespace))
            for chunk in batch:
                record = response.vectors.get(chunk['id'])
                if record is None or record.metadata != chunk['metadata']:
                    missing.append(chunk['id'])
        if not missing:
            return
        if attempt + 1 < attempts:
            time.sleep(delay)
    raise RuntimeError(f'Verification failed for {len(missing)} chunks')


def ingest(index, ai, chunks, namespace, model, dimensions=512):
    """Upload prepared chunks, then verify storage and category-filtered search.

    This function receives clients instead of creating them, making offline tests
    possible with fake services. It raises on failure; main records that outcome.
    LangChain add_texts() uploads synchronously in batches of at most 64. The batch
    adapter embeds contextual input while the vector store saves readable passages.
    Direct Pinecone SDK fetch/query calls then verify storage and category search.

    Used by:
        This file: main(), for live ingestion only; skipped when --dry-run is supplied.
        Direct test/fixture callers in tests/test_policy_pipeline.py: test_repeat_upsert_uses_same_ids().
    """
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset:offset+64]
        # LangChain embeds contextual text through the batch adapter, preserving
        # the same embedding inputs, IDs, and stored metadata layout as before.
        store = vector_store(index, PreparedBatchEmbeddings(ai.embeddings, batch), namespace)
        retry(lambda: store.add_texts(
            texts=[chunk['metadata']['text'] for chunk in batch],
            metadatas=[dict(chunk['metadata']) for chunk in batch],
            ids=[chunk['id'] for chunk in batch],
            embedding_chunk_size=64, batch_size=64, async_req=False,
        ))
    verify(index, chunks, namespace)
   
    # Check filtered semantic retrieval, not only record fetch visibility.
    # Braces create a set of unique categories; next(...) selects one sample per category.
    for category in sorted({c['metadata']['category'] for c in chunks}):
        sample = next(c for c in chunks if c['metadata']['category'] == category)
        vector = embed(ai, [sample['embedding_text']], model, dimensions)[0]
        for attempt in range(12):
            result = retry(lambda: index.query(vector=vector, namespace=namespace, top_k=3,
                           filter={'category': {'$eq': category}}, include_metadata=True))
            # This is a search smoke check, not proof the best passage was retrieved.
            if result.matches and all(m.metadata.get('category') == category for m in result.matches):
                break
            if attempt == 11:
                raise RuntimeError(f'Filtered retrieval verification failed: {category}')
            time.sleep(5)


def main():
    """Coordinate CLI arguments, local preparation, service calls, and reporting.

    Used by:
        This file: the if __name__ == "__main__" entry-point block. Run with uv run --locked python ingest_policies.py.
    """
    
    # argparse turns command-line flags into fields on args and provides --help.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',
                         type=Path,
                         default=Path('Dataset/Policy document.json'),
                         help='Policy JSON file (default: %(default)s).')
    
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Validate, prepare, and display statistics without uploading.')
    
    # These values are passed explicitly to prepare(), making the chunking controls
    # discoverable here rather than relying on defaults hidden in another module.
    parser.add_argument('--chunk-size',
                        type=int,
                        default=600,
                        metavar='TOKENS',
                        help='Maximum tokens per embedding input, including title/heading (default: %(default)s).')
    
    parser.add_argument('--chunk-overlap',
                        type=int,
                        default=100,
                        metavar='TOKENS',
                        help='Overlap within a section; must be smaller than chunk size (default: %(default)s).')
    
    args = parser.parse_args()

    
    # Validate before reading data or connecting to services. A section's title and
    # heading must also fit; prepare() performs that content-dependent budget check.
    if args.chunk_size <= 0:
        parser.error('--chunk-size must be a positive integer')

    if not 0 <= args.chunk_overlap < args.chunk_size:
        parser.error('--chunk-overlap must be nonnegative and smaller than --chunk-size')

    # Load configuration from .env/environment; secrets remain in the environment
    # and are consumed later by clients(), not stored in the Settings object.
    settings = Settings.read()
    documents = load_documents(args.dataset)

    chunks, namespace, parameters = prepare(
        documents, args.dataset.as_posix(), settings.embedding_model,
        size=args.chunk_size, overlap=args.chunk_overlap,
        dimensions=settings.embedding_dimensions,
    )
    
    # Counter tallies how many documents belong to each category.
    summary = {'documents': len(documents), 'chunks': len(chunks), 'namespace': namespace,
               'embedding_model': settings.embedding_model,
               'embedding_dimensions': settings.embedding_dimensions,
               'categories': dict(Counter(d.metadata.category for d in documents)),
               'chunking': {
                   'strategy': 'Markdown sections, paragraph accumulation, token-window fallback',
                   'tokenizer': 'cl100k_base',
                   'chunk_size_tokens_including_title_and_heading': args.chunk_size,
                   'overlap_tokens_within_section': args.chunk_overlap,
                   'overlap_crosses_headings': False,
                   'fixed_word_limit': None,
               },
               'chunk_statistics': chunk_statistics(chunks)}
    print('Preparation summary (shown before any OpenAI/Pinecone requests):')
    print(json.dumps(summary, indent=2))
    print('Chunk size is a ceiling, not a target. Short sections remain small.')
    print('Overlap repeats context within a section; it does not cross headings.')
    print('Words are informational only. Display word wrapping does not affect chunking.')
    print('Changing chunk size or overlap produces a different namespace.')


    # Return before creating clients or a report: this is the no-upload path.
    if args.dry_run:
        print('Dry run passed. No external services called and no report written.')
        return
    
    # **summary copies its key/value pairs into the new dictionary.
    # The catalog lets chat route against the vocabulary from this exact ingestion.
    report = {**summary,
              'status': 'running',
              'index': settings.index,
              'parameters': parameters,
              'catalog': catalog(documents), 'verified_at': None, 'error': None}
    
    path = Path(settings.report)
    path.parent.mkdir(parents=True, exist_ok=True)    
    # Invalidate any previous success before touching the namespace.
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    try:
        ai, index = clients(settings, create=True)
        ingest(index, ai, chunks, namespace, settings.embedding_model, settings.embedding_dimensions)
        # Record success only after both fetch and search verification finish.
        report.update(status='verified', verified_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        # Save the exception class, not raw service messages that could contain secrets.
        report.update(status='failed', error=type(exc).__name__)
        raise
    finally:
        # finally runs on success and failure, leaving a truthful readiness report.
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Ingestion verified. Set PINECONE_NAMESPACE={namespace} in .env')


# Execute the CLI only when this file is run directly. Importing its functions
# from tests or another module must not start a real upload.
if __name__ == '__main__':
    main()
