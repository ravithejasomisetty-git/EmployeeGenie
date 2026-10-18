# Employee Genie

A local Streamlit application that answers questions about supplied policy documents using LangChain, OpenAI embeddings, and Pinecone. It classifies questions into categories, retrieves matching passages with metadata filters, and produces cited answers.

## Current capabilities

- Standalone JSON ingestion with strict validation, heading-aware token chunking, inherited metadata, deterministic IDs, and dataset-version namespaces.
- Configurable embedding model and dimensions; the default is `text-embedding-3-small` with 512 dimensions.
- Structured question classification and answer generation with `ChatOpenAI` (default: `gpt-4.1-mini`).
- Multi-category retrieval across HR, onboarding, compliance, and product. Compensation is an HR subcategory.
- Streaming answers, session-based follow-ups, expandable source passages, and optional routing/retrieval diagnostics.
- Citation correction runs silently; a failed correction returns a safe fallback response.
- Offline tests and a separate, opt-in live evaluation script.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). Run commands from the project root. Python 3.11+ is supported by the manifest; `.python-version` selects Python 3.14 for development. The project has been tested locally on Windows with Python 3.14.

```powershell
uv sync --locked
# Fresh checkout only: do not overwrite an existing .env.
Copy-Item .env.example .env
```

Add your OpenAI and Pinecone API keys to `.env`. Review the index, cloud, region, model, and dimensions there. Existing process environment variables take priority over `.env`; defaults apply only to absent variables, not blank values. Hosted service use can incur charges.

`pyproject.toml` declares dependencies; `uv.lock` pins their resolved versions. The dev group includes pytest and is installed by default. This script application is not built as a Python wheel, and environment activation is unnecessary.

## Ingest documents

The dataset contains 80 newly authored fictional records, with 20 in each category. Default preparation produces 320 chunks. Cedarfield Labs and the Harbor workspace product are invented for this demo. All policies, amounts, deadlines, and entitlements are synthetic, not real employment or legal guidance. See `Dataset/README.md` for provenance and intended use.

```powershell
uv run --locked python ingest_policies.py --dry-run
uv run --locked python ingest_policies.py
```

Dry run validates and prepares locally without OpenAI/Pinecone calls or report writes. Tokenization may download its public vocabulary on first use. Both modes show chunking configuration, token/word statistics, per-category counts, dimensions, and namespace.

Defaults: 600 tokens including title/heading, with up to 100 tokens of overlap within each section. Short sections remain small. Paragraphs stay together when possible; oversized text uses token windows, without a sentence-boundary guarantee. Word counts are descriptive, not chunk-size rules.

```powershell
uv run --locked python ingest_policies.py --dry-run --chunk-size 800 --chunk-overlap 120
uv run --locked python ingest_policies.py --chunk-size 800 --chunk-overlap 120
uv run --locked python ingest_policies.py --help
```

Use the same settings for preview and upload. Live ingestion creates a missing Pinecone index or checks its dimensions and cosine metric. It uploads through `PineconeVectorStore.add_texts()`, verifies stored metadata and category-filtered search, and writes `artifacts/ingestion-report.json`. Only a successful run is marked `verified`. Copy the printed namespace into `PINECONE_NAMESPACE` in `.env`.

Identical prepared data/settings use the same namespace and IDs. Changes to text, source filename, chunking, embedding dimensions/model, or preparation version produce a new namespace. Old namespaces are retained. Do not run concurrent ingestions against the same report/namespace or query a namespace being reingested.

An existing index with different dimensions needs a different index name, or must be explicitly deleted and recreated by its owner. A namespace cannot change an index's dimensions. The code never deletes indexes automatically.

## Run the chat

```powershell
uv run --locked python -m streamlit run policy_chat_app.py
```

Open the local URL printed in the terminal, usually http://localhost:8501. Keep the terminal running; press Ctrl+C to stop. A verified local report, matching index/model/dimensions/namespace, and configured keys are required. The UI never triggers ingestion. A fresh clone does not include the ignored local report; run ingestion in that environment before chatting.

Questions need at least two whitespace-separated words containing letters or numbers, up to 6,000 characters. Spaces and punctuation are allowed; spaces-only, punctuation-only, and one-word inputs are rejected before changing history or calling RAG. The two-word rule is UI-only; direct backend/evaluation calls check blank input and length.

Try "How many leaves is an employee eligible for annually?" or "Explain onboarding buddies and annual compensation review." Follow-ups use the last six messages in the browser session. Clear conversation removes that history; it is not stored in a database.

## Execution flow

```text
Ingestion:
ingest_policies.main()
  -> Settings.read() -> load_documents() -> prepare() -> chunk_statistics()
  -> dry run: stop
  -> live: catalog() -> write running report -> clients(create=True)
       -> ingest() -> LangChain embeddings/upload
       -> verify() -> category-search checks -> write verified/failed report

Chat:
policy_chat_app -> readiness checks -> input validation -> recent history
  -> get_rag() -> RAG.ask()
       -> classify() -> structured() -> validate_routes()
            -> clarify/out_of_scope: return guidance
            -> retrieve() -> embed() -> metadata-filtered Pinecone searches
                 -> select_evidence()
                      -> no passages: insufficient-evidence response
                      -> generate() -> structured() -> validate_answer()
                           -> invalid citations: one silent correction, then a safe fallback
  -> show_result() -> save answer in session history
```

Classification uses the ingested catalog to choose focused question/category routes. It does not establish that an answer exists. Each route searches within its category; subcategory routes also include broader candidates from the same category. Each filtered search requests at most five candidates. Candidates are deduplicated and ranked by similarity across routes; only the top five unique passages are passed to the LLM for answer synthesis. Multi-topic questions can have uneven coverage within this five-passage budget.

The answer system prompt requests evidence-only claims, preservation of qualifications, and explicit handling of missing/conflicting information. Status can be `answered`, `partial`, or `insufficient_evidence`. Citation validation checks source labels, not whether a passage proves a claim. Similarity scores are not calibrated answer-confidence probabilities. No separate factual-verification model is implemented.


## Code map

| File | Responsibility |
|---|---|
| `ingest_policies.py` | CLI, statistics, upload orchestration, verification, reports |
| `policy_helper/policy_document_preparation.py` | Schemas, cleaning, token chunking, IDs, namespace fingerprint, catalog |
| `policy_helper/ai_service_clients.py` | Settings, LangChain model/vector-store wrappers, Pinecone index setup, retries |
| `policy_helper/policy_question_answering.py` | Routing, filtered retrieval, evidence selection, answer generation, citation checks |
| `policy_chat_app.py` | Streamlit validation, conversation history, answer/source display |
| `evaluate_policy_answers.py` | Opt-in live routing evaluation and review output |
| `tests/test_policy_pipeline.py` | Offline pipeline and Streamlit tests |

Docstrings describe function contracts and comments explain non-obvious behavior. `system_prompt` holds application instructions; `user_prompt` is the original user input. The model payload also contains history and catalog/evidence, and focused `Route.question` values are classifier-generated search questions.

LangChain uses streaming prompt/model chains, schema-constrained responses, `OpenAIEmbeddings`, and `PineconeVectorStore`. Initial answer generation streams partial answer text; citation corrections run silently. Valid declared references are displayed when inline markers are omitted. `PreparedBatchEmbeddings` preserves title + heading + passage embedding inputs while storing just the passage under metadata key `text`. Retrieval converts LangChain Documents back to the source dictionaries used by the UI. The Pinecone SDK remains for index administration, fetch verification, and ingestion search checks.

## Verification

```powershell
uv run --locked python -m pytest -q -p no:cacheprovider
uv run --locked python ingest_policies.py --dry-run
# Optional: real hosted calls; requires verified ingestion and credentials.
uv run --locked python evaluate_policy_answers.py
```

Offline tests use real LangChain prompt/vector-store adapters with fake provider responses. They check validation, chunk sizes, namespace behavior, upload compatibility, multi-category filters, citation numbering, and UI interactions. They do not demonstrate live API access or factual model accuracy.

`evaluation/questions.json` contains 33 draft routing cases. Live evaluation runs each independently without history and writes routing accuracy, answers, and sources to `artifacts/evaluation.json`. Review expected labels and factual support manually; the script does not calculate answer correctness, expected-document recall, or calibrated confidence. A 90% routing score is a proposed evaluation target, not an achieved result.

## Repository and local files

Commit source, tests, synthetic dataset and its README, evaluation cases, README, `.env.example`, `.gitignore`, `.python-version`, `pyproject.toml`, and `uv.lock`.

`.gitignore` excludes `.env` variants except `.env.example`, Streamlit secrets, virtual environments, Python/test caches, logs, local IDE files, and `artifacts/`. Keep your local ingestion report: the app needs it. Ignore rules do not remove files already tracked by Git; inspect staged files before committing. No application license is selected here; add one only after deciding the intended distribution terms.

## Dependencies

Use `uv add package-name` or `uv add --dev package-name` for dependency changes.
After editing dependencies manually, run `uv lock`, `uv sync --locked`, and the offline tests.
Use `uv sync --no-dev --locked` for runtime dependencies only.

On Windows, stop Streamlit before synchronizing dependencies: loaded modules can block package removal.

## Limits

This is a local shared-access demo, without authentication, authorization, deployment configuration, persistent chat storage, live policy updates, hybrid search, or reranking. Document labels can be coarse, retrieval can miss exceptions, and at most five passages enter an answer. Prompt instructions reduce but cannot eliminate hallucination or prompt injection. Read the sources and snapshot caveats; linked pages are not fetched. This application does not claim production readiness or current-policy accuracy.
