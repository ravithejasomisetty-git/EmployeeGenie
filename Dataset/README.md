# Synthetic policy dataset

`Policy document.json` contains 80 original fictional demonstration records: 20 each for HR, compliance, product, and onboarding. It replaces the previously supplied third-party documentation dataset; its prose was newly authored from fictional scenarios rather than paraphrased from those documents.

Cedarfield Labs is an invented organization. Harbor is an invented workspace application. Names are illustrative and imply no association with any real entity. All credits, dates, time limits, procedures, benefits, and access rules are invented. This material is not real employment, financial, legal, or compliance guidance.

Each record retains the application's schema: `id`, `metadata`, and Markdown `content`. Metadata includes category, subcategory, topics, audience roles, and `applicability_scope="fictional_demo"`. Empty country lists do not assert real worldwide legal applicability.

The active dataset contains no external URLs, copied product code, or source-site identifiers. No external-source attribution is asserted for this newly authored text. No formal distribution license has been selected for the project or dataset; choose one explicitly before advertising reuse permissions.

To ingest this dataset, run `uv run --locked python ingest_policies.py --dry-run`, then `uv run --locked python ingest_policies.py`. Set the verified namespace in `.env` and restart Streamlit. Old cloud namespaces are not automatically removed. The latest local verification report is stored in the ignored `artifacts/ingestion-report.json`; the app needs that report to enable chat.
