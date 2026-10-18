"""Employee Genie application package.

This file marks policy_helper as a Python package so imports such as
`from policy_helper.policy_question_answering import RAG` work. It intentionally starts no services.

Learning order: policy_document_preparation.py (prepare text), ai_service_clients.py (connect to providers),
ingest_policies.py at the project root (load Pinecone), policy_question_answering.py (answer questions),
then policy_chat_app.py at the project root (display the chat).
"""
