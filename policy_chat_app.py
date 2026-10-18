"""Local chat UI. Start with: uv run --locked streamlit run policy_chat_app.py.

Streamlit executes this file from top to bottom when a browser session opens
and again when a widget changes. session_state preserves that session's messages
across reruns; cache_resource reuses expensive service clients. Keeping these
separate avoids reconnecting unnecessarily without sharing users' chat histories.

Readiness is checked before enabling chat. This file never ingests documents:
the separate ingest_policies.py command must first produce a verified report.
Valid submissions contain at least two whitespace-separated words with letters
or numbers. Spaces/punctuation are allowed inside otherwise valid questions.
The question and last six messages flow into the LangChain-backed RAG service.
"""
import logging
import os
from pathlib import Path
import traceback

import streamlit as st

from policy_helper.policy_question_answering import RAG, read_report
from policy_helper.ai_service_clients import Settings

# Render inexpensive page elements first, before any possible service connection.
st.set_page_config(page_title='Employee Genie', page_icon=':material/menu_book:')
st.title('Employee Genie')
st.caption('Explore HR, onboarding, compliance, and product documentation.')

settings = Settings.read()
# 'with' directs the enclosed UI elements into a particular container.
with st.sidebar:
    st.header('Your knowledge library')
    st.write('Ask one question or combine topics. Answers include passages from the supplied dataset.')
    diagnostics = st.toggle('Show retrieval details', value=False)
    if st.button('Clear conversation', icon=':material/delete_sweep:', width='stretch'):
        st.session_state.messages = []
        # Restart rendering so the old messages disappear immediately.
        st.rerun()
    st.caption('Answers reflect the ingested snapshot. Check cited conditions and policy dates.')

# Initialize once per browser session, rather than erasing messages on every rerun.
if 'messages' not in st.session_state:
    st.session_state.messages = []


@st.cache_resource(max_entries=2, ttl=300)
def get_rag(config, report_stamp):
    """Reuse a service object for up to 300 seconds, with at most two cached entries.

    The decorator wraps this function with a cache. Although report_stamp is not
    used in the body, it is part of the cache key: a changed report modification
    time forces a fresh object instead of reusing an outdated dataset catalog.
    Cached resources may be shared across sessions; chat messages are not stored
    on this RAG object.

    Used by:
        This file: question-submission block. Streamlit caches the returned service object between calls.
    """
    return RAG(config)


def show_result(result, show_answer=True):
    """Render a stored or newly generated answer, sources, and optional diagnostics.

    Source positions match the [1]-style labels assigned during generation.
    All selected passages are available for inspection, including passages the
    answer did not cite. Expanders organize display; their contents still execute.

    Used by:
        This file: conversation-history rendering loop and the question-submission block after a successful answer.
    """
    if show_answer:
        st.markdown(result['answer'])
    if result.get('sources'):
        with st.expander('Source passages'):
            for label, source in enumerate(result['sources'], 1):
                meta = source['metadata']
                st.markdown(f"**[{label}] {meta['title']}**")
                st.caption(f"{meta['category']} / {meta['subcategory']} · {meta['document_id']}")
                # Plain text prevents embedded document HTML from becoming UI.
                st.text(meta['text'])
    if diagnostics:
        with st.expander('Retrieval details'):
            st.json({key: result[key] for key in ('routing', 'diagnostics', 'seconds') if key in result})


# Rebuild the visible conversation from session data on every Streamlit rerun.
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        if message['role'] == 'assistant':
            show_result(message['result'])
        else:
            st.write(message['content'])

# These checks read local settings/report only. They do not prove that remote
# services are reachable; service authentication happens when a question is asked.
ready = False
try:
    report = read_report(settings)
    missing = [name for name in ('OPENAI_API_KEY', 'PINECONE_API_KEY') if not os.getenv(name)]
    if missing:
        raise ValueError('Set ' + ', '.join(missing) + ' in your .env file.')
    ready = True
    with st.sidebar:
        st.success(f"{report['documents']} documents · {report['chunks']} passages")
except (ValueError, OSError, KeyError) as exc:
    st.info(str(exc))
    with st.expander('Set up the knowledge library'):
        st.write('Copy .env.example to .env, add your API keys, then run ingestion and copy its namespace to .env.')
        st.code('uv sync --locked\nuv run --locked python ingest_policies.py --dry-run\nuv run --locked python ingest_policies.py', language='powershell')

# No question can be submitted until local ingestion/configuration checks pass.
user_prompt = st.chat_input('Ask about your policies or workspace product documentation', disabled=not ready, max_chars=6000)
if user_prompt is not None:
    # Count whitespace-separated words only if they contain a letter or number.
    # isalnum() recognizes Unicode letters/numbers; punctuation-only tokens such
    # as '!!!' do not count. Normal punctuation in 'benefits?' is allowed.
    word_count = sum(
        any(character.isalnum() for character in word)
        for word in user_prompt.split()
    )
    if word_count < 2:
        st.warning('Please enter a question with at least 2 words containing letters or numbers. Spaces and special characters alone are not accepted.')
        # Stop only this render. The chat box remains available for correction.
        # No history entry or RAG service call is made for this invalid input.
        st.stop()

    # Snapshot recent history BEFORE appending the new question, so it is not
    # sent twice. Pass only role/content, leaving UI diagnostics out of model input.
    history = [{'role': m['role'], 'content': m['content']} for m in st.session_state.messages[-6:]]
    st.session_state.messages.append({'role': 'user', 'content': user_prompt})
    with st.chat_message('user'):
        st.write(user_prompt)
    with st.chat_message('assistant'):
        answer_preview = st.empty()
        try:
            with st.spinner('Finding supporting policy passages…'):
                rag = get_rag(settings, Path(settings.report).stat().st_mtime_ns)
                # One backend call coordinates classification, retrieval, and generation.
                result = rag.ask(
                    user_prompt, history,
                    on_answer=lambda text: answer_preview.markdown(text),
                )
            # Keep one answer location for the preview and validated final text.
            answer_preview.markdown(result['answer'])
            show_result(result, show_answer=False)
            # Store plain text for future model context and the full result for UI replay.
            st.session_state.messages.append({'role': 'assistant', 'content': result['answer'], 'result': result})
        except Exception as exc:
            answer_preview.empty()
            # Log the failing source location without exposing raw SDK error details.
            frames = traceback.extract_tb(exc.__traceback__)
            if frames:
                origin = frames[-1]
                logging.getLogger(__name__).error(
                    'Question failed (%s at %s:%d in %s)',
                    type(exc).__name__, Path(origin.filename).name, origin.lineno, origin.name,
                )
            else:
                logging.getLogger(__name__).error('Question failed (%s)', type(exc).__name__)
            st.error('Unable to complete the question. Check your configuration, service access, and connection, then retry.')
            st.caption('For a more specific diagnosis, check the Streamlit terminal for the failure location.')
            # Keep conversation consistent after a failed request.
            st.session_state.messages.pop()
