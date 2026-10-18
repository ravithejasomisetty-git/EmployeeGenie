"""Offline ingestion, retrieval, streaming and Streamlit interface tests.

Run with uv run --locked python -m pytest -q.
Provider calls are mocked; tokenization may download its vocabulary on first use.
These tests verify application behavior rather than live model accuracy.
"""
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
from langchain_core.runnables import RunnableLambda

import pytest
import tiktoken

from ingest_policies import chunk_statistics, ingest, verify
from policy_helper.policy_document_preparation import clean_content, load_documents, prepare
from policy_helper.policy_question_answering import (Classification, GeneratedAnswer, RAG, Route, read_report,
                               select_evidence, validate_answer, validate_routes)
from policy_helper.ai_service_clients import Settings, clients, embed, PreparedBatchEmbeddings

DATASET = Path('Dataset/Policy document.json')


def test_langchain_prompt_roles_and_literal_braces():
    """Keep system instructions separate and do not interpret JSON braces as templates."""
    rag = object.__new__(RAG)
    rag.ai = Mock()
    captured = []

    def answer(prompt_value):
        """Capture rendered LangChain messages; used by the fake chat runnable."""
        captured.extend(prompt_value.to_messages())
        return GeneratedAnswer(status='insufficient_evidence', answer='Not established.', citations=[])

    rag.ai.chat.with_structured_output.return_value = RunnableLambda(answer)
    result = rag.structured(GeneratedAnswer, 'Rules with {literal braces}', {'question':'Explain {benefits}'})
    assert result.status == 'insufficient_evidence'
    assert [message.type for message in captured] == ['system', 'human']
    assert captured[0].content == 'Rules with {literal braces}'
    assert '{benefits}' in captured[1].content
    assert rag.ai.chat.with_structured_output.call_args.kwargs == {'method':'json_schema','strict':True}


def test_prepared_embedding_adapter_preserves_context():
    """Embed the same contextual input as before while storing the original passage."""
    chunks = [
        {'embedding_text':'Title A\nSection\n\nShared words', 'metadata':{'text':'Shared words'}},
        {'embedding_text':'Title B\nSection\n\nShared words', 'metadata':{'text':'Shared words'}},
    ]
    base = Mock()
    base.embed_documents.return_value = [[.1]*512, [.2]*512]
    adapter = PreparedBatchEmbeddings(base, chunks)
    assert len(adapter.embed_documents(['Shared words','Shared words'])) == 2
    base.embed_documents.assert_called_once_with([chunk['embedding_text'] for chunk in chunks])
    with pytest.raises(ValueError, match='batch does not match'):
        adapter.embed_documents(['Wrong passage'])


def test_langchain_refusal_does_not_become_policy_answer():
    """Missing structured output fails clearly rather than displaying invented content."""
    rag = object.__new__(RAG)
    rag.ai = Mock()
    rag.ai.chat.with_structured_output.return_value = RunnableLambda(lambda messages: None)
    with pytest.raises(ValueError, match='valid structured response'):
        rag.structured(GeneratedAnswer, 'Use evidence only', {'question':'Which benefits?'})


def test_configured_dimensions_reach_openai():
    """The embedding helper uses the configured LangChain model and vector length."""
    ai = Mock()
    ai.embeddings.model = 'text-embedding-3-small'
    ai.embeddings.dimensions = 512
    ai.embeddings.embed_documents.side_effect = lambda texts: [[0.1]*512 for _ in texts]
    assert len(embed(ai, ['policy'], 'text-embedding-3-small', 512)[0]) == 512
    assert ai.embeddings.dimensions == 512
    ai.embeddings.embed_documents.assert_called_once_with(['policy'])


def test_dimensions_change_namespace(prepared):
    """Identical text with a different vector length is a distinct dataset version."""
    docs, (_, original, _) = prepared
    _, changed, parameters = prepare(docs, DATASET.as_posix(), 'text-embedding-3-small', dimensions=1536)
    assert changed != original
    assert parameters['dimensions'] == 1536


def test_old_index_dimensions_rejected(monkeypatch):
    """A mismatch must not delete or recreate the existing index."""
    import policy_helper.ai_service_clients as services
    monkeypatch.setenv('OPENAI_API_KEY', 'fake')
    monkeypatch.setenv('PINECONE_API_KEY', 'fake')
    pc = Mock()
    pc.has_index.return_value = True
    pc.describe_index.return_value = NS(dimension=1536, metric='cosine')
    monkeypatch.setattr(services, 'Pinecone', Mock(return_value=pc))
    monkeypatch.setattr(services, 'ChatOpenAI', Mock())
    monkeypatch.setattr(services, 'OpenAIEmbeddings', Mock())
    settings = Settings('index', 'ns', 'text-embedding-3-small', 'chat', 'report', 'aws', 'us-east-1')
    with pytest.raises(ValueError, match='new index name'):
        clients(settings, create=True)
    pc.create_index.assert_not_called()
    pc.delete_index.assert_not_called()


def test_report_dimensions_must_match(tmp_path):
    """An old namespace cannot be queried using a new embedding length."""
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'status':'verified','index':'index','namespace':'ns',
                               'parameters':{'model':'model','dimensions':1536}}))
    settings = Settings('index','ns','model','chat',str(path),'aws','us-east-1')
    with pytest.raises(ValueError, match='dimensions differ'):
        read_report(settings)


def test_chunk_statistics_measures_embedding_prefix_separately():
    """Report real token counts and descriptive word counts, including empty input."""
    chunks = [{'embedding_text': 'Policy title\nEligibility\n\nEmployees qualify.',
               'metadata': {'text': 'Employees qualify.', 'category': 'hr'}}]
    result = chunk_statistics(chunks)
    encoder = tiktoken.get_encoding('cl100k_base')
    assert result['embedding_input_tokens']['maximum'] == len(encoder.encode(chunks[0]['embedding_text']))
    assert result['passage_tokens']['maximum'] < result['embedding_input_tokens']['maximum']
    assert result['passage_words_whitespace_split']['total'] == 2
    assert result['small_passages_under_100_tokens'] == 1
    assert result['chunks_by_category'] == {'hr': 1}
    assert chunk_statistics([])['embedding_input_tokens']['total'] == 0


def test_dry_run_prints_custom_chunk_settings(monkeypatch, capsys):
    """CLI controls reach preparation and dry runs never create service clients."""
    import ingest_policies as cli
    monkeypatch.setattr('sys.argv', ['ingest_policies.py', '--dry-run', '--chunk-size', '800', '--chunk-overlap', '120'])
    connect = Mock(side_effect=AssertionError('Dry run must not connect'))
    monkeypatch.setattr(cli, 'clients', connect)
    cli.main()
    output = capsys.readouterr().out
    assert '"chunk_size_tokens_including_title_and_heading": 800' in output
    assert '"overlap_tokens_within_section": 120' in output
    assert '"embedding_input_tokens"' in output
    assert '"median"' in output
    connect.assert_not_called()


@pytest.mark.parametrize('size,overlap', [('0','0'), ('600','600'), ('600','-1')])
def test_invalid_chunk_options_fail_before_service_calls(monkeypatch, size, overlap):
    """Reject invalid budgets with argparse's normal command-line error exit."""
    import ingest_policies as cli
    monkeypatch.setattr('sys.argv', ['ingest_policies.py','--chunk-size',size,'--chunk-overlap',overlap])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.fixture(scope='module')
def prepared():
    """Prepare the real dataset once and share it across tests in this module.

    A fixture is a reusable test setup step. pytest injects its returned value
    into any test with a parameter named prepared. No vectors are uploaded.
    """
    docs = load_documents(DATASET)
    return docs, prepare(docs, DATASET.as_posix(), 'text-embedding-3-small')


def test_real_dataset_and_chunks(prepared):
    """Check token limits, inherited categories, unique IDs, and namespace stability."""
    docs, (chunks, namespace, _) = prepared
    assert len(docs) == 80
    assert len({c['id'] for c in chunks}) == len(chunks)
    encoder = tiktoken.get_encoding('cl100k_base')
    parents = {d.id: d for d in docs}
    for chunk in chunks:
        assert len(encoder.encode(chunk['embedding_text'])) <= 600
        assert chunk['metadata']['category'] == parents[chunk['metadata']['document_id']].metadata.category
        assert chunk['metadata']['text'].strip()
    assert prepare(docs, DATASET.as_posix(), 'text-embedding-3-small')[1] == namespace
    # A deep copy isolates this mutation from the shared fixture and later tests.
    changed = [d.model_copy(deep=True) for d in docs]
    changed[0].content += '\nA new policy clause.'
    assert prepare(changed, DATASET.as_posix(), 'text-embedding-3-small')[1] != namespace


@pytest.mark.parametrize('mutation', ['duplicate', 'blank', 'category', 'type'])
def test_invalid_dataset_fails(tmp_path, mutation):
    """Run the same rejection test for four independently damaged inputs.

    parametrize asks pytest to execute this function once for each mutation value.
    pytest.raises succeeds only if the enclosed operation raises the expected error.
    """
    raw = json.loads(DATASET.read_text(encoding='utf-8'))[:2]
    if mutation == 'duplicate': raw[1]['id'] = raw[0]['id']
    if mutation == 'blank': raw[0]['content'] = ' '
    if mutation == 'category': raw[0]['metadata']['category'] = 'unknown'
    if mutation == 'type': raw[0]['metadata']['topics'] = 'salary'
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(raw), encoding='utf-8')
    with pytest.raises(ValueError): load_documents(path)


def test_cleaning_preserves_warning():
    """Markup removal must keep policy caveats and text inside template sections."""
    text = '---\ntitle: Test\n---\n<span>Use The Loop for current guidance.</span>\n{{% details %}}Exception{{% /details %}}'
    clean = clean_content(text)
    assert 'Use The Loop' in clean and 'Exception' in clean and '<span>' not in clean


def test_upload_verification_failure(prepared):
    """A fake service returning no stored vectors must not pass verification."""
    chunks = prepared[1][0][:2]
    index = Mock()
    index.fetch.return_value = NS(vectors={})
    with pytest.raises(RuntimeError): verify(index, chunks, 'test', attempts=1, delay=0)


def test_repeat_upsert_uses_same_ids(prepared):
    """Two identical uploads send identical records instead of random new IDs."""
    chunks = prepared[1][0][:2]
    index = Mock()
    # return_value is the canned response returned whenever this fake method runs.
    index.fetch.return_value = NS(vectors={c['id']: NS(metadata=c['metadata']) for c in chunks})
    index.query.return_value = NS(matches=[NS(metadata={'category': 'hr'})])
    ai = Mock()
    ai.embeddings.model = 'text-embedding-3-small'
    ai.embeddings.dimensions = 512
    ai.embeddings.embed_documents.side_effect = lambda texts: [[0.1]*512 for _ in texts]
    ingest(index, ai, chunks, 'test', 'text-embedding-3-small')
    ingest(index, ai, chunks, 'test', 'text-embedding-3-small')
    # Mocks record calls so we can compare the keyword arguments sent to Pinecone.
    first, second = index.upsert.call_args_list
    assert first.kwargs['vectors'] == second.kwargs['vectors']


def test_multicategory_retrieval_filters_and_coverage():
    """Ensure every selected category contributes evidence and filters remain scoped."""
    # Bypass __init__ only in this focused test, avoiding readiness/network setup.
    rag = object.__new__(RAG)
    rag.settings = NS(embedding_model='test', namespace='verified', embedding_dimensions=512)
    rag.ai = Mock()
    rag.ai.embeddings.model = 'test'
    rag.ai.embeddings.dimensions = 512
    rag.ai.embeddings.embed_documents.side_effect = lambda texts: [[0.1]*512 for _ in texts]
    def query(**kwargs):
        """Return a Pinecone-style dictionary consumed by the real LangChain store."""
        cat = kwargs['filter']['category']['$eq']
        sub = kwargs['filter'].get('subcategory', {}).get('$eq', 'other')
        return {'matches': [{'id':f'{cat}-{sub}', 'score':.8,
            'metadata':{'category':cat,'subcategory':sub,'text':'support'}}]}
    rag.index = Mock()
    # side_effect delegates each fake request to our function, allowing its
    # response to depend on the actual metadata filter under test.
    rag.index.query.side_effect = query
    routes = [Route(question='review', category='hr', subcategory='compensation'),
              Route(question='new hire', category='onboarding', subcategory=None),
              Route(question='training', category='compliance', subcategory=None)]
    evidence, details = rag.retrieve(routes)
    assert {e['metadata']['category'] for e in evidence} == {'hr','onboarding','compliance'}
    assert details[0]['filter']['subcategory'] == {'$eq':'compensation'}
    assert details[1]['filter'] == {'category':{'$eq':'hr'}}
    assert all(call.kwargs['namespace'] == 'verified' for call in rag.index.query.call_args_list)


def test_subcategory_must_exist():
    """A schema-valid string must still match the ingested category vocabulary."""
    decision = Classification(status='retrieve', routes=[Route(question='x', category='hr', subcategory='fake')],
                              explanation='', clarification=None)
    with pytest.raises(ValueError): validate_routes(decision, [{'category':'hr','subcategory':'benefits'}])


@pytest.mark.parametrize('body,labels', [('Claim [9]', [9]), ('Claim [1]', []), ('Uncited claim', [])])
def test_bad_citations_rejected(body, labels):
    """Reject nonexistent, mismatched, or missing source numbers in policy answers."""
    with pytest.raises(ValueError):
        validate_answer(GeneratedAnswer(status='answered', answer=body, citations=labels), 2)


def test_valid_partial_answer():
    """Allow a cited answer that explicitly leaves an unsupported part unanswered."""
    answer = GeneratedAnswer(status='partial', answer='Training is required [1]. Salary details are absent.', citations=[1])
    assert validate_answer(answer, 1) == answer


def test_clarification_does_not_retrieve():
    """Ambiguous questions ask for detail without spending a vector-search call."""
    rag = object.__new__(RAG)
    rag.classify = Mock(return_value=Classification(status='clarify', routes=[], explanation='', clarification='Which policy?'))
    rag.retrieve = Mock()
    assert rag.ask('What is the policy?')['answer'] == 'Which policy?'
    rag.retrieve.assert_not_called()


def test_evidence_deduplicated():
    """A passage found by two routes should appear only once in final evidence."""
    assert len(select_evidence([[{'id':'a'}], [{'id':'a'},{'id':'b'}]])) == 2


def test_report_must_be_verified(tmp_path):
    """The chat readiness gate must reject a report marked failed."""
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'status':'failed'}))
    settings = Settings('index','ns','model','chat',str(path),'aws','us-east-1')
    with pytest.raises(ValueError, match='not verified'): read_report(settings)


def test_ui_setup_without_api_calls(monkeypatch, tmp_path):
    """An unconfigured app shows setup guidance and disables question submission.

    AppTest runs Streamlit in-process without launching a browser or local server.
    """
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('INGESTION_REPORT', str(tmp_path / 'missing.json'))
    app = AppTest.from_file(Path('policy_chat_app.py').resolve(), default_timeout=20).run()
    assert not app.exception
    assert app.chat_input[0].disabled
    assert any('ingestion report' in item.value for item in app.info)


def test_end_to_end_multicategory_answer(tmp_path):
    """Exercise classify -> retrieve -> generate with deterministic fake responses.

    This checks application wiring and citation handling. Because we supply the
    model decisions ourselves, it does not measure real classification accuracy.
    """
    path = tmp_path / 'verified.json'
    cat = [{'category':'hr','subcategory':'compensation'},
           {'category':'onboarding','subcategory':'device_setup'}]
    path.write_text(json.dumps({'status':'verified','index':'index','namespace':'ns',
                               'parameters':{'model':'embedding','dimensions':512},'catalog':cat}))
    settings = Settings('index','ns','embedding','chat',str(path),'aws','us-east-1')
    decision = Classification(status='retrieve', explanation='Two topics', clarification=None,
        routes=[Route(question='review eligibility', category='hr', subcategory='compensation'),
                Route(question='laptop setup', category='onboarding', subcategory='device_setup')])
    ai = Mock()
    # The iterator supplies outputs to the runnable: classify first, answer second.
    outputs = iter([decision, GeneratedAnswer(
        status='answered', answer='Review eligibility [1]. Laptop setup [2].', citations=[1,2])])
    ai.chat.with_structured_output.return_value = RunnableLambda(lambda messages: next(outputs))
    ai.embeddings.model = 'text-embedding-3-small'
    ai.embeddings.dimensions = 512
    ai.embeddings.embed_documents.side_effect = lambda texts: [[0.1]*512 for _ in texts]
    def query(**kwargs):
        """Return category-matched Pinecone records for the LangChain vector store."""
        category = kwargs['filter']['category']['$eq']
        subcategory = next(c['subcategory'] for c in cat if c['category'] == category)
        return {'matches':[{'id':category,'score':.9,'metadata':{'category':category,
            'subcategory':subcategory,'title':category,'document_id':category,'text':'Supporting passage'}}]}
    index = Mock()
    index.query.side_effect = query
    ai.embeddings.model = 'embedding'
    result = RAG(settings,ai,index).ask('Explain review eligibility and laptop setup.')
    assert result['status'] == 'answered'
    assert len(result['sources']) == 2
    assert {source['metadata']['category'] for source in result['sources']} == {'hr','onboarding'}
    assert '[1]' in result['answer'] and '[2]' in result['answer']


def test_ui_chat_and_clear(monkeypatch, tmp_path, caplog):
    """Simulate sending a message, showing sources, and clearing the conversation.

    Dummy credentials and a mocked RAG object enable the UI without authenticating
    to any provider. They are local test strings, not usable account keys.
    """
    from streamlit.testing.v1 import AppTest
    import policy_helper.policy_question_answering as module
    path = tmp_path / 'report.json'
    path.write_text(json.dumps({'status':'verified','index':'test-ui','namespace':'test-ui',
        'parameters':{'model':'text-embedding-3-small','dimensions':512},'documents':80,'chunks':629,'catalog':[]}))
    monkeypatch.setenv('INGESTION_REPORT', str(path))
    monkeypatch.setenv('PINECONE_INDEX','test-ui')
    monkeypatch.setenv('PINECONE_NAMESPACE','test-ui')
    monkeypatch.setenv('OPENAI_API_KEY','test-key')
    monkeypatch.setenv('PINECONE_API_KEY','test-key')
    fake = Mock()
    fake.ask.return_value = {'status':'answered','answer':'Grounded answer [1].',
        'sources':[{'id':'a','metadata':{'title':'Test policy','category':'hr',
                    'subcategory':'benefits','document_id':'test','text':'Evidence'}}],
        'routing':{'routes':[]},'diagnostics':[]}
    # Replace the service constructor before AppTest executes policy_chat_app.py.
    monkeypatch.setattr(module, 'RAG', Mock(return_value=fake))
    app = AppTest.from_file(Path('policy_chat_app.py').resolve(), default_timeout=20).run()
    assert not app.chat_input[0].disabled
    # Invalid input must not enter history or reach the mocked RAG service.
    for invalid in ('   ', '!!! @#$', 'benefits', 'benefits !!!'):
        app.chat_input[0].set_value(invalid).run()
        assert not app.exception
        assert app.warning
        assert not app.session_state.messages
        fake.ask.assert_not_called()
    # A short two-word follow-up with punctuation remains valid.
    app.chat_input[0].set_value('Which benefits?').run()
    assert not app.exception
    fake.ask.assert_called_once()
    assert len(app.session_state.messages) == 2
    app.sidebar.button[0].click().run()
    fake.ask.reset_mock()
    app.chat_input[0].set_value('What benefits exist?').run()
    assert not app.exception
    assert any('Grounded answer' in item.value for item in app.markdown)
    assert len(app.session_state.messages) == 2
    app.sidebar.button[0].click().run()
    assert not app.session_state.messages
    fake.ask.side_effect = ValueError('sensitive provider details')
    with caplog.at_level('ERROR'):
        app.chat_input[0].set_value('How is compensation reviewed?').run()
    assert app.error
    assert 'Question failed (ValueError at ' in caplog.text
    assert 'sensitive provider details' not in caplog.text
    assert not app.session_state.messages


def test_streaming_answer_updates_before_completion():
    """Show incremental answer text while still validating the final schema."""
    from langchain_core.messages import AIMessageChunk
    rag = object.__new__(RAG)
    rag.ai = Mock()

    def chunks(messages):
        yield AIMessageChunk(content='{"status":"answered","answer":"Leave')
        yield AIMessageChunk(content=' entitlement [1]","citations":[1]}')

    rag.ai.chat.bind.return_value = RunnableLambda(chunks)
    updates = []
    result = rag.structured(GeneratedAnswer, 'Use evidence', {}, on_answer=updates.append)
    assert updates == ['Leave', 'Leave entitlement [1]']
    assert result.citations == [1]
    assert result.answer == updates[-1]


def test_streaming_rejects_incomplete_final_json():
    from langchain_core.messages import AIMessageChunk
    rag = object.__new__(RAG)
    rag.ai = Mock()
    rag.ai.chat.bind.return_value = RunnableLambda(
        lambda messages: AIMessageChunk(content='{"answer":"unfinished')
    )
    with pytest.raises(ValueError):
        rag.structured(GeneratedAnswer, 'Use evidence', {}, on_answer=lambda text: None)


def test_citation_retry_receives_failed_answer_and_allowed_labels():
    rag = object.__new__(RAG)
    payloads = []
    invalid = GeneratedAnswer(status='answered', answer='Leave [9].', citations=[9])
    valid = GeneratedAnswer(status='answered', answer='Leave [1].', citations=[1])

    def structured(schema, system, payload, **kwargs):
        payloads.append(json.loads(json.dumps(payload)))
        return invalid if len(payloads) == 1 else valid

    rag.structured = structured
    result = rag.generate('Annual leave?', [{'metadata': {'text': 'Leave policy'}}], [])
    assert result == valid
    correction = payloads[1]['citation_correction']
    assert correction['previous_answer'] == invalid.model_dump()
    assert correction['allowed_labels'] == [1]
    assert 'invalid' in correction['validation_error']


def test_citation_retry_failure_returns_safe_fallback():
    rag = object.__new__(RAG)
    rag.structured = Mock(return_value=GeneratedAnswer(
        status='answered', answer='Unsupported answer.', citations=[]))
    updates = []
    result = rag.generate('Annual leave?', [{'metadata': {'text': 'Leave policy'}}], [],
                          on_answer=updates.append)
    assert result.status == 'insufficient_evidence'
    assert result.citations == []
    assert 'could not validate the source references' in result.answer
    assert 'Unsupported answer.' not in result.answer
    assert updates[-1] == result.answer
    assert validate_answer(result, 1) == result
    assert rag.structured.call_count == 2


def test_citation_correction_does_not_stream_a_second_time():
    rag = object.__new__(RAG)
    calls = []
    invalid = GeneratedAnswer(status='answered', answer='First draft [9].', citations=[9])
    valid = GeneratedAnswer(status='answered', answer='Corrected answer [1].', citations=[1])
    updates = []

    def structured(schema, system, payload, on_answer=None):
        calls.append(on_answer)
        response = invalid if len(calls) == 1 else valid
        if on_answer is not None:
            on_answer(response.answer)
        return response

    rag.structured = structured
    result = rag.generate('Annual leave?', [{'metadata': {'text': 'Leave policy'}}], [],
                          on_answer=updates.append)
    assert updates == ['First draft [9].']
    assert calls[0] is not None
    assert calls[1] is None
    assert result == valid


def test_declared_valid_citations_are_displayed_when_inline_markers_omitted():
    rag = object.__new__(RAG)
    rag.structured = Mock(return_value=GeneratedAnswer(
        status='answered', answer='Full-time employees have 20 planned leave days per calendar year in this fictional scheme.',
        citations=[1]))
    result = rag.generate('How many leaves is an employee eligible for annually?',
                          [{'metadata': {'text': 'Full-time employees have 20 planned leave days per calendar year.'}}], [])
    assert result.status == 'answered'
    assert result.answer.endswith('Sources: [1]')
    assert result.citations == [1]
    assert rag.structured.call_count == 1
    assert validate_answer(result, 1) == result


def test_top_five_evidence_ranked_and_deduplicated():
    groups = [
        [{'id': str(i), 'score': i / 10} for i in range(8)],
        [{'id': '7', 'score': .95}, {'id': 'extra', 'score': .85}],
    ]
    selected = select_evidence(groups)
    assert [item['id'] for item in selected] == ['7', 'extra', '6', '5', '4']
    assert selected[0]['score'] == .95
    assert len(select_evidence([])) == 0


def test_retrieval_requests_five_and_returns_only_five(monkeypatch):
    from langchain_core.documents import Document
    import policy_helper.policy_question_answering as module
    rag = object.__new__(RAG)
    rag.index = Mock()
    rag.ai = Mock()
    rag.settings = NS(namespace='demo', embedding_model='demo', embedding_dimensions=2)
    store = Mock()
    store.similarity_search_by_vector_with_score.return_value = [
        (Document(id=str(i), page_content=f'Policy {i}', metadata={'category': 'hr', 'subcategory': 'leave'}), i / 10)
        for i in range(8)
    ]
    monkeypatch.setattr(module, 'vector_store', lambda *args: store)
    monkeypatch.setattr(module, 'embed', lambda *args: [[0, 1]])
    evidence, _ = rag.retrieve([Route(question='Annual leave?', category='hr', subcategory='leave')])
    assert len(evidence) == 5
    assert [item['id'] for item in evidence] == ['7', '6', '5', '4', '3']
    assert all(call.kwargs['k'] == 5 for call in store.similarity_search_by_vector_with_score.call_args_list)
