"""Run the draft evaluation questions against the real hosted RAG pipeline.

Start with: uv run --locked python evaluate_policy_answers.py.

Unlike pytest's offline tests, this command uses configured API keys, makes model
and Pinecone requests, and can incur charges. Run after verified ingestion.
It measures routing agreement and saves answers/sources for human review; it
does not automatically judge whether a policy answer is factually supported.
Each case calls the LangChain-backed RAG.ask() independently, without prior chat
history. This script therefore does not measure follow-up resolution or apply
Streamlit's two-word input validation.
"""
import json
from pathlib import Path

from policy_helper.policy_question_answering import RAG
from policy_helper.ai_service_clients import Settings


def main():
    """Compare each live routing result with its draft expectation and save details.

    Used by:
        This file: the if __name__ == "__main__" entry-point block. Run explicitly with uv run --locked python evaluate_policy_answers.py; this makes hosted-service calls.
    """
    cases = json.loads(Path('evaluation/questions.json').read_text(encoding='utf-8'))
    rag = RAG(Settings.read())
    results = []
    # Each JSON case contains the question, expected categories, and expected status.
    # Starting enumerate at 1 makes terminal progress easier to read.
    for number, case in enumerate(cases, 1):
        # Evaluation supplies a user prompt just as the Streamlit input does.
        user_prompt = case['question']
        result = rag.ask(user_prompt)
        # A set removes repeated routes to the same category; sorting makes the
        # comparison independent of the order in which the model listed categories.
        actual = sorted({route['category'] for route in result['routing']['routes']})
        correct = actual == sorted(case['categories']) and result['routing']['status'] == case['status']
        results.append({'case': case, 'routing_correct': correct, 'result': result})
        print(f'{number}/{len(cases)}: {"PASS" if correct else "REVIEW"}')
    # Python treats True as 1 and False as 0 when summing: correct / total is accuracy.
    # Review expected labels before treating this score as an acceptance benchmark.
    report = {'routing_accuracy': sum(r['routing_correct'] for r in results)/len(results), 'cases': results}
    output = Path('artifacts/evaluation.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Routing accuracy: {report["routing_accuracy"]:.1%}. Review answer support in {output}.')


# Importing this module must not launch paid evaluation requests.
if __name__ == '__main__':
    main()
