"""
Faithfulness evaluation for the Multilingual RAG Assistant (LLM-as-judge).

For each test question:
  1. Retrieve chunks and generate an answer (1 Gemini call).
  2. Ask Gemini to judge whether the answer is grounded in the retrieved
     sources, scoring 1 (faithful) or 0 (unfaithful/hallucinated) (1 Gemini call).

Computes:
  - Faithfulness Score: average grounding score across questions.
  - Hallucination Rate: fraction of answers judged unfaithful.

Run from the project root (venv active):
  python -m evaluation.evaluate_faithfulness

FREE-TIER NOTE: gemini-2.5-flash free tier allows ~5 requests/minute and ~20/day.
Each question uses 2 calls. PAUSE_SECONDS spaces calls out to respect the per-minute
limit. With PAUSE_SECONDS=30 and 6 questions, a full run takes a few minutes but
stays under the limits. The script also retries automatically on a 429.
"""

import json
import os
import time

from backend.main import retrieve, generate_answer, client

MAX_QUESTIONS = 6        # how many test questions to evaluate
PAUSE_SECONDS = 30       # pause between questions to respect 5/minute limit
MAX_RETRIES = 3          # retries if a 429 (rate limit) is hit


def load_test_set():
    path = os.path.join("evaluation", "test_set.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["tests"]


def call_with_retry(fn, *args):
    """Call a function; if a 429 rate-limit error occurs, wait and retry."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = 35  # wait just over a minute window
                print(f"     (rate limited; waiting {wait}s and retrying...)")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded max retries due to rate limiting.")


def judge_faithfulness(question, sources, answer):
    """Ask Gemini to judge if the answer is grounded in the sources. Returns 1 or 0."""
    context = "\n\n".join(sources)
    judge_prompt = f"""You are a strict evaluator. Decide whether the ANSWER is fully
supported by the CONTEXT. If every claim in the answer is supported by the context,
reply with exactly "1". If the answer contains any information not present in the
context (a hallucination), reply with exactly "0". Reply with only "1" or "0".

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

VERDICT (1 or 0):"""
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=judge_prompt,
    )
    verdict = response.text.strip()
    return 1 if verdict.startswith("1") else 0


def main():
    tests = load_test_set()[:MAX_QUESTIONS]
    print(f"Running faithfulness evaluation on {len(tests)} questions "
          f"(~{len(tests) * 2} Gemini calls, ~{PAUSE_SECONDS}s between each)...\n")

    total_faithful = 0
    completed = 0

    for i, t in enumerate(tests, start=1):
        question = t["question"]
        language = t.get("language", "en")

        chunks = retrieve(question, language=language)
        answer = call_with_retry(generate_answer, question, chunks)
        score = call_with_retry(judge_faithfulness, question, chunks, answer)

        total_faithful += score
        completed += 1
        verdict = "FAITHFUL" if score else "HALLUCINATED"
        print(f"[{i}] {verdict}  \"{question[:50]}\"")
        print(f"     answer: {answer[:80]}")

        if i < len(tests):
            time.sleep(PAUSE_SECONDS)

    print("\n" + "=" * 55)
    print("FAITHFULNESS EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Questions completed: {completed}")
    print(f"Faithfulness Score:  {total_faithful}/{completed} = "
          f"{total_faithful / completed:.2f}" if completed else "no data")
    print(f"Hallucination Rate:  {(completed - total_faithful) / completed:.2f}"
          if completed else "")


if __name__ == "__main__":
    main()