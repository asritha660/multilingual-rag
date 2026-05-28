"""
Retrieval evaluation for the Multilingual RAG Assistant.

Runs each question in the test set through the retrieval pipeline and
measures whether the expected information was found in the retrieved chunks.

Computes:
  - Hit Rate (Recall proxy): did ANY retrieved chunk contain an expected keyword?
  - Precision@K: fraction of retrieved chunks that contained an expected keyword.
  - Average latency of retrieval.

Run from the project root (with the venv active):
  python -m evaluation.evaluate_retrieval

NOTE: This uses your retrieval logic directly (no Gemini calls, no API quota).
A document must already be ingested into the vector store (upload sample.pdf first).
"""

import json
import time
import os

# Reuse the retrieval logic from the backend
from backend.main import retrieve


def load_test_set():
    path = os.path.join("evaluation", "test_set.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["tests"]


def keyword_in_chunks(keywords, chunks):
    """Return how many of the retrieved chunks contain at least one expected keyword."""
    text_blocks = [c.lower() for c in chunks]
    hits = 0
    for block in text_blocks:
        if any(kw.lower() in block for kw in keywords):
            hits += 1
    return hits


def main():
    tests = load_test_set()
    print(f"Running retrieval evaluation on {len(tests)} test questions...\n")

    total_hit = 0
    total_precision = 0.0
    total_latency = 0.0

    for i, t in enumerate(tests, start=1):
        question = t["question"]
        keywords = t["expected_keywords"]
        language = t.get("language", "en")

        start = time.time()
        chunks = retrieve(question, language=language)
        elapsed = time.time() - start

        hits = keyword_in_chunks(keywords, chunks)
        hit = 1 if hits > 0 else 0
        precision = hits / len(chunks) if chunks else 0.0

        total_hit += hit
        total_precision += precision
        total_latency += elapsed

        status = "PASS" if hit else "FAIL"
        print(f"[{i}] {status}  ({language})  \"{question[:55]}\"")
        print(f"     retrieved {len(chunks)} chunks, {hits} relevant "
              f"-> precision={precision:.2f}, latency={elapsed:.2f}s")
        if not hit:
            print(f"     >>> EXPECTED one of: {keywords}")
            print(f"     >>> BUT RETRIEVED CHUNKS WERE:")
            for j, ch in enumerate(chunks):
                print(f"         chunk {j}: {ch[:150]}")
    n = len(tests)
    print("\n" + "=" * 55)
    print("RETRIEVAL EVALUATION SUMMARY")
    print("=" * 55)
    print(f"Hit Rate (recall proxy):  {total_hit}/{n} = {total_hit / n:.2f}")
    print(f"Average Precision@K:      {total_precision / n:.2f}")
    print(f"Average retrieval latency:{total_latency / n:.2f}s")


if __name__ == "__main__":
    main()
