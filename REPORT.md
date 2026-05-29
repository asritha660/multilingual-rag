# Multilingual RAG — Evaluation Report

## Overview

This report documents the measured performance of the Multilingual RAG Assistant on a curated test set. Evaluation covers retrieval quality, faithfulness of generated answers, and end-to-end latency. The test set lives in `evaluation/test_set.json` and the scripts that produced these numbers are `evaluation/evaluate_retrieval.py` and `evaluation/evaluate_faithfulness.py`.

## Test Corpus

The system is evaluated against `test_document.pdf`, a fictional "Brindleton Bike Share Rider Handbook." The document is approximately 10 KB of structured prose covering program overview, fleet, membership tiers, station network, safety rules, damage and theft policy, customer support, privacy, and an FAQ. It is intentionally varied in structure (prose paragraphs, a pricing table, numbered FAQ sections) to exercise different aspects of chunking and retrieval. The corpus is fully fabricated and was authored specifically as a test artifact; it has no overlap with the system's training data or any real organization.

After ingestion the document is split into 46 chunks of 250 characters with 50-character overlap, embedded with the `paraphrase-multilingual-MiniLM-L12-v2` model, and stored in a persistent ChromaDB collection.

## Test Set

Seven questions: six in English and one in Hindi. The Hindi question validates the multilingual retrieval path, which skips the English-only cross-encoder reranker and uses hybrid top-6 instead of reranked top-4. The final question is a deliberate stress-test for keyword dilution: the term "payment" appears in three sections of the document but only one section actually answers the question.

| # | Question | Language | Expected Keywords |
|---|---|---|---|
| 1 | What types of bikes are in the fleet? | English | pedal, electric |
| 2 | How much does the annual membership cost? | English | 169 |
| 3 | What is the minimum age requirement to use the service? | English | 16 |
| 4 | When does the program suspend service? | English | severe, weather |
| 5 | What is the maximum charge for bike damage? | English | 250 |
| 6 | इस सेवा का उपयोग करने की न्यूनतम उम्र क्या है? (Hindi: "What is the minimum age to use this service?") | Hindi | 16 |
| 7 | What payment methods are accepted? (stress-test for keyword dilution) | English | Visa, Mastercard, American Express |

## Retrieval Results

Measured by `evaluate_retrieval.py`. Hit Rate is a recall proxy: a question is a "hit" if at least one expected keyword appears in any of the retrieved chunks. Precision@K counts the fraction of retrieved chunks that contain an expected keyword.

| Metric | Value |
|---|---|
| Hit Rate (recall proxy) | **7/7 = 1.00** |
| Average Precision@K | **0.35** |
| Average retrieval latency | **0.17s** |

### Per-question breakdown

| # | Retrieved | Relevant | Precision | Latency |
|---|---|---|---|---|
| 1 | 4 | 3 | 0.75 | 0.48s |
| 2 | 4 | 1 | 0.25 | 0.22s |
| 3 | 4 | 2 | 0.50 | 0.16s |
| 4 | 4 | 1 | 0.25 | 0.09s |
| 5 | 4 | 1 | 0.25 | 0.09s |
| 6 (Hindi) | 6 | 1 | 0.17 | 0.03s |
| 7 (stress) | 4 | 1 | 0.25 | 0.11s |

## Faithfulness Results

Measured by `evaluate_faithfulness.py` using Gemini 2.5 Flash as an LLM judge. The judge is given the retrieved context and the generated answer, and asked whether every claim in the answer is supported by the context (binary 0/1 per question, then averaged).

| Metric | Value |
|---|---|
| Faithfulness Score | **6/6 = 1.00** |
| Hallucination Rate | **0.00** |

Every generated answer in the evaluated set was judged fully grounded in the retrieved context. The Hindi answer correctly returned "16 वर्ष" (16 years), confirming both the multilingual generation path and the language-matching behavior.

Note: the faithfulness script evaluates 6 questions rather than 7 (the stress-test question was added to the retrieval test set after the faithfulness script was last revised). Expanding the faithfulness coverage to all 7 questions is a small, mechanical follow-up.

## End-to-End Latency

Captured per-query in PostgreSQL via the `query_logs` table populated by the FastAPI backend.

| Stage | Time |
|---|---|
| Retrieval (vector + BM25 + optional rerank) | ~0.17-0.41s |
| Generation (Gemini 2.5 Flash) | ~2.6s |
| **Total end-to-end** | **~2.9s** |

Generation dominates total latency, consistent with public benchmarks for Gemini 2.5 Flash on prompts of this size. Retrieval latency varies with whether the cross-encoder reranker runs (English questions, ~0.2-0.5s) or is skipped for non-English (~0.03-0.1s).

## Precision Analysis

Hit Rate is saturated at 7/7, including a stress-test question designed to probe keyword dilution. The correct chunk is consistently found in the top-K results. **The genuine engineering weakness is precision, not recall.** Across most questions, only 1 of the 4 retrieved chunks actually contains the answer; the other 3 are topically related but non-answering distractors.

This is the expected behavior of a small-corpus retrieval system at K=4: once the corpus is small enough and the embedding model strong enough, recall saturates quickly, and the next axis to optimize becomes precision. Concrete next steps would be:

1. **Tighten K from 4 to 2-3.** This accepts a small recall risk for cleaner prompts to the LLM, which often improves answer quality and reduces token usage.
2. **Stricter rerank threshold.** The cross-encoder already orders chunks by relevance; thresholding on the rerank score (rather than always taking top-K) would drop low-confidence distractors.
3. **Section-level metadata.** Tagging chunks at ingest time (FAQ vs. policy vs. fleet description) would let the retriever bias toward the right section type for "what payment methods" or "what is the age limit" style questions.

The first two are 30-minute changes. The third is a more substantial reingest-and-test cycle.

## Comparison to Baseline

A reasonable baseline for a single-retriever system on a 7-question test set would be approximately 0.6-0.8 hit rate with comparable precision. **7/7 hit rate with hybrid retrieval plus conditional reranking is above baseline**, and the system also recovers cleanly on a deliberately constructed keyword-dilution stress test. Faithfulness at 1.0 with a 0% hallucination rate indicates that when retrieval succeeds, generation is reliably grounded.

The honest caveat is that this test set is small and was authored by the same person who tuned the pipeline. A larger, independently-authored test set would be the next rigor step.

## Reproducibility

To reproduce these numbers:

```powershell
# Activate venv
.\venv\Scripts\Activate.ps1

# Ingest the bundled test corpus (one time)
python backend\ingest.py test_document.pdf

# Retrieval evaluation (deterministic, no LLM calls)
python -m evaluation.evaluate_retrieval

# Faithfulness evaluation (uses Gemini quota; ~3-5 min with throttling)
python -m evaluation.evaluate_faithfulness
```

Retrieval evaluation is fully deterministic given a fixed test set and ingested corpus. Faithfulness has minor variance because the LLM judge is non-deterministic, but in repeated runs against this corpus the score has been stable at 1.0.

## Limitations

- **Small test set (7 questions).** Results are directionally meaningful but not statistically tight. A production deployment should expand this to 50-100 questions across multiple document types and languages.
- **Single document corpus.** Multi-document retrieval has not been stress-tested.
- **Faithfulness coverage of 6/7 questions** (the stress-test question was added to the retrieval set after the faithfulness script was last revised).
- **Faithfulness judge is the same model family as the generator** (Gemini judging Gemini). A stronger judge model or a different judge family would give a more independent measurement.
- **No translation-quality metric** for the Hindi path. The Hindi answer was qualitatively verified to be in Hindi and topically correct, but no BLEU- or COMET-style score is computed.
- **Free-tier rate limits** on Gemini (20 requests/day, 5/minute) constrain how often the faithfulness eval can be run end-to-end.

## Conclusion

The Multilingual RAG Assistant achieves saturated recall (7/7 hit rate) and perfect faithfulness (6/6, 0% hallucinations) with ~2.9s end-to-end latency on a 7-question multilingual test set that includes a deliberate keyword-dilution stress test. The system's real remaining weakness is precision (0.35 average), which is the expected ceiling for K=4 on a small corpus and has clear, concrete mitigation paths. The evaluation harness — versioned test set, separate retrieval and faithfulness scripts, latency telemetry in PostgreSQL — is in place and can be re-run against any future change to chunking, retrieval, or generation.
