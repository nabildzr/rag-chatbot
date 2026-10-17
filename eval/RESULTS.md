# Retrieval evaluation

`go run ./cmd/eval` uploads `eval/corpus` and asks the 37 questions in
`eval/questions.json` through `POST /search`, across all indexed documents, then
checks whether the expected section comes back. Add `-tune` to search without
score filters and get a suggested `MIN_SCORE`.

The corpus has 4 documents with questions (2 English, 2 Indonesian) and 4
distractor documents in other formats (DOCX, PPTX, TXT, MD) that no question
targets.

- **same**: question in the document's language
- **cross**: question in the other language (Indonesian ↔ English)
- **offtopic**: nothing relevant exists; correct when no source is returned

Each section below is one configuration, measured on the same corpus.

## 1. nomic-embed-text, dense only, MIN_SCORE 0.64

| type | n | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| same | 17 | 76% | 100% | 0.84 |
| cross | 15 | 20% | 27% | 0.22 |
| **in-scope total** | 32 | 50% | 66% | 0.55 |
| offtopic (no sources) | 5 | 40% | | |

nomic-embed-text is trained mostly on English: questions in Indonesian about
English documents (and the reverse) barely retrieve anything.

## 2. bge-m3, dense only, MIN_SCORE 0.45

| type | n | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| same | 17 | 100% | 100% | 1.00 |
| cross | 15 | 93% | 100% | 0.97 |
| **in-scope total** | 32 | 97% | 100% | 0.98 |
| offtopic (no sources) | 5 | 80% | | |

Threshold from `-tune`: the expected chunk always scored ≥ 0.482, the best
off-topic chunk ≤ 0.475. 0.45 sits below that gap on purpose (favouring
recall), so one off-topic question still returns a weak source; the model then
answers that it found nothing.

## 3. bge-m3 + hybrid search (dense + BM25, RRF)

Ten **keyword** questions were added: short queries that are an exact term from
the text (`/readyz`, `OOMKilled`, `QRIS`, `Rp3.500`, `Mendut`, ...). Same data,
dense-only (`HYBRID_SEARCH=false`) vs hybrid:

| type | n | dense hit@1 | hybrid hit@1 | dense MRR | hybrid MRR |
|---|---|---|---|---|---|
| same | 17 | 100% | 100% | 1.00 | 1.00 |
| cross | 15 | 93% | 93% | 0.97 | 0.97 |
| keyword | 10 | 80% | **100%** | 0.83 | **1.00** |
| **in-scope total** | 42 | 93% | **98%** | 0.95 | **0.99** |
| offtopic (no sources) | 5 | 80% | 80% | | |

Dense-only missed `Mendut` entirely (its cosine was below `MIN_SCORE`) and ranked
`Rp3.500` third. Hybrid orders candidates by reciprocal rank fusion of the dense
and BM25 rankings, and keeps a chunk that contains every keyword of the
question even when its cosine is under the threshold.

A first attempt that only used BM25 to add candidates and then re-ranked
everything by cosine scored exactly like dense-only: on a corpus this small the
dense top 20 already contains every chunk, so keywords must influence the
ordering and the threshold, not only the candidate set.

## 4. Reranking (optional, off by default)

Same data as section 3, plus the documents uploaded during development as extra
distractors. The reranker re-scores the 10 best fused candidates.

| configuration | in-scope hit@1 | MRR | off-topic rejected | time per question (CPU) |
|---|---|---|---|---|
| hybrid, no reranker | **98%** | **0.99** | 60% | ~0.2 s |
| + mmarco-mMiniLMv2-L12 (118M) | 90% | 0.95 | (needs tuning) | ~1.5 s |
| + bge-reranker-v2-m3 (568M) | not finished | | | ~60 s |

On a CPU-only machine reranking does not pay off here: the small multilingual
cross-encoder ranks worse than hybrid search (it scored one correct chunk 0.009),
and bge-reranker-v2-m3 is far too slow. Reranking is therefore opt-in
(`RERANK_URL`) and meant for a GPU host.

Run evaluations against a separate collection so test documents do not end up
next to real ones, e.g. start a second API with
`COLLECTION=eval PORT=18080 CHAT_RATE_PER_MIN=0 INGEST_RATE_PER_MIN=0` and pass
`-api http://localhost:18080`.
