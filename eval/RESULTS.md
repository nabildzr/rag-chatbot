# Retrieval evaluation

`go run ./cmd/eval` uploads `eval/corpus` (2 English, 2 Indonesian documents),
asks the 37 questions in `eval/questions.json` through `POST /search` across all
indexed documents, and checks whether the expected section comes back.

- **same**: question in the document's language
- **cross**: question in the other language (Indonesian ↔ English)
- **offtopic**: nothing relevant exists; correct when no source is returned

Each row below is a configuration change measured on the same corpus.

## 1. Baseline: nomic-embed-text, dense only, MIN_SCORE 0.64

| type | n | hit@1 | hit@5 | MRR |
|---|---|---|---|---|
| same | 17 | 76% | 94% | 0.83 |
| cross | 15 | 20% | 27% | 0.22 |
| **in-scope total** | 32 | 50% | 62% | 0.54 |
| offtopic (no sources) | 5 | 40% | | |
