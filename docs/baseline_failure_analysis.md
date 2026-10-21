# PharmaGuard-RAG — Baseline Failure Analysis

## Experiment Overview

- Total questions: 24
- Human-verified reviews: 0
- Retrieval: multilingual-e5-small + ChromaDB
- Generation: Qwen2.5 3B via Ollama

## Failure Classification

| Classification | Count |
|---|---:|
| answered_pending_review | 8 |
| citation_failure | 1 |
| expected_abstention | 4 |
| possible_false_abstention | 9 |
| technical_failure | 2 |

## Cases Requiring Investigation

| ID | Classification | Status |
|---|---|---|
| F03 | citation_failure | citation_error |
| F05 | technical_failure | execution_error |
| F06 | possible_false_abstention | insufficient_evidence |
| F08 | possible_false_abstention | insufficient_evidence |
| P01 | possible_false_abstention | insufficient_evidence |
| P02 | possible_false_abstention | insufficient_evidence |
| P03 | technical_failure | execution_error |
| P06 | possible_false_abstention | insufficient_evidence |
| C01 | possible_false_abstention | insufficient_evidence |
| C02 | possible_false_abstention | insufficient_evidence |
| C03 | possible_false_abstention | insufficient_evidence |
| C04 | possible_false_abstention | insufficient_evidence |

## Known Limitations

- Gold annotations are not yet complete.
- Retrieval Hit@4 and MRR@4 are not final.
- Citation structure does not prove faithfulness.
- Failure classifications require human review.

## Next Steps

1. Verify gold evidence against source PDFs.
2. Review retrieved evidence for rejected questions.
3. Investigate generation truncation and citation errors.
4. Evaluate answer faithfulness and completeness.
5. Apply revisions and rerun the same evaluation set.
