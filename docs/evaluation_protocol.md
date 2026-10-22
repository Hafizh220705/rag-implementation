# PharmaGuard-RAG Evaluation Protocol

## Status of the dataset

`evaluation/datasets/pharmaguard_eval_v1.json` contains 24 development
questions: 8 factual, 6 paraphrase, 6 cross-document, and 4
insufficient-evidence cases. The same set is used during development, so it is
not an independent held-out benchmark.

At the time of this document, every case remains `verified=false`. Retrieval
metrics must therefore report coverage as 0/24 and must not be presented as
final accuracy.

## Evidence verification workflow

1. Validate structure and existing references:

   ```powershell
   python scripts/validate_evaluation_dataset.py
   ```

2. Inspect the retrieved candidates for one question. When a Baseline v2
   report is unavailable, the script falls back to Baseline v1:

   ```powershell
   python scripts/review_evidence.py F06
   ```

3. Open the original PDF shown by the command and verify the stated PDF page.
   Check enough surrounding context to ensure the passage has not been
   misinterpreted.

4. Record only genuinely verified chunk IDs in `gold_chunk_ids`, write a
   concise evidence-grounded `reference_answer`, add review notes, and set
   `verified=true` only after completing the source check.

5. Re-run the validator. Unknown chunk IDs, incomplete verified answers, and
   cross-document cases with evidence from fewer than two documents are
   rejected.

## Retrieval metrics

- **Hit@4:** 1 when at least one verified gold chunk occurs in the first four
  retrieved chunks; otherwise 0.
- **MRR@4:** reciprocal rank of the first verified gold chunk within the first
  four results; otherwise 0.
- **Cross-document Coverage@4:** fraction of required gold documents
  represented in the first four results.

Only verified answerable questions are included in these denominators.
Verified insufficient-evidence cases are evaluated separately through
abstention accuracy.

## Human answer review

Create a review file from a complete generation report:

```powershell
python scripts/create_human_review.py
```

For every case, assess:

- `evidence_sufficient`: retrieved passages contain enough information for the
  requested scope.
- `faithfulness`: every factual claim is supported by the cited passage.
- `citation_correctness`: each citation resolves to the passage that supports
  the nearby claim.
- `answer_completeness`: the answer covers the supported parts of the question
  without requiring unsupported detail.
- `abstention_correct`: abstention is appropriate because the corpus lacks the
  requested evidence.

Use `true`, `false`, or `null` when the reviewer cannot decide. Add concrete
notes and set the review record's `verified=true` only after checking the
original document. Report the number of verified reviews separately from the
24-question total.

Structural citation validation does not establish semantic faithfulness.
Status counts are diagnostic outcomes, not accuracy metrics.
