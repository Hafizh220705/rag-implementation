
# Known Issues — PharmaGuard-RAG Baseline

## KI-001 — False Abstention
- Definition questions sometimes return insufficient_evidence
  despite relevant retrieved passages.
- Example: "Apa yang dimaksud dengan farmakovigilans?"
- Priority: High

## KI-002 — Incomplete Generation
- Some generated answers are truncated.
- Example: obligations of marketing authorization holders.
- Priority: High

## KI-003 — Incomplete Procedural Evidence
- Some procedural questions retrieve related explanations
  but not complete step-by-step evidence.
- Example: adverse drug reaction reporting mechanism.
- Priority: Medium

## KI-004 — Unsupported Citation Risk
- Valid source labels do not guarantee factual faithfulness.
- Requires manual citation and claim verification.
- Priority: High
