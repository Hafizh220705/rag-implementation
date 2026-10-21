# PharmaGuard-RAG — Project Plan (Initial)

## Assessment and scope
- Assessment: KALBE AI Research Engineer Intern · Multimodal AI — Option 1 (RAG Implementation).
- Research scope: Indonesian-language questions answered from selected public BPOM pharmacovigilance PDFs.
- Intended user: a reader looking for source-backed explanations of published pharmacovigilance documents.
- Non-goals: diagnosis, patient-specific medical guidance, automated regulatory compliance decisions, multimodal OCR/VLM, deployment.

## Problem statement
Pharmacovigilance knowledge appears across Indonesian regulations, training modules, and practical guidance. Search can be tedious, and sources differ in legal status and publication year. This proof of concept will retrieve relevant passages and generate traceable, evidence-grounded answers.

## Research question and hypothesis
**RQ:** Does hybrid retrieval (dense + BM25 using reciprocal rank fusion) improve evidence retrieval compared with dense-only retrieval on a curated Indonesian pharmacovigilance corpus?

**Hypothesis:** Hybrid retrieval *may* help for queries with exact terms and paraphrased expressions. This is untested and must not be reported as an empirical result.

## Baseline architecture (to implement)
BPOM PDFs → PDF extraction → page-aware chunking and metadata → multilingual E5 embeddings → ChromaDB dense retrieval → Qwen2.5:3b via Ollama → Indonesian answer + verifiable document/page/chunk IDs.

## Planned observability / experiment
- Langfuse spans for retrieval, context construction, generation, and citation validation. It is optional to RAG functionality.
- Gold set planned: 24 questions (8 factual, 6 paraphrase, 6 cross-document, 4 insufficient-evidence).
- Retrieval metrics: Hit@4, MRR@4, Evidence Coverage@4. Compute on answerable cases.
- Manual checks: faithfulness, answer relevance, citation correctness, insufficient-evidence handling.
- Optional experiment: dense-only vs dense+BM25 RRF, holding corpus/chunks, evaluation set, and final Top-K constant.
- Measurements: retrieval and generation latency; report measured results only.

## Data governance
- Candidate corpus: four BPOM documents listed in `data/manifest.json`.
- The 2026 BPOM regulation has a different legal status from the 2019–2020 educational references. Never assume those older references set current legal obligations.
- Audit PDF text and pages before finalizing corpus. Save checksums and extracted character counts.
- Do not rehost source PDFs without confirming applicable redistribution terms. Source links and a retrieval procedure are sufficient for this exercise.

## Completion criteria
1. CLI works end-to-end with document ingestion, dense retrieval, LLM generation, source attribution.
2. README has exact models, dependencies, hardware/setup info and runnable commands.
3. Saved questions, retrieved evidence, answers and references are available in repo.
4. Observed results and limitations are distinguished from untested hypotheses.
5. External sources and use of AI assistance are disclosed truthfully.

## Stage 1 gate
- Python 3.11 environment and project install pass.
- Ollama Qwen2.5:3b local inference responds.
- multilingual-e5-small encodes an Indonesian query/passage pair.
- Candidate corpus is downloaded and audited for readable pages, counts, and SHA-256 checksums.
