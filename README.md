# PharmaGuard-RAG

Evidence-Grounded RAG for Indonesian Pharmacovigilance Documents.

> **Status: Stage 1 project starter only.** This repository does **not** implement ingestion, retrieval, generation, attribution, or Langfuse yet. Those come in later milestones.

## Windows / PowerShell setup
1. Install Python 3.11, Git, and Ollama (if not installed already).
2. Open PowerShell at the project root, then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,models]"
Copy-Item .env.example .env
python scripts/check_environment.py
```

If PowerShell blocks activation, you can use `.\.venv\Scripts\python.exe` directly without changing execution policy.

## Model checks
```powershell
ollama pull qwen2.5:3b
ollama list
python scripts/smoke_ollama.py
python scripts/smoke_embedding.py
```

The embedding test downloads the model from Hugging Face on first use and intentionally runs on CPU to avoid CUDA setup issues during Stage 1.

## Corpus download
Read `data/manifest.json` for the **official source URLs** and the exact local filenames expected. Download each source PDF to `data/raw/` using those filenames. The 2026 regulation source is an official JDIH landing page; use its document download button.

```powershell
python scripts/audit_corpus.py
```

The audit prints document page/character counts and saves results to `outputs/evaluations/corpus_audit.json`. Low-text pages need manual review (covers, scanned pages, and figures can be expected); this heuristic does not certify ingestion quality.

## Stage 1 definition of done
- Python 3.11 installed and package editable install passes.
- Qwen2.5:3b and multilingual-e5-small smoke tests pass locally.
- The four PDF candidates are downloaded and audited.
- Source/legal status differences are understood and the candidate corpus is frozen for Stage 2.
- `docs/project_plan.md` reviewed and committed on `chore/setup`.

## Assessment requirements
KALBE Option 1 requires end-to-end ingestion, retrieval, LLM answer generation, attribution, reproducibility and saved examples. The baseline is not done until all of those are verified. Optional experiments (Langfuse, hybrid retrieval) come later. Cite reused resources and disclose AI assistance accurately.


## Observability with Langfuse

PharmaGuard-RAG supports optional Langfuse tracing.

Each RAG execution can record:

- User question and final answer
- Retrieved chunk IDs and similarity scores
- Retrieval latency
- LLM prompts and generated response
- Generation latency
- Pipeline status and total latency

### Configuration

Set the following environment variables in `.env`:

LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com

Set `LANGFUSE_ENABLED=false` to run without tracing.

### Run Tracing Test

```powershell
python scripts/test_langfuse_manual.py
python scripts/test_rag_traced_manual.py
```

### Known Limitations

- Tracing does not guarantee answer correctness.
- The baseline may incorrectly abstain despite relevant evidence.
- Citation validity does not guarantee claim-level faithfulness.
- Langfuse Cloud receives the telemetry data configured for tracing.
- Evaluation and failure analysis are performed separately.

## AI Assistance Disclosure

AI assistance was used for development guidance,
code drafting, debugging, and documentation.
The implementation, experiments, and results must
be reviewed and verified by the project author.

