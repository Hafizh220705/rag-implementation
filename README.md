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
