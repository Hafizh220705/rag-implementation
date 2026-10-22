from time import perf_counter

from pharmaguard_rag.evidence import build_evidence, render_selection
from pharmaguard_rag.prompting import (
    build_grounded_prompts, build_selection_retry_prompt, selection_schema,
)
from pharmaguard_rag.retrieval_dense import required_document_ids

INSUFFICIENT_ANSWER = "Bukti dalam dokumen yang ditemukan belum cukup untuk menjawab pertanyaan tersebut."
DEFAULT_MINIMUM_RETRIEVAL_SIMILARITY = 0.86
MAX_GENERATION_ATTEMPTS = 2


class RAGPipelineError(RuntimeError):
    """Preserve stage diagnostics when an end-to-end run fails."""

    def __init__(self, stage: str, cause: Exception, retrieved_chunks=None,
                 retrieval_latency_seconds=None, generation_latency_seconds=None,
                 raw_answer: str = ""):
        super().__init__(str(cause))
        self.stage = stage
        self.cause = cause
        self.retrieved_chunks = retrieved_chunks or []
        self.retrieval_latency_seconds = retrieval_latency_seconds
        self.generation_latency_seconds = generation_latency_seconds
        self.raw_answer = raw_answer


class RAGPipeline:
    """Retrieve evidence, let the model select it, validate and render quotes."""

    def __init__(self, retriever, generator,
                 minimum_retrieval_similarity=DEFAULT_MINIMUM_RETRIEVAL_SIMILARITY):
        if not -1 <= minimum_retrieval_similarity <= 1:
            raise ValueError("minimum_retrieval_similarity must be between -1 and 1.")
        self.retriever = retriever
        self.generator = generator
        self.minimum_retrieval_similarity = minimum_retrieval_similarity

    def close(self):
        if hasattr(self.retriever, "close"):
            self.retriever.close()

    def ask(self, question: str, top_k: int = 4) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        start = perf_counter()
        required = required_document_ids(question)
        result = {
            "question": question, "status": "insufficient_evidence",
            "answer": INSUFFICIENT_ANSWER, "raw_answer": "", "initial_raw_answer": "",
            "generation_attempts": 0, "generation_raw_attempts": [],
            "abstention_reason": None, "cited_sources": [], "retrieved_chunks": [],
            "required_document_ids": required, "missing_document_ids": [],
            "evidence_pages": [], "evidence_candidates": [], "selected_evidence": [],
            "validation_errors": [], "generation_diagnostics": [],
            "answer_mode": "evidence_excerpts",
            "retrieval_latency_seconds": None, "generation_latency_seconds": None,
        }

        def finish(reason=None):
            result["abstention_reason"] = reason
            result["latency_seconds"] = round(perf_counter() - start, 3)
            return result

        retrieval_start = perf_counter()
        try:
            chunks = self.retriever.search(question=question, top_k=top_k)
            result["retrieved_chunks"] = chunks
        except Exception as error:
            raise RAGPipelineError("retrieval", error,
                retrieval_latency_seconds=round(perf_counter() - retrieval_start, 3)) from error
        result["retrieval_latency_seconds"] = round(perf_counter() - retrieval_start, 3)
        if not chunks:
            return finish("no_retrieved_chunks")
        missing = set(required) - {chunk["document_id"] for chunk in chunks}
        result["missing_document_ids"] = sorted(missing)
        if missing:
            return finish("missing_required_sources")
        if chunks[0].get("similarity", 1.0) < self.minimum_retrieval_similarity:
            return finish("low_retrieval_similarity")
        try:
            pages = (self.retriever.evidence_pages(chunks)
                     if hasattr(self.retriever, "evidence_pages") else chunks)
            evidence = build_evidence(question, pages)
            result["evidence_pages"] = pages
            result["evidence_candidates"] = evidence
            result["retrieval_latency_seconds"] = round(perf_counter() - retrieval_start, 3)
            if not evidence:
                return finish("no_complete_evidence")
            if set(required) - {item["document_id"] for item in evidence}:
                return finish("missing_complete_required_evidence")
            system, prompt = build_grounded_prompts(question, evidence, pages, required)
        except Exception as error:
            raise RAGPipelineError("prompting", error, chunks,
                                   result["retrieval_latency_seconds"]) from error

        generation_start = perf_counter()
        current_prompt = prompt
        for _ in range(MAX_GENERATION_ATTEMPTS):
            try:
                raw = self.generator.generate(system_prompt=system, user_prompt=current_prompt,
                                              response_format=selection_schema(evidence, required))
            except Exception as error:
                raise RAGPipelineError("generation", error, chunks,
                    result["retrieval_latency_seconds"],
                    round(perf_counter() - generation_start, 3),
                    getattr(error, "partial_response", "")) from error
            result["generation_raw_attempts"].append(raw)
            result["generation_attempts"] += 1
            result["raw_answer"] = raw
            result["initial_raw_answer"] = (result["generation_raw_attempts"][0]
                                            if result["generation_attempts"] > 1 else "")
            result["generation_diagnostics"].append(getattr(self.generator, "last_diagnostics", {}))
            result["generation_latency_seconds"] = round(perf_counter() - generation_start, 3)
            try:
                answer, sources, selected = render_selection(raw, evidence, pages, required)
                if not selected:
                    return finish("model_abstention")
                result.update(status="answered", answer=answer, cited_sources=sources,
                              selected_evidence=selected)
                return finish()
            except ValueError as error:
                result["validation_errors"].append(str(error))
                current_prompt = build_selection_retry_prompt(prompt, str(error))
        return finish("evidence_validation_failed")
