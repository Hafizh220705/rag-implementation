from time import perf_counter
from pharmaguard_rag.citations import resolve_citations
from pharmaguard_rag.prompting import build_grounded_prompts

INSUFFICIENT_ANSWER = (
    "Bukti dalam dokumen yang ditemukan belum cukup "
    "untuk menjawab pertanyaan tersebut."
)

class RAGPipeline:
    """End-to-end retrieval-augmented generation."""

    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def ask(
        self,
        question: str,
        top_k: int = 4,
    ) -> dict:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        start_time = perf_counter()

        # Step 1: Retrieve relevant passages
        retrieved_chunks = self.retriever.search(
            question=question,
            top_k=top_k,
        )

        if not retrieved_chunks:
            return {
                "question": question,
                "status": "insufficient_evidence",
                "answer": INSUFFICIENT_ANSWER,
                "raw_answer": "",
                "cited_sources": [],
                "retrieved_chunks": [],
                "latency_seconds": round(
                    perf_counter() - start_time, 3
                ),
            }

        # Step 2: Build evidence-grounded prompts
        system_prompt, user_prompt = build_grounded_prompts(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        # Step 3: Generate answer with Ollama
        raw_answer = self.generator.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Step 4: Handle model abstention
        if raw_answer.strip().upper().startswith(
            "BUKTI_TIDAK_CUKUP"
        ):
            status = "insufficient_evidence"
            answer = INSUFFICIENT_ANSWER
            cited_sources = []

        else:
            # Step 5: Validate citation labels
            try:
                answer, cited_sources = resolve_citations(
                    answer=raw_answer,
                    retrieved_chunks=retrieved_chunks,
                )

                status = "answered"

            except ValueError:
                status = "citation_error"
                answer = (
                    "Model menghasilkan jawaban tanpa "
                    "sitasi sumber yang valid. "
                    "Jawaban tidak ditampilkan."
                )
                cited_sources = []

        return {
            "question": question,
            "status": status,
            "answer": answer,
            "raw_answer": raw_answer,
            "cited_sources": cited_sources,
            "retrieved_chunks": retrieved_chunks,
            "latency_seconds": round(
                perf_counter() - start_time, 3
            ),
        }