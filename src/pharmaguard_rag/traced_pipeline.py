from time import perf_counter
from pharmaguard_rag.observability import OptionalLangfuse
from pharmaguard_rag.pipeline import RAGPipeline

class TracedRetriever:
    """Add tracing without changing retrieval logic."""

    def __init__(self, retriever, observer):
        self.retriever = retriever
        self.observer = observer

    def search(self, question: str, top_k: int = 4):
        with self.observer.observation(
            name="dense-retrieval",
            as_type="retriever",
            input={
                "question": question,
                "top_k": top_k,
            },
            metadata={
                "retrieval_method": "dense",
                "vector_store": "ChromaDB",
            },
        ) as span:

            start = perf_counter()

            results = self.retriever.search(
                question=question,
                top_k=top_k,
            )

            latency = perf_counter() - start

            retrieved_info = [
                {
                    "rank": chunk["rank"],
                    "chunk_id": chunk["chunk_id"],
                    "document_id": chunk["document_id"],
                    "pdf_page": chunk["pdf_page"],
                    "similarity": chunk["similarity"],
                }
                for chunk in results
            ]

            self.observer.update(
                span,
                output={
                    "retrieved_count": len(results),
                    "chunks": retrieved_info,
                },
                metadata={
                    "latency_seconds": round(latency, 3),
                },
            )

            return results

class TracedGenerator:
    """Add tracing without changing Ollama generation."""

    def __init__(self, generator, observer):
        self.generator = generator
        self.observer = observer
        self.model_name = getattr(
            generator,
            "model_name",
            "unknown",
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        **kwargs,
    ) -> str:

        with self.observer.observation(
            name="ollama-generation",
            as_type="generation",
            model=self.model_name,
            input={
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            metadata={
                "provider": "ollama",
                "temperature": 0,
            },
        ) as span:

            start = perf_counter()

            response = self.generator.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                **kwargs,
            )

            latency = perf_counter() - start

            self.observer.update(
                span,
                output=response,
                metadata={
                    "latency_seconds": round(latency, 3),
                },
            )

            return response

class TracedRAGPipeline:
    """Run the existing RAG pipeline with Langfuse traces."""

    def __init__(
        self,
        retriever,
        generator,
        observer=None,
    ):
        self.observer = (
            observer
            if observer is not None
            else OptionalLangfuse()
        )

        self.pipeline = RAGPipeline(
            retriever=TracedRetriever(
                retriever,
                self.observer,
            ),
            generator=TracedGenerator(
                generator,
                self.observer,
            ),
        )

    def ask(
        self,
        question: str,
        top_k: int = 4,
    ) -> dict:

        with self.observer.observation(
            name="pharmaguard-rag",
            as_type="span",
            input={
                "question": question,
                "top_k": top_k,
            },
            metadata={
                "application": "PharmaGuard-RAG",
                "pipeline_version": "baseline-v1",
            },
        ) as span:

            result = self.pipeline.ask(
                question=question,
                top_k=top_k,
            )

            self.observer.update(
                span,
                output={
                    "status": result["status"],
                    "answer": result["answer"],
                    "cited_source_count": len(
                        result["cited_sources"]
                    ),
                },
                metadata={
                    "latency_seconds": result[
                        "latency_seconds"
                    ],
                },
            )

            return result

    def flush(self):
        """Send pending traces before the script exits."""

        self.observer.flush()