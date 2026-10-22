import argparse
import hashlib
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.generation import OllamaGenerator
from pharmaguard_rag.pipeline import RAGPipeline, RAGPipelineError
from pharmaguard_rag.retrieval_dense import DenseRetriever
from pharmaguard_rag.traced_pipeline import TracedRAGPipeline
from pharmaguard_rag.vector_store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
INDEX_DIR = ROOT / "data/indexes/chroma"


def build_pipeline(trace: bool):
    if not CHUNKS_PATH.is_file():
        raise FileNotFoundError(
            "Processed chunks are missing. Run scripts/build_chunks.py."
        )

    if not INDEX_DIR.is_dir():
        raise FileNotFoundError(
            "Chroma index is missing. Run scripts/build_index.py."
        )

    corpus_sha256 = hashlib.sha256(
        CHUNKS_PATH.read_bytes()
    ).hexdigest()
    embedder = E5Embedder(device="cpu")
    store = ChromaVectorStore(
        persist_directory=INDEX_DIR,
        model_name=embedder.model_name,
        corpus_sha256=corpus_sha256,
    )
    retriever = DenseRetriever(embedder, store)
    generator = OllamaGenerator()

    if trace:
        return TracedRAGPipeline(retriever, generator)

    return RAGPipeline(retriever, generator)


def print_result(result, show_passages):
    print("\nStatus:", result["status"])
    print("\nJawaban:\n", result["answer"], sep="")

    if result["cited_sources"]:
        print("\nSumber yang disitasi:")
        for source in result["cited_sources"]:
            print(
                f"- {source['document_id']} | PDF hlm. "
                f"{source['pdf_page']} | {source['chunk_id']}"
            )

    if show_passages:
        print("\nPassage hasil retrieval:")
        for chunk in result["retrieved_chunks"]:
            preview = " ".join(chunk["text"].split())[:500]
            print(
                f"\n[{chunk['rank']}] {chunk['chunk_id']} | "
                f"similarity={chunk['similarity']:.4f}\n{preview}"
            )

    print(
        "\nLatency: retrieval={retrieval}s generation={generation}s "
        "total={total}s".format(
            retrieval=result["retrieval_latency_seconds"],
            generation=result["generation_latency_seconds"],
            total=result["latency_seconds"],
        )
    )


def main() -> int:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Ask PharmaGuard-RAG from the terminal."
    )
    parser.add_argument("question", nargs="*")
    parser.add_argument(
        "--top-k",
        type=int,
        default=int(os.getenv("TOP_K", "4")),
    )
    parser.add_argument(
        "--show-passages",
        action="store_true",
        help="Print retrieved evidence passages.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Use optional Langfuse tracing configuration.",
    )
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    try:
        pipeline = build_pipeline(args.trace)
    except Exception as error:
        print(f"Setup error: {error}", file=sys.stderr)
        return 1

    questions = [" ".join(args.question).strip()]
    interactive = not questions[0]

    if interactive:
        print("PharmaGuard-RAG interactive CLI")
        print("Ketik 'quit' untuk keluar.")

    try:
        while True:
            if interactive:
                try:
                    question = input("\nPertanyaan: ").strip()
                except EOFError:
                    break
                if question.lower() in {"quit", "exit"}:
                    break
                if not question:
                    continue
            else:
                question = questions[0]

            try:
                result = pipeline.ask(question, top_k=args.top_k)
                print_result(result, args.show_passages)
            except RAGPipelineError as error:
                print(f"Pipeline error at {error.stage}: {error}", file=sys.stderr)
                return 1

            if not interactive:
                break
    finally:
        pipeline.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
