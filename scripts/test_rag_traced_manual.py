import hashlib
import json
from pathlib import Path
from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.vector_store import ChromaVectorStore
from pharmaguard_rag.retrieval_dense import DenseRetriever
from pharmaguard_rag.generation import OllamaGenerator
from pharmaguard_rag.traced_pipeline import TracedRAGPipeline

ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
INDEX_DIR = ROOT / "data/indexes/chroma"
OUTPUT_PATH = ROOT / "outputs/examples/rag_langfuse_smoke.json"

MODEL_NAME = "intfloat/multilingual-e5-small"

QUESTIONS = [
    "Apa yang dimaksud dengan farmakovigilans?",
    "Apa kewajiban pemilik izin edar dalam pelaksanaan farmakovigilans?",
    "Bagaimana mekanisme pelaporan efek samping obat?",
    "Bagaimana prosedur mengganti oli mesin kendaraan?",
]

def main():
    print("=== PHARMAGUARD END-TO-END RAG TEST ===")

    corpus_sha256 = hashlib.sha256(
        CHUNKS_PATH.read_bytes()
    ).hexdigest()

    # Initialize embedding model
    embedder = E5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
    )

    # Load existing vector database
    store = ChromaVectorStore(
        persist_directory=INDEX_DIR,
        model_name=MODEL_NAME,
        corpus_sha256=corpus_sha256,
    )

    print(f"Indexed chunks: {store.count()}")

    # Initialize retrieval and generation
    retriever = DenseRetriever(embedder, store)
    generator = OllamaGenerator()

    pipeline = TracedRAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    results = []

    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        result = pipeline.ask(
            question=question,
            top_k=4,
        )

        results.append(result)

        print("Status:", result["status"])
        
        if result["status"] != "answered":
            print(
                "Raw Model Output:",
                repr(result["raw_answer"]),
            )

        print("Latency:", result["latency_seconds"], "s")

        print("\nANSWER:")
        print(result["answer"])

        print("\nCITED SOURCES:")
        for source in result["cited_sources"]:
            print(
                f"- {source['chunk_id']} | "
                f"{source['title']} | "
                f"PDF Page {source['pdf_page']}"
            )

        print("\nRETRIEVED EVIDENCE:")
        for chunk in result["retrieved_chunks"]:
            print(
                f"- {chunk['chunk_id']} | "
                f"similarity={chunk['similarity']:.4f}"
            )

        if result["status"] == "citation_error":
            print("\nRAW MODEL OUTPUT:")
            print(result["raw_answer"])

    # Save reproducible example results
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== TEST COMPLETED ===")
    print(f"Results saved to: {OUTPUT_PATH}")
    
    pipeline.flush()
    print("[PASS] Langfuse flush completed")

if __name__ == "__main__":
    main()