import hashlib
import json
from pathlib import Path
from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.vector_store import ChromaVectorStore
from pharmaguard_rag.retrieval_dense import DenseRetriever

ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
INDEX_DIR = ROOT / "data/indexes/chroma"
OUTPUT_PATH = ROOT / "outputs/examples/dense_retrieval_smoke.json"

MODEL_NAME = "intfloat/multilingual-e5-small"

QUESTIONS = [
    "Apa yang dimaksud dengan farmakovigilans?",
    "Bagaimana mekanisme pelaporan efek samping obat?",
    "Apa kewajiban pemegang izin edar dalam pelaksanaan farmakovigilans?",
    "Bagaimana prosedur mengganti oli mesin kendaraan?",
]

def main():
    print("=== PHARMAGUARD DENSE RETRIEVAL TEST ===")

    # Calculate corpus checksum
    corpus_sha256 = hashlib.sha256(
        CHUNKS_PATH.read_bytes()
    ).hexdigest()

    # Load embedding model
    embedder = E5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
    )

    # Load existing ChromaDB index
    store = ChromaVectorStore(
        persist_directory=INDEX_DIR,
        model_name=MODEL_NAME,
        corpus_sha256=corpus_sha256,
    )

    print(f"Indexed chunks: {store.count()}")

    # Initialize retriever
    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=store,
    )

    saved_results = []

    for question in QUESTIONS:
        print("\n" + "=" * 70)
        print("QUESTION:", question)
        print("=" * 70)

        results = retriever.search(
            question=question,
            top_k=4,
        )

        saved_results.append({
            "question": question,
            "retrieved_chunks": results,
        })

        for chunk in results:
            print(f"\nRank: {chunk['rank']}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Document: {chunk['document_id']}")
            print(f"Title: {chunk['title']}")
            print(f"PDF Page: {chunk['pdf_page']}")
            print(
                f"Similarity: {chunk['similarity']:.4f}"
            )
            print("Passage:")
            print(chunk["text"][:500])
            print("-" * 70)

    # Save outputs for evaluation
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            saved_results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== RETRIEVAL TEST COMPLETED ===")
    print(f"Results saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()