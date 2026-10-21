import hashlib
import json
import chromadb
from datetime import datetime, timezone
from pathlib import Path
from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.vector_store import ChromaVectorStore

ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = ROOT / "data" / "indexes" / "chroma"
REPORT_DIR = ROOT / "outputs" / "evaluations"

MODEL_NAME = "intfloat/multilingual-e5-small"
BATCH_SIZE = 16

def load_chunks():
    with CHUNKS_PATH.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

def main():
    print("=== PHARMAGUARD INDEXING ===")

    chunks = load_chunks()

    if not chunks:
        raise ValueError("No chunks found.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    corpus_sha256 = hashlib.sha256(
        CHUNKS_PATH.read_bytes()
    ).hexdigest()

    print(f"Loaded chunks: {len(chunks)}")

    embedder = E5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
    )

    store = ChromaVectorStore(
        persist_directory=INDEX_DIR,
        model_name=MODEL_NAME,
        corpus_sha256=corpus_sha256,
    )

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]

        passages = [chunk["text"] for chunk in batch]

        embeddings = embedder.encode_passages(passages)

        if embeddings.shape[1] != embedder.dimension:
            raise ValueError("Unexpected embedding dimension.")

        store.upsert_chunks(batch, embeddings)

        print(
            f"Indexed: {min(start + BATCH_SIZE, len(chunks))}"
            f"/{len(chunks)}"
        )

    indexed_count = store.count()

    if indexed_count != len(chunks):
        raise ValueError(
            f"Index count mismatch: "
            f"{indexed_count} != {len(chunks)}"
        )

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "embedding_model": MODEL_NAME,
        "embedding_dimension": embedder.dimension,
        "normalized_embeddings": True,
        "vector_database": "ChromaDB",
        "chromadb_version": chromadb.__version__,
        "distance_metric": "cosine",
        "collection_name": store.collection.name,
        "corpus_sha256": corpus_sha256,
        "batch_size": BATCH_SIZE,
        "total_chunks": len(chunks),
        "indexed_count": indexed_count,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORT_DIR / "indexing_report.json"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== INDEXING COMPLETED ===")
    print(f"Indexed vectors: {indexed_count}")
    print(f"Vector dimension: {embedder.dimension}")
    print(f"Database path: {INDEX_DIR}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()