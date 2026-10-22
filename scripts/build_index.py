import argparse
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
    parser = argparse.ArgumentParser(description="Build a fresh index or validate an existing frozen index.")
    parser.add_argument("--check", action="store_true", help="Validate a read-only snapshot; no embedding inference.")
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--report", type=Path, default=REPORT_DIR / "indexing_report.json")
    args = parser.parse_args()
    index_dir = args.index_dir if args.index_dir.is_absolute() else ROOT / args.index_dir
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    if not args.check and ((index_dir.exists() and any(index_dir.iterdir())) or report_path.exists()):
        parser.error("Index or report exists. Use a new --index-dir and --report; frozen artifacts cannot be overwritten.")
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

    if args.check:
        store = ChromaVectorStore(index_dir, MODEL_NAME, corpus_sha256)
        try:
            indexed_ids = set(store.collection.get(include=[])["ids"])
            if indexed_ids != set(chunk_ids):
                raise ValueError("Index IDs do not match the frozen chunks.")
            print(f"[OK] Existing index: {store.count()} vectors; model, corpus hash and chunk IDs match.")
        finally:
            store.close()
        return

    embedder = E5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
    )

    store = ChromaVectorStore(
        persist_directory=index_dir,
        model_name=MODEL_NAME,
        corpus_sha256=corpus_sha256,
        read_only=False,
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

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== INDEXING COMPLETED ===")
    print(f"Indexed vectors: {indexed_count}")
    print(f"Vector dimension: {embedder.dimension}")
    store.close()
    print(f"Database path: {index_dir}")
    print(f"Report: {report_path}")

if __name__ == "__main__":
    main()
