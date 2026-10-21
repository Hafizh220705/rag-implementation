import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

def load_jsonl(filename):
    path = PROCESSED_DIR / filename

    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def main():
    pages = load_jsonl("pages.jsonl")
    chunks = load_jsonl("chunks.jsonl")

    report = json.loads(
        (PROCESSED_DIR / "chunking_report.json")
        .read_text(encoding="utf-8")
    )

    print("=== CHUNK VALIDATION ===")

    # 1. Check total records
    assert len(pages) == report["total_pages"]
    assert len(chunks) == report["total_chunks"]

    print("[PASS] Total records match report")

    # 2. Check unique chunk IDs
    chunk_ids = [c["chunk_id"] for c in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))

    print("[PASS] All chunk IDs are unique")

    # 3. Check source pages
    valid_pages = {
        (p["document_id"], p["pdf_page"])
        for p in pages
    }

    # 4. Check metadata and token limits
    required_fields = [
        "chunk_id",
        "document_id",
        "source_file",
        "pdf_page",
        "text",
        "token_count",
        "embedding_input_tokens",
    ]

    for chunk in chunks:
        for field in required_fields:
            assert field in chunk, (
                f"Missing {field}: {chunk.get('chunk_id')}"
            )

        assert chunk["text"].strip()
        assert chunk["token_count"] <= report["chunk_size"]
        assert chunk["embedding_input_tokens"] <= 512

        page_key = (
            chunk["document_id"],
            chunk["pdf_page"],
        )

        assert page_key in valid_pages

    print("[PASS] Metadata and source pages valid")
    print("[PASS] Token limits valid")

    # 5. Compare document distribution
    counts = Counter(c["document_id"] for c in chunks)

    for doc_id, stats in report["documents"].items():
        actual = counts[doc_id]
        expected = stats["chunks"]

        assert actual == expected

        print(f"[PASS] {doc_id}: {actual} chunks")

    # 6. Print samples for manual inspection
    print("\n=== CHUNK SAMPLES ===")

    for doc_id in report["documents"]:
        sample = next(
            c for c in chunks
            if c["document_id"] == doc_id
        )

        print(f"\nDocument: {doc_id}")
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"Page: {sample['pdf_page']}")
        print(f"Tokens: {sample['token_count']}")
        print(f"Preview: {sample['text'][:250]}")

    print("\n[SUCCESS] Chunk validation passed")

if __name__ == "__main__":
    main()