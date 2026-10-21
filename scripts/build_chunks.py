import json
from datetime import datetime, timezone
from pathlib import Path
from transformers import AutoTokenizer
from pharmaguard_rag.ingestion import extract_pages
from pharmaguard_rag.chunking import chunk_page

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = ROOT / "data" / "manifest.json"

MODEL_NAME = "intfloat/multilingual-e5-small"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 75

def save_jsonl(path, records):
    """Save records as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

def main():
    # Load dataset manifest
    with MANIFEST_PATH.open(encoding="utf-8") as file:
        manifest = json.load(file)

    documents = manifest["documents"]

    # Load tokenizer once
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    all_pages = []
    all_chunks = []
    document_stats = {}
    seen_chunk_ids = set()

    for document in documents:
        doc_id = document["id"]
        pdf_path = RAW_DIR / document["file_name"]

        print(f"\nProcessing {doc_id}: {pdf_path.name}")

        pages = list(extract_pages(pdf_path, doc_id))

        chunk_count = 0
        low_text_count = 0

        for page in pages:
            # Preserve document provenance
            metadata = {
                "title": document["title"],
                "publisher": document.get("publisher"),
                "year": document.get("year"),
                "document_type": document.get("document_type"),
                "legal_status": document.get("legal_status"),
                "source_url": document.get("source_url"),
            }

            page.update(metadata)
            all_pages.append(page)

            # Keep low-text pages for audit, but
            # exclude them from baseline indexing.
            if page["low_text"]:
                low_text_count += 1
                continue

            chunks = chunk_page(
                page=page,
                tokenizer=tokenizer,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

            for chunk in chunks:
                chunk.update(metadata)

                # Validate unique chunk identifiers
                if chunk["chunk_id"] in seen_chunk_ids:
                    raise ValueError(
                        f"Duplicate chunk ID: {chunk['chunk_id']}"
                    )

                seen_chunk_ids.add(chunk["chunk_id"])

                # Check actual embedding input length
                embedding_input = "passage: " + chunk["text"]

                input_tokens = tokenizer.encode(
                    embedding_input,
                    add_special_tokens=True,
                    truncation=False,
                    verbose=False,
                )

                if len(input_tokens) > 512:
                    raise ValueError(
                        f"Chunk exceeds model limit: "
                        f"{chunk['chunk_id']}"
                    )

                chunk["embedding_input_tokens"] = len(
                    input_tokens
                )

            all_chunks.extend(chunks)
            chunk_count += len(chunks)

        document_stats[doc_id] = {
            "pages": len(pages),
            "low_text_pages": low_text_count,
            "chunks": chunk_count,
        }

        print(f"Pages: {len(pages)}")
        print(f"Low-text pages: {low_text_count}")
        print(f"Chunks: {chunk_count}")

    # Validate nonempty corpus
    if not all_chunks:
        raise ValueError("No chunks were generated")

    # Save processed data
    save_jsonl(
        PROCESSED_DIR / "pages.jsonl",
        all_pages,
    )

    save_jsonl(
        PROCESSED_DIR / "chunks.jsonl",
        all_chunks,
    )

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "embedding_tokenizer": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "total_pages": len(all_pages),
        "total_chunks": len(all_chunks),
        "documents": document_stats,
        "limitations": [
            "Image-only content is not extracted.",
            "Low-text pages are excluded from chunks.",
            "Chunking is page-aware, not yet section-aware.",
        ],
    }

    report_path = PROCESSED_DIR / "chunking_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n=== INGESTION SUMMARY ===")
    print(f"Total pages: {len(all_pages)}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Output: {PROCESSED_DIR}")

if __name__ == "__main__":
    main()