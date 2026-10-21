from pathlib import Path
from transformers import AutoTokenizer
from pharmaguard_rag.ingestion import extract_pages
from pharmaguard_rag.chunking import chunk_page

MODEL_NAME = "intfloat/multilingual-e5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

pdf_path = Path("data/raw/D1_perbpom_4_2026.pdf")

pages = extract_pages(
    pdf_path=pdf_path,
    document_id="D1",
)

total_chunks = 0

for page in pages:
    if page["low_text"]:
        continue

    chunks = chunk_page(
        page=page,
        tokenizer=tokenizer,
        chunk_size=450,
        chunk_overlap=75,
    )

    total_chunks += len(chunks)

    print(
        f"Page {page['pdf_page']} | "
        f"Chunks: {len(chunks)}"
    )

    if page["pdf_page"] == 1:
        for chunk in chunks:
            print("\n--- CHUNK ---")
            print("ID:", chunk["chunk_id"])
            print("Tokens:", chunk["token_count"])
            print("Text:", chunk["text"][:300])

print(f"\nTotal chunks: {total_chunks}")