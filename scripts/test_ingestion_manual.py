from pharmaguard_rag.ingestion import extract_pages
from pathlib import Path

pdf_path = Path(
    "data/raw/D1_perbpom_4_2026.pdf"
)

pages = extract_pages(
    pdf_path=pdf_path,
    document_id="D1",
)

for page in pages:
    print(
        f"Page {page['pdf_page']} | "
        f"Characters: {page['char_count']} | "
        f"Low Text: {page['low_text']}"
    )
