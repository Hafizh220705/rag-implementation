import pymupdf
from pathlib import Path
from collections.abc import Iterator

def extract_pages(
    pdf_path: Path,
    document_id: str,
) -> Iterator[dict]:

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF tidak ditemukan: {pdf_path}"
        )

    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()

            yield {
                "document_id": document_id,
                "source_file": pdf_path.name,
                "pdf_page": page_index + 1,
                "text": text,
                "char_count": len(text),
                "low_text": len(text) < 100,
            }