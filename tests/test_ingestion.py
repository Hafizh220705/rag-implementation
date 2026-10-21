import pymupdf
import pytest
from pharmaguard_rag.ingestion import extract_pages

def test_extract_pages(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    # Create a temporary PDF with 2 pages
    pdf = pymupdf.open()

    page1 = pdf.new_page()
    page1.insert_text(
        (72, 72),
        "Farmakovigilans adalah kegiatan pemantauan. " * 5
    )

    pdf.new_page()
    pdf.save(pdf_path)
    pdf.close()

    # Run ingestion
    pages = list(extract_pages(pdf_path, "TEST"))

    assert len(pages) == 2

    assert pages[0]["document_id"] == "TEST"
    assert pages[0]["pdf_page"] == 1
    assert pages[0]["char_count"] >= 100
    assert pages[0]["low_text"] is False

    assert pages[1]["pdf_page"] == 2
    assert pages[1]["low_text"] is True


def test_missing_pdf(tmp_path):
    missing_file = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        list(extract_pages(missing_file, "TEST"))
