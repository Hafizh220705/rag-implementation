import pytest

from pharmaguard_rag.citations import resolve_citations


def make_chunks():
    return [
        {
            "chunk_id": "D1_p026_c001",
            "document_id": "D1",
            "title": "Peraturan BPOM 4/2026",
            "source_file": "D1.pdf",
            "pdf_page": 26,
            "source_url": "https://example.test/d1",
        },
        {
            "chunk_id": "D3_p002_c001",
            "document_id": "D3",
            "title": "Modul Farmakovigilans Dasar",
            "source_file": "D3.pdf",
            "pdf_page": 2,
            "source_url": "https://example.test/d3",
        },
    ]


def test_resolves_single_citation():
    answer, sources = resolve_citations(
        "Pemilik izin edar memiliki kewajiban [S1].",
        make_chunks(),
    )

    assert "[D1, PDF hlm. 26, D1_p026_c001]" in answer
    assert [source["label"] for source in sources] == ["S1"]


def test_resolves_combined_citations():
    answer, sources = resolve_citations(
        "Definisi tersebut terdapat di kedua dokumen [S1, S2].",
        make_chunks(),
    )

    assert "[S1" not in answer
    assert "[D1, PDF hlm. 26, D1_p026_c001]" in answer
    assert "[D3, PDF hlm. 2, D3_p002_c001]" in answer
    assert [source["label"] for source in sources] == ["S1", "S2"]


def test_duplicate_labels_are_deduplicated():
    answer, sources = resolve_citations(
        "Informasi yang sama [S1, S1] dan ditegaskan lagi [S1].",
        make_chunks(),
    )

    assert len(sources) == 1
    assert sources[0]["chunk_id"] == "D1_p026_c001"
    assert answer.count("D1_p026_c001") == 2


@pytest.mark.parametrize(
    "answer",
    [
        "Sumber tidak ada [S3].",
        "Format tidak valid [S1 S2].",
        "Placeholder tidak valid [Sfoo].",
        "Label dengan typo [Bs1] dan label valid [S1].",
        "Jawaban tanpa sumber.",
    ],
)
def test_rejects_invalid_or_missing_citations(answer):
    with pytest.raises(ValueError):
        resolve_citations(answer, make_chunks())
