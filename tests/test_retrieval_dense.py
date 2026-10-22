import numpy as np
import pytest
from pharmaguard_rag.retrieval_dense import DenseRetriever, required_document_ids, retrieval_query

class MockEmbedder:
    def __init__(self):
        self.last_question = None

    def encode_query(self, question):
        self.last_question = question
        return np.array([1.0, 0.0], dtype=np.float32)

class MockCollection:
    def __init__(self):
        self.last_n_results = None

    def query(self, query_embeddings, n_results, include):
        self.last_n_results = n_results

        return {
            "ids": [[
                "D1_p026_c001",
                "D3_p018_c001",
            ]],
            "documents": [[
                "Kewajiban pemilik izin edar.",
                "Definisi farmakovigilans.",
            ]],
            "metadatas": [[
                {
                    "document_id": "D1",
                    "pdf_page": 26,
                    "title": "Peraturan BPOM 4/2026",
                },
                {
                    "document_id": "D3",
                    "pdf_page": 18,
                    "title": "Modul Farmakovigilans Dasar",
                },
            ]],
            "distances": [[0.10, 0.25]],
        }

class MockVectorStore:
    def __init__(self, count=2):
        self.index_count = count
        self.collection = MockCollection()

    def count(self):
        return self.index_count

def test_retrieval_returns_ranked_chunks():
    embedder = MockEmbedder()
    store = MockVectorStore()

    retriever = DenseRetriever(embedder, store)

    results = retriever.search(
        "Apa kewajiban pemilik izin edar?",
        top_k=4,
    )

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[0]["chunk_id"] == "D1_p026_c001"
    assert results[0]["pdf_page"] == 26
    assert results[0]["similarity"] == pytest.approx(0.90)

    assert results[1]["rank"] == 2
    assert results[1]["similarity"] == pytest.approx(0.75)

    assert store.collection.last_n_results == 2

def test_empty_question():
    retriever = DenseRetriever(
        MockEmbedder(),
        MockVectorStore(),
    )

    with pytest.raises(ValueError):
        retriever.search("   ")

def test_invalid_top_k():
    retriever = DenseRetriever(
        MockEmbedder(),
        MockVectorStore(),
    )

    with pytest.raises(ValueError):
        retriever.search("Apa itu farmakovigilans?", top_k=0)

def test_empty_index():
    retriever = DenseRetriever(
        MockEmbedder(),
        MockVectorStore(count=0),
    )

    with pytest.raises(ValueError):
        retriever.search("Apa itu farmakovigilans?")


@pytest.mark.parametrize("question, expected", [
    ("Peraturan BPOM Nomor 4 Tahun 2026 dan Modul Dasar", ["D1", "D3"]),
    ("Modul Industri Farmasi dan Modul Farmakovigilans Dasar", ["D2", "D3"]),
    ("Buku Saku ESO dibandingkan dengan modul dasar", ["D3", "D4"]),
    ("Menurut D2 dan D4?", ["D2", "D4"]),
    ("Apa itu farmakovigilans?", []),
])
def test_explicit_document_aliases(question, expected):
    assert required_document_ids(question) == expected


def test_general_semantic_query_is_unchanged():
    query = "Apa kewajiban pemilik izin edar?"
    embedder = MockEmbedder()
    DenseRetriever(embedder, MockVectorStore()).search(query)
    assert embedder.last_question == query


def test_personnel_synonym_does_not_insert_a_responsible_actor():
    query = retrieval_query("Siapa menyediakan personel khusus keamanan obat?")
    assert "penanggung jawab farmakovigilans" in query
    assert "Pemilik Izin Edar" not in query


def test_explicit_documents_get_quotas_without_unrequested_sources():
    class Collection:
        def __init__(self):
            self.calls = []

        def query(self, **kwargs):
            self.calls.append(kwargs)
            doc = kwargs["where"]["document_id"]
            return {"ids": [[doc + "_p001_c001"]], "documents": [["Bukti lengkap."]],
                    "metadatas": [[{"document_id": doc, "pdf_page": 1}]],
                    "distances": [[0.12 if doc == "D3" else 0.1]]}

    store = MockVectorStore(count=10)
    store.collection = Collection()
    embedder = MockEmbedder()
    result = DenseRetriever(embedder, store).search("Apa manfaat pelaporan dalam Modul Dasar dan Buku Saku ESO?", top_k=4)
    assert {r["document_id"] for r in result} == {"D3", "D4"}
    assert [c["n_results"] for c in store.collection.calls] == [2, 2]
    assert [r["rank"] for r in result] == [1, 2]
    assert "Modul Dasar" not in embedder.last_question
    assert "manfaat pelaporan" in embedder.last_question


def test_too_small_budget_does_not_silently_drop_a_required_document():
    with pytest.raises(ValueError, match="every explicitly requested"):
        DenseRetriever(MockEmbedder(), MockVectorStore()).search("D1 dan D3", top_k=1)


def test_same_page_overlap_is_merged_without_persisting_or_crossing_gaps():
    overlap = "Penanggung jawab harus memahami profil keamanan dan dapat"

    class Collection:
        def get(self, **kwargs):
            assert kwargs["where"] == {"$and": [{"document_id": "D1"}, {"pdf_page": 26}]}
            return {"ids": ["D1_p026_c001", "D1_p026_c002", "D1_p026_c003"],
                    "documents": ["Pendahuluan. " + overlap, overlap + " melaksanakan tugasnya.",
                                  "Hal lain tidak memiliki overlap."]}

    store = MockVectorStore()
    store.collection = Collection()
    chunk = {"document_id": "D1", "pdf_page": 26, "chunk_id": "D1_p026_c001"}
    pages = DenseRetriever(MockEmbedder(), store).evidence_pages([chunk, chunk])
    assert len(pages) == 2
    assert overlap + " melaksanakan tugasnya." in pages[0]["text"]
    assert pages[0]["text"].count(overlap) == 1
    assert pages[0]["context_chunk_ids"] == ["D1_p026_c001", "D1_p026_c002"]
    assert pages[1]["context_chunk_ids"] == ["D1_p026_c003"]
