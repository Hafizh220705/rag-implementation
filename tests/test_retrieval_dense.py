import numpy as np
import pytest
from pharmaguard_rag.retrieval_dense import DenseRetriever

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