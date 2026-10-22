import hashlib
from pathlib import Path

import numpy as np
import pytest

from pharmaguard_rag.vector_store import ChromaVectorStore


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def index_path(tmp_path):
    path = tmp_path / "index"
    writer = ChromaVectorStore(path, "test-model", "test-corpus", read_only=False)
    try:
        writer.upsert_chunks([{"chunk_id": "D1_p001_c001", "document_id": "D1", "pdf_page": 1,
                               "text": "Bukti uji tidak memakai model nyata."}],
                             np.array([[1.0, 0.0]], dtype=np.float32))
    finally:
        writer.close()
    return path


def test_snapshot_queries_leave_frozen_index_bytes_unchanged(index_path):
    before = hashes(index_path)
    reader = ChromaVectorStore(index_path, "test-model", "test-corpus")
    snapshot = Path(reader._snapshot.name)
    try:
        assert reader.count() == 1
        assert reader.collection.query(query_embeddings=[[1.0, 0.0]], n_results=1)["ids"][0] == ["D1_p001_c001"]
        with pytest.raises(PermissionError):
            reader.upsert_chunks([], np.array([]))
    finally:
        reader.close()
    assert hashes(index_path) == before
    assert not snapshot.exists()


@pytest.mark.parametrize("model, corpus", [("other", "test-corpus"), ("test-model", "other")])
def test_mismatched_index_is_rejected_without_mutating_original(index_path, model, corpus):
    before = hashes(index_path)
    with pytest.raises(ValueError, match="another"):
        ChromaVectorStore(index_path, model, corpus)
    assert hashes(index_path) == before


def test_query_does_not_create_missing_index(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        ChromaVectorStore(missing, "model", "corpus")
    assert not missing.exists()
