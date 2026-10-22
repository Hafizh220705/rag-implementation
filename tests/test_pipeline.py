import json

import pytest

from pharmaguard_rag.generation import GenerationTruncatedError
from pharmaguard_rag.pipeline import RAGPipeline, RAGPipelineError


def chunk(doc="D1", text="Pemilik Izin Edar harus menyampaikan nama dan kontak kepada Kepala Badan."):
    return {"rank": 1, "chunk_id": f"{doc}_p026_c001", "document_id": doc,
            "title": f"Dokumen {doc}", "pdf_page": 26, "text": text, "similarity": 0.9}


class FakeRetriever:
    def __init__(self, results=None, error=None):
        self.results = [chunk()] if results is None else results
        self.error = error

    def search(self, question, top_k=4):
        if self.error:
            raise self.error
        return self.results


class FakeGenerator:
    def __init__(self, responses=None, error=None):
        self.responses = responses or ['{"evidence_ids": ["E1"]}']
        self.error = error
        self.calls = 0

    def generate(self, system_prompt, user_prompt, response_format=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.responses[min(self.calls - 1, len(self.responses) - 1)]


def test_valid_selection_preserves_actor_and_source():
    result = RAGPipeline(FakeRetriever(), FakeGenerator()).ask("Informasi nama dan kontak?")
    assert result["status"] == "answered"
    assert "Pemilik Izin Edar harus menyampaikan" in result["answer"]
    assert "D1_p026_c001" in result["answer"]
    assert result["selected_evidence"][0]["spans"]


@pytest.mark.parametrize("raw", [
    '{"evidence_ids": ["Bs1"]}',
    '{"evidence_ids": ["E999"]}',
    'KTD (Kesalahan Teknis Dokter) wajib dilaporkan [S1].',
    'Perjanjian kerja sama harus ditetapkan oleh Kepala Badan [S1].',
    '{"evidence_ids": ["E1"], "answer": "Tenaga kesehatan wajib menunjuk personel."}',
    'BUKTI_TIDAK_CUKUP\nNama dan kontak [S1].',
])
def test_untrusted_free_text_and_unknown_ids_never_reach_answer(raw):
    generator = FakeGenerator([raw])
    result = RAGPipeline(FakeRetriever(), generator).ask("Informasi kontak?")
    assert result["status"] == "insufficient_evidence"
    assert result["abstention_reason"] == "evidence_validation_failed"
    assert result["cited_sources"] == []
    assert generator.calls == 2


def test_selection_retry_recovers_without_rewriting_claims():
    generator = FakeGenerator(['{"evidence_ids": ["Bs1"]}', '{"evidence_ids": ["E1"]}'])
    result = RAGPipeline(FakeRetriever(), generator).ask("Informasi kontak?")
    assert result["status"] == "answered"
    assert result["generation_attempts"] == 2
    assert len(result["validation_errors"]) == 1


def test_explicit_abstention_is_not_retried_to_force_answer():
    generator = FakeGenerator(['{"evidence_ids": []}'])
    result = RAGPipeline(FakeRetriever(), generator).ask("Informasi kontak?")
    assert result["abstention_reason"] == "model_abstention"
    assert generator.calls == 1


def test_missing_requested_document_abstains_before_generation():
    generator = FakeGenerator()
    result = RAGPipeline(FakeRetriever([chunk("D1")]), generator).ask(
        "Bandingkan Peraturan BPOM 4/2026 dan Modul Dasar tentang nama kontak."
    )
    assert result["missing_document_ids"] == ["D3"]
    assert result["abstention_reason"] == "missing_required_sources"
    assert generator.calls == 0


def test_cannot_attribute_d2_quote_to_d3():
    results = [chunk("D2"), chunk("D3")]
    generator = FakeGenerator(['{"evidence_ids": {"D2": "E2", "D3": "E1"}}'])
    result = RAGPipeline(FakeRetriever(results), generator).ask(
        "Nama kontak menurut Modul Industri Farmasi dan Modul Dasar?"
    )
    assert result["status"] == "insufficient_evidence"
    assert "claimed document" in result["validation_errors"][0]


def test_low_similarity_skips_generation():
    low = {**chunk(), "similarity": 0.8}
    generator = FakeGenerator()
    result = RAGPipeline(FakeRetriever([low]), generator).ask("Informasi kontak?")
    assert result["abstention_reason"] == "low_retrieval_similarity"
    assert generator.calls == 0


def test_no_complete_sentence_skips_generation():
    generator = FakeGenerator()
    result = RAGPipeline(FakeRetriever([chunk(text="Memahami profil keamanan dan dapat")]), generator).ask("Profil keamanan?")
    assert result["abstention_reason"] == "no_complete_evidence"
    assert generator.calls == 0


def test_generation_error_preserves_diagnostics():
    pipeline = RAGPipeline(FakeRetriever(), FakeGenerator(error=GenerationTruncatedError("partial")))
    with pytest.raises(RAGPipelineError) as error:
        pipeline.ask("Informasi kontak?")
    assert error.value.stage == "generation"
    assert error.value.raw_answer == "partial"
    assert error.value.retrieved_chunks[0]["document_id"] == "D1"
    assert error.value.generation_latency_seconds is not None


def test_retrieval_error_identifies_stage():
    pipeline = RAGPipeline(FakeRetriever(error=RuntimeError("missing index")), FakeGenerator())
    with pytest.raises(RAGPipelineError) as error:
        pipeline.ask("Pertanyaan")
    assert error.value.stage == "retrieval"


@pytest.mark.parametrize("question, k", [("", 4), ("Pertanyaan", 0)])
def test_invalid_input_rejected(question, k):
    with pytest.raises(ValueError):
        RAGPipeline(FakeRetriever(), FakeGenerator()).ask(question, top_k=k)
