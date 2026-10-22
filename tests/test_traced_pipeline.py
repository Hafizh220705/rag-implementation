import pytest
from contextlib import contextmanager
from pharmaguard_rag.observability import OptionalLangfuse
from pharmaguard_rag.traced_pipeline import TracedRAGPipeline

class FakeRetriever:
    def search(self, question, top_k=4):
        return [
            {
                "rank": 1,
                "chunk_id": "D3_p018_c001",
                "document_id": "D3",
                "source_file": "D3_modul_dasar_2020.pdf",
                "title": "Modul Farmakovigilans Dasar",
                "pdf_page": 18,
                "similarity": 0.89,
                "text": (
                    "Farmakovigilans adalah kegiatan "
                    "pemantauan keamanan obat."
                ),
            }
        ]

class FakeGenerator:
    model_name = "qwen2.5:3b"

    def __init__(self, response):
        self.response = response

    def generate(self, system_prompt, user_prompt, **kwargs):
        return self.response

class FakeObserver:
    def __init__(self):
        self.observations = []
        self.updates = []
        self.flushed = False

    @contextmanager
    def observation(self, name, as_type="span", **kwargs):
        self.observations.append(name)
        yield object()

    def update(self, observation, **kwargs):
        self.updates.append(kwargs)

    def flush(self):
        self.flushed = True

@pytest.mark.parametrize(
    "model_response, expected_status",
    [
        (
            '{"evidence_ids": ["E1"]}',
            "answered",
        ),
        (
            '{"evidence_ids": []}',
            "insufficient_evidence",
        ),
    ],
)

def test_traced_pipeline_preserves_rag_behavior(
    model_response,
    expected_status,
):
    observer = FakeObserver()

    pipeline = TracedRAGPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(model_response),
        observer=observer,
    )

    result = pipeline.ask(
        question="Apa itu farmakovigilans?",
        top_k=4,
    )

    assert result["status"] == expected_status

    assert observer.observations == [
        "pharmaguard-rag",
        "dense-retrieval",
        "ollama-generation",
    ]

    assert len(observer.updates) == 3

    if expected_status == "answered":
        assert len(result["cited_sources"]) == 1
        assert result["cited_sources"][0]["chunk_id"] == (
            "D3_p018_c001"
        )
    else:
        assert result["cited_sources"] == []

    pipeline.flush()
    assert observer.flushed is True

def test_langfuse_disabled_does_not_block(monkeypatch):
    monkeypatch.setenv("LANGFUSE_ENABLED", "false")

    observer = OptionalLangfuse()

    assert observer.enabled is False

    pipeline = TracedRAGPipeline(
        retriever=FakeRetriever(),
        generator=FakeGenerator(
            '{"evidence_ids": ["E1"]}'
        ),
        observer=observer,
    )

    result = pipeline.ask(
        "Apa itu farmakovigilans?"
    )

    assert result["status"] == "answered"
    assert result["cited_sources"]

    pipeline.flush()
