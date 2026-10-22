import io
import json

import pytest

import pharmaguard_rag.generation as generation_module
from pharmaguard_rag.generation import (
    GenerationTruncatedError,
    OllamaGenerator,
)


def fake_response(payload):
    return io.BytesIO(json.dumps(payload).encode("utf-8"))


def test_generation_returns_complete_content(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return fake_response({
            "message": {"content": "Jawaban ringkas [S1]."},
            "done_reason": "stop",
        })

    monkeypatch.setattr(generation_module, "urlopen", fake_urlopen)

    generator = OllamaGenerator(timeout=10)
    result = generator.generate("system", "question")

    assert result == "Jawaban ringkas [S1]."
    assert captured["timeout"] == 10
    assert captured["payload"]["options"]["repeat_penalty"] == 1.1


def test_generation_rejects_truncated_output(monkeypatch):
    monkeypatch.setattr(
        generation_module,
        "urlopen",
        lambda request, timeout: fake_response({
            "message": {"content": "Jawaban parsial [S1]"},
            "done_reason": "length",
        }),
    )

    with pytest.raises(GenerationTruncatedError) as exc_info:
        OllamaGenerator().generate("system", "question")

    assert exc_info.value.partial_response == "Jawaban parsial [S1]"
    assert "Partial output was rejected" in str(exc_info.value)


def test_generation_rejects_empty_output(monkeypatch):
    monkeypatch.setattr(
        generation_module,
        "urlopen",
        lambda request, timeout: fake_response({
            "message": {"content": "   "},
            "done_reason": "stop",
        }),
    )

    with pytest.raises(RuntimeError, match="empty response"):
        OllamaGenerator().generate("system", "question")


def test_selection_passes_schema_and_records_token_diagnostics(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data))
        return fake_response({"message": {"content": '{"evidence_ids": ["E1"]}'},
                              "done_reason": "stop", "prompt_eval_count": 100, "eval_count": 10})

    monkeypatch.setattr(generation_module, "urlopen", fake_urlopen)
    generator = OllamaGenerator()
    schema = {"type": "object"}
    generator.generate("system", "question", response_format=schema)
    assert captured["format"] == schema
    assert captured["options"]["num_predict"] == 128
    assert captured["options"]["num_ctx"] == 8192
    assert generator.last_diagnostics["eval_count"] == 10
