"""Test local Ollama. This does not test RAG/grounding."""
from __future__ import annotations
import os
import time
import requests
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
    started = time.perf_counter()
    resp = requests.post(f"{base}/api/generate", json={
        "model": model,
        "prompt": "Jawab dalam satu kalimat berbahasa Indonesia: Apa fungsi sistem RAG?",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 96},
    }, timeout=240)
    resp.raise_for_status()
    data = resp.json()
    print(f"Model: {model}")
    print(f"Elapsed seconds: {time.perf_counter() - started:.2f}")
    print(f"Response: {data.get('response', '')}")
    print("Ollama smoke test PASSED")


if __name__ == "__main__":
    main()
