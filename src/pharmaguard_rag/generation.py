import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"

class OllamaGenerator:
    """Generate text using a local Ollama model."""

    def __init__(
        self,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout: int = 240,
    ):
        self.model_name = (
            model_name
            or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        )

        self.base_url = (
            base_url
            or os.getenv("OLLAMA_BASE_URL", DEFAULT_BASE_URL)
        ).rstrip("/")

        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Send a chat request and return the answer."""

        if not user_prompt.strip():
            raise ValueError("User prompt cannot be empty.")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 768,
                "seed": 42,
            },
        }

        body = json.dumps(payload).encode("utf-8")

        request = Request(
            url=f"{self.base_url}/api/chat",
            data=body,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:
                result = json.load(response)

        except HTTPError as error:
            details = error.read().decode(
                "utf-8",
                errors="replace",
            )
            raise RuntimeError(
                f"Ollama HTTP {error.code}: {details}"
            ) from error

        except URLError as error:
            raise RuntimeError(
                "Cannot connect to Ollama. "
                "Ensure the Ollama server is running."
            ) from error
        
        content = (
            result.get("message", {})
            .get("content", "")
            .strip()
        )

        if not content:
            raise RuntimeError(
                "Ollama returned an empty response."
            )

        # Check whether generation was truncated
        if result.get("done_reason") == "length":
            raise RuntimeError(
                "Ollama output was truncated because the "
                "generation limit was reached. "
                "Reduce answer length or increase num_predict."
            )

        return content