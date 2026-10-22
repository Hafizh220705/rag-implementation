import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = "qwen2.5:3b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class GenerationTruncatedError(RuntimeError):
    """Raised when Ollama stops because the output limit was reached."""

    def __init__(self, partial_response: str):
        super().__init__(
            "Ollama output was truncated because the generation "
            "limit was reached. Partial output was rejected."
        )
        self.partial_response = partial_response
        self.done_reason = "length"


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
        response_format: dict | None = None,
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
                "repeat_penalty": 1.1,
            },
        }

        if response_format is not None:
            payload["format"] = response_format
            payload["options"].update(num_ctx=8192, num_predict=128)
        self.last_diagnostics = {}

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
        
        self.last_diagnostics = {key: result.get(key) for key in (
            "prompt_eval_count", "eval_count", "done_reason", "total_duration",
        )}
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
            raise GenerationTruncatedError(content)

        return content
