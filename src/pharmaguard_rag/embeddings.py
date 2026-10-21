import os
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-small"

class E5Embedder:
    """Generate normalized query and passage embeddings."""

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "cpu",
    ):
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        )

        self.model = SentenceTransformer(
            self.model_name,
            device=device,
        )

        self.dimension = self.model.get_embedding_dimension()

    def _encode(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        if not texts:
            raise ValueError("Input texts cannot be empty")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_passages(
        self,
        passages: list[str],
    ) -> np.ndarray:
        """Encode document chunks for indexing."""

        texts = [
            f"passage: {text}"
            for text in passages
        ]

        return self._encode(texts)

    def encode_query(
        self,
        question: str,
    ) -> np.ndarray:
        """Encode a user question for retrieval."""

        if not question.strip():
            raise ValueError("Question cannot be empty")

        texts = [f"query: {question.strip()}"]

        return self._encode(texts)[0]