"""Test that multilingual E5 generates vector embeddings, not retrieval quality."""
from __future__ import annotations
from sentence_transformers import SentenceTransformer

MODEL = "intfloat/multilingual-e5-small"


def main() -> None:
    print(f"Loading embedding model: {MODEL}")
    model = SentenceTransformer(MODEL, device="cpu")
    texts = [
        "query: Apa yang dimaksud dengan farmakovigilans?",
        "passage: Farmakovigilans merupakan kegiatan pemantauan keamanan obat.",
    ]
    vectors = model.encode(texts, normalize_embeddings=True)
    print(f"Embedding shape: {vectors.shape}")
    print(f"Cosine similarity (smoke-test only): {float(vectors[0] @ vectors[1]):.4f}")
    if vectors.shape != (2, 384):
        raise RuntimeError(f"Unexpected vector shape: {vectors.shape}")
    print("Embedding smoke test PASSED")


if __name__ == "__main__":
    main()
