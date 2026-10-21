import numpy as np
from pharmaguard_rag.embeddings import E5Embedder

def main():
    embedder = E5Embedder(device="cpu")

    question = (
        "Apa yang dimaksud dengan farmakovigilans?"
    )

    passages = [
        (
            "Farmakovigilans merupakan kegiatan "
            "pemantauan keamanan obat untuk mendeteksi, "
            "menilai, dan mencegah efek samping obat."
        ),
        (
            "Pemeliharaan kendaraan bermotor dilakukan "
            "secara berkala untuk memastikan kondisi mesin."
        ),
    ]

    query_embedding = embedder.encode_query(question)

    passage_embeddings = embedder.encode_passages(
        passages
    )

    scores = passage_embeddings @ query_embedding

    print("=== EMBEDDING TEST ===")
    print("Model:", embedder.model_name)
    print("Dimension:", embedder.dimension)
    print("Query shape:", query_embedding.shape)
    print("Passages shape:", passage_embeddings.shape)

    print(
        "Query vector norm:",
        round(float(np.linalg.norm(query_embedding)), 4),
    )

    for index, score in enumerate(scores, start=1):
        print(
            f"Passage {index} cosine similarity: "
            f"{score:.4f}"
        )

if __name__ == "__main__":
    main()