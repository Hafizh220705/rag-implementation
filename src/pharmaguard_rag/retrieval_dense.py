from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.vector_store import ChromaVectorStore

class DenseRetriever:
    """Semantic retrieval using E5 embeddings and ChromaDB."""

    def __init__(
        self,
        embedder: E5Embedder,
        vector_store: ChromaVectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        question: str,
        top_k: int = 4,
    ) -> list[dict]:
        """Retrieve top-k relevant document chunks."""

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        index_count = self.vector_store.count()

        if index_count == 0:
            raise ValueError(
                "Vector index is empty. Run indexing first."
            )

        # Step 1: Encode user question
        query_embedding = self.embedder.encode_query(
            question
        )

        # Step 2: Search ChromaDB
        results = self.vector_store.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=min(top_k, index_count),
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        # Step 3: Parse retrieved results
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        retrieved_chunks = []

        for rank, chunk_id in enumerate(ids, start=1):
            index = rank - 1

            metadata = metadatas[index] or {}
            distance = float(distances[index])

            retrieved_chunks.append({
                "rank": rank,
                "chunk_id": chunk_id,
                "document_id": metadata.get(
                    "document_id"
                ),
                "title": metadata.get("title"),
                "source_file": metadata.get(
                    "source_file"
                ),
                "pdf_page": metadata.get("pdf_page"),
                "document_type": metadata.get(
                    "document_type"
                ),
                "legal_status": metadata.get(
                    "legal_status"
                ),
                "source_url": metadata.get(
                    "source_url"
                ),
                "text": documents[index],
                "distance": distance,
                "similarity": 1.0 - distance,
            })

        return retrieved_chunks