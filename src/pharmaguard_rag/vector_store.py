import chromadb
from pathlib import Path

COLLECTION_NAME = "pharmaguard_dense_e5_v1"

class ChromaVectorStore:
    """Persistent vector storage for PharmaGuard-RAG."""

    def __init__(
        self,
        persist_directory: str | Path,
        model_name: str,
        corpus_sha256: str,
    ):
        self.client = chromadb.PersistentClient(
            path=str(persist_directory)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
            metadata={
                "embedding_model": model_name,
                "corpus_sha256": corpus_sha256,
            },
        )

        # Prevent mixing embeddings from different
        # models or corpus versions.
        if self.collection.count() > 0:
            metadata = self.collection.metadata or {}

            if metadata.get("embedding_model") != model_name:
                raise ValueError(
                    "Existing index uses another embedding model."
                )

            if metadata.get("corpus_sha256") != corpus_sha256:
                raise ValueError(
                    "Existing index uses another corpus version."
                )

    def upsert_chunks(self, chunks, embeddings):
        """Insert or update chunks with their embeddings."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        ids = []
        documents = []
        metadatas = []

        metadata_fields = (
            "document_id",
            "source_file",
            "pdf_page",
            "chunk_number",
            "title",
            "publisher",
            "year",
            "document_type",
            "legal_status",
            "source_url",
            "token_count",
        )

        for chunk in chunks:
            ids.append(chunk["chunk_id"])
            documents.append(chunk["text"])

            metadata = {
                key: chunk[key]
                for key in metadata_fields
                if key in chunk
                and isinstance(
                    chunk[key], (str, int, float, bool)
                )
            }

            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

    def count(self):
        return self.collection.count()