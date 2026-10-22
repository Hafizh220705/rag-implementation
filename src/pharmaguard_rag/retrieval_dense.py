from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pharmaguard_rag.embeddings import E5Embedder
    from pharmaguard_rag.vector_store import ChromaVectorStore


# Corpus title aliases, not evaluation-question rules. Match explicit mentions
# only; a general pharmacovigilance question keeps global dense retrieval.
DOCUMENT_PATTERNS = {
    "D1": r"\b(?:D1|(?:peraturan|perbpom)\s+(?:BPOM\s+)?(?:nomor\s+)?4\s*(?:/|tahun\s*)\s*2026)\b",
    "D2": r"\b(?:D2|modul\s+(?:farmakovigilans\s+)?(?:untuk\s+)?industri\s+farmasi)\b",
    "D3": r"\b(?:D3|modul\s+(?:farmakovigilans\s+)?dasar)\b",
    "D4": r"\b(?:D4|buku\s+saku(?:\s+ESO)?)\b",
}


def required_document_ids(question: str) -> list[str]:
    return [doc for doc, pattern in DOCUMENT_PATTERNS.items()
            if re.search(pattern, question, re.IGNORECASE)]


def retrieval_query(question: str) -> str:
    """Ground a narrow personnel synonym without assuming the responsible actor."""
    if (re.search(r"\b(?:personel|petugas)\s+khusus\b", question, re.I)
            and re.search(r"keamanan\s+obat|farmakovigilans", question, re.I)):
        return question + " penunjukan penanggung jawab farmakovigilans"
    if required_document_ids(question):
        for pattern in DOCUMENT_PATTERNS.values():
            question = re.sub(pattern, " ", question, flags=re.I)
        question = " ".join(question.split())
    return question

class DenseRetriever:
    """Semantic retrieval using E5 embeddings and ChromaDB."""

    def __init__(
        self,
        embedder: E5Embedder,
        vector_store: ChromaVectorStore,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def close(self):
        self.vector_store.close()

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
        required = required_document_ids(question)
        if len(required) > top_k:
            raise ValueError("top_k must cover every explicitly requested document.")
        query_embedding = self.embedder.encode_query(
            retrieval_query(question)
        )

        # Step 2: Search ChromaDB
        if required:
            chunks = []
            for index, document_id in enumerate(required):
                quota = top_k // len(required) + int(index < top_k % len(required))
                results = self.vector_store.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(quota, index_count),
                    where={"document_id": document_id},
                    include=["documents", "metadatas", "distances"],
                )
                chunks.extend(self._parse(results))
            chunks.sort(key=lambda chunk: chunk["distance"])
            for rank, chunk in enumerate(chunks, 1):
                chunk["rank"] = rank
            return chunks

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

        return self._parse(results)

    @staticmethod
    def _parse(results: dict) -> list[dict]:
        """Parse Chroma's nested query result without changing score semantics."""
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

    def evidence_pages(self, chunks: list[dict]) -> list[dict]:
        """Read neighboring chunks on the same page; never re-embed or persist.

        Overlap is joined only when the actual texts share a suffix/prefix.
        A page boundary or unproven gap is never silently joined.
        """
        pages = []
        seen = set()
        for chunk in chunks:
            key = (chunk["document_id"], chunk["pdf_page"])
            if key in seen:
                continue
            seen.add(key)
            result = self.vector_store.collection.get(
                where={"$and": [
                    {"document_id": key[0]}, {"pdf_page": key[1]},
                ]}, include=["documents", "metadatas"],
            )
            siblings = sorted(zip(result["ids"], result["documents"]), key=lambda pair: pair[0])
            groups = []
            for chunk_id, text in siblings:
                if not groups:
                    groups.append((text, [chunk_id]))
                    continue
                previous, ids = groups[-1]
                overlap = next((size for size in range(min(len(previous), len(text)), 29, -1)
                                if previous.endswith(text[:size])), 0)
                if overlap:
                    groups[-1] = (previous + text[overlap:], ids + [chunk_id])
                else:
                    groups.append((text, [chunk_id]))
            for text, ids in groups:
                pages.append({**chunk, "text": text, "chunk_id": ids[0],
                              "context_chunk_ids": ids})
        return pages
