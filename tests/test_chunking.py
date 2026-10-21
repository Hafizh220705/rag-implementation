import pytest

from pharmaguard_rag.chunking import (
    clean_text,
    chunk_page,
)

class MockTokenizer:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}

    def encode(
        self,
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    ):
        ids = []

        for word in text.split():
            if word not in self.vocab:
                token_id = len(self.vocab) + 1
                self.vocab[word] = token_id
                self.reverse_vocab[token_id] = word

            ids.append(self.vocab[word])

        return ids


    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        return " ".join(
            self.reverse_vocab[token_id]
            for token_id in token_ids
        )

def test_clean_text():
    result = clean_text("Halo   dunia\n\n\nTes")

    assert result == "Halo dunia\n\nTes"


def test_chunk_page():
    tokenizer = MockTokenizer()

    page = {
        "document_id": "TEST",
        "source_file": "sample.pdf",
        "pdf_page": 1,
        "text": " ".join(f"kata{i}" for i in range(12)),
    }

    chunks = chunk_page(
        page=page,
        tokenizer=tokenizer,
        chunk_size=5,
        chunk_overlap=2,
    )

    assert len(chunks) == 4

    # Check unique chunk IDs
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))

    # Check maximum chunk size
    assert all(c["token_count"] <= 5 for c in chunks)

    # Check overlap between consecutive chunks
    for first, second in zip(chunks, chunks[1:]):
        first_words = first["text"].split()
        second_words = second["text"].split()

        assert first_words[-2:] == second_words[:2]


def test_invalid_chunk_size():
    tokenizer = MockTokenizer()

    page = {
        "document_id": "TEST",
        "source_file": "sample.pdf",
        "pdf_page": 1,
        "text": "Contoh teks",
    }

    with pytest.raises(ValueError):
        chunk_page(page, tokenizer, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_page(
            page,
            tokenizer,
            chunk_size=5,
            chunk_overlap=5,
        )