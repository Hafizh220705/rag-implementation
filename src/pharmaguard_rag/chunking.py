import re

def clean_text(text: str) -> str:
    """Normalize whitespace without removing important text."""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize repeated spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def chunk_page(
    page: dict,
    tokenizer,
    chunk_size: int = 450,
    chunk_overlap: int = 75,
) -> list[dict]:
    """Split one PDF page into token-based chunks."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Invalid chunk overlap")

    text = clean_text(page["text"])

    if not text:
        return []

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=False,
        verbose=False,
    )

    chunks = []
    start = 0
    chunk_number = 1

    while start < len(token_ids):
        end = min(start + chunk_size, len(token_ids))

        chunk_tokens = token_ids[start:end]

        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        if chunk_text:
            chunk_id = (
                f"{page['document_id']}_"
                f"p{page['pdf_page']:03d}_"
                f"c{chunk_number:03d}"
            )

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": page["document_id"],
                "source_file": page["source_file"],
                "pdf_page": page["pdf_page"],
                "chunk_number": chunk_number,
                "text": chunk_text,
                "token_count": len(chunk_tokens),
            })

            chunk_number += 1

        if end == len(token_ids):
            break

        start = end - chunk_overlap

    return chunks