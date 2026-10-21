import re

SOURCE_PATTERN = re.compile(r"\[S(\d+)\]")

def resolve_citations(
    answer: str,
    retrieved_chunks: list[dict],
) -> tuple[str, list[dict]]:
    """Validate and resolve LLM source labels."""

    source_map = {
        f"S{index}": chunk
        for index, chunk in enumerate(
            retrieved_chunks,
            start=1,
        )
    }

    referenced_labels = SOURCE_PATTERN.findall(answer)

    if not referenced_labels:
        raise ValueError("Answer contains no source citations.")

    cited_sources = []
    seen_labels = set()

    for number in referenced_labels:
        label = f"S{number}"

        if label not in source_map:
            raise ValueError(
                f"Unknown citation label: [{label}]"
            )

        if label in seen_labels:
            continue

        seen_labels.add(label)
        chunk = source_map[label]

        cited_sources.append({
            "label": label,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "title": chunk.get("title"),
            "source_file": chunk.get("source_file"),
            "pdf_page": chunk["pdf_page"],
            "source_url": chunk.get("source_url"),
        })

    def replace_label(match):
        label = f"S{match.group(1)}"
        chunk = source_map[label]

        return (
            f"[{chunk['document_id']}, "
            f"PDF hlm. {chunk['pdf_page']}, "
            f"{chunk['chunk_id']}]"
        )

    resolved_answer = SOURCE_PATTERN.sub(
        replace_label,
        answer,
    )

    return resolved_answer, cited_sources