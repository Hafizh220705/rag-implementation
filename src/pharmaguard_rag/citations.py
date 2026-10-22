import re

SOURCE_GROUP_PATTERN = re.compile(
    r"\[(S\d+(?:\s*,\s*S\d+)*)\]",
    flags=re.IGNORECASE,
)
SOURCE_TOKEN_PATTERN = re.compile(r"S(\d+)", flags=re.IGNORECASE)
SOURCE_PLACEHOLDER_PATTERN = re.compile(
    r"\[[^\]]*[Ss](?:\s*\d+|[A-Za-z]+)[^\]]*\]",
    flags=re.IGNORECASE,
)


def extract_citation_labels(answer: str) -> list[list[str]]:
    """Return normalized source-label groups in order of appearance."""

    groups = []

    for match in SOURCE_GROUP_PATTERN.finditer(answer):
        labels = [
            f"S{number}"
            for number in SOURCE_TOKEN_PATTERN.findall(match.group(1))
        ]

        groups.append(labels)

    return groups

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

    matches = list(SOURCE_GROUP_PATTERN.finditer(answer))

    if not matches:
        if SOURCE_PLACEHOLDER_PATTERN.search(answer):
            raise ValueError("Answer contains an invalid citation format.")

        raise ValueError("Answer contains no source citations.")

    cited_sources = []
    seen_labels = set()

    for match in matches:
        labels = [
            f"S{number}"
            for number in SOURCE_TOKEN_PATTERN.findall(match.group(1))
        ]

        for label in labels:
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

    def replace_group(match):
        labels = []

        for number in SOURCE_TOKEN_PATTERN.findall(match.group(1)):
            label = f"S{number}"

            if label not in labels:
                labels.append(label)

        references = []

        for label in labels:
            chunk = source_map[label]

            references.append(
                f"[{chunk['document_id']}, "
                f"PDF hlm. {chunk['pdf_page']}, "
                f"{', '.join(chunk.get('context_chunk_ids', [chunk['chunk_id']]))}]"
            )

        return "; ".join(references)

    resolved_answer = SOURCE_GROUP_PATTERN.sub(
        replace_group,
        answer,
    )

    if SOURCE_PLACEHOLDER_PATTERN.search(resolved_answer):
        raise ValueError("Answer contains an unresolved citation placeholder.")

    return resolved_answer, cited_sources
