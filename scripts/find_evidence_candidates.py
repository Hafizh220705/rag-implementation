import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHUNKS_PATH = (
    ROOT / "data" / "processed" / "chunks.jsonl"
)

STOPWORDS = {
    "apa", "yang", "dan", "atau", "dalam",
    "dengan", "untuk", "dari", "pada",
    "bagaimana", "menurut", "adalah",
    "harus", "dapat", "ini", "itu",
}

def tokenize(text):
    words = re.findall(
        r"\b\w+\b",
        text.lower(),
    )

    return {
        word
        for word in words
        if len(word) >= 3
        and word not in STOPWORDS
    }

def load_chunks():
    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python scripts/find_evidence_candidates.py '
            '"kata kunci"'
        )
        return

    query = " ".join(sys.argv[1:])
    query_words = tokenize(query)

    if not query_words:
        raise ValueError(
            "Enter more specific search terms."
        )

    matches = []

    for chunk in load_chunks():
        chunk_words = tokenize(chunk["text"])

        common = query_words & chunk_words

        if not common:
            continue

        matches.append({
            "score": len(common),
            "matched_words": sorted(common),
            "chunk": chunk,
        })

    matches.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    print(f"\nQuery: {query}")
    print(f"Candidates: {len(matches)}")

    for item in matches[:10]:
        chunk = item["chunk"]

        print("\n" + "-" * 60)
        print("Chunk ID:", chunk["chunk_id"])
        print("Document:", chunk["document_id"])
        print("PDF Page:", chunk["pdf_page"])
        print("Matching words:", item["matched_words"])
        print("Text:")
        print(chunk["text"][:650])

if __name__ == "__main__":
    main()