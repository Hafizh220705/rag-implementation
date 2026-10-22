import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVAL_PATH = (
    ROOT / "evaluation/datasets/pharmaguard_eval_v1.json"
)

DEFAULT_REPORT_PATH = (
    ROOT / "outputs/evaluations/generation_baseline_v2.json"
)

CHUNKS_PATH = (
    ROOT / "data/processed/chunks.jsonl"
)

def main():
    parser = argparse.ArgumentParser(
        description="Review candidate evidence for an evaluation question"
    )

    parser.add_argument("question_id")
    parser.add_argument("chunk_ids", nargs="*")
    parser.add_argument(
        "--report",
        type=Path,
        help=(
            "Use retrieved candidates from a generation report when "
            "chunk IDs are omitted."
        ),
    )

    args = parser.parse_args()

    # Load evaluation questions
    questions = json.loads(
        EVAL_PATH.read_text(encoding="utf-8")
    )

    question = next(
        (
            item for item in questions
            if item["question_id"] == args.question_id
        ),
        None,
    )

    if question is None:
        raise ValueError(
            f"Unknown question ID: {args.question_id}"
        )

    chunk_ids = list(args.chunk_ids)

    if not chunk_ids:
        report_path = args.report or DEFAULT_REPORT_PATH

        if not report_path.is_absolute():
            report_path = ROOT / report_path

        if not report_path.exists() and args.report is None:
            report_path = (
                ROOT
                / "outputs/evaluations/generation_baseline_v1.json"
            )

        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        result = next(
            (
                item for item in report["per_question"]
                if item["question_id"] == args.question_id
            ),
            None,
        )

        if result is None:
            raise ValueError(
                f"Question {args.question_id} not found in "
                f"{report_path}"
            )

        chunk_ids = [
            item["chunk_id"]
            for item in result["retrieved_chunks"]
        ]

        print("Candidate source:", report_path)

    # Load corpus chunks
    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        chunks = {
            chunk["chunk_id"]: chunk
            for line in file
            if line.strip()
            for chunk in [json.loads(line)]
        }

    print("\n=== EVIDENCE REVIEW ===")
    print("Question ID:", question["question_id"])
    print("Category:", question["category"])
    print("Question:", question["question"])

    for chunk_id in chunk_ids:
        if chunk_id not in chunks:
            print(f"\n[NOT FOUND] {chunk_id}")
            continue

        chunk = chunks[chunk_id]

        print("\n" + "=" * 70)
        print("Chunk ID:", chunk_id)
        print("Document:", chunk["document_id"])
        print("Title:", chunk.get("title"))
        print("PDF Page:", chunk["pdf_page"])
        print(
            "Original PDF:",
            ROOT / "data/raw" / chunk["source_file"],
        )
        print("Source URL:", chunk.get("source_url"))
        print("=" * 70)

        # Show full chunk rather than truncated preview
        print(chunk["text"])

    print("\nReview these passages against the original PDFs.")
    print("Only then annotate gold_chunk_ids and verified.")

if __name__ == "__main__":
    main()
