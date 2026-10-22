import json
from pathlib import Path

from pharmaguard_rag.evaluation_validation import (
    validate_evaluation_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation/datasets/pharmaguard_eval_v1.json"
CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"


def main() -> int:
    cases = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        chunk_ids = {
            json.loads(line)["chunk_id"]
            for line in file
            if line.strip()
        }

    report = validate_evaluation_dataset(cases, chunk_ids)

    print("=== EVALUATION DATASET VALIDATION ===")
    print("Total questions:", report["total_questions"])
    print("Category counts:", report["category_counts"])
    print("Verified:", report["verified_questions"])
    print("Pending:", report["pending_questions"])

    if report["errors"]:
        print("\nErrors:")
        for error in report["errors"]:
            print("-", error)
        return 1

    if report["pending_questions"]:
        print(
            "\n[VALID, PENDING REVIEW] Structure and references are "
            "valid, but unverified cases are excluded from metrics."
        )
    else:
        print("\n[VALID] All annotations are verified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
