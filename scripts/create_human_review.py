import argparse
import json
from pathlib import Path

from pharmaguard_rag.evaluation_metrics import (
    classify_generation_case,
)

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "outputs/evaluations/generation_baseline_v3_final.json"
)

OUTPUT_PATH = (
    ROOT / "evaluation/datasets/generation_human_review_v3.json"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if output_path.exists():
        raise FileExistsError(
            "Review file already exists. "
            "Refusing to overwrite manual annotations."
        )

    report = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    records = []

    for item in report["per_question"]:
        status = item["status"]
        failure_candidate = classify_generation_case(
            status=status,
            should_abstain=item["should_abstain"],
        )

        records.append({
            "question_id": item["question_id"],
            "category": item["category"],
            "question": item["question"],
            "should_abstain": item["should_abstain"],
            "model_status": status,
            "model_answer": item["answer"],
            "raw_answer": item["raw_answer"],
            "cited_sources": item["cited_sources"],
            "retrieved_chunks": item["retrieved_chunks"],
            "evidence_pages": item.get("evidence_pages", []),
            "selected_evidence": item.get("selected_evidence", []),
            "abstention_reason": item.get("abstention_reason"),
            "failure_candidate": failure_candidate,
            "human_review": {
                "evidence_sufficient": None,
                "faithfulness": None,
                "citation_correctness": None,
                "answer_completeness": None,
                "abstention_correct": None,
                "review_notes": "",
                "verified": False,
            },
        })

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== HUMAN REVIEW DATASET ===")
    print("Total cases:", len(records))
    print("Pending manual review:", len(records))
    print("Output:", output_path)

if __name__ == "__main__":
    main()
