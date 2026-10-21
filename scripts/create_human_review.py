import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT / "outputs/evaluations/generation_baseline_v1.json"
)

OUTPUT_PATH = (
    ROOT / "evaluation/datasets/generation_human_review_v1.json"
)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            "Review file already exists. "
            "Refusing to overwrite manual annotations."
        )

    report = json.loads(
        INPUT_PATH.read_text(encoding="utf-8")
    )

    records = []

    for item in report["per_question"]:
        status = item["status"]

        if status == "execution_error":
            failure_candidate = "technical_failure"
        elif status == "citation_error":
            failure_candidate = "citation_failure"
        elif (
            status == "insufficient_evidence"
            and not item["should_abstain"]
        ):
            failure_candidate = "possible_false_abstention"
        elif (
            status == "answered"
            and item["should_abstain"]
        ):
            failure_candidate = "possible_unsupported_answer"
        else:
            failure_candidate = None

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

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
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
    print("Output:", OUTPUT_PATH)

if __name__ == "__main__":
    main()