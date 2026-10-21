import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BASELINE_PATH = (
    ROOT / "outputs/evaluations/generation_baseline_v1.json"
)

REVIEW_PATH = (
    ROOT / "evaluation/datasets/generation_human_review_v1.json"
)

OUTPUT_PATH = (
    ROOT / "outputs/evaluations/failure_analysis_v1.json"
)

MARKDOWN_PATH = (
    ROOT / "docs/baseline_failure_analysis.md"
)


def classify_case(case):
    status = case["status"]
    should_abstain = case["should_abstain"]

    if status == "execution_error":
        return "technical_failure"

    if status == "citation_error":
        return "citation_failure"

    if status == "insufficient_evidence":
        if should_abstain:
            return "expected_abstention"
        return "possible_false_abstention"

    if status == "answered":
        if should_abstain:
            return "possible_unsupported_answer"
        return "answered_pending_review"

    return "unknown_status"


def main():
    baseline = json.loads(
        BASELINE_PATH.read_text(encoding="utf-8")
    )

    reviews = json.loads(
        REVIEW_PATH.read_text(encoding="utf-8")
    )

    cases = baseline["per_question"]

    if len(cases) != len(reviews):
        raise ValueError(
            "Baseline and review dataset sizes do not match."
        )

    review_by_id = {
        item["question_id"]: item
        for item in reviews
    }

    if len(review_by_id) != len(reviews):
        raise ValueError("Duplicate review question IDs.")

    if set(review_by_id) != {
        item["question_id"] for item in cases
    }:
        raise ValueError(
            "Baseline and review question IDs do not match."
        )

    classified = []
    counts = Counter()

    for case in cases:
        question_id = case["question_id"]
        classification = classify_case(case)

        review = review_by_id[question_id]["human_review"]

        counts[classification] += 1

        classified.append({
            "question_id": question_id,
            "category": case["category"],
            "question": case["question"],
            "status": case["status"],
            "classification": classification,
            "human_verified": review["verified"],
            "latency_seconds": case["latency_seconds"],
            "execution_error": case["execution_error"],
            "retrieved_chunk_ids": [
                item["chunk_id"]
                for item in case["retrieved_chunks"]
            ],
        })

    verified_count = sum(
        item["human_verified"] for item in classified
    )

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": "generation_baseline_v1",
        "total_questions": len(cases),
        "human_verified": verified_count,
        "status_counts": dict(
            Counter(item["status"] for item in cases)
        ),
        "failure_classifications": dict(counts),
        "limitations": [
            "Classification is based on model status.",
            "Possible false abstentions are not confirmed.",
            "Faithfulness has not been fully evaluated.",
            "Gold evidence verification is incomplete.",
        ],
        "cases": classified,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Generate a human-readable Markdown report
    lines = [
        "# PharmaGuard-RAG — Baseline Failure Analysis",
        "",
        "## Experiment Overview",
        "",
        f"- Total questions: {len(cases)}",
        f"- Human-verified reviews: {verified_count}",
        "- Retrieval: multilingual-e5-small + ChromaDB",
        "- Generation: Qwen2.5 3B via Ollama",
        "",
        "## Failure Classification",
        "",
        "| Classification | Count |",
        "|---|---:|",
    ]

    for name, count in sorted(counts.items()):
        lines.append(f"| {name} | {count} |")

    lines.extend([
        "",
        "## Cases Requiring Investigation",
        "",
        "| ID | Classification | Status |",
        "|---|---|---|",
    ])

    for item in classified:
        if item["classification"] in {
            "possible_false_abstention",
            "possible_unsupported_answer",
            "citation_failure",
            "technical_failure",
        }:
            lines.append(
                f"| {item['question_id']} | "
                f"{item['classification']} | "
                f"{item['status']} |"
            )

    lines.extend([
        "",
        "## Known Limitations",
        "",
        "- Gold annotations are not yet complete.",
        "- Retrieval Hit@4 and MRR@4 are not final.",
        "- Citation structure does not prove faithfulness.",
        "- Failure classifications require human review.",
        "",
        "## Next Steps",
        "",
        "1. Verify gold evidence against source PDFs.",
        "2. Review retrieved evidence for rejected questions.",
        "3. Investigate generation truncation and citation errors.",
        "4. Evaluate answer faithfulness and completeness.",
        "5. Apply revisions and rerun the same evaluation set.",
    ])

    MARKDOWN_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    MARKDOWN_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("=== BASELINE FAILURE ANALYSIS ===")
    print("Total questions:", len(cases))
    print("Human verified:", verified_count)

    for name, count in sorted(counts.items()):
        print(f"{name}: {count}")

    print("\nJSON report:", OUTPUT_PATH)
    print("Markdown report:", MARKDOWN_PATH)

if __name__ == "__main__":
    main()