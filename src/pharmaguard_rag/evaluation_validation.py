from collections import Counter

EXPECTED_CATEGORY_COUNTS = {
    "factual": 8,
    "paraphrase": 6,
    "cross_document": 6,
    "insufficient_evidence": 4,
}

def _document_id(chunk_id: str) -> str:
    return chunk_id.split("_p", 1)[0]

def validate_evaluation_dataset(
    cases: list[dict],
    available_chunk_ids: set[str],
) -> dict:
    """Validate structure and verified annotations without inventing labels."""

    errors = []
    question_ids = [case.get("question_id") for case in cases]

    duplicate_ids = sorted(
        question_id
        for question_id, count in Counter(question_ids).items()
        if question_id is not None and count > 1
    )

    if duplicate_ids:
        errors.append(f"Duplicate question IDs: {duplicate_ids}")

    category_counts = Counter(case.get("category") for case in cases)

    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        errors.append(
            "Unexpected category distribution: "
            f"{dict(category_counts)}"
        )

    verified_ids = []
    pending_ids = []
    verified_answerable_ids = []
    verified_abstention_ids = []

    for case in cases:
        question_id = case.get("question_id", "<missing>")
        verified = case.get("verified") is True
        should_abstain = case.get("should_abstain") is True
        gold_ids = case.get("gold_chunk_ids") or []
        reference_answer = (case.get("reference_answer") or "").strip()

        unknown_chunks = sorted(
            set(gold_ids) - available_chunk_ids
        )

        if unknown_chunks:
            errors.append(
                f"{question_id}: unknown gold chunks {unknown_chunks}"
            )

        if verified:
            verified_ids.append(question_id)

            if should_abstain:
                verified_abstention_ids.append(question_id)
            else:
                verified_answerable_ids.append(question_id)

                if not gold_ids:
                    errors.append(
                        f"{question_id}: verified answerable case "
                        "has no gold chunks"
                    )

                if not reference_answer:
                    errors.append(
                        f"{question_id}: verified answerable case "
                        "has no reference answer"
                    )

                if case.get("category") == "cross_document":
                    documents = {
                        _document_id(chunk_id)
                        for chunk_id in gold_ids
                    }

                    if len(documents) < 2:
                        errors.append(
                            f"{question_id}: verified cross-document "
                            "case requires at least two documents"
                        )

        else:
            pending_ids.append(question_id)

    return {
        "valid": not errors,
        "errors": errors,
        "total_questions": len(cases),
        "category_counts": dict(category_counts),
        "verified_questions": len(verified_ids),
        "verified_question_ids": verified_ids,
        "verified_answerable_questions": len(
            verified_answerable_ids
        ),
        "verified_answerable_question_ids": verified_answerable_ids,
        "verified_abstention_questions": len(
            verified_abstention_ids
        ),
        "verified_abstention_question_ids": verified_abstention_ids,
        "pending_questions": len(pending_ids),
        "pending_question_ids": pending_ids,
    }