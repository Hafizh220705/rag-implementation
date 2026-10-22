def classify_generation_case(
    status: str,
    should_abstain: bool,
) -> str | None:
    """Classify a generation outcome as a review candidate."""

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


def score_retrieval_case(
    case: dict,
    retrieved_ids: list[str],
    k: int = 4,
) -> dict:
    """Evaluate retrieval against manually verified gold chunks."""

    if not case.get("verified"):
        raise ValueError("Question has not been verified.")

    if case["should_abstain"]:
        raise ValueError(
            "Insufficient-evidence cases require abstention evaluation."
        )

    if k <= 0:
        raise ValueError("k must be positive.")

    gold_ids = set(case["gold_chunk_ids"])

    if not gold_ids:
        raise ValueError(
            f"Missing gold chunks: {case['question_id']}"
        )

    top_ids = retrieved_ids[:k]

    # Find the rank of the first relevant chunk.
    first_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(top_ids, start=1)
            if chunk_id in gold_ids
        ),
        None,
    )

    hit = 1 if first_rank is not None else 0
    reciprocal_rank = (
        1.0 / first_rank if first_rank is not None else 0.0
    )

    # Additional metric for cross-document questions.
    document_coverage = None

    if case["category"] == "cross_document":
        gold_documents = {
            chunk_id.split("_p", 1)[0]
            for chunk_id in gold_ids
        }

        if len(gold_documents) < 2:
            raise ValueError(
                "Cross-document gold evidence must "
                "include at least two documents."
            )

        retrieved_documents = {
            chunk_id.split("_p", 1)[0]
            for chunk_id in top_ids
        }

        document_coverage = (
            len(gold_documents & retrieved_documents)
            / len(gold_documents)
        )

    return {
        "question_id": case["question_id"],
        "category": case["category"],
        "hit": hit,
        "reciprocal_rank": reciprocal_rank,
        "first_relevant_rank": first_rank,
        "document_coverage": document_coverage,
        "gold_chunk_ids": sorted(gold_ids),
        "retrieved_chunk_ids": top_ids,
    }


def summarize_retrieval(scores: list[dict]) -> dict:
    """Aggregate metrics across verified answerable questions."""

    if not scores:
        raise ValueError("No evaluated questions available.")

    n = len(scores)

    cross_document_scores = [
        score["document_coverage"]
        for score in scores
        if score["document_coverage"] is not None
    ]

    return {
        "evaluated_questions": n,
        "hit_at_k": sum(
            score["hit"] for score in scores
        ) / n,
        "mrr_at_k": sum(
            score["reciprocal_rank"] for score in scores
        ) / n,
        "cross_document_coverage_at_k": (
            sum(cross_document_scores)
            / len(cross_document_scores)
            if cross_document_scores else None
        ),
        "cross_document_questions": len(
            cross_document_scores
        ),
    }


def score_abstention_case(
    case: dict,
    pipeline_status: str,
) -> dict:
    """Score one verified insufficient-evidence question separately."""

    if not case.get("verified"):
        raise ValueError("Question has not been verified.")

    if not case["should_abstain"]:
        raise ValueError("Question is not an abstention case.")

    return {
        "question_id": case["question_id"],
        "pipeline_status": pipeline_status,
        "abstention_correct": int(
            pipeline_status == "insufficient_evidence"
        ),
    }


def summarize_abstention(scores: list[dict]) -> dict:
    """Aggregate verified insufficient-evidence cases."""

    if not scores:
        raise ValueError("No abstention scores available.")

    return {
        "evaluated_questions": len(scores),
        "question_ids": [score["question_id"] for score in scores],
        "abstention_accuracy": sum(
            score["abstention_correct"] for score in scores
        ) / len(scores),
    }
