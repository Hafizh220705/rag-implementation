import pytest

from pharmaguard_rag.evaluation_metrics import (
    classify_generation_case,
    score_abstention_case,
    score_retrieval_case,
    summarize_abstention,
    summarize_retrieval,
)


def answerable_case(**overrides):
    case = {
        "question_id": "F01",
        "category": "factual",
        "verified": True,
        "should_abstain": False,
        "gold_chunk_ids": ["D1_p001_c001"],
    }
    case.update(overrides)
    return case


@pytest.mark.parametrize(
    "status, should_abstain, expected",
    [
        ("answered", False, "answered_pending_review"),
        ("answered", True, "possible_unsupported_answer"),
        ("insufficient_evidence", False, "possible_false_abstention"),
        ("insufficient_evidence", True, "expected_abstention"),
        ("citation_error", False, "citation_failure"),
        ("execution_error", False, "technical_failure"),
    ],
)
def test_generation_classification(status, should_abstain, expected):
    assert classify_generation_case(status, should_abstain) == expected


def test_hit_and_mrr_use_first_relevant_rank_within_k():
    score = score_retrieval_case(
        answerable_case(),
        [
            "D2_p001_c001",
            "D1_p001_c001",
            "D3_p001_c001",
            "D4_p001_c001",
            "D1_p002_c001",
        ],
        k=4,
    )

    assert score["hit"] == 1
    assert score["first_relevant_rank"] == 2
    assert score["reciprocal_rank"] == pytest.approx(0.5)
    assert len(score["retrieved_chunk_ids"]) == 4


def test_miss_outside_k_does_not_count():
    score = score_retrieval_case(
        answerable_case(),
        [
            "D2_p001_c001",
            "D2_p002_c001",
            "D3_p001_c001",
            "D4_p001_c001",
            "D1_p001_c001",
        ],
        k=4,
    )

    assert score["hit"] == 0
    assert score["first_relevant_rank"] is None
    assert score["reciprocal_rank"] == 0.0


def test_cross_document_coverage_counts_required_documents():
    score = score_retrieval_case(
        answerable_case(
            question_id="C01",
            category="cross_document",
            gold_chunk_ids=["D2_p002_c001", "D3_p002_c001"],
        ),
        ["D2_p002_c001", "D2_p003_c001", "D1_p001_c001"],
        k=4,
    )

    assert score["document_coverage"] == pytest.approx(0.5)


def test_summary_denominator_is_scored_verified_cases_only():
    scores = [
        score_retrieval_case(
            answerable_case(question_id="F01"),
            ["D1_p001_c001"],
        ),
        score_retrieval_case(
            answerable_case(question_id="F02"),
            ["D2_p001_c001"],
        ),
    ]

    summary = summarize_retrieval(scores)

    assert summary["evaluated_questions"] == 2
    assert summary["hit_at_k"] == pytest.approx(0.5)
    assert summary["mrr_at_k"] == pytest.approx(0.5)


def test_unverified_and_abstention_cases_are_not_retrieval_scored():
    with pytest.raises(ValueError, match="not been verified"):
        score_retrieval_case(
            answerable_case(verified=False),
            ["D1_p001_c001"],
        )

    with pytest.raises(ValueError, match="abstention evaluation"):
        score_retrieval_case(
            answerable_case(should_abstain=True),
            ["D1_p001_c001"],
        )


def test_abstention_is_scored_separately():
    case = answerable_case(
        question_id="I01",
        category="insufficient_evidence",
        should_abstain=True,
        gold_chunk_ids=[],
    )
    score = score_abstention_case(case, "insufficient_evidence")
    summary = summarize_abstention([score])

    assert score["abstention_correct"] == 1
    assert summary["abstention_accuracy"] == 1.0
    assert summary["question_ids"] == ["I01"]
