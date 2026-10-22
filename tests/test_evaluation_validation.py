from pharmaguard_rag.evaluation_validation import (
    validate_evaluation_dataset,
)


def make_dataset():
    cases = []
    counts = {
        "factual": 8,
        "paraphrase": 6,
        "cross_document": 6,
        "insufficient_evidence": 4,
    }

    index = 1
    for category, count in counts.items():
        for _ in range(count):
            cases.append({
                "question_id": f"Q{index:02d}",
                "category": category,
                "question": "Question",
                "should_abstain": category == "insufficient_evidence",
                "gold_chunk_ids": [],
                "reference_answer": "",
                "verified": False,
            })
            index += 1

    return cases


def test_pending_dataset_is_valid_but_not_scored():
    report = validate_evaluation_dataset(make_dataset(), set())

    assert report["valid"] is True
    assert report["verified_questions"] == 0
    assert report["pending_questions"] == 24


def test_verified_cross_document_case_requires_two_documents():
    cases = make_dataset()
    case = next(
        item for item in cases
        if item["category"] == "cross_document"
    )
    case.update({
        "verified": True,
        "gold_chunk_ids": ["D2_p002_c001"],
        "reference_answer": "Reference",
    })

    report = validate_evaluation_dataset(
        cases,
        {"D2_p002_c001"},
    )

    assert report["valid"] is False
    assert "at least two documents" in " ".join(report["errors"])


def test_unknown_gold_chunk_is_rejected():
    cases = make_dataset()
    cases[0]["gold_chunk_ids"] = ["UNKNOWN_p001_c001"]

    report = validate_evaluation_dataset(cases, set())

    assert report["valid"] is False
    assert "unknown gold chunks" in " ".join(report["errors"])
