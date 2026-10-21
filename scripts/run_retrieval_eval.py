import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.vector_store import ChromaVectorStore
from pharmaguard_rag.retrieval_dense import DenseRetriever
from pharmaguard_rag.evaluation_metrics import (
    score_retrieval_case,
    summarize_retrieval,
)

ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT / "evaluation/datasets/pharmaguard_eval_v1.json"
)

CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
INDEX_DIR = ROOT / "data/indexes/chroma"

OUTPUT_PATH = (
    ROOT / "outputs/evaluations/retrieval_baseline_v1.json"
)

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 4


def main():
    print("=== PHARMAGUARD RETRIEVAL EVALUATION ===")

    questions = json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

    verified_cases = [
        item for item in questions
        if item["verified"] and not item["should_abstain"]
    ]

    print("Total questions:", len(questions))
    print("Verified answerable:", len(verified_cases))
    print("Pending verification:", sum(
        not item["verified"] for item in questions
    ))

    if not verified_cases:
        print(
            "\n[SKIP] No verified answerable questions. "
            "Complete gold evidence annotation first."
        )
        return

    corpus_sha256 = hashlib.sha256(
        CHUNKS_PATH.read_bytes()
    ).hexdigest()

    embedder = E5Embedder(
        model_name=MODEL_NAME,
        device="cpu",
    )

    store = ChromaVectorStore(
        persist_directory=INDEX_DIR,
        model_name=MODEL_NAME,
        corpus_sha256=corpus_sha256,
    )

    retriever = DenseRetriever(
        embedder=embedder,
        vector_store=store,
    )

    scores = []

    for case in verified_cases:
        results = retriever.search(
            question=case["question"],
            top_k=TOP_K,
        )

        retrieved_ids = [
            chunk["chunk_id"] for chunk in results
        ]

        score = score_retrieval_case(
            case=case,
            retrieved_ids=retrieved_ids,
            k=TOP_K,
        )

        scores.append(score)

        print(
            f"{case['question_id']} | "
            f"Hit@{TOP_K}: {score['hit']} | "
            f"RR: {score['reciprocal_rank']:.4f} | "
            f"First rank: {score['first_relevant_rank']}"
        )

    summary = summarize_retrieval(scores)

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "retrieval_method": "dense",
        "embedding_model": MODEL_NAME,
        "top_k": TOP_K,
        "total_dataset_questions": len(questions),
        "verified_answerable_questions": len(
            verified_cases
        ),
        "summary": summary,
        "per_question": scores,
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

    print("\n=== EVALUATION SUMMARY ===")
    print("Evaluated:", summary["evaluated_questions"])
    print(f"Hit@{TOP_K}: {summary['hit_at_k']:.4f}")
    print(f"MRR@{TOP_K}: {summary['mrr_at_k']:.4f}")

    coverage = summary["cross_document_coverage_at_k"]

    if coverage is not None:
        print(
            f"Cross-document Coverage@{TOP_K}: "
            f"{coverage:.4f}"
        )

    print("Report:", OUTPUT_PATH)

if __name__ == "__main__":
    main()