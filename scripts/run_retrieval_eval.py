import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.evaluation_metrics import (
    score_retrieval_case,
    summarize_retrieval,
)
from pharmaguard_rag.evaluation_validation import (
    validate_evaluation_dataset,
)
from pharmaguard_rag.retrieval_dense import DenseRetriever
from pharmaguard_rag.vector_store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation/datasets/pharmaguard_eval_v1.json"
CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
INDEX_DIR = ROOT / "data/indexes/chroma"
OUTPUT_PATH = ROOT / "outputs/evaluations/retrieval_baseline_v3.json"

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 4


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(report: dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    if output_path.exists():
        parser.error("Output exists. Use a new filename.")
    print("=== PHARMAGUARD RETRIEVAL EVALUATION V3 ===")

    questions = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    with CHUNKS_PATH.open(encoding="utf-8") as file:
        chunks = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    validation = validate_evaluation_dataset(
        questions,
        {chunk["chunk_id"] for chunk in chunks},
    )

    if not validation["valid"]:
        raise ValueError(
            "Invalid evaluation dataset: "
            + "; ".join(validation["errors"])
        )

    verified_cases = [
        item for item in questions
        if item["verified"] and not item["should_abstain"]
    ]
    verified_abstention_cases = [
        item for item in questions
        if item["verified"] and item["should_abstain"]
    ]

    print("Total questions:", len(questions))
    print("Verified answerable:", len(verified_cases))
    print("Verified abstention:", len(verified_abstention_cases))
    print("Pending verification:", validation["pending_questions"])

    scores = []
    collection_name = None
    corpus_sha256 = file_sha256(CHUNKS_PATH)

    if verified_cases:
        embedder = E5Embedder(model_name=MODEL_NAME, device="cpu")
        store = ChromaVectorStore(
            persist_directory=INDEX_DIR,
            model_name=MODEL_NAME,
            corpus_sha256=corpus_sha256,
        )
        collection_name = store.collection.name
        retriever = DenseRetriever(
            embedder=embedder,
            vector_store=store,
        )

        for case in verified_cases:
            results = retriever.search(
                question=case["question"],
                top_k=TOP_K,
            )
            score = score_retrieval_case(
                case=case,
                retrieved_ids=[
                    chunk["chunk_id"] for chunk in results
                ],
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
        store.close()

    else:
        summary = {
            "evaluated_questions": 0,
            "hit_at_k": None,
            "mrr_at_k": None,
            "cross_document_coverage_at_k": None,
            "cross_document_questions": 0,
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "retrieval_baseline_v3",
        "retrieval_method": "dense_with_explicit_document_constraints",
        "embedding_model": MODEL_NAME,
        "top_k": TOP_K,
        "dataset_sha256": file_sha256(DATASET_PATH),
        "corpus_sha256": corpus_sha256,
        "collection_name": collection_name,
        "total_dataset_questions": len(questions),
        "metric_denominator": "verified answerable questions only",
        "evaluated_question_ids": [
            case["question_id"] for case in verified_cases
        ],
        "verified_abstention_question_ids": [
            case["question_id"]
            for case in verified_abstention_cases
        ],
        "pending_question_ids": validation["pending_question_ids"],
        "summary": summary,
        "per_question": scores,
        "limitations": [
            "Unverified questions are excluded from retrieval metrics.",
            "Insufficient-evidence questions require separate "
            "abstention evaluation.",
            "This development set is not a held-out benchmark.",
        ],
    }

    write_report(report, output_path)

    if scores:
        print("\n=== EVALUATION SUMMARY ===")
        print("Evaluated:", summary["evaluated_questions"])
        print(f"Hit@{TOP_K}: {summary['hit_at_k']:.4f}")
        print(f"MRR@{TOP_K}: {summary['mrr_at_k']:.4f}")
    else:
        print(
            "\n[NOT SCORED] No verified answerable questions. "
            "Coverage is reported as 0/24, not as final accuracy."
        )

    print("Report:", output_path)


if __name__ == "__main__":
    main()
