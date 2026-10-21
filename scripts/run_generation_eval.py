import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.vector_store import ChromaVectorStore
from pharmaguard_rag.retrieval_dense import DenseRetriever
from pharmaguard_rag.generation import OllamaGenerator
from pharmaguard_rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT / "evaluation/datasets/pharmaguard_eval_v1.json"
)

CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
INDEX_DIR = ROOT / "data/indexes/chroma"

OUTPUT_PATH = (
    ROOT / "outputs/evaluations/generation_baseline_v1.json"
)

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 4

SMOKE_IDS = ["F01", "F05", "F03", "I01"]

def load_dataset():
    return json.loads(
        DATASET_PATH.read_text(encoding="utf-8")
    )

def save_report(results):
    statuses = Counter(
        item["status"] for item in results
    )

    latencies = [
        item["latency_seconds"]
        for item in results
        if item["latency_seconds"] is not None
    ]

    # These are review candidates, NOT error metrics.
    abstention_candidates = [
        item["question_id"]
        for item in results
        if not item["should_abstain"]
        and item["status"] == "insufficient_evidence"
    ]

    unexpected_answer_candidates = [
        item["question_id"]
        for item in results
        if item["should_abstain"]
        and item["status"] == "answered"
    ]

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "pipeline": "baseline-v1",
        "retrieval_method": "dense",
        "embedding_model": MODEL_NAME,
        "generation_model": "qwen2.5:3b",
        "top_k": TOP_K,
        "summary": {
            "evaluated_cases": len(results),
            "status_counts": dict(statuses),
            "mean_latency_seconds": (
                round(sum(latencies) / len(latencies), 3)
                if latencies else None
            ),
            "abstentions_to_review": abstention_candidates,
            "unexpected_answers_to_review": (
                unexpected_answer_candidates
            ),
        },
        "limitations": [
            "Gold evidence is not fully verified.",
            "Status counts are not accuracy metrics.",
            "Faithfulness requires manual review.",
            "Citation structure does not prove support.",
        ],
        "per_question": results,
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

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all 24 questions.",
    )

    parser.add_argument(
        "--ids",
        nargs="+",
        help="Evaluate selected question IDs.",
    )

    args = parser.parse_args()

    if args.all and args.ids:
        parser.error("Use either --all or --ids.")

    dataset = load_dataset()

    if args.all:
        selected = dataset
    else:
        selected_ids = set(args.ids or SMOKE_IDS)

        selected = [
            case for case in dataset
            if case["question_id"] in selected_ids
        ]

        found_ids = {
            case["question_id"] for case in selected
        }

        missing = selected_ids - found_ids

        if missing:
            raise ValueError(
                f"Unknown question IDs: {sorted(missing)}"
            )

    print("=== GENERATION EVALUATION ===")
    print("Selected questions:", len(selected))

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

    retriever = DenseRetriever(embedder, store)
    generator = OllamaGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        generator=generator,
    )

    results = []

    for case in selected:
        question_id = case["question_id"]

        print("\n" + "=" * 60)
        print(question_id, "-", case["question"])

        try:
            response = pipeline.ask(
                question=case["question"],
                top_k=TOP_K,
            )

            result = {
                "question_id": question_id,
                "category": case["category"],
                "question": case["question"],
                "gold_verified": case["verified"],
                "should_abstain": case["should_abstain"],
                "status": response["status"],
                "answer": response["answer"],
                "raw_answer": response["raw_answer"],
                "cited_sources": response["cited_sources"],
                "retrieved_chunks": response["retrieved_chunks"],
                "latency_seconds": response["latency_seconds"],
                "execution_error": None,
            }

        except Exception as error:
            # Record a technical failure separately
            # from a model abstention.
            result = {
                "question_id": question_id,
                "category": case["category"],
                "question": case["question"],
                "gold_verified": case["verified"],
                "should_abstain": case["should_abstain"],
                "status": "execution_error",
                "answer": "",
                "raw_answer": "",
                "cited_sources": [],
                "retrieved_chunks": [],
                "latency_seconds": None,
                "execution_error": str(error),
            }

        results.append(result)

        print("Status:", result["status"])
        print("Answer:", result["answer"][:400])
        print("Citations:", len(result["cited_sources"]))
        print("Latency:", result["latency_seconds"])

        if result["execution_error"]:
            print("ERROR:", result["execution_error"])

        # Save incrementally so completed runs
        # are not lost if a later question fails.
        save_report(results)

    print("\n=== GENERATION SUMMARY ===")

    report = json.loads(
        OUTPUT_PATH.read_text(encoding="utf-8")
    )

    for status, count in (
        report["summary"]["status_counts"].items()
    ):
        print(f"{status}: {count}")

    print(
        "Mean latency:",
        report["summary"]["mean_latency_seconds"],
    )

    print("Report:", OUTPUT_PATH)

if __name__ == "__main__":
    main()