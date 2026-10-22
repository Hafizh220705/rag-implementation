import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pharmaguard_rag.embeddings import E5Embedder
from pharmaguard_rag.generation import OllamaGenerator
from pharmaguard_rag.pipeline import (
    DEFAULT_MINIMUM_RETRIEVAL_SIMILARITY,
    RAGPipeline,
    RAGPipelineError,
)
from pharmaguard_rag.retrieval_dense import DenseRetriever
from pharmaguard_rag.vector_store import ChromaVectorStore


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evaluation/datasets/pharmaguard_eval_v1.json"
CHUNKS_PATH = ROOT / "data/processed/chunks.jsonl"
INDEX_DIR = ROOT / "data/indexes/chroma"
DEFAULT_OUTPUT_PATH = (
    ROOT / "outputs/evaluations/generation_baseline_v3_final.json"
)

MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 4
SMOKE_IDS = ["F01", "F05", "F03", "I01"]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def save_report(results, output_path, metadata):
    statuses = Counter(item["status"] for item in results)
    latencies = [
        item["latency_seconds"]
        for item in results
        if item["latency_seconds"] is not None
    ]
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
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "baseline-v3",
        "retrieval_method": "dense_with_explicit_document_constraints",
        "answer_mode": "evidence_excerpts",
        "generation_options": {"num_ctx": 8192, "num_predict": 128,
                               "temperature": 0, "seed": 42, "repeat_penalty": 1.1},
        "max_generation_attempts": 2,
        "embedding_model": metadata["embedding_model"],
        "generation_model": metadata["generation_model"],
        "top_k": metadata["top_k"],
        "minimum_retrieval_similarity": metadata[
            "minimum_retrieval_similarity"
        ],
        "dataset_sha256": metadata["dataset_sha256"],
        "corpus_sha256": metadata["corpus_sha256"],
        "collection_name": metadata["collection_name"],
        "source_sha256": metadata["source_sha256"],
        "development_dataset_note": (
            "The same 24-question set was used during development; "
            "it is not an independent held-out benchmark."
        ),
        "summary": {
            "evaluated_cases": len(results),
            "evaluated_question_ids": [
                item["question_id"] for item in results
            ],
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
            "This development set is not a held-out benchmark.",
        ],
        "per_question": results,
    }

    atomic_write_json(output_path, report)


def select_cases(dataset, evaluate_all, selected_ids):
    if evaluate_all:
        return dataset

    requested_ids = set(selected_ids or SMOKE_IDS)
    selected = [
        case for case in dataset
        if case["question_id"] in requested_ids
    ]
    found_ids = {case["question_id"] for case in selected}
    missing = requested_ids - found_ids

    if missing:
        raise ValueError(f"Unknown question IDs: {sorted(missing)}")

    return selected


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
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output path. Relative paths are resolved from the "
            "repository root."
        ),
    )
    args = parser.parse_args()

    if args.all and args.ids:
        parser.error("Use either --all or --ids.")

    output_path = args.output or DEFAULT_OUTPUT_PATH
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    if output_path.exists():
        parser.error("Output exists. Use a new filename; saved experiments are immutable.")

    dataset = load_dataset()
    selected = select_cases(dataset, args.all, args.ids)

    print("=== GENERATION EVALUATION V3 ===")
    print("Selected questions:", len(selected))
    print("Output:", output_path)

    corpus_sha256 = file_sha256(CHUNKS_PATH)
    dataset_sha256 = file_sha256(DATASET_PATH)

    embedder = E5Embedder(model_name=MODEL_NAME, device="cpu")
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
        minimum_retrieval_similarity=(
            DEFAULT_MINIMUM_RETRIEVAL_SIMILARITY
        ),
    )

    metadata = {
        "embedding_model": embedder.model_name,
        "generation_model": generator.model_name,
        "top_k": TOP_K,
        "minimum_retrieval_similarity": (
            DEFAULT_MINIMUM_RETRIEVAL_SIMILARITY
        ),
        "dataset_sha256": dataset_sha256,
        "corpus_sha256": corpus_sha256,
        "collection_name": store.collection.name,
        "source_sha256": {str(path.relative_to(ROOT)): file_sha256(path)
                          for path in sorted((ROOT / "src/pharmaguard_rag").glob("*.py"))},
    }
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
                "initial_raw_answer": response[
                    "initial_raw_answer"
                ],
                "generation_attempts": response[
                    "generation_attempts"
                ],
                "generation_raw_attempts": response[
                    "generation_raw_attempts"
                ],
                "abstention_reason": response[
                    "abstention_reason"
                ],
                "cited_sources": response["cited_sources"],
                "retrieved_chunks": response["retrieved_chunks"],
                "retrieval_latency_seconds": response[
                    "retrieval_latency_seconds"
                ],
                "generation_latency_seconds": response[
                    "generation_latency_seconds"
                ],
                "latency_seconds": response["latency_seconds"],
                "error_stage": None,
                "error_type": None,
                "execution_error": None,
            }
            for key in ("required_document_ids", "missing_document_ids", "evidence_pages",
                        "evidence_candidates", "selected_evidence", "validation_errors",
                        "generation_diagnostics", "answer_mode"):
                result[key] = response[key]

        except RAGPipelineError as error:
            stage_latencies = [
                value for value in (
                    error.retrieval_latency_seconds,
                    error.generation_latency_seconds,
                )
                if value is not None
            ]
            result = {
                "question_id": question_id,
                "category": case["category"],
                "question": case["question"],
                "gold_verified": case["verified"],
                "should_abstain": case["should_abstain"],
                "status": "execution_error",
                "answer": "",
                "raw_answer": error.raw_answer,
                "initial_raw_answer": "",
                "generation_attempts": None,
                "generation_raw_attempts": [],
                "abstention_reason": None,
                "cited_sources": [],
                "retrieved_chunks": error.retrieved_chunks,
                "retrieval_latency_seconds": (
                    error.retrieval_latency_seconds
                ),
                "generation_latency_seconds": (
                    error.generation_latency_seconds
                ),
                "latency_seconds": (
                    round(sum(stage_latencies), 3)
                    if stage_latencies else None
                ),
                "error_stage": error.stage,
                "error_type": type(error.cause).__name__,
                "execution_error": str(error),
            }

        except Exception as error:
            result = {
                "question_id": question_id,
                "category": case["category"],
                "question": case["question"],
                "gold_verified": case["verified"],
                "should_abstain": case["should_abstain"],
                "status": "execution_error",
                "answer": "",
                "raw_answer": "",
                "initial_raw_answer": "",
                "generation_attempts": None,
                "generation_raw_attempts": [],
                "abstention_reason": None,
                "cited_sources": [],
                "retrieved_chunks": [],
                "retrieval_latency_seconds": None,
                "generation_latency_seconds": None,
                "latency_seconds": None,
                "error_stage": "unknown",
                "error_type": type(error).__name__,
                "execution_error": str(error),
            }

        results.append(result)
        print("Status:", result["status"])
        print("Answer:", result["answer"][:400])
        print("Citations:", len(result["cited_sources"]))
        print("Latency:", result["latency_seconds"])

        if result["execution_error"]:
            print(
                "ERROR:",
                result["error_stage"],
                result["error_type"],
                result["execution_error"],
            )

        # Atomic incremental saves preserve completed cases.
        save_report(results, output_path, metadata)

    report = json.loads(output_path.read_text(encoding="utf-8"))
    print("\n=== GENERATION SUMMARY ===")
    for status, count in report["summary"]["status_counts"].items():
        print(f"{status}: {count}")
    print("Mean latency:", report["summary"]["mean_latency_seconds"])
    print("Report:", output_path)
    store.close()


if __name__ == "__main__":
    main()
