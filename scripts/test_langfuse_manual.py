from pharmaguard_rag.observability import OptionalLangfuse

def main():
    print("=== LANGFUSE CONNECTION TEST ===")

    observer = OptionalLangfuse()

    print("Langfuse enabled:", observer.enabled)

    if not observer.enabled:
        print(
            "[SKIP] Langfuse is disabled. "
            "RAG can still run without tracing."
        )
        return

    # Verify credentials and connectivity
    try:
        authenticated = observer.client.auth_check()
    except Exception as error:
        print("[WARNING] Authentication check failed:", error)
        return

    if not authenticated:
        print("[WARNING] Langfuse authentication failed.")
        return

    print("[PASS] Langfuse authentication successful")

    # Create parent trace
    with observer.observation(
        name="pharmaguard-observability-smoke",
        as_type="span",
        input={
            "question": "Apa itu farmakovigilans?"
        },
    ) as root:

        # Simulate retrieval
        with observer.observation(
            name="dense-retrieval",
            as_type="retriever",
            input={
                "question": "Apa itu farmakovigilans?",
                "top_k": 4,
            },
        ) as retrieval:

            observer.update(
                retrieval,
                output={
                    "retrieved_chunk_ids": [
                        "D3_p018_c001",
                        "D2_p002_c001",
                    ]
                },
            )

        # Simulate generation
        with observer.observation(
            name="ollama-generation",
            as_type="generation",
            model="qwen2.5:3b",
            input={
                "prompt": "Jelaskan farmakovigilans."
            },
        ) as generation:

            observer.update(
                generation,
                output=(
                    "Farmakovigilans berkaitan dengan "
                    "pemantauan keamanan obat."
                ),
            )

        observer.update(
            root,
            output={
                "status": "smoke_test_completed",
            },
        )

    observer.flush()

    print("[PASS] Test observations created")
    print(
        "Check the Langfuse Tracing dashboard "
        "to verify the trace was received."
    )

if __name__ == "__main__":
    main()