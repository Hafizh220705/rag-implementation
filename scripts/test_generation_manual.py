from pharmaguard_rag.generation import OllamaGenerator

def main():
    generator = OllamaGenerator()

    print("=== OLLAMA GENERATION TEST ===")
    print(f"Model: {generator.model_name}")

    system_prompt = (
        "Anda adalah asisten AI berbahasa Indonesia. "
        "Ikuti instruksi pengguna secara singkat."
    )

    user_prompt = (
        "Jawab dengan tepat satu kata: SIAP"
    )

    response = generator.generate(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    print("\nResponse:")
    print(response)

    print("\n[SUCCESS] Ollama API responded")

if __name__ == "__main__":
    main()