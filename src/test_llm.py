from llm.client import GeminiClient


def main():

    print("=" * 70)
    print("GEMINI CONNECTION TEST")
    print("=" * 70)

    client = GeminiClient()

    response = client.generate(
        """
        You are testing an AI generation pipeline.

        Respond with exactly:

        GENAR LLM CONNECTION OK
        """
    )

    print("\nModel response:")
    print(response)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()