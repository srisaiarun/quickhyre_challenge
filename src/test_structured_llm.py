from llm.client import GeminiClient


def main():

    print("=" * 70)
    print("STRUCTURED GEMINI TEST")
    print("=" * 70)

    client = GeminiClient()

    prompt = """
You are testing a grounded pharmacovigilance generation system.

The supplied evidence is:

[EV-CASE-001]
metric: total_cases
value: 1024

[EV-CASE-002]
metric: serious_cases
value: 1023

Write one concise factual claim using ONLY these
two evidence items.

The claim must preserve the exact numbers.
"""

    result = client.generate_json(
        prompt=prompt,
        system_instruction="""
You are a strict evidence-grounded
pharmacovigilance reporting assistant.

Use only the supplied evidence.

Every factual claim must cite the relevant
evidence IDs.

Do not invent information.
""",
    )

    print("\nStructured response:")
    print(result)

    print("\n" + "=" * 70)

    print("\nSection:")
    print(result["section"])

    print("\nClaims:")

    for claim in result["claims"]:
        print(
            f"Text: {claim['text']}"
        )
        print(
            f"Evidence IDs: "
            f"{claim['evidence_ids']}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()