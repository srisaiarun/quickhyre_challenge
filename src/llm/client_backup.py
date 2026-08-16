from __future__ import annotations

import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq


load_dotenv()


class GeminiClient:
    """
    LLM client used by the GENAR application.

    Provider strategy:
      1. Gemini is the primary provider.
      2. Groq is the automatic fallback provider.

    Both providers return:
        {
          "section": "...",
          "claims": [
            {
              "text": "...",
              "evidence_ids": ["EV-..."]
            }
          ]
        }

    Evidence grounding and numeric validation remain outside
    this class and are performed by the existing validators.
    """

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        groq_model: str = "openai/gpt-oss-120b",
    ):
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add it to the project .env file."
            )

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not configured. "
                "Add it to the project .env file."
            )

        self.client = genai.Client(api_key=gemini_api_key)
        self.model = model

        self.groq_client = Groq(api_key=groq_api_key)
        self.groq_model = groq_model

    # =========================================================
    # PLAIN TEXT GENERATION
    # =========================================================

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2000,
    ) -> str:
        """Generate plain text with Gemini -> Groq fallback."""

        try:
            print("      LLM provider: Gemini")

            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            if system_instruction:
                config.system_instruction = system_instruction

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            if not response.text:
                raise RuntimeError("Gemini returned an empty response.")

            return response.text.strip()

        except Exception as gemini_error:
            print("\nGemini plain-text generation failed.")
            print(f"Reason: {gemini_error}")
            print("      Falling back to Groq...")

            messages: list[dict[str, str]] = []

            if system_instruction:
                messages.append(
                    {
                        "role": "system",
                        "content": system_instruction,
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": prompt,
                }
            )

            response = self.groq_client.chat.completions.create(
                model=self.groq_model,
                messages=messages,
                temperature=min(max(temperature, 0.0), 1.0),
                max_completion_tokens=max_output_tokens,
                reasoning_effort="low",
                reasoning_format="hidden",
            )

            if not response.choices:
                raise RuntimeError("Groq returned no choices.")

            text = response.choices[0].message.content

            if not text:
                raise RuntimeError("Groq returned an empty response.")

            return text.strip()

    # =========================================================
    # STRUCTURED JSON GENERATION
    # =========================================================

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 6000,
        max_retries: int = 3,
    ) -> dict:
        """
        Generate structured JSON.

        Gemini is attempted first. Temporary Gemini failures are
        retried. If Gemini remains unavailable, Groq is used.

        Final evidence/numeric validation remains in the section
        generators and evidence validator.
        """

        gemini_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_schema=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "section": types.Schema(
                        type=types.Type.STRING
                    ),
                    "claims": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "text": types.Schema(
                                    type=types.Type.STRING
                                ),
                                "evidence_ids": types.Schema(
                                    type=types.Type.ARRAY,
                                    items=types.Schema(
                                        type=types.Type.STRING
                                    ),
                                ),
                            },
                            required=[
                                "text",
                                "evidence_ids",
                            ],
                        ),
                    ),
                },
                required=[
                    "section",
                    "claims",
                ],
            ),
        )

        if system_instruction:
            gemini_config.system_instruction = system_instruction

        last_gemini_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                print(
                    f"      Gemini attempt "
                    f"{attempt + 1}/{max_retries + 1}..."
                )

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=gemini_config,
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                result = self._parse_and_validate_json(
                    response.text.strip(),
                    provider="Gemini",
                )

                print("      Gemini generation: SUCCESS")
                return result

            except Exception as exc:
                last_gemini_error = exc
                error_text = str(exc)

                if attempt >= max_retries:
                    print("\nGemini retries exhausted.")
                    break

                if not self._is_retryable_error(error_text):
                    print("\nGemini returned a non-retryable error.")
                    print(f"Reason: {error_text}")
                    break

                wait_seconds = min(2 ** attempt, 8)

                print("\nGemini generation failed temporarily.")
                print(f"Reason: {error_text}")
                print(
                    f"Retrying in {wait_seconds}s "
                    f"(attempt {attempt + 1}/{max_retries})..."
                )

                time.sleep(wait_seconds)

        print("\n" + "=" * 60)
        print("GEMINI FAILED — USING GROQ FALLBACK")
        print("=" * 60)

        if last_gemini_error is not None:
            print(f"Gemini final error: {last_gemini_error}")

        return self._generate_json_with_groq(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    # =========================================================
    # GROQ JSON GENERATION
    # =========================================================

    def _generate_json_with_groq(
        self,
        prompt: str,
        system_instruction: str | None,
        temperature: float,
        max_output_tokens: int,
    ) -> dict:
        """
        Generate structured JSON using Groq GPT-OSS.

        GPT-OSS 120B supports strict JSON Schema output. We use
        constrained JSON Schema rather than the older json_object
        mode because GENAR depends on deterministic downstream
        parsing and validation.

        The prompt is preserved intact because it contains the
        evidence required for grounding. The fallback system
        instruction is deliberately compact to conserve TPM.
        """

        fallback_system = (
            "Return only the GENAR JSON object. "
            "Use only supplied evidence and instructions. "
            "Do not invent facts, numbers, dates, percentages, "
            "causality, safety signals, or expectedness. "
            "Every factual claim must cite supplied evidence IDs. "
            "Keep claims concise and grounded."
        )

        if system_instruction:
            fallback_system += (
                "\n\nSection instructions:\n"
                + system_instruction
            )

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": fallback_system,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # -----------------------------------------------------
        # TPM budgeting
        # -----------------------------------------------------
        #
        # Groq's on-demand GPT-OSS tier can enforce an 8K TPM
        # limit. Reasoning tokens are part of completion usage.
        # We therefore leave a margin and use low reasoning.
        #
        # Approximation used only for request budgeting.
        # The full evidence prompt is never truncated.
        # -----------------------------------------------------

        input_chars = sum(
            len(message["content"])
            for message in messages
        )

        estimated_input_tokens = input_chars / 3.5

        GROQ_TPM_LIMIT = 8000
        SAFETY_MARGIN = 250

        available_completion = int(
            GROQ_TPM_LIMIT
            - SAFETY_MARGIN
            - estimated_input_tokens
        )

        # GPT-OSS can reason before producing JSON, so 576 tokens
        # is often too small. Prefer ~800 when the TPM budget allows.
        groq_max_tokens = min(
            max_output_tokens,
            900,
            available_completion,
        )

        if groq_max_tokens < 650:
            # If the full prompt leaves too little completion space,
            # retry with an ultra-compact system instruction. The
            # user prompt itself already contains the section rules.
            compact_messages = [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY the required JSON structure. "
                        "Use only the supplied evidence. "
                        "Do not invent or calculate facts/numbers. "
                        "Every factual claim must include evidence IDs.\n\n"
                        + prompt
                    ),
                }
            ]

            compact_chars = sum(
                len(message["content"])
                for message in compact_messages
            )

            estimated_input_tokens = compact_chars / 3.5

            available_completion = int(
                GROQ_TPM_LIMIT
                - SAFETY_MARGIN
                - estimated_input_tokens
            )

            messages = compact_messages

            groq_max_tokens = min(
                max_output_tokens,
                900,
                available_completion,
            )

        if groq_max_tokens < 650:
            raise RuntimeError(
                "Groq fallback prompt is too large for the "
                "available TPM budget. "
                f"Estimated input tokens: "
                f"{estimated_input_tokens:.0f}; "
                f"available completion tokens: "
                f"{groq_max_tokens}."
            )

        print(f"      Groq model: {self.groq_model}")
        print(
            f"      Groq estimated input tokens: "
            f"{estimated_input_tokens:.0f}"
        )
        print(
            f"      Groq max completion tokens: "
            f"{groq_max_tokens}"
        )
        print(
            f"      Groq estimated request tokens: "
            f"{estimated_input_tokens + groq_max_tokens:.0f}"
        )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "genar_report_section",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string"
                        },
                        "claims": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "text": {
                                        "type": "string"
                                    },
                                    "evidence_ids": {
                                        "type": "array",
                                        "items": {
                                            "type": "string"
                                        },
                                    },
                                },
                                "required": [
                                    "text",
                                    "evidence_ids",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": [
                        "section",
                        "claims",
                    ],
                    "additionalProperties": False,
                },
            },
        }

        try:
            response = (
                self.groq_client
                .chat
                .completions
                .create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=min(max(temperature, 0.0), 1.0),
                    max_completion_tokens=groq_max_tokens,
                    reasoning_effort="low",
                    reasoning_format="hidden",
                    response_format=response_format,
                )
            )

            if not response.choices:
                raise RuntimeError(
                    "Groq returned no choices."
                )

            message = response.choices[0].message
            raw_text = message.content

            if not raw_text:
                raise RuntimeError(
                    "Groq returned an empty response."
                )

            result = self._parse_and_validate_json(
                raw_text.strip(),
                provider="Groq",
            )

            print(
                "      Groq fallback generation: SUCCESS"
            )

            return result

        except Exception as exc:
            raise RuntimeError(
                "Both Gemini and Groq failed to produce "
                "valid structured JSON.\n"
                f"Groq error: {exc}"
            ) from exc

    # =========================================================
    # JSON PARSING + BASIC STRUCTURE VALIDATION
    # =========================================================

    @staticmethod
    def _parse_and_validate_json(
        raw_text: str,
        provider: str,
    ) -> dict:
        """Parse JSON and validate the application-level shape."""

        try:
            result = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{provider} returned malformed JSON. "
                f"JSON error: {exc}"
            ) from exc

        if not isinstance(result, dict):
            raise RuntimeError(
                f"{provider} JSON response is not an object."
            )

        if "section" not in result:
            raise RuntimeError(
                f"{provider} response is missing 'section'."
            )

        if not isinstance(result["section"], str):
            raise RuntimeError(
                f"{provider} 'section' must be a string."
            )

        if "claims" not in result:
            raise RuntimeError(
                f"{provider} response is missing 'claims'."
            )

        if not isinstance(result["claims"], list):
            raise RuntimeError(
                f"{provider} 'claims' must be a list."
            )

        for claim_index, claim in enumerate(
            result["claims"],
            start=1,
        ):
            if not isinstance(claim, dict):
                raise RuntimeError(
                    f"Claim {claim_index} is not an object."
                )

            if "text" not in claim:
                raise RuntimeError(
                    f"Claim {claim_index} is missing 'text'."
                )

            if "evidence_ids" not in claim:
                raise RuntimeError(
                    f"Claim {claim_index} is missing "
                    "'evidence_ids'."
                )

            if not isinstance(claim["text"], str):
                raise RuntimeError(
                    f"Claim {claim_index} text must be a string."
                )

            if not isinstance(claim["evidence_ids"], list):
                raise RuntimeError(
                    f"Claim {claim_index} evidence_ids "
                    "must be a list."
                )

            for evidence_id in claim["evidence_ids"]:
                if not isinstance(evidence_id, str):
                    raise RuntimeError(
                        f"Claim {claim_index} contains a "
                        "non-string evidence ID."
                    )

        return result

    # =========================================================
    # ERROR CLASSIFICATION
    # =========================================================

    @staticmethod
    def _is_retryable_error(
        error_text: str,
    ) -> bool:
        """Identify Gemini errors for which another attempt is reasonable."""

        text = error_text.lower()

        retryable_markers = (
            "429",
            "resource_exhausted",
            "quota exceeded",
            "rate limit",
            "too many requests",
            "500",
            "502",
            "503",
            "504",
            "unavailable",
            "high demand",
            "temporarily",
            "temporary",
            "timeout",
            "timed out",
            "malformed json",
            "invalid or truncated json",
            "empty response",
            "connection reset",
            "connection error",
        )

        return any(
            marker in text
            for marker in retryable_markers
        )