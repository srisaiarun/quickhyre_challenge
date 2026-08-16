from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from groq import Groq


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CLIENT
# ============================================================

class GeminiClient:
    """
    LLM client used by the GENAR application.

    Provider strategy:

        1. Gemini is the primary provider.
        2. Gemini failures are retried when appropriate.
        3. Groq is automatically used as the fallback provider.

    Both providers must return:

    {
        "section": "...",
        "claims": [
            {
                "text": "...",
                "evidence_ids": ["EV-..."]
            }
        ]
    }

    Important:

    Evidence grounding and numeric/date validation are NOT
    performed here.

    Those remain the responsibility of the existing GENAR
    section generators and validators.
    """

    # ========================================================
    # CONSTANTS
    # ========================================================

    DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
    DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

    # Groq on-demand GPT-OSS TPM budget.
    GROQ_TPM_LIMIT = 8000

    # Leave enough room for tokenizer estimation errors,
    # provider overhead and reasoning.
    GROQ_SAFETY_MARGIN = 350

    # We deliberately keep the completion reasonably small.
    # GENAR claims are concise and don't need thousands of tokens.
    GROQ_MAX_COMPLETION = 700

    # Minimum useful completion size.
    GROQ_MIN_COMPLETION = 180

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        groq_model: str = DEFAULT_GROQ_MODEL,
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

        self.client = genai.Client(
            api_key=gemini_api_key
        )

        self.model = model

        self.groq_client = Groq(
            api_key=groq_api_key
        )

        self.groq_model = groq_model

    # ========================================================
    # PLAIN TEXT GENERATION
    # ========================================================

    def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.2,
        max_output_tokens: int = 2000,
    ) -> str:
        """
        Generate plain text.

        Provider order:

            Gemini -> Groq
        """

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
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as gemini_error:

            print(
                "\nGemini plain-text generation failed."
            )

            print(
                f"Reason: {gemini_error}"
            )

            print(
                "      Falling back to Groq..."
            )

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

            try:
                response = (
                    self.groq_client
                    .chat
                    .completions
                    .create(
                        model=self.groq_model,
                        messages=messages,
                        temperature=min(
                            max(temperature, 0.0),
                            1.0,
                        ),
                        max_completion_tokens=max_output_tokens,
                        reasoning_effort="low",
                        reasoning_format="hidden",
                    )
                )

                if not response.choices:
                    raise RuntimeError(
                        "Groq returned no choices."
                    )

                text = (
                    response.choices[0]
                    .message
                    .content
                )

                if not text:
                    raise RuntimeError(
                        "Groq returned an empty response."
                    )

                return text.strip()

            except Exception as groq_error:

                raise RuntimeError(
                    "Both Gemini and Groq failed "
                    "to generate plain text.\n"
                    f"Gemini error: {gemini_error}\n"
                    f"Groq error: {groq_error}"
                ) from groq_error

    # ========================================================
    # STRUCTURED JSON GENERATION
    # ========================================================

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 6000,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """
        Generate structured GENAR JSON.

        Provider strategy:

            Gemini
              ↓
            retry
              ↓
            Groq fallback
              ↓
            JSON validation

        Important:

        The Groq fallback automatically compacts redundant
        whitespace and removes the separate system instruction
        when the TPM budget is tight.

        The actual evidence content is preserved.
        """

        # ----------------------------------------------------
        # GEMINI JSON SCHEMA
        # ----------------------------------------------------

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
            gemini_config.system_instruction = (
                system_instruction
            )

        last_gemini_error: Exception | None = None

        # ----------------------------------------------------
        # GEMINI RETRIES
        # ----------------------------------------------------

        for attempt in range(max_retries + 1):

            try:

                print(
                    f"      Gemini attempt "
                    f"{attempt + 1}/"
                    f"{max_retries + 1}..."
                )

                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=gemini_config,
                    )
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                result = (
                    self._parse_and_validate_json(
                        response.text.strip(),
                        provider="Gemini",
                    )
                )

                print(
                    "      Gemini generation: SUCCESS"
                )

                return result

            except Exception as exc:

                last_gemini_error = exc

                error_text = str(exc)

                # ------------------------------------------------
                # STOP EARLY FOR NON-RETRYABLE ERRORS
                # ------------------------------------------------

                if not self._is_retryable_error(
                    error_text
                ):

                    print(
                        "\nGemini returned a "
                        "non-retryable error."
                    )

                    print(
                        f"Reason: {error_text}"
                    )

                    break

                # ------------------------------------------------
                # FINAL ATTEMPT
                # ------------------------------------------------

                if attempt >= max_retries:

                    print(
                        "\nGemini retries exhausted."
                    )

                    break

                # ------------------------------------------------
                # EXPONENTIAL BACKOFF
                # ------------------------------------------------

                wait_seconds = min(
                    2 ** attempt,
                    8,
                )

                print(
                    "\nGemini generation failed "
                    "temporarily."
                )

                print(
                    f"Reason: {error_text}"
                )

                print(
                    f"Retrying in "
                    f"{wait_seconds}s "
                    f"(attempt "
                    f"{attempt + 1}/"
                    f"{max_retries})..."
                )

                time.sleep(
                    wait_seconds
                )

        # ----------------------------------------------------
        # GROQ FALLBACK
        # ----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "GEMINI FAILED — USING GROQ FALLBACK"
        )

        print(
            "=" * 60
        )

        if last_gemini_error is not None:

            print(
                "Gemini final error: "
                f"{last_gemini_error}"
            )

        return self._generate_json_with_groq(
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    # ========================================================
    # GROQ JSON GENERATION
    # ========================================================

    def _generate_json_with_groq(
        self,
        prompt: str,
        system_instruction: str | None,
        temperature: float,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        """
        Generate structured JSON using Groq.

        This method is deliberately designed around the
        8K TPM constraint of the current Groq tier.

        Strategy:

            1. Try compact system + original prompt.
            2. If TPM is tight, remove system instruction.
            3. Compact whitespace inside the prompt.
            4. Calculate safe completion budget.
            5. Send strict JSON schema request.
        """

        # ----------------------------------------------------
        # COMPACT FALLBACK SYSTEM
        # ----------------------------------------------------

        fallback_system = (
            "Return only the GENAR JSON object. "
            "Use only supplied evidence and instructions. "
            "Do not invent facts, numbers, dates, percentages, "
            "causality, safety signals, or expectedness. "
            "Every factual claim must cite supplied evidence IDs."
        )

        if system_instruction:

            fallback_system += (
                "\n\nSection instructions:\n"
                + system_instruction
            )

        # ----------------------------------------------------
        # FIRST MESSAGE VERSION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CALCULATE TOKEN ESTIMATE
        # ----------------------------------------------------

        estimated_input_tokens = (
            self._estimate_message_tokens(
                messages
            )
        )

        available_completion = (
            self.GROQ_TPM_LIMIT
            - self.GROQ_SAFETY_MARGIN
            - estimated_input_tokens
        )

        # ----------------------------------------------------
        # IF PROMPT IS LARGE:
        #
        # Remove the redundant system instruction.
        #
        # The section prompt already contains all important
        # grounding instructions.
        # ----------------------------------------------------

        if available_completion < (
            self.GROQ_MIN_COMPLETION + 50
        ):

            print(
                "      Groq prompt is large; "
                "removing redundant system "
                "instruction..."
            )

            compact_user_prompt = (
                self._compact_prompt(
                    prompt
                )
            )

            messages = [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY the required JSON "
                        "object. Use only supplied evidence. "
                        "Do not invent or calculate facts. "
                        "Every factual claim must include "
                        "evidence IDs.\n\n"
                        + compact_user_prompt
                    ),
                }
            ]

            estimated_input_tokens = (
                self._estimate_message_tokens(
                    messages
                )
            )

            available_completion = (
                self.GROQ_TPM_LIMIT
                - self.GROQ_SAFETY_MARGIN
                - estimated_input_tokens
            )

        # ----------------------------------------------------
        # SECOND COMPACTION PASS
        # ----------------------------------------------------

        if available_completion < (
            self.GROQ_MIN_COMPLETION + 50
        ):

            print(
                "      Groq prompt still large; "
                "using maximum whitespace "
                "compaction..."
            )

            compact_prompt = (
                self._aggressively_compact_prompt(
                    prompt
                )
            )

            messages = [
                {
                    "role": "user",
                    "content": (
                        "Return ONLY valid JSON.\n"
                        "Use ONLY supplied evidence.\n"
                        "Every factual claim needs evidence IDs.\n\n"
                        + compact_prompt
                    ),
                }
            ]

            estimated_input_tokens = (
                self._estimate_message_tokens(
                    messages
                )
            )

            available_completion = (
                self.GROQ_TPM_LIMIT
                - self.GROQ_SAFETY_MARGIN
                - estimated_input_tokens
            )

        # ----------------------------------------------------
        # FINAL TOKEN BUDGET
        # ----------------------------------------------------

        groq_max_tokens = min(
            max_output_tokens,
            self.GROQ_MAX_COMPLETION,
            max(
                0,
                available_completion,
            ),
        )

        # ----------------------------------------------------
        # ABSOLUTE FAILURE
        # ----------------------------------------------------

        if groq_max_tokens < self.GROQ_MIN_COMPLETION:

            raise RuntimeError(
                "Groq fallback prompt remains too large "
                "for the available TPM budget.\n"
                f"Estimated input tokens: "
                f"{estimated_input_tokens:.0f}\n"
                f"Available completion tokens: "
                f"{max(0, available_completion)}\n"
                "The section prompt/evidence context "
                "must be reduced."
            )

        # ----------------------------------------------------
        # LOGGING
        # ----------------------------------------------------

        print(
            f"      Groq model: "
            f"{self.groq_model}"
        )

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

        # ----------------------------------------------------
        # STRICT JSON SCHEMA
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # GROQ REQUEST
        # ----------------------------------------------------

        try:

            response = (
                self.groq_client
                .chat
                .completions
                .create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=min(
                        max(
                            temperature,
                            0.0,
                        ),
                        1.0,
                    ),
                    max_completion_tokens=(
                        groq_max_tokens
                    ),
                    reasoning_effort="low",
                    reasoning_format="hidden",
                    response_format=response_format,
                )
            )

            if not response.choices:

                raise RuntimeError(
                    "Groq returned no choices."
                )

            message = (
                response.choices[0]
                .message
            )

            raw_text = message.content

            if not raw_text:

                raise RuntimeError(
                    "Groq returned an empty response."
                )

            result = (
                self._parse_and_validate_json(
                    raw_text.strip(),
                    provider="Groq",
                )
            )

            print(
                "      Groq fallback generation: SUCCESS"
            )

            return result

        except Exception as exc:

            raise RuntimeError(
                "Both Gemini and Groq failed "
                "to produce valid structured JSON.\n"
                f"Groq error: {exc}"
            ) from exc

    # ========================================================
    # TOKEN ESTIMATION
    # ========================================================

    @staticmethod
    def _estimate_message_tokens(
        messages: list[dict[str, str]]
    ) -> int:
        """
        Conservative approximate token estimator.

        We intentionally use ~3.5 characters/token rather than
        a smaller ratio so that we do not accidentally exceed
        the Groq TPM limit.
        """

        total_chars = 0

        for message in messages:

            content = message.get(
                "content",
                "",
            )

            total_chars += len(content)

            # Small allowance for role/message overhead.
            total_chars += 20

        return int(
            total_chars / 3.5
        )

    # ========================================================
    # PROMPT COMPACTION
    # ========================================================

    @staticmethod
    def _compact_prompt(
        prompt: str
    ) -> str:
        """
        Moderate whitespace compaction.

        This preserves all textual content while removing
        unnecessary blank lines and indentation.
        """

        text = prompt.replace(
            "\r\n",
            "\n",
        )

        # Remove trailing spaces.
        text = re.sub(
            r"[ \t]+\n",
            "\n",
            text,
        )

        # Collapse 3+ newlines to 2.
        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text,
        )

        # Remove indentation from JSON-like evidence.
        text = re.sub(
            r"\n[ \t]+",
            "\n",
            text,
        )

        return text.strip()

    @staticmethod
    def _aggressively_compact_prompt(
        prompt: str
    ) -> str:
        """
        Aggressive but content-preserving whitespace
        compaction.

        This does NOT remove evidence or factual content.

        It only removes formatting whitespace.
        """

        text = prompt.replace(
            "\r\n",
            "\n",
        )

        # Remove leading/trailing whitespace per line.
        lines = [
            line.strip()
            for line in text.split("\n")
        ]

        # Remove empty lines.
        lines = [
            line
            for line in lines
            if line
        ]

        text = "\n".join(lines)

        # Collapse repeated spaces.
        text = re.sub(
            r"[ \t]{2,}",
            " ",
            text,
        )

        return text.strip()

    # ========================================================
    # JSON PARSING + STRUCTURE VALIDATION
    # ========================================================

    @staticmethod
    def _parse_and_validate_json(
        raw_text: str,
        provider: str,
    ) -> dict[str, Any]:
        """
        Parse JSON and validate the basic GENAR structure.

        This does NOT perform evidence validation.

        The downstream GENAR validator remains responsible for:

            - evidence grounding
            - numeric validation
            - date validation
            - section-specific rules
        """

        try:

            result = json.loads(
                raw_text
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                f"{provider} returned malformed JSON. "
                f"JSON error: {exc}"
            ) from exc

        # ----------------------------------------------------
        # ROOT OBJECT
        # ----------------------------------------------------

        if not isinstance(
            result,
            dict,
        ):

            raise RuntimeError(
                f"{provider} JSON response "
                "is not an object."
            )

        # ----------------------------------------------------
        # SECTION
        # ----------------------------------------------------

        if "section" not in result:

            raise RuntimeError(
                f"{provider} response is "
                "missing 'section'."
            )

        if not isinstance(
            result["section"],
            str,
        ):

            raise RuntimeError(
                f"{provider} 'section' "
                "must be a string."
            )

        # ----------------------------------------------------
        # CLAIMS
        # ----------------------------------------------------

        if "claims" not in result:

            raise RuntimeError(
                f"{provider} response is "
                "missing 'claims'."
            )

        if not isinstance(
            result["claims"],
            list,
        ):

            raise RuntimeError(
                f"{provider} 'claims' "
                "must be a list."
            )

        # ----------------------------------------------------
        # CLAIM VALIDATION
        # ----------------------------------------------------

        for claim_index, claim in enumerate(
            result["claims"],
            start=1,
        ):

            if not isinstance(
                claim,
                dict,
            ):

                raise RuntimeError(
                    f"Claim {claim_index} "
                    "is not an object."
                )

            if "text" not in claim:

                raise RuntimeError(
                    f"Claim {claim_index} "
                    "is missing 'text'."
                )

            if "evidence_ids" not in claim:

                raise RuntimeError(
                    f"Claim {claim_index} "
                    "is missing 'evidence_ids'."
                )

            if not isinstance(
                claim["text"],
                str,
            ):

                raise RuntimeError(
                    f"Claim {claim_index} "
                    "text must be a string."
                )

            if not isinstance(
                claim["evidence_ids"],
                list,
            ):

                raise RuntimeError(
                    f"Claim {claim_index} "
                    "evidence_ids must be "
                    "a list."
                )

            for evidence_id in (
                claim["evidence_ids"]
            ):

                if not isinstance(
                    evidence_id,
                    str,
                ):

                    raise RuntimeError(
                        f"Claim {claim_index} "
                        "contains a non-string "
                        "evidence ID."
                    )

        return result

    # ========================================================
    # ERROR CLASSIFICATION
    # ========================================================

    @staticmethod
    def _is_retryable_error(
        error_text: str,
    ) -> bool:
        """
        Determine whether a Gemini error should be retried.
        """

        text = error_text.lower()

        retryable_markers = (
            # Rate/quota
            "429",
            "resource_exhausted",
            "quota exceeded",
            "rate limit",
            "too many requests",

            # Server errors
            "500",
            "502",
            "503",
            "504",
            "internal server error",
            "service unavailable",
            "unavailable",
            "high demand",

            # Temporary/network
            "temporarily",
            "temporary",
            "timeout",
            "timed out",
            "connection reset",
            "connection error",

            # Structured-output failures
            "malformed json",
            "invalid or truncated json",
            "empty response",
        )

        return any(
            marker in text
            for marker in retryable_markers
        )