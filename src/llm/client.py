from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class GeminiClient:
    """
    Thin wrapper around the Google GenAI SDK.

    The rest of the application communicates with Gemini
    through this class.
    """

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
    ):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add it to the project .env file."
            )

        self.client = genai.Client(
            api_key=api_key
        )

        self.model = model

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
        """
        Generate plain text from Gemini.
        """

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
        Generate structured JSON from Gemini.

        The response is requested using Gemini's JSON mode
        and validated against the expected structure.

        Temporary Gemini service failures and malformed/
        truncated JSON responses are retried.
        """

        # -----------------------------------------------------
        # RESPONSE SCHEMA
        # -----------------------------------------------------

        config = types.GenerateContentConfig(
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
            config.system_instruction = system_instruction

        last_error = None

        # -----------------------------------------------------
        # RETRY LOOP
        # -----------------------------------------------------

        for attempt in range(max_retries + 1):

            try:

                print(
                    f"      Gemini attempt "
                    f"{attempt + 1}/{max_retries + 1}..."
                )

                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                )

                # -------------------------------------------------
                # EMPTY RESPONSE
                # -------------------------------------------------

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                raw_text = response.text.strip()

                # -------------------------------------------------
                # PARSE JSON
                # -------------------------------------------------

                try:

                    result = json.loads(
                        raw_text
                    )

                except json.JSONDecodeError as exc:

                    print(
                        "\nGemini returned malformed JSON."
                    )

                    print(
                        f"JSON error: {exc}"
                    )

                    raise RuntimeError(
                        "Gemini returned invalid or "
                        "truncated JSON."
                    ) from exc

                # -------------------------------------------------
                # BASIC STRUCTURE VALIDATION
                # -------------------------------------------------

                if not isinstance(
                    result,
                    dict,
                ):
                    raise RuntimeError(
                        "Gemini JSON response is not "
                        "an object."
                    )

                if "section" not in result:
                    raise RuntimeError(
                        "Gemini response is missing "
                        "'section'."
                    )

                if "claims" not in result:
                    raise RuntimeError(
                        "Gemini response is missing "
                        "'claims'."
                    )

                if not isinstance(
                    result["claims"],
                    list,
                ):
                    raise RuntimeError(
                        "'claims' must be a list."
                    )

                # -------------------------------------------------
                # CLAIM VALIDATION
                # -------------------------------------------------

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
                            "is missing "
                            "'evidence_ids'."
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
                            "evidence_ids must "
                            "be a list."
                        )

                    # Evidence IDs should be strings.
                    for evidence_id in claim[
                        "evidence_ids"
                    ]:

                        if not isinstance(
                            evidence_id,
                            str,
                        ):
                            raise RuntimeError(
                                f"Claim {claim_index} "
                                "contains a non-string "
                                "evidence ID."
                            )

                # -------------------------------------------------
                # SUCCESS
                # -------------------------------------------------

                return result

            # -----------------------------------------------------
            # ERROR HANDLING
            # -----------------------------------------------------

            except Exception as exc:

                last_error = exc

                error_text = str(exc)

                is_temporary = (
                    "503" in error_text
                    or "UNAVAILABLE"
                    in error_text
                    or "high demand"
                    in error_text.lower()
                    or "temporarily"
                    in error_text.lower()
                    or "invalid or truncated JSON"
                    in error_text
                    or "malformed JSON"
                    in error_text
                )

                # Permanent error.
                if not is_temporary:
                    raise

                # No retries remaining.
                if attempt >= max_retries:
                    break

                wait_seconds = 2 ** attempt

                print(
                    "\nGemini generation failed "
                    "temporarily."
                )

                print(
                    f"Reason: {error_text}"
                )

                print(
                    f"Retrying in {wait_seconds}s "
                    f"(attempt {attempt + 1}/"
                    f"{max_retries})..."
                )

                time.sleep(
                    wait_seconds
                )

        # ---------------------------------------------------------
        # ALL RETRIES FAILED
        # ---------------------------------------------------------

        raise RuntimeError(
            "Gemini failed to produce valid "
            "structured JSON after "
            f"{max_retries} retries."
        ) from last_error