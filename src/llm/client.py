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
            config.system_instruction = (
                system_instruction
            )

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

    def generate_json(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float = 0.1,
        max_output_tokens: int = 3000,
        max_retries: int = 3,
    ) -> dict:
        """
        Generate structured JSON from Gemini.

        Retries temporary server/service failures such as
        HTTP 503 before returning a final error.
        """

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
            config.system_instruction = (
                system_instruction
            )

        last_error = None

        for attempt in range(max_retries + 1):

            try:
                response = (
                    self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=config,
                    )
                )

                if not response.text:
                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                try:
                    result = json.loads(
                        response.text
                    )

                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        "Gemini returned invalid JSON."
                    ) from exc

                if not isinstance(result, dict):
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

                return result

            except Exception as exc:

                last_error = exc

                error_text = str(exc)

                is_temporary = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "high demand"
                    in error_text.lower()
                    or "temporarily"
                    in error_text.lower()
                )

                if not is_temporary:
                    raise

                if attempt >= max_retries:
                    break

                wait_seconds = 2 ** attempt

                print(
                    f"\nGemini temporarily unavailable. "
                    f"Retrying in {wait_seconds}s "
                    f"(attempt {attempt + 1}/"
                    f"{max_retries})..."
                )

                time.sleep(wait_seconds)

        raise RuntimeError(
            "Gemini remained unavailable after "
            f"{max_retries} retries."
        ) from last_error