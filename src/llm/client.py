from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class GeminiClient:
    """
    Thin wrapper around the Google GenAI SDK.

    The application communicates with Gemini through
    this class instead of calling the SDK throughout
    the codebase.
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
        Generate a text response from Gemini.

        No function calling or external tools are enabled.
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