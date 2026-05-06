"""LLM client abstraction.

Protocol-based so the generator doesn't care which model is behind it.
Two implementations ship now:

  - OllamaClient: calls a local Ollama server (default Qwen 2.5 7B)
  - MockClient:   returns canned responses for tests

Future swap-ins (AnthropicClient, OpenAIClient, etc) implement the same
Protocol and plug in by config.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal interface: take system + user, return raw JSON string."""

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        ...


class OllamaClient:
    """Wraps the `ollama` Python SDK against a local Ollama server."""

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        host: str | None = None,
    ) -> None:
        self.model = model
        self.host = host

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        # Lazy import so tests without the package still run.
        import ollama

        client = ollama.Client(host=self.host) if self.host else ollama
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class GroqClient:
    """Wraps Groq's hosted chat completions API.

    Groq runs models on LPU (Language Processing Unit) hardware which is
    significantly faster than GPU-based inference. Free tier has generous
    rate limits suitable for development and small-scale production.

    Requires GROQ_API_KEY environment variable (or pass api_key explicitly).
    Sign up at https://console.groq.com to get a key.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        api_key: str | None = None,
    ) -> None:
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required for GroqClient. "
                "Set the environment variable or pass api_key explicitly."
            )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        # Lazy import so tests without the package still run.
        from groq import Groq

        client = Groq(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class GeminiClient:
    """Wraps Google's Gemini API.

    Uses the new `google-genai` SDK. Strong multilingual capabilities —
    particularly notable on specialized concepts where Llama / Qwen / Aya
    fall short. JSON mode supported via response_mime_type.

    Requires GEMINI_API_KEY environment variable (or pass api_key).
    Get a key at https://aistudio.google.com/app/apikey.

    Recommended models:
      - gemini-2.5-pro     : best quality, slightly slower
      - gemini-2.5-flash   : faster, cheaper, still very capable
      - gemini-2.0-flash   : older, fastest, OK quality
    """

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        api_key: str | None = None,
    ) -> None:
        import os
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for GeminiClient. "
                "Set the environment variable or pass api_key explicitly."
            )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        # Lazy import so tests without the package still run.
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        return response.text


class MockClient:
    """Test double. Returns a canned response; records every call for assertion."""

    def __init__(self, canned_response: str) -> None:
        self.canned_response = canned_response
        self.calls: list[dict] = []

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        self.calls.append(
            {"system": system, "user": user, "temperature": temperature}
        )
        return self.canned_response
