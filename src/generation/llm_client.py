"""LLM client abstraction.

Protocol-based so the generator doesn't care which model is behind it.
Implementations shipped here:

  - OllamaClient: calls a local Ollama server (default Qwen 2.5 7B)
  - GroqClient:   Groq's hosted API
  - GeminiClient: Google Gemini (the production default)
  - MockClient:   returns canned responses for tests

Future swap-ins (AnthropicClient, OpenAIClient, etc) implement the same
Protocol and plug in by config.

Two operational properties every real client here shares:

  1. The underlying SDK client is built ONCE and reused. Constructing one
     per call re-does TLS setup and credential handling on every request.
  2. Every call carries a timeout (`LLM_TIMEOUT_SECONDS`, default 90).
     Without one, a hung upstream holds a FastAPI worker thread forever —
     and the generator's retry loop means up to three of them per request.

Timeout wiring is defensive: these SDKs move fast and the argument names
differ between versions, so a client that rejects the timeout argument is
built without it and logs a warning once, rather than failing every call.
"""

from __future__ import annotations

import logging
import os
from typing import Protocol


logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 90.0


def resolve_timeout(explicit: float | None = None) -> float:
    """Per-call LLM timeout in seconds. Env-overridable; 0 disables."""
    if explicit is not None:
        return explicit
    raw = os.environ.get("LLM_TIMEOUT_SECONDS")
    if raw is None:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "LLM_TIMEOUT_SECONDS=%r is not a number; using %.0fs",
            raw, DEFAULT_TIMEOUT_SECONDS,
        )
        return DEFAULT_TIMEOUT_SECONDS


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
        timeout_seconds: float | None = None,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout_seconds = resolve_timeout(timeout_seconds)
        self._client = None

    def _get_client(self):
        if self._client is None:
            import ollama

            if not self.host:
                self._client = ollama
            else:
                try:
                    self._client = ollama.Client(
                        host=self.host, timeout=self.timeout_seconds
                    )
                except TypeError:
                    logger.warning(
                        "Installed ollama SDK rejects a timeout argument; "
                        "requests to %s will not time out.", self.host,
                    )
                    self._client = ollama.Client(host=self.host)
        return self._client

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        response = self._get_client().chat(
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
        timeout_seconds: float | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is required for GroqClient. "
                "Set the environment variable or pass api_key explicitly."
            )
        self.timeout_seconds = resolve_timeout(timeout_seconds)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq

            try:
                self._client = Groq(
                    api_key=self.api_key, timeout=self.timeout_seconds
                )
            except TypeError:
                logger.warning(
                    "Installed groq SDK rejects a timeout argument; "
                    "requests will not time out."
                )
                self._client = Groq(api_key=self.api_key)
        return self._client

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        response = self._get_client().chat.completions.create(
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

    Uses the `google-genai` SDK. Strong multilingual capabilities —
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
        timeout_seconds: float | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for GeminiClient. "
                "Set the environment variable or pass api_key explicitly."
            )
        self.timeout_seconds = resolve_timeout(timeout_seconds)
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai

            # google-genai takes its timeout in MILLISECONDS, inside
            # HttpOptions. Both the class and the argument have moved between
            # releases, so fall back rather than break every request.
            if self.timeout_seconds > 0:
                try:
                    from google.genai import types

                    self._client = genai.Client(
                        api_key=self.api_key,
                        http_options=types.HttpOptions(
                            timeout=int(self.timeout_seconds * 1000)
                        ),
                    )
                    return self._client
                except (TypeError, AttributeError, ValueError) as exc:
                    logger.warning(
                        "Installed google-genai does not accept an HttpOptions "
                        "timeout (%s); requests will not time out.", exc,
                    )
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.75,
    ) -> str:
        from google.genai import types

        response = self._get_client().models.generate_content(
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
