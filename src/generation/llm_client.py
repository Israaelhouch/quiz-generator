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
