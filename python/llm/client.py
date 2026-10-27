"""LLM backends behind one small interface.

Every backend implements ``complete(system, user) -> str``. The search code
never talks to a provider directly, so switching between a local model, a
hosted model and the test mock is a command-line flag.

``OpenAICompatibleClient`` covers every provider that speaks the OpenAI chat
API: Ollama, Groq, OpenRouter, Together and vLLM. ``MockClient`` returns
canned or generated answers and is used by the tests and by offline runs.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Protocol, Sequence


class LLMError(RuntimeError):
    """Raised when a backend cannot return an answer."""


class LLMClient(Protocol):
    backend: str
    model: str

    def complete(self, system: str, user: str) -> str:
        ...


# Known OpenAI-compatible endpoints and the environment variable holding
# their key. Ollama runs locally and ignores the key.
PROVIDERS = {
    "ollama": ("http://localhost:11434/v1", None),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
}

DEFAULT_MODELS = {
    "mock": "mock",
    "ollama": "qwen2.5:7b",
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "meta-llama/llama-3.3-70b-instruct",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
}

BACKENDS = ("mock", *PROVIDERS, "openai-compatible")


class OpenAICompatibleClient:
    """Chat-completions client for any OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        backend: str = "openai-compatible",
        temperature: float = 0.0,
        seed: int | None = 0,
        timeout_seconds: float = 60.0,
        max_tokens: int = 800,
    ):
        try:
            from openai import OpenAI
        except ImportError as error:  # pragma: no cover - depends on environment
            raise LLMError("The 'openai' package is required: pip install openai") from error

        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        # Local servers accept any key, but the client library requires one.
        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout_seconds,
            max_retries=2,
        )

    def complete(self, system: str, user: str) -> str:
        options = {}
        if self.seed is not None:
            options["seed"] = self.seed
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **options,
            )
        except Exception as error:
            raise LLMError(f"{self.backend} request failed: {error}") from error
        content = response.choices[0].message.content if response.choices else None
        if not content:
            raise LLMError(f"{self.backend} returned an empty answer")
        return content


class MockClient:
    """
    Offline backend.

    ``responses`` may be a list of strings returned in turn (cycling), or a
    function of ``(system, user)``. With neither, the mock wraps the parent
    formula from the prompt in a simple operator, which is enough to run the
    whole pipeline without a network.
    """

    backend = "mock"

    def __init__(
        self,
        responses: Sequence[str] | Callable[[str, str], str] | None = None,
        model: str = "mock",
    ):
        self.model = model
        self._responses = responses
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if callable(self._responses):
            return self._responses(system, user)
        if self._responses:
            response = self._responses[(self.calls - 1) % len(self._responses)]
            if isinstance(response, Exception):
                raise response
            return response
        return _default_mock_answer(user)


def _default_mock_answer(user: str) -> str:
    from python.dsl.json_io import dumps
    from python.dsl.parser import parse
    from python.dsl.ast import TimeSeriesOp, UnaryOp

    from .prompt import parent_expression_from_prompt

    parent = parse(parent_expression_from_prompt(user))
    # Deterministic choice of edit from the prompt text.
    choice = int(hashlib.sha256(user.encode("utf-8")).hexdigest(), 16) % 3
    if choice == 0:
        proposal = UnaryOp("RANK", parent)
    elif choice == 1:
        proposal = TimeSeriesOp("TS_MEAN", parent, 5)
    else:
        proposal = TimeSeriesOp("TS_MEAN", parent, 20)
    return dumps(proposal)


def load_env_file(path: Path | str = ".env") -> None:
    """Set ``KEY=value`` lines from a local .env file without overriding the shell."""

    path = Path(path)
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def make_client(
    backend: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> LLMClient:
    """Build a client from command-line style settings."""

    if backend not in BACKENDS:
        raise ValueError(f"Unknown LLM backend {backend!r}; choose from {', '.join(BACKENDS)}")
    if backend == "mock":
        return MockClient(model=model or "mock")

    load_env_file()
    if backend == "openai-compatible":
        if not base_url or not model:
            raise ValueError("openai-compatible needs --llm-base-url and --llm-model")
        return OpenAICompatibleClient(
            base_url, model, api_key or os.environ.get("LLM_API_KEY"), backend=backend
        )

    default_url, key_variable = PROVIDERS[backend]
    key = api_key or (os.environ.get(key_variable) if key_variable else None)
    if key_variable and not key:
        raise ValueError(f"{backend} needs an API key: set {key_variable} in the environment or .env")
    return OpenAICompatibleClient(
        base_url or default_url,
        model or DEFAULT_MODELS[backend],
        key,
        backend=backend,
    )
