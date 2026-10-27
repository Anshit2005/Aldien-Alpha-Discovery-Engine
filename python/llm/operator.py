"""LLM proposal operator: ask a model for a variation of a parent formula.

One call:

    parent -> similar library factors + recent failures -> prompt
           -> cache (hit: stored answer) or client (miss: new answer, stored)
           -> strip fences / prose -> loads() -> validate() -> depth -> simplify
           -> ASTNode, or None after logging why it was rejected

``propose`` never raises: any failure returns ``None`` so evolution falls back
to its random mutation. Only formulas that pass the same validator as random
mutations can enter the population.
"""

from __future__ import annotations

from collections import Counter, deque
import json
from pathlib import Path
from typing import Any

from python.dsl.ast import ASTNode
from python.dsl.json_io import loads
from python.dsl.simplify import simplify
from python.dsl.validator import MAX_DEPTH, ValidationError, validate

from .cache import ResponseCache, request_key
from .client import LLMClient
from .prompt import system_prompt, user_prompt
from .retrieval import FactorRetriever


class ProposalRejected(ValueError):
    def __init__(self, reason: str, detail: str):
        super().__init__(detail)
        self.reason = reason


def extract_json(response: str) -> str:
    """Return the outermost ``{...}`` of a reply, ignoring fences and prose."""

    start = response.find("{")
    end = response.rfind("}")
    if start < 0 or end < start:
        raise ProposalRejected("no_json", "reply contains no JSON object")
    return response[start : end + 1]


class LLMProposalOperator:
    def __init__(
        self,
        client: LLMClient,
        cache: ResponseCache | None = None,
        retriever: FactorRetriever | None = None,
        max_depth: int = MAX_DEPTH,
        horizon: int = 20,
        reference_count: int = 4,
        failures_in_prompt: int = 3,
        failures_path: Path | str | None = None,
    ):
        self.client = client
        self.cache = cache if cache is not None else ResponseCache()
        self.retriever = retriever if retriever is not None else FactorRetriever()
        self.max_depth = max_depth
        self.horizon = horizon
        self.reference_count = reference_count
        self.failures_path = Path(failures_path) if failures_path is not None else None
        self._system = system_prompt(max_depth)
        self._recent_failures: deque[dict[str, str]] = deque(maxlen=failures_in_prompt)
        self.calls = 0
        self.cache_hits = 0
        self.accepted = 0
        self.rejected: Counter[str] = Counter()

    def propose(self, parent: ASTNode, parent_fitness: Any | None = None) -> ASTNode | None:
        self.calls += 1
        user = user_prompt(
            parent,
            parent_fitness,
            self.retriever.similar(parent, self.reference_count),
            list(self._recent_failures),
            horizon=self.horizon,
        )
        key = request_key(self.client.backend, self.client.model, self._system, user)
        response = self.cache.get(key)
        try:
            if response is not None:
                self.cache_hits += 1
            else:
                try:
                    response = self.client.complete(self._system, user)
                except Exception as error:
                    # Not cached: a later call may succeed.
                    raise ProposalRejected("client_error", str(error)) from error
                self.cache.put(key, response)
            proposal = self._parse(response, parent)
        except ProposalRejected as rejection:
            self._record_failure(key, response or "", rejection)
            return None
        self.accepted += 1
        return proposal

    def _parse(self, response: str, parent: ASTNode) -> ASTNode:
        try:
            node = loads(extract_json(response))
        except ProposalRejected:
            raise
        except ValueError as error:
            reason = "invalid_json" if str(error).startswith("Invalid JSON") else "invalid_formula"
            raise ProposalRejected(reason, str(error)) from error
        try:
            validate(node)
        except ValidationError as error:
            reason = "too_deep" if "depth" in str(error).lower() else "invalid_formula"
            raise ProposalRejected(reason, str(error)) from error
        from python.search.evolution import EvolutionSearch

        depth = EvolutionSearch._depth(node)
        if depth > self.max_depth:
            raise ProposalRejected("too_deep", f"depth {depth} > {self.max_depth}")
        node = simplify(node)
        if node == simplify(parent):
            raise ProposalRejected("same_as_parent", "proposal is the parent formula")
        return node

    def _record_failure(self, key: str, response: str, rejection: ProposalRejected) -> None:
        self.rejected[rejection.reason] += 1
        error = f"{rejection.reason}: {rejection}"
        if rejection.reason != "client_error":
            # Network problems say nothing about the model's formulas.
            self._recent_failures.append({"response": response, "error": error})
        if self.failures_path is not None:
            self.failures_path.parent.mkdir(parents=True, exist_ok=True)
            with self.failures_path.open("a", encoding="utf-8") as file:
                record = {"prompt_hash": key, "response": response, "error": error}
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def stats(self) -> dict[str, Any]:
        rejected = sum(self.rejected.values())
        return {
            "backend": self.client.backend,
            "model": self.client.model,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "accepted": self.accepted,
            "rejected": rejected,
            "rejected_by_reason": dict(sorted(self.rejected.items())),
            "acceptance_rate": self.accepted / self.calls if self.calls else None,
        }
