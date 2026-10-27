"""Persistent cache of LLM answers, keyed by the exact request.

The key is ``sha256(backend, model, system, user)``. Because evolution is
seeded and prompts are built deterministically, re-running a seed sends the
same prompts in the same order, so every answer comes from the cache: the
rerun is free, needs no network and reproduces the original run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def request_key(backend: str, model: str, system: str, user: str) -> str:
    payload = json.dumps([backend, model, system, user], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    """JSON-lines file of ``{"key", "response"}`` records; in memory if no path."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else None
        self._entries: dict[str, str] = {}
        if self.path is not None and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    self._entries[record["key"]] = record["response"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # A partially written last line must not break later runs.
                    continue

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def put(self, key: str, response: str) -> None:
        if key in self._entries:
            return
        self._entries[key] = response
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"key": key, "response": response}, ensure_ascii=False) + "\n")
