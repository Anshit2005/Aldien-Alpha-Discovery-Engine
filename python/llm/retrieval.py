"""Find known factors similar to a parent formula.

Similarity is the Jaccard overlap of the operators and variables two formulas
use. The library is small (a few dozen entries), so this is exact, instant
and easy to explain; no embedding model or vector database is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from python.dsl.ast import ASTNode, BinaryOp, TimeSeriesOp, UnaryOp, Variable
from python.dsl.parser import parse
from python.dsl.simplify import simplify


LIBRARY_PATH = Path(__file__).with_name("library.json")


@dataclass(frozen=True)
class LibraryFactor:
    name: str
    expression: str
    meaning: str


def load_library(path: Path | str = LIBRARY_PATH) -> list[LibraryFactor]:
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    return [LibraryFactor(entry["name"], entry["expression"], entry["meaning"]) for entry in entries]


def formula_tokens(node: ASTNode) -> set[str]:
    """Operators and variables used by a formula."""

    if isinstance(node, Variable):
        return {node.name}
    if isinstance(node, (UnaryOp, TimeSeriesOp)):
        return {node.operator} | formula_tokens(node.operand)
    if isinstance(node, BinaryOp):
        return {node.operator} | formula_tokens(node.left) | formula_tokens(node.right)
    return set()


class FactorRetriever:
    def __init__(self, library: list[LibraryFactor] | None = None):
        self.library = library if library is not None else load_library()
        self._parsed = [simplify(parse(factor.expression)) for factor in self.library]
        self._tokens = [formula_tokens(node) for node in self._parsed]

    def similar(self, parent: ASTNode, k: int = 4) -> list[LibraryFactor]:
        """The ``k`` most similar library factors, excluding the parent itself."""

        canonical = simplify(parent)
        tokens = formula_tokens(canonical)
        scored = []
        for factor, node, factor_tokens in zip(self.library, self._parsed, self._tokens):
            if node == canonical:
                continue
            union = tokens | factor_tokens
            similarity = len(tokens & factor_tokens) / len(union) if union else 0.0
            scored.append((-similarity, factor.name, factor))
        scored.sort(key=lambda item: item[:2])
        return [factor for _, _, factor in scored[:k]]
