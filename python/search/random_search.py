"""Random-search baseline using the evolutionary search's DSL tree generator."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any, Callable

import pandas as pd

from python.dsl.ast import ASTNode, BinaryOp, TimeSeriesOp, UnaryOp
from python.dsl.simplify import simplify
from python.dsl.validator import MAX_DEPTH, ValidationError, validate
from python.evaluation.fitness import FitnessEvaluator, FitnessResult
from python.search.direction import directional_fitness, signal_direction
from python.search.evolution import EvolutionSearch, train_score


@dataclass(frozen=True)
class RandomSearchConfig:
    evaluation_budget: int = 257
    seed: int = 17
    horizon: int = 20
    max_depth: int = 8
    seed_pool: str = "none"
    validation_top_k: int = 10
    progress_interval_seconds: float = 60.0
    min_coverage: float = 0.5

    def __post_init__(self):
        if self.evaluation_budget < 1:
            raise ValueError("evaluation_budget must be positive")
        if self.horizon not in {1, 5, 20, 30}:
            raise ValueError("horizon must be one of 1, 5, 20, or 30")
        if not 1 <= self.max_depth <= MAX_DEPTH:
            raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")
        if self.seed_pool not in {"none", "expanded"}:
            raise ValueError("seed_pool must be 'none' or 'expanded'")
        if self.validation_top_k < 0:
            raise ValueError("validation_top_k cannot be negative")
        if self.progress_interval_seconds <= 0:
            raise ValueError("progress_interval_seconds must be positive")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be between 0 and 1")


@dataclass
class RandomCandidate:
    expression: str
    score: float
    fitness: FitnessResult

    @property
    def direction(self) -> int:
        """Sign of train ICIR; later splits are evaluated as direction * formula."""
        return signal_direction(self.fitness.icir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "score": self.score,
            "direction": self.direction,
            "fitness": self.fitness.to_dict(),
        }


@dataclass
class RandomSearchResult:
    config: RandomSearchConfig
    best: list[RandomCandidate]
    validation: list[dict[str, Any]]
    unique_evaluations: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "random_search",
            "config": asdict(self.config),
            "objective": "absolute_train_icir",
            "best": [candidate.to_dict() for candidate in self.best],
            "validation": self.validation,
            "unique_train_evaluations": self.unique_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
        }


class RandomSearch:
    """Evaluate a reproducible sample of unique valid DSL expressions."""

    def __init__(
        self,
        research_data: pd.DataFrame,
        config: RandomSearchConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.config = config or RandomSearchConfig()
        self.progress_callback = progress_callback
        self.rng = random.Random(self.config.seed)
        self.evaluator = FitnessEvaluator(research_data)
        if "split" not in self.evaluator.data.columns:
            raise ValueError("Research data must include the signal 'split' column")
        target_split = f"target_split_{self.config.horizon}d"
        if target_split not in self.evaluator.data.columns:
            raise ValueError(f"Research data must include '{target_split}'")

        # Use the exact AST generation mechanism and operator/window constraints
        # used by evolutionary search, while keeping this run's RNG independent.
        self._generator = EvolutionSearch.__new__(EvolutionSearch)
        self._generator.rng = self.rng

    def run(self) -> RandomSearchResult:
        started = time.perf_counter()
        last_progress = started
        expressions: set[str] = set()
        candidates: list[RandomCandidate] = []
        attempts = 0
        max_attempts = self.config.evaluation_budget * 1000

        self._emit(
            "Starting random search: "
            f"budget={self.config.evaluation_budget}, horizon={self.config.horizon}d, "
            f"max_depth={self.config.max_depth}, seed={self.config.seed}, "
            f"seed_pool={self.config.seed_pool}; "
            "fitness selection uses absolute train ICIR."
        )
        if self.config.seed_pool == "expanded":
            for expression in EvolutionSearch.SEEDS:
                node = simplify(self._generator._validated(expression))
                if EvolutionSearch._depth(node) <= self.config.max_depth:
                    self._evaluate(node, expressions, candidates)
                    if len(candidates) == self.config.evaluation_budget:
                        break
        while len(candidates) < self.config.evaluation_budget and attempts < max_attempts:
            attempts += 1
            node = simplify(self._generator._random_tree(self.config.max_depth))
            try:
                validate(node)
            except ValidationError:
                continue
            expression = str(node)
            if expression in expressions:
                continue
            self._evaluate(node, expressions, candidates)

            now = time.perf_counter()
            if now - last_progress >= self.config.progress_interval_seconds:
                self._emit(
                    f"Progress: {len(candidates)}/{self.config.evaluation_budget} "
                    f"unique formulas evaluated; elapsed={now - started:.0f}s."
                )
                last_progress = now

        if len(candidates) != self.config.evaluation_budget:
            raise RuntimeError(
                "Could not generate enough unique valid formulas: "
                f"{len(candidates)}/{self.config.evaluation_budget} after {attempts} attempts"
            )

        candidates.sort(key=lambda candidate: (candidate.score, candidate.expression), reverse=True)
        finalists = candidates[: self.config.validation_top_k]
        validation = self._validate_finalists(finalists)
        elapsed = time.perf_counter() - started
        self._emit(
            f"Random search complete: {len(candidates)} unique train evaluations, "
            f"{sum('fitness' in item for item in validation)} validation evaluations, "
            f"elapsed={elapsed:.1f}s."
        )
        return RandomSearchResult(
            config=self.config,
            best=finalists,
            validation=validation,
            unique_evaluations=len(candidates),
            elapsed_seconds=elapsed,
        )

    def _evaluate(
        self,
        node: ASTNode,
        expressions: set[str],
        candidates: list[RandomCandidate],
    ) -> None:
        expression = str(node)
        if expression in expressions:
            return
        expressions.add(expression)
        try:
            fitness = self.evaluator.evaluate(
                node, horizon=self.config.horizon, split="train"
            )
            score = train_score(fitness, self.config.min_coverage)
        except (ValueError, FloatingPointError, OverflowError):
            fitness = FitnessResult(
                horizon=self.config.horizon,
                mean_ic=float("nan"),
                ic_std=float("nan"),
                icir=float("nan"),
                num_days=0,
                num_observations=0,
                coverage=0.0,
            )
            score = 0.0
        candidates.append(RandomCandidate(expression, score, fitness))

    def _validate_finalists(self, finalists: list[RandomCandidate]) -> list[dict[str, Any]]:
        if "validation" not in set(self.evaluator.data["split"]):
            return [
                {"expression": candidate.expression, "status": "validation_split_unavailable"}
                for candidate in finalists
            ]
        results = []
        for candidate in finalists:
            from python.dsl.parser import parse

            fitness = self.evaluator.evaluate(
                parse(candidate.expression), horizon=self.config.horizon, split="validation"
            )
            raw = fitness.to_dict()
            results.append({
                "expression": candidate.expression,
                "direction": candidate.direction,
                "fitness": raw,
                "directional": directional_fitness(raw, candidate.direction),
            })
        return results

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)


def operator_set(node: ASTNode) -> set[str]:
    """Return operators used by an expression for diversity summaries."""
    if isinstance(node, (UnaryOp, TimeSeriesOp)):
        return {node.operator} | operator_set(node.operand)
    if isinstance(node, BinaryOp):
        return {node.operator} | operator_set(node.left) | operator_set(node.right)
    return set()
