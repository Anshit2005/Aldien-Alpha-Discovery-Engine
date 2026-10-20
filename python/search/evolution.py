"""Deterministic, single-process evolutionary search over the alpha DSL."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import time
from typing import Any
from typing import Callable

import numpy as np
import pandas as pd

from python.dsl.ast import ASTNode, BinaryOp, Number, TimeSeriesOp, UnaryOp, Variable
from python.dsl.validator import (
    ALLOWED_BINARY_OPERATORS,
    ALLOWED_TIME_SERIES_OPERATORS,
    ALLOWED_UNARY_OPERATORS,
    ALLOWED_VARIABLES,
    MAX_DEPTH,
    MAX_WINDOW,
    ValidationError,
    validate,
)
from python.evaluation.fitness import FitnessEvaluator, FitnessResult


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 32
    generations: int = 10
    elite_count: int = 4
    elitism_enabled: bool = True
    tournament_size: int = 3
    mutation_rate: float = 0.7
    crossover_rate: float = 0.3
    max_depth: int = MAX_DEPTH
    seed: int = 17
    horizon: int = 20
    search_split: str = "train"
    seed_pool: str = "expanded"
    validation_top_k: int = 10
    progress_interval_seconds: float = 60.0

    def __post_init__(self):
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 0:
            raise ValueError("generations cannot be negative")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be in [1, population_size)")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be positive")
        if self.mutation_rate < 0 or self.crossover_rate < 0:
            raise ValueError("mutation_rate and crossover_rate cannot be negative")
        if self.mutation_rate + self.crossover_rate <= 0:
            raise ValueError("At least one genetic operator must be enabled")
        if not 1 <= self.max_depth <= MAX_DEPTH:
            raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}")
        if self.search_split != "train":
            raise ValueError("Evolutionary search must use the train split")
        if self.seed_pool not in {"legacy", "expanded"}:
            raise ValueError("seed_pool must be 'legacy' or 'expanded'")
        if self.validation_top_k < 0:
            raise ValueError("validation_top_k cannot be negative")
        if self.progress_interval_seconds <= 0:
            raise ValueError("progress_interval_seconds must be positive")


@dataclass
class ScoredCandidate:
    expression: str
    score: float
    fitness: FitnessResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "score": self.score,
            "fitness": self.fitness.to_dict(),
        }


@dataclass
class EvolutionResult:
    config: EvolutionConfig
    best: list[ScoredCandidate]
    validation: list[dict[str, Any]]
    generations: list[dict[str, Any]]
    unique_evaluations: int
    cache_hits: int
    validation_evaluations: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "objective": "absolute_train_icir",
            "best": [candidate.to_dict() for candidate in self.best],
            "validation": self.validation,
            "generations": self.generations,
            "unique_train_evaluations": self.unique_evaluations,
            "cache_hits": self.cache_hits,
            "validation_evaluations": self.validation_evaluations,
            "elapsed_seconds": self.elapsed_seconds,
        }


class EvolutionSearch:
    """Evolve symbolic expressions while selecting exclusively on train ICIR."""

    LEGACY_SEEDS = (
        "CLOSE",
        "RANK(CLOSE)",
        "SUB(CLOSE,TS_MEAN(CLOSE,5))",
        "SUB(CLOSE,TS_MEAN(CLOSE,20))",
        "DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20))",
        "RANK(DIV(SUB(CLOSE,TS_MEAN(CLOSE,20)),TS_STD(CLOSE,20)))",
        "DIV(SUB(HIGH,LOW),CLOSE)",
        "RANK(DIV(VOLUME,TS_MEAN(VOLUME,20)))",
    )
    # Expanded domain-informed pool. The original eight formulas above remain
    # byte-for-byte unchanged; additions span normalized price, volatility,
    # intraday movement, volume activity, price-volume interaction, and range.
    SEEDS = (
        *LEGACY_SEEDS,
        "DIV(CLOSE,TS_MEAN(CLOSE,20))",
        "DIV(SUB(CLOSE,TS_MEAN(CLOSE,60)),TS_MEAN(CLOSE,60))",
        "DIV(TS_STD(CLOSE,20),TS_MEAN(CLOSE,20))",
        "DIV(SUB(CLOSE,OPEN),ADD(SUB(HIGH,LOW),0.001))",
        "DIV(SUB(VOLUME,TS_MEAN(VOLUME,20)),TS_MEAN(VOLUME,20))",
        "MUL(SIGN(SUB(CLOSE,OPEN)),DIV(VOLUME,TS_MEAN(VOLUME,20)))",
        "DIV(SUB(CLOSE,TS_MIN(LOW,20)),SUB(TS_MAX(HIGH,20),TS_MIN(LOW,20)))",
        "DIV(SUB(CLOSE,TS_MAX(CLOSE,20)),TS_MAX(CLOSE,20))",
    )

    WINDOWS = (2, 5, 10, 20, 40, 60, 120, 252)

    def __init__(
        self,
        research_data: pd.DataFrame,
        config: EvolutionConfig | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.config = config or EvolutionConfig()
        self.progress_callback = progress_callback
        self.rng = random.Random(self.config.seed)
        self.evaluator = FitnessEvaluator(research_data)
        if "split" not in self.evaluator.data.columns:
            raise ValueError("Research data must include the signal 'split' column")
        target_split = f"target_split_{self.config.horizon}d"
        if target_split not in self.evaluator.data.columns:
            raise ValueError(f"Research data must include '{target_split}'")
        self._fitness_cache: dict[str, ScoredCandidate] = {}
        self._cache_hits = 0
        self._started_at = 0.0
        self._last_progress_at = 0.0
        self._generation = 0
        self._generation_candidate_count = 0
        self._generation_candidates_scored = 0

    def run(self) -> EvolutionResult:
        started = time.perf_counter()
        self._started_at = started
        self._last_progress_at = started
        self._emit(
            "Starting evolutionary search: "
            f"population={self.config.population_size}, "
            f"generations={self.config.generations}, "
            f"horizon={self.config.horizon}d, seed={self.config.seed}; "
            "fitness selection uses absolute train ICIR."
        )
        population = self._initial_population()
        history: list[dict[str, Any]] = []

        for generation in range(self.config.generations + 1):
            self._generation = generation + 1
            self._generation_candidate_count = len(set(map(str, population)))
            self._generation_candidates_scored = 0
            self._emit(
                f"Generation {generation + 1}/{self.config.generations + 1}: "
                f"evaluating {self._generation_candidate_count} candidates."
            )
            scored = self._score_population(population)
            summary = self._generation_summary(generation, scored)
            history.append(summary)
            self._emit(
                f"Generation {generation + 1}/{self.config.generations + 1} "
                f"complete: best |train ICIR|={summary['best_train_score']:.4f}, "
                f"mean score={summary['mean_train_score']:.4f}, "
                f"unique train evaluations={len(self._fitness_cache)}, "
                f"elapsed={self._elapsed():.0f}s."
            )
            if generation == self.config.generations:
                break
            population = self._next_generation(scored)

        ranked = sorted(
            self._fitness_cache.values(), key=self._sort_key, reverse=True
        )
        best = ranked[: self.config.validation_top_k]
        self._emit(
            f"Train search complete. Evaluating {len(best)} top candidates "
            "on validation; validation does not affect selection."
        )
        validation = self._validate_finalists(best)
        self._emit(
            f"Search complete: {len(self._fitness_cache)} unique train evaluations, "
            f"{sum('fitness' in item for item in validation)} validation evaluations, "
            f"elapsed={time.perf_counter() - started:.1f}s."
        )
        return EvolutionResult(
            config=self.config,
            best=best,
            validation=validation,
            generations=history,
            unique_evaluations=len(self._fitness_cache),
            cache_hits=self._cache_hits,
            validation_evaluations=sum("fitness" in item for item in validation),
            elapsed_seconds=time.perf_counter() - started,
        )

    def _initial_population(self) -> list[ASTNode]:
        population: list[ASTNode] = []
        seen: set[str] = set()
        seeds = self.LEGACY_SEEDS if self.config.seed_pool == "legacy" else self.SEEDS
        for expression in seeds:
            try:
                node = self._validated(expression)
            except (ValueError, ValidationError):
                continue
            if self._depth(node) <= self.config.max_depth:
                population.append(node)
                seen.add(str(node))
            if len(population) == self.config.population_size:
                return population

        attempts = 0
        while len(population) < self.config.population_size and attempts < 5000:
            attempts += 1
            node = self._random_tree(self.config.max_depth)
            expression = str(node)
            if expression not in seen:
                population.append(node)
                seen.add(expression)
        if len(population) < self.config.population_size:
            raise RuntimeError("Unable to create a unique initial population")
        return population

    @staticmethod
    def _validated(expression: str) -> ASTNode:
        from python.dsl.parser import parse

        node = parse(expression)
        validate(node)
        return node

    def _random_tree(self, depth_left: int) -> ASTNode:
        if depth_left <= 1 or self.rng.random() < 0.28:
            return self._random_terminal()
        kind = self.rng.choices(
            ("terminal", "unary", "binary", "timeseries"),
            weights=(0.12, 0.20, 0.42, 0.26),
            k=1,
        )[0]
        if kind == "terminal":
            return self._random_terminal()
        if kind == "unary":
            return UnaryOp(self.rng.choice(sorted(ALLOWED_UNARY_OPERATORS)), self._random_tree(depth_left - 1))
        if kind == "binary":
            return BinaryOp(
                self.rng.choice(sorted(ALLOWED_BINARY_OPERATORS)),
                self._random_tree(depth_left - 1),
                self._random_tree(depth_left - 1),
            )
        return TimeSeriesOp(
            self.rng.choice(sorted(ALLOWED_TIME_SERIES_OPERATORS)),
            self._random_tree(depth_left - 1),
            self.rng.choice(self.WINDOWS),
        )

    def _random_terminal(self) -> ASTNode:
        if self.rng.random() < 0.82:
            return Variable(self.rng.choice(sorted(ALLOWED_VARIABLES)))
        return Number(float(self.rng.choice((-1, 0, 1, 2))))

    def _score_population(self, population: list[ASTNode]) -> list[ScoredCandidate]:
        unique: dict[str, ASTNode] = {str(node): node for node in population}
        scored = []
        for node in unique.values():
            scored.append(self._score(node))
            self._generation_candidates_scored += 1
            now = time.perf_counter()
            if now - self._last_progress_at >= self.config.progress_interval_seconds:
                best_score = max(
                    (candidate.score for candidate in scored), default=0.0
                )
                self._emit(
                    f"Progress: generation {self._generation}, "
                    f"{self._generation_candidates_scored}/"
                    f"{self._generation_candidate_count} candidates scored; "
                    f"best score this generation={best_score:.4f}, "
                    f"unique train evaluations={len(self._fitness_cache)}, "
                    f"elapsed={self._elapsed():.0f}s."
                )
                self._last_progress_at = now
        return sorted(scored, key=self._sort_key, reverse=True)

    def _emit(self, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(message)

    def _elapsed(self) -> float:
        return time.perf_counter() - self._started_at

    def _score(self, node: ASTNode) -> ScoredCandidate:
        expression = str(node)
        cached = self._fitness_cache.get(expression)
        if cached is not None:
            self._cache_hits += 1
            return cached
        try:
            validate(node)
            fitness = self.evaluator.evaluate(
                node, horizon=self.config.horizon, split=self.config.search_split
            )
            score = abs(float(fitness.icir)) if np.isfinite(fitness.icir) else 0.0
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
        candidate = ScoredCandidate(expression, score, fitness)
        self._fitness_cache[expression] = candidate
        return candidate

    @staticmethod
    def _sort_key(candidate: ScoredCandidate) -> tuple[float, str]:
        return candidate.score, candidate.expression

    def _next_generation(self, scored: list[ScoredCandidate]) -> list[ASTNode]:
        elite_nodes = (
            [self._validated(item.expression) for item in scored[: self.config.elite_count]]
            if self.config.elitism_enabled
            else []
        )
        next_population = list(elite_nodes)
        seen = {str(node) for node in next_population}
        attempts = 0
        while len(next_population) < self.config.population_size and attempts < self.config.population_size * 100:
            attempts += 1
            parent = self._tournament(scored)
            child = self._validated(parent.expression)
            if self.rng.random() < self.config.crossover_rate:
                other = self._validated(self._tournament(scored).expression)
                child = self._crossover(child, other)
            if self.rng.random() < self.config.mutation_rate:
                child = self._mutate(child)
            try:
                validate(child)
            except ValidationError:
                continue
            if self._depth(child) > self.config.max_depth or str(child) in seen:
                continue
            next_population.append(child)
            seen.add(str(child))

        while len(next_population) < self.config.population_size:
            candidate = self._random_tree(self.config.max_depth)
            expression = str(candidate)
            if expression not in seen:
                next_population.append(candidate)
                seen.add(expression)
        return next_population

    def _tournament(self, scored: list[ScoredCandidate]) -> ScoredCandidate:
        contestants = self.rng.sample(scored, min(self.config.tournament_size, len(scored)))
        return max(contestants, key=self._sort_key)

    def _mutate(self, node: ASTNode) -> ASTNode:
        paths = self._paths(node)
        target = self.rng.choice(paths)
        subtree = self._subtree(node, target)
        operation = self.rng.choice(("replace", "wrap", "change"))
        if operation == "replace":
            replacement = self._random_tree(min(4, self.config.max_depth))
        elif operation == "wrap":
            wrapper = self.rng.choice(("unary", "binary", "timeseries"))
            if wrapper == "unary":
                replacement = UnaryOp(self.rng.choice(sorted(ALLOWED_UNARY_OPERATORS)), subtree)
            elif wrapper == "binary":
                replacement = BinaryOp(
                    self.rng.choice(sorted(ALLOWED_BINARY_OPERATORS)),
                    subtree,
                    self._random_terminal(),
                )
            else:
                replacement = TimeSeriesOp(
                    self.rng.choice(sorted(ALLOWED_TIME_SERIES_OPERATORS)),
                    subtree,
                    self.rng.choice(self.WINDOWS),
                )
        else:
            replacement = self._change_operator(subtree)
        return self._replace(node, target, replacement)

    def _change_operator(self, node: ASTNode) -> ASTNode:
        if isinstance(node, UnaryOp):
            return UnaryOp(self.rng.choice(sorted(ALLOWED_UNARY_OPERATORS - {node.operator})), node.operand)
        if isinstance(node, BinaryOp):
            return BinaryOp(self.rng.choice(sorted(ALLOWED_BINARY_OPERATORS - {node.operator})), node.left, node.right)
        if isinstance(node, TimeSeriesOp):
            if self.rng.random() < 0.5:
                operator = self.rng.choice(sorted(ALLOWED_TIME_SERIES_OPERATORS - {node.operator}))
                return TimeSeriesOp(operator, node.operand, node.window)
            return TimeSeriesOp(node.operator, node.operand, self.rng.choice(self.WINDOWS))
        return self._random_terminal()

    def _crossover(self, left: ASTNode, right: ASTNode) -> ASTNode:
        left_path = self.rng.choice(self._paths(left))
        right_path = self.rng.choice(self._paths(right))
        return self._replace(left, left_path, self._subtree(right, right_path))

    @staticmethod
    def _paths(node: ASTNode, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
        paths = [prefix]
        if isinstance(node, UnaryOp):
            paths.extend(EvolutionSearch._paths(node.operand, prefix + ("operand",)))
        elif isinstance(node, TimeSeriesOp):
            paths.extend(EvolutionSearch._paths(node.operand, prefix + ("operand",)))
        elif isinstance(node, BinaryOp):
            paths.extend(EvolutionSearch._paths(node.left, prefix + ("left",)))
            paths.extend(EvolutionSearch._paths(node.right, prefix + ("right",)))
        return paths

    @staticmethod
    def _subtree(node: ASTNode, path: tuple[str, ...]) -> ASTNode:
        for part in path:
            node = getattr(node, part)
        return node

    @staticmethod
    def _replace(node: ASTNode, path: tuple[str, ...], replacement: ASTNode) -> ASTNode:
        if not path:
            return replacement
        part, *remaining = path
        tail = tuple(remaining)
        if isinstance(node, UnaryOp):
            return UnaryOp(node.operator, EvolutionSearch._replace(node.operand, tail, replacement))
        if isinstance(node, TimeSeriesOp):
            return TimeSeriesOp(node.operator, EvolutionSearch._replace(node.operand, tail, replacement), node.window)
        if isinstance(node, BinaryOp):
            if part == "left":
                return BinaryOp(node.operator, EvolutionSearch._replace(node.left, tail, replacement), node.right)
            return BinaryOp(node.operator, node.left, EvolutionSearch._replace(node.right, tail, replacement))
        raise ValueError("Cannot replace a child of a leaf node")

    @staticmethod
    def _depth(node: ASTNode) -> int:
        if isinstance(node, (Variable, Number)):
            return 1
        if isinstance(node, UnaryOp):
            return 1 + EvolutionSearch._depth(node.operand)
        if isinstance(node, TimeSeriesOp):
            return 1 + EvolutionSearch._depth(node.operand)
        if isinstance(node, BinaryOp):
            return 1 + max(EvolutionSearch._depth(node.left), EvolutionSearch._depth(node.right))
        raise TypeError(f"Unknown AST node: {type(node).__name__}")

    def _generation_summary(self, generation: int, scored: list[ScoredCandidate]) -> dict[str, Any]:
        return {
            "generation": generation,
            "population_size": len(scored),
            "best_train_score": scored[0].score if scored else 0.0,
            "best_train_icir": scored[0].fitness.icir if scored else float("nan"),
            "mean_train_score": float(np.mean([item.score for item in scored])) if scored else 0.0,
            "unique_expressions": len({item.expression for item in scored}),
        }

    def _validate_finalists(self, finalists: list[ScoredCandidate]) -> list[dict[str, Any]]:
        if not finalists:
            return []
        if "validation" not in set(self.evaluator.data["split"]):
            return [{"expression": item.expression, "status": "validation_split_unavailable"} for item in finalists]
        results = []
        for candidate in finalists:
            ast = self._validated(candidate.expression)
            fitness = self.evaluator.evaluate(ast, horizon=self.config.horizon, split="validation")
            results.append({"expression": candidate.expression, "fitness": fitness.to_dict()})
        return results
