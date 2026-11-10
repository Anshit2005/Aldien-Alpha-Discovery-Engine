"""
Fitness evaluation on the C++ engine, with the same interface as
python.evaluation.fitness.FitnessEvaluator.

The research panel is copied into C++ once, when the evaluator is created
(prices, the four forward returns, and the split columns as small integer
codes, with dates and instruments as integer ids). After that each call only
converts the formula and returns a FitnessResult, including the daily IC
series, so the search can use either evaluator.

    evaluator = CppFitnessEvaluator(research_data)
    result = evaluator.evaluate(parse("RANK(CLOSE)"), horizon=20, split="train")
    result.icir, result.daily_ic

Needs the compiled module engine/build/alpha_engine_cpp (see README).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from python.cpp_engine import alpha_engine_cpp
from python.dsl.ast import ASTNode
from python.evaluation.cpp_ast_bridge import CppASTConversionError, python_ast_to_cpp
from python.evaluation.fitness import (
    FitnessEvaluationError,
    FitnessEvaluator,
    FitnessResult,
)


class CppEvaluationError(FitnessEvaluationError, ValueError):
    """The C++ engine rejected a formula; a ValueError, as Python evaluation errors are."""


class CppFitnessEvaluator(FitnessEvaluator):
    """FitnessEvaluator whose alpha and fitness are computed in C++."""

    engine_name = "cpp"

    def __init__(self, research_data: pd.DataFrame):
        # Sorts by instrument, then datetime, exactly like the Python evaluator.
        super().__init__(research_data)
        data = self.data
        instruments = pd.factorize(data["instrument"], sort=True)[0].astype(np.int32)
        days, dates = pd.factorize(data["datetime"], sort=True)
        self.dates = pd.DatetimeIndex(dates, name="datetime")
        horizons = sorted(self.HORIZONS)
        if "split" in data.columns:
            split = self._split_codes(data["split"])
            target_split = [
                self._split_codes(data[f"target_split_{horizon}d"])
                if f"target_split_{horizon}d" in data.columns
                # No target column: no row's target is in any split.
                else np.full(len(data), -1, dtype=np.int8)
                for horizon in horizons
            ]
        else:
            split = np.empty(0, dtype=np.int8)
            target_split = []
        self.engine = alpha_engine_cpp.Engine(
            instrument=instruments,
            day=days.astype(np.int32),
            open=data["open"].to_numpy(dtype=float),
            high=data["high"].to_numpy(dtype=float),
            low=data["low"].to_numpy(dtype=float),
            close=data["close"].to_numpy(dtype=float),
            volume=data["volume"].to_numpy(dtype=float),
            returns=[data[self.HORIZONS[horizon]].to_numpy(dtype=float) for horizon in horizons],
            split=split,
            target_split=target_split,
        )

    @classmethod
    def _split_codes(cls, column: pd.Series) -> np.ndarray:
        return pd.Categorical(column, categories=cls.SPLITS).codes.astype(np.int8)

    def _cpp_ast(self, ast: ASTNode):
        try:
            return python_ast_to_cpp(ast)
        except CppASTConversionError as error:
            raise CppEvaluationError(str(error)) from error

    def evaluate_alpha(self, ast: ASTNode) -> np.ndarray:
        """Alpha value of every row, in the order of ``self.data``."""

        try:
            return self.engine.evaluate(self._cpp_ast(ast))
        except RuntimeError as error:
            raise CppEvaluationError(str(error)) from error

    def evaluate(
        self,
        ast: ASTNode,
        horizon: int = 20,
        split: str | None = None,
    ) -> FitnessResult:
        self._check_request(horizon, split)
        code = -1 if split is None else self.SPLITS.index(split)
        try:
            result = self.engine.fitness(self._cpp_ast(ast), horizon, code)
        except RuntimeError as error:
            raise CppEvaluationError(str(error)) from error
        daily_ic = pd.Series(
            result.daily_ic, index=self.dates[result.days], name="ic"
        )
        return FitnessResult(
            horizon=horizon,
            mean_ic=float(result.mean_ic),
            ic_std=float(result.ic_std),
            icir=float(result.icir),
            num_days=int(result.num_days),
            num_observations=int(result.num_observations),
            coverage=float(result.coverage),
            daily_ic=daily_ic,
        )
