"""
Choose the fitness engine: the Python reference or the C++ engine.

Both return the same FitnessResult (with the daily IC series); they agree to
rounding (about 1e-7 in ICIR), so a search gives the same formulas on either.

    evaluator = make_fitness_evaluator(data, "cpp")
"""

from __future__ import annotations

import pandas as pd

from python.evaluation.fitness import FitnessEvaluator

ENGINES = ("python", "cpp")


def cpp_engine_error() -> str | None:
    """Why the C++ engine cannot be used here, or None when it can."""

    try:
        import python.evaluation.cpp_fitness  # noqa: F401
    except ImportError as error:
        return (
            "the C++ engine is not built (engine/build/alpha_engine_cpp); "
            f"see README, 'Optional: build the C++ engine' ({error})"
        )
    return None


def make_fitness_evaluator(data: pd.DataFrame, engine: str = "python") -> FitnessEvaluator:
    if engine == "python":
        return FitnessEvaluator(data)
    if engine == "cpp":
        problem = cpp_engine_error()
        if problem is not None:
            raise RuntimeError(f"--engine cpp: {problem}")
        from python.evaluation.cpp_fitness import CppFitnessEvaluator

        return CppFitnessEvaluator(data)
    raise ValueError(f"engine must be one of {ENGINES}, not {engine!r}")


def add_engine_argument(parser) -> None:
    parser.add_argument(
        "--engine", choices=ENGINES, default="python",
        help="fitness engine: the Python reference (default) or the compiled C++ engine",
    )


def check_engine_argument(parser, args) -> None:
    """parser.error() early when --engine cpp is asked for but not built."""

    if getattr(args, "engine", "python") == "cpp":
        problem = cpp_engine_error()
        if problem is not None:
            parser.error(f"--engine cpp: {problem}")
