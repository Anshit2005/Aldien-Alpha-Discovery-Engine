import pandas as pd

from python.cpp_engine import alpha_engine_cpp
from python.evaluation.cpp_ast_bridge import (
    python_ast_to_cpp,
)
from python.dsl.parser import parse
from python.dsl.validator import validate
from python.evaluation.fitness import FitnessResult


class CppAlphaEngine:
    """
    Python interface to the C++ alpha execution engine.
    """

    REQUIRED_COLUMNS = {
        "datetime",
        "instrument",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    RETURN_COLUMNS = {
        1: "forward_return_1d",
        5: "forward_return_5d",
        20: "forward_return_20d",
        30: "forward_return_30d",
    }

    def __init__(self, data: pd.DataFrame):
        self.data = self._prepare_data(data)

    # ========================================================
    # DATA PREPARATION
    # ========================================================

    @staticmethod
    def _prepare_data(
        data: pd.DataFrame
    ) -> pd.DataFrame:

        missing = (
            CppAlphaEngine.REQUIRED_COLUMNS
            - set(data.columns)
        )

        if missing:
            raise ValueError(
                f"Missing required columns: "
                f"{sorted(missing)}"
            )

        df = data.copy()

        df["datetime"] = pd.to_datetime(
            df["datetime"]
        )

        # Important:
        # C++ time-series operators assume that
        # each instrument's rows are chronological.
        df = (
            df.sort_values(
                ["instrument", "datetime"]
            )
            .reset_index(drop=True)
        )

        return df

    # ========================================================
    # PYTHON → C++ DATA CONVERSION
    # ========================================================

    def _to_cpp_data(self, source_data=None):

        data = alpha_engine_cpp.MarketData()
        source_data = self.data if source_data is None else source_data

        data.datetime = (
            source_data["datetime"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        data.instrument = (
            source_data["instrument"]
            .astype(str)
            .tolist()
        )

        data.open = (
            source_data["open"]
            .tolist()
        )

        data.high = (
            source_data["high"]
            .tolist()
        )

        data.low = (
            source_data["low"]
            .tolist()
        )

        data.close = (
            source_data["close"]
            .tolist()
        )

        data.volume = (
            source_data["volume"]
            .tolist()
        )

        # Forward returns are optional for
        # normal alpha evaluation, but are
        # required for fitness evaluation.

        for _, column in self.RETURN_COLUMNS.items():

            if column in source_data.columns:

                setattr(
                    data,
                    column,
                    source_data[column].tolist()
                )

        return data

    # ========================================================
    # LOW-LEVEL AST API
    # ========================================================

    def evaluate_ast(self, ast):
        """
        Evaluate an already-parsed Python DSL AST
        using the C++ engine.
        """

        cpp_ast = (
            ast if isinstance(ast, alpha_engine_cpp.ASTNode)
            else python_ast_to_cpp(ast)
        )

        cpp_data = self._to_cpp_data()

        return alpha_engine_cpp.evaluate(
            cpp_data,
            cpp_ast
        )

    def evaluate(self, ast):
        """Backward-compatible alias for evaluating a Python or C++ AST."""
        if isinstance(ast, str):
            ast = parse(ast)
            validate(ast)
        return self.evaluate_ast(ast)

    # --------------------------------------------------------

    def evaluate_fitness_ast(
        self,
        ast,
        horizon=20,
        split=None,
    ):
        """
        Evaluate an already-parsed Python DSL AST
        using the C++ fitness engine.
        """

        if horizon not in self.RETURN_COLUMNS:

            raise ValueError(
                f"Unsupported horizon: {horizon}. "
                f"Supported horizons: "
                f"{sorted(self.RETURN_COLUMNS)}"
            )

        column = self.RETURN_COLUMNS[horizon]

        if column not in self.data.columns:

            raise ValueError(
                f"Missing required return column: "
                f"{column}"
            )

        source_data = self.data
        split_row_count = None
        if split is not None:
            if split not in {"train", "validation", "test", "holdout"}:
                raise ValueError(
                    "Unsupported split: "
                    f"{split}. Supported splits: "
                    "['holdout', 'test', 'train', 'validation']"
                )
            target_split_column = f"target_split_{horizon}d"
            missing = {"split", target_split_column} - set(self.data.columns)
            if missing:
                raise ValueError(
                    "Split-aware evaluation requires columns: "
                    f"{sorted(missing)}"
                )
            source_data = self.data.copy()
            split_rows = source_data["split"].eq(split)
            split_row_count = int(split_rows.sum())
            target_matches = source_data[target_split_column].eq(split)
            source_data.loc[
                ~(split_rows & target_matches), column
            ] = float("nan")

        cpp_ast = (
            ast if isinstance(ast, alpha_engine_cpp.ASTNode)
            else python_ast_to_cpp(ast)
        )

        cpp_data = self._to_cpp_data(source_data)

        result = alpha_engine_cpp.evaluate_fitness(
            cpp_data,
            cpp_ast,
            horizon
        )
        if split_row_count is not None and split_row_count > 0:
            return FitnessResult(
                horizon=horizon,
                mean_ic=result.mean_ic,
                ic_std=result.ic_std,
                icir=result.icir,
                num_days=result.num_days,
                num_observations=result.num_observations,
                coverage=result.num_observations / split_row_count,
            )
        return result

    # ========================================================
    # HIGH-LEVEL EXPRESSION API
    # ========================================================

    def evaluate_expression(
        self,
        expression: str
    ):
        """
        Parse, validate and evaluate a DSL
        expression using the C++ engine.

        Pipeline:

            Expression
                ↓
              Parser
                ↓
             Validator
                ↓
            Python AST
                ↓
            C++ AST
                ↓
            C++ Engine
        """

        # 1. Convert expression → Python AST
        ast = parse(expression)

        # 2. Check semantic/research constraints
        validate(ast)

        # 3. Send AST to C++
        return self.evaluate_ast(ast)

    # --------------------------------------------------------

    def evaluate_fitness_expression(
        self,
        expression: str,
        horizon=20,
        split=None,
    ):
        """
        Parse, validate and calculate alpha
        fitness using the C++ engine.

        Pipeline:

            Expression
                ↓
              Parser
                ↓
             Validator
                ↓
            Python AST
                ↓
            C++ AST
                ↓
          C++ Fitness Engine
                ↓
         FitnessResult
        """

        # 1. Parse expression
        ast = parse(expression)

        # 2. Validate expression
        validate(ast)

        # 3. Evaluate fitness using C++
        return self.evaluate_fitness_ast(
            ast,
            horizon,
            split
        )
