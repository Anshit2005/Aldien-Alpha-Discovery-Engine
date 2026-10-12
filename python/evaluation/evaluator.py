import numpy as np
import pandas as pd

from python.dsl.ast import (
    ASTNode,
    Variable,
    Number,
    UnaryOp,
    BinaryOp,
    TimeSeriesOp,
)


class AlphaEvaluationError(Exception):
    """Raised when an alpha cannot be evaluated."""
    pass


class AlphaEvaluator:
    """
    Evaluates a validated alpha AST on market data.

    Expected input DataFrame columns:

        datetime
        instrument
        open
        high
        low
        close
        volume

    Output:

        datetime | instrument | alpha
    """

    VARIABLE_MAP = {
        "OPEN": "open",
        "HIGH": "high",
        "LOW": "low",
        "CLOSE": "close",
        "VOLUME": "volume",
    }

    def __init__(self, data: pd.DataFrame):
        self.data = self._prepare_data(data)

    # =========================================================
    # Public API
    # =========================================================

    def evaluate(self, ast: ASTNode) -> pd.DataFrame:
        """
        Evaluate an AST and return a structured DataFrame.

        Returns:
            DataFrame with columns:

                datetime
                instrument
                alpha
        """

        result = self._evaluate_node(ast)

        if not isinstance(result, pd.Series):
            raise AlphaEvaluationError(
                "Alpha expression must produce a Series."
            )

        return pd.DataFrame({
            "datetime": self.data["datetime"].to_numpy(),
            "instrument": self.data["instrument"].to_numpy(),
            "alpha": result.to_numpy(dtype=float),
        })

    # =========================================================
    # Data Preparation
    # =========================================================

    def _prepare_data(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        required = {
            "datetime",
            "instrument",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = required - set(data.columns)

        if missing:
            raise AlphaEvaluationError(
                f"Missing required columns: {sorted(missing)}"
            )

        df = data.copy()

        df["datetime"] = pd.to_datetime(
            df["datetime"]
        )

        # Rolling operations must be performed
        # chronologically for each instrument.
        df = (
            df.sort_values(
                ["instrument", "datetime"]
            )
            .reset_index(drop=True)
        )

        return df

    # =========================================================
    # AST Evaluation
    # =========================================================

    def _evaluate_node(
        self,
        node: ASTNode,
    ) -> pd.Series:

        if isinstance(node, Variable):
            return self._evaluate_variable(node)

        if isinstance(node, Number):
            return self._evaluate_number(node)

        if isinstance(node, UnaryOp):
            return self._evaluate_unary(node)

        if isinstance(node, BinaryOp):
            return self._evaluate_binary(node)

        if isinstance(node, TimeSeriesOp):
            return self._evaluate_timeseries(node)

        raise AlphaEvaluationError(
            f"Unsupported AST node: {type(node).__name__}"
        )

    # =========================================================
    # Variables
    # =========================================================

    def _evaluate_variable(
        self,
        node: Variable,
    ) -> pd.Series:

        variable = node.name.upper()

        if variable not in self.VARIABLE_MAP:
            raise AlphaEvaluationError(
                f"Unsupported variable: {variable}"
            )

        column = self.VARIABLE_MAP[variable]

        return pd.Series(
            self.data[column].to_numpy(
                dtype=float
            ),
            index=self.data.index,
        )

    # =========================================================
    # Constants
    # =========================================================

    def _evaluate_number(
        self,
        node: Number,
    ) -> pd.Series:

        return pd.Series(
            float(node.value),
            index=self.data.index,
            dtype=float,
        )

    # =========================================================
    # Unary Operators
    # =========================================================

    def _evaluate_unary(
        self,
        node: UnaryOp,
    ) -> pd.Series:

        operand = self._evaluate_node(
            node.operand
        )

        operator = node.operator.upper()

        if operator == "ABS":
            return operand.abs()

        if operator == "LOG":

            # log(x) is undefined for x <= 0.
            result = operand.copy()

            result[result <= 0] = np.nan

            return np.log(result)

        if operator == "SIGN":
            return np.sign(operand)

        if operator == "RANK":
            return self._cross_sectional_rank(
                operand
            )

        raise AlphaEvaluationError(
            f"Unsupported unary operator: {operator}"
        )

    # =========================================================
    # Binary Operators
    # =========================================================

    def _evaluate_binary(
        self,
        node: BinaryOp,
    ) -> pd.Series:

        left = self._evaluate_node(
            node.left
        )

        right = self._evaluate_node(
            node.right
        )

        operator = node.operator.upper()

        if operator == "ADD":
            return left + right

        if operator == "SUB":
            return left - right

        if operator == "MUL":
            return left * right

        if operator == "DIV":
            return self._safe_divide(
                left,
                right,
            )

        raise AlphaEvaluationError(
            f"Unsupported binary operator: {operator}"
        )

    # =========================================================
    # Time-Series Operators
    # =========================================================

    def _evaluate_timeseries(
        self,
        node: TimeSeriesOp,
    ) -> pd.Series:

        operand = self._evaluate_node(
            node.operand
        )

        operator = node.operator.upper()

        window = int(node.window)

        temp = pd.DataFrame({
            "instrument": self.data[
                "instrument"
            ],
            "value": operand,
        })

        grouped = temp.groupby(
            "instrument",
            sort=False,
        )["value"]

        if operator == "TS_MEAN":

            result = grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=window,
                ).mean()
            )

        elif operator == "TS_STD":

            result = grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=window,
                ).std()
            )

        elif operator == "TS_MIN":

            result = grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=window,
                ).min()
            )

        elif operator == "TS_MAX":

            result = grouped.transform(
                lambda x: x.rolling(
                    window=window,
                    min_periods=window,
                ).max()
            )

        else:

            raise AlphaEvaluationError(
                f"Unsupported time-series operator: {operator}"
            )

        return pd.Series(
            result.to_numpy(
                dtype=float
            ),
            index=self.data.index,
        )

    # =========================================================
    # Safe Division
    # =========================================================

    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
    ) -> pd.Series:

        numerator = numerator.astype(float)

        denominator = denominator.astype(float)

        result = pd.Series(
            np.nan,
            index=numerator.index,
            dtype=float,
        )

        valid = (
            numerator.notna()
            & denominator.notna()
            & (denominator != 0)
        )

        result.loc[valid] = (
            numerator.loc[valid]
            / denominator.loc[valid]
        )

        return result

    # =========================================================
    # Cross-Sectional Rank
    # =========================================================

    def _cross_sectional_rank(
        self,
        values: pd.Series,
    ) -> pd.Series:

        temp = pd.DataFrame({
            "datetime": self.data[
                "datetime"
            ],
            "value": values,
        })

        # Rank stocks against other stocks
        # on the SAME date.
        ranks = (
            temp.groupby(
                "datetime",
                sort=False,
            )["value"]
            .rank(
                method="average",
                pct=True,
            )
        )

        return pd.Series(
            ranks.to_numpy(
                dtype=float
            ),
            index=self.data.index,
        )