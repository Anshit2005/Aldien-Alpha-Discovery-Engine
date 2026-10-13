import numpy as np
import pandas as pd

from dataclasses import dataclass

from python.dsl.ast import ASTNode
from python.evaluation.evaluator import AlphaEvaluator


class FitnessEvaluationError(Exception):
    """Raised when alpha fitness cannot be calculated."""
    pass


@dataclass
class FitnessResult:
    """
    Summary of an alpha's predictive performance.
    """

    horizon: int
    mean_ic: float
    ic_std: float
    icir: float
    num_days: int
    num_observations: int
    coverage: float

    def to_dict(self):
        return {
            "horizon": self.horizon,
            "mean_ic": self.mean_ic,
            "ic_std": self.ic_std,
            "icir": self.icir,
            "num_days": self.num_days,
            "num_observations": self.num_observations,
            "coverage": self.coverage,
        }


class FitnessEvaluator:
    """
    Calculates quantitative fitness metrics for
    symbolic alpha expressions.

    Pipeline:

        AST
        ↓
        AlphaEvaluator
        ↓
        Alpha values
        ↓
        Future returns
        ↓
        Daily IC
        ↓
        Mean IC / ICIR
    """

    HORIZONS = {
        1: "forward_return_1d",
        5: "forward_return_5d",
        20: "forward_return_20d",
        30: "forward_return_30d",
    }

    def __init__(
        self,
        research_data: pd.DataFrame,
    ):
        self.data = self._prepare_data(
            research_data
        )

    # =========================================================
    # Public API
    # =========================================================

    def evaluate(
        self,
        ast: ASTNode,
        horizon: int = 20,
    ) -> FitnessResult:
        """
        Evaluate an alpha at a particular
        future-return horizon.

        Example:

            evaluate(ast, horizon=20)
        """

        if horizon not in self.HORIZONS:
            raise FitnessEvaluationError(
                f"Unsupported horizon: {horizon}. "
                f"Supported horizons: "
                f"{sorted(self.HORIZONS)}"
            )

        # -----------------------------------------------------
        # 1. Evaluate alpha
        # -----------------------------------------------------

        evaluator = AlphaEvaluator(
            self.data
        )

        alpha_df = evaluator.evaluate(
            ast
        )

        # -----------------------------------------------------
        # 2. Merge alpha with future returns
        # -----------------------------------------------------

        return_column = self.HORIZONS[
            horizon
        ]

        merged = alpha_df.merge(
            self.data[
                [
                    "datetime",
                    "instrument",
                    return_column,
                ]
            ],
            on=[
                "datetime",
                "instrument",
            ],
            how="left",
        )

        # -----------------------------------------------------
        # 3. Calculate daily IC
        # -----------------------------------------------------

        daily_ic = self._calculate_daily_ic(
            merged,
            return_column,
        )

        # -----------------------------------------------------
        # 4. Calculate summary statistics
        # -----------------------------------------------------

        return self._calculate_fitness_result(
            daily_ic=daily_ic,
            merged=merged,
            return_column=return_column,
            horizon=horizon,
        )

    # =========================================================
    # Data Preparation
    # =========================================================

    @staticmethod
    def _prepare_data(
        data: pd.DataFrame,
    ) -> pd.DataFrame:

        required = {
            "datetime",
            "instrument",
            "forward_return_1d",
            "forward_return_5d",
            "forward_return_20d",
            "forward_return_30d",
        }

        missing = required - set(
            data.columns
        )

        if missing:
            raise FitnessEvaluationError(
                f"Missing required columns: "
                f"{sorted(missing)}"
            )

        df = data.copy()

        df["datetime"] = pd.to_datetime(
            df["datetime"]
        )

        df = df.sort_values(
            [
                "instrument",
                "datetime",
            ]
        ).reset_index(drop=True)

        return df

    # =========================================================
    # Daily IC
    # =========================================================

    @staticmethod
    def _calculate_daily_ic(
        merged: pd.DataFrame,
        return_column: str,
    ) -> pd.Series:
        """
        Calculate cross-sectional Pearson
        correlation between alpha and future
        return independently for each date.

        IC_t = Corr(alpha_t, return_t+h)
        """

        def calculate_ic(group):

            valid = (
                group["alpha"].notna()
                & group[return_column].notna()
                & np.isfinite(
                    group["alpha"]
                )
                & np.isfinite(
                    group[return_column]
                )
            )

            valid_data = group.loc[
                valid,
                [
                    "alpha",
                    return_column,
                ],
            ]

            # Correlation requires at least
            # two observations.
            if len(valid_data) < 2:
                return np.nan

            # If either variable is constant,
            # correlation is undefined.
            if (
                valid_data["alpha"].nunique()
                < 2
                or
                valid_data[return_column].nunique()
                < 2
            ):
                return np.nan

            return valid_data[
                "alpha"
            ].corr(
                valid_data[return_column]
            )

        daily_ic = (
            merged
            .groupby("datetime")
            .apply(
                calculate_ic,
                include_groups=False,
            )
        )

        daily_ic.name = "ic"

        return daily_ic

    # =========================================================
    # Fitness Statistics
    # =========================================================

    @staticmethod
    def _calculate_fitness_result(
        daily_ic: pd.Series,
        merged: pd.DataFrame,
        return_column: str,
        horizon: int,
    ) -> FitnessResult:

        valid_ic = daily_ic.dropna()

        # -----------------------------------------------------
        # No valid IC
        # -----------------------------------------------------

        if len(valid_ic) == 0:

            return FitnessResult(
                horizon=horizon,
                mean_ic=np.nan,
                ic_std=np.nan,
                icir=np.nan,
                num_days=0,
                num_observations=0,
                coverage=0.0,
            )

        # -----------------------------------------------------
        # Mean IC
        # -----------------------------------------------------

        mean_ic = float(
            valid_ic.mean()
        )

        # -----------------------------------------------------
        # IC standard deviation
        # -----------------------------------------------------

        ic_std = float(
            valid_ic.std(
                ddof=1
            )
        )

        # -----------------------------------------------------
        # ICIR
        # -----------------------------------------------------

        if (
            np.isfinite(ic_std)
            and ic_std > 0
        ):
            icir = float(
                mean_ic / ic_std
            )
        else:
            icir = np.nan

        # -----------------------------------------------------
        # Number of valid days
        # -----------------------------------------------------

        num_days = int(
            len(valid_ic)
        )

        # -----------------------------------------------------
        # Number of usable observations
        # -----------------------------------------------------

        valid_observations = (
            merged["alpha"].notna()
            & merged[return_column].notna()
            & np.isfinite(
                merged["alpha"]
            )
            & np.isfinite(
                merged[return_column]
            )
        )

        num_observations = int(
            valid_observations.sum()
        )

        # -----------------------------------------------------
        # Coverage
        # -----------------------------------------------------

        total_observations = len(
            merged
        )

        if total_observations > 0:
            coverage = (
                num_observations
                / total_observations
            )
        else:
            coverage = 0.0

        return FitnessResult(
            horizon=horizon,
            mean_ic=mean_ic,
            ic_std=ic_std,
            icir=icir,
            num_days=num_days,
            num_observations=num_observations,
            coverage=float(coverage),
        )