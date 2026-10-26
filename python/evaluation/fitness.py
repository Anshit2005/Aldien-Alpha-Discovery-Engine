import numpy as np
import pandas as pd

from dataclasses import dataclass, field

from python.dsl.ast import ASTNode
from python.evaluation.evaluator import AlphaEvaluator


class FitnessEvaluationError(Exception):
    """Raised when alpha fitness cannot be calculated."""
    pass


@dataclass
class FitnessResult:
    """
    Summary of an alpha's predictive performance.

    ``daily_ic`` holds the per-date IC series behind the summary
    statistics (NaN on dates where IC is undefined). It is kept in
    memory for statistical testing and is not part of ``to_dict()``.
    """

    horizon: int
    mean_ic: float
    ic_std: float
    icir: float
    num_days: int
    num_observations: int
    coverage: float
    daily_ic: pd.Series | None = field(
        default=None,
        repr=False,
        compare=False,
    )

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

        # Built on first use and reused for every evaluation:
        # constructing it copies and sorts the full panel.
        self._alpha_evaluator: AlphaEvaluator | None = None

    # =========================================================
    # Public API
    # =========================================================

    def evaluate(
        self,
        ast: ASTNode,
        horizon: int = 20,
        split: str | None = None,
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

        if split is not None:
            if split not in {"train", "validation", "test", "holdout"}:
                raise FitnessEvaluationError(
                    f"Unsupported split: {split}. Supported splits: "
                    "['holdout', 'test', 'train', 'validation']"
                )
            split_column = "split"
            target_split_column = f"target_split_{horizon}d"
            missing_split_columns = {
                split_column, target_split_column
            } - set(self.data.columns)
            if missing_split_columns:
                raise FitnessEvaluationError(
                    "Split-aware evaluation requires columns: "
                    f"{sorted(missing_split_columns)}"
                )

        # -----------------------------------------------------
        # 1. Evaluate alpha on the full panel
        #
        # Time-series operators need each instrument's full
        # history, so the alpha is computed before any split
        # filtering.
        # -----------------------------------------------------

        alpha = self._evaluate_alpha(
            ast
        )

        # -----------------------------------------------------
        # 2. Align future returns
        #
        # Alpha rows are in the same order as self.data, so
        # returns are attached by position.
        # -----------------------------------------------------

        return_column = self.HORIZONS[
            horizon
        ]

        returns = self.data[return_column].to_numpy(
            dtype=float,
            copy=True,
        )

        dates = self.data["datetime"]

        if split is not None:
            signal_rows = self.data["split"].eq(split).to_numpy()
            target_matches = (
                self.data[f"target_split_{horizon}d"]
                .eq(split)
                .to_numpy()
            )

            # A target that crosses into another split must
            # not contribute to this split's fitness.
            returns[~target_matches] = np.nan

            alpha = alpha[signal_rows]
            returns = returns[signal_rows]
            dates = dates[signal_rows]

        # -----------------------------------------------------
        # 3. Calculate daily IC
        # -----------------------------------------------------

        daily_ic = self._calculate_daily_ic(
            dates,
            alpha,
            returns,
        )

        # -----------------------------------------------------
        # 4. Calculate summary statistics
        # -----------------------------------------------------

        return self._calculate_fitness_result(
            daily_ic=daily_ic,
            alpha=alpha,
            returns=returns,
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
    # Alpha Evaluation
    # =========================================================

    def _evaluate_alpha(
        self,
        ast: ASTNode,
    ) -> np.ndarray:

        if self._alpha_evaluator is None:

            evaluator = AlphaEvaluator(
                self.data
            )

            # Positional alignment depends on both classes
            # ordering rows identically.
            if not (
                evaluator.data["instrument"].to_numpy()
                == self.data["instrument"].to_numpy()
            ).all() or not (
                evaluator.data["datetime"].to_numpy()
                == self.data["datetime"].to_numpy()
            ).all():
                raise FitnessEvaluationError(
                    "Alpha evaluator row order does not match "
                    "fitness data row order"
                )

            self._alpha_evaluator = evaluator

        alpha_df = self._alpha_evaluator.evaluate(
            ast
        )

        return alpha_df["alpha"].to_numpy(
            dtype=float
        )

    # =========================================================
    # Daily IC
    # =========================================================

    @staticmethod
    def _calculate_daily_ic(
        dates: pd.Series,
        alpha: np.ndarray,
        returns: np.ndarray,
    ) -> pd.Series:
        """
        Calculate cross-sectional Pearson
        correlation between alpha and future
        return independently for each date.

        IC_t = Corr(alpha_t, return_t+h)

        Computed with grouped sums rather than a
        Python function per date. A date's IC is
        NaN when it has fewer than two valid
        observations or when either variable is
        constant on that date.
        """

        codes, unique_dates = pd.factorize(
            dates,
            sort=True,
        )

        num_dates = len(unique_dates)

        daily_ic = np.full(
            num_dates,
            np.nan,
        )

        valid = (
            np.isfinite(alpha)
            & np.isfinite(returns)
        )

        if valid.any():

            day = codes[valid]
            x = alpha[valid]
            y = returns[valid]

            count = np.bincount(
                day,
                minlength=num_dates,
            )

            with np.errstate(
                invalid="ignore",
                divide="ignore",
            ):

                # Centre each date's values on that date's
                # mean before forming products, which keeps
                # the sums numerically stable.
                mean_x = np.bincount(
                    day, weights=x, minlength=num_dates
                ) / count

                mean_y = np.bincount(
                    day, weights=y, minlength=num_dates
                ) / count

                dx = x - mean_x[day]
                dy = y - mean_y[day]

                sxy = np.bincount(
                    day, weights=dx * dy, minlength=num_dates
                )
                sxx = np.bincount(
                    day, weights=dx * dx, minlength=num_dates
                )
                syy = np.bincount(
                    day, weights=dy * dy, minlength=num_dates
                )

                ic = sxy / np.sqrt(
                    sxx * syy
                )

            # Constant values can leave tiny non-zero
            # deviations from rounding, so constancy is
            # tested on the raw values.
            grouped_x = pd.Series(x).groupby(day)
            grouped_y = pd.Series(y).groupby(day)

            varying = np.zeros(
                num_dates,
                dtype=bool,
            )

            observed = np.flatnonzero(
                count > 0
            )

            varying[observed] = (
                (grouped_x.min() < grouped_x.max()).to_numpy()
                & (grouped_y.min() < grouped_y.max()).to_numpy()
            )

            defined = (
                (count >= 2)
                & varying
                & np.isfinite(ic)
            )

            daily_ic[defined] = np.clip(
                ic[defined],
                -1.0,
                1.0,
            )

        return pd.Series(
            daily_ic,
            index=pd.DatetimeIndex(
                unique_dates,
                name="datetime",
            ),
            name="ic",
        )

    # =========================================================
    # Fitness Statistics
    # =========================================================

    @staticmethod
    def _calculate_fitness_result(
        daily_ic: pd.Series,
        alpha: np.ndarray,
        returns: np.ndarray,
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
                daily_ic=daily_ic,
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

        num_observations = int(
            (
                np.isfinite(alpha)
                & np.isfinite(returns)
            ).sum()
        )

        # -----------------------------------------------------
        # Coverage
        # -----------------------------------------------------

        total_observations = len(
            alpha
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
            daily_ic=daily_ic,
        )
