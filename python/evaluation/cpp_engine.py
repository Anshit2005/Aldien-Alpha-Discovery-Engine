import pandas as pd

from python.cpp_engine import alpha_engine_cpp


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

    @staticmethod
    def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
        missing = (
            CppAlphaEngine.REQUIRED_COLUMNS
            - set(data.columns)
        )

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        df = data.copy()

        df["datetime"] = pd.to_datetime(df["datetime"])

        df = (
            df.sort_values(
                ["instrument", "datetime"]
            )
            .reset_index(drop=True)
        )

        return df

    def _to_cpp_data(self):
        data = alpha_engine_cpp.MarketData()

        data.datetime = (
            self.data["datetime"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        data.instrument = (
            self.data["instrument"]
            .astype(str)
            .tolist()
        )

        data.open = self.data["open"].tolist()
        data.high = self.data["high"].tolist()
        data.low = self.data["low"].tolist()
        data.close = self.data["close"].tolist()
        data.volume = self.data["volume"].tolist()

        for horizon, column in self.RETURN_COLUMNS.items():
            if column in self.data.columns:
                setattr(
                    data,
                    column,
                    self.data[column].tolist()
                )

        return data

    def evaluate(self, ast):
        """
        Evaluate an AST using the C++ engine.

        Returns:
            numpy.ndarray
        """

        cpp_data = self._to_cpp_data()

        return alpha_engine_cpp.evaluate(
            cpp_data,
            ast
        )

    def evaluate_fitness(
        self,
        ast,
        horizon=20
    ):
        """
        Evaluate alpha fitness using C++.
        """

        if horizon not in self.RETURN_COLUMNS:
            raise ValueError(
                f"Unsupported horizon: {horizon}"
            )

        column = self.RETURN_COLUMNS[horizon]

        if column not in self.data.columns:
            raise ValueError(
                f"Missing column: {column}"
            )

        cpp_data = self._to_cpp_data()

        return alpha_engine_cpp.evaluate_fitness(
            cpp_data,
            ast,
            horizon
        )