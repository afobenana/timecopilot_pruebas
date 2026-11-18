import pandas as pd
from tabicl.forecast import TabICLForecaster as _TabICLForecaster

from ..utils.forecaster import Forecaster, QuantileConverter


class TabICLv2(Forecaster):
    def __init__(
        self,
        alias: str = "TabICLv2",
        max_context_length: int = 4096,
        point_estimate: str = "median",
        tabicl_config: dict | None = None,
    ) -> None:
        self.alias = alias
        self._forecaster = _TabICLForecaster(
            max_context_length=max_context_length,
            point_estimate=point_estimate,
            tabicl_config=tabicl_config,
        )

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        if h < 1:
            raise ValueError(f"h must be positive, got {h}")

        freq = self._maybe_infer_freq(df, freq)
        converter = QuantileConverter(level=level, quantiles=quantiles)
        requested = converter.quantiles
        model_quantiles = (
            [float(q) for q in requested] if requested is not None else [0.5]
        )

        context = df[["unique_id", "ds", "y"]].rename(
            columns={"unique_id": "item_id", "ds": "timestamp", "y": "target"}
        )
        context["timestamp"] = pd.to_datetime(context["timestamp"])
        context = context.sort_values(["item_id", "timestamp"])

        future_parts = []
        for item_id, group in context.groupby("item_id", sort=False):
            dates = pd.date_range(
                start=group["timestamp"].iloc[-1],
                periods=h + 1,
                freq=freq,
            )[1:]
            future_parts.append(
                pd.DataFrame({"item_id": item_id, "timestamp": dates})
            )
        future = pd.concat(future_parts, ignore_index=True)

        prediction = self._forecaster.predict_df(
            context,
            future_df=future,
            quantiles=model_quantiles,
        ).reset_index()

        rename = {
            "item_id": "unique_id",
            "timestamp": "ds",
            "target": self.alias,
        }
        if requested is not None:
            rename.update(
                {float(q): f"{self.alias}-q-{int(float(q) * 100)}" for q in requested}
            )

        result = prediction.rename(columns=rename)
        if requested is not None:
            result = converter.maybe_convert_quantiles_to_level(
                result, models=[self.alias]
            )
        return result


__all__ = ["TabICLv2"]