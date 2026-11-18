# timecopilot/models/foundation/citras_fm.py

import numpy as np
import pandas as pd
from citras_fm import CitrasFM as _CitrasFM

from ..utils.forecaster import Forecaster, QuantileConverter


class CITRASFM(Forecaster):
    def __init__(
        self,
        alias: str = "CITRAS-FM",
        model_id: str = "hitachi-nlp/citras-fm",
        revision: str = "ckpt-2026-07-08",
        map_location: str = "cpu",
    ) -> None:
        self.alias = alias
        self.model_id = model_id
        self.revision = revision
        self.map_location = map_location
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = _CitrasFM.from_pretrained(
                self.model_id,
                revision=self.revision,
                map_location=self.map_location,
            )
        return self._model

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        freq = self._maybe_infer_freq(df, freq)
        converter = QuantileConverter(level=level, quantiles=quantiles)
        model = self._get_model()
        native_levels = np.asarray(model.quantiles, dtype=float)
        requested = converter.quantiles

        if requested is not None:
            unsupported = [q for q in requested if q < native_levels[0] or q > native_levels[-1]]
            if unsupported:
                raise ValueError(
                    f"{self.alias} solo devuelve cuantiles entre "
                    f"{native_levels[0]:.1f} y {native_levels[-1]:.1f}; "
                    f"no puede producir {unsupported}."
                )

        forecasts = []
        for unique_id, group in df.groupby("unique_id", sort=False):
            group = group.sort_values("ds")
            target = group["y"].to_numpy(dtype=np.float32)
            median, native = model.forecast(target, horizon=h)

            median = np.asarray(median).reshape(-1)
            native = native.detach().cpu().numpy().reshape(h, -1)

            last_ds = pd.Timestamp(group["ds"].iloc[-1])
            future_ds = pd.date_range(last_ds, periods=h + 1, freq=freq)[1:]
            result = pd.DataFrame(
                {"unique_id": unique_id, "ds": future_ds, self.alias: median}
            )

            if requested is not None:
                for q in requested:
                    values = np.array(
                        [np.interp(q, native_levels, row) for row in native]
                    )
                    result[f"{self.alias}-q-{int(q * 100)}"] = values

            forecasts.append(result)

        result = pd.concat(forecasts, ignore_index=True)
        if requested is not None:
            result = converter.maybe_convert_quantiles_to_level(
                result, models=[self.alias]
            )
        return result


__all__ = ["CITRASFM"]