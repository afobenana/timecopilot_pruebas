from __future__ import annotations

import math

import numpy as np
import pandas as pd

from timecopilot.models.utils.forecaster import Forecaster
from time_moe.models.modeling_time_moe import TimeMoeForPrediction

class TimeMoE(Forecaster):
    _COVERAGE_BY_QUANTILE = {
        10: 0.8,
        20: 0.6,
        30: 0.4,
        50: None,
        70: 0.4,
        80: 0.6,
        90: 0.8,
    }
    _SUPPORTED_LEVELS = {40, 60, 80}
    _MAX_SEQUENCE_LENGTH = 4096

    def __init__(
        self,
        repo_id: str = "Maple728/TimeMoE-50M",
        alias: str = "TimeMoE-50M",
        context_length: int = 512,
        seasonality: int = 12,
        calibration_windows: int = 5,
        calibration_step_size: int | None = None,
        device_map: str = "auto",
        revision: str | None = None,
    ) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if seasonality < 1:
            raise ValueError("seasonality must be positive")
        if calibration_windows < 1:
            raise ValueError("calibration_windows must be positive")
        if calibration_step_size is not None and calibration_step_size < 1:
            raise ValueError("calibration_step_size must be positive")

        self.repo_id = repo_id
        self.alias = alias
        self.context_length = context_length
        self.seasonality = seasonality
        self.calibration_windows = calibration_windows
        self.calibration_step_size = calibration_step_size
        self.device_map = device_map
        self.revision = revision
        self.model =  None

    def _load_model(self):
        if self.model is None:
            import torch

            self.model = TimeMoeForPrediction.from_pretrained(
                self.repo_id,
                device_map=self.device_map,
                torch_dtype="auto",
                revision=self.revision,
            )
            self.model.eval()
            self._torch = torch

        return self.model

    def _predict_point(self, values: np.ndarray, h: int) -> np.ndarray:
        model = self._load_model()
        torch = self._torch

        context = values[-self.context_length :]
        device = next(model.parameters()).device
        x = torch.as_tensor(
            context,
            dtype=model.dtype,
            device=device,
        ).unsqueeze(0)

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        normalized = (x - mean) / std

        with torch.inference_mode():
            output = model.generate(
                normalized,
                max_new_tokens=h,
            )

        prediction = output[0, -h:] * std[0, 0] + mean[0, 0]
        prediction = prediction.float().cpu().numpy()

        if len(prediction) != h or not np.isfinite(prediction).all():
            raise ValueError("TimeMoE returned invalid predictions")

        return prediction

    def _scale(self, values: np.ndarray) -> float:
        if len(values) > self.seasonality:
            seasonal_errors = np.abs(
                values[self.seasonality :] - values[: -self.seasonality]
            )
            scale = float(np.mean(seasonal_errors))
        else:
            scale = 0.0

        if not np.isfinite(scale) or scale <= 1e-8:
            scale = float(np.mean(np.abs(values - np.median(values))))

        return scale if np.isfinite(scale) and scale > 1e-8 else 1.0

    @staticmethod
    def _conformal_radius(errors: list[float], coverage: float) -> float:
        n = len(errors)
        rank = math.ceil((n + 1) * coverage)

        if rank > n:
            raise ValueError(
                f"Only {n} calibration errors are available; more are needed "
                f"for a {coverage:.0%} interval"
            )

        return float(np.partition(np.asarray(errors), rank - 1)[rank - 1])

    def _calibration_radii(
        self,
        df: pd.DataFrame,
        h: int,
    ) -> dict[float, np.ndarray]:
        step_size = self.calibration_step_size or h
        errors_by_horizon: list[list[float]] = [[] for _ in range(h)]

        for _, series in df.groupby("unique_id", sort=False):
            values = (
                series.sort_values("ds")["y"]
                .to_numpy(dtype=float)
            )
            origins = list(range(2, len(values) - h + 1, step_size))
            origins = origins[-self.calibration_windows :]

            for origin in origins:
                train = values[:origin]
                actual = values[origin : origin + h]
                predicted = self._predict_point(train, h)
                normalized_errors = np.abs(actual - predicted) / self._scale(train)

                for horizon_idx, error in enumerate(normalized_errors):
                    errors_by_horizon[horizon_idx].append(float(error))

        radii: dict[float, list[float]] = {0.4: [], 0.6: [], 0.8: []}
        for horizon_idx, errors in enumerate(errors_by_horizon):
            for coverage in radii:
                try:
                    radius = self._conformal_radius(errors, coverage)
                except ValueError as exc:
                    raise ValueError(
                        f"Not enough rolling-origin errors for horizon "
                        f"{horizon_idx + 1}: {exc}. Add more history or reduce "
                        "the forecast horizon."
                    ) from exc
                radii[coverage].append(radius)

        return {
            coverage: np.asarray(values)
            for coverage, values in radii.items()
        }

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        if h < 1:
            raise ValueError("h must be positive")
        if self.context_length + h > self._MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "TimeMoE requires context_length + h <= 4096"
            )
        if level is not None and quantiles is not None:
            raise ValueError("Pass level or quantiles, not both")

        freq = self._maybe_infer_freq(df, freq)

        requested_quantiles = None
        if quantiles is not None:
            requested_quantiles = sorted({round(float(q) * 100) for q in quantiles})
            unsupported = set(requested_quantiles) - set(self._COVERAGE_BY_QUANTILE)
            if unsupported:
                raise ValueError(
                    f"Unsupported quantiles: {sorted(unsupported)}; "
                    "supported values are 0.1 through 0.9 by 0.1"
                )

        requested_levels = None
        if level is not None:
            requested_levels = sorted({int(lv) for lv in level})
            unsupported = set(requested_levels) - self._SUPPORTED_LEVELS
            if unsupported:
                raise ValueError(
                    f"Unsupported interval levels: {sorted(unsupported)}"
                )

        needs_intervals = requested_quantiles is not None or requested_levels is not None
        radii = self._calibration_radii(df, h) if needs_intervals else None

        forecasts = []
        for unique_id, series in df.groupby("unique_id", sort=False):
            series = series.sort_values("ds")
            values = series["y"].to_numpy(dtype=float)
            point = self._predict_point(values, h)
            scale = self._scale(values)
            future_ds = pd.date_range(
                start=pd.to_datetime(series["ds"].iloc[-1]),
                periods=h + 1,
                freq=freq,
            )[1:]

            result = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": future_ds,
                    self.alias: point,
                }
            )

            if requested_quantiles is not None:
                for q in requested_quantiles:
                    coverage = self._COVERAGE_BY_QUANTILE[q]
                    if coverage is None:
                        prediction = point
                    else:
                        direction = -1 if q < 50 else 1
                        prediction = (
                            point + direction * scale * radii[coverage]
                        )
                    result[f"{self.alias}-q-{q}"] = prediction

            if requested_levels is not None:
                for interval_level in requested_levels:
                    coverage = interval_level / 100
                    radius = scale * radii[coverage]
                    result[f"{self.alias}-lo-{interval_level}"] = point - radius
                    result[f"{self.alias}-hi-{interval_level}"] = point + radius

            forecasts.append(result)

        return pd.concat(forecasts, ignore_index=True)