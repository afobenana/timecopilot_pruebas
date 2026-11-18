import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils.forecaster import Forecaster, QuantileConverter


class MOMENT(Forecaster):
    """MOMENT-1-base forecaster with fold-local head fine-tuning."""

    def __init__(
        self,
        repo_id: str = "AutonLab/MOMENT-1-base",
        alias: str = "MOMENT",
        context_length: int = 512,
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        calibration_windows: int = 3,
    ) -> None:
        self.repo_id = repo_id
        self.alias = alias
        self.context_length = context_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.calibration_windows = calibration_windows
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _context(self, values: np.ndarray, end: int) -> tuple[np.ndarray, np.ndarray]:
        start = max(0, end - self.context_length)
        context = np.asarray(values[start:end], dtype=np.float32)
        observed = np.isfinite(context)

        x = np.zeros(self.context_length, dtype=np.float32)
        input_mask = np.zeros(self.context_length, dtype=np.float32)

        size = len(context)
        if size:
            x[-size:] = np.nan_to_num(context, nan=0.0, posinf=0.0, neginf=0.0)
            input_mask[-size:] = observed.astype(np.float32)

        return x, input_mask

    def _load_model(self, horizon: int) -> torch.nn.Module:
        from momentfm import MOMENTPipeline

        model = MOMENTPipeline.from_pretrained(
            self.repo_id,
            model_kwargs={
                "task_name": "forecasting",
                "forecast_horizon": horizon,
                "head_dropout": 0.1,
                "freeze_encoder": True,
                "freeze_embedder": True,
                "freeze_head": False,
            },
        )
        # MOMENT replaces the pretrained reconstruction head with a new,
        # randomly initialized forecasting head here.
        model.init()
        return model.to(self.device)

    def _train_model(
        self,
        model: torch.nn.Module,
        train_x: list[np.ndarray],
        train_masks: list[np.ndarray],
        train_y: list[np.ndarray],
    ) -> None:
        if not train_x:
            raise ValueError(
                "Not enough historical data to fine-tune MOMENT's forecasting head."
            )

        dataset = TensorDataset(
            torch.from_numpy(np.stack(train_x)[:, None, :]),
            torch.from_numpy(np.stack(train_masks)),
            torch.from_numpy(np.stack(train_y)[:, None, :]),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("MOMENT has no trainable forecasting-head parameters.")

        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.learning_rate,
            weight_decay=0.0,
        )
        criterion = torch.nn.MSELoss()

        model.train()
        for _ in range(self.epochs):
            for x, input_mask, target in loader:
                x = x.to(self.device)
                input_mask = input_mask.to(self.device)
                target = target.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                prediction = model(x_enc=x, input_mask=input_mask).forecast
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()

        model.eval()

    def _predict(
        self,
        model: torch.nn.Module,
        contexts: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> np.ndarray:
        predictions = []

        with torch.inference_mode():
            for start in range(0, len(contexts), self.batch_size):
                x = torch.from_numpy(
                    np.stack(contexts[start : start + self.batch_size])[:, None, :]
                ).to(self.device)
                input_mask = torch.from_numpy(
                    np.stack(masks[start : start + self.batch_size])
                ).to(self.device)

                output = model(x_enc=x, input_mask=input_mask).forecast
                predictions.append(output[:, 0, :].cpu().numpy())

        return np.concatenate(predictions, axis=0)

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        required = {"unique_id", "ds", "y"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Input is missing required columns: {sorted(missing)}")

        df = df[["unique_id", "ds", "y"]].copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(["unique_id", "ds"])
        freq = self._maybe_infer_freq(df, freq)
        converter = QuantileConverter(level=level, quantiles=quantiles)
        needs_calibration = converter.quantiles is not None
        holdout_size = h * self.calibration_windows if needs_calibration else 0

        groups = list(df.groupby("unique_id", sort=False))
        train_x, train_masks, train_y = [], [], []

        for _, series in groups:
            values = series["y"].to_numpy(dtype=np.float32)
            train_end = len(values) - holdout_size

            for origin in range(1, train_end - h + 1):
                target = values[origin : origin + h]
                if not np.isfinite(target).all():
                    continue

                x, input_mask = self._context(values, origin)
                train_x.append(x)
                train_masks.append(input_mask)
                train_y.append(target)

        model = self._load_model(h)
        self._train_model(model, train_x, train_masks, train_y)

        calibration_contexts, calibration_masks, calibration_targets = [], [], []
        if needs_calibration:
            for _, series in groups:
                values = series["y"].to_numpy(dtype=np.float32)
                holdout_start = len(values) - holdout_size

                for window in range(self.calibration_windows):
                    origin = holdout_start + window * h
                    target = values[origin : origin + h]
                    if origin < 1 or len(target) != h or not np.isfinite(target).all():
                        continue

                    x, input_mask = self._context(values, origin)
                    calibration_contexts.append(x)
                    calibration_masks.append(input_mask)
                    calibration_targets.append(target)

            if not calibration_targets:
                raise ValueError(
                    "No valid calibration windows. Reduce calibration_windows "
                    "or provide longer series."
                )

            calibration_predictions = self._predict(
                model,
                calibration_contexts,
                calibration_masks,
            )
            residuals = np.stack(calibration_targets) - calibration_predictions
            quantile_offsets = np.quantile(
                residuals,
                converter.quantiles,
                axis=0,
            )
        else:
            quantile_offsets = None

        final_contexts, final_masks = [], []
        for _, series in groups:
            values = series["y"].to_numpy(dtype=np.float32)
            x, input_mask = self._context(values, len(values))
            final_contexts.append(x)
            final_masks.append(input_mask)

        point_forecasts = self._predict(model, final_contexts, final_masks)
        results = []

        for row, (unique_id, series) in enumerate(groups):
            last_date = series["ds"].iloc[-1]
            future_dates = pd.date_range(
                start=last_date,
                periods=h + 1,
                freq=freq,
            )[1:]
            point = point_forecasts[row]

            result = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": future_dates,
                    self.alias: point,
                }
            )

            if quantile_offsets is not None:
                for index, q in enumerate(converter.quantiles):
                    result[f"{self.alias}-q-{int(q * 100)}"] = (
                        point + quantile_offsets[index]
                    )

            results.append(result)

        result = pd.concat(results, ignore_index=True)
        if converter.quantiles is not None:
            result = converter.maybe_convert_quantiles_to_level(
                result,
                models=[self.alias],
            )
        return result


__all__ = ["MOMENT"]