import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..utils.forecaster import Forecaster, QuantileConverter


class MOMENT(Forecaster):
    """MOMENT-1 forecaster with bounded fold-local head fine-tuning."""

    def __init__(
        self,
        repo_id: str = "AutonLab/MOMENT-1-base",
        alias: str = "MOMENT",
        context_length: int = 512,
        epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        calibration_windows: int = 3,
        max_train_windows_per_series: int = 64,
        train_stride: int = 12,
    ) -> None:
        self.repo_id = repo_id
        self.alias = alias
        self.context_length = context_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.calibration_windows = calibration_windows
        self.max_train_windows_per_series = max_train_windows_per_series
        self.train_stride = train_stride

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def supports_native_futr_exog(self) -> bool:
        return False

    def _context(
        self,
        values: np.ndarray,
        end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        start = max(0, end - self.context_length)

        context = np.asarray(
            values[start:end],
            dtype=np.float32,
        )

        observed = np.isfinite(context)

        x = np.zeros(
            self.context_length,
            dtype=np.float32,
        )

        input_mask = np.zeros(
            self.context_length,
            dtype=np.float32,
        )

        size = len(context)

        if size:
            x[-size:] = np.nan_to_num(
                context,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            input_mask[-size:] = observed.astype(
                np.float32
            )

        return x, input_mask

    def _load_model(
        self,
        horizon: int,
    ) -> torch.nn.Module:
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

        model.init()
        model = model.to(self.device)

        return model

    def _build_training_windows(
        self,
        groups: list[tuple[object, pd.DataFrame]],
        h: int,
        holdout_size: int,
    ) -> tuple[
        list[np.ndarray],
        list[np.ndarray],
        list[np.ndarray],
    ]:
        train_x = []
        train_masks = []
        train_y = []

        for _, series in groups:
            values = series["y"].to_numpy(
                dtype=np.float32
            )

            train_end = len(values) - holdout_size

            if train_end < h + 1:
                continue

            latest_origin = train_end - h

            origins = np.arange(
                latest_origin,
                0,
                -self.train_stride,
                dtype=int,
            )

            origins = origins[
                : self.max_train_windows_per_series
            ]

            origins = origins[::-1]

            for origin in origins:
                target = values[
                    origin : origin + h
                ]

                if len(target) != h:
                    continue

                if not np.isfinite(target).all():
                    continue

                x, input_mask = self._context(
                    values,
                    origin,
                )

                train_x.append(x)
                train_masks.append(input_mask)
                train_y.append(target)

        return train_x, train_masks, train_y

    def _train_model(
        self,
        model: torch.nn.Module,
        train_x: list[np.ndarray],
        train_masks: list[np.ndarray],
        train_y: list[np.ndarray],
    ) -> None:
        if not train_x:
            raise ValueError(
                "No hay suficientes datos históricos para "
                "entrenar la cabeza de forecasting de MOMENT."
            )

        dataset = TensorDataset(
            torch.from_numpy(
                np.stack(train_x)[:, None, :]
            ),
            torch.from_numpy(
                np.stack(train_masks)
            ),
            torch.from_numpy(
                np.stack(train_y)[:, None, :]
            ),
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
        )

        trainable_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError(
                "MOMENT no tiene parámetros entrenables "
                "en la cabeza de forecasting."
            )

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=self.learning_rate,
            weight_decay=0.0,
        )

        criterion = torch.nn.MSELoss()

        print(
            f"[{self.alias}] device={self.device} | "
            f"ventanas={len(dataset)} | "
            f"batches={len(loader)} | "
            f"epochs={self.epochs}",
            flush=True,
        )

        model.train()

        for epoch in range(self.epochs):
            total_loss = 0.0

            for batch_index, (
                x,
                input_mask,
                target,
            ) in enumerate(loader, start=1):
                x = x.to(
                    self.device,
                    non_blocking=True,
                )

                input_mask = input_mask.to(
                    self.device,
                    non_blocking=True,
                )

                target = target.to(
                    self.device,
                    non_blocking=True,
                )

                optimizer.zero_grad(
                    set_to_none=True
                )

                prediction = model(
                    x_enc=x,
                    input_mask=input_mask,
                ).forecast

                loss = criterion(
                    prediction,
                    target,
                )

                loss.backward()
                optimizer.step()

                total_loss += float(
                    loss.detach().cpu()
                )

                if (
                    batch_index == 1
                    or batch_index % 10 == 0
                    or batch_index == len(loader)
                ):
                    print(
                        f"[{self.alias}] "
                        f"epoch={epoch + 1}/{self.epochs} | "
                        f"batch={batch_index}/{len(loader)} | "
                        f"loss={loss.item():.6f}",
                        flush=True,
                    )

            mean_loss = total_loss / len(loader)

            print(
                f"[{self.alias}] "
                f"epoch={epoch + 1} terminada | "
                f"loss_media={mean_loss:.6f}",
                flush=True,
            )

        model.eval()

    def _predict(
        self,
        model: torch.nn.Module,
        contexts: list[np.ndarray],
        masks: list[np.ndarray],
    ) -> np.ndarray:
        if not contexts:
            raise ValueError(
                "No hay contextos para generar predicciones."
            )

        predictions = []

        model.eval()

        with torch.inference_mode():
            for start in range(
                0,
                len(contexts),
                self.batch_size,
            ):
                stop = start + self.batch_size

                x = torch.from_numpy(
                    np.stack(
                        contexts[start:stop]
                    )[:, None, :]
                ).to(
                    self.device,
                    non_blocking=True,
                )

                input_mask = torch.from_numpy(
                    np.stack(
                        masks[start:stop]
                    )
                ).to(
                    self.device,
                    non_blocking=True,
                )

                output = model(
                    x_enc=x,
                    input_mask=input_mask,
                ).forecast

                predictions.append(
                    output[:, 0, :]
                    .detach()
                    .cpu()
                    .numpy()
                )

        return np.concatenate(
            predictions,
            axis=0,
        )

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
    ) -> pd.DataFrame:
        required = {
            "unique_id",
            "ds",
            "y",
        }

        missing = required.difference(
            df.columns
        )

        if missing:
            raise ValueError(
                f"Faltan columnas obligatorias: "
                f"{sorted(missing)}"
            )

        if h <= 0:
            raise ValueError(
                "h debe ser un entero positivo."
            )

        df = df[
            [
                "unique_id",
                "ds",
                "y",
            ]
        ].copy()

        df["ds"] = pd.to_datetime(
            df["ds"]
        )

        if df.duplicated(
            ["unique_id", "ds"]
        ).any():
            raise ValueError(
                "df contiene claves unique_id/ds duplicadas."
            )

        df = df.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        freq = self._maybe_infer_freq(
            df,
            freq,
        )

        converter = QuantileConverter(
            level=level,
            quantiles=quantiles,
        )

        requested_quantiles = converter.quantiles
        needs_calibration = (
            requested_quantiles is not None
        )

        holdout_size = (
            h * self.calibration_windows
            if needs_calibration
            else 0
        )

        groups = list(
            df.groupby(
                "unique_id",
                sort=False,
            )
        )

        if not groups:
            raise ValueError(
                "El dataframe no contiene series."
            )

        for unique_id, series in groups:
            minimum_length = (
                h + holdout_size + 1
            )

            if len(series) < minimum_length:
                raise ValueError(
                    f"La serie {unique_id} tiene "
                    f"{len(series)} observaciones; "
                    f"se necesitan al menos "
                    f"{minimum_length}."
                )

        train_x, train_masks, train_y = (
            self._build_training_windows(
                groups=groups,
                h=h,
                holdout_size=holdout_size,
            )
        )

        print(
            f"[{self.alias}] Cargando "
            f"{self.repo_id}...",
            flush=True,
        )

        model = self._load_model(h)

        print(
            f"[{self.alias}] Modelo cargado.",
            flush=True,
        )

        self._train_model(
            model=model,
            train_x=train_x,
            train_masks=train_masks,
            train_y=train_y,
        )

        quantile_offsets = None

        if needs_calibration:
            calibration_contexts = []
            calibration_masks = []
            calibration_targets = []

            for _, series in groups:
                values = series["y"].to_numpy(
                    dtype=np.float32
                )

                holdout_start = (
                    len(values) - holdout_size
                )

                for window in range(
                    self.calibration_windows
                ):
                    origin = (
                        holdout_start
                        + window * h
                    )

                    target = values[
                        origin : origin + h
                    ]

                    if origin < 1:
                        continue

                    if len(target) != h:
                        continue

                    if not np.isfinite(
                        target
                    ).all():
                        continue

                    x, input_mask = self._context(
                        values,
                        origin,
                    )

                    calibration_contexts.append(
                        x
                    )

                    calibration_masks.append(
                        input_mask
                    )

                    calibration_targets.append(
                        target
                    )

            if not calibration_targets:
                raise ValueError(
                    "No se pudieron construir ventanas "
                    "de calibración válidas."
                )

            calibration_predictions = (
                self._predict(
                    model=model,
                    contexts=calibration_contexts,
                    masks=calibration_masks,
                )
            )

            calibration_targets_array = (
                np.stack(
                    calibration_targets
                )
            )

            residuals = (
                calibration_targets_array
                - calibration_predictions
            )

            quantile_offsets = np.quantile(
                residuals,
                requested_quantiles,
                axis=0,
            )

        final_contexts = []
        final_masks = []

        for _, series in groups:
            values = series["y"].to_numpy(
                dtype=np.float32
            )

            x, input_mask = self._context(
                values,
                len(values),
            )

            final_contexts.append(x)
            final_masks.append(input_mask)

        point_forecasts = self._predict(
            model=model,
            contexts=final_contexts,
            masks=final_masks,
        )

        results = []

        for row, (
            unique_id,
            series,
        ) in enumerate(groups):
            last_date = pd.Timestamp(
                series["ds"].iloc[-1]
            )

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
                for index, quantile in enumerate(
                    requested_quantiles
                ):
                    result[
                        f"{self.alias}-q-"
                        f"{int(round(quantile * 100))}"
                    ] = (
                        point
                        + quantile_offsets[index]
                    )

            results.append(result)

        result = pd.concat(
            results,
            ignore_index=True,
        )

        expected_rows = (
            len(groups) * h
        )

        if len(result) != expected_rows:
            raise ValueError(
                f"{self.alias} generó "
                f"{len(result)} filas, pero se "
                f"esperaban {expected_rows}."
            )

        if requested_quantiles is not None:
            result = (
                converter.maybe_convert_quantiles_to_level(
                    result,
                    models=[self.alias],
                )
            )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result


__all__ = ["MOMENT"]