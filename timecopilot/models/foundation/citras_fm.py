import numpy as np
import pandas as pd
import torch

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

    def supports_native_futr_exog(self) -> bool:
        return True

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()

        return np.asarray(value)

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        X_df: pd.DataFrame | None = None,
        *,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
        required_df_cols = {"unique_id", "ds", "y"}
        missing_df_cols = required_df_cols - set(df.columns)

        if missing_df_cols:
            raise ValueError(
                f"Faltan columnas obligatorias en df: "
                f"{sorted(missing_df_cols)}"
            )

        if h <= 0:
            raise ValueError("h debe ser un entero positivo.")

        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        if df.duplicated(["unique_id", "ds"]).any():
            raise ValueError(
                "df tiene claves unique_id/ds duplicadas."
            )

        df = df.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        freq = self._maybe_infer_freq(df, freq)

        converter = QuantileConverter(
            level=level,
            quantiles=quantiles,
        )

        exog_cols: list[str] = []

        if X_df is not None:
            X_df = X_df.copy()

            required_x_cols = {"unique_id", "ds"}
            missing_x_cols = required_x_cols - set(X_df.columns)

            if missing_x_cols:
                raise ValueError(
                    f"Faltan columnas obligatorias en X_df: "
                    f"{sorted(missing_x_cols)}"
                )

            X_df["ds"] = pd.to_datetime(X_df["ds"])

            if "y" in X_df.columns:
                raise ValueError(
                    "X_df debe contener timestamps y covariables, "
                    "pero no la columna objetivo 'y'."
                )

            if futr_exog_list is not None:
                exog_cols = list(futr_exog_list)
            else:
                exog_cols = [
                    column
                    for column in X_df.columns
                    if column not in {"unique_id", "ds"}
                ]

            if not exog_cols:
                raise ValueError(
                    "X_df no contiene columnas exógenas."
                )

            if len(exog_cols) != len(set(exog_cols)):
                raise ValueError(
                    "futr_exog_list contiene columnas duplicadas."
                )

            missing_future = (
                {"unique_id", "ds"} | set(exog_cols)
            ) - set(X_df.columns)

            missing_history = set(exog_cols) - set(df.columns)

            if missing_future or missing_history:
                raise ValueError(
                    "Faltan covariables. "
                    f"En X_df: {sorted(missing_future)}. "
                    f"En df: {sorted(missing_history)}."
                )

            if X_df.duplicated(["unique_id", "ds"]).any():
                raise ValueError(
                    "X_df tiene claves unique_id/ds duplicadas."
                )

            df_ids = set(df["unique_id"].unique())
            x_ids = set(X_df["unique_id"].unique())

            if df_ids != x_ids:
                missing_in_x = df_ids - x_ids
                extra_in_x = x_ids - df_ids

                raise ValueError(
                    "df y X_df deben contener los mismos unique_id. "
                    f"Faltan en X_df: {sorted(missing_in_x)}. "
                    f"Sobran en X_df: {sorted(extra_in_x)}."
                )

            X_df = X_df.sort_values(
                ["unique_id", "ds"]
            ).reset_index(drop=True)

        elif futr_exog_list is not None:
            raise ValueError(
                "futr_exog_list requiere que se proporcione X_df."
            )

        model = self._get_model()

        native_quantiles = np.asarray(
            model.quantiles,
            dtype=float,
        ).reshape(-1)

        requested_quantiles = converter.quantiles

        if requested_quantiles is not None:
            unsupported = [
                quantile
                for quantile in requested_quantiles
                if (
                    quantile < native_quantiles.min()
                    or quantile > native_quantiles.max()
                )
            ]

            if unsupported:
                raise ValueError(
                    f"{self.alias} solo devuelve cuantiles entre "
                    f"{native_quantiles.min():.3f} y "
                    f"{native_quantiles.max():.3f}; "
                    f"no puede producir {unsupported}."
                )

        forecasts: list[pd.DataFrame] = []

        for unique_id, group in df.groupby(
            "unique_id",
            sort=False,
        ):
            group = group.sort_values("ds")

            target = group["y"].to_numpy(
                dtype=np.float32
            )

            if target.size == 0:
                raise ValueError(
                    f"La serie {unique_id} no tiene observaciones."
                )

            if not np.isfinite(target).all():
                raise ValueError(
                    f"La columna y contiene valores no finitos "
                    f"para {unique_id}."
                )

            last_ds = pd.Timestamp(group["ds"].iloc[-1])

            future_ds = pd.date_range(
                start=last_ds,
                periods=h + 1,
                freq=freq,
            )[1:]

            if X_df is None:
                median, native_predictions = model.forecast(
                    target,
                    horizon=h,
                )

            else:
                future = (
                    X_df.loc[
                        X_df["unique_id"] == unique_id,
                        ["ds", *exog_cols],
                    ]
                    .sort_values("ds")
                    .reset_index(drop=True)
                )

                if len(future) != h:
                    raise ValueError(
                        f"X_df debe tener exactamente {h} filas "
                        f"para {unique_id}, pero tiene {len(future)}."
                    )

                # Normalizamos ambas secuencias a datetime64[ns].
                # Así evitamos falsos negativos por diferencias de
                # nombre, resolución temporal o metadatos del índice.
                actual_future_ds = pd.to_datetime(
                    future["ds"]
                ).to_numpy(dtype="datetime64[ns]")

                expected_future_ds = pd.DatetimeIndex(
                    future_ds
                ).to_numpy(dtype="datetime64[ns]")

                if not np.array_equal(
                    actual_future_ds,
                    expected_future_ds,
                ):
                    mismatches = np.flatnonzero(
                        actual_future_ds != expected_future_ds
                    )

                    mismatch_details = [
                        {
                            "position": int(index),
                            "expected": str(
                                expected_future_ds[index]
                            ),
                            "received": str(
                                actual_future_ds[index]
                            ),
                        }
                        for index in mismatches[:10]
                    ]

                    raise ValueError(
                        f"Los timestamps futuros de {unique_id} "
                        f"no coinciden con los {h} pasos esperados. "
                        f"Diferencias: {mismatch_details}"
                    )

                past_covariates = group[
                    exog_cols
                ].to_numpy(dtype=np.float32)

                future_covariates = future[
                    exog_cols
                ].to_numpy(dtype=np.float32)

                known_covariates = np.concatenate(
                    [
                        past_covariates,
                        future_covariates,
                    ],
                    axis=0,
                )

                expected_shape = (
                    len(group) + h,
                    len(exog_cols),
                )

                if known_covariates.shape != expected_shape:
                    raise ValueError(
                        f"Forma incorrecta de known_cov para "
                        f"{unique_id}. "
                        f"Esperado: {expected_shape}. "
                        f"Recibido: {known_covariates.shape}."
                    )

                if not np.isfinite(
                    known_covariates
                ).all():
                    raise ValueError(
                        f"Las covariables contienen valores "
                        f"no finitos para {unique_id}."
                    )

                median, native_predictions = model.forecast(
                    target,
                    horizon=h,
                    known_cov=known_covariates,
                )

            median = self._to_numpy(
                median
            ).reshape(-1)

            native_predictions = self._to_numpy(
                native_predictions
            ).reshape(h, -1)

            if median.shape[0] != h:
                raise ValueError(
                    f"{self.alias} devolvió {median.shape[0]} "
                    f"predicciones puntuales para {unique_id}; "
                    f"se esperaban {h}."
                )

            if native_predictions.shape[1] != len(
                native_quantiles
            ):
                raise ValueError(
                    f"{self.alias} devolvió "
                    f"{native_predictions.shape[1]} cuantiles, "
                    f"pero model.quantiles contiene "
                    f"{len(native_quantiles)}."
                )

            result = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": future_ds,
                    self.alias: median,
                }
            )

            for index, quantile in enumerate(
                native_quantiles
            ):
                result[
                    f"{self.alias}-q-{quantile:g}"
                ] = native_predictions[:, index]

            forecasts.append(result)

        if not forecasts:
            raise ValueError(
                f"{self.alias} no generó ninguna predicción."
            )

        result = pd.concat(
            forecasts,
            ignore_index=True,
        )

        expected_rows = df["unique_id"].nunique() * h

        if len(result) != expected_rows:
            counts = (
                result.groupby("unique_id")
                .size()
                .to_dict()
            )

            raise ValueError(
                f"{self.alias} generó {len(result)} filas, "
                f"pero se esperaban {expected_rows}. "
                f"Filas por serie: {counts}"
            )

        if result.duplicated(
            ["unique_id", "ds"]
        ).any():
            raise ValueError(
                f"{self.alias} generó claves "
                f"unique_id/ds duplicadas."
            )

        if requested_quantiles is not None:
            result = converter.maybe_convert_quantiles_to_level(
                result,
                models=[self.alias],
            )

        return result


__all__ = ["CITRASFM"]