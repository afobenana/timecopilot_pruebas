import numpy as np
import pandas as pd
import torch
from toto.data.util.dataset import MaskedTimeseries

from foundationforecast.models.toto import Toto as _Toto

from ..utils.forecaster import Forecaster, QuantileConverter


class Toto(_Toto, Forecaster):
    """
    Toto con soporte de exógenas futuras para Toto 1.0.

    Toto 2.0 conserva el comportamiento univariante original.
    """

    def supports_native_futr_exog(self) -> bool:
        return not self._is_toto2()

    @staticmethod
    def _to_datetime64_ns(values) -> np.ndarray:
        return pd.to_datetime(
            values
        ).to_numpy(dtype="datetime64[ns]")

    def _validate_exog_inputs(
        self,
        df: pd.DataFrame,
        X_df: pd.DataFrame,
        h: int,
        freq: str,
        futr_exog_list: list[str] | None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        list[str],
    ]:
        required_df = {
            "unique_id",
            "ds",
            "y",
        }

        missing_df = required_df - set(df.columns)

        if missing_df:
            raise ValueError(
                f"Faltan columnas obligatorias en df: "
                f"{sorted(missing_df)}"
            )

        required_future = {
            "unique_id",
            "ds",
        }

        missing_future_keys = (
            required_future - set(X_df.columns)
        )

        if missing_future_keys:
            raise ValueError(
                f"Faltan columnas obligatorias en X_df: "
                f"{sorted(missing_future_keys)}"
            )

        df = df.copy()
        X_df = X_df.copy()

        df["ds"] = pd.to_datetime(df["ds"])
        X_df["ds"] = pd.to_datetime(X_df["ds"])

        if "y" in X_df.columns:
            raise ValueError(
                "X_df no debe contener la columna objetivo 'y'."
            )

        if df.duplicated(["unique_id", "ds"]).any():
            raise ValueError(
                "df contiene claves unique_id/ds duplicadas."
            )

        if X_df.duplicated(["unique_id", "ds"]).any():
            raise ValueError(
                "X_df contiene claves unique_id/ds duplicadas."
            )

        exog_cols = (
            list(futr_exog_list)
            if futr_exog_list is not None
            else [
                column
                for column in X_df.columns
                if column not in {
                    "unique_id",
                    "ds",
                }
            ]
        )

        if not exog_cols:
            raise ValueError(
                "X_df no contiene variables exógenas."
            )

        if len(exog_cols) != len(set(exog_cols)):
            raise ValueError(
                "futr_exog_list contiene columnas duplicadas."
            )

        missing_history = (
            set(exog_cols) - set(df.columns)
        )

        missing_future = (
            set(exog_cols) - set(X_df.columns)
        )

        if missing_history or missing_future:
            raise ValueError(
                "Faltan variables exógenas. "
                f"En df: {sorted(missing_history)}. "
                f"En X_df: {sorted(missing_future)}."
            )

        historical_ids = set(
            df["unique_id"].unique()
        )

        future_ids = set(
            X_df["unique_id"].unique()
        )

        if historical_ids != future_ids:
            raise ValueError(
                "df y X_df deben contener los mismos unique_id. "
                f"Faltan en X_df: "
                f"{sorted(historical_ids - future_ids)}. "
                f"Sobran en X_df: "
                f"{sorted(future_ids - historical_ids)}."
            )

        df = df.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        X_df = X_df.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        numeric_columns = [
            "y",
            *exog_cols,
        ]

        for column in numeric_columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if not np.isfinite(
                df[column].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"La columna histórica '{column}' contiene "
                    f"nulos, infinitos o valores no numéricos."
                )

        for column in exog_cols:
            X_df[column] = pd.to_numeric(
                X_df[column],
                errors="coerce",
            )

            if not np.isfinite(
                X_df[column].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"La exógena futura '{column}' contiene "
                    f"nulos, infinitos o valores no numéricos."
                )

        for unique_id, group in df.groupby(
            "unique_id",
            sort=False,
        ):
            future_group = X_df.loc[
                X_df["unique_id"] == unique_id
            ].sort_values("ds")

            if len(future_group) != h:
                raise ValueError(
                    f"X_df debe tener exactamente {h} filas "
                    f"para {unique_id}, pero contiene "
                    f"{len(future_group)}."
                )

            expected_dates = pd.date_range(
                start=pd.Timestamp(group["ds"].iloc[-1]),
                periods=h + 1,
                freq=freq,
            )[1:]

            expected_values = self._to_datetime64_ns(
                expected_dates
            )

            actual_values = self._to_datetime64_ns(
                future_group["ds"]
            )

            if not np.array_equal(
                actual_values,
                expected_values,
            ):
                raise ValueError(
                    f"Los timestamps futuros de {unique_id} "
                    f"no coinciden con los {h} pasos esperados."
                )

        return df, X_df, exog_cols

    def _build_exog_batch(
        self,
        batch_groups: list[
            tuple[object, pd.DataFrame]
        ],
        X_df: pd.DataFrame,
        exog_cols: list[str],
        h: int,
    ) -> tuple[
        MaskedTimeseries,
        torch.Tensor,
    ]:
        batch_size = len(batch_groups)
        n_exog = len(exog_cols)
        n_variates = 1 + n_exog

        series = torch.zeros(
            (
                batch_size,
                n_variates,
                self.context_length,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        padding_mask = torch.zeros(
            (
                batch_size,
                n_variates,
                self.context_length,
            ),
            dtype=torch.bool,
            device=self.device,
        )

        id_mask = torch.zeros(
            (
                batch_size,
                n_variates,
                self.context_length,
            ),
            dtype=torch.long,
            device=self.device,
        )

        timestamp_seconds = torch.zeros(
            (
                batch_size,
                n_variates,
                self.context_length,
            ),
            dtype=torch.long,
            device=self.device,
        )

        time_interval_seconds = torch.ones(
            (
                batch_size,
                n_variates,
            ),
            dtype=torch.long,
            device=self.device,
        )

        future_exog = torch.zeros(
            (
                batch_size,
                n_exog,
                h,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        for batch_index, (
            unique_id,
            group,
        ) in enumerate(batch_groups):
            group = group.sort_values("ds")

            historical_values = group[
                [
                    "y",
                    *exog_cols,
                ]
            ].to_numpy(
                dtype=np.float32
            ).T

            historical_values = historical_values[
                :,
                -self.context_length :,
            ]

            series_length = historical_values.shape[1]

            series[
                batch_index,
                :,
                -series_length:,
            ] = torch.from_numpy(
                historical_values
            ).to(self.device)

            padding_mask[
                batch_index,
                :,
                -series_length:,
            ] = True

            timestamps = (
                group["ds"]
                .iloc[-series_length:]
                .astype("int64")
                .to_numpy()
                // 1_000_000_000
            )

            timestamp_tensor = torch.from_numpy(
                timestamps.astype(np.int64)
            ).to(self.device)

            timestamp_seconds[
                batch_index,
                :,
                -series_length:,
            ] = timestamp_tensor.unsqueeze(0).expand(
                n_variates,
                -1,
            )

            if len(timestamps) > 1:
                interval = int(
                    np.median(
                        np.diff(timestamps)
                    )
                )

                time_interval_seconds[
                    batch_index,
                    :,
                ] = max(interval, 1)

            future_group = (
                X_df.loc[
                    X_df["unique_id"] == unique_id,
                    exog_cols,
                ]
                .to_numpy(dtype=np.float32)
                .T
            )

            future_exog[
                batch_index
            ] = torch.from_numpy(
                future_group
            ).to(self.device)

        masked_timeseries = MaskedTimeseries(
            series=series,
            padding_mask=padding_mask,
            id_mask=id_mask,
            timestamp_seconds=timestamp_seconds,
            time_interval_seconds=time_interval_seconds,
            num_exogenous_variables=n_exog,
        )

        return masked_timeseries, future_exog

    def _forecast_toto1_with_exog(
        self,
        model,
        df: pd.DataFrame,
        X_df: pd.DataFrame,
        h: int,
        exog_cols: list[str],
        quantiles: list[float] | None,
    ) -> tuple[
        np.ndarray,
        np.ndarray | None,
        list[object],
    ]:
        groups = list(
            df.groupby(
                "unique_id",
                sort=False,
            )
        )

        point_batches = []
        quantile_batches = []
        ordered_ids = []

        for start in range(
            0,
            len(groups),
            self.batch_size,
        ):
            batch_groups = groups[
                start : start + self.batch_size
            ]

            inputs, future_exog = (
                self._build_exog_batch(
                    batch_groups=batch_groups,
                    X_df=X_df,
                    exog_cols=exog_cols,
                    h=h,
                )
            )

            with torch.inference_mode():
                forecast = model.forecast(
                    inputs,
                    prediction_length=h,
                    num_samples=self.num_samples,
                    samples_per_batch=self.samples_per_batch,
                    use_kv_cache=True,
                    future_exogenous_variables=future_exog,
                )

            # Forma del forecast:
            # batch x (target + exógenas) x h.
            # Solo conservamos el primer canal, que es y.
            point_batch = (
                forecast.median[
                    :,
                    0,
                    :,
                ]
                .detach()
                .cpu()
                .numpy()
            )

            point_batches.append(
                point_batch
            )

            if quantiles is not None:
                quantile_tensor = torch.tensor(
                    quantiles,
                    dtype=torch.float32,
                    device=self.device,
                )

                native_quantiles = (
                    forecast.quantile(
                        quantile_tensor
                    )
                )

                # La posición de la dimensión de cuantiles
                # depende de la versión de Toto. Normalizamos
                # a batch x h x n_quantiles.
                quantile_array = (
                    native_quantiles
                    .detach()
                    .cpu()
                    .numpy()
                )

                if quantile_array.ndim != 4:
                    raise ValueError(
                        f"Toto devolvió cuantiles con forma "
                        f"inesperada: {quantile_array.shape}."
                    )

                # Forma habitual:
                # n_quantiles x batch x variates x h.
                if quantile_array.shape[0] == len(
                    quantiles
                ):
                    quantile_array = (
                        quantile_array[
                            :,
                            :,
                            0,
                            :,
                        ]
                        .transpose(1, 2, 0)
                    )

                # Forma alternativa:
                # batch x variates x h x n_quantiles.
                elif quantile_array.shape[-1] == len(
                    quantiles
                ):
                    quantile_array = quantile_array[
                        :,
                        0,
                        :,
                        :,
                    ]

                else:
                    raise ValueError(
                        f"No se pudo identificar la dimensión "
                        f"de cuantiles en la salida de Toto: "
                        f"{quantile_array.shape}."
                    )

                quantile_batches.append(
                    quantile_array
                )

            ordered_ids.extend(
                unique_id
                for unique_id, _ in batch_groups
            )

        point_forecasts = np.concatenate(
            point_batches,
            axis=0,
        )

        if quantiles is not None:
            quantile_forecasts = np.concatenate(
                quantile_batches,
                axis=0,
            )
        else:
            quantile_forecasts = None

        return (
            point_forecasts,
            quantile_forecasts,
            ordered_ids,
        )

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
        if h < 1:
            raise ValueError(
                f"h debe ser positivo, recibido: {h}."
            )

        df_for_freq = df[
            [
                "unique_id",
                "ds",
                "y",
            ]
        ].copy()

        df_for_freq["ds"] = pd.to_datetime(
            df_for_freq["ds"]
        )

        freq = self._maybe_infer_freq(
            df_for_freq,
            freq,
        )

        # Sin X_df, usar exactamente la implementación
        # original de FoundationForecast.
        if X_df is None:
            if futr_exog_list is not None:
                raise ValueError(
                    "futr_exog_list requiere X_df."
                )

            return super()._forecast_univariate(
                df=df_for_freq,
                h=h,
                freq=freq,
                level=level,
                quantiles=quantiles,
            )

        if self._is_toto2():
            raise ValueError(
                f"{self.repo_id} es un checkpoint Toto 2.0. "
                f"Toto 2.0 todavía no admite variables "
                f"exógenas futuras. Usa "
                f"Datadog/Toto-Open-Base-1.0 o ejecuta este "
                f"modelo sin X_df."
            )

        df, X_df, exog_cols = (
            self._validate_exog_inputs(
                df=df,
                X_df=X_df,
                h=h,
                freq=freq,
                futr_exog_list=futr_exog_list,
            )
        )

        converter = QuantileConverter(
            level=level,
            quantiles=quantiles,
        )

        requested_quantiles = (
            converter.quantiles
        )

        with self._get_model() as model:
            (
                point_forecasts,
                quantile_forecasts,
                ordered_ids,
            ) = self._forecast_toto1_with_exog(
                model=model,
                df=df,
                X_df=X_df,
                h=h,
                exog_cols=exog_cols,
                quantiles=requested_quantiles,
            )

        results = []

        grouped = {
            unique_id: group
            for unique_id, group in df.groupby(
                "unique_id",
                sort=False,
            )
        }

        for row_index, unique_id in enumerate(
            ordered_ids
        ):
            group = grouped[unique_id]

            future_dates = pd.date_range(
                start=pd.Timestamp(
                    group["ds"].iloc[-1]
                ),
                periods=h + 1,
                freq=freq,
            )[1:]

            result = pd.DataFrame(
                {
                    "unique_id": unique_id,
                    "ds": future_dates,
                    self.alias: point_forecasts[
                        row_index
                    ],
                }
            )

            if (
                requested_quantiles is not None
                and quantile_forecasts is not None
            ):
                for quantile_index, quantile in enumerate(
                    requested_quantiles
                ):
                    result[
                        f"{self.alias}-q-"
                        f"{int(round(quantile * 100))}"
                    ] = quantile_forecasts[
                        row_index,
                        :,
                        quantile_index,
                    ]

            results.append(result)

        forecast_df = pd.concat(
            results,
            ignore_index=True,
        )

        expected_rows = (
            df["unique_id"].nunique() * h
        )

        if len(forecast_df) != expected_rows:
            raise ValueError(
                f"{self.alias} generó "
                f"{len(forecast_df)} filas, pero se "
                f"esperaban {expected_rows}."
            )

        if forecast_df.duplicated(
            ["unique_id", "ds"]
        ).any():
            raise ValueError(
                f"{self.alias} generó claves "
                f"unique_id/ds duplicadas."
            )

        if requested_quantiles is not None:
            forecast_df = (
                converter.maybe_convert_quantiles_to_level(
                    forecast_df,
                    models=[self.alias],
                )
            )

        return forecast_df


__all__ = ["Toto"]