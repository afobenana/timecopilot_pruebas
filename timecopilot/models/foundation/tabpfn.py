import numpy as np
import pandas as pd

from foundationforecast.models.tabpfn import (
    DEFAULT_QUANTILE_CONFIG,
    TABPFN_V2_MODEL,
    TABPFN_V3_MODEL,
    QuantileConverter,
    TabPFN as _TabPFN,
    TimeSeriesDataFrame,
    generate_test_X,
    resolve_quantile_values,
)

from ..utils.forecaster import Forecaster


class TabPFN(_TabPFN, Forecaster):
    """
    TabPFN-TS con soporte para variables exógenas conocidas
    durante el horizonte futuro.
    """

    def supports_native_futr_exog(self) -> bool:
        return True

    @staticmethod
    def _normalise_datetime(values) -> np.ndarray:
        return pd.to_datetime(
            values
        ).to_numpy(dtype="datetime64[ns]")

    def _prepare_exogenous_data(
        self,
        df: pd.DataFrame,
        X_df: pd.DataFrame | None,
        h: int,
        freq: str,
        futr_exog_list: list[str] | None,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame | None,
        list[str],
    ]:
        required = {
            "unique_id",
            "ds",
            "y",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Faltan columnas obligatorias en df: "
                f"{sorted(missing)}"
            )

        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])

        if df.duplicated(
            ["unique_id", "ds"]
        ).any():
            raise ValueError(
                "df contiene claves unique_id/ds duplicadas."
            )

        df = df.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        if X_df is None:
            if futr_exog_list is not None:
                raise ValueError(
                    "futr_exog_list requiere que se proporcione X_df."
                )

            return (
                df[["unique_id", "ds", "y"]],
                None,
                [],
            )

        X_df = X_df.copy()

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

        X_df["ds"] = pd.to_datetime(
            X_df["ds"]
        )

        if "y" in X_df.columns:
            raise ValueError(
                "X_df debe contener covariables futuras, "
                "pero no la columna objetivo 'y'."
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
                "X_df no contiene columnas exógenas."
            )

        if len(exog_cols) != len(
            set(exog_cols)
        ):
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

        if X_df.duplicated(
            ["unique_id", "ds"]
        ).any():
            raise ValueError(
                "X_df contiene claves unique_id/ds duplicadas."
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

        for column in exog_cols:
            historical_values = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            future_values = pd.to_numeric(
                X_df[column],
                errors="coerce",
            )

            if not np.isfinite(
                historical_values.to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"La exógena histórica '{column}' contiene "
                    f"nulos, infinitos o valores no numéricos."
                )

            if not np.isfinite(
                future_values.to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"La exógena futura '{column}' contiene "
                    f"nulos, infinitos o valores no numéricos."
                )

            df[column] = historical_values
            X_df[column] = future_values

        X_df = X_df.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        for unique_id, group in df.groupby(
            "unique_id",
            sort=False,
        ):
            expected_dates = pd.date_range(
                start=pd.Timestamp(
                    group["ds"].iloc[-1]
                ),
                periods=h + 1,
                freq=freq,
            )[1:]

            future_group = X_df.loc[
                X_df["unique_id"] == unique_id
            ].sort_values("ds")

            if len(future_group) != h:
                raise ValueError(
                    f"X_df debe contener exactamente {h} filas "
                    f"para {unique_id}, pero contiene "
                    f"{len(future_group)}."
                )

            expected_values = (
                self._normalise_datetime(
                    expected_dates
                )
            )

            actual_values = (
                self._normalise_datetime(
                    future_group["ds"]
                )
            )

            if not np.array_equal(
                actual_values,
                expected_values,
            ):
                raise ValueError(
                    f"Los timestamps futuros de {unique_id} "
                    f"no coinciden con los {h} pasos esperados."
                )

        return (
            df[
                [
                    "unique_id",
                    "ds",
                    "y",
                    *exog_cols,
                ]
            ],
            X_df[
                [
                    "unique_id",
                    "ds",
                    *exog_cols,
                ]
            ],
            exog_cols,
        )

    def _forecast_with_exog(
        self,
        model,
        df: pd.DataFrame,
        X_df: pd.DataFrame,
        h: int,
        quantiles: list[float] | None,
        exog_cols: list[str],
    ) -> pd.DataFrame:
        renamer = {
            "unique_id": "item_id",
            "ds": "timestamp",
            "y": "target",
        }

        context_df = df.rename(
            columns=renamer
        ).copy()

        context_df["item_id"] = (
            context_df["item_id"].astype(str)
        )

        context_tsdf = TimeSeriesDataFrame(
            context_df.set_index(
                ["item_id", "timestamp"]
            )
        )

        if self.context_length > 0:
            context_tsdf = (
                context_tsdf.slice_by_timestep(
                    -self.context_length,
                    None,
                )
            )

        # generate_test_X crea la estructura temporal correcta,
        # pero inicialmente no contiene los valores futuros
        # de las variables exógenas.
        future_tsdf = generate_test_X(
            context_tsdf,
            h,
        )

        future_exog = X_df.rename(
            columns={
                "unique_id": "item_id",
                "ds": "timestamp",
            }
        ).copy()

        future_exog["item_id"] = (
            future_exog["item_id"].astype(str)
        )

        future_exog = future_exog.set_index(
            ["item_id", "timestamp"]
        ).sort_index()

        # Normalizar el índice antes de alinear.
        future_tsdf = future_tsdf.sort_index()

        missing_future_keys = (
            future_tsdf.index.difference(
                future_exog.index
            )
        )

        extra_future_keys = (
            future_exog.index.difference(
                future_tsdf.index
            )
        )

        if (
            len(missing_future_keys) > 0
            or len(extra_future_keys) > 0
        ):
            raise ValueError(
                "Las claves futuras generadas por TabPFN "
                "no coinciden con X_df. "
                f"Faltan en X_df: "
                f"{list(missing_future_keys[:10])}. "
                f"Sobran en X_df: "
                f"{list(extra_future_keys[:10])}."
            )

        # Insertar las covariables conocidas en el futuro.
        for column in exog_cols:
            future_tsdf[column] = (
                future_exog.loc[
                    future_tsdf.index,
                    column,
                ].to_numpy()
            )

        # TabPFN-TS genera sus features después de que las
        # exógenas históricas y futuras estén colocadas.
        context_tsdf, future_tsdf = (
            self.feature_transformer.transform(
                context_tsdf,
                future_tsdf,
            )
        )

        if quantiles is None:
            forecast_df = model.predict(
                context_tsdf,
                future_tsdf,
            )
        else:
            forecast_df = model.predict(
                context_tsdf,
                future_tsdf,
                quantiles=list(
                    DEFAULT_QUANTILE_CONFIG
                ),
            )

        forecast_df = (
            forecast_df.reset_index()
        )

        output_renamer = {
            "item_id": "unique_id",
            "timestamp": "ds",
            "target": self.alias,
        }

        forecast_df = forecast_df.rename(
            columns=output_renamer
        )

        if quantiles is None:
            forecast_df = forecast_df[
                [
                    "unique_id",
                    "ds",
                    self.alias,
                ]
            ]

        else:
            native_columns = list(
                DEFAULT_QUANTILE_CONFIG
            )

            native_values = forecast_df[
                native_columns
            ].to_numpy()

            resolved_values = (
                resolve_quantile_values(
                    DEFAULT_QUANTILE_CONFIG,
                    native_values,
                    quantiles,
                )
            )

            forecast_df = forecast_df[
                [
                    "unique_id",
                    "ds",
                    self.alias,
                ]
            ]

            forecast_df = (
                self._assign_quantile_forecasts(
                    pd.DataFrame(
                        forecast_df
                    ),
                    self.alias,
                    quantiles,
                    resolved_values,
                )
            )

        forecast_df["ds"] = pd.to_datetime(
            forecast_df["ds"]
        )

        return pd.DataFrame(
            forecast_df
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
            ["unique_id", "ds", "y"]
        ].copy()

        df_for_freq["ds"] = pd.to_datetime(
            df_for_freq["ds"]
        )

        freq = self._maybe_infer_freq(
            df_for_freq,
            freq,
        )

        prepared_df, prepared_X_df, exog_cols = (
            self._prepare_exogenous_data(
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
            if prepared_X_df is None:
                # Conservamos exactamente la ruta original
                # de FoundationForecast cuando no hay exógenas.
                forecast_df = self._forecast(
                    model=model,
                    df=prepared_df,
                    h=h,
                    quantiles=requested_quantiles,
                )
            else:
                forecast_df = (
                    self._forecast_with_exog(
                        model=model,
                        df=prepared_df,
                        X_df=prepared_X_df,
                        h=h,
                        quantiles=requested_quantiles,
                        exog_cols=exog_cols,
                    )
                )

        expected_rows = (
            prepared_df["unique_id"].nunique()
            * h
        )

        if len(forecast_df) != expected_rows:
            counts = (
                forecast_df.groupby(
                    "unique_id"
                )
                .size()
                .to_dict()
            )

            raise ValueError(
                f"{self.alias} generó "
                f"{len(forecast_df)} filas, "
                f"pero se esperaban {expected_rows}. "
                f"Filas por serie: {counts}"
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


__all__ = [
    "TABPFN_V2_MODEL",
    "TABPFN_V3_MODEL",
    "TabPFN",
]
