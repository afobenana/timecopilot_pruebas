import numpy as np
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

    def supports_native_futr_exog(self) -> bool:
        return True

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
            raise ValueError(f"h debe ser positivo, recibido: {h}.")

        required = {"unique_id", "ds", "y"}
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Faltan columnas obligatorias en df: {sorted(missing)}"
            )

        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"])
        df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

        if df.duplicated(["unique_id", "ds"]).any():
            raise ValueError("df tiene claves unique_id/ds duplicadas.")

        freq = self._maybe_infer_freq(df, freq)

        converter = QuantileConverter(
            level=level,
            quantiles=quantiles,
        )
        requested = converter.quantiles
        model_quantiles = (
            [float(q) for q in requested]
            if requested is not None
            else [0.5]
        )

        exog_cols: list[str] = []

        if X_df is not None:
            X_df = X_df.copy()

            missing_x = {"unique_id", "ds"} - set(X_df.columns)
            if missing_x:
                raise ValueError(
                    f"Faltan columnas obligatorias en X_df: {sorted(missing_x)}"
                )

            X_df["ds"] = pd.to_datetime(X_df["ds"])

            if "y" in X_df.columns:
                raise ValueError(
                    "X_df debe contener covariables, no la columna 'y'."
                )

            exog_cols = (
                list(futr_exog_list)
                if futr_exog_list is not None
                else [
                    col
                    for col in X_df.columns
                    if col not in {"unique_id", "ds"}
                ]
            )

            if not exog_cols:
                raise ValueError("X_df no contiene columnas exógenas.")

            if len(exog_cols) != len(set(exog_cols)):
                raise ValueError(
                    "futr_exog_list contiene columnas duplicadas."
                )

            missing_future = set(exog_cols) - set(X_df.columns)
            missing_history = set(exog_cols) - set(df.columns)

            if missing_future or missing_history:
                raise ValueError(
                    f"Faltan covariables; "
                    f"X_df: {sorted(missing_future)}, "
                    f"df: {sorted(missing_history)}"
                )

            if X_df.duplicated(["unique_id", "ds"]).any():
                raise ValueError(
                    "X_df tiene claves unique_id/ds duplicadas."
                )

            df_ids = set(df["unique_id"].unique())
            x_ids = set(X_df["unique_id"].unique())

            if df_ids != x_ids:
                raise ValueError(
                    "df y X_df deben contener los mismos unique_id. "
                    f"Faltan: {sorted(df_ids - x_ids)}. "
                    f"Adicionales: {sorted(x_ids - df_ids)}."
                )

            for col in exog_cols:
                if df[col].isna().any():
                    raise ValueError(
                        f"La exógena histórica '{col}' contiene nulos."
                    )

                if X_df[col].isna().any():
                    raise ValueError(
                        f"La exógena futura '{col}' contiene nulos."
                    )

            X_df = X_df.sort_values(
                ["unique_id", "ds"]
            ).reset_index(drop=True)

        elif futr_exog_list is not None:
            raise ValueError("futr_exog_list requiere X_df.")

        context = df[
            ["unique_id", "ds", "y", *exog_cols]
        ].rename(
            columns={
                "unique_id": "item_id",
                "ds": "timestamp",
                "y": "target",
            }
        )

        context = context.sort_values(
            ["item_id", "timestamp"]
        ).reset_index(drop=True)

        if X_df is None:
            future_parts = []

            for item_id, group in context.groupby(
                "item_id",
                sort=False,
            ):
                dates = pd.date_range(
                    start=pd.Timestamp(group["timestamp"].iloc[-1]),
                    periods=h + 1,
                    freq=freq,
                )[1:]

                future_parts.append(
                    pd.DataFrame(
                        {
                            "item_id": item_id,
                            "timestamp": dates,
                            "target": np.nan,
                        }
                    )
                )

            future = pd.concat(
                future_parts,
                ignore_index=True,
            )

        else:
            future = X_df[
                ["unique_id", "ds", *exog_cols]
            ].rename(
                columns={
                    "unique_id": "item_id",
                    "ds": "timestamp",
                }
            )

            future["target"] = np.nan

            future = future[
                ["item_id", "timestamp", "target", *exog_cols]
            ].sort_values(
                ["item_id", "timestamp"]
            ).reset_index(drop=True)

        # Validar primero todas las fechas.
        for item_id, group in context.groupby(
            "item_id",
            sort=False,
        ):
            expected_dates = pd.date_range(
                start=pd.Timestamp(group["timestamp"].iloc[-1]),
                periods=h + 1,
                freq=freq,
            )[1:].to_numpy(dtype="datetime64[ns]")

            actual_dates = pd.to_datetime(
                future.loc[
                    future["item_id"] == item_id,
                    "timestamp",
                ]
            ).to_numpy(dtype="datetime64[ns]")

            if len(actual_dates) != h:
                raise ValueError(
                    f"El horizonte de {item_id} contiene "
                    f"{len(actual_dates)} filas; se esperaban {h}."
                )

            if not np.array_equal(actual_dates, expected_dates):
                raise ValueError(
                    f"Los timestamps futuros de {item_id} "
                    f"no coinciden con los {h} pasos esperados."
                )

        # Este bloque está fuera del bucle de validación.
        # Se hace una predicción por serie.
        predictions = []

        for item_id, context_item in context.groupby(
            "item_id",
            sort=False,
        ):
            future_item = future.loc[
                future["item_id"] == item_id
            ].copy()

            context_item = context_item.reset_index(drop=True)
            future_item = future_item.reset_index(drop=True)

            prediction_item = self._forecaster.predict_df(
                context_item,
                future_df=future_item,
                quantiles=model_quantiles,
            ).reset_index()

            rename = {
                "item_id": "unique_id",
                "timestamp": "ds",
                "target": self.alias,
            }

            if requested is not None:
                rename.update(
                    {
                        float(q): (
                            f"{self.alias}-q-"
                            f"{int(round(float(q) * 100))}"
                        )
                        for q in requested
                    }
                )

            prediction_item = prediction_item.rename(columns=rename)

            if "unique_id" not in prediction_item.columns:
                prediction_item["unique_id"] = item_id

            if "ds" not in prediction_item.columns:
                raise ValueError(
                    f"TabICL no devolvió timestamps para {item_id}. "
                    f"Columnas: {prediction_item.columns.tolist()}"
                )

            if self.alias not in prediction_item.columns:
                raise ValueError(
                    f"TabICL no devolvió la predicción puntual para "
                    f"{item_id}. Columnas: "
                    f"{prediction_item.columns.tolist()}"
                )

            prediction_item["ds"] = pd.to_datetime(
                prediction_item["ds"]
            )

            expected_item = future_item[
                ["item_id", "timestamp"]
            ].rename(
                columns={
                    "item_id": "unique_id",
                    "timestamp": "ds",
                }
            )

            expected_item["ds"] = pd.to_datetime(
                expected_item["ds"]
            )

            # Si TabICL devuelve varias filas para una clave,
            # conservar la que tenga predicción puntual.
            prediction_item["_has_prediction"] = (
                prediction_item[self.alias].notna().astype(int)
            )

            prediction_item = (
                prediction_item.sort_values(
                    ["unique_id", "ds", "_has_prediction"],
                    ascending=[True, True, False],
                )
                .drop_duplicates(
                    ["unique_id", "ds"],
                    keep="first",
                )
                .drop(columns="_has_prediction")
            )

            prediction_item = expected_item.merge(
                prediction_item,
                on=["unique_id", "ds"],
                how="left",
                validate="one_to_one",
            )

            missing_prediction = prediction_item[self.alias].isna()

            if missing_prediction.any():
                missing_rows = prediction_item.loc[
                    missing_prediction,
                    ["unique_id", "ds"],
                ]

                raise ValueError(
                    f"{self.alias} no generó todas las predicciones "
                    f"para {item_id}:\n{missing_rows}"
                )

            predictions.append(prediction_item)

        result = pd.concat(
            predictions,
            ignore_index=True,
        )

        expected_rows = df["unique_id"].nunique() * h

        if len(result) != expected_rows:
            raise ValueError(
                f"{self.alias} generó {len(result)} filas, "
                f"pero se esperaban {expected_rows}."
            )

        if result.duplicated(["unique_id", "ds"]).any():
            raise ValueError(
                f"{self.alias} generó claves unique_id/ds duplicadas."
            )

        if requested is not None:
            result = converter.maybe_convert_quantiles_to_level(
                result,
                models=[self.alias],
            )

        return result


__all__ = ["TabICLv2"]