import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset

from foundationforecast.core.forecaster import QuantileConverter
from foundationforecast.core.gluonts_forecaster import fix_freq
from foundationforecast.core.utils import PanelData, process_panel_from_df
from foundationforecast.models.moirai import Moirai as _Moirai

from ..utils.forecaster import Forecaster


class Moirai(_Moirai, Forecaster):
    """Moirai con soporte de exógenas conocidas en el futuro."""

    def supports_native_futr_exog(self) -> bool:
        return True

    @staticmethod
    def _normalise_dates(values) -> np.ndarray:
        return pd.to_datetime(values).to_numpy(
            dtype="datetime64[ns]"
        )

    def _forecast_native_futr_exog(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None,
        level: list[int | float] | None,
        quantiles: list[float] | None,
        panel: PanelData | None,
        horizon_df: pd.DataFrame,
        futr_exog_list: list[str],
    ) -> pd.DataFrame | None:
        if not futr_exog_list:
            return None

        required_df = {"unique_id", "ds", "y"}
        missing_df = required_df - set(df.columns)

        if missing_df:
            raise ValueError(
                f"Faltan columnas en df: {sorted(missing_df)}"
            )

        required_future = {
            "unique_id",
            "ds",
            *futr_exog_list,
        }
        missing_future = required_future - set(
            horizon_df.columns
        )
        missing_history = set(futr_exog_list) - set(
            df.columns
        )

        if missing_future or missing_history:
            raise ValueError(
                "Faltan exógenas. "
                f"En df: {sorted(missing_history)}. "
                f"En horizon_df: {sorted(missing_future)}."
            )

        df = df.copy()
        horizon_df = horizon_df.copy()

        df["ds"] = pd.to_datetime(df["ds"])
        horizon_df["ds"] = pd.to_datetime(
            horizon_df["ds"]
        )

        df = df.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        horizon_df = horizon_df.sort_values(
            ["unique_id", "ds"]
        ).reset_index(drop=True)

        if df.duplicated(["unique_id", "ds"]).any():
            raise ValueError(
                "df contiene claves unique_id/ds duplicadas."
            )

        if horizon_df.duplicated(
            ["unique_id", "ds"]
        ).any():
            raise ValueError(
                "horizon_df contiene claves "
                "unique_id/ds duplicadas."
            )

        historical_ids = set(df["unique_id"].unique())
        future_ids = set(
            horizon_df["unique_id"].unique()
        )

        if historical_ids != future_ids:
            raise ValueError(
                "df y horizon_df deben contener los "
                "mismos unique_id."
            )

        freq = self._maybe_infer_freq(df, freq)

        for column in ["y", *futr_exog_list]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

            if not np.isfinite(
                df[column].to_numpy(dtype=float)
            ).all():
                raise ValueError(
                    f"La columna histórica '{column}' "
                    f"contiene valores no finitos."
                )

        for column in futr_exog_list:
            horizon_df[column] = pd.to_numeric(
                horizon_df[column],
                errors="coerce",
            )

            if not np.isfinite(
                horizon_df[column].to_numpy(
                    dtype=float
                )
            ).all():
                raise ValueError(
                    f"La exógena futura '{column}' "
                    f"contiene valores no finitos."
                )

        combined_parts = []

        for unique_id, history in df.groupby(
            "unique_id",
            sort=False,
        ):
            future = horizon_df.loc[
                horizon_df["unique_id"] == unique_id
            ].sort_values("ds")

            if len(future) != h:
                raise ValueError(
                    f"horizon_df debe contener {h} filas "
                    f"para {unique_id}, pero contiene "
                    f"{len(future)}."
                )

            expected_dates = pd.date_range(
                start=pd.Timestamp(
                    history["ds"].iloc[-1]
                ),
                periods=h + 1,
                freq=freq,
            )[1:]

            if not np.array_equal(
                self._normalise_dates(future["ds"]),
                self._normalise_dates(expected_dates),
            ):
                raise ValueError(
                    f"Los timestamps futuros de {unique_id} "
                    f"no coinciden con los {h} pasos esperados."
                )

            history_part = history[
                [
                    "unique_id",
                    "ds",
                    "y",
                    *futr_exog_list,
                ]
            ].copy()

            future_part = future[
                [
                    "unique_id",
                    "ds",
                    *futr_exog_list,
                ]
            ].copy()

            # GluonTS eliminará estas últimas h posiciones
            # del target gracias a future_length=h.
            future_part["y"] = np.nan

            future_part = future_part[
                [
                    "unique_id",
                    "ds",
                    "y",
                    *futr_exog_list,
                ]
            ]

            combined_parts.append(
                pd.concat(
                    [history_part, future_part],
                    ignore_index=True,
                )
            )

        combined_df = pd.concat(
            combined_parts,
            ignore_index=True,
        )

        if panel is None:
            panel = process_panel_from_df(df)

        qc = QuantileConverter(
            level=level,
            quantiles=quantiles,
        )

        # Las dimensiones del predictor deben coincidir
        # con las columnas entregadas al dataset.
        self.feat_dynamic_real_dim = len(
            futr_exog_list
        )
        self.past_feat_dynamic_real_dim = 0

        gluonts_dataset = (
            PandasDataset.from_long_dataframe(
                combined_df,
                item_id="unique_id",
                timestamp="ds",
                target="y",
                feat_dynamic_real=futr_exog_list,
                past_feat_dynamic_real=None,
                future_length=h,
                freq=fix_freq(freq),
            )
        )

        with self.get_predictor(
            prediction_length=h
        ) as predictor:
            forecasts = predictor.predict(
                gluonts_dataset,
                num_samples=self.num_samples,
            )

        forecast_df = self.gluonts_fcsts_to_df(
            forecasts,
            freq=freq,
            model_name=self.alias,
            quantiles=qc.quantiles,
            h=h,
            panel=panel,
        )

        if qc.quantiles is not None:
            forecast_df = (
                qc.maybe_convert_quantiles_to_level(
                    forecast_df,
                    models=[self.alias],
                )
            )

        return forecast_df


__all__ = ["Moirai"]
