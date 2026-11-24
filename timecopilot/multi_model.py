from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable

import pandas as pd
import utilsforecast.processing as ufp

from foundationforecast.core.exog.covariates import (
    exog_strategy_disabled,
    normalize_exog_strategy,
    resolve_exog_columns_from_df,
    resolve_horizon_exog_df,
)
from foundationforecast.core.exog.futr_exog import (
    futr_exog_unsupported_message,
)
from foundationforecast.core.forecaster import (
    Forecaster,
    maybe_infer_freq,
)
from foundationforecast.core.utils import (
    PanelData,
    process_panel_from_df,
)

logger = logging.getLogger("foundationforecast.multi_model")


def _accepts_kwarg(fn: Callable[..., object], name: str) -> bool:
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return any(
        param.name == name or param.kind == inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )


def _with_panel_kwargs(
    fn: Callable[..., object],
    known_kwargs: dict[str, object],
    panel: PanelData | None,
) -> dict[str, object]:
    call_kwargs = dict(known_kwargs)
    if panel is not None and _accepts_kwarg(fn, "panel"):
        call_kwargs["panel"] = panel
    return call_kwargs


def _optional_exog_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    """Omit exog keys unless the caller supplied horizon data or column names."""
    exog_keys = ("X_df", "futr_df", "futr_exog_list")
    if not any(kwargs.get(key) is not None for key in exog_keys):
        return {k: v for k, v in kwargs.items() if k not in exog_keys}
    filtered: dict[str, object] = {
        k: v for k, v in kwargs.items() if k not in exog_keys
    }
    for key in exog_keys:
        if kwargs.get(key) is not None:
            filtered[key] = kwargs[key]
    return filtered


def _model_uses_horizon_exog(
    model: Forecaster,
    *,
    attr: str,
    df: pd.DataFrame,
    kwargs: dict[str, object],
) -> bool:
    strategy = normalize_exog_strategy(getattr(model, "exog_strategy", "auto"))
    if exog_strategy_disabled(strategy):
        return False
    if attr == "forecast":
        horizon_df = resolve_horizon_exog_df(
            kwargs.get("X_df"),  # type: ignore[arg-type]
            kwargs.get("futr_df"),  # type: ignore[arg-type]
        )
        return horizon_df is not None
    if attr == "cross_validation":
        futr_exog_list = kwargs.get("futr_exog_list")
        cols = resolve_exog_columns_from_df(
            df,
            futr_exog_list if isinstance(futr_exog_list, list) else None,
        )
        return bool(cols)
    return False


def _validate_models_horizon_exog(
    models: list[Forecaster],
    *,
    attr: str,
    df: pd.DataFrame,
    kwargs: dict[str, object],
) -> None:
    for model in models:
        if not _model_uses_horizon_exog(model, attr=attr, df=df, kwargs=kwargs):
            continue
        if not model.supports_native_futr_exog():
            raise ValueError(futr_exog_unsupported_message(model))


class MultiModelForecasterMixin:
    """Orchestrate multiple Forecaster instances through a single interface."""

    models: list[Forecaster]
    fallback_model: Forecaster | None
    clean_cache: bool
    log_model_times: bool

    def _validate_unique_aliases(self, models: list[Forecaster]) -> None:
        aliases = [model.alias for model in models]
        duplicates = {a for a in aliases if aliases.count(a) > 1}
        if duplicates:
            raise ValueError(
                f"Duplicate model aliases found: {sorted(duplicates)}. "
                "Each model must have a unique alias."
            )

    def _clean_model_cache(self) -> None:
        import gc

        models_to_clear = list(self.models)
        if self.fallback_model is not None:
            models_to_clear.append(self.fallback_model)
        for model in models_to_clear:
            clear_cache = getattr(model, "clear_model_cache", None)
            if clear_cache is not None:
                clear_cache()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _timing_enabled(self) -> bool:
        return bool(getattr(self, "log_model_times", True))

    @staticmethod
    def _safe_row_count(result: object) -> int | None:
        try:
            return len(result)  # type: ignore[arg-type]
        except TypeError:
            return None

    def _log_start(self, alias: str, operation: str) -> None:
        if not self._timing_enabled():
            return
        message = f"[START] model={alias} operation={operation}"
        print(message, flush=True)
        logger.info(message)

    def _log_end(
        self,
        alias: str,
        operation: str,
        elapsed: float,
        result: object,
        *,
        used_fallback: bool = False,
    ) -> None:
        if not self._timing_enabled():
            return
        rows = self._safe_row_count(result)
        message = (
            f"[END] model={alias} operation={operation} "
            f"seconds={elapsed:.3f} minutes={elapsed / 60.0:.3f} "
            f"rows={rows if rows is not None else 'unknown'} "
            f"fallback={used_fallback}"
        )
        print(message, flush=True)
        logger.info(message)

    def _log_error(
        self,
        alias: str,
        operation: str,
        elapsed: float,
        error: Exception,
        *,
        fallback_will_run: bool,
    ) -> None:
        if not self._timing_enabled():
            return
        message = (
            f"[ERROR] model={alias} operation={operation} "
            f"seconds={elapsed:.3f} minutes={elapsed / 60.0:.3f} "
            f"error_type={type(error).__name__} "
            f"fallback_will_run={fallback_will_run} error={error}"
        )
        print(message, flush=True)
        logger.exception(message)

    def _call_models(
        self,
        attr: str,
        merge_on: list[str],
        df: pd.DataFrame,
        h: int | None,
        freq: str | None,
        level: list[int | float] | None,
        quantiles: list[float] | None,
        panel: PanelData | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        Forecaster.validate_input(df, h)
        freq = maybe_infer_freq(df, freq)
        if panel is None and attr == "forecast":
            panel = process_panel_from_df(df)
        _validate_models_horizon_exog(
            self.models,
            attr=attr,
            df=df,
            kwargs=kwargs,
        )

        res_df: pd.DataFrame | None = None
        for model in self.models:
            alias = getattr(model, "alias", model.__class__.__name__)
            known_kwargs: dict[str, object] = {
                "df": df,
                "h": h,
                "freq": freq,
                "level": level,
            }
            if attr != "detect_anomalies":
                known_kwargs["quantiles"] = quantiles

            fn = getattr(model, attr)
            exog_kwargs = _optional_exog_kwargs(kwargs)
            call_kwargs = _with_panel_kwargs(fn, known_kwargs, panel)
            start = time.perf_counter()
            self._log_start(alias, attr)
            used_fallback = False

            try:
                res_df_model = fn(**call_kwargs, **exog_kwargs)
            except (ValueError, RuntimeError) as error:
                elapsed = time.perf_counter() - start
                self._log_error(
                    alias,
                    attr,
                    elapsed,
                    error,
                    fallback_will_run=self.fallback_model is not None,
                )
                if self.fallback_model is None:
                    raise

                used_fallback = True
                fallback_alias = getattr(
                    self.fallback_model,
                    "alias",
                    self.fallback_model.__class__.__name__,
                )
                fallback_fn = getattr(self.fallback_model, attr)
                fallback_kwargs = _with_panel_kwargs(
                    fallback_fn,
                    known_kwargs,
                    panel,
                )
                fallback_start = time.perf_counter()
                self._log_start(f"{fallback_alias} for {alias}", attr)
                try:
                    res_df_model = fallback_fn(
                        **fallback_kwargs,
                        **exog_kwargs,
                    )
                except (ValueError, RuntimeError) as fallback_error:
                    fallback_elapsed = time.perf_counter() - fallback_start
                    self._log_error(
                        f"{fallback_alias} for {alias}",
                        attr,
                        fallback_elapsed,
                        fallback_error,
                        fallback_will_run=False,
                    )
                    raise
                else:
                    fallback_elapsed = time.perf_counter() - fallback_start
                    self._log_end(
                        f"{fallback_alias} for {alias}",
                        attr,
                        fallback_elapsed,
                        res_df_model,
                        used_fallback=True,
                    )

                res_df_model = res_df_model.rename(
                    columns={
                        col: (
                            col.replace(self.fallback_model.alias, model.alias)
                            if col.startswith(self.fallback_model.alias)
                            else col
                        )
                        for col in res_df_model.columns
                    }
                )
            else:
                elapsed = time.perf_counter() - start
                self._log_end(
                    alias,
                    attr,
                    elapsed,
                    res_df_model,
                    used_fallback=False,
                )

            if used_fallback:
                total_elapsed = time.perf_counter() - start
                if self._timing_enabled():
                    message = (
                        f"[END] model={alias} operation={attr} "
                        f"seconds={total_elapsed:.3f} "
                        f"minutes={total_elapsed / 60.0:.3f} "
                        f"rows={self._safe_row_count(res_df_model)} fallback=True"
                    )
                    print(message, flush=True)
                    logger.info(message)

            if res_df is None:
                res_df = res_df_model
            else:
                if "y" in res_df_model:
                    res_df_model = res_df_model.drop(columns=["y"])
                res_df = ufp.join(res_df, res_df_model, on=merge_on, how="left")

            if self.clean_cache:
                self._clean_model_cache()

        if res_df is None:
            raise ValueError("At least one model is required.")
        return res_df

    def forecast(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        panel: PanelData | None = None,
        X_df: pd.DataFrame | None = None,
        *,
        futr_df: pd.DataFrame | None = None,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
        return self._call_models(
            "forecast",
            merge_on=["unique_id", "ds"],
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
            panel=panel,
            X_df=X_df,
            futr_df=futr_df,
            futr_exog_list=futr_exog_list,
        )

    def cross_validation(
        self,
        df: pd.DataFrame,
        h: int,
        freq: str | None = None,
        n_windows: int = 1,
        step_size: int | None = None,
        level: list[int | float] | None = None,
        quantiles: list[float] | None = None,
        *,
        futr_exog_list: list[str] | None = None,
    ) -> pd.DataFrame:
        return self._call_models(
            "cross_validation",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,
            freq=freq,
            level=level,
            quantiles=quantiles,
            n_windows=n_windows,
            step_size=step_size,
            futr_exog_list=futr_exog_list,
        )

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        h: int | None = None,
        freq: str | None = None,
        n_windows: int | None = None,
        level: int | float = 99,
    ) -> pd.DataFrame:
        return self._call_models(
            "detect_anomalies",
            merge_on=["unique_id", "ds", "cutoff"],
            df=df,
            h=h,
            freq=freq,
            level=level,  # type: ignore[arg-type]
            quantiles=None,
            n_windows=n_windows,
        )
