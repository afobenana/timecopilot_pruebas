import importlib
import sys
import types
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from gluonts.torch.model.predictor import PyTorchPredictor

from ..utils.gluonts_forecaster import GluonTSForecaster


def _ensure_legacy_gluonts_loss() -> None:
    module_name = "gluonts.torch.modules.loss"

    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name not in {module_name, "gluonts.torch.modules"}:
            raise

        module = types.ModuleType(module_name)

        # Lag-Llama importa estas clases, pero la inferencia no evalúa la loss.
        class DistributionLoss:
            pass

        class NegativeLogLikelihood(DistributionLoss):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

        DistributionLoss.__module__ = module_name
        NegativeLogLikelihood.__module__ = module_name
        module.DistributionLoss = DistributionLoss
        module.NegativeLogLikelihood = NegativeLogLikelihood
        sys.modules[module_name] = module


class LagLlama(GluonTSForecaster):
    """Lag-Llama forecaster using its upstream GluonTS implementation.

    The upstream Lag-Llama source must be importable on ``PYTHONPATH``.
    Its checkpoint is unpickled, so only use checkpoints from trusted sources.
    """

    def __init__(
        self,
        repo_id: str = "time-series-foundation-models/Lag-Llama",
        filename: str = "lag-llama.ckpt",
        alias: str = "Lag-Llama",
        num_samples: int = 100,
        use_kv_cache: bool = False,
    ) -> None:
        super().__init__(
            repo_id=repo_id,
            filename=filename,
            alias=alias,
            num_samples=num_samples,
        )
        self.use_kv_cache = use_kv_cache
        self._checkpoint: dict[str, Any] | None = None
        self._predictor: PyTorchPredictor | None = None
        self._prediction_length: int | None = None

    def load(self) -> dict[str, Any]:
        # The official checkpoint pickles GluonTS objects; weights_only=True
        # cannot load it. Only use this with the trusted Lag-Llama checkpoint.
        return torch.load(
            self.checkpoint_path,
            map_location=self.map_location,
            weights_only=False,
        )

    def _load_checkpoint(self) -> dict[str, Any]:
        if self._checkpoint is None:
            self._checkpoint = self.load()
        return self._checkpoint

    def _create_predictor(self, prediction_length: int) -> PyTorchPredictor:
        _ensure_legacy_gluonts_loss()
        from lag_llama.gluon.estimator import LagLlamaEstimator

        checkpoint = self._load_checkpoint()
        hyper_parameters = checkpoint["hyper_parameters"]
        model_kwargs = dict(hyper_parameters["model_kwargs"])
        distr_output = model_kwargs.get("distr_output")
        if distr_output.__class__.__name__ != "StudentTOutput":
            raise ValueError(
                f"Distribución del checkpoint no soportada: {type(distr_output).__name__}"
            )
        model_kwargs = dict(hyper_parameters["model_kwargs"])
        checkpoint_lags = model_kwargs.pop("lags_seq")
        model_kwargs.pop("distr_output", None)  # Se pasa explícitamente como texto
        context_length = model_kwargs.pop(
            "context_length",
            hyper_parameters.get("context_length"),
        )

        estimator = LagLlamaEstimator(
            prediction_length=prediction_length,
            context_length=context_length,
            ckpt_path=None,
            lags_seq=[],
            distr_output="studentT",
            device=torch.device(self.map_location),
            **model_kwargs,
        )
        estimator.lags_seq = checkpoint_lags

        module = estimator.create_lightning_module(
            use_kv_cache=self.use_kv_cache,
        )
        incompatible = module.load_state_dict(
            checkpoint["state_dict"],
            strict=False,
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Lag-Llama checkpoint weights do not match the model: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        module.eval()

        return estimator.create_predictor(
            estimator.create_transformation(),
            module,
        )

    @contextmanager
    def get_predictor(self, prediction_length: int) -> Iterator[PyTorchPredictor]:
        if (
            self._predictor is None
            or self._prediction_length != prediction_length
        ):
            self._predictor = self._create_predictor(prediction_length)
            self._prediction_length = prediction_length
        yield self._predictor


__all__ = ["LagLlama"]