from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

def _load_torch_predictor(function_name: str) -> Callable[[Any, Any], Tuple[np.ndarray, np.ndarray]]:
    try:
        from . import _deep_torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PyTorch is required for the selected deep-learning model. "
            "Install a working PyTorch build or choose a non-Torch model such as Ridge, Huber, or XGBoost."
        ) from exc

    return getattr(_deep_torch, function_name)


def fit_predict_mlp(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _load_torch_predictor("fit_predict_mlp")(cfg, split)


def fit_predict_lstm(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _load_torch_predictor("fit_predict_lstm")(cfg, split)


def fit_predict_gru(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _load_torch_predictor("fit_predict_gru")(cfg, split)


def fit_predict_cnn1d(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _load_torch_predictor("fit_predict_cnn1d")(cfg, split)


def fit_predict_patchtst(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _load_torch_predictor("fit_predict_patchtst")(cfg, split)


def fit_predict_itransformer(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _load_torch_predictor("fit_predict_itransformer")(cfg, split)


def fit_predict_nhits(cfg: Any, split: Any) -> Tuple[np.ndarray, np.ndarray]:
    return _load_torch_predictor("fit_predict_nhits")(cfg, split)
