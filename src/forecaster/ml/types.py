# src/forecaster/ml/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import pandas as pd


@dataclass
class SplitData:
    X_train_df: pd.DataFrame
    X_val_df: pd.DataFrame
    X_test_df: pd.DataFrame

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    targ_train: pd.DatetimeIndex
    targ_val: pd.DatetimeIndex
    targ_test: pd.DatetimeIndex

    y_anom_full: pd.Series
    H: int = 1

    meta: Optional[Dict[str, Any]] = None


@dataclass
class TrainPredictResult:
    pred_val: Dict[str, np.ndarray]
    pred_test: Dict[str, np.ndarray]
    models: Dict[str, Any]
    info: Optional[Dict[str, Any]] = None
