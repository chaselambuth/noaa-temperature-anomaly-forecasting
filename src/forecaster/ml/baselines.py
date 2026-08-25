from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

def climatology_baseline(
    target_idx: pd.DatetimeIndex,
    y_series,
    *,
    years: int = 15,
    min_count: int = 1,
    fallback: float = np.nan,
) -> np.ndarray:
    years = int(years)
    min_count = int(min_count)
    if years < 1:
        raise ValueError("years must be >= 1.")
    if min_count < 1:
        raise ValueError("min_count must be >= 1.")

    y = pd.Series(y_series).sort_index()
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()

    preds = []
    for d in pd.DatetimeIndex(target_idx).to_period("M").to_timestamp():
        past = y.loc[(y.index < d) & (y.index.month == d.month)].tail(years)
        if len(past) < min_count:
            preds.append(float(fallback))
        else:
            preds.append(float(past.mean()))
    return np.asarray(preds, dtype=float)

def baseline_blend_for(
    feature_idx: pd.DatetimeIndex,
    target_idx: pd.DatetimeIndex,
    y_anom_full,
    *,
    mode: Literal["A", "B"] | str = "A",
    persist_w: float = 0.6,
    lastyear_w: float = 0.4,
) -> np.ndarray:

    y = pd.Series(y_anom_full).sort_index()
    y.index = pd.DatetimeIndex(y.index).to_period("M").to_timestamp()

    feature_idx = pd.DatetimeIndex(feature_idx).to_period("M").to_timestamp()
    target_idx = pd.DatetimeIndex(target_idx).to_period("M").to_timestamp()

    mode = str(mode or "A").strip().upper()
    if mode not in {"A", "B"}:
        raise ValueError("mode must be 'A' or 'B'.")

    persist = y.reindex(feature_idx).to_numpy(dtype=float)
    if mode == "B":
        last_year_idx = feature_idx - pd.DateOffset(months=12)
    else:
        last_year_idx = target_idx - pd.DateOffset(months=12)
    last_year = y.reindex(last_year_idx).to_numpy(dtype=float)

    persist = np.where(np.isfinite(persist), persist, 0.0)
    last_year = np.where(np.isfinite(last_year), last_year, 0.0)
    return float(persist_w) * persist + float(lastyear_w) * last_year
