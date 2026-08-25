from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Time Helpers
def month_start(x) -> pd.DatetimeIndex:
    dt = pd.to_datetime(x)
    return pd.DatetimeIndex(dt).to_period("M").to_timestamp()


def month_start_ts(x) -> pd.Timestamp:
    
    return pd.Timestamp(pd.to_datetime(x)).to_period("M").to_timestamp()


def season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"


def month_key(dt: pd.Timestamp) -> str:
    return f"M{dt.month:02d}"

# Metrics 
def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() == 0:
        return float("nan")
    return float(np.sqrt(mean_squared_error(y_true[good], y_pred[good])))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() == 0:
        return float("nan")
    return float(mean_absolute_error(y_true[good], y_pred[good]))


def bias(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() == 0:
        return float("nan")
    return float(np.mean(y_pred[good] - y_true[good]))


def tol_accuracy(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    good = np.isfinite(y_true) & np.isfinite(y_pred)
    if good.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_pred[good] - y_true[good]) <= float(tol)))

# Tables
def eval_by_month(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_idx: pd.DatetimeIndex,
    tols: Sequence[float] = (0.25, 0.35, 0.50),
) -> pd.DataFrame:
  
    target_idx = pd.DatetimeIndex(target_idx)
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)

    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}, index=target_idx)

    err = df["y_pred"] - df["y_true"]
    df["abs_err"] = np.abs(err)
    df["sq_err"] = err ** 2
    df["bias"] = err
    df["mkey"] = [month_key(d) for d in df.index]

    out = df.groupby("mkey").agg(
        Count=("abs_err", "count"),
        RMSE=("sq_err", lambda s: float(np.sqrt(np.mean(s)))),
        MAE=("abs_err", "mean"),
        Bias=("bias", "mean"),
    )

    for tol in tols:
        acc = (np.abs(df["y_pred"] - df["y_true"]) <= float(tol)).groupby(df["mkey"]).mean()
        out[f"Acc@{tol:.2f}"] = acc.astype(float)

    # Always order M01..M12
    return out.reindex([f"M{m:02d}" for m in range(1, 13)])


def overall_table(
    y_true: np.ndarray,
    pred_dict: Dict[str, np.ndarray],
    target_idx: Optional[pd.DatetimeIndex] = None,
    tols: Sequence[float] = (0.25, 0.35, 0.50),
    sort_by: str = "RMSE",
) -> pd.DataFrame:

    rows: List[List[object]] = []
    for name, p in pred_dict.items():
        row: List[object] = [name, rmse(y_true, p), mae(y_true, p), bias(y_true, p)]
        for t in tols:
            row.append(tol_accuracy(y_true, p, t))
        rows.append(row)

    cols = ["Model", "RMSE", "MAE", "Bias"] + [f"Acc@{t:.2f}" for t in tols]
    df = pd.DataFrame(rows, columns=cols)

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True)

    return df


def summarize_choice_run(
    y_test: np.ndarray,
    pred_test: np.ndarray,
    targ_test: pd.DatetimeIndex,
    tols: Sequence[float] = (0.25, 0.35, 0.50),
) -> Dict[str, float]:
 
    out = {
        "RMSE": rmse(y_test, pred_test),
        "MAE": mae(y_test, pred_test),
        "Bias": bias(y_test, pred_test),
    }
    for t in tols:
        out[f"Acc@{t:.2f}"] = tol_accuracy(y_test, pred_test, t)
    return out
