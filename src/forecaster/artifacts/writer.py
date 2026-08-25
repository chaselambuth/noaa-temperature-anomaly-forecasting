from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

# helper functions
def _utc_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")


def _jsonable(x: Any):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.ndarray,)):
        return x.tolist()
    if isinstance(x, (pd.Timestamp,)):
        return x.isoformat()
    if isinstance(x, (pd.DatetimeIndex,)):
        return [t.isoformat() for t in x.to_list()]
    return x


def _safe_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_jsonable)


def _safe_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)


def _safe_write_txt(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _get_default_out_dir() -> Path:
    return Path.cwd() / "runs"


def _df_preds_for_test(
    split: Any,
    final_pred_test: np.ndarray,
    base_test: np.ndarray,
) -> pd.DataFrame:

    targ = pd.DatetimeIndex(getattr(split, "targ_test"))
    y_true = np.asarray(getattr(split, "y_test").values, float)

    final_pred_test = np.asarray(final_pred_test, float)
    base_test = np.asarray(base_test, float)

    n = len(targ)
    if len(final_pred_test) != n:
        raise ValueError(f"final_pred_test length {len(final_pred_test)} != len(targ_test) {n}")
    if len(base_test) != n:
        raise ValueError(f"base_test length {len(base_test)} != len(targ_test) {n}")

    df = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred_final": final_pred_test,
            "y_pred_baseline": base_test,
            "err_final": final_pred_test - y_true,
            "abs_err_final": np.abs(final_pred_test - y_true),
        },
        index=targ,
    )
    df.index.name = "target_month"
    df["month_key"] = [f"M{d.month:02d}" for d in df.index]
    return df


def _df_preds_for_all(
    split: Any,
    pred_test_all: Mapping[str, np.ndarray],
) -> pd.DataFrame:

    targ = pd.DatetimeIndex(getattr(split, "targ_test"))
    n = len(targ)

    cols: dict[str, np.ndarray] = {}
    for name, values in pred_test_all.items():
        arr = np.asarray(values, float)
        if len(arr) != n:
            raise ValueError(f"{name} prediction length {len(arr)} != len(targ_test) {n}")
        cols[str(name)] = arr

    df = pd.DataFrame(cols, index=targ)
    df.index.name = "target_month"
    return df

# main writer
def write_run_artifacts(
    *,
    split: Any,
    overall_val: pd.DataFrame,
    overall_test: pd.DataFrame,
    choice_table: Optional[pd.DataFrame],
    final_pred_test: np.ndarray,
    base_test: np.ndarray,
    pred_test_all: Optional[Mapping[str, np.ndarray]] = None,
    out_dir: str | Path | None = None,
    run_name: str = "official_backtest",
) -> str:

    out_dir = Path(out_dir) if out_dir is not None else _get_default_out_dir()
    run_dir = out_dir / f"{run_name}_{_utc_run_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "n_train": int(len(getattr(split, "targ_train"))),
            "n_val": int(len(getattr(split, "targ_val"))),
            "n_test": int(len(getattr(split, "targ_test"))),
        },
        "ranges": {
            "train_target_min": str(pd.DatetimeIndex(getattr(split, "targ_train")).min()),
            "train_target_max": str(pd.DatetimeIndex(getattr(split, "targ_train")).max()),
            "val_target_min": str(pd.DatetimeIndex(getattr(split, "targ_val")).min()),
            "val_target_max": str(pd.DatetimeIndex(getattr(split, "targ_val")).max()),
            "test_target_min": str(pd.DatetimeIndex(getattr(split, "targ_test")).min()),
            "test_target_max": str(pd.DatetimeIndex(getattr(split, "targ_test")).max()),
        },
        "columns": {
            "X_train_cols": list(getattr(split, "X_train").columns),
        },
    }
    
    if is_dataclass(split):
        d = asdict(split)
        for k in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test", "w_train"]:
            if k in d:
                d.pop(k, None)
        meta["split_dataclass_fields"] = d

    _safe_write_json(run_dir / "run_meta.json", meta)

    _safe_write_csv(run_dir / "overall_val.csv", overall_val)
    _safe_write_csv(run_dir / "overall_test.csv", overall_test)

    if choice_table is not None:
        _safe_write_csv(run_dir / "month_choice_table.csv", choice_table)

    # Predictions table for the test set
    preds_test = _df_preds_for_test(split, final_pred_test, base_test)
    _safe_write_csv(run_dir / "preds_test.csv", preds_test)
    if pred_test_all is not None:
        preds_test_all = _df_preds_for_all(split, pred_test_all)
        _safe_write_csv(run_dir / "preds_test_all.csv", preds_test_all)

    # Prints a Summary of the run
    best_val = None
    if isinstance(overall_val, pd.DataFrame) and (len(overall_val) > 0) and ("Model" in overall_val.columns):
        best_val = str(overall_val.iloc[0]["Model"])

    summary_lines = [
        f"run_dir: {run_dir}",
        f"created_utc: {meta['created_utc']}",
        f"best_overall_val: {best_val}",
        f"n_train: {meta['counts']['n_train']}, n_val: {meta['counts']['n_val']}, n_test: {meta['counts']['n_test']}",
        f"test_target_range: {meta['ranges']['test_target_min']} -> {meta['ranges']['test_target_max']}",
        "",
        "Files written:",
        "- run_meta.json",
        "- overall_val.csv",
        "- overall_test.csv",
        "- month_choice_table.csv (if enabled)",
        "- preds_test.csv",
        "- preds_test_all.csv (if available)",
    ]
    _safe_write_txt(run_dir / "README.txt", "\n".join(summary_lines) + "\n")

    return str(run_dir)
