from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import pandas as pd
import numpy as np

from .config import ForecasterConfig
from .data.sources import (
    load_land_series_and_anoms,
    load_enso_monthly,
    load_teleconnections,
)
from .data.sst import load_multi_region_sst_pcs

from .ml.core import (
    build_features_and_target,
    train_predict_models,
)
from .ml.calibration import apply_qmap_to_predictions
from .ml.selection import (
    build_val_month_choice_table,
    apply_month_choice_table_to_test,
)
from .ml.evaluation import overall_table, eval_by_month
from .artifacts.writer import write_run_artifacts

# Result container
@dataclass
class RunResult:
    run_id: str
    overall_val: pd.DataFrame
    overall_test: pd.DataFrame
    per_month_test: pd.DataFrame
    month_choice_table: Optional[pd.DataFrame]
    best_overall_model: str
    used_month_choice: bool

# Utils
def _cfg_get(cfg: object, key: str, default: object = None) -> object:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _vprint(cfg: ForecasterConfig, *args: object) -> None:
    if bool(_cfg_get(cfg, "verbose", False)):
        print(*args)


def _apply_residual_deseasonalize(y_anom_full: pd.Series) -> pd.Series:
    y = pd.Series(y_anom_full, copy=True).sort_index()
    month_means = y.groupby(y.index.month).mean()
    y_adj = y - y.index.month.map(month_means)
    y_adj.name = getattr(y_anom_full, "name", "y_anom")
    return y_adj


def _assert_expected_window(
    name: str,
    idx: pd.DatetimeIndex,
    start,
    end,
    *,
    horizon: int,
) -> None:
  
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)

    if len(idx) == 0:
        raise ValueError(f"{name} split empty.")

    max_allowed_end = end - pd.DateOffset(months=horizon)

    if idx.min() > start or idx.max() < max_allowed_end:
        raise ValueError(
            f"{name} window collapsed.\n"
            f"Expected {start.date()} -> {end.date()} (H={horizon})\n"
            f"Actual   {idx.min().date()} -> {idx.max().date()}"
        )

# Baseline
def _baseline_blend_for(
    feature_idx: pd.DatetimeIndex,
    target_idx: pd.DatetimeIndex,
    y_anom_full: pd.Series,
    *,
    mode: str = "A",
    persist_w: float = 0.6,
    lastyear_w: float = 0.4,
) -> np.ndarray:

    y_anom_full = y_anom_full.sort_index()
    mode = str(mode or "A").strip().upper()

    persist = y_anom_full.reindex(feature_idx).values.astype(float)

    if mode == "B":
        last_year = y_anom_full.reindex(feature_idx - pd.DateOffset(months=12)).values
    else:
        last_year = y_anom_full.reindex(target_idx - pd.DateOffset(months=12)).values

    last_year = np.where(np.isfinite(last_year), last_year, 0.0)
    persist = np.where(np.isfinite(persist), persist, 0.0)

    return persist_w * persist + lastyear_w * last_year


def _add_baseline_candidate(
    pred_val: Dict[str, np.ndarray],
    pred_test: Dict[str, np.ndarray],
    base_val: np.ndarray,
    base_test: np.ndarray,
    name: str = "Baseline",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    pv = dict(pred_val)
    pt = dict(pred_test)
    pv[name] = np.asarray(base_val, float)
    pt[name] = np.asarray(base_test, float)
    return pv, pt

# Main pipeline
def run_backtest(cfg: ForecasterConfig) -> RunResult:

    _vprint(cfg, "[BOOT] Starting backtest")

    # 1) LAND SERIES
    y_full, y_anom_full, time_index = load_land_series_and_anoms(cfg)
    y_anom_full = pd.Series(y_anom_full, index=time_index, name="y_anom").sort_index()
    if bool(getattr(cfg, "residual_deseasonalize", False)):
        y_anom_full = _apply_residual_deseasonalize(y_anom_full)

    # 2) EXTERNAL SOURCES
    enso = load_enso_monthly(cfg.enso_csv, time_index) if getattr(cfg, "enso_csv", None) else None

    ao = nao = pna = None
    if getattr(cfg, "use_teleconnections", False):
        ao, nao, pna = load_teleconnections(time_index, cfg)

    Z_sst_multi = load_multi_region_sst_pcs(cfg, time_index) if getattr(cfg, "use_sst", False) else None

    # 3) FEATURES + SPLITS
    split = build_features_and_target(
        cfg=cfg,
        y_anom_full=y_anom_full,
        time_index=time_index,
        enso=enso,
        ao=ao,
        nao=nao,
        pna=pna,
        Z_sst_multi=Z_sst_multi,
        ar_lags=cfg.ar_lags,
    )

    _vprint(cfg, "\n===== SPLIT INFO (TARGET MONTHS) =====")
    _vprint(cfg, f"Train: {split.targ_train.min().date()} -> {split.targ_train.max().date()} (n={len(split.targ_train)})")
    _vprint(cfg, f"Val  : {split.targ_val.min().date()}   -> {split.targ_val.max().date()}   (n={len(split.targ_val)})")
    _vprint(cfg, f"Test : {split.targ_test.min().date()}  -> {split.targ_test.max().date()}  (n={len(split.targ_test)})")

    # Horizon-aware guards
    _assert_expected_window(
        "VAL",
        split.targ_val,
        cfg.val_target_start,
        cfg.val_target_end,
        horizon=cfg.horizon,
    )
    _assert_expected_window(
        "TEST",
        split.targ_test,
        cfg.test_target_start,
        cfg.test_target_end,
        horizon=cfg.horizon,
    )

    # 4) BASELINE
    base_val = _baseline_blend_for(
        split.X_val.index, split.targ_val, y_anom_full,
        mode=getattr(cfg, "baseline_mode", "A"),
    )
    base_test = _baseline_blend_for(
        split.X_test.index, split.targ_test, y_anom_full,
        mode=getattr(cfg, "baseline_mode", "A"),
    )

    # 5) MODELS
    pred_val, pred_test = train_predict_models(cfg, split)

    # 6) QMAP
    if bool(_cfg_get(cfg, "use_qmap", False)):
        pred_val, pred_test = apply_qmap_to_predictions(
            cfg=cfg,
            split=split,
            pred_val=pred_val,
            pred_test=pred_test,
        )

    # 7) ADD BASELINE
    pred_val_all, pred_test_all = _add_baseline_candidate(
        pred_val, pred_test, base_val, base_test
    )

    # 8) OVERALL METRICS
    overall_val = overall_table(split.y_val, pred_val_all, split.targ_val)
    overall_test = overall_table(split.y_test, pred_test_all, split.targ_test)

    best_overall = str(overall_val.iloc[0]["Model"])

    # 9) MONTH-CHOICE
    choice_table = None
    final_pred_test = np.asarray(pred_test_all[best_overall], float)
    used_month_choice = False

    if bool(_cfg_get(cfg, "use_month_alpha_choice", False)):
        choice_table = build_val_month_choice_table(
            y_val=split.y_val,
            targ_val=split.targ_val,
            pred_val_all=pred_val_all,
            alpha=float(getattr(cfg, "alpha", 0.67)),
            acc_tol=float(getattr(cfg, "acc_tol_for_score", 0.50)),
            tiebreak=str(getattr(cfg, "tiebreak", "mae")),
            month_group_size=int(getattr(cfg, "month_group_size", 1)),
        )

        final_pred_test = apply_month_choice_table_to_test(
            choice_table=choice_table,
            targ_test=split.targ_test,
            pred_test_all=pred_test_all,
            fallback=best_overall,
        )
        used_month_choice = True

    if used_month_choice:
        pred_test_all = dict(pred_test_all)
        pred_test_all["Ensemble_MonthChoice"] = final_pred_test
        overall_test = overall_table(split.y_test, pred_test_all, split.targ_test)

    # 10) PER-MONTH TEST
    per_month_test = eval_by_month(split.y_test, final_pred_test, split.targ_test)

    # 11) WRITE ARTIFACTS
    run_id = write_run_artifacts(
        split=split,
        overall_val=overall_val,
        overall_test=overall_test,
        choice_table=choice_table,
        final_pred_test=final_pred_test,
        base_test=base_test,
        pred_test_all=pred_test_all,
        out_dir=cfg.runs_dir,
        run_name="official_backtest",
    )

    return RunResult(
        run_id=run_id,
        overall_val=overall_val,
        overall_test=overall_test,
        per_month_test=per_month_test,
        month_choice_table=choice_table,
        best_overall_model=best_overall,
        used_month_choice=used_month_choice,
    )
