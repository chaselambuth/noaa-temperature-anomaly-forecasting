from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
import numpy as np
import pandas as pd
from .evaluation import month_start

# Small helpers to get the season from the month
def _season_from_month(m: int) -> str:
    if m in (12, 1, 2):
        return "DJF"
    if m in (3, 4, 5):
        return "MAM"
    if m in (6, 7, 8):
        return "JJA"
    return "SON"

# Fit the empirical quantile mapping
def _fit_empirical_qmap(pred_cal: np.ndarray, obs_cal: np.ndarray, n_q: int) -> Tuple[np.ndarray, np.ndarray]:
    pred_cal = np.asarray(pred_cal, float)
    obs_cal = np.asarray(obs_cal, float)
    good = np.isfinite(pred_cal) & np.isfinite(obs_cal)
    pred_cal = pred_cal[good]
    obs_cal = obs_cal[good]

    if pred_cal.size < 10:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    qs = np.linspace(0.0, 1.0, int(n_q))
    pred_q = np.quantile(pred_cal, qs)
    obs_q = np.quantile(obs_cal, qs)

    # Enforce monotonicity
    pred_q = np.maximum.accumulate(pred_q)
    obs_q = np.maximum.accumulate(obs_q)

    eps = 1e-12
    for i in range(1, len(pred_q)):
        if pred_q[i] <= pred_q[i - 1]:
            pred_q[i] = pred_q[i - 1] + eps

    return pred_q, obs_q

# Apply the empirical quantile mapping
def _apply_empirical_qmap(x: float, pred_q: np.ndarray, obs_q: np.ndarray, clip_to_range: bool) -> float:
    x = float(x)
    if clip_to_range:
        x = float(np.clip(x, float(pred_q[0]), float(pred_q[-1])))
    return float(np.interp(x, pred_q, obs_q))

# Build the train and holdout indices
def _build_train_holdout_indices(train_targ: pd.DatetimeIndex, cal_target_months: int) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    train_targ = pd.DatetimeIndex(train_targ).sort_values()
    n = len(train_targ)
    n_cal = int(cal_target_months)

    if n_cal <= 0 or n <= max(24, n_cal):
        n_cal = max(24, int(0.2 * n))
    n_cal = min(n_cal, n - 24)
    n_cal = max(12, n_cal)

    cal = train_targ[-n_cal:]
    fit = train_targ[:-n_cal]
    return pd.DatetimeIndex(fit), pd.DatetimeIndex(cal)

# Compute the mapping
def _compute_mapping(
    pred_cal: np.ndarray,
    obs_cal: np.ndarray,
    targ_cal: pd.DatetimeIndex,
    n_q: int,
    min_bucket: int,
    fallback_to_season: bool,
    fallback_global: bool,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    targ_cal = pd.DatetimeIndex(targ_cal)
    mapping: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    if fallback_global:
        mapping["_GLOBAL"] = _fit_empirical_qmap(pred_cal, obs_cal, n_q)

    if fallback_to_season:
        seasons = np.array([_season_from_month(m) for m in targ_cal.month], dtype=object)
        for s in ["DJF", "MAM", "JJA", "SON"]:
            mask = seasons == s
            if int(mask.sum()) >= int(min_bucket):
                mapping[s] = _fit_empirical_qmap(pred_cal[mask], obs_cal[mask], n_q)

    for m in range(1, 13):
        key = f"M{m:02d}"
        mask = targ_cal.month == m
        if int(mask.sum()) >= int(min_bucket):
            mapping[key] = _fit_empirical_qmap(pred_cal[mask], obs_cal[mask], n_q)

    return mapping

# Apply the mapping
def _apply_mapping(
    pred: np.ndarray,
    targ: pd.DatetimeIndex,
    mapping: Dict[str, Tuple[np.ndarray, np.ndarray]],
    clip_to_range: bool,
    fallback_to_season: bool,
) -> np.ndarray:
    targ = pd.DatetimeIndex(targ)
    out = np.asarray(pred, float).copy()

    for i, dt in enumerate(targ):
        mkey = f"M{dt.month:02d}"
        skey = _season_from_month(dt.month)

        if mkey in mapping:
            pred_q, obs_q = mapping[mkey]
        elif fallback_to_season and (skey in mapping):
            pred_q, obs_q = mapping[skey]
        elif "_GLOBAL" in mapping:
            pred_q, obs_q = mapping["_GLOBAL"]
        else:
            continue

        out[i] = _apply_empirical_qmap(out[i], pred_q, obs_q, clip_to_range)

    return out

# Public API used by pipeline to apply the quantile mapping to the predictions
def apply_qmap_to_predictions(
    cfg: Any,
    split: Any,
    pred_val: Dict[str, np.ndarray],
    pred_test: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

    # config
    cal_months = int(getattr(cfg, "qmap_cal_target_months", 120))
    n_q = int(getattr(cfg, "qmap_nquantiles", 201))
    min_bucket = int(getattr(cfg, "qmap_min_samples_bucket", 8))
    fallback_to_season = bool(getattr(cfg, "qmap_fallback_to_season", True))
    fallback_global = bool(getattr(cfg, "qmap_fallback_global", True))
    clip_to_range = bool(getattr(cfg, "qmap_clip_pred_to_cal_range", True))
    H = int(getattr(cfg, "H", getattr(cfg, "horizon", 1)))

    # Indices in TARGET-month space
    train_targ = pd.DatetimeIndex(split.targ_train)
    val_targ = pd.DatetimeIndex(split.targ_val)
    test_targ = pd.DatetimeIndex(split.targ_test)

    fit_targ, cal_targ = _build_train_holdout_indices(train_targ, cal_months)

    # Map target months to feature months (t = targ - H)
    fit_feat = month_start(pd.DatetimeIndex(fit_targ) - pd.DateOffset(months=H))
    cal_feat = month_start(pd.DatetimeIndex(cal_targ) - pd.DateOffset(months=H))

    # DataFrames are feature-month indexed
    X_train = split.X_train
    y_train = split.y_train

    X_fit = X_train.loc[X_train.index.intersection(fit_feat)]
    y_fit = y_train.loc[y_train.index.intersection(fit_feat)]

    X_cal = X_train.loc[X_train.index.intersection(cal_feat)]
    y_cal = y_train.loc[y_train.index.intersection(cal_feat)]

    if len(X_fit) < 24 or len(X_cal) < 12:
        print("[WARN] QMAP skipped (not enough train data for holdout).")
        return pred_val, pred_test

    # If upstream provided pred_train aligned to split.targ_train, we can use true TRAIN-holdout preds.
    pred_train: Optional[Dict[str, np.ndarray]] = getattr(split, "pred_train", None)

    if isinstance(pred_train, dict) and len(pred_train) > 0:
        train_pred_index = pd.DatetimeIndex(split.targ_train)  # target-month index for pred_train arrays
        cal_mask = train_pred_index.isin(cal_targ)

        # y_train is feature-month indexed; align to train_pred_index by shifting back H
        train_feat_index = month_start(train_pred_index - pd.DateOffset(months=H))
        obs_cal = y_train.reindex(train_feat_index).values.astype(float)[cal_mask]
        src_index = train_pred_index[cal_mask]  # target months for calibration buckets
        src_kind = "TRAIN-holdout"
    else:
        print("[WARN] QMAP skipped (no TRAIN-holdout predictions available: split.pred_train missing).")
        return pred_val, pred_test

    pred_val_out = dict(pred_val)
    pred_test_out = dict(pred_test)

    for name in list(pred_val.keys()):
        # Calibration predictions + bucket index (target months)
        if isinstance(pred_train, dict) and (name in pred_train):
            pcal = np.asarray(pred_train[name], float)[cal_mask]
            targ_cal_for_map = src_index
        else:
            pcal = np.asarray(pred_val[name], float)
            targ_cal_for_map = src_index

        mapping = _compute_mapping(
            pred_cal=pcal,
            obs_cal=obs_cal,
            targ_cal=pd.DatetimeIndex(targ_cal_for_map),
            n_q=n_q,
            min_bucket=min_bucket,
            fallback_to_season=fallback_to_season,
            fallback_global=fallback_global,
        )

        pred_val_out[name + "_qmapM"] = _apply_mapping(
            pred=np.asarray(pred_val[name], float),
            targ=val_targ,
            mapping=mapping,
            clip_to_range=clip_to_range,
            fallback_to_season=fallback_to_season,
        )
        pred_test_out[name + "_qmapM"] = _apply_mapping(
            pred=np.asarray(pred_test[name], float),
            targ=test_targ,
            mapping=mapping,
            clip_to_range=clip_to_range,
            fallback_to_season=fallback_to_season,
        )

        months_mapped = sum(1 for mm in range(1, 13) if f"M{mm:02d}" in mapping)
        print(f"[QMAP] {name}: source={src_kind}, months_mapped={months_mapped}/12")

    return pred_val_out, pred_test_out
