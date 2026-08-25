from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso, HuberRegressor

# Hold the split data
@dataclass
class BacktestSplit:
    # Feature matrices
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame

    # Targets 
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    # Target-month indices for each split
    targ_train: pd.DatetimeIndex
    targ_val: pd.DatetimeIndex
    targ_test: pd.DatetimeIndex

    # Optional recency weights
    w_train: Optional[np.ndarray] = None

# Small helpers to convert to month-start timestamps
def _month_start_index(x: Any) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(x))
    # Normalize to month-start timestamps
    return idx.to_period("M").to_timestamp()

def _month_start_ts(x: Any) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(x)).to_period("M").to_timestamp()

def _shift_cols(df: pd.DataFrame, lags: Tuple[int, ...], suffix: str) -> pd.DataFrame:
    out = []
    for lag in lags:
        tmp = df.shift(lag).add_suffix(f"_{suffix}lag{lag}")
        out.append(tmp)
    return pd.concat(out, axis=1)

def make_recency_weights(index: pd.DatetimeIndex, half_life_months: float = 48.0) -> np.ndarray:
    # Exponentially increases weight toward more recent samples, returned weights have mean 1.0.
    n = len(index)
    age = np.arange(n, dtype=float)
    lam = np.log(2.0) / float(half_life_months)
    w = np.exp(lam * age)
    return w / np.mean(w)


def build_features_and_target(
    cfg: Any,
    *,
    y_anom_full: np.ndarray | pd.Series,
    time_index: pd.DatetimeIndex,
    enso: Optional[pd.Series] = None,
    ao: Optional[pd.Series] = None,
    nao: Optional[pd.Series] = None,
    pna: Optional[pd.Series] = None,
    Z_sst_multi: Optional[Dict[str, pd.DataFrame | np.ndarray]] = None,
    ar_lags: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
) -> BacktestSplit:
    # Rows are indexed by feature month. The supervised target is shifted forward
    # by cfg.horizon months, so every split below is still chronological in target time.
    idx = _month_start_index(time_index)

    # Y series at feature-month resolution before any horizon shift is applied.
    if isinstance(y_anom_full, pd.Series):
        y_series = y_anom_full.copy()
        y_series.index = _month_start_index(y_series.index)
        y_series = y_series.reindex(idx)
    else:
        y_series = pd.Series(np.asarray(y_anom_full, float), index=idx, name="y_anom")

    horizon = int(getattr(cfg, "horizon", 1))
    ar_lags = tuple(int(lag) for lag in (ar_lags or tuple(range(1, 13))))
    pc_lags = tuple(getattr(cfg, "pc_lags", (1, 2, 3, 6, 12)))
    y_lags = tuple(getattr(cfg, "y_lags", (1, 2, 3, 6, 12)))

    tele_lags = tuple(getattr(cfg, "tele_lags", (0, 1, 2, 3, 6, 12)))
    sst_pc_lags = tuple(getattr(cfg, "sst_pc_lags", (0, 1, 3, 6, 12)))

    X_parts: list[pd.DataFrame] = []

    # Y lags
    ylag_df = _shift_cols(y_series.to_frame("y"), y_lags, suffix="y")
    X_parts.append(ylag_df)

    # Autoregressive lags
    if ar_lags:
        ar_features = {
            f"lag_{lag}": y_series.shift(int(lag)).values for lag in ar_lags
        }
        X_parts.append(pd.DataFrame(ar_features, index=idx))

    # Seasonality features 
    m = idx.month.astype(float)
    X_parts.append(
        pd.DataFrame(
            {
                "sin_month": np.sin(2 * np.pi * m / 12.0),
                "cos_month": np.cos(2 * np.pi * m / 12.0),
            },
            index=idx,
        )
    )

    # Land PCs 
    land_pcs = getattr(cfg, "land_pcs", None)
    if land_pcs is not None:
        pcs = land_pcs.copy()
        pcs.index = _month_start_index(pcs.index)
        pcs = pcs.reindex(idx)
        X_parts.append(_shift_cols(pcs, pc_lags, suffix="pc"))

    # ENSO lag 0 means the current feature-month Nino 3.4 value; positive lags
    # add older ENSO values without duplicating lag 0.
    if enso is not None:
        s = enso.copy()
        s.index = _month_start_index(s.index)
        s = s.reindex(idx)
        X_parts.append(pd.DataFrame({"enso": s.values}, index=idx))

        enso_lags = tuple(getattr(cfg, "enso_lags", ()))
        for lag in enso_lags:
            lag = int(lag)
            if lag == 0:
                continue
            X_parts.append(pd.DataFrame({f"enso_lag{lag}": s.shift(lag).values}, index=idx))

    # Teleconnections (optional)
    def _add_tele(name: str, series: Optional[pd.Series]) -> None:
        if series is None:
            return
        s = series.copy()
        s.index = _month_start_index(s.index)
        s = s.reindex(idx)
        X_parts.append(pd.DataFrame({name: s.values}, index=idx))
        for lag in tele_lags:
            lag = int(lag)
            if lag == 0:
                continue
            X_parts.append(pd.DataFrame({f"{name}_lag{lag}": s.shift(lag).values}, index=idx))

    _add_tele("ao", ao)
    _add_tele("nao", nao)
    _add_tele("pna", pna)

    # SST PCs (optional, multi-region)
    # Accept either DataFrame indexed by month or raw ndarray aligned to idx length.
    if Z_sst_multi is not None:
        for region, Z in Z_sst_multi.items():
            if isinstance(Z, pd.DataFrame):
                df = Z.copy()
                df.index = _month_start_index(df.index)
                df = df.reindex(idx)
            else:
                arr = np.asarray(Z, float)
                if arr.shape[0] != len(idx):
                    raise ValueError(
                        f"SST region {region}: array rows {arr.shape[0]} != len(time_index) {len(idx)}"
                    )
                df = pd.DataFrame(
                    arr,
                    index=idx,
                    columns=[f"SST_{region}_PC{i+1}" for i in range(arr.shape[1])],
                )

            # lagging
            for lag in sst_pc_lags:
                lag = int(lag)
                if lag == 0:
                    X_parts.append(df)
                else:
                    tmp = df.shift(lag).add_suffix(f"_lag{lag}")
                    X_parts.append(tmp)

    # Combine all feature parts
    X_feat = pd.concat(X_parts, axis=1)

    y_target = y_series.shift(-horizon)
    y_target.name = "y_anom"

    # Drop rows with any missing values or missing target (aligned on FEATURE month)
    common = X_feat.index.intersection(y_target.dropna().index)
    X_feat = X_feat.loc[common]
    y_target = y_target.loc[common]

    valid = (~X_feat.isna().any(axis=1)) & np.isfinite(y_target.values)
    X_feat = X_feat.loc[valid]
    y_target = y_target.loc[valid]

    # Target-month timestamps corresponding to each supervised row
    feature_idx = pd.DatetimeIndex(X_feat.index)
    target_idx = _month_start_index(feature_idx + pd.DateOffset(months=horizon))

    # Official splits (by TARGET month)
    train_start = _month_start_ts(getattr(cfg, "train_target_start", "1950-01-01"))
    train_end = _month_start_ts(getattr(cfg, "train_target_end", "2010-12-01"))
    val_start = _month_start_ts(getattr(cfg, "val_target_start", "2011-01-01"))
    val_end = _month_start_ts(getattr(cfg, "val_target_end", "2018-12-01"))
    test_start = _month_start_ts(getattr(cfg, "test_target_start", "2019-01-01"))
    test_end = _month_start_ts(getattr(cfg, "test_target_end", "2025-12-01"))

    train_mask = (target_idx >= train_start) & (target_idx <= train_end)
    val_mask = (target_idx >= val_start) & (target_idx <= val_end)
    test_mask = (target_idx >= test_start) & (target_idx <= test_end)

    X_train = X_feat.loc[train_mask]
    y_train = y_target.loc[train_mask]
    X_val = X_feat.loc[val_mask]
    y_val = y_target.loc[val_mask]
    X_test = X_feat.loc[test_mask]
    y_test = y_target.loc[test_mask]

    targ_train = pd.DatetimeIndex(target_idx[train_mask])
    targ_val = pd.DatetimeIndex(target_idx[val_mask])
    targ_test = pd.DatetimeIndex(target_idx[test_mask])

    # Recency weights 
    w_train = None
    if bool(getattr(cfg, "use_recency_weights", False)):
        half_life = float(getattr(cfg, "half_life_months", 48.0))
        w_train = make_recency_weights(pd.DatetimeIndex(X_train.index), half_life_months=half_life)

    return BacktestSplit(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        targ_train=targ_train,
        targ_val=targ_val,
        targ_test=targ_test,
        w_train=w_train,
    )

# Baseline and training helpers
def baseline_predict(split: BacktestSplit, which: str = "test") -> np.ndarray:
    
    which = str(which).lower().strip()
    if which == "val":
        y = split.y_val
        targ = split.targ_val
    elif which == "train":
        y = split.y_train
        targ = split.targ_train
    else:
        y = split.y_test
        targ = split.targ_test

    # Approximate y_anom(t) using previous supervised row in FEATURE time (for H=1: y(feature t-1) == y_anom(target t))
    persist = y.shift(1).values.astype(float)

    # Approximate last-year same month in TARGET time by shifting the TARGET index (we reindex y onto (targ - 12m); since y is indexed by FEATURE month, this is only approximate)
    lastyr = y.reindex(pd.DatetimeIndex(targ) - pd.DateOffset(months=12)).values.astype(float)

    lastyr = np.where(np.isfinite(lastyr), lastyr, 0.0)
    persist = np.where(np.isfinite(persist), persist, 0.0)

    return 0.6 * persist + 0.4 * lastyr

# Train candidate models on TRAIN, predict VAL and TEST.
def train_predict_models(cfg: Any, split: BacktestSplit) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

    pred_val: Dict[str, np.ndarray] = {}
    pred_test: Dict[str, np.ndarray] = {}
    random_seed = int(getattr(cfg, "random_seed", 42))

    # Helpers for QMAP holdout
    def _month_start(di: pd.DatetimeIndex) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(di).to_period("M").to_timestamp()

    def _build_train_holdout(train_targ: pd.DatetimeIndex, cal_months: int) -> Tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
        tt = _month_start(pd.DatetimeIndex(train_targ))
        if len(tt) == 0:
            return tt, tt
        cal_months = int(max(0, cal_months))
        if cal_months <= 0:
            return tt, tt[:0]
        if cal_months >= len(tt):
            cal_months = max(0, len(tt) - 1)
        cal_targ = tt[-cal_months:] if cal_months > 0 else tt[:0]
        fit_targ = tt[:-cal_months] if cal_months > 0 else tt
        return fit_targ, cal_targ

    use_qmap = bool(getattr(cfg, "use_qmap", False))
    H = int(getattr(cfg, "H", getattr(cfg, "horizon", 1)))
    cal_months = int(getattr(cfg, "qmap_cal_target_months", 120))

    X_train_df = split.X_train
    y_train_s = split.y_train

    train_targ = pd.DatetimeIndex(split.targ_train)
    fit_targ, cal_targ = _build_train_holdout(train_targ, cal_months)

    # Map TARGET months -> FEATURE months (t = targ - H)
    fit_feat = _month_start(pd.DatetimeIndex(fit_targ) - pd.DateOffset(months=H))
    cal_feat = _month_start(pd.DatetimeIndex(cal_targ) - pd.DateOffset(months=H))

    # Reindex to preserve chronological order and make alignment deterministic
    X_fit_df = X_train_df.reindex(fit_feat)
    y_fit_s = y_train_s.reindex(fit_feat)

    X_cal_df = X_train_df.reindex(cal_feat)
    y_cal_s = y_train_s.reindex(cal_feat)

    # Build weights aligned to X_train index if provided, then subset to fit window
    w_fit: Optional[np.ndarray] = None
    if split.w_train is not None:
        w_ser = pd.Series(np.asarray(split.w_train, float), index=pd.DatetimeIndex(X_train_df.index))
        w_fit = w_ser.reindex(X_fit_df.index).values

    # QMAP requires meaningful holdout size; if not, we still do normal val/test preds.
    can_holdout = use_qmap and (X_fit_df.notna().all(axis=1).sum() >= 24) and (X_cal_df.notna().all(axis=1).sum() >= 12)

    pred_train: Dict[str, np.ndarray] = {}
    if use_qmap:
        # Create containers aligned to split.targ_train (TARGET months), fill only the holdout calibration months, everything else remains NaN.
        pass

    # Convenient numpy views for the normal full-train fit
    Xtr = X_train_df.values
    ytr = y_train_s.values
    Xva = split.X_val.values
    Xte = split.X_test.values
    w_tr = split.w_train

    # Ridge
    if bool(getattr(cfg, "use_ridge", True)):
        ridge_alpha = float(getattr(cfg, "ridge_alpha", 10.0))
        ridge_final = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=ridge_alpha, random_state=random_seed))])

        # full TRAIN fit (for VAL/TEST)
        if w_tr is None:
            ridge_final.fit(Xtr, ytr)
        else:
            ridge_final.fit(Xtr, ytr, model__sample_weight=w_tr)
        pred_val["Ridge"] = ridge_final.predict(Xva)
        pred_test["Ridge"] = ridge_final.predict(Xte)

        # holdout preds 
        if can_holdout:
            ridge_hold = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=ridge_alpha, random_state=random_seed))])
            X_fit = X_fit_df.values
            y_fit = y_fit_s.values
            X_cal = X_cal_df.values
            if w_fit is None:
                ridge_hold.fit(X_fit, y_fit)
            else:
                ridge_hold.fit(X_fit, y_fit, model__sample_weight=w_fit)
            cal_pred = ridge_hold.predict(X_cal).astype(float)

            out = np.full(len(train_targ), np.nan, dtype=float)
            cal_mask = _month_start(train_targ).isin(_month_start(cal_targ))
            out[cal_mask] = cal_pred
            pred_train["Ridge"] = out

    # ElasticNet
    if bool(getattr(cfg, "use_enet", True)):
        enet_alpha = float(getattr(cfg, "enet_alpha", 0.01))
        enet_l1 = float(getattr(cfg, "enet_l1_ratio", 0.2))

        enet_final = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1, random_state=random_seed, max_iter=20000)),
            ]
        )
        if w_tr is None:
            enet_final.fit(Xtr, ytr)
        else:
            enet_final.fit(Xtr, ytr, model__sample_weight=w_tr)
        pred_val["ElasticNet"] = enet_final.predict(Xva)
        pred_test["ElasticNet"] = enet_final.predict(Xte)

        if can_holdout:
            enet_hold = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1, random_state=random_seed, max_iter=20000)),
                ]
            )
            X_fit = X_fit_df.values
            y_fit = y_fit_s.values
            X_cal = X_cal_df.values
            if w_fit is None:
                enet_hold.fit(X_fit, y_fit)
            else:
                enet_hold.fit(X_fit, y_fit, model__sample_weight=w_fit)
            cal_pred = enet_hold.predict(X_cal).astype(float)

            out = np.full(len(train_targ), np.nan, dtype=float)
            cal_mask = _month_start(train_targ).isin(_month_start(cal_targ))
            out[cal_mask] = cal_pred
            pred_train["ElasticNet"] = out

    # XGBoost
    if bool(getattr(cfg, "use_xgb", True)):
        try:
            import xgboost as xgb

            xgb_params = dict(
                n_estimators=int(getattr(cfg, "xgb_n_estimators", 1200)),
                learning_rate=float(getattr(cfg, "xgb_learning_rate", 0.03)),
                max_depth=int(getattr(cfg, "xgb_max_depth", 4)),
                subsample=float(getattr(cfg, "xgb_subsample", 0.8)),
                colsample_bytree=float(getattr(cfg, "xgb_colsample_bytree", 0.8)),
                reg_lambda=float(getattr(cfg, "xgb_reg_lambda", 1.0)),
                objective="reg:squarederror",
                tree_method=str(getattr(cfg, "xgb_tree_method", "hist")),
                random_state=random_seed,
            )

            xgb_final = xgb.XGBRegressor(**xgb_params)
            if w_tr is None:
                xgb_final.fit(Xtr, ytr)
            else:
                xgb_final.fit(Xtr, ytr, sample_weight=w_tr)
            pred_val["XGBoost"] = xgb_final.predict(Xva)
            pred_test["XGBoost"] = xgb_final.predict(Xte)

            if can_holdout:
                xgb_hold = xgb.XGBRegressor(**xgb_params)
                X_fit = X_fit_df.values
                y_fit = y_fit_s.values
                X_cal = X_cal_df.values
                if w_fit is None:
                    xgb_hold.fit(X_fit, y_fit)
                else:
                    xgb_hold.fit(X_fit, y_fit, sample_weight=w_fit)
                cal_pred = xgb_hold.predict(X_cal).astype(float)

                out = np.full(len(train_targ), np.nan, dtype=float)
                cal_mask = _month_start(train_targ).isin(_month_start(cal_targ))
                out[cal_mask] = cal_pred
                pred_train["XGBoost"] = out
        except Exception as e:
            print(f"[WARN] XGBoost failed: {e}")

    # LightGBM
    if bool(getattr(cfg, "use_lgbm", True)):
        try:
            import lightgbm as lgb

            lgbm_params = dict(
                n_estimators=int(getattr(cfg, "lgbm_n_estimators", 2000)),
                learning_rate=float(getattr(cfg, "lgbm_learning_rate", 0.03)),
                num_leaves=int(getattr(cfg, "lgbm_num_leaves", 31)),
                max_depth=int(getattr(cfg, "lgbm_max_depth", -1)),
                subsample=float(getattr(cfg, "lgbm_subsample", 0.8)),
                colsample_bytree=float(getattr(cfg, "lgbm_colsample_bytree", 0.8)),
                reg_lambda=float(getattr(cfg, "lgbm_reg_lambda", 1.0)),
                random_state=random_seed,
                verbosity=-1,
            )

            lgb_final = lgb.LGBMRegressor(**lgbm_params)
            if w_tr is None:
                lgb_final.fit(split.X_train, ytr)
            else:
                lgb_final.fit(split.X_train, ytr, sample_weight=w_tr)
            pred_val["LightGBM"] = lgb_final.predict(split.X_val)
            pred_test["LightGBM"] = lgb_final.predict(split.X_test)

            if can_holdout:
                lgb_hold = lgb.LGBMRegressor(**lgbm_params)
                X_fit = X_fit_df
                y_fit = y_fit_s.values
                X_cal = X_cal_df
                if w_fit is None:
                    lgb_hold.fit(X_fit, y_fit)
                else:
                    lgb_hold.fit(X_fit, y_fit, sample_weight=w_fit)
                cal_pred = np.asarray(lgb_hold.predict(X_cal), float)

                out = np.full(len(train_targ), np.nan, dtype=float)
                cal_mask = _month_start(train_targ).isin(_month_start(cal_targ))
                out[cal_mask] = cal_pred
                pred_train["LightGBM"] = out
        except Exception as e:
            print(f"[WARN] LightGBM failed: {e}")

    # Lasso
    if bool(getattr(cfg, "use_lasso", True)):
        lasso_alpha = float(getattr(cfg, "lasso_alpha", 0.001))
        lasso_max_iter = int(getattr(cfg, "lasso_max_iter", 20000))

        lasso_final = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Lasso(alpha=lasso_alpha, random_state=random_seed, max_iter=lasso_max_iter)),
            ]
        )

        # full TRAIN fit (for VAL/TEST)
        if w_tr is None:
            lasso_final.fit(Xtr, ytr)
        else:
            lasso_final.fit(Xtr, ytr, model__sample_weight=w_tr)
        pred_val["Lasso"] = lasso_final.predict(Xva)
        pred_test["Lasso"] = lasso_final.predict(Xte)

        # holdout preds (for QMAP calibration)
        if can_holdout:
            lasso_hold = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Lasso(alpha=lasso_alpha, random_state=random_seed, max_iter=lasso_max_iter)),
                ]
            )
            X_fit = X_fit_df.values
            y_fit = y_fit_s.values
            X_cal = X_cal_df.values
            if w_fit is None:
                lasso_hold.fit(X_fit, y_fit)
            else:
                lasso_hold.fit(X_fit, y_fit, model__sample_weight=w_fit)
            cal_pred = lasso_hold.predict(X_cal).astype(float)

            out = np.full(len(train_targ), np.nan, dtype=float)
            cal_mask = _month_start(train_targ).isin(_month_start(cal_targ))
            out[cal_mask] = cal_pred
            pred_train["Lasso"] = out

    # Huber (robust regression)
    if bool(getattr(cfg, "use_huber", True)):
        huber_eps = float(getattr(cfg, "huber_epsilon", 1.35))
        huber_alpha = float(getattr(cfg, "huber_alpha", 0.0001))
        huber_max_iter = int(getattr(cfg, "huber_max_iter", 20000))

        huber_final = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", HuberRegressor(epsilon=huber_eps, alpha=huber_alpha, max_iter=huber_max_iter)),
            ]
        )

        if w_tr is None:
            huber_final.fit(Xtr, ytr)
        else:
            huber_final.fit(Xtr, ytr, model__sample_weight=w_tr)
        pred_val["Huber"] = huber_final.predict(Xva)
        pred_test["Huber"] = huber_final.predict(Xte)

        if can_holdout:
            huber_hold = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", HuberRegressor(epsilon=huber_eps, alpha=huber_alpha, max_iter=huber_max_iter)),
                ]
            )
            X_fit = X_fit_df.values
            y_fit = y_fit_s.values
            X_cal = X_cal_df.values
            if w_fit is None:
                huber_hold.fit(X_fit, y_fit)
            else:
                huber_hold.fit(X_fit, y_fit, model__sample_weight=w_fit)
            cal_pred = huber_hold.predict(X_cal).astype(float)

            out = np.full(len(train_targ), np.nan, dtype=float)
            cal_mask = _month_start(train_targ).isin(_month_start(cal_targ))
            out[cal_mask] = cal_pred
            pred_train["Huber"] = out


    # RandomForest 
    if bool(getattr(cfg, "use_rf", False)):
        try:
            from sklearn.ensemble import RandomForestRegressor

            max_features = getattr(cfg, "rf_max_features", None)
            # sklearn no longer accepts "auto" for RF regressor; map it to 1.0 (old behavior).
            if isinstance(max_features, str) and max_features.strip().lower() == "auto":
                max_features = 1.0

            rf_params = dict(
                n_estimators=int(getattr(cfg, "rf_n_estimators", 800)),
                max_depth=(None if getattr(cfg, "rf_max_depth", None) in (None, "None") else int(getattr(cfg, "rf_max_depth"))),
                min_samples_leaf=int(getattr(cfg, "rf_min_samples_leaf", 1)),
                min_samples_split=int(getattr(cfg, "rf_min_samples_split", 2)),
                max_features=max_features,
                n_jobs=int(getattr(cfg, "rf_n_jobs", -1)),
                random_state=random_seed,
            )

            rf_final = RandomForestRegressor(**rf_params)
            if w_tr is None:
                rf_final.fit(Xtr, ytr)
            else:
                rf_final.fit(Xtr, ytr, sample_weight=w_tr)
            pred_val["RandomForest"] = rf_final.predict(Xva)
            pred_test["RandomForest"] = rf_final.predict(Xte)

            if can_holdout:
                rf_hold = RandomForestRegressor(**rf_params)
                X_fit = X_fit_df.values
                y_fit = y_fit_s.values
                X_cal = X_cal_df.values
                if w_fit is None:
                    rf_hold.fit(X_fit, y_fit)
                else:
                    rf_hold.fit(X_fit, y_fit, sample_weight=w_fit)
                cal_pred = rf_hold.predict(X_cal).astype(float)

                out = np.full(len(train_targ), np.nan, dtype=float)
                cal_mask = _month_start(train_targ).isin(_month_start(cal_targ))
                out[cal_mask] = cal_pred
                pred_train["RandomForest"] = out
        except Exception as e:
            print(f"[WARN] RandomForest failed: {e}")

    # CatBoost 
    if bool(getattr(cfg, "use_cat", False)):
        try:
            from catboost import CatBoostRegressor

            cat_params = dict(
                iterations=int(getattr(cfg, "cat_iterations", 4000)),
                learning_rate=float(getattr(cfg, "cat_learning_rate", 0.03)),
                depth=int(getattr(cfg, "cat_depth", 6)),
                l2_leaf_reg=float(getattr(cfg, "cat_l2_leaf_reg", 3.0)),
                loss_function="RMSE",
                random_seed=random_seed,
                verbose=False,
            )

            cat_final = CatBoostRegressor(**cat_params)
            if w_tr is None:
                cat_final.fit(Xtr, ytr)
            else:
                cat_final.fit(Xtr, ytr, sample_weight=w_tr)
            pred_val["CatBoost"] = np.asarray(cat_final.predict(Xva), float)
            pred_test["CatBoost"] = np.asarray(cat_final.predict(Xte), float)

            if can_holdout:
                cat_hold = CatBoostRegressor(**cat_params)
                X_fit = X_fit_df.values
                y_fit = y_fit_s.values
                X_cal = X_cal_df.values
                if w_fit is None:
                    cat_hold.fit(X_fit, y_fit)
                else:
                    cat_hold.fit(X_fit, y_fit, sample_weight=w_fit)
                cal_pred = np.asarray(cat_hold.predict(X_cal), float)

                out = np.full(len(train_targ), np.nan, dtype=float)
                cal_mask = _month_start(train_targ).isin(_month_start(cal_targ))
                out[cal_mask] = cal_pred
                pred_train["CatBoost"] = out
        except Exception as e:
            print(f"[WARN] CatBoost failed: {e}")

    # MLP (PyTorch, optional)
    if bool(getattr(cfg, "use_mlp", False)):
        from .deep import fit_predict_mlp

        mlp_val, mlp_test = fit_predict_mlp(cfg, split)
        pred_val["MLP"] = np.asarray(mlp_val, float)
        pred_test["MLP"] = np.asarray(mlp_test, float)

    # LSTM (PyTorch, optional)
    if bool(getattr(cfg, "use_lstm", False)):
        from .deep import fit_predict_lstm

        lstm_val, lstm_test = fit_predict_lstm(cfg, split)
        pred_val["LSTM"] = np.asarray(lstm_val, float)
        pred_test["LSTM"] = np.asarray(lstm_test, float)

    # GRU (PyTorch, optional)
    if bool(getattr(cfg, "use_gru", False)):
        from .deep import fit_predict_gru

        gru_val, gru_test = fit_predict_gru(cfg, split)
        pred_val["GRU"] = np.asarray(gru_val, float)
        pred_test["GRU"] = np.asarray(gru_test, float)

    # CNN1D (PyTorch, optional)
    if bool(getattr(cfg, "use_cnn1d", False)):
        from .deep import fit_predict_cnn1d

        cnn_val, cnn_test = fit_predict_cnn1d(cfg, split)
        pred_val["CNN1D"] = np.asarray(cnn_val, float)
        pred_test["CNN1D"] = np.asarray(cnn_test, float)

    # PatchTST-style transformer (PyTorch, optional)
    if bool(getattr(cfg, "use_patchtst", False)):
        from .deep import fit_predict_patchtst

        patch_val, patch_test = fit_predict_patchtst(cfg, split)
        pred_val["PatchTST"] = np.asarray(patch_val, float)
        pred_test["PatchTST"] = np.asarray(patch_test, float)

    # iTransformer-style inverted transformer (PyTorch, optional)
    if bool(getattr(cfg, "use_itransformer", False)):
        from .deep import fit_predict_itransformer

        itransformer_val, itransformer_test = fit_predict_itransformer(cfg, split)
        pred_val["iTransformer"] = np.asarray(itransformer_val, float)
        pred_test["iTransformer"] = np.asarray(itransformer_test, float)

    # N-HiTS-style multi-rate residual network (PyTorch, optional)
    if bool(getattr(cfg, "use_nhits", False)):
        from .deep import fit_predict_nhits

        nhits_val, nhits_test = fit_predict_nhits(cfg, split)
        pred_val["NHiTS"] = np.asarray(nhits_val, float)
        pred_test["NHiTS"] = np.asarray(nhits_test, float)

    # Attach holdout predictions for QMAP if requested and available (if can_holdout and len(pred_train) > 0)
    if use_qmap:
        if can_holdout and len(pred_train) > 0:
            setattr(split, "pred_train", pred_train)
        else:
            setattr(split, "pred_train", {})

    return pred_val, pred_test
