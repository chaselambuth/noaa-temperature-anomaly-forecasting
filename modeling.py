from __future__ import annotations

import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]

from src.forecaster.config import ForecasterConfig
from src.forecaster.experiments import (
    predict_selected_model_for_choice,
    prepare_residual_problem,
    refit_selected_model,
)
from src.forecaster.pipeline import run_backtest 

# Model names / mappings
UI_TO_INTERNAL = {
    "baseline": "Baseline",
    "ridge": "Ridge",
    "elasticnet": "ElasticNet",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "randomforest": "RandomForest",
    "lasso": "Lasso",
    "huber": "Huber",
    "mlp": "MLP",
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn1d": "CNN1D",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "nhits": "NHiTS",
}

INTERNAL_TO_FLAG = {
    "Ridge": "use_ridge",
    "ElasticNet": "use_enet",
    "XGBoost": "use_xgb",
    "LightGBM": "use_lgbm",
    "CatBoost": "use_cat",
    "RandomForest": "use_rf",
    "Lasso": "use_lasso",
    "Huber": "use_huber",
    "MLP": "use_mlp",
    "LSTM": "use_lstm",
    "GRU": "use_gru",
    "CNN1D": "use_cnn1d",
    "PatchTST": "use_patchtst",
    "iTransformer": "use_itransformer",
    "NHiTS": "use_nhits",
}

ALL_MODEL_FLAGS = [
    "use_ridge",
    "use_enet",
    "use_xgb",
    "use_lgbm",
    "use_cat",
    "use_rf",
    "use_lasso",
    "use_huber",
    "use_mlp",
    "use_lstm",
    "use_gru",
    "use_cnn1d",
    "use_patchtst",
    "use_itransformer",
    "use_nhits",
]

VALID_SEASONS = {"DJF", "MAM", "JJA", "SON"}
VALID_SEASONAL_ENSEMBLE_GROUPS = {"JFM", "AMJ", "JAS", "OND"}
DEFAULT_AR_LAGS = (1, 2, 3, 6, 12, 18, 24)
DEFAULT_ENSO_LAGS = (0,)
DEFAULT_LOCAL_ENSO_CSV = Path(__file__).resolve().parent / "noaa_raw_data" / "nina34.anom.csv"
PORTABLE_DATA_DEFAULTS = {
    "tavg_file": "noaa_raw_data/nclimgrid_tavg.nc",
    "sst_dir": "noaa_raw_data/sst_cache",
    "enso_csv": "noaa_raw_data/nina34.anom.csv",
}
DEFAULT_RESIDUAL_MONTH_GROUP_MODELS = {
    "JFM": ["Lasso"],
    "AMJ": ["LSTM"],
    "JAS": ["CatBoost"],
    "OND": ["Huber"],
}
NOTEBOOK08_SELECTED_PARAMS = {
    "GRU": {"torch_seq_hidden_size": 24, "torch_seq_num_layers": 1, "torch_seq_dropout": 0.0, "torch_lr": 0.001, "torch_epochs": 150},
    "MLP": {"torch_hidden1": 64, "torch_hidden2": 32, "torch_dropout": 0.1, "torch_lr": 0.001, "torch_epochs": 200},
    "LSTM": {"torch_seq_hidden_size": 24, "torch_seq_num_layers": 1, "torch_seq_dropout": 0.0, "torch_lr": 0.001, "torch_epochs": 150},
    "CNN1D": {"torch_cnn_channels": 32, "torch_cnn_kernel_size": 5, "torch_lr": 0.0007, "torch_epochs": 200},
    "PatchTST": {"torch_patch_len": 4, "torch_patch_stride": 2, "torch_transformer_d_model": 32, "torch_transformer_nhead": 4, "torch_transformer_layers": 1, "torch_seq_dropout": 0.1, "torch_lr": 0.001, "torch_epochs": 150},
    "iTransformer": {"torch_itransformer_d_model": 32, "torch_itransformer_nhead": 4, "torch_itransformer_layers": 1, "torch_seq_dropout": 0.1, "torch_lr": 0.001, "torch_epochs": 150},
    "NHiTS": {"torch_nhits_hidden_size": 48, "torch_nhits_num_blocks": 2, "torch_nhits_pool_sizes": (1, 2, 4), "torch_dropout": 0.05, "torch_lr": 0.001, "torch_epochs": 150},
    "ElasticNet": {"enet_alpha": 0.01, "enet_l1_ratio": 0.5},
    "Lasso": {"lasso_alpha": 0.003},
    "Ridge": {"ridge_alpha": 50.0},
    "Huber": {"huber_epsilon": 1.35, "huber_alpha": 0.0001},
    "CatBoost": {"cat_iterations": 5000, "cat_learning_rate": 0.02, "cat_depth": 8, "cat_l2_leaf_reg": 5.0},
    "XGBoost": {"xgb_n_estimators": 1600, "xgb_learning_rate": 0.02, "xgb_max_depth": 5, "xgb_subsample": 0.9, "xgb_colsample_bytree": 0.9},
    "LightGBM": {"lgbm_n_estimators": 1000, "lgbm_learning_rate": 0.05, "lgbm_num_leaves": 15, "lgbm_max_depth": 4},
}


# Basic parsing helpers
def _to_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_int(x: Any, default: int) -> int:
    if x is None or x == "":
        return default
    return int(x)


def _to_float(x: Any, default: float) -> float:
    if x is None or x == "":
        return default
    return float(x)


def _to_int_tuple(x: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    if x is None:
        return tuple(default)
    if isinstance(x, (list, tuple)):
        return tuple(int(v) for v in x)
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",") if p.strip()]
        return tuple(int(v) for v in parts)
    return tuple(default)


def _portable_path_or_env(payload: dict, key: str, env_name: str) -> Any:
    value = payload.get(key)
    if value:
        portable_default = PORTABLE_DATA_DEFAULTS.get(key)
        normalized = str(value).replace("\\", "/")
        if normalized == portable_default and not Path(str(value)).expanduser().exists():
            return os.getenv(env_name) or value
        return value
    return os.getenv(env_name)


def _clean_records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or len(df) == 0:
        return []

    out = df.reset_index().copy()

    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)

    out = out.replace({np.nan: None})
    return out.to_dict(orient="records")

# Model / season helpers
def _normalize_model_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in UI_TO_INTERNAL:
        raise ValueError(f"Unsupported model name: {name}")
    return UI_TO_INTERNAL[key]

def _season_from_month(month: int) -> str:
    month = int(month)
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"

def _seasonal_ensemble_group_from_month(month: int) -> str:
    month = int(month)
    if month in (1, 2, 3):
        return "JFM"
    if month in (4, 5, 6):
        return "AMJ"
    if month in (7, 8, 9):
        return "JAS"
    return "OND"

def _normalize_model_list(models: list[str] | None) -> list[str]:
    if not models:
        return []

    out = []
    seen = set()
    for m in models:
        mm = _normalize_model_name(m)
        if mm not in seen:
            seen.add(mm)
            out.append(mm)
    return out

def _normalize_seasonal_map(seasonal_map: dict[str, str] | None) -> dict[str, str]:
    if not seasonal_map:
        return {}

    out: dict[str, str] = {}
    for season, model in seasonal_map.items():
        season_key = str(season).strip().upper()
        if season_key not in VALID_SEASONS:
            raise ValueError(f"Invalid season '{season}'. Expected one of {sorted(VALID_SEASONS)}")
        out[season_key] = _normalize_model_name(model)
    return out

def _normalize_seasonal_ensemble_map(
    seasonal_ensemble_map: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    if not seasonal_ensemble_map:
        return {}

    out: dict[str, list[str]] = {}
    for group, models in seasonal_ensemble_map.items():
        group_key = str(group).strip().upper()
        if group_key not in VALID_SEASONAL_ENSEMBLE_GROUPS:
            raise ValueError(
                f"Invalid seasonal ensemble group '{group}'. Expected one of {sorted(VALID_SEASONAL_ENSEMBLE_GROUPS)}"
            )

        normalized_models = _normalize_model_list(models or [])
        if len(normalized_models) == 0:
            raise ValueError(f"Seasonal ensemble group '{group_key}' requires at least one model")
        if len(normalized_models) > 5:
            raise ValueError(f"Seasonal ensemble group '{group_key}' supports up to 5 models")

        out[group_key] = normalized_models

    return out

def _build_model_flags(enabled_models: list[str]) -> dict[str, bool]:
    flags = {k: False for k in ALL_MODEL_FLAGS}
    for name in enabled_models:
        flag = INTERNAL_TO_FLAG.get(name)
        if flag:
            flags[flag] = True
    return flags

# Backtest selection helpers
def _extract_metric_row(df: pd.DataFrame, model_name: str) -> dict:
    if df is None or len(df) == 0:
        return {}

    if "Model" not in df.columns:
        return {}

    sub = df[df["Model"] == model_name]
    if sub.empty:
        return {}

    row = sub.iloc[0].to_dict()
    for k, v in list(row.items()):
        if isinstance(v, (np.floating, np.integer)):
            row[k] = v.item()
    return row

def _find_run_dir(run_id: str) -> Path | None:
    p = Path(run_id)
    return p if p.exists() else None


def _load_preds_test(run_id: str) -> pd.DataFrame | None:
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        return None

    preds_path = run_dir / "preds_test.csv"
    if not preds_path.exists():
        return None

    df = pd.read_csv(preds_path)
    if "target_month" in df.columns:
        df["target_month"] = pd.to_datetime(df["target_month"])
    return df

def _latest_forecast_from_preds(preds_df: pd.DataFrame | None) -> dict:
    if preds_df is None or len(preds_df) == 0:
        return {
            "target_month": None,
            "predicted_anomaly": None,
        }

    df = preds_df.copy()
    df = df.sort_values("target_month")

    last = df.iloc[-1]
    target_month = last["target_month"]
    if pd.notna(target_month):
        target_month = pd.Timestamp(target_month).strftime("%Y-%m")

    pred_value = last.get("y_pred_final", None)
    if pd.isna(pred_value):
        pred_value = None
    elif pred_value is not None:
        pred_value = float(pred_value)

    return {
        "target_month": target_month,
        "predicted_anomaly": pred_value,
    }

def _series_from_preds(preds_df: pd.DataFrame | None) -> dict | None:
    if preds_df is None or len(preds_df) == 0:
        return None

    df = preds_df.copy().sort_values("target_month")
    if "target_month" not in df.columns or "y_true" not in df.columns or "y_pred_final" not in df.columns:
        return None

    series = {
        "dates": [pd.Timestamp(dt).strftime("%Y-%m") for dt in df["target_month"]],
        "y_true": [float(v) for v in df["y_true"].astype(float)],
        "y_pred": [float(v) for v in df["y_pred_final"].astype(float)],
    }

    if "y_pred_baseline" in df.columns:
        series["y_base"] = [float(v) for v in df["y_pred_baseline"].astype(float)]

    return series

def _make_series_from_array(
    targ_test: pd.Series | pd.DatetimeIndex | list,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_base: np.ndarray | None = None,
) -> dict:
    dates = pd.DatetimeIndex(targ_test)

    series = {
        "dates": [d.strftime("%Y-%m") for d in dates],
        "y_true": [float(v) for v in np.asarray(y_true, float)],
        "y_pred": [float(v) for v in np.asarray(y_pred, float)],
    }
    if y_base is not None:
        series["y_base"] = [float(v) for v in np.asarray(y_base, float)]
    return series


def _average_selected_predictions(
    selected_models: list[str],
    targ_test: pd.DatetimeIndex,
    y_true: pd.Series,
    pred_test_df: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    missing = [m for m in selected_models if m not in pred_test_df.columns]
    if missing:
        raise ValueError(f"Missing prediction columns for selected models: {missing}")

    pred = pred_test_df[selected_models].mean(axis=1).values.astype(float)
    y_base = pred_test_df["Baseline"].values if "Baseline" in pred_test_df.columns else None
    series = _make_series_from_array(targ_test, y_true.values, pred, y_base=y_base)
    return pred, series

def _seasonal_selected_predictions(
    seasonal_models: dict[str, str],
    targ_test: pd.DatetimeIndex,
    y_true: pd.Series,
    pred_test_df: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    required = sorted(set(seasonal_models.values()))
    missing = [m for m in required if m not in pred_test_df.columns]
    if missing:
        raise ValueError(f"Missing prediction columns for seasonal model map: {missing}")

    out = []
    for dt in pd.DatetimeIndex(targ_test):
        season = _season_from_month(dt.month)
        if season not in seasonal_models:
            raise ValueError(f"Seasonal map missing season '{season}'")
        model_name = seasonal_models[season]
        out.append(float(pred_test_df.loc[dt, model_name]))

    pred = np.asarray(out, float)
    series = _make_series_from_array(targ_test, y_true.values, pred)
    return pred, series

def _seasonal_ensemble_predictions(
    seasonal_ensemble_models: dict[str, list[str]],
    targ_test: pd.DatetimeIndex,
    y_true: pd.Series,
    pred_test_df: pd.DataFrame,
) -> tuple[np.ndarray, dict]:
    required = sorted({model for models in seasonal_ensemble_models.values() for model in models})
    missing = [m for m in required if m not in pred_test_df.columns]
    if missing:
        raise ValueError(f"Missing prediction columns for seasonal ensemble model map: {missing}")

    out = []
    for dt in pd.DatetimeIndex(targ_test):
        group = _seasonal_ensemble_group_from_month(dt.month)
        if group not in seasonal_ensemble_models or len(seasonal_ensemble_models[group]) == 0:
            raise ValueError(f"Seasonal ensemble map missing group '{group}'")
        row = pred_test_df.loc[dt, seasonal_ensemble_models[group]]
        out.append(float(np.asarray(row, float).mean()))

    pred = np.asarray(out, float)
    y_base = pred_test_df["Baseline"].values if "Baseline" in pred_test_df.columns else None
    series = _make_series_from_array(targ_test, y_true.values, pred, y_base=y_base)
    return pred, series

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(y_pred - y_true))


def _acc_at_tol(y_true: np.ndarray, y_pred: np.ndarray, tol: float) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_pred - y_true) <= float(tol)))


def _build_metric_row(model_label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Model": model_label,
        "RMSE": _rmse(y_true, y_pred),
        "MAE": _mae(y_true, y_pred),
        "Bias": _bias(y_true, y_pred),
        "Acc@0.25": _acc_at_tol(y_true, y_pred, 0.25),
        "Acc@0.35": _acc_at_tol(y_true, y_pred, 0.35),
        "Acc@0.50": _acc_at_tol(y_true, y_pred, 0.50),
    }


def _load_prediction_matrix_from_run(run_id: str, overall_test: pd.DataFrame) -> pd.DataFrame | None:
    run_dir = _find_run_dir(run_id)
    if run_dir is None:
        return None

    pred_matrix_path = run_dir / "preds_test_all.csv"
    if not pred_matrix_path.exists():
        return None

    df = pd.read_csv(pred_matrix_path)
    if "target_month" in df.columns:
        df["target_month"] = pd.to_datetime(df["target_month"])
        df = df.set_index("target_month")
    return df


def _sanitize_model_group_map(seasonal_ensemble_models: dict[str, list[str]]) -> dict[str, list[str]]:
    if not seasonal_ensemble_models:
        return {k: list(v) for k, v in DEFAULT_RESIDUAL_MONTH_GROUP_MODELS.items()}
    return _normalize_seasonal_ensemble_map(seasonal_ensemble_models)


def _selected_params_for_models(model_names: list[str]) -> dict[str, dict]:
    missing = [name for name in model_names if name not in NOTEBOOK08_SELECTED_PARAMS]
    if missing:
        raise ValueError(f"No experiment-selected parameters are available for: {missing}")
    return {name: dict(NOTEBOOK08_SELECTED_PARAMS[name]) for name in model_names}

def _combine_group_predictions(
    *,
    group_models: dict[str, list[str]],
    target_idx: pd.DatetimeIndex,
    predictions_by_model: dict[str, np.ndarray],
) -> np.ndarray:
    out: list[float] = []
    for i, dt in enumerate(pd.DatetimeIndex(target_idx)):
        group = _seasonal_ensemble_group_from_month(dt.month)
        models = group_models.get(group) or []
        if not models:
            raise ValueError(f"Missing residual model selection for month group '{group}'")
        missing = [m for m in models if m not in predictions_by_model]
        if missing:
            raise ValueError(f"Missing residual predictions for {group}: {missing}")
        out.append(float(np.mean([predictions_by_model[m][i] for m in models])))
    return np.asarray(out, float)

def _write_residual_artifacts(
    *,
    runs_dir: str,
    target_idx: pd.DatetimeIndex,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_base: np.ndarray,
    pred_test_all: dict[str, np.ndarray],
    overall_test: pd.DataFrame,
    summary: dict,
) -> str:
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_utc")
    run_dir = Path(runs_dir) / f"residual_climatology_app_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    preds_df = pd.DataFrame(
        {
            "target_month": pd.DatetimeIndex(target_idx),
            "y_true": np.asarray(y_true, float),
            "y_pred_final": np.asarray(y_pred, float),
            "y_pred_baseline": np.asarray(y_base, float),
            "err_final": np.asarray(y_pred, float) - np.asarray(y_true, float),
            "abs_err_final": np.abs(np.asarray(y_pred, float) - np.asarray(y_true, float)),
        }
    )
    preds_df["month_key"] = [f"M{d.month:02d}" for d in pd.DatetimeIndex(preds_df["target_month"])]
    preds_df.to_csv(run_dir / "preds_test.csv", index=False)

    pred_all_df = pd.DataFrame({"target_month": pd.DatetimeIndex(target_idx)})
    for name, values in pred_test_all.items():
        pred_all_df[name] = np.asarray(values, float)
    pred_all_df.to_csv(run_dir / "preds_test_all.csv", index=False)
    overall_test.to_csv(run_dir / "overall_test.csv", index=False)
    pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

    return str(run_dir)

def _run_residual_climatology_app(
    *,
    payload: dict,
    horizon: int,
    tavg_file: str,
    sst_dir: str,
    enso_csv: str | None,
    runs_dir: str,
    seasonal_ensemble_models: dict[str, list[str]],
) -> dict:
    mode = str(payload.get("mode", "single")).strip().lower()
    climatology_years = _to_int(payload.get("climatology_years"), 18)
    deep_internal_val_months = _to_int(payload.get("deep_internal_val_months"), 24)
    group_order = ["JFM", "AMJ", "JAS", "OND"]

    if mode == "single":
        selected_model = _normalize_model_name(payload.get("model_name", "lstm"))
        group_models = {group: [selected_model] for group in group_order}
        strategy_kind = "single"
    elif mode == "ensemble":
        selected_models = _normalize_model_list(payload.get("ensemble_models", []))
        if not selected_models:
            raise ValueError("Residual ensemble mode requires at least one model.")
        group_models = {group: list(selected_models) for group in group_order}
        strategy_kind = "ensemble"
    else:
        group_models = _sanitize_model_group_map(seasonal_ensemble_models)
        strategy_kind = "month_group"

    enabled_models: list[str] = []
    for models in group_models.values():
        for model in models:
            if model not in enabled_models:
                enabled_models.append(model)
    selected_params = _selected_params_for_models(enabled_models)

    ar_lags = _to_int_tuple(payload.get("ar_lags"), DEFAULT_AR_LAGS)
    enso_lags = _to_int_tuple(payload.get("enso_lags"), DEFAULT_ENSO_LAGS)

    base_cfg = {
        "tavg_file": str(tavg_file),
        "sst_dir": str(sst_dir),
        "enso_csv": str(enso_csv) if enso_csv else None,
        "H": int(horizon),
        "K": _to_int(payload.get("K"), 10),
        "pc_lags": _to_int_tuple(payload.get("pc_lags"), (1, 2, 3, 6, 12)),
        "y_lags": _to_int_tuple(payload.get("y_lags"), (1, 2, 3, 6, 12)),
        "ar_lags": ar_lags,
        "enso_lags": enso_lags,
        "sst_pc_lags": _to_int_tuple(payload.get("sst_pc_lags"), (0, 1, 3, 6, 12)),
        "use_sst": _to_bool(payload.get("use_sst"), False),
        "use_teleconnections": _to_bool(payload.get("use_teleconnections"), False),
        "use_qmap": _to_bool(payload.get("use_qmap"), False),
        "use_month_alpha_choice": True,
        "month_group_size": _to_int(payload.get("month_group_size"), 3),
        "random_seed": _to_int(payload.get("random_seed"), 42),
        "use_recency_weights": True,
        "half_life_months": _to_float(payload.get("half_life_months"), 48.0),
        "residual_deseasonalize": _to_bool(payload.get("residual_deseasonalize"), False),
        "baseline_mode": str(payload.get("baseline_mode", "A")),
        "plot_test": False,
        "verbose": False,
    }

    problem = prepare_residual_problem(
        base_cfg,
        climatology_years=climatology_years,
        ar_lags=ar_lags,
        include_teleconnections=bool(base_cfg["use_teleconnections"]),
        include_sst=bool(base_cfg["use_sst"]),
    )

    pred_val_by_model: dict[str, np.ndarray] = {}
    pred_test_by_model: dict[str, np.ndarray] = {}
    per_model_rows: list[dict] = []

    for model_name in enabled_models:
        params = selected_params[model_name]
        choice_result = predict_selected_model_for_choice(problem, model_name, params, base_cfg=base_cfg)
        refit_result = refit_selected_model(
            problem,
            model_name,
            params,
            base_cfg=base_cfg,
            deep_internal_val_months=deep_internal_val_months,
        )
        pred_val_by_model[model_name] = np.asarray(choice_result["pred_val"], float)
        pred_test_by_model[model_name] = np.asarray(refit_result["pred_test"], float)
        per_model_rows.append(_build_metric_row(f"Residual_{model_name}", problem["split_full"].y_test.values, pred_test_by_model[model_name]))

    targ_val = pd.DatetimeIndex(problem["split_full"].targ_val)
    targ_test = pd.DatetimeIndex(problem["split_full"].targ_test)
    y_val_true = problem["split_full"].y_val.values.astype(float)
    y_test_true = problem["split_full"].y_test.values.astype(float)

    y_val_pred = _combine_group_predictions(
        group_models=group_models,
        target_idx=targ_val,
        predictions_by_model=pred_val_by_model,
    )
    y_test_pred = _combine_group_predictions(
        group_models=group_models,
        target_idx=targ_test,
        predictions_by_model=pred_test_by_model,
    )

    if strategy_kind == "single" and len(enabled_models) == 1:
        strategy_label = f"Climatology{climatology_years}_Residual_{enabled_models[0]}"
    elif strategy_kind == "ensemble":
        strategy_label = f"Climatology{climatology_years}_ResidualEnsemble"
    else:
        strategy_label = f"Climatology{climatology_years}_ResidualMonthGroup"
    test_metrics = _build_metric_row(strategy_label, y_test_true, y_test_pred)
    val_metrics = _build_metric_row(strategy_label, y_val_true, y_val_pred)
    climatology_metrics = _build_metric_row(f"Climatology{climatology_years}", y_test_true, problem["clim_test"])

    overall_test = pd.DataFrame([test_metrics, climatology_metrics, *per_model_rows]).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    pred_test_all = {f"Residual_{name}": values for name, values in pred_test_by_model.items()}
    pred_test_all[f"Climatology{climatology_years}"] = np.asarray(problem["clim_test"], float)
    pred_test_all[strategy_label] = y_test_pred

    summary = {
        "climatology_years": climatology_years,
        "strategy": strategy_label,
        "month_group_models": "; ".join(f"{group}:{'/'.join(models)}" for group, models in group_models.items()),
        "horizon": horizon,
    }
    run_id = _write_residual_artifacts(
        runs_dir=str(runs_dir),
        target_idx=targ_test,
        y_true=y_test_true,
        y_pred=y_test_pred,
        y_base=np.asarray(problem["clim_test"], float),
        pred_test_all=pred_test_all,
        overall_test=overall_test,
        summary=summary,
    )

    series = _make_series_from_array(targ_test, y_test_true, y_test_pred, y_base=np.asarray(problem["clim_test"], float))
    series["baseline_label"] = f"{climatology_years}-year climatology"

    latest_forecast = {
        "target_month": targ_test[-1].strftime("%Y-%m") if len(targ_test) else None,
        "predicted_anomaly": float(y_test_pred[-1]) if len(y_test_pred) else None,
    }

    return {
        "ok": True,
        "mode": "climatology_residual",
        "horizon": horizon,
        "selected_models": [f"Residual_{name}" for name in enabled_models],
        "seasonal_ensemble_models": group_models,
        "enabled_models": enabled_models,
        "config": {
            "tavg_file": str(tavg_file),
            "sst_dir": str(sst_dir),
            "use_enso": bool(enso_csv),
            "enso_csv": str(enso_csv) if enso_csv else None,
            "enso_lags": list(enso_lags),
            "ar_lags": list(ar_lags),
            "runs_dir": str(runs_dir),
            "use_sst": bool(base_cfg["use_sst"]),
            "use_teleconnections": bool(base_cfg["use_teleconnections"]),
            "use_qmap": bool(base_cfg["use_qmap"]),
            "use_climatology_residual": True,
            "climatology_years": climatology_years,
            "random_seed": int(base_cfg["random_seed"]),
            "half_life_months": float(base_cfg["half_life_months"]),
        },
        "summary": {
            "run_id": run_id,
            "best_overall_model": strategy_label,
            "used_month_choice": strategy_kind == "month_group",
            "climatology_years": climatology_years,
            "month_group_models": summary["month_group_models"],
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "latest_forecast": latest_forecast,
        "series": series,
        "overall_test_table": _clean_records(overall_test),
        "overall_val_table": [],
        "per_month_test_table": [],
    }

# Main service
def run_forecast_app(payload: dict | None = None) -> dict:
    payload = payload or {}

    mode = str(payload.get("mode", "single")).strip().lower()
    if mode not in {"single", "ensemble", "seasonal", "seasonal_ensemble"}:
        return {"ok": False, "error": "mode must be 'single', 'ensemble', 'seasonal', or 'seasonal_ensemble'"}

    horizon = _to_int(payload.get("horizon", payload.get("H")), 1)

    single_model = payload.get("model_name", "mlp")
    ensemble_models = payload.get("ensemble_models", [])
    seasonal_map = payload.get("seasonal_models", {})
    seasonal_ensemble_map = payload.get("seasonal_ensemble_models", {})

    selected_ensemble: list[str] = []
    seasonal_models: dict[str, str] = {}
    seasonal_ensemble_models: dict[str, list[str]] = {}

    try:
        selected_single = _normalize_model_name(single_model)
        if mode == "ensemble":
            selected_ensemble = _normalize_model_list(ensemble_models)
        elif mode == "seasonal":
            seasonal_models = _normalize_seasonal_map(seasonal_map)
        elif mode == "seasonal_ensemble":
            seasonal_ensemble_models = _normalize_seasonal_ensemble_map(seasonal_ensemble_map)
    except Exception as e:
        return {"ok": False, "error": str(e)}

    if mode == "single":
        enabled_models = [selected_single]
    elif mode == "ensemble":
        if len(selected_ensemble) == 0:
            return {"ok": False, "error": "ensemble mode requires at least 1 model"}
        if len(selected_ensemble) > 5:
            return {"ok": False, "error": "ensemble mode supports up to 5 models"}
        enabled_models = list(selected_ensemble)
    elif mode == "seasonal":
        if len(seasonal_models) == 0:
            return {"ok": False, "error": "seasonal mode requires at least one seasonal model mapping"}
        enabled_models = list(seasonal_models.values())
    else:
        missing_groups = sorted(VALID_SEASONAL_ENSEMBLE_GROUPS.difference(seasonal_ensemble_models.keys()))
        if missing_groups:
            return {"ok": False, "error": f"seasonal_ensemble mode requires all groups: {missing_groups}"}
        enabled_models = []
        for models in seasonal_ensemble_models.values():
            for model_name in models:
                if model_name not in enabled_models:
                    enabled_models.append(model_name)

    # Include any season-specific models so the pipeline actually trains them.
    for m in seasonal_models.values():
        if m not in enabled_models:
            enabled_models.append(m)
    for models in seasonal_ensemble_models.values():
        for m in models:
            if m not in enabled_models:
                enabled_models.append(m)

    tavg_file = _portable_path_or_env(payload, "tavg_file", "TAVG_FILE")
    sst_dir = _portable_path_or_env(payload, "sst_dir", "SST_DIR")
    enso_csv_value = _portable_path_or_env(payload, "enso_csv", "ENSO_CSV")
    if not enso_csv_value and DEFAULT_LOCAL_ENSO_CSV.exists():
        enso_csv_value = str(DEFAULT_LOCAL_ENSO_CSV)
    use_enso = _to_bool(payload.get("use_enso"), bool(enso_csv_value))
    enso_csv = enso_csv_value if use_enso else None
    runs_dir = payload.get("runs_dir") or os.getenv("RUNS_DIR", "runs")

    if not tavg_file:
        return {"ok": False, "error": "Missing required setting: tavg_file"}
    if not sst_dir:
        return {"ok": False, "error": "Missing required setting: sst_dir"}

    if _to_bool(payload.get("use_climatology_residual"), False):
        try:
            return _run_residual_climatology_app(
                payload=payload,
                horizon=horizon,
                tavg_file=str(tavg_file),
                sst_dir=str(sst_dir),
                enso_csv=str(enso_csv) if enso_csv else None,
                runs_dir=str(runs_dir),
                seasonal_ensemble_models=seasonal_ensemble_models,
            )
        except Exception as e:
            message = str(e).strip() or e.__class__.__name__
            return {
                "ok": False,
                "error": f"{e.__class__.__name__}: {message}",
                "traceback": traceback.format_exc(),
            }

    flags = _build_model_flags(enabled_models)

    use_month_alpha_choice = False

    cfg = ForecasterConfig(
        # Required paths
        tavg_file=str(tavg_file),
        sst_dir=str(sst_dir),
        enso_csv=str(enso_csv) if enso_csv else None,

        # Forecast settings
        H=horizon,
        K=_to_int(payload.get("K"), 10),
        pc_lags=_to_int_tuple(payload.get("pc_lags"), (1, 2, 3, 6, 12)),
        y_lags=_to_int_tuple(payload.get("y_lags"), (1, 2, 3, 6, 12)),
        ar_lags=_to_int_tuple(payload.get("ar_lags"), DEFAULT_AR_LAGS),
        enso_lags=_to_int_tuple(payload.get("enso_lags"), DEFAULT_ENSO_LAGS),
        tele_lags=_to_int_tuple(payload.get("tele_lags"), (0, 1, 2, 3, 6, 12)),
        sst_pc_lags=_to_int_tuple(payload.get("sst_pc_lags"), (0, 1, 3, 6, 12)),

        # Sources
        use_sst=_to_bool(payload.get("use_sst"), False),
        use_teleconnections=_to_bool(payload.get("use_teleconnections"), False),

        # Preprocessing / calibration
        use_qmap=_to_bool(payload.get("use_qmap"), False),
        use_month_alpha_choice=use_month_alpha_choice,
        alpha=_to_float(payload.get("alpha"), 0.67),
        acc_tol_for_score=_to_float(payload.get("acc_tol_for_score"), 0.50),
        month_group_size=_to_int(payload.get("month_group_size"), 1),
        residual_deseasonalize=_to_bool(payload.get("residual_deseasonalize"), False),
        baseline_mode=str(payload.get("baseline_mode", "A")),
        baseline_persist_w=_to_float(payload.get("baseline_persist_w"), 0.6),
        baseline_lastyear_w=_to_float(payload.get("baseline_lastyear_w"), 0.4),
        random_seed=_to_int(payload.get("random_seed"), 42),
        half_life_months=_to_float(payload.get("half_life_months"), 48.0),

        # Enabled models
        use_ridge=flags["use_ridge"],
        use_enet=flags["use_enet"],
        use_xgb=flags["use_xgb"],
        use_lgbm=flags["use_lgbm"],
        use_cat=flags["use_cat"],
        use_rf=flags["use_rf"],
        use_lasso=flags["use_lasso"],
        use_huber=flags["use_huber"],
        use_mlp=flags["use_mlp"],
        use_lstm=flags["use_lstm"],
        use_gru=flags["use_gru"],
        use_cnn1d=flags["use_cnn1d"],
        use_patchtst=flags["use_patchtst"],
        use_itransformer=flags["use_itransformer"],
        use_nhits=flags["use_nhits"],

        # XGBoost params
        ridge_alpha=_to_float(payload.get("ridge_alpha"), 50.0),
        enet_alpha=_to_float(payload.get("enet_alpha"), 0.01),
        enet_l1_ratio=_to_float(payload.get("enet_l1_ratio"), 0.5),
        lasso_alpha=_to_float(payload.get("lasso_alpha"), 0.003),
        lasso_max_iter=_to_int(payload.get("lasso_max_iter"), 20000),
        huber_epsilon=_to_float(payload.get("huber_epsilon"), 1.35),
        huber_alpha=_to_float(payload.get("huber_alpha"), 0.0001),
        huber_max_iter=_to_int(payload.get("huber_max_iter"), 20000),

        # XGBoost params
        xgb_n_estimators=_to_int(payload.get("xgb_n_estimators"), 1600),
        xgb_learning_rate=_to_float(payload.get("xgb_learning_rate"), 0.02),
        xgb_max_depth=_to_int(payload.get("xgb_max_depth"), 5),
        xgb_subsample=_to_float(payload.get("xgb_subsample"), 0.9),
        xgb_colsample_bytree=_to_float(payload.get("xgb_colsample_bytree"), 0.9),
        xgb_reg_lambda=_to_float(payload.get("xgb_reg_lambda"), 1.0),
        xgb_tree_method=str(payload.get("xgb_tree_method", "hist")),

        # LightGBM params
        lgbm_n_estimators=_to_int(payload.get("lgbm_n_estimators"), 1000),
        lgbm_learning_rate=_to_float(payload.get("lgbm_learning_rate"), 0.05),
        lgbm_num_leaves=_to_int(payload.get("lgbm_num_leaves"), 15),
        lgbm_max_depth=_to_int(payload.get("lgbm_max_depth"), 4),
        lgbm_subsample=_to_float(payload.get("lgbm_subsample"), 0.8),
        lgbm_colsample_bytree=_to_float(payload.get("lgbm_colsample_bytree"), 0.8),
        lgbm_reg_lambda=_to_float(payload.get("lgbm_reg_lambda"), 1.0),

        # CatBoost params
        cat_iterations=_to_int(payload.get("cat_iterations"), 5000),
        cat_learning_rate=_to_float(payload.get("cat_learning_rate"), 0.02),
        cat_depth=_to_int(payload.get("cat_depth"), 8),
        cat_l2_leaf_reg=_to_float(payload.get("cat_l2_leaf_reg"), 5.0),

        # PyTorch params
        torch_epochs=_to_int(payload.get("torch_epochs"), 200),
        torch_batch_size=_to_int(payload.get("torch_batch_size"), 32),
        torch_lr=_to_float(payload.get("torch_lr"), 1e-3),
        torch_patience=_to_int(payload.get("torch_patience"), 20),
        torch_hidden1=_to_int(payload.get("torch_hidden1"), 64),
        torch_hidden2=_to_int(payload.get("torch_hidden2"), 32),
        torch_dropout=_to_float(payload.get("torch_dropout"), 0.1),
        torch_seq_hidden_size=_to_int(payload.get("torch_seq_hidden_size"), 24),
        torch_seq_num_layers=_to_int(payload.get("torch_seq_num_layers"), 1),
        torch_seq_dropout=_to_float(payload.get("torch_seq_dropout"), 0.0),
        torch_cnn_channels=_to_int(payload.get("torch_cnn_channels"), 32),
        torch_cnn_kernel_size=_to_int(payload.get("torch_cnn_kernel_size"), 5),
        torch_patch_len=_to_int(payload.get("torch_patch_len"), 4),
        torch_patch_stride=_to_int(payload.get("torch_patch_stride"), 2),
        torch_transformer_d_model=_to_int(payload.get("torch_transformer_d_model"), 32),
        torch_transformer_nhead=_to_int(payload.get("torch_transformer_nhead"), 4),
        torch_transformer_layers=_to_int(payload.get("torch_transformer_layers"), 2),
        torch_itransformer_d_model=_to_int(payload.get("torch_itransformer_d_model"), 32),
        torch_itransformer_nhead=_to_int(payload.get("torch_itransformer_nhead"), 4),
        torch_itransformer_layers=_to_int(payload.get("torch_itransformer_layers"), 2),
        torch_nhits_hidden_size=_to_int(payload.get("torch_nhits_hidden_size"), 64),
        torch_nhits_num_blocks=_to_int(payload.get("torch_nhits_num_blocks"), 3),
        torch_nhits_pool_sizes=_to_int_tuple(payload.get("torch_nhits_pool_sizes"), (1, 2, 4)),

        # RF params
        rf_n_estimators=_to_int(payload.get("rf_n_estimators"), 800),
        rf_max_depth=None if payload.get("rf_max_depth") in (None, "", "None") else int(payload["rf_max_depth"]),
        rf_min_samples_leaf=_to_int(payload.get("rf_min_samples_leaf"), 1),
        rf_min_samples_split=_to_int(payload.get("rf_min_samples_split"), 2),
        rf_max_features=payload.get("rf_max_features", "sqrt"),
        rf_n_jobs=_to_int(payload.get("rf_n_jobs"), -1),

        # Output
        runs_dir=str(runs_dir),
        plot_test=_to_bool(payload.get("plot_test"), False),
        verbose=_to_bool(payload.get("verbose"), False),
    )

    try:
        result = run_backtest(cfg)

        overall_val = result.overall_val.copy() if result.overall_val is not None else pd.DataFrame()
        overall_test = result.overall_test.copy() if result.overall_test is not None else pd.DataFrame()

        # Single-model mode is easy because the pipeline already gives us its row directly.
        if mode == "single" and not seasonal_models and not seasonal_ensemble_models:
            selected_label = selected_single
            preds_test = _load_preds_test(result.run_id)

            return {
                "ok": True,
                "mode": "single",
                "horizon": horizon,
                "selected_model": selected_label,
                "enabled_models": enabled_models,
                "config": {
                    "tavg_file": cfg.tavg_file,
                    "sst_dir": cfg.sst_dir,
                    "use_enso": bool(cfg.enso_csv),
                    "enso_csv": cfg.enso_csv,
                    "enso_lags": list(cfg.enso_lags),
                    "ar_lags": list(cfg.ar_lags),
                    "runs_dir": cfg.runs_dir,
                    "use_sst": cfg.use_sst,
                    "use_teleconnections": cfg.use_teleconnections,
                    "use_qmap": cfg.use_qmap,
                    "random_seed": int(cfg.random_seed),
                },
                "summary": {
                    "run_id": result.run_id,
                    "best_overall_model": result.best_overall_model,
                    "used_month_choice": bool(result.used_month_choice),
                },
                "val_metrics": _extract_metric_row(overall_val, selected_label),
                "test_metrics": _extract_metric_row(overall_test, selected_label),
                "latest_forecast": _latest_forecast_from_preds(preds_test),
                "series": _series_from_preds(preds_test),
                "overall_val_table": _clean_records(overall_val),
                "overall_test_table": _clean_records(overall_test),
                "per_month_test_table": _clean_records(result.per_month_test),
            }

        # For ensemble / seasonal custom strategies, we need per-model prediction series.
        pred_test_df = _load_prediction_matrix_from_run(result.run_id, overall_test)
        preds_test_final = _load_preds_test(result.run_id)

        if pred_test_df is None:
            return {
                "ok": False,
                "error": (
                    "Custom ensemble/seasonal evaluation needs a wider test prediction file "
                    "such as preds_test_all.csv. Your current writer only exposes final "
                    "preds_test.csv. Add a per-model prediction export, then this route will work."
                ),
                "run_id": result.run_id,
                "overall_test_table": _clean_records(overall_test),
            }

        # We need true targets aligned to test prediction index.
        if preds_test_final is None or "y_true" not in preds_test_final.columns:
            return {
                "ok": False,
                "error": "Could not load y_true from preds_test.csv for ensemble evaluation.",
                "run_id": result.run_id,
            }

        preds_test_final = preds_test_final.sort_values("target_month")
        targ_test = pd.DatetimeIndex(preds_test_final["target_month"])
        y_true = pd.Series(preds_test_final["y_true"].values.astype(float), index=targ_test, name="y_true")

        pred_test_df = pred_test_df.reindex(targ_test)

        if mode == "seasonal_ensemble":
            strategy_label = "SeasonalEnsemble"
            y_pred, series = _seasonal_ensemble_predictions(
                seasonal_ensemble_models=seasonal_ensemble_models,
                targ_test=targ_test,
                y_true=y_true,
                pred_test_df=pred_test_df,
            )
        elif seasonal_models:
            strategy_label = "SeasonalCustom"
            y_pred, series = _seasonal_selected_predictions(
                seasonal_models=seasonal_models,
                targ_test=targ_test,
                y_true=y_true,
                pred_test_df=pred_test_df,
            )
        else:
            strategy_label = "CustomEnsemble"
            y_pred, series = _average_selected_predictions(
                selected_models=selected_ensemble,
                targ_test=targ_test,
                y_true=y_true,
                pred_test_df=pred_test_df,
            )

        test_metrics = _build_metric_row(strategy_label, y_true.values, y_pred)

        latest_forecast = {
            "target_month": series["dates"][-1] if series["dates"] else None,
            "predicted_anomaly": series["y_pred"][-1] if series["y_pred"] else None,
        }

        return {
            "ok": True,
            "mode": mode,
            "horizon": horizon,
            "selected_models": selected_ensemble,
            "seasonal_models": seasonal_models,
            "seasonal_ensemble_models": seasonal_ensemble_models,
            "enabled_models": enabled_models,
            "config": {
                "tavg_file": cfg.tavg_file,
                "sst_dir": cfg.sst_dir,
                "use_enso": bool(cfg.enso_csv),
                "enso_csv": cfg.enso_csv,
                "enso_lags": list(cfg.enso_lags),
                "ar_lags": list(cfg.ar_lags),
                "runs_dir": cfg.runs_dir,
                "use_sst": cfg.use_sst,
                "use_teleconnections": cfg.use_teleconnections,
                "use_qmap": cfg.use_qmap,
                "random_seed": int(cfg.random_seed),
            },
                "summary": {
                    "run_id": result.run_id,
                    "best_overall_model": result.best_overall_model,
                    "used_month_choice": bool(result.used_month_choice),
                },
                "val_metrics": None,
                "test_metrics": test_metrics,
                "series": series,
                "latest_forecast": latest_forecast,
                "overall_val_table": _clean_records(overall_val),
                "overall_test_table": _clean_records(overall_test),
            "per_month_test_table": _clean_records(result.per_month_test),
        }

    except Exception as e:
        message = str(e).strip() or e.__class__.__name__
        return {
            "ok": False,
            "error": f"{e.__class__.__name__}: {message}",
            "traceback": traceback.format_exc(),
        }
