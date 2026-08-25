from __future__ import annotations

import ast
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ForecasterConfig
from .data.sources import load_enso_monthly, load_land_series_and_anoms, load_teleconnections
from .data.sst import load_multi_region_sst_pcs
from .ml.baselines import climatology_baseline
from .ml.core import BacktestSplit, build_features_and_target, make_recency_weights, train_predict_models
from .ml.evaluation import bias, mae, overall_table, rmse
from .ml.selection import build_month_choice_table, make_month_choice_prediction, rank_models_in_group


DEFAULT_MODEL_TOGGLE_FIELDS: Dict[str, str] = {
    "Ridge": "use_ridge",
    "Huber": "use_huber",
    "XGBoost": "use_xgb",
    "LSTM": "use_lstm",
    "PatchTST": "use_patchtst",
    "iTransformer": "use_itransformer",
    "NHiTS": "use_nhits",
}

ALL_MODEL_TOGGLE_FIELDS: Dict[str, str] = {
    **DEFAULT_MODEL_TOGGLE_FIELDS,
    "ElasticNet": "use_enet",
    "RandomForest": "use_rf",
    "LightGBM": "use_lgbm",
    "CatBoost": "use_cat",
    "Lasso": "use_lasso",
    "MLP": "use_mlp",
    "GRU": "use_gru",
    "CNN1D": "use_cnn1d",
}

DEEP_MODELS = {"LSTM", "PatchTST", "iTransformer", "NHiTS"}

def find_project_root(start: Path) -> Path:

    start = Path(start).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "forecaster" / "__init__.py").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root containing src/forecaster.")

def _build_config(config_cls: type[ForecasterConfig], values: Mapping[str, Any]) -> ForecasterConfig:
    fields = set(getattr(config_cls, "__dataclass_fields__", {}).keys())
    init_kwargs = {k: v for k, v in values.items() if k in fields}
    extra_attrs = {k: v for k, v in values.items() if k not in fields}

    cfg = config_cls(**init_kwargs)
    for key, value in extra_attrs.items():
        object.__setattr__(cfg, key, value)
    return cfg

def build_single_model_cfg(
    base_cfg: Mapping[str, Any],
    model_name: str,
    params: Optional[Mapping[str, Any]] = None,
    *,
    model_toggle_fields: Optional[Mapping[str, str]] = None,
    config_cls: type[ForecasterConfig] = ForecasterConfig,
) -> ForecasterConfig:
 
    toggle_fields = dict(model_toggle_fields or DEFAULT_MODEL_TOGGLE_FIELDS)
    if model_name not in toggle_fields:
        raise KeyError(f"No model toggle field configured for {model_name!r}.")

    cfg_kwargs = dict(base_cfg)
    for toggle_field in set(ALL_MODEL_TOGGLE_FIELDS.values()).union(toggle_fields.values()):
        cfg_kwargs[toggle_field] = False
    cfg_kwargs[toggle_fields[model_name]] = True
    cfg_kwargs.update(dict(params or {}))

    return _build_config(config_cls, cfg_kwargs)

def load_selected_settings(
    path: Path,
    *,
    model_column: str = "model",
    params_column: str = "params",
) -> tuple[pd.DataFrame, Dict[str, dict]]:
 
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Selected settings file not found: {path}")

    df = pd.read_csv(path)
    selected: Dict[str, dict] = {}
    for _, row in df.iterrows():
        selected[str(row[model_column])] = ast.literal_eval(str(row[params_column]))
    return df, selected

def prepare_model_problem(
    base_cfg: Mapping[str, Any],
    *,
    ar_lags: Optional[Sequence[int]] = None,
    include_teleconnections: Optional[bool] = None,
    include_sst: Optional[bool] = None,
    config_cls: type[ForecasterConfig] = ForecasterConfig,
) -> Dict[str, Any]:

    cfg = _build_config(config_cls, base_cfg)
    y_full, y_anom_full, time_index = load_land_series_and_anoms(cfg)
    y_anom_full = pd.Series(y_anom_full, index=time_index, name="y_anom").sort_index()

    enso = load_enso_monthly(cfg.enso_csv, time_index) if getattr(cfg, "enso_csv", None) else None

    use_tele = bool(getattr(cfg, "use_teleconnections", False) if include_teleconnections is None else include_teleconnections)
    ao = nao = pna = None
    if use_tele:
        ao, nao, pna = load_teleconnections(time_index, cfg)

    use_sst = bool(getattr(cfg, "use_sst", False) if include_sst is None else include_sst)
    z_sst_multi = load_multi_region_sst_pcs(cfg, time_index) if use_sst else None

    split = build_features_and_target(
        cfg=cfg,
        y_anom_full=y_anom_full,
        time_index=time_index,
        enso=enso,
        ao=ao,
        nao=nao,
        pna=pna,
        Z_sst_multi=z_sst_multi,
        ar_lags=tuple(ar_lags if ar_lags is not None else getattr(cfg, "ar_lags", tuple(range(1, 13)))),
    )

    return {
        "base_cfg": dict(base_cfg),
        "cfg": cfg,
        "y_full": y_full,
        "y_anom_full": y_anom_full,
        "time_index": time_index,
        "enso": enso,
        "ao": ao,
        "nao": nao,
        "pna": pna,
        "Z_sst_multi": z_sst_multi,
        "split": split,
    }

def evaluate_model_option(
    problem: Mapping[str, Any],
    model_name: str,
    option_name: str,
    params: Mapping[str, Any],
    *,
    base_cfg: Optional[Mapping[str, Any]] = None,
    metadata: Optional[Mapping[str, Any]] = None,
    model_toggle_fields: Optional[Mapping[str, str]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Fit one model/parameter option and return a summary row plus payload.
    """
    cfg = None
    meta = dict(metadata or {})
    cfg_values = dict(base_cfg or problem.get("base_cfg") or {})

    try:
        cfg = build_single_model_cfg(
            cfg_values,
            model_name,
            params,
            model_toggle_fields=model_toggle_fields,
        )
        pred_val_dict, pred_test_dict = train_predict_models(cfg, problem["split"])
        if model_name not in pred_val_dict or model_name not in pred_test_dict:
            raise RuntimeError(f"{model_name} predictions were not returned.")

        pred_val = np.asarray(pred_val_dict[model_name], float)
        pred_test = np.asarray(pred_test_dict[model_name], float)

        row = {
            "model": model_name,
            "option": option_name,
            "status": "ok",
            "params": repr(dict(params)),
            **meta,
            "val_rmse": rmse(problem["split"].y_val.values, pred_val),
            "val_mae": mae(problem["split"].y_val.values, pred_val),
            "val_bias": bias(problem["split"].y_val.values, pred_val),
            "test_rmse": rmse(problem["split"].y_test.values, pred_test),
            "test_mae": mae(problem["split"].y_test.values, pred_test),
            "test_bias": bias(problem["split"].y_test.values, pred_test),
            "error": "",
        }
        payload = {"pred_val": pred_val, "pred_test": pred_test, "cfg": cfg, "params": dict(params)}
        return row, payload
    except Exception as exc:
        row = {
            "model": model_name,
            "option": option_name,
            "status": "failed",
            "params": repr(dict(params)),
            **meta,
            "val_rmse": np.nan,
            "val_mae": np.nan,
            "val_bias": np.nan,
            "test_rmse": np.nan,
            "test_mae": np.nan,
            "test_bias": np.nan,
            "error": f"{type(exc).__name__}: {exc}",
        }
        payload = {
            "pred_val": None,
            "pred_test": None,
            "cfg": cfg,
            "params": dict(params),
            "traceback": traceback.format_exc(),
        }
        return row, payload

def choose_best_options(tuning_results_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = tuning_results_df.loc[tuning_results_df["status"] == "ok"].copy()
    if ok_df.empty:
        return ok_df
    ok_df = ok_df.sort_values(["model", "val_rmse", "val_mae", "test_rmse"]).reset_index(drop=True)
    best_df = ok_df.groupby("model", as_index=False).first()
    return best_df.sort_values(["val_rmse", "val_mae"]).reset_index(drop=True)

def prepare_residual_problem(
    base_cfg: Mapping[str, Any],
    climatology_years: int,
    *,
    ar_lags: Optional[Sequence[int]] = None,
    include_teleconnections: Optional[bool] = None,
    include_sst: Optional[bool] = None,
) -> Dict[str, Any]:
   
    problem = prepare_model_problem(
        base_cfg,
        ar_lags=ar_lags,
        include_teleconnections=include_teleconnections,
        include_sst=include_sst,
    )
    split_full = problem["split"]
    y_anom_full = problem["y_anom_full"]

    clim_train = climatology_baseline(split_full.targ_train, y_anom_full, years=climatology_years)
    clim_val = climatology_baseline(split_full.targ_val, y_anom_full, years=climatology_years)
    clim_test = climatology_baseline(split_full.targ_test, y_anom_full, years=climatology_years)

    # Train models on residuals, but keep the original split and climatology arrays
    # so validation/test scores can be reported on the final anomaly scale.
    split = BacktestSplit(
        X_train=split_full.X_train,
        X_val=split_full.X_val,
        X_test=split_full.X_test,
        y_train=pd.Series(split_full.y_train.values - clim_train, index=split_full.y_train.index, name="resid_train"),
        y_val=pd.Series(split_full.y_val.values - clim_val, index=split_full.y_val.index, name="resid_val"),
        y_test=pd.Series(split_full.y_test.values - clim_test, index=split_full.y_test.index, name="resid_test"),
        targ_train=split_full.targ_train,
        targ_val=split_full.targ_val,
        targ_test=split_full.targ_test,
        w_train=getattr(split_full, "w_train", None),
    )

    return {
        **problem,
        "split_full": split_full,
        "split": split,
        "clim_train": clim_train,
        "clim_val": clim_val,
        "clim_test": clim_test,
        "climatology_years": int(climatology_years),
        "climatology_label": f"Climatology{int(climatology_years)}",
    }

def predict_selected_model_for_choice(
    problem: Mapping[str, Any],
    model_name: str,
    params: Mapping[str, Any],
    *,
    base_cfg: Optional[Mapping[str, Any]] = None,
    model_toggle_fields: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
   
    cfg = build_single_model_cfg(
        base_cfg or problem["base_cfg"],
        model_name,
        params,
        model_toggle_fields=model_toggle_fields,
    )
    pred_val_dict, pred_test_dict = train_predict_models(cfg, problem["split"])
    pred_val = np.asarray(pred_val_dict[model_name], float) + problem["clim_val"]
    pred_test = np.asarray(pred_test_dict[model_name], float) + problem["clim_test"]

    return {
        "cfg": cfg,
        "pred_val": pred_val,
        "pred_test": pred_test,
        "val_rmse": rmse(problem["split_full"].y_val.values, pred_val),
        "val_mae": mae(problem["split_full"].y_val.values, pred_val),
        "val_bias": bias(problem["split_full"].y_val.values, pred_val),
        "test_rmse": rmse(problem["split_full"].y_test.values, pred_test),
        "test_mae": mae(problem["split_full"].y_test.values, pred_test),
        "test_bias": bias(problem["split_full"].y_test.values, pred_test),
    }

def build_refit_split(
    problem: Mapping[str, Any],
    model_name: str,
    *,
    deep_internal_val_months: int = 24,
    deep_models: Optional[set[str]] = None,
) -> BacktestSplit:

    split_full = problem["split_full"]
    resid_split = problem["split"]

    x_tv = pd.concat([resid_split.X_train, resid_split.X_val], axis=0)
    y_tv = pd.concat([resid_split.y_train, resid_split.y_val], axis=0)
    targ_tv = pd.DatetimeIndex(list(split_full.targ_train) + list(split_full.targ_val))

    if len(x_tv) == 0:
        raise ValueError("Cannot build a refit split from an empty train+val set.")

    deep_set = set(deep_models or DEEP_MODELS)
    if model_name in deep_set:
        if len(x_tv) < 2:
            raise ValueError("Deep-model refit needs at least two train+val rows.")
        internal_val = int(min(max(deep_internal_val_months, 1), len(x_tv) - 1))
        x_train_refit = x_tv.iloc[:-internal_val].copy()
        y_train_refit = y_tv.iloc[:-internal_val].copy()
        targ_train_refit = targ_tv[:-internal_val]
        x_val_refit = x_tv.iloc[-internal_val:].copy()
        y_val_refit = y_tv.iloc[-internal_val:].copy()
        targ_val_refit = targ_tv[-internal_val:]
    else:
        x_train_refit = x_tv.copy()
        y_train_refit = y_tv.copy()
        targ_train_refit = targ_tv
        internal_val = int(min(max(deep_internal_val_months, 1), len(x_tv)))
        x_val_refit = x_tv.iloc[-internal_val:].copy()
        y_val_refit = y_tv.iloc[-internal_val:].copy()
        targ_val_refit = targ_tv[-internal_val:]

    if bool(getattr(problem["cfg"], "use_recency_weights", False)):
        w_train_refit = make_recency_weights(
            pd.DatetimeIndex(x_train_refit.index),
            half_life_months=float(getattr(problem["cfg"], "half_life_months", 48.0)),
        )
    else:
        w_train_refit = None

    return BacktestSplit(
        X_train=x_train_refit,
        X_val=x_val_refit,
        X_test=resid_split.X_test,
        y_train=y_train_refit,
        y_val=y_val_refit,
        y_test=resid_split.y_test,
        targ_train=targ_train_refit,
        targ_val=targ_val_refit,
        targ_test=resid_split.targ_test,
        w_train=w_train_refit,
    )

def refit_selected_model(
    problem: Mapping[str, Any],
    model_name: str,
    params: Mapping[str, Any],
    *,
    base_cfg: Optional[Mapping[str, Any]] = None,
    deep_internal_val_months: int = 24,
    model_toggle_fields: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
 
    cfg = build_single_model_cfg(
        base_cfg or problem["base_cfg"],
        model_name,
        params,
        model_toggle_fields=model_toggle_fields,
    )
    refit_split = build_refit_split(
        problem,
        model_name,
        deep_internal_val_months=deep_internal_val_months,
    )
    _, pred_test_dict = train_predict_models(cfg, refit_split)
    pred_test = np.asarray(pred_test_dict[model_name], float) + problem["clim_test"]

    return {
        "cfg": cfg,
        "refit_split": refit_split,
        "pred_test": pred_test,
        "test_rmse": rmse(problem["split_full"].y_test.values, pred_test),
        "test_mae": mae(problem["split_full"].y_test.values, pred_test),
        "test_bias": bias(problem["split_full"].y_test.values, pred_test),
    }

def evaluate_residual_month_choice(
    problem: Mapping[str, Any],
    selected_params: Mapping[str, Mapping[str, Any]],
    month_groups: Mapping[str, Sequence[int]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    deep_internal_val_months: int = 24,
) -> Dict[str, Any]:

    meta = dict(metadata or {})
    choice_rows = []
    refit_rows = []
    pred_val_choice_all = {problem["climatology_label"]: problem["clim_val"]}
    pred_test_all = {problem["climatology_label"]: problem["clim_test"]}

    for model_name, params in selected_params.items():
        choice_rr = predict_selected_model_for_choice(problem, model_name, params)
        refit_rr = refit_selected_model(
            problem,
            model_name,
            params,
            deep_internal_val_months=deep_internal_val_months,
        )
        candidate_name = f"Residual_{model_name}"
        pred_val_choice_all[candidate_name] = choice_rr["pred_val"]
        pred_test_all[candidate_name] = refit_rr["pred_test"]

        choice_rows.append(
            {
                **meta,
                "climatology_years": int(problem["climatology_years"]),
                "climatology_label": problem["climatology_label"],
                "model": model_name,
                "candidate": candidate_name,
                "params": repr(dict(params)),
                "val_rmse": choice_rr["val_rmse"],
                "val_mae": choice_rr["val_mae"],
                "val_bias": choice_rr["val_bias"],
                "test_rmse_before_refit": choice_rr["test_rmse"],
                "test_mae_before_refit": choice_rr["test_mae"],
            }
        )
        refit_rows.append(
            {
                **meta,
                "climatology_years": int(problem["climatology_years"]),
                "climatology_label": problem["climatology_label"],
                "model": model_name,
                "candidate": candidate_name,
                "params": repr(dict(params)),
                "test_rmse": refit_rr["test_rmse"],
                "test_mae": refit_rr["test_mae"],
                "test_bias": refit_rr["test_bias"],
                "refit_train_rows": len(refit_rr["refit_split"].y_train),
                "internal_val_rows": len(refit_rr["refit_split"].y_val),
            }
        )

    choice_candidate_df = pd.DataFrame(choice_rows).sort_values(["val_rmse", "val_mae"]).reset_index(drop=True)
    refit_test_df = pd.DataFrame(refit_rows).sort_values(["test_rmse", "test_mae"]).reset_index(drop=True)

    group_rankings = {
        group: rank_models_in_group(
            y_true=problem["split_full"].y_val.values,
            target_idx=problem["split_full"].targ_val,
            pred_dict=pred_val_choice_all,
            months=months,
        )
        for group, months in month_groups.items()
    }
    month_choice_table = build_month_choice_table(group_rankings)
    month_choice_pred = make_month_choice_prediction(
        targ_idx=problem["split_full"].targ_test,
        pred_test_all=pred_test_all,
        month_choice_table=month_choice_table,
        month_groups=dict(month_groups),
    )

    month_choice_metrics_df = pd.DataFrame(
        [
            {
                **meta,
                "climatology_years": int(problem["climatology_years"]),
                "climatology_label": problem["climatology_label"],
                "strategy": "Ensemble_MonthChoice",
                "test_rmse": rmse(problem["split_full"].y_test.values, month_choice_pred),
                "test_mae": mae(problem["split_full"].y_test.values, month_choice_pred),
                "test_bias": bias(problem["split_full"].y_test.values, month_choice_pred),
            }
        ]
    )

    pred_test_all_with_choice = {
        **pred_test_all,
        "Ensemble_MonthChoice": month_choice_pred,
    }
    comparison_meta = {"H": meta["H"]} if "H" in meta else {}

    overall_test_df = (
        overall_table(
            problem["split_full"].y_test.values,
            pred_test_all_with_choice,
            problem["split_full"].targ_test,
        )
        .sort_values(["RMSE", "MAE"])
        .reset_index(drop=True)
    )
    overall_test_df.insert(0, "climatology_label", problem["climatology_label"])
    overall_test_df.insert(0, "climatology_years", int(problem["climatology_years"]))
    for key, value in reversed(list(comparison_meta.items())):
        overall_test_df.insert(0, key, value)

    best_global = refit_test_df.iloc[0]
    climatology_row = overall_test_df.loc[
        overall_test_df["Model"] == problem["climatology_label"]
    ].iloc[0]
    month_choice_row = month_choice_metrics_df.iloc[0]

    benchmark_rows = [
        {
            **comparison_meta,
            "climatology_years": int(problem["climatology_years"]),
            "climatology_label": problem["climatology_label"],
            "strategy": "global_single_best_refit_model",
            "source": str(best_global["candidate"]),
            "test_rmse": float(best_global["test_rmse"]),
            "test_mae": float(best_global["test_mae"]),
            "test_bias": float(best_global["test_bias"]),
        }
    ]
    for row in overall_test_df.itertuples(index=False):
        benchmark_rows.append(
            {
                **comparison_meta,
                "climatology_years": int(problem["climatology_years"]),
                "climatology_label": problem["climatology_label"],
                "strategy": str(row.Model),
                "source": str(row.Model),
                "test_rmse": float(row.RMSE),
                "test_mae": float(row.MAE),
                "test_bias": float(row.Bias),
            }
        )
    benchmark_df = (
        pd.DataFrame(benchmark_rows)
        .sort_values(["test_rmse", "test_mae"])
        .reset_index(drop=True)
    )

    def _edge_date(index: pd.DatetimeIndex, position: int) -> str:
        return pd.Timestamp(index[position]).strftime("%Y-%m-%d")

    winner_summary = "; ".join(f"{row.Group}:{row.Winner}" for row in month_choice_table.itertuples(index=False))
    summary_df = pd.DataFrame(
        [
            {
                **meta,
                "climatology_years": int(problem["climatology_years"]),
                "climatology_label": problem["climatology_label"],
                "selected_model_count": int(len(selected_params)),
                "train_target_start": _edge_date(problem["split_full"].targ_train, 0),
                "train_target_end": _edge_date(problem["split_full"].targ_train, -1),
                "val_target_start": _edge_date(problem["split_full"].targ_val, 0),
                "val_target_end": _edge_date(problem["split_full"].targ_val, -1),
                "test_target_start": _edge_date(problem["split_full"].targ_test, 0),
                "test_target_end": _edge_date(problem["split_full"].targ_test, -1),
                "best_global_source": str(best_global["candidate"]),
                "best_global_test_rmse": float(best_global["test_rmse"]),
                "best_global_test_mae": float(best_global["test_mae"]),
                "ensemble_month_choice_test_rmse": float(month_choice_row["test_rmse"]),
                "ensemble_month_choice_test_mae": float(month_choice_row["test_mae"]),
                "ensemble_month_choice_test_bias": float(month_choice_row["test_bias"]),
                "climatology_test_rmse": float(climatology_row["RMSE"]),
                "climatology_test_mae": float(climatology_row["MAE"]),
                "dRMSE_ensemble_vs_best_global": float(month_choice_row["test_rmse"] - best_global["test_rmse"]),
                "dMAE_ensemble_vs_best_global": float(month_choice_row["test_mae"] - best_global["test_mae"]),
                "dRMSE_ensemble_vs_climatology": float(month_choice_row["test_rmse"] - climatology_row["RMSE"]),
                "dMAE_ensemble_vs_climatology": float(month_choice_row["test_mae"] - climatology_row["MAE"]),
                "month_choice_winners": winner_summary,
                "deep_internal_val_months": int(deep_internal_val_months),
            }
        ]
    )

    month_choice_table = month_choice_table.copy()
    for key, value in reversed(list(meta.items())):
        month_choice_table.insert(0, key, value)
    month_choice_table.insert(0, "climatology_label", problem["climatology_label"])
    month_choice_table.insert(0, "climatology_years", int(problem["climatology_years"]))

    return {
        "problem": problem,
        "choice_candidate_df": choice_candidate_df,
        "refit_test_df": refit_test_df,
        "month_choice_table": month_choice_table,
        "month_choice_metrics_df": month_choice_metrics_df,
        "overall_test_df": overall_test_df,
        "benchmark_df": benchmark_df,
        "summary_df": summary_df,
        "month_choice_pred": month_choice_pred,
    }

def evaluate_climatology_window(
    base_cfg: Mapping[str, Any],
    climatology_years: int,
    selected_params: Mapping[str, Mapping[str, Any]],
    month_groups: Mapping[str, Sequence[int]],
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    deep_internal_val_months: int = 24,
) -> Dict[str, Any]:
  
    problem = prepare_residual_problem(base_cfg, climatology_years=climatology_years)
    return evaluate_residual_month_choice(
        problem,
        selected_params,
        month_groups,
        metadata=metadata,
        deep_internal_val_months=deep_internal_val_months,
    )

def evaluate_horizon(
    base_cfg_template: Mapping[str, Any],
    horizon: int,
    selected_params: Mapping[str, Mapping[str, Any]],
    month_groups: Mapping[str, Sequence[int]],
    *,
    climatology_years: int,
    metadata: Optional[Mapping[str, Any]] = None,
    deep_internal_val_months: int = 24,
) -> Dict[str, Any]:
 
    base_cfg = dict(base_cfg_template)
    base_cfg["H"] = int(horizon)
    meta = {"H": int(horizon), **dict(metadata or {})}
    return evaluate_climatology_window(
        base_cfg,
        climatology_years=climatology_years,
        selected_params=selected_params,
        month_groups=month_groups,
        metadata=meta,
        deep_internal_val_months=deep_internal_val_months,
    )

def summarize_run(
    rr: Any,
    label: Any,
    *,
    label_field: str = "run",
    prefer_month_choice: bool = True,
) -> Dict[str, Any]:
  
    overall_val = rr.overall_val.copy().sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    overall_test = rr.overall_test.copy().sort_values(["RMSE", "MAE"]).reset_index(drop=True)
    baseline_rows = overall_test[overall_test["Model"] == "Baseline"]

    if prefer_month_choice and "Ensemble_MonthChoice" in set(overall_test["Model"]):
        final_row = overall_test[overall_test["Model"] == "Ensemble_MonthChoice"].iloc[0]
        final_model = "Ensemble_MonthChoice"
    else:
        best_val_model = str(overall_val.iloc[0]["Model"])
        matched = overall_test[overall_test["Model"] == best_val_model]
        final_row = matched.iloc[0] if not matched.empty else overall_test.iloc[0]
        final_model = str(final_row["Model"])

    choice_table = getattr(rr, "month_choice_table", None)
    if choice_table is not None and "Winner" in choice_table.columns:
        chosen_models = ", ".join(sorted(choice_table["Winner"].dropna().astype(str).unique()))
    else:
        chosen_models = ""

    out = {
        label_field: label,
        "final_model": final_model,
        "chosen_models": chosen_models,
        "global_val_best_model": str(overall_val.iloc[0]["Model"]),
        "global_val_rmse": float(overall_val.iloc[0]["RMSE"]),
        "global_val_mae": float(overall_val.iloc[0]["MAE"]),
        "test_rmse": float(final_row["RMSE"]),
        "test_mae": float(final_row["MAE"]),
        "test_bias": float(final_row["Bias"]),
        "run_id": rr.run_id,
    }
    if not baseline_rows.empty:
        baseline_row = baseline_rows.iloc[0]
        out.update(
            {
                "baseline_test_rmse": float(baseline_row["RMSE"]),
                "baseline_test_mae": float(baseline_row["MAE"]),
                "dRMSE_vs_baseline": float(final_row["RMSE"] - baseline_row["RMSE"]),
                "dMAE_vs_baseline": float(final_row["MAE"] - baseline_row["MAE"]),
            }
        )
    return out
