from pathlib import Path

import numpy as np
import pandas as pd

from forecaster.experiments import build_single_model_cfg, choose_best_options, load_selected_settings
from forecaster.ml.baselines import baseline_blend_for, climatology_baseline
from forecaster.ml.selection import (
    build_contiguous_month_groups,
    build_group_member_map,
    build_month_choice_table,
    build_strategy_table,
    make_month_choice_prediction,
    make_seasonal_ensemble_prediction,
    rank_models_in_group,
)


def test_climatology_baseline_uses_prior_same_months():
    idx = pd.date_range("2017-01-01", periods=24, freq="MS")
    y = pd.Series(np.arange(24, dtype=float), index=idx)

    pred = climatology_baseline(pd.DatetimeIndex(["2019-01-01"]), y, years=2)

    assert pred.tolist() == [6.0]


def test_baseline_blend_for_matches_notebook_formula():
    idx = pd.date_range("2017-01-01", periods=36, freq="MS")
    y = pd.Series(np.arange(36, dtype=float), index=idx)
    feature_idx = pd.DatetimeIndex(["2018-01-01"])
    target_idx = pd.DatetimeIndex(["2018-02-01"])

    pred = baseline_blend_for(feature_idx, target_idx, y, persist_w=0.6, lastyear_w=0.4)

    assert pred.tolist() == [7.6]


def test_group_choice_helpers_rank_and_predict():
    target_idx = pd.date_range("2020-01-01", periods=12, freq="MS")
    y = np.arange(12, dtype=float)
    preds = {
        "Good": y.copy(),
        "Bad": y + 10,
    }
    groups = build_contiguous_month_groups(3)
    rankings = {
        group: rank_models_in_group(y, target_idx, preds, months)
        for group, months in groups.items()
    }

    choice_table = build_month_choice_table(rankings)
    out = make_month_choice_prediction(target_idx, preds, choice_table, groups)

    assert set(choice_table["Winner"]) == {"Good"}
    assert np.allclose(out, y)


def test_seasonal_ensemble_helpers_average_members():
    target_idx = pd.date_range("2020-01-01", periods=3, freq="MS")
    groups = {"Q1": [1, 2, 3]}
    rankings = {
        "Q1": pd.DataFrame(
            [
                {"Model": "A", "RMSE": 1.0, "MAE": 1.0, "Bias": 0.0, "Count": 3},
                {"Model": "B", "RMSE": 2.0, "MAE": 2.0, "Bias": 0.0, "Count": 3},
            ]
        )
    }
    strategy = build_strategy_table(rankings, top_k=2)
    members = build_group_member_map(strategy)
    preds = {"A": np.array([1.0, 2.0, 3.0]), "B": np.array([3.0, 4.0, 5.0])}

    out = make_seasonal_ensemble_prediction(target_idx, preds, members, groups)

    assert np.allclose(out, [2.0, 3.0, 4.0])


def test_build_single_model_cfg_disables_other_models():
    base_cfg = {
        "tavg_file": "dummy",
        "sst_dir": "dummy",
        "use_ridge": True,
        "use_enet": True,
        "use_xgb": True,
        "train_target_start": "2000-01-01",
    }

    cfg = build_single_model_cfg(base_cfg, "Ridge", {"ridge_alpha": 2.5})

    assert cfg.use_ridge is True
    assert cfg.use_enet is False
    assert cfg.use_xgb is False
    assert cfg.ridge_alpha == 2.5
    assert cfg.train_target_start == "2000-01-01"


def test_load_selected_settings_and_choose_best(tmp_path: Path):
    path = tmp_path / "selected.csv"
    pd.DataFrame(
        [
            {"model": "Ridge", "params": "{'ridge_alpha': 1.0}"},
            {"model": "ElasticNet", "params": "{'enet_alpha': 0.1}"},
        ]
    ).to_csv(path, index=False)

    _, selected = load_selected_settings(path)
    assert selected["Ridge"] == {"ridge_alpha": 1.0}

    best = choose_best_options(
        pd.DataFrame(
            [
                {"model": "Ridge", "status": "ok", "val_rmse": 2.0, "val_mae": 1.0, "test_rmse": 9.0},
                {"model": "Ridge", "status": "ok", "val_rmse": 1.0, "val_mae": 1.0, "test_rmse": 9.0},
                {"model": "Bad", "status": "failed", "val_rmse": np.nan, "val_mae": np.nan, "test_rmse": np.nan},
            ]
        )
    )

    assert best.iloc[0]["model"] == "Ridge"
    assert best.iloc[0]["val_rmse"] == 1.0
