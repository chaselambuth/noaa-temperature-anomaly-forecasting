import os

from flask import Flask, jsonify, render_template, request

from modeling import run_forecast_app

app = Flask(__name__)

RECOMMENDED_CONFIG = {
    "mode": "single",
    "horizon": 1,
    "model_name": "lstm",
    "seasonal_ensemble_models": {},
    "K": 10,
    "pc_lags": [1, 2, 3, 6, 12],
    "y_lags": [1, 2, 3, 6, 12],
    "sst_pc_lags": [0, 1, 3, 6, 12],
    "ar_lags": [1, 2, 3, 6, 12, 18, 24],
    "use_enso": True,
    "enso_lags": [0],
    "use_sst": False,
    "use_teleconnections": False,
    "use_qmap": False,
    "residual_deseasonalize": False,
    "half_life_months": 48,
    "month_group_size": 12,
    "use_climatology_residual": True,
    "climatology_years": 18,
    "deep_internal_val_months": 24,
    "model_params": {
        "ridge_alpha": 50.0,
        "enet_alpha": 0.01,
        "enet_l1_ratio": 0.5,
        "lasso_alpha": 0.003,
        "huber_epsilon": 1.35,
        "huber_alpha": 0.0001,
        "xgb_n_estimators": 1600,
        "xgb_learning_rate": 0.02,
        "xgb_max_depth": 5,
        "xgb_subsample": 0.9,
        "xgb_colsample_bytree": 0.9,
        "lgbm_n_estimators": 1000,
        "lgbm_learning_rate": 0.05,
        "lgbm_num_leaves": 15,
        "lgbm_max_depth": 4,
        "cat_iterations": 5000,
        "cat_learning_rate": 0.02,
        "cat_depth": 8,
        "cat_l2_leaf_reg": 5.0,
        "torch_hidden1": 64,
        "torch_hidden2": 32,
        "torch_dropout": 0.1,
        "torch_seq_hidden_size": 24,
        "torch_seq_num_layers": 1,
        "torch_seq_dropout": 0.0,
        "torch_cnn_channels": 32,
        "torch_cnn_kernel_size": 5,
    },
}

RECOMMENDED_RESULT = {
    "1": {
        "model_name": "lstm",
        "label": "Residual LSTM with 18-year climatology",
        "test_rmse": 1.113049529369226,
        "test_mae": 0.8463652472790572,
        "test_bias": -0.003061070511041344,
        "note": "Notebook 06 carry-forward selection; 11-year LSTM remains the strict RMSE-only sensitivity check.",
    },
}


def _default_path(env_name: str, local_relative: str) -> str:
    return local_relative.replace("\\", "/")


@app.get("/")
def home():
    default_config = {
        **RECOMMENDED_CONFIG,
        "tavg_file": _default_path("TAVG_FILE", "noaa_raw_data/nclimgrid_tavg.nc"),
        "sst_dir": _default_path("SST_DIR", "noaa_raw_data/sst_cache"),
        "enso_csv": _default_path("ENSO_CSV", "noaa_raw_data/nina34.anom.csv"),
        "runs_dir": _default_path("RUNS_DIR", "runs"),
    }
    return render_template(
        "index.html",
        experiment_config=default_config,
        experiment_best_by_horizon=RECOMMENDED_RESULT,
    )


@app.get("/health")
def health():
    return jsonify({"ok": True, "message": "Forecaster app is running"})


@app.get("/models")
def models():
    return jsonify(
        {
            "ok": True,
            "models": [
                "Ridge",
                "ElasticNet",
                "XGBoost",
                "LightGBM",
                "CatBoost",
                "RandomForest",
                "Lasso",
                "Huber",
                "MLP",
                "LSTM",
                "GRU",
                "CNN1D",
            ],
            "seasons": ["DJF", "MAM", "JJA", "SON"],
            "modes": ["single", "ensemble", "seasonal_ensemble"],
            "enso_options": [
                {"value": "off", "label": "Off", "lags": []},
                {"value": "enso_0_1_3_6_12", "label": "ENSO lags 0, 1, 3, 6, 12", "lags": [0, 1, 3, 6, 12]},
            ],
            "recommended_config": RECOMMENDED_CONFIG,
            "best_by_horizon": RECOMMENDED_RESULT,
        }
    )


@app.post("/run")
def run():
    payload = request.get_json(force=True) or {}
    result = run_forecast_app(payload)
    status = 200 if result.get("ok") else 400
    return jsonify(result), status


if __name__ == "__main__":
    # Docker-friendly host binding
    debug = os.getenv("FLASK_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    app.run(host="0.0.0.0", port=5000, debug=debug, use_reloader=debug)
