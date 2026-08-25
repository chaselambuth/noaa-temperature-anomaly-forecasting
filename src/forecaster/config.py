from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Optional, Any


def default_sst_regions() -> Dict[str, Dict[str, Any]]:
    return {
        "TPAC": dict(lat=(-30, 30), lon=(120, 290), k=8),
        "NPAC": dict(lat=(20, 70), lon=(110, 260), k=6),
        "NATL": dict(lat=(0, 70), lon=(280, 360), k=6),
        "IND":  dict(lat=(-30, 30), lon=(40, 110),  k=5),
    }

@dataclass(frozen=True)
class ForecasterConfig:
    # Paths
    tavg_file: str
    sst_dir: str
    enso_csv: Optional[str] = None

    # Chronological splits use target months.
    train_target_start = "1950-01-01"
    train_target_end   = "2009-12-01"

    val_target_start   = "2010-01-01"
    val_target_end     = "2017-12-01"

    test_target_start  = "2018-01-01"
    test_target_end    = "2025-12-01"


    debug_missingness = False

    # Forecast horizon (months-ahead)
    H: int = 1

    # Land PCA + lags
    K: int = 10
    pc_lags: Sequence[int] = (1, 2, 3, 6, 12)
    y_lags: Sequence[int]  = (1, 2, 3, 6, 12)

    # AR lag structure used to build autoregressive features
    # Default: 1–12 months
    ar_lags: tuple[int, ...] = tuple(range(1, 13))

    # ENSO lags (lag 0 is included automatically by the feature builder)
    enso_lags: Sequence[int] = (0, 1, 3, 6, 12)

    # Teleconnections
    use_teleconnections: bool = True
    ao_url: str = "https://psl.noaa.gov/data/correlation/ao.data"
    nao_url: str = "https://psl.noaa.gov/data/correlation/nao.data"
    pna_url: str = "https://psl.noaa.gov/data/correlation/pna.data"
    tele_lags: Sequence[int] = (0, 1, 2, 3, 6, 12)

    # SST multi-region PCA
    use_sst: bool = True
    sst_pc_lags: Sequence[int] = (0, 1, 3, 6, 12)
    sst_regions: Dict[str, Dict[str, Any]] = field(default_factory=default_sst_regions)

    # Cache files (optional). If None, sst.py will build defaults under sst_dir.
    sst_pcs_cache_files: Optional[Dict[str, str]] = None

    # Coverage filtering for PCA columns (TRAIN-only coverage)
    coverage_candidates: Sequence[float] = (
        0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.35, 0.20, 0.10, 0.09
    )

    # If coverage filter drops all points, keep top-covered cells
    sst_fallback_keep_min: int = 250
    sst_fallback_keep_max: int = 5000

    # Training / weighting
    random_seed: int = 42
    use_recency_weights: bool = True
    half_life_months: float = 48.0

    # Models toggles
    use_ridge: bool = True
    use_enet: bool = False
    use_xgb: bool = True
    use_lgbm: bool = False
    use_cat: bool = False
    use_rf: bool = False
    use_lasso: bool = False
    use_huber: bool = True
    
    # PyTorch MLP toggle
    use_mlp: bool = False
    use_lstm: bool = False
    use_gru: bool = False
    use_cnn1d: bool = False
    use_patchtst: bool = False
    use_itransformer: bool = False
    use_nhits: bool = False

    # Ridge / ElasticNet params
    ridge_alpha: float = 50.0
    enet_alpha: float = 0.01
    enet_l1_ratio: float = 0.5

    # Lasso / Huber params
    lasso_alpha: float = 0.003
    lasso_max_iter: int = 20000
    huber_epsilon: float = 1.35
    huber_alpha: float = 0.0001
    huber_max_iter: int = 20000

    # RandomForest params (NEW)
    rf_n_estimators: int = 800
    rf_max_depth: Optional[int] = None
    rf_min_samples_leaf: int = 1
    rf_min_samples_split: int = 2
    # IMPORTANT: sklearn no longer accepts "auto" for RandomForestRegressor
    rf_max_features: Optional[str] = "sqrt"  # "sqrt" or "log2" or None
    rf_n_jobs: int = -1

    # XGBoost params (mirrors your reference)
    xgb_n_estimators: int = 1600
    xgb_learning_rate: float = 0.02
    xgb_max_depth: int = 5
    xgb_subsample: float = 0.9
    xgb_colsample_bytree: float = 0.9
    xgb_reg_lambda: float = 1.0
    xgb_tree_method: str = "hist"

    # LightGBM params
    lgbm_n_estimators: int = 1000
    lgbm_learning_rate: float = 0.05
    lgbm_num_leaves: int = 15
    lgbm_max_depth: int = 4
    lgbm_subsample: float = 0.8
    lgbm_colsample_bytree: float = 0.8
    lgbm_reg_lambda: float = 1.0

    # CatBoost params
    cat_iterations: int = 5000
    cat_learning_rate: float = 0.02
    cat_depth: int = 8
    cat_l2_leaf_reg: float = 5.0

    # PyTorch MLP params
    torch_epochs: int = 200
    torch_batch_size: int = 32
    torch_lr: float = 1e-3
    torch_patience: int = 20
    torch_hidden1: int = 64
    torch_hidden2: int = 32
    torch_dropout: float = 0.1

    # PyTorch sequence models (LSTM / GRU)
    torch_seq_hidden_size: int = 24
    torch_seq_num_layers: int = 1
    torch_seq_dropout: float = 0.0

    # PyTorch 1D CNN
    torch_cnn_channels: int = 32
    torch_cnn_kernel_size: int = 5

    # PyTorch PatchTST-style transformer
    torch_patch_len: int = 4
    torch_patch_stride: int = 2
    torch_transformer_d_model: int = 32
    torch_transformer_nhead: int = 4
    torch_transformer_layers: int = 2

    # PyTorch iTransformer-style inverted transformer
    torch_itransformer_d_model: int = 32
    torch_itransformer_nhead: int = 4
    torch_itransformer_layers: int = 2

    # PyTorch N-HiTS-style multi-rate residual network
    torch_nhits_hidden_size: int = 64
    torch_nhits_num_blocks: int = 3
    torch_nhits_pool_sizes: tuple[int, ...] = (1, 2, 4)

    # QMAP
    use_qmap: bool = True
    qmap_cal_target_months: int = 120
    qmap_nquantiles: int = 201
    qmap_clip_pred_to_cal_range: bool = True
    qmap_fallback_global: bool = True
    qmap_min_samples_bucket: int = 8
    qmap_fallback_to_season: bool = True

    # Metrics / selection
    tols: Sequence[float] = (0.25, 0.35, 0.50)

    # Alpha month-choice (baseline included)
    use_month_alpha_choice: bool = True
    alpha: float = 0.67
    acc_tol_for_score: float = 0.50
    tiebreak: str = "mae"  # "mae" or "rmse"

    # Grouped month-choice: Must divide 12 exactly.
    month_group_size: int = 1

    # Optional second-stage deseasonalization applied AFTER the normal
    # training-period monthly climatology anomaly is computed.
    # This is useful for testing whether any residual month-of-year structure
    # remains in the anomaly series.
    residual_deseasonalize: bool = False

    # Baseline options 
    # Option A:
    #   0.6 * y_anom(feature_month=t) + 0.4 * y_anom(target_month=t+H-12)
    # Option B:
    #   0.6 * y_anom(feature_month=t) + 0.4 * y_anom(feature_month=t-12)
    baseline_mode: str = "A"          # "A" or "B"
    baseline_persist_w: float = 0.6
    baseline_lastyear_w: float = 0.4

    # Output
    runs_dir: str = "runs"
    plot_test: bool = True

    # Debug / verbosity
    verbose: bool = False

    # Compatibility helpers
    @property
    def horizon(self) -> int:

        return int(self.H)

    @property
    def COVERAGE_CANDIDATES(self) -> Sequence[float]:

        return tuple(float(x) for x in self.coverage_candidates)

    def with_overrides(self, **kwargs: Any) -> "ForecasterConfig":
        data = {**self.__dict__}
        data.update(kwargs)
        return ForecasterConfig(**data)  
