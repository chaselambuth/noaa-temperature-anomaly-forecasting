import numpy as np
import pandas as pd
import pytest

from forecaster.config import ForecasterConfig
from forecaster.ml.core import build_features_and_target


# Deterministic RNG
@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)

# Synthetic monthly anomaly series
@pytest.fixture(scope="session")
def fake_series_fixture(rng):
    idx = pd.date_range("1950-01-01", periods=930, freq="MS")

    # A bit of structure: seasonal + slight trend + noise
    seasonal = 0.5 * np.sin(2 * np.pi * (idx.month / 12.0))
    trend = 0.001 * np.arange(len(idx))
    noise = rng.normal(0, 0.8, size=len(idx))

    y = (seasonal + trend + noise).astype(float)
    return y, idx


# Config used in tests
@pytest.fixture(scope="session")
def test_config():

    return ForecasterConfig(
        tavg_file="dummy.nc",
        sst_dir="dummy",

        use_sst=False,
        use_teleconnections=False,

        # keep QMAP enabled for calibration tests
        use_qmap=True,
        use_month_alpha_choice=False,

        # keep models minimal
        use_ridge=True,
        use_enet=False,
        use_xgb=False,
        use_lgbm=False,
        use_cat=False,
        use_rf=False,

        H=1,
        verbose=False,
        plot_test=False,
    )


# Split fixture (BacktestSplit) + attach split.pred_train for QMAP
@pytest.fixture
def split_fixture(fake_series_fixture, test_config, rng):

    y_anom_full, time_index = fake_series_fixture

    split = build_features_and_target(
        cfg=test_config,
        y_anom_full=y_anom_full,
        time_index=time_index,
        enso=None,
        ao=None,
        nao=None,
        pna=None,
        Z_sst_multi=None,
    )

    # Attach TRAIN-holdout prediction container aligned to targ_train (TARGET-month index)
    # apply_qmap_to_predictions will slice this using a cal_mask over split.targ_train.
    n_train_targ = len(pd.DatetimeIndex(split.targ_train))
    split.pred_train = {
        "ModelA": rng.normal(0, 1, size=n_train_targ).astype(float)
    }

    return split


# Fixture requested by test_pipeline_smoke.py
@pytest.fixture
def fake_pipeline_data(fake_series_fixture):
    y_anom_full, time_index = fake_series_fixture
    return {"y_anom_full": y_anom_full, "time_index": time_index}


# Monkeypatch pipeline IO so smoke tests never load real files
@pytest.fixture
def patched_pipeline(monkeypatch, fake_series_fixture):
    y_anom_full, time_index = fake_series_fixture

    def fake_land_loader(cfg):
        # pipeline expects: y_full, y_anom_full, time_index
        y_full = np.asarray(y_anom_full, float)
        y_anom = np.asarray(y_anom_full, float)
        return y_full, y_anom, pd.DatetimeIndex(time_index)

    # Patch where pipeline imports them from
    monkeypatch.setattr("forecaster.data.sources.load_land_series_and_anoms", fake_land_loader)
    monkeypatch.setattr("forecaster.data.sources.load_enso_monthly", lambda *a, **k: None)
    monkeypatch.setattr("forecaster.data.sources.load_teleconnections", lambda *a, **k: (None, None, None))
    monkeypatch.setattr("forecaster.data.sst.load_multi_region_sst_pcs", lambda *a, **k: None)

    return True
