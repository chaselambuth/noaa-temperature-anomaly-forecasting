import pandas as pd
from forecaster.config import ForecasterConfig
from forecaster.ml.core import build_features_and_target


def test_feature_target_alignment(fake_series_fixture):

    cfg = ForecasterConfig(
        tavg_file="dummy.nc",
        sst_dir="dummy",
        H=1,
    )

    y_anom_full, time_index = fake_series_fixture

    split = build_features_and_target(
        cfg=cfg,
        y_anom_full=y_anom_full,
        time_index=time_index,
    )

    # For each split, ensure:
    # target_month == feature_month + H
    for X, targ in [
        (split.X_train, split.targ_train),
        (split.X_val, split.targ_val),
        (split.X_test, split.targ_test),
    ]:
        feature_months = pd.DatetimeIndex(X.index)
        expected_target = feature_months + pd.DateOffset(months=cfg.H)

        assert all(
            pd.DatetimeIndex(targ) == expected_target
        )
