import pandas as pd
from forecaster.config import ForecasterConfig
from forecaster.ml.core import build_features_and_target


def test_target_splits_do_not_overlap(fake_series_fixture):
    """
    Ensures TRAIN / VAL / TEST splits are disjoint in TARGET-month space.
    """

    cfg = ForecasterConfig(
        tavg_file="dummy.nc",
        sst_dir="dummy",
    )

    y_anom_full, time_index = fake_series_fixture

    split = build_features_and_target(
        cfg=cfg,
        y_anom_full=y_anom_full,
        time_index=time_index,
    )

    train = set(split.targ_train)
    val = set(split.targ_val)
    test = set(split.targ_test)

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
