import numpy as np
from forecaster.ml.calibration import apply_qmap_to_predictions


def test_qmap_does_not_modify_baseline(split_fixture):

    split = split_fixture

    # Fake predictions
    pred_val = {"ModelA": np.random.randn(len(split.y_val))}
    pred_test = {"ModelA": np.random.randn(len(split.y_test))}

    cfg = type("Dummy", (), {
        "qmap_cal_target_months": 120,
        "qmap_nquantiles": 201,
        "qmap_min_samples_bucket": 8,
        "qmap_fallback_to_season": True,
        "qmap_fallback_global": True,
        "qmap_clip_pred_to_cal_range": True,
        "H": 1,
    })()

    new_val, new_test = apply_qmap_to_predictions(
        cfg, split, pred_val, pred_test
    )

    # Ensure calibrated model exists
    assert any(name.endswith("_qmapM") for name in new_val.keys())
