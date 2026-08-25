import numpy as np
import pandas as pd
from forecaster.ml.selection import build_val_month_choice_table


def test_month_choice_table_structure():

    months = pd.date_range("2015-01-01", periods=36, freq="MS")

    y = np.random.randn(36)

    preds = {
        "A": np.random.randn(36),
        "B": np.random.randn(36),
    }

    table = build_val_month_choice_table(
        y_val=y,
        targ_val=months,
        pred_val_all=preds,
        alpha=0.5,
        acc_tol=0.35,
    )

    assert list(table.index) == [f"M{m:02d}" for m in range(1, 13)]
