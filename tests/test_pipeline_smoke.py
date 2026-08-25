from pathlib import Path

import pandas as pd


def test_pipeline_runs_minimal(patched_pipeline, tmp_path):
    # patched_pipeline fixture monkeypatches the IO loaders

    from forecaster.pipeline import run_backtest
    from forecaster.config import ForecasterConfig

    cfg = ForecasterConfig(
        tavg_file="dummy",
        sst_dir="dummy",
        use_sst=False,
        use_teleconnections=False,
        use_qmap=False,
        use_month_alpha_choice=False,
        use_xgb=False,
        use_lgbm=False,
        use_cat=False,
        use_rf=False,
        runs_dir=str(tmp_path),
        verbose=False,
        plot_test=False,
    )

    result = run_backtest(cfg)

    # basic sanity: produced outputs
    assert result.overall_test is not None
    assert len(result.per_month_test) > 0

    run_dir = Path(result.run_id)
    assert run_dir.exists()
    assert (run_dir / "preds_test.csv").exists()
    assert (run_dir / "preds_test_all.csv").exists()

    preds_test_all = pd.read_csv(run_dir / "preds_test_all.csv")
    assert "Baseline" in preds_test_all.columns
