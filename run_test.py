from __future__ import annotations

from pathlib import Path

from src.forecaster.config import ForecasterConfig
from src.forecaster.pipeline import run_backtest


def main() -> None:
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "noaa_raw_data"

    cfg = ForecasterConfig(
        # Required paths, relative to this repository
        tavg_file=str(data_dir / "nclimgrid_tavg.nc"),
        sst_dir=str(data_dir / "sst_cache"),

        # Optional external data
        enso_csv=str(data_dir / "nina34.anom.csv"),

        # Full experiment settings
        use_qmap=True,
        use_month_alpha_choice=True,

        # Enable models
        use_ridge=True,
        use_enet=True,
        use_xgb=True,
        use_lgbm=True,
        use_cat=True,
        use_rf=False,

        # Output
        plot_test=True,
        verbose=True,
    )

    run_backtest(cfg)
    print("Full backtest finished")


if __name__ == "__main__":
    main()
