from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Union

import numpy as np
import pandas as pd
import xarray as xr

# Time helper functions
def month_start(x) -> pd.DatetimeIndex:
    dt = pd.to_datetime(x)
    return pd.DatetimeIndex(dt).to_period("M").to_timestamp()

def month_start_ts(x) -> pd.Timestamp:
    return pd.Timestamp(pd.to_datetime(x)).to_period("M").to_timestamp()

# Clean and preprocess xarray DataArrays to ensure consistent data handling
def drop_lev_if_present(da: xr.DataArray) -> xr.DataArray:
    if "lev" in da.dims:
        if da.sizes.get("lev", 1) == 1:
            return da.squeeze("lev", drop=True)
        return da.isel(lev=0, drop=True)
    return da

def sanitize_fill_values(da: xr.DataArray) -> xr.DataArray:
    da = da.astype("float32")
    fill_candidates = []
    for key in ["_FillValue", "missing_value"]:
        if key in da.attrs:
            try:
                fill_candidates.append(float(da.attrs[key]))
            except Exception:
                pass
    for fv in fill_candidates:
        da = da.where(da != fv)
    da = da.where(np.isfinite(da))
    da = da.where(np.abs(da) < 1e20)
    return da

# Land series
@dataclass(frozen=True)
class LandSeriesMeta:
    var_name: str
    units: Optional[str]
    time_name: str
    lat_name: str
    lon_name: str

def _infer_first_var(ds: xr.Dataset) -> str:
    if not ds.data_vars:
        raise ValueError("Dataset has no data variables.")
    return list(ds.data_vars.keys())[0]

def load_land_series_and_anoms(cfg) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    ds = xr.open_dataset(cfg.tavg_file)

    var_name = "tavg" if "tavg" in ds.data_vars else _infer_first_var(ds)

    if "lat" not in ds or "lon" not in ds:
        raise ValueError("Expected dataset coords 'lat' and 'lon' in land file.")

    tavg = drop_lev_if_present(ds[var_name])
    if "time" not in tavg.dims:
        raise ValueError("Expected a 'time' dimension in land dataset.")

    lat_name = "lat"
    lon_name = "lon"
    time_name = "time"

    lat_vals = np.asarray(ds[lat_name].values, dtype=float)
    weights_lat = xr.DataArray(
        np.cos(np.deg2rad(lat_vals)).astype("float32"),
        coords={lat_name: ds[lat_name].values},
        dims=(lat_name,),
    )

    time_vals = pd.to_datetime(ds[time_name].values)
    mean_values: list[float] = []

    for i in range(int(tavg.sizes[time_name])):
        t_slice = sanitize_fill_values(tavg.isel({time_name: i}))
        mean_da = t_slice.weighted(weights_lat).mean((lat_name, lon_name), skipna=True)
        mean_values.append(float(np.asarray(mean_da.values)))

    y_full = pd.Series(mean_values, index=time_vals).sort_index()
    y_full.index = month_start(y_full.index)
    y_full = y_full[~y_full.index.duplicated(keep="first")]

    # TRAIN-only climatology
    train_target_start = month_start_ts(getattr(cfg, "train_target_start", "1950-01-01"))
    train_target_end = month_start_ts(getattr(cfg, "train_target_end", "2010-12-01"))

    train_months = month_start(pd.date_range(train_target_start, train_target_end, freq="MS"))
    y_train_for_clim = y_full.reindex(train_months).dropna()
    if len(y_train_for_clim) < 24:
        raise ValueError("Training window too small for monthly climatology (need >= 24 months).")

    train_mu_1d = y_train_for_clim.groupby(y_train_for_clim.index.month).mean()
    y_anom_full = y_full - y_full.index.month.map(train_mu_1d)

    time_index = pd.DatetimeIndex(y_full.index)
    return y_full.values.astype(float), y_anom_full.values.astype(float), time_index

# ENSO 
def load_enso_monthly(a: Any, b: Any) -> Optional[pd.Series]:   
    if (isinstance(a, (str, Path)) or a is None) and not hasattr(b, "enso_csv"):
        csv_path = None if a is None else str(a)
        index = pd.DatetimeIndex(b)
    else:
        # Style (time_index, cfg)
        index = pd.DatetimeIndex(a)
        cfg = b
        csv_path = getattr(cfg, "enso_csv", None)

    if not csv_path or not os.path.exists(str(csv_path)):
        if csv_path:
            print(f"[WARN] ENSO file not found: {csv_path} (continuing without ENSO)")
        return None

    df = pd.read_csv(str(csv_path))
    orig_cols = list(df.columns)
    df.columns = [c.strip().lower() for c in df.columns]

    date_col = next((c for c in ["date", "time", "datetime", "month"] if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"ENSO: could not find date column. Columns: {orig_cols}")

    val_col = next(
        (c for c in ["nino34", "nina34", "anom", "anomaly", "value", "enso", "index"] if c in df.columns),
        None,
    )
    if val_col is None:
        non_date = [c for c in df.columns if c != date_col]
        if len(non_date) == 1:
            val_col = non_date[0]
        else:
            raise ValueError(f"ENSO: could not identify value column. Columns: {orig_cols}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    vals = pd.to_numeric(df[val_col], errors="coerce").replace(-99.99, np.nan)
    s = pd.Series(vals.values, index=df[date_col].values).sort_index()
    s.name = "nino34"

    s.index = month_start(s.index)
    s = s[~s.index.duplicated(keep="first")]

    idx = month_start(index)
    return s.reindex(idx).interpolate(limit_direction="both")

# TELECONNECTIONS (PSL year + 12 values)
def _read_text_from_url(url: str) -> str:
    import urllib.request
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8", errors="ignore")

def _read_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def load_psl_year12_index(path_or_url: Optional[str], index: pd.DatetimeIndex, name: str) -> Optional[pd.Series]:
    
    if not path_or_url:
        return None

    try:
        if str(path_or_url).startswith("http"):
            text = _read_text_from_url(str(path_or_url))
        else:
            if not os.path.exists(str(path_or_url)):
                print(f"[WARN] teleconnection file not found: {path_or_url}")
                return None
            text = _read_text_from_file(str(path_or_url))
    except Exception as e:
        print(f"[WARN] Failed to load {name} from {path_or_url}: {e}")
        return None

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            yr = int(parts[0])
        except Exception:
            continue

        vals = []
        ok = True
        for j in range(1, 13):
            try:
                vals.append(float(parts[j]))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        rows.append((yr, vals))

    if len(rows) == 0:
        print(f"[WARN] {name}: no usable rows parsed from {path_or_url}")
        return None

    dates = []
    values = []
    for yr, vals in rows:
        for m in range(1, 13):
            dates.append(pd.Timestamp(year=yr, month=m, day=1))
            values.append(vals[m - 1])

    s = pd.Series(values, index=pd.DatetimeIndex(dates)).sort_index()
    s = s.replace([-99.9, -99.90, -999.0, -999.00, -99.99], np.nan)
    s.index = month_start(s.index)
    s = s[~s.index.duplicated(keep="first")]
    s.name = name

    idx = month_start(index)
    return s.reindex(idx).interpolate(limit_direction="both")

def load_teleconnections(
    time_index: pd.DatetimeIndex,
    cfg,
) -> Tuple[Optional[pd.Series], Optional[pd.Series], Optional[pd.Series]]:

    if not getattr(cfg, "use_teleconnections", False):
        return None, None, None

    ao_src = getattr(cfg, "ao_url", None)
    nao_src = getattr(cfg, "nao_url", None)
    pna_src = getattr(cfg, "pna_url", None)

    ao = load_psl_year12_index(ao_src, time_index, "ao")
    nao = load_psl_year12_index(nao_src, time_index, "nao")
    pna = load_psl_year12_index(pna_src, time_index, "pna")
    return ao, nao, pna
