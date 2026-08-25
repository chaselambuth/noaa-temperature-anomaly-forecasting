from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA

# Time helpers
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

# Align by month periods to prevent subtle timestamp/day mismatches.
def safe_align_xarray_to_pandas_time(x_da: xr.DataArray, t_index: pd.DatetimeIndex) -> xr.DataArray:
    t_dt = month_start(t_index)
    t_per = t_dt.to_period("M")

    xa_dt = month_start(pd.to_datetime(x_da["time"].values))
    xa_per = xa_dt.to_period("M")

    common = np.intersect1d(xa_per.values, t_per.values)
    if common.size == 0:
        raise ValueError("No overlapping months between xarray time and desired index.")

    keep = np.isin(xa_per.values, common)
    x_sel = x_da.isel(time=np.where(keep)[0])
    new_time = xa_per[keep].to_timestamp()

    x_sel = x_sel.assign_coords(time=("time", new_time))

    # drop duplicate months if any
    _, uniq_idx = np.unique(new_time.values, return_index=True)
    x_sel = x_sel.isel(time=np.sort(uniq_idx))

    # reindex to the desired month-start index
    return x_sel.reindex(time=t_dt)

# Convert 0..360 to -180..180 style.
def _lon_to_180(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

# Convert -180..180 to 0..360 style.
def _lon_to_360(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lon360 = lon % 360.0
    lon360 = np.where(lon360 < 0, lon360 + 360.0, lon360)
    return lon360

# Robust lon selection: supports datasets with lon in [-180,180] OR [0,360] and bounds that cross the dateline (e.g., 280..360 or -20..20)
def _select_lon(da: xr.DataArray, lon_bounds: Tuple[float, float]) -> xr.DataArray:
    if "lon" not in da.coords:
        raise ValueError("SST DataArray missing 'lon' coordinate")

    lon_vals = da["lon"].values.astype(float)
    lon_min = float(np.nanmin(lon_vals))
    lon_max = float(np.nanmax(lon_vals))

    lo, hi = float(lon_bounds[0]), float(lon_bounds[1])

    # Determine dataset convention
    dataset_is_360 = (lon_min >= 0.0) and (lon_max > 180.0)
    dataset_is_180 = (lon_min < 0.0) and (lon_max <= 180.0)

    # Normalize requested bounds into dataset convention
    if dataset_is_360 and (lo < 0.0 or hi <= 180.0):
        lo_n, hi_n = float(_lon_to_360(lo)), float(_lon_to_360(hi))
    elif dataset_is_180 and (lo >= 0.0 or hi > 180.0):
        lo_n, hi_n = float(_lon_to_180(lo)), float(_lon_to_180(hi))
    else:
        lo_n, hi_n = lo, hi

    # If bounds "wrap" (dateline crossing), select two slices and concat.
    wraps = lo_n > hi_n
    if not wraps:
        return da.sel(lon=slice(lo_n, hi_n))

    left = da.sel(lon=slice(lo_n, float(np.nanmax(da["lon"].values))))
    right = da.sel(lon=slice(float(np.nanmin(da["lon"].values)), hi_n))
    return xr.concat([left, right], dim="lon")

def stack_and_impute_for_pca_adaptive(
    da: xr.DataArray,
    space_dims: Tuple[str, str],
    train_end_time: pd.Timestamp,
    coverage_candidates: Iterable[float],
    *,
    fallback_keep_max: int = 5000,
    fallback_keep_min: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    cov_list = tuple(float(x) for x in coverage_candidates)

    X_stacked = da.stack(space=space_dims).transpose("time", "space")
    train_end_time = month_start_ts(train_end_time)

    X_train = X_stacked.sel(time=slice(X_stacked["time"].values.min(), train_end_time)).values
    if X_train.shape[0] == 0:
        raise ValueError("Training slice empty after alignment.")

    finite_mask = np.isfinite(X_train)
    coverage = finite_mask.mean(axis=0)

    keep_mask = None
    chosen = None
    for thr in cov_list:
        km = coverage >= float(thr)
        if int(np.sum(km)) > 0:
            keep_mask = km
            chosen = float(thr)
            break

    if keep_mask is None:
        order = np.argsort(coverage)[::-1]
        n_space = int(len(order))
        n_keep = int(min(fallback_keep_max, max(fallback_keep_min, 0.01 * n_space)))
        keep_mask = np.zeros_like(coverage, dtype=bool)
        keep_mask[order[:n_keep]] = True
        chosen = float(np.max(coverage)) if np.isfinite(np.max(coverage)) else 0.0
        print(
            f"[WARN] SST coverage filter dropped all points; "
            f"fallback keeping top {n_keep}/{n_space} cells (best coverage={chosen:.3f})."
        )

    # Info if we had to relax
    if chosen is not None and cov_list and chosen != cov_list[0]:
        print(f"[WARN] SST coverage threshold relaxed to {chosen:.2f}")

    X_keep = X_stacked.isel(space=np.where(keep_mask)[0])

    X_train_keep = X_keep.sel(time=slice(X_keep["time"].values.min(), train_end_time)).values
    col_mean = np.nanmean(X_train_keep, axis=0)

    X_all = X_keep.values
    nan_mask = ~np.isfinite(X_all)
    if nan_mask.any():
        col_idx = np.where(nan_mask)[1]
        X_all[nan_mask] = col_mean[col_idx]

    return X_all, keep_mask

# Local ERSST reader
def list_local_nc_files(folder: str) -> List[str]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"SST_DIR not found: {folder}")
    files = sorted([str(f) for f in p.glob("ersst.v6.*.nc")])
    if not files:
        files = sorted([str(f) for f in p.glob("*.nc")])
    if not files:
        raise FileNotFoundError(f"No .nc files found in {folder}")
    return files

# Extract month from filename if present
def _file_period_month(path_str: str) -> Optional[pd.Period]:
    base = Path(path_str).name
    parts = base.split(".")
    for token in parts:
        if len(token) == 6 and token.isdigit():
            try:
                return pd.Period(token, freq="M")
            except Exception:
                return None
    return None

# Open and concatenate SST data from a list of files
def open_and_concat_sst(files: List[str]) -> xr.DataArray:
    arrays = []
    for fp in files:
        try:
            ds = xr.open_dataset(fp, engine="netcdf4")
        except Exception:
            ds = xr.open_dataset(fp)  

        if "sst" not in ds:
            raise ValueError(f"'sst' not in {fp}. Vars: {list(ds.data_vars)}")

        da = sanitize_fill_values(drop_lev_if_present(ds["sst"]))
        arrays.append(da)

    return xr.concat(arrays, dim="time").sortby("time")

# Load SST PCs from local ERSST files
def load_sst_pcs_from_local_ersst(
    *,
    index: pd.DatetimeIndex,
    pca_fit_end_time: pd.Timestamp,
    sst_k: int,
    lat_bounds: Tuple[float, float],
    lon_bounds: Tuple[float, float],
    folder: str,
    pcs_cache_file: Optional[str],
    coverage_candidates: Iterable[float],
) -> np.ndarray:
    idx = month_start(index)
    pca_fit_end_time = month_start_ts(pca_fit_end_time)

    # cache load
    if pcs_cache_file:
        p = Path(pcs_cache_file)
        if p.exists():
            data = np.load(str(p))
            Z = data["Z"]
            if Z.shape[0] == len(idx) and Z.shape[1] == int(sst_k):
                print(f"[INFO] Loaded cached SST PCs: {pcs_cache_file}")
                return Z
            print("[WARN] SST PC cache mismatch; recomputing.")

    files = list_local_nc_files(folder)

    # try to trim file list if filenames include YYYYMM
    idx_min = pd.Timestamp(idx.min()).to_period("M")
    idx_max = pd.Timestamp(idx.max()).to_period("M")
    parsed = [(f, _file_period_month(f)) for f in files]
    if any(p is not None for _, p in parsed):
        kept = [f for f, p in parsed if (p is not None and (p >= idx_min) and (p <= idx_max))]
        if kept:
            files = kept

    sst = open_and_concat_sst(files)

    # Lat slice (handles increasing/decreasing lat ordering as long as bounds are correct)
    sst = sst.sel(lat=slice(float(lat_bounds[0]), float(lat_bounds[1])))

    # Robust lon slice
    sst = _select_lon(sst, lon_bounds)

    # Align to desired monthly index
    sst = safe_align_xarray_to_pandas_time(sst, idx)

    # anomaly climatology over TRAIN-fit period 
    sst_train = sst.sel(time=slice(idx.min(), pca_fit_end_time))
    clim = sst_train.groupby("time.month").mean("time", skipna=True)
    sst_anom = drop_lev_if_present(sst.groupby("time.month") - clim)

    X_all, _ = stack_and_impute_for_pca_adaptive(
        sst_anom,
        ("lat", "lon"),
        train_end_time=pca_fit_end_time,
        coverage_candidates=coverage_candidates,
    )

    t_all = month_start(pd.to_datetime(sst_anom["time"].values))
    mask_fit = t_all <= pca_fit_end_time
    X_fit = X_all[mask_fit]

    pca = PCA(n_components=int(sst_k))
    pca.fit(X_fit)

    Z = pca.transform(X_all)

    evr = float(np.sum(pca.explained_variance_ratio_))
    print(f"[INFO] SST PCA K={sst_k}, EVR sum={evr:.4f}")

    if pcs_cache_file:
        Path(pcs_cache_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pcs_cache_file, Z=Z)
        print(f"[INFO] Saved SST PCs cache: {pcs_cache_file}")

    return Z

# Load multi-region SST PCs
def load_multi_region_sst_pcs(cfg, time_index: pd.DatetimeIndex) -> Optional[Dict[str, np.ndarray]]:
    use_sst = bool(getattr(cfg, "use_sst", getattr(cfg, "USE_SST", False)))
    if not use_sst:
        return None

    folder = getattr(cfg, "sst_dir", getattr(cfg, "SST_DIR", None))
    if not folder:
        raise ValueError("cfg.sst_dir (or cfg.SST_DIR) is required when use_sst=True")

    regions = getattr(cfg, "sst_regions", getattr(cfg, "SST_REGIONS", None))
    if not regions:
        raise ValueError("cfg.sst_regions (or cfg.SST_REGIONS) is required when use_sst=True")

    cache_files = getattr(cfg, "sst_pcs_cache_files", getattr(cfg, "SST_PCS_CACHE_FILES", None))
    if cache_files is None:
        cache_files = {}
        for name, rcfg in regions.items():
            k = int(rcfg["k"])
            cache_files[name] = str(Path(folder) / f"sst_{name}_pcs_k{k}.npz")

    coverage_candidates = getattr(
        cfg,
        "coverage_candidates",
        getattr(cfg, "COVERAGE_CANDIDATES", (0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.35, 0.20, 0.10, 0.09)),
    )

    H = int(getattr(cfg, "H", getattr(cfg, "horizon", 1)))
    train_end = month_start_ts(getattr(cfg, "train_target_end", getattr(cfg, "TRAIN_TARGET_END", "2010-12-01")))
    pca_fit_end = month_start_ts(train_end - pd.DateOffset(months=H))

    # Fallback keep sizes (controls the "why only 100 cells" situation)
    fb_min = int(getattr(cfg, "sst_fallback_keep_min", 250))
    fb_max = int(getattr(cfg, "sst_fallback_keep_max", 5000))

    out: Dict[str, np.ndarray] = {}
    for name, rcfg in regions.items():
        latb = tuple(rcfg["lat"])
        lonb = tuple(rcfg["lon"])
        k = int(rcfg["k"])

        cache = None
        if isinstance(cache_files, dict):
            cache = cache_files.get(name)

        print(f"[INFO] Loading SST region {name} (K={k})")
        Z = load_sst_pcs_from_local_ersst(
            index=time_index,
            pca_fit_end_time=pca_fit_end,
            sst_k=k,
            lat_bounds=(float(latb[0]), float(latb[1])),
            lon_bounds=(float(lonb[0]), float(lonb[1])),
            folder=folder,
            pcs_cache_file=cache,
            coverage_candidates=coverage_candidates,
        )
        out[name] = Z

    stack_and_impute_for_pca_adaptive.__defaults__ = (
        stack_and_impute_for_pca_adaptive.__defaults__[:-2] + (fb_max, fb_min)
        if stack_and_impute_for_pca_adaptive.__defaults__ and len(stack_and_impute_for_pca_adaptive.__defaults__) >= 2
        else stack_and_impute_for_pca_adaptive.__defaults__
    )

    return out
