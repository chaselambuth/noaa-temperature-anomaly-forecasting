from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .evaluation import rmse, mae, tol_accuracy, month_key

# helper functions
def _validate_choice_inputs(alpha: float, tiebreak: str, month_group_size: int) -> None:
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError("alpha must be in [0, 1].")

    if tiebreak not in ("mae", "rmse"):
        raise ValueError("tiebreak must be 'mae' or 'rmse'.")

    mgs = int(month_group_size)
    if mgs < 1 or mgs > 12:
        raise ValueError("month_group_size must be an integer in [1, 12].")

    if 12 % mgs != 0:
        raise ValueError("month_group_size must divide 12 exactly (allowed: 1, 2, 3, 4, 6, 12).")


def _group_label_for_month(month: int, month_group_size: int) -> str:

    m = int(month)
    g = int(month_group_size)
    start = ((m - 1) // g) * g + 1
    end = start + g - 1
    return f"G{start:02d}_{end:02d}"


def _group_bounds_from_label(group_label: str) -> tuple[int, int]:

    body = str(group_label).strip().upper().replace("G", "")
    lo, hi = body.split("_")
    return int(lo), int(hi)


def _score_candidates_for_mask(
    y_val: np.ndarray,
    pred_val_all: Dict[str, np.ndarray],
    mask: np.ndarray,
    alpha: float,
    acc_tol: float,
    tiebreak: str,
    eps: float,
) -> tuple[str, float, float, float, float]:

    if int(mask.sum()) == 0:
        return "", np.nan, np.nan, np.nan, np.nan

    stats = []
    y_m = y_val[mask]

    for name, p in pred_val_all.items():
        p_m = p[mask]
        r = rmse(y_m, p_m)
        a = tol_accuracy(y_m, p_m, acc_tol)
        m_mae = mae(y_m, p_m)

        if not np.isfinite(a):
            a = 0.0

        stats.append((name, float(r), float(a), float(m_mae)))

    rmses = np.array([s[1] for s in stats], float)
    rmin = float(np.min(rmses))
    rmax = float(np.max(rmses))
    denom = (rmax - rmin) + float(eps)

    scored = []
    for (name, r, a, m_mae) in stats:
        r_norm = (r - rmin) / denom
        score = float(alpha) * float(r_norm) + (1.0 - float(alpha)) * (1.0 - float(a))
        scored.append((name, float(score), float(r), float(m_mae), float(a)))

    if tiebreak == "rmse":
        scored.sort(key=lambda x: (x[1], x[2]))  # score, RMSE
    else:
        scored.sort(key=lambda x: (x[1], x[3]))  # score, MAE

    winner, win_score, win_rmse, win_mae, win_acc = scored[0]
    return winner, win_score, win_rmse, win_mae, win_acc

# public API
def build_val_month_choice_table(
    y_val: np.ndarray,
    targ_val: pd.DatetimeIndex,
    pred_val_all: Dict[str, np.ndarray],
    alpha: float,
    acc_tol: float,
    tiebreak: str = "mae",  # "mae" or "rmse"
    eps: float = 1e-8,
    month_group_size: int = 1,
) -> pd.DataFrame:

    _validate_choice_inputs(alpha=float(alpha), tiebreak=str(tiebreak), month_group_size=int(month_group_size))

    dt = pd.DatetimeIndex(targ_val)
    y_val = np.asarray(y_val, float)
    pred_val_all = {k: np.asarray(v, float) for k, v in pred_val_all.items()}

    if len(dt) != len(y_val):
        raise ValueError("len(targ_val) must match len(y_val).")

    for name, arr in pred_val_all.items():
        if len(arr) != len(dt):
            raise ValueError(f"Prediction array length mismatch for model {name!r}: {len(arr)} != {len(dt)}")

    months = [f"M{m:02d}" for m in range(1, 13)]
    group_size = int(month_group_size)

    unique_groups = []
    seen = set()
    for m in range(1, 13):
        g = _group_label_for_month(m, group_size)
        if g not in seen:
            unique_groups.append(g)
            seen.add(g)

    group_results: Dict[str, dict] = {}

    for g in unique_groups:
        g_lo, g_hi = _group_bounds_from_label(g)
        group_months = set(range(g_lo, g_hi + 1))

        mask = np.array([pd.Timestamp(d).month in group_months for d in dt], dtype=bool)

        winner, score, group_rmse, group_mae, group_acc = _score_candidates_for_mask(
            y_val=y_val,
            pred_val_all=pred_val_all,
            mask=mask,
            alpha=float(alpha),
            acc_tol=float(acc_tol),
            tiebreak=str(tiebreak),
            eps=float(eps),
        )

        group_results[g] = {
            "Group": g,
            "Winner": winner,
            "Score": score,
            "RMSE": group_rmse,
            "MAE": group_mae,
            "Acc": group_acc,
            "Count": int(mask.sum()),
            "GroupStartMonth": int(g_lo),
            "GroupEndMonth": int(g_hi),
        }

    rows = []
    for m in range(1, 13):
        mkey = f"M{m:02d}"
        g = _group_label_for_month(m, group_size)
        info = group_results[g]
        rows.append(
            [
                mkey,
                info["Group"],
                info["Winner"],
                info["Score"],
                info["RMSE"],
                info["MAE"],
                info["Acc"],
                info["Count"],
                info["GroupStartMonth"],
                info["GroupEndMonth"],
            ]
        )

    return (
        pd.DataFrame(
            rows,
            columns=[
                "Month",
                "Group",
                "Winner",
                "Score",
                "RMSE",
                "MAE",
                "Acc",
                "Count",
                "GroupStartMonth",
                "GroupEndMonth",
            ],
        )
        .set_index("Month")
        .reindex(months)
    )

def apply_month_choice_table_to_test(
    choice_table: pd.DataFrame,
    targ_test: pd.DatetimeIndex,
    pred_test_all: Dict[str, np.ndarray],
    fallback: str,
) -> np.ndarray:
   
    dt = pd.DatetimeIndex(targ_test)
    pred_test_all = {k: np.asarray(v, float) for k, v in pred_test_all.items()}

    if len(pred_test_all) == 0:
        raise ValueError("pred_test_all is empty.")

    if fallback not in pred_test_all:
        fallback = sorted(pred_test_all.keys())[0]

    n = len(dt)
    for name, arr in pred_test_all.items():
        if len(arr) != n:
            raise ValueError(f"Prediction array length mismatch for model {name!r}: {len(arr)} != {n}")

    out = np.zeros(n, dtype=float)

    for i, d in enumerate(dt):
        mkey = month_key(d)

        winner = fallback
        if choice_table is not None and mkey in choice_table.index:
            w = choice_table.loc[mkey, "Winner"]
            if isinstance(w, str) and len(w) > 0:
                winner = w

        if winner not in pred_test_all:
            winner = fallback

        out[i] = float(pred_test_all[winner][i])

    return out

def month_choice_winner_map(choice_table: pd.DataFrame) -> Dict[str, str]:
  
    out: Dict[str, str] = {}
    if choice_table is None:
        return out

    for m in range(1, 13):
        k = f"M{m:02d}"
        if k in choice_table.index:
            w = choice_table.loc[k, "Winner"]
            if isinstance(w, str) and w:
                out[k] = w
    return out

def build_contiguous_month_groups(group_size: int) -> Dict[str, list[int]]:

    group_size = int(group_size)
    if group_size < 1 or group_size > 12 or 12 % group_size != 0:
        raise ValueError("group_size must divide 12 exactly (allowed: 1, 2, 3, 4, 6, 12).")

    groups: Dict[str, list[int]] = {}
    for start in range(1, 13, group_size):
        end = start + group_size - 1
        groups[f"G{start:02d}_{end:02d}"] = list(range(start, end + 1))
    return groups

def group_label_for_month(month: int, month_groups: Dict[str, Sequence[int]]) -> str:

    m = int(month)
    if m < 1 or m > 12:
        raise ValueError("month must be in [1, 12].")

    for label, months in month_groups.items():
        if m in {int(x) for x in months}:
            return str(label)
    raise ValueError(f"Month {m} not covered by month_groups.")

def rank_models_in_group(
    y_true,
    target_idx: pd.DatetimeIndex,
    pred_dict: Dict[str, np.ndarray],
    months: Sequence[int],
    *,
    acc_tol: float | None = None,
) -> pd.DataFrame:

    dt = pd.DatetimeIndex(target_idx)
    month_set = {int(m) for m in months}
    if not month_set:
        raise ValueError("months must contain at least one calendar month.")

    y_true_arr = np.asarray(y_true, float)
    if len(y_true_arr) != len(dt):
        raise ValueError("len(y_true) must match len(target_idx).")

    mask = np.array([pd.Timestamp(d).month in month_set for d in dt], dtype=bool)
    y_g = y_true_arr[mask]

    rows = []
    for name, pred in pred_dict.items():
        pred_arr = np.asarray(pred, float)
        if len(pred_arr) != len(dt):
            raise ValueError(f"Prediction array length mismatch for model {name!r}: {len(pred_arr)} != {len(dt)}")

        p_g = pred_arr[mask]
        row = {
            "Model": name,
            "RMSE": rmse(y_g, p_g),
            "MAE": mae(y_g, p_g),
            "Bias": float(np.nanmean(p_g - y_g)) if len(y_g) else np.nan,
            "Count": int(mask.sum()),
        }
        if acc_tol is not None:
            row[f"Acc@{float(acc_tol):g}"] = tol_accuracy(y_g, p_g, float(acc_tol))
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)

def build_month_choice_table(group_rankings: Dict[str, pd.DataFrame]) -> pd.DataFrame:
   
    rows = []
    for group, df in group_rankings.items():
        if df is None or df.empty:
            raise ValueError(f"No ranking rows supplied for group {group!r}.")

        winner = df.iloc[0]
        rows.append(
            {
                "Group": str(group),
                "Winner": str(winner["Model"]),
                "Val_RMSE": float(winner["RMSE"]),
                "Val_MAE": float(winner["MAE"]),
                "Val_Bias": float(winner["Bias"]),
                "Count": int(winner["Count"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Group").reset_index(drop=True)

def make_month_choice_prediction(
    targ_idx: pd.DatetimeIndex,
    pred_test_all: Dict[str, np.ndarray],
    month_choice_table: pd.DataFrame,
    month_groups: Dict[str, Sequence[int]],
    *,
    fallback: str | None = None,
) -> np.ndarray:
   
    if month_choice_table is None or month_choice_table.empty:
        raise ValueError("month_choice_table must contain at least one winner row.")
    if "Group" not in month_choice_table.columns or "Winner" not in month_choice_table.columns:
        raise ValueError("month_choice_table must contain Group and Winner columns.")
    if not pred_test_all:
        raise ValueError("pred_test_all is empty.")

    dt = pd.DatetimeIndex(targ_idx)
    pred_test_all = {k: np.asarray(v, float) for k, v in pred_test_all.items()}
    n = len(dt)

    for name, arr in pred_test_all.items():
        if len(arr) != n:
            raise ValueError(f"Prediction array length mismatch for model {name!r}: {len(arr)} != {n}")

    if fallback is not None and fallback not in pred_test_all:
        raise ValueError(f"fallback model {fallback!r} is not present in pred_test_all.")

    winner_map = dict(zip(month_choice_table["Group"].astype(str), month_choice_table["Winner"].astype(str)))
    out = np.zeros(n, dtype=float)

    for i, d in enumerate(dt):
        group = group_label_for_month(pd.Timestamp(d).month, month_groups)
        winner = winner_map.get(group, fallback)
        if winner not in pred_test_all:
            if fallback is None:
                raise KeyError(f"Winner {winner!r} for group {group!r} is not present in pred_test_all.")
            winner = fallback
        out[i] = float(pred_test_all[winner][i])

    return out

def build_strategy_table(group_rankings: Dict[str, pd.DataFrame], top_k: int) -> pd.DataFrame:

    top_k = int(top_k)
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    rows = []
    for group, df in group_rankings.items():
        if df is None or df.empty:
            raise ValueError(f"No ranking rows supplied for group {group!r}.")

        top = df.head(top_k)
        rows.append(
            {
                "Group": str(group),
                "Members": ", ".join(top["Model"].astype(str).tolist()),
                "TopK": int(top_k),
                "Leader": str(top.iloc[0]["Model"]),
                "Leader_RMSE": float(top.iloc[0]["RMSE"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Group").reset_index(drop=True)

def build_group_member_map(strategy_df: pd.DataFrame) -> Dict[str, list[str]]:

    if "Group" not in strategy_df.columns or "Members" not in strategy_df.columns:
        raise ValueError("strategy_df must contain Group and Members columns.")

    out: Dict[str, list[str]] = {}
    for _, row in strategy_df.iterrows():
        members = [x.strip() for x in str(row["Members"]).split(",") if x.strip()]
        out[str(row["Group"])] = members
    return out

def make_seasonal_ensemble_prediction(
    targ_idx: pd.DatetimeIndex,
    pred_test_all: Dict[str, np.ndarray],
    group_member_map: Dict[str, Sequence[str]],
    month_groups: Dict[str, Sequence[int]],
    *,
    fallback: str | None = None,
) -> np.ndarray:
   
    if not pred_test_all:
        raise ValueError("pred_test_all is empty.")

    dt = pd.DatetimeIndex(targ_idx)
    pred_test_all = {k: np.asarray(v, float) for k, v in pred_test_all.items()}
    n = len(dt)

    for name, arr in pred_test_all.items():
        if len(arr) != n:
            raise ValueError(f"Prediction array length mismatch for model {name!r}: {len(arr)} != {n}")

    if fallback is not None and fallback not in pred_test_all:
        raise ValueError(f"fallback model {fallback!r} is not present in pred_test_all.")

    out = np.zeros(n, dtype=float)
    for i, d in enumerate(dt):
        group = group_label_for_month(pd.Timestamp(d).month, month_groups)
        members = list(group_member_map.get(group, []))
        if not members and fallback is not None:
            members = [fallback]
        if not members:
            raise KeyError(f"No ensemble members configured for group {group!r}.")

        preds = []
        for member in members:
            if member not in pred_test_all:
                if fallback is None:
                    raise KeyError(f"Member {member!r} for group {group!r} is not present in pred_test_all.")
                member = fallback
            preds.append(float(pred_test_all[member][i]))
        out[i] = float(np.mean(preds))

    return out
