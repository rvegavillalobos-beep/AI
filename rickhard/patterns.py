"""When do modules fail? Shift, weekday, hour of day and start-up effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as _st

from .config import ShiftPlan
from .stats import chi2_independence, fisher_2x2, welch, wilson

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def add_time_features(df: pd.DataFrame, shifts: ShiftPlan) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        for c in ("Hour", "Weekday", "WeekdayName", "Shift", "ProdDate"):
            out[c] = pd.Series(dtype=object)
        return out
    d = pd.to_datetime(out["Date"])
    out["Hour"] = d.dt.hour
    out["Weekday"] = d.dt.dayofweek
    out["WeekdayName"] = [WEEKDAYS[i] for i in out["Weekday"]]
    out["Shift"] = shifts.assign(out["Hour"].to_numpy())
    first_start = min(shifts.starts)
    prod = d.dt.normalize()
    out["ProdDate"] = prod.where(out["Hour"] >= first_start, prod - pd.Timedelta(days=1))
    return out


GROUP_COLUMNS = ["N", "Passed", "FPY", "CI_lo", "CI_hi"]


def by_group(first: pd.DataFrame, col: str, order: list | None = None) -> pd.DataFrame:
    """FPY (with 95 % CI) and mean placement per group."""
    if first.empty or col not in first:
        return pd.DataFrame(columns=[col] + GROUP_COLUMNS)
    g = first.assign(_p=first["Status"] == "PASS").groupby(col, observed=True)
    t = pd.DataFrame({"N": g.size(), "Passed": g["_p"].sum().astype(int)}).reset_index()
    t["FPY"] = t["Passed"] / t["N"] * 100
    ci = [wilson(int(k), int(n)) for k, n in zip(t["Passed"], t["N"])]
    t["CI_lo"] = [c[0] for c in ci]
    t["CI_hi"] = [c[1] for c in ci]
    for v in ("Fit_Tx", "Fit_Ty", "Fit_Rot", "WorstAbs"):
        if v in first:
            t[v] = g[v].mean().to_numpy()
    if order is not None:
        present = set(t[col])
        cats = [o for o in order if o in present]
        t[col] = pd.Categorical(t[col], categories=cats, ordered=True)
        t = t.sort_values(col)
        t[col] = t[col].astype(str)
    return t.reset_index(drop=True)


def group_test(first: pd.DataFrame, col: str) -> float:
    """Chi-square p-value: does FPY depend on ``col``?"""
    if first.empty or col not in first:
        return float("nan")
    tab = pd.crosstab(first[col], first["Status"] == "PASS")
    return chi2_independence(tab.to_numpy())


def group_mean_test(first: pd.DataFrame, col: str, value: str) -> float:
    """Kruskal–Wallis p-value: does ``value`` differ between ``col`` groups?"""
    if first.empty or col not in first or value not in first:
        return float("nan")
    groups = [g[value].dropna().to_numpy(float) for _, g in first.groupby(col, observed=True)]
    groups = [g for g in groups if len(g) >= 5]
    if len(groups) < 2:
        return float("nan")
    try:
        return float(_st.kruskal(*groups).pvalue)
    except ValueError:
        return float("nan")


def suggest_stop_gap(first: pd.DataFrame) -> float:
    """A 'line stop' threshold: 5× the median time between modules, ≥ 60 min."""
    if len(first) < 10:
        return 60.0
    t = np.sort(pd.to_datetime(first["Date"]).to_numpy())
    gaps = np.diff(t).astype("timedelta64[s]").astype(float) / 60.0
    gaps = gaps[gaps > 0]
    med = float(np.median(gaps)) if len(gaps) else 10.0
    return float(max(60.0, round(5 * med / 10) * 10))


def heatmap(first: pd.DataFrame, value: str = "nok") -> pd.DataFrame:
    """Weekday × hour table of NOK rate (%) or volume."""
    if first.empty:
        return pd.DataFrame()
    f = first.assign(_nok=(first["Status"] != "PASS").astype(float))
    if value == "volume":
        t = f.pivot_table(index="Weekday", columns="Hour", values="_nok", aggfunc="size", fill_value=0)
    else:
        t = f.pivot_table(index="Weekday", columns="Hour", values="_nok", aggfunc="mean") * 100
    t = t.reindex(index=range(7), columns=range(24))
    t.index = WEEKDAYS
    return t


def hourly_profile(first: pd.DataFrame, value: str) -> pd.DataFrame:
    if first.empty or value not in first:
        return pd.DataFrame(columns=["Hour", "N", "Mean", "SD", "Lo", "Hi"])
    g = first.groupby("Hour")[value]
    t = pd.DataFrame({"N": g.count(), "Mean": g.mean(), "SD": g.std(ddof=1)}).reindex(range(24))
    t.index.name = "Hour"
    t = t.reset_index()
    se = t["SD"] / np.sqrt(t["N"].clip(lower=1))
    t["Lo"] = t["Mean"] - 1.96 * se
    t["Hi"] = t["Mean"] + 1.96 * se
    return t


def startup_flags(first: pd.DataFrame, gap_min: float = 60.0, k: int = 3) -> np.ndarray:
    """True for the first ``k`` modules after the line stood still for
    ``gap_min`` minutes or more (first runs, all battery types)."""
    if first.empty:
        return np.array([], bool)
    t_all = pd.to_datetime(first["Date"]).to_numpy()
    order = np.argsort(t_all, kind="stable")
    t = t_all[order]
    gaps = np.diff(t).astype("timedelta64[s]").astype(float) / 60.0
    since = np.zeros(len(t), int)
    c = 10 ** 6
    for i in range(len(t)):
        c = 0 if (i == 0 or gaps[i - 1] >= gap_min) else c + 1
        since[i] = c
    flags = np.zeros(len(t), bool)
    flags[order] = since < k
    return flags


def startup_effect(first: pd.DataFrame, gap_min: float = 60.0, k: int = 3) -> dict:
    if len(first) < 20:
        return {}
    f = startup_flags(first, gap_min, k)
    ok = (first["Status"] == "PASS").to_numpy()
    a, b = int((~ok & f).sum()), int((ok & f).sum())
    c, d = int((~ok & ~f).sum()), int((ok & ~f).sum())
    out = {
        "n_startup": int(f.sum()), "n_steady": int((~f).sum()),
        "fpy_startup": b / (a + b) * 100 if a + b else float("nan"),
        "fpy_steady": d / (c + d) * 100 if c + d else float("nan"),
        "p_value": fisher_2x2(a, b, c, d),
        "gap_min": gap_min, "k": k,
    }
    for v in ("Fit_Tx", "Fit_Ty", "Fit_Rot"):
        if v in first:
            vals = first[v].to_numpy(float)
            diff, p = welch(vals[~f], vals[f])
            out[f"{v}_shift"] = diff
            out[f"{v}_p"] = p
    return out
