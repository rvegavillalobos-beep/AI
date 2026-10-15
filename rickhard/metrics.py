"""Quality KPIs: first runs, FPY, Pareto, failure mechanisms, capability, shape link."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import BATTERY_TYPES, CORNERS, DEV_COLS, MECHANISMS
from .stats import cpk_ci, fisher_2x2, normal_tail_ppm, wilson

# ---------------------------------------------------------------------------
# First runs and FPY
# ---------------------------------------------------------------------------

def first_runs(runs: pd.DataFrame, incomplete_as_fail: bool) -> pd.DataFrame:
    """The first measurement (Run 1) of every module.

    With ``incomplete_as_fail`` a first run with a missing corner is a FAIL
    (strict NOK, as in the original app); otherwise it stays INCOMPLETE.
    """
    if runs.empty:
        return runs.copy()
    fr = (
        runs.sort_values(["RunNum", "Date"], kind="stable")
        .drop_duplicates("PartID", keep="first")
        .sort_values("Date", kind="stable")
        .copy()
    )
    if incomplete_as_fail:
        fr.loc[fr["MissingCorners"] > 0, "Status"] = "FAIL"
    return fr.reset_index(drop=True)


def fpy(df: pd.DataFrame) -> float:
    return float((df["Status"] == "PASS").mean() * 100) if len(df) else float("nan")


def fpy_ci(df: pd.DataFrame) -> tuple[float, float]:
    return wilson(int((df["Status"] == "PASS").sum()), len(df))


def status_counts(df: pd.DataFrame) -> dict:
    vc = df["Status"].value_counts() if len(df) else pd.Series(dtype=int)
    return {s: int(vc.get(s, 0)) for s in ("PASS", "FAIL", "INCOMPLETE")}


WEEKLY_COLUMNS = ["WeekKey", "CalendarWeek", "Total", "Passed", "Failed", "Incomplete", "PassRate",
                  "CI_lo", "CI_hi", "LowSample", "MA_FPY", "NearMiss", "NearMissRate"]


def weekly_fpy(first: pd.DataFrame, ma_window: int = 3, low_sample: int = 5,
               near_miss: float | None = None) -> pd.DataFrame:
    if first.empty:
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    f = first.assign(_p=first["Status"] == "PASS", _f=first["Status"] == "FAIL",
                     _i=first["Status"] == "INCOMPLETE")
    if near_miss is not None and "Margin" in f:
        f["_nm"] = f["_p"] & (f["Margin"] < near_miss)
    else:
        f["_nm"] = False
    g = f.groupby(["WeekKey", "CalendarWeek"], sort=True)
    w = pd.DataFrame({
        "Total": g.size(),
        "Passed": g["_p"].sum().astype(int),
        "Failed": g["_f"].sum().astype(int),
        "Incomplete": g["_i"].sum().astype(int),
        "NearMiss": g["_nm"].sum().astype(int),
    }).reset_index().sort_values("WeekKey")
    w["PassRate"] = w["Passed"] / w["Total"] * 100
    ci = [wilson(int(k), int(n)) for k, n in zip(w["Passed"], w["Total"])]
    w["CI_lo"] = [c[0] for c in ci]
    w["CI_hi"] = [c[1] for c in ci]
    w["LowSample"] = w["Total"] < low_sample
    w["MA_FPY"] = w["PassRate"].rolling(window=max(1, int(ma_window)), min_periods=1).mean()
    w["NearMissRate"] = w["NearMiss"] / w["Total"] * 100
    return w[WEEKLY_COLUMNS].reset_index(drop=True)


def daily_fpy(first: pd.DataFrame) -> pd.DataFrame:
    if first.empty:
        return pd.DataFrame(columns=["Day", "Total", "Passed", "PassRate"])
    d = first.assign(Day=pd.to_datetime(first["Date"]).dt.normalize(), _p=first["Status"] == "PASS")
    g = d.groupby("Day")
    out = pd.DataFrame({"Total": g.size(), "Passed": g["_p"].sum()}).reset_index()
    out["PassRate"] = out["Passed"] / out["Total"] * 100
    return out


# ---------------------------------------------------------------------------
# Why modules fail
# ---------------------------------------------------------------------------

def corner_pareto(first: pd.DataFrame, limit: float) -> pd.DataFrame:
    """Which corner/axis is out of spec in NOK modules (a module can count in
    several bars). Missing corners are their own cause."""
    cols = ["Cause", "Corner", "Axis", "Modules", "Share"]
    nok = first[first["Status"] != "PASS"]
    if nok.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for c in CORNERS:
        for a in ("X", "Y"):
            v = nok[f"{c}_{a}"].to_numpy(float)
            with np.errstate(invalid="ignore"):
                n = int((np.abs(v) > limit).sum())
            if n:
                rows.append({"Cause": f"{c} {a}", "Corner": c, "Axis": a, "Modules": n})
    n_missing = int((nok["MissingCorners"] > 0).sum())
    if n_missing:
        rows.append({"Cause": "Missing corner", "Corner": "-", "Axis": "-", "Modules": n_missing})
    out = pd.DataFrame(rows, columns=cols[:-1])
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["Share"] = out["Modules"] / len(nok) * 100
    return out.sort_values("Modules", ascending=False).reset_index(drop=True)


def mechanism_summary(first: pd.DataFrame) -> pd.DataFrame:
    """NOK first runs by failure mechanism and battery type."""
    nok = first[first["Status"] != "PASS"]
    if nok.empty or "Mechanism" not in nok:
        return pd.DataFrame(columns=["Mechanism", "BatteryType", "Modules", "Share"])
    t = nok.groupby(["Mechanism", "BatteryType"]).size().rename("Modules").reset_index()
    tot = t.groupby("BatteryType")["Modules"].transform("sum")
    t["Share"] = t["Modules"] / tot * 100
    t["_o"] = t["Mechanism"].map({m: i for i, m in enumerate(MECHANISMS)})
    return t.sort_values(["BatteryType", "_o"]).drop(columns="_o").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Capability
# ---------------------------------------------------------------------------

def _diagnose(offset: float, pp: float, ppk: float, half_tol: float) -> str:
    """Offset (systematic, fixable by compensation) and/or scatter (spread,
    not fixable by an offset)."""
    is_offset = abs(offset) >= half_tol / 4
    scatter = np.isfinite(pp) and pp < 1.0
    if is_offset and scatter:
        return "Offset + scatter"
    if is_offset:
        return "Offset"
    if scatter:
        return "Scatter"
    if np.isfinite(ppk) and ppk >= 1.33:
        return "Capable"
    return "Marginal"


def capability_series(v, lo: float, hi: float) -> dict:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    n = len(v)
    if n == 0:
        return {"N": 0}
    mu = float(v.mean())
    sd = float(v.std(ddof=1)) if n > 1 else float("nan")
    mr = np.abs(np.diff(v))
    sw = float(mr.mean() / 1.128) if len(mr) else float("nan")
    tol = hi - lo
    cp = tol / (6 * sw) if sw and sw > 0 else float("nan")
    cpk = min(hi - mu, mu - lo) / (3 * sw) if sw and sw > 0 else float("nan")
    pp = tol / (6 * sd) if sd and sd > 0 else float("nan")
    ppk = min(hi - mu, mu - lo) / (3 * sd) if sd and sd > 0 else float("nan")
    lo_ci, hi_ci = cpk_ci(ppk, n)
    return {
        "N": n, "Mean": mu, "Std": sd, "StdWithin": sw, "Min": float(v.min()), "Max": float(v.max()),
        "P05": float(np.percentile(v, 5)), "P95": float(np.percentile(v, 95)),
        "Cp": cp, "Cpk": cpk, "Pp": pp, "Ppk": ppk, "PpkLo": lo_ci, "PpkHi": hi_ci,
        "PPM": normal_tail_ppm(mu, sd, lo, hi),
        "PPMCentered": normal_tail_ppm((lo + hi) / 2, sd, lo, hi),
        "OutPct": float(((v < lo) | (v > hi)).mean() * 100),
    }


CAP_COLUMNS = ["BatteryType", "Corner", "Axis", "Coord", "N", "Mean", "Std", "StdWithin", "Min", "Max",
               "P05", "P95", "Cp", "Cpk", "Pp", "Ppk", "PpkLo", "PpkHi", "PPM", "PPMCentered", "OutPct",
               "Diagnosis"]


def capability(first: pd.DataFrame, limit: float, z_limits: tuple[float, float] | None = None) -> pd.DataFrame:
    """Per battery type, corner and axis: mean, σ (overall and within),
    Cp/Cpk (short-term, σ from moving ranges), Pp/Ppk (long-term) with a 95 %
    CI, expected PPM and a plain diagnosis."""
    rows = []
    axes = ["X", "Y"] + (["Z"] if z_limits is not None and "FL_Z" in first else [])
    for t in BATTERY_TYPES:
        sub = first[first["BatteryType"] == t].sort_values("Date")
        if sub.empty:
            continue
        for c in CORNERS:
            for a in axes:
                lo, hi = (z_limits if a == "Z" else (-limit, limit))
                r = capability_series(sub[f"{c}_{a}"].to_numpy(float), lo, hi)
                if r["N"] == 0:
                    continue
                r.update({"BatteryType": t, "Corner": c, "Axis": a, "Coord": f"{c}_{a}",
                          "Diagnosis": _diagnose(r["Mean"] - (lo + hi) / 2, r["Pp"], r["Ppk"], (hi - lo) / 2)})
                rows.append(r)
    return pd.DataFrame(rows, columns=CAP_COLUMNS)


# ---------------------------------------------------------------------------
# Deformation ↔ NOK
# ---------------------------------------------------------------------------

DIAG_BANDS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, np.inf]


def shape_link(sq_first: pd.DataFrame, diag_tol: float) -> dict:
    """How deformation relates to NOK on complete first runs (PASS/FAIL)."""
    d = sq_first[sq_first["Status"].isin(["PASS", "FAIL"])] if len(sq_first) else sq_first
    out: dict = {"n": int(len(d))}
    if d.empty:
        return out
    fail = (d["Status"] == "FAIL").to_numpy()
    deformed = (d["SquarenessStatus"] == "DEFORMED").to_numpy()
    a, b = int((deformed & fail).sum()), int((deformed & ~fail).sum())
    c, e = int((~deformed & fail).sum()), int((~deformed & ~fail).sum())
    out.update({
        "n_fail": int(fail.sum()), "n_deformed": a + b, "n_square": c + e,
        "fail_rate_deformed": a / (a + b) * 100 if a + b else float("nan"),
        "fail_rate_square": c / (c + e) * 100 if c + e else float("nan"),
        "p_value": fisher_2x2(a, b, c, e),
        "fpy": float((~fail).mean() * 100),
    })
    if "PassIfSquare" in d:
        pis = d["PassIfSquare"].to_numpy(bool)
        out["nok_if_square"] = int((fail & pis).sum())
        out["fpy_if_square"] = float(pis.mean() * 100)
    if "PassIfPlaced" in d:
        out["fpy_if_placed"] = float(d["PassIfPlaced"].to_numpy(bool).mean() * 100)
    bands = pd.cut(d["DeltaDiag"], DIAG_BANDS, right=False)
    by_band = (d.assign(Band=bands, Fail=fail).groupby("Band", observed=True)
               .agg(N=("Fail", "size"), Fails=("Fail", "sum")).reset_index())
    by_band["FailRate"] = by_band["Fails"] / by_band["N"] * 100
    ci = [wilson(int(k), int(n)) for k, n in zip(by_band["Fails"], by_band["N"])]
    by_band["CI_lo"] = [x[0] for x in ci]
    by_band["CI_hi"] = [x[1] for x in ci]
    by_band["Label"] = [f"{iv.left:g}–{iv.right:g}" if np.isfinite(iv.right) else f"≥ {iv.left:g}"
                        for iv in by_band["Band"]]
    by_band["AboveTol"] = [iv.left >= diag_tol for iv in by_band["Band"]]
    out["by_band"] = by_band.drop(columns="Band")
    return out


def weekly_deformation(sq: pd.DataFrame) -> pd.DataFrame:
    cols = ["WeekKey", "CalendarWeek", "BatteryType", "N", "Deformed", "Share", "MedianDiag"]
    if sq.empty:
        return pd.DataFrame(columns=cols)
    g = sq.assign(_d=sq["SquarenessStatus"] == "DEFORMED").groupby(["WeekKey", "CalendarWeek", "BatteryType"])
    w = pd.DataFrame({"N": g.size(), "Deformed": g["_d"].sum().astype(int),
                      "MedianDiag": g["DeltaDiag"].median()}).reset_index().sort_values("WeekKey")
    w["Share"] = w["Deformed"] / w["N"] * 100
    return w[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Headline KPIs
# ---------------------------------------------------------------------------

def kpis(first: pd.DataFrame, runs: pd.DataFrame, weekly: pd.DataFrame, cap: pd.DataFrame,
         sq_first: pd.DataFrame, near_miss: float, target: float) -> dict:
    n = len(first)
    counts = status_counts(first)
    out = {
        "modules": n,
        "runs": int(len(runs)),
        "remeasured": int(runs.loc[runs["RunNum"] > 1, "PartID"].nunique()) if len(runs) else 0,
        "pass": counts["PASS"], "fail": counts["FAIL"], "incomplete": counts["INCOMPLETE"],
        "fpy": fpy(first),
        "fpy_ci": fpy_ci(first),
        "gap": fpy(first) - target if n else float("nan"),
        "spark": [],
    }
    passed = first[first["Status"] == "PASS"]
    out["near_miss"] = int((passed["Margin"] < near_miss).sum()) if len(passed) else 0
    out["near_miss_rate"] = out["near_miss"] / n * 100 if n else float("nan")
    out["median_margin"] = float(first["Margin"].median()) if n else float("nan")
    w = weekly[weekly["Total"] > 0] if len(weekly) else weekly
    if len(w):
        last = w.iloc[-1]
        out["last_week"] = str(last["CalendarWeek"])
        out["last_week_fpy"] = float(last["PassRate"])
        out["last_week_n"] = int(last["Total"])
        prev = w.iloc[-5:-1]
        prev_n = prev["Total"].sum()
        out["prev_fpy"] = float(prev["Passed"].sum() / prev_n * 100) if prev_n else float("nan")
        out["spark"] = w["PassRate"].tail(12).tolist()
    if len(cap):
        cxy = cap[cap["Axis"].isin(["X", "Y"])].dropna(subset=["Ppk"])
        if len(cxy):
            worst = cxy.sort_values("Ppk").iloc[0]
            out["worst_ppk"] = float(worst["Ppk"])
            out["worst_coord"] = f"{worst['BatteryType']} {worst['Corner']} {worst['Axis']}"
    if len(sq_first):
        out["deformed_rate"] = float((sq_first["SquarenessStatus"] == "DEFORMED").mean() * 100)
    return out


def worst_coordinate_share(first: pd.DataFrame) -> pd.DataFrame:
    """How often each corner/axis is the module's worst coordinate (%)."""
    if first.empty or "WorstCoord" not in first:
        return pd.DataFrame({"Coord": list(DEV_COLS), "Share": 0.0})
    s = first["WorstCoord"].dropna().value_counts(normalize=True) * 100
    return pd.DataFrame({"Coord": list(DEV_COLS), "Share": [float(s.get(c, 0.0)) for c in DEV_COLS]})
