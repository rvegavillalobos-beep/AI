"""Re-measurements: what changes between a module's runs.

When a module is measured again shortly afterwards without rework, the
difference between the two runs is measurement (and re-seating) noise. That
gives a practical estimate of the station's repeatability and a %GRR against
the tolerance, the way an MSA study would, from production data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import BATTERY_TYPES, DEV_COLS


def run_pairs(runs: pd.DataFrame) -> pd.DataFrame:
    """Consecutive run pairs (k, k+1) of the same module, with deltas b − a."""
    if runs.empty:
        return pd.DataFrame()
    r = runs.sort_values(["PartID", "RunNum"]).reset_index(drop=True)
    nxt = r.groupby("PartID").shift(-1)
    m = nxt["RunNum"].notna().to_numpy()
    if not m.any():
        return pd.DataFrame()
    a, b = r[m], nxt[m]
    out = pd.DataFrame({
        "PartID": a["PartID"].to_numpy(),
        "BatteryType": a["BatteryType"].to_numpy(),
        "Run_a": a["RunNum"].astype(int).to_numpy(),
        "Run_b": b["RunNum"].astype(int).to_numpy(),
        "Date_a": pd.to_datetime(a["Date"]).to_numpy(),
        "Date_b": pd.to_datetime(b["Date"]).to_numpy(),
        "Status_a": a["Status"].to_numpy(),
        "Status_b": b["Status"].to_numpy(),
        "Worst_a": a["WorstAbs"].to_numpy(float),
        "Worst_b": b["WorstAbs"].to_numpy(float),
    })
    out["GapMin"] = (out["Date_b"] - out["Date_a"]).dt.total_seconds() / 60.0
    for c in DEV_COLS:
        out[f"a_{c}"] = a[c].to_numpy(float)
        out[f"b_{c}"] = b[c].to_numpy(float)
        out[f"d_{c}"] = out[f"b_{c}"] - out[f"a_{c}"]
    return out


REP_COLUMNS = ["BatteryType", "Coord", "N", "MeanDelta", "SDDelta", "SigmaRepeat", "GRR", "Verdict"]


def repeatability(pairs: pd.DataFrame, limit: float, max_gap_min: float = 120.0) -> pd.DataFrame:
    """σ_repeat per type and coordinate from short-gap pairs: SD(Δ)/√2
    (robust SD via the MAD), and %GRR = 6·σ_repeat / tolerance width."""
    if pairs.empty:
        return pd.DataFrame(columns=REP_COLUMNS)
    p = pairs[pairs["GapMin"] <= max_gap_min]
    rows = []
    for t in BATTERY_TYPES:
        sub = p[p["BatteryType"] == t]
        for c in DEV_COLS:
            d = sub[f"d_{c}"].to_numpy(float)
            d = d[np.isfinite(d)]
            if len(d) < 5:
                continue
            sd = float(np.median(np.abs(d - np.median(d))) / 0.6745) or float(d.std(ddof=1))
            sr = sd / np.sqrt(2)
            grr = 6 * sr / (2 * limit) * 100
            verdict = "Good" if grr < 10 else ("Acceptable" if grr < 30 else "Not acceptable")
            rows.append({"BatteryType": t, "Coord": c, "N": len(d), "MeanDelta": float(np.mean(d)),
                         "SDDelta": sd, "SigmaRepeat": sr, "GRR": grr, "Verdict": verdict})
    return pd.DataFrame(rows, columns=REP_COLUMNS)


def transitions(pairs: pd.DataFrame, first_only: bool = True) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    p = pairs[pairs["Run_a"] == 1] if first_only else pairs
    if p.empty:
        return pd.DataFrame()
    order = ["PASS", "FAIL", "INCOMPLETE"]
    t = pd.crosstab(p["Status_a"], p["Status_b"])
    return t.reindex(index=[o for o in order if o in t.index], columns=[o for o in order if o in t.columns],
                     fill_value=0)


def summary(pairs: pd.DataFrame, rep: pd.DataFrame) -> dict:
    out: dict = {}
    if pairs.empty:
        return out
    p1 = pairs[pairs["Run_a"] == 1]
    nok = p1[p1["Status_a"] != "PASS"]
    out["pairs"] = int(len(pairs))
    out["modules_remeasured"] = int(pairs["PartID"].nunique())
    out["nok_remeasured"] = int(len(nok))
    out["nok_to_ok"] = int((nok["Status_b"] == "PASS").sum())
    out["nok_to_ok_rate"] = out["nok_to_ok"] / len(nok) * 100 if len(nok) else float("nan")
    out["median_gap_min"] = float(pairs["GapMin"].median())
    if len(rep):
        out["median_sigma"] = float(rep["SigmaRepeat"].median())
        out["max_grr"] = float(rep["GRR"].max())
        out["median_grr"] = float(rep["GRR"].median())
        worst = rep.sort_values("GRR").iloc[-1]
        out["worst_coord"] = f"{worst['BatteryType']} {worst['Coord'].replace('_', ' ')}"
    return out
