"""Interventions log and before/after impact analysis.

Log what was done to the line (an end-stop moved, a guide realigned, a new
tool) and the app tells you, with statistics, whether it changed anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import DEV_COLS
from .stats import fisher_2x2, levene, welch, wilson

EVENT_COLUMNS = ["Date", "Scope", "Category", "Title", "Notes", "dX", "dY", "dYaw"]
CATEGORIES = ["Adjustment", "Maintenance", "Tooling", "Material", "Training", "Software", "Other"]
SCOPES = ["All", "Type S", "Type M"]


def normalize_events(records) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame(list(records) if records is not None else [])
    for c in EVENT_COLUMNS:
        if c not in df:
            df[c] = np.nan if c in ("dX", "dY", "dYaw") else ""
    df = df[EVENT_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df["Scope"] = df["Scope"].where(df["Scope"].isin(SCOPES), "All")
    df["Category"] = df["Category"].where(df["Category"].isin(CATEGORIES), "Other")
    for c in ("Title", "Notes"):
        df[c] = df[c].fillna("").astype(object).map(str)
    df.loc[df["Title"].str.strip() == "", "Title"] = "Untitled event"
    for c in ("dX", "dY", "dYaw"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def events_to_records(df) -> list[dict]:
    d = normalize_events(df)
    out = []
    for r in d.to_dict("records"):
        r["Date"] = pd.Timestamp(r["Date"]).isoformat()
        for c in ("dX", "dY", "dYaw"):
            r[c] = None if pd.isna(r[c]) else float(r[c])
        out.append(r)
    return out


def scope_types(scope: str) -> tuple[str, ...]:
    return ("Type S", "Type M") if scope == "All" else (scope,)


BA_VARIABLES = [("Fit_Tx", "Shift X (mm)"), ("Fit_Ty", "Shift Y (mm)"), ("Fit_Rot", "Yaw (°)"),
                ("ShapeMax", "Shape residual (mm)"), ("WorstAbs", "Worst corner |dev| (mm)")]


def before_after(first: pd.DataFrame, when, scope: str = "All", window_days: float = 14) -> dict:
    """Compare the ``window_days`` before and after an event."""
    when = pd.Timestamp(when)
    d = first[first["BatteryType"].isin(scope_types(scope))]
    dates = pd.to_datetime(d["Date"])
    before = d[(dates < when) & (dates >= when - pd.Timedelta(days=window_days))]
    after = d[(dates >= when) & (dates < when + pd.Timedelta(days=window_days))]
    out: dict = {"n_before": len(before), "n_after": len(after), "when": when, "window_days": window_days,
                 "before": before, "after": after}
    if len(before) < 5 or len(after) < 5:
        out["enough"] = False
        return out
    out["enough"] = True
    pb, pa = int((before["Status"] == "PASS").sum()), int((after["Status"] == "PASS").sum())
    out["fpy_before"], out["fpy_after"] = pb / len(before) * 100, pa / len(after) * 100
    out["fpy_before_ci"], out["fpy_after_ci"] = wilson(pb, len(before)), wilson(pa, len(after))
    out["fpy_p"] = fisher_2x2(len(before) - pb, pb, len(after) - pa, pa)
    rows = []
    variables = BA_VARIABLES + [(c, c.replace("_", " ") + " (mm)") for c in DEV_COLS]
    for col, label in variables:
        if col not in d:
            continue
        a, b = before[col].to_numpy(float), after[col].to_numpy(float)
        diff, p = welch(a, b)
        ratio, p_sd = levene(a, b)
        rows.append({"Variable": label, "Column": col,
                     "Before": float(np.nanmean(a)) if np.isfinite(a).any() else np.nan,
                     "After": float(np.nanmean(b)) if np.isfinite(b).any() else np.nan,
                     "Change": diff, "p (mean)": p,
                     "SD before": float(np.nanstd(a, ddof=1)) if np.isfinite(a).sum() > 1 else np.nan,
                     "SD after": float(np.nanstd(b, ddof=1)) if np.isfinite(b).sum() > 1 else np.nan,
                     "SD ratio": ratio, "p (spread)": p_sd})
    out["table"] = pd.DataFrame(rows)
    gain = out["fpy_after"] - out["fpy_before"]
    sig = np.isfinite(out["fpy_p"]) and out["fpy_p"] < 0.05
    t = out["table"]
    moved = t[(t["p (mean)"] < 0.05) & t["Column"].isin(["Fit_Tx", "Fit_Ty", "Fit_Rot"])]
    tighter = t[(t["p (spread)"] < 0.05) & (t["SD ratio"] < 1) & t["Column"].isin(["Fit_Tx", "Fit_Ty", "Fit_Rot"])]
    if sig and gain > 0:
        verdict = ("good", f"FPY improved by {gain:+.1f} pp — statistically significant.")
    elif sig and gain < 0:
        verdict = ("bad", f"FPY dropped by {gain:+.1f} pp — statistically significant.")
    elif len(moved) or len(tighter):
        what = ", ".join(sorted(set(moved["Variable"]) | set(tighter["Variable"])))
        verdict = ("info", f"The process changed ({what}), but the FPY change ({gain:+.1f} pp) is not yet "
                           "significant. More modules will tell.")
    else:
        verdict = ("neutral", f"No significant change detected ({gain:+.1f} pp FPY).")
    out["verdict"] = verdict
    return out


def rolling_series(first: pd.DataFrame, column: str, window: int = 25) -> pd.DataFrame:
    d = first.sort_values("Date")
    if column == "PASS":
        s = (d["Status"] == "PASS").astype(float) * 100
    else:
        s = d[column].astype(float)
    return pd.DataFrame({"Date": pd.to_datetime(d["Date"]).to_numpy(),
                         "Value": s.rolling(window, min_periods=max(3, window // 3)).mean().to_numpy(),
                         "Raw": s.to_numpy()})
