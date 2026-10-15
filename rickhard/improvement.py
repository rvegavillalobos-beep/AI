"""Error-anatomy summaries and the what-if simulator ("path to target").

Every complete module's deviations are written in the orthonormal mode basis
(see ``geometry.mode_basis``). Levers act on those coefficients:

* compensation: shifts the placement modes by a constant,
* placement variation: shrinks shift X/Y and yaw towards their mean,
* shape variation / shape bias: shrinks or removes the non-rigid modes.

The modified deviations are re-judged against the tolerance, so the
predicted FPY is computed on your own modules, not on a distribution model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .config import Nominals, normalize_type
from .constants import BATTERY_TYPES, CORNERS, MODE_LABELS, MODES, SHAPE_MODES
from .geometry import dev_matrix, mode_basis

A_COLS = [f"A_{m}" for m in MODES]


# ---------------------------------------------------------------------------
# Variance / bias budget
# ---------------------------------------------------------------------------

BUDGET_COLUMNS = ["BatteryType", "Mode", "Label", "Bias", "SD", "VarShare", "BiasShare", "TotalShare", "N"]


def budget(vec: pd.DataFrame) -> pd.DataFrame:
    """Per type and mode: bias (mean amplitude), spread (σ), and their share
    of the total mean-square corner deviation."""
    rows = []
    for t in BATTERY_TYPES:
        sub = vec[(vec["BatteryType"] == t)][A_COLS].dropna()
        if len(sub) < 3:
            continue
        mean = sub.mean()
        var = sub.var(ddof=1)
        tot_var = float(var.sum())
        tot_bias = float((mean ** 2).sum())
        tot = tot_var + tot_bias
        for m in MODES:
            c = f"A_{m}"
            rows.append({
                "BatteryType": t, "Mode": m, "Label": MODE_LABELS[m],
                "Bias": float(mean[c]), "SD": float(np.sqrt(var[c])),
                "VarShare": float(var[c] / tot_var * 100) if tot_var else np.nan,
                "BiasShare": float(mean[c] ** 2 / tot_bias * 100) if tot_bias else np.nan,
                "TotalShare": float((var[c] + mean[c] ** 2) / tot * 100) if tot else np.nan,
                "N": int(len(sub)),
            })
    return pd.DataFrame(rows, columns=BUDGET_COLUMNS)


def mode_fail_link(vec: pd.DataFrame) -> pd.DataFrame:
    """NOK rate in the bottom vs top quartile of |amplitude − mean| of each mode."""
    rows = []
    d = vec[vec["Status"].isin(["PASS", "FAIL"])]
    for t in BATTERY_TYPES:
        sub = d[d["BatteryType"] == t].dropna(subset=A_COLS)
        if len(sub) < 40:
            continue
        fail = (sub["Status"] == "FAIL").to_numpy()
        for m in MODES:
            a = np.abs(sub[f"A_{m}"].to_numpy(float) - sub[f"A_{m}"].mean())
            q1, q3 = np.percentile(a, [25, 75])
            lo, hi = fail[a <= q1], fail[a >= q3]
            rows.append({"BatteryType": t, "Mode": m,
                         "FailLow": lo.mean() * 100 if len(lo) else np.nan,
                         "FailHigh": hi.mean() * 100 if len(hi) else np.nan,
                         "Lift": (hi.mean() - lo.mean()) * 100 if len(lo) and len(hi) else np.nan})
    return pd.DataFrame(rows, columns=["BatteryType", "Mode", "FailLow", "FailHigh", "Lift"])


def weekly_modes(vec: pd.DataFrame) -> pd.DataFrame:
    if vec.empty:
        return pd.DataFrame()
    g = vec.groupby(["BatteryType", "WeekKey", "CalendarWeek"])
    mean = g[A_COLS].mean().add_suffix("_mean")
    sd = g[A_COLS].std().add_suffix("_sd")
    n = g.size().rename("N")
    return pd.concat([mean, sd, n], axis=1).reset_index().sort_values(["BatteryType", "WeekKey"])


# ---------------------------------------------------------------------------
# PCA (data-driven modes)
# ---------------------------------------------------------------------------

def pca(vec: pd.DataFrame, bat_type: str, nominals: Nominals, k: int = 4) -> dict | None:
    sub = vec[vec["BatteryType"] == bat_type]
    D = dev_matrix(sub)
    D = D[~np.isnan(D).any(axis=1)]
    if len(D) < 15:
        return None
    mu = D.mean(axis=0)
    X = D - mu
    _, s, Vt = np.linalg.svd(X, full_matrices=False)
    var = s ** 2 / (len(D) - 1)
    share = var / var.sum() * 100
    E, _ = mode_basis(bat_type, nominals)
    comps = []
    for i in range(min(k, len(s))):
        v = Vt[i]
        proj = (E.T @ v) ** 2
        proj = proj / proj.sum() * 100
        order = np.argsort(proj)[::-1]
        label = " + ".join(f"{MODE_LABELS[MODES[j]]} {proj[j]:.0f}%" for j in order[:2] if proj[j] >= 15)
        comps.append({"k": i + 1, "share": float(share[i]), "sd": float(np.sqrt(var[i])),
                      "vector": v, "mix": {MODES[j]: float(proj[j]) for j in range(8)},
                      "label": label or MODE_LABELS[MODES[order[0]]]})
    return {"mean": mu, "components": comps, "n": int(len(D))}


# ---------------------------------------------------------------------------
# What-if levers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Levers:
    compensate: bool = False             # apply the per-type compensation
    shift_reduction: float = 0.0         # 0..1, shrinks shift X/Y spread
    yaw_reduction: float = 0.0           # 0..1, shrinks yaw spread
    shape_reduction: float = 0.0         # 0..1, shrinks shape-mode spread
    remove_shape_bias: bool = False      # systematic shape → 0


def apply_levers(vec: pd.DataFrame, levers: Levers, comps: dict | None, nominals: Nominals) -> np.ndarray:
    """(n, 8) deviations after the levers (NaN rows for incomplete runs)."""
    D = dev_matrix(vec)
    out = np.full_like(D, np.nan)
    if not len(vec):
        return out
    ok = ~np.isnan(D).any(axis=1)
    types = vec["BatteryType"].map(normalize_type).to_numpy()
    for t in np.unique(types):
        m = ok & (types == t)
        if not m.any():
            continue
        E, rot_norm = mode_basis(t, nominals)
        C = D[m] @ E
        if levers.compensate and comps and comps.get(t) is not None and np.isfinite(comps[t]).all():
            dx, dy, yaw = comps[t]
            C[:, 0] += 2.0 * dx
            C[:, 1] += 2.0 * dy
            C[:, 2] += np.radians(yaw) * rot_norm
        mean = C.mean(axis=0)
        for j, mode in enumerate(MODES):
            r = 0.0
            if mode in ("Tx", "Ty"):
                r = levers.shift_reduction
            elif mode == "Rot":
                r = levers.yaw_reduction
            elif mode in SHAPE_MODES:
                r = levers.shape_reduction
            C[:, j] = mean[j] + (1.0 - r) * (C[:, j] - mean[j])
            if mode in SHAPE_MODES and levers.remove_shape_bias:
                C[:, j] -= mean[j]
        out[m] = C @ E.T
    return out


def fpy_after(vec: pd.DataFrame, levers: Levers, comps: dict | None, limit: float, nominals: Nominals) -> float:
    if not len(vec):
        return float("nan")
    D = apply_levers(vec, levers, comps, nominals)
    with np.errstate(invalid="ignore"):
        ok = (np.abs(D) <= limit).all(axis=1) & ~np.isnan(D).any(axis=1)
    return float(ok.mean() * 100)


def waterfall(vec: pd.DataFrame, levers: Levers, comps: dict | None, limit: float,
              nominals: Nominals) -> pd.DataFrame:
    """Cumulative FPY as each lever is switched on, in a fixed order."""
    steps = [("Today", Levers())]
    cur = Levers()
    if levers.compensate:
        cur = replace(cur, compensate=True)
        steps.append(("Compensation", cur))
    if levers.yaw_reduction > 0:
        cur = replace(cur, yaw_reduction=levers.yaw_reduction)
        steps.append((f"Yaw spread −{levers.yaw_reduction:.0%}", cur))
    if levers.shift_reduction > 0:
        cur = replace(cur, shift_reduction=levers.shift_reduction)
        steps.append((f"Shift spread −{levers.shift_reduction:.0%}", cur))
    if levers.remove_shape_bias:
        cur = replace(cur, remove_shape_bias=True)
        steps.append(("No systematic shape", cur))
    if levers.shape_reduction > 0:
        cur = replace(cur, shape_reduction=levers.shape_reduction)
        steps.append((f"Shape spread −{levers.shape_reduction:.0%}", cur))
    rows, prev = [], None
    for label, lv in steps:
        f = fpy_after(vec, lv, comps, limit, nominals)
        rows.append({"Step": label, "FPY": f, "Delta": 0.0 if prev is None else f - prev})
        prev = f
    return pd.DataFrame(rows)


def required_reduction(vec: pd.DataFrame, target: float, comps: dict | None, limit: float,
                       nominals: Nominals, which: str = "placement", compensate: bool = True) -> float | None:
    """Smallest spread reduction (0..1) that reaches ``target`` FPY, or None
    if even a 100 % reduction is not enough."""

    def lv(r):
        if which == "placement":
            return Levers(compensate, r, r, 0.0, False)
        if which == "yaw":
            return Levers(compensate, 0.0, r, 0.0, False)
        if which == "shift":
            return Levers(compensate, r, 0.0, 0.0, False)
        return Levers(compensate, r, r, r, False)

    if fpy_after(vec, lv(0.0), comps, limit, nominals) >= target:
        return 0.0
    if fpy_after(vec, lv(1.0), comps, limit, nominals) < target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(14):
        mid = (lo + hi) / 2
        if fpy_after(vec, lv(mid), comps, limit, nominals) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def sensitivity_curves(vec: pd.DataFrame, comps: dict | None, limit: float, nominals: Nominals,
                       compensate: bool = True, steps: int = 11) -> dict[str, tuple]:
    xs = np.linspace(0, 1, steps)
    curves = {
        "Yaw spread": [fpy_after(vec, Levers(compensate, 0.0, r, 0.0, False), comps, limit, nominals) for r in xs],
        "Shift spread": [fpy_after(vec, Levers(compensate, r, 0.0, 0.0, False), comps, limit, nominals) for r in xs],
        "Shape spread": [fpy_after(vec, Levers(compensate, 0.0, 0.0, r, False), comps, limit, nominals) for r in xs],
    }
    return {k: (xs, v) for k, v in curves.items()}


def ceilings(vec: pd.DataFrame, comps: dict | None, limit: float, nominals: Nominals) -> dict:
    """FPY with perfect placement / perfect shape (theoretical limits)."""
    return {
        "today": fpy_after(vec, Levers(), comps, limit, nominals),
        "compensated": fpy_after(vec, Levers(compensate=True), comps, limit, nominals),
        "perfect_placement": fpy_after(vec, Levers(True, 1.0, 1.0, 0.0, False), comps, limit, nominals),
        "perfect_shape": fpy_after(vec, Levers(False, 0.0, 0.0, 1.0, True), comps, limit, nominals),
    }


def mode_shape_points(bat_type: str, nominals: Nominals, vector8, scale: float) -> dict:
    """Corner positions of the nominal outline displaced by ``scale`` × vector."""
    nom = nominals.get(bat_type)
    v = np.asarray(vector8, float) * scale
    return {c: (nom[c][0] + v[2 * i], nom[c][1] + v[2 * i + 1]) for i, c in enumerate(CORNERS)}


def mode_vector(bat_type: str, mode: str, nominals: Nominals) -> np.ndarray:
    """Deviation pattern of 1 mm (RMS at the corners) of ``mode``."""
    E, _ = mode_basis(bat_type, nominals)
    return E[:, MODES.index(mode)] * 2.0
