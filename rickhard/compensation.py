"""Rigid compensation (X/Y translation + yaw about the nominal module center).

Methods
-------
* **Median-based** (original app): undo the median centroid offset and the
  median yaw. Centers the *average* module.
* **Rigid fit (mean)**: undo the mean least-squares placement (all four
  corners vote for the yaw, not only the front/rear midpoints).
* **Robust optimum**: maximise the *expected* FPY when every setting is
  blurred by placement noise σ. A module passes only if all four corners are
  inside the tolerance, so for each yaw the X/Y settings that make one module
  pass form a rectangle; with Gaussian noise the pass probability is a
  product of two error-function differences, and the expected FPY over all
  modules on a whole X×Y grid is a single matrix product. That makes the
  search exhaustive, smooth (it prefers the middle of a good region, not a
  lucky edge) and fast.

Plus: exact evaluation, FPY landscape, yaw sweep, walk-forward backtest,
bootstrap confidence interval of the gain, and a readiness checklist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import ndtr

from .config import Nominals, normalize_type
from .constants import DEV_COLS
from .geometry import add_vectors, apply_compensation, corner_offsets, dev_matrix, evaluate_status, rel_nominal, with_deviations

METHODS = ("No compensation", "Median-based", "Rigid fit (mean)", "Robust optimum")


# ---------------------------------------------------------------------------
# Simple suggestions
# ---------------------------------------------------------------------------

def median_suggestion(vec: pd.DataFrame, method: str = "median") -> tuple[float, float, float]:
    """Original method: minus the median (or mean) centroid offset and yaw."""
    if vec.empty:
        return (np.nan, np.nan, np.nan)
    f = (lambda s: s.dropna().mean()) if method == "mean" else (lambda s: s.dropna().median())
    return (-float(f(vec["Centroid_X"])), -float(f(vec["Centroid_Y"])), -float(f(vec["Rotation_Angle"])))


def rigid_mean_suggestion(vec: pd.DataFrame) -> tuple[float, float, float]:
    if vec.empty or vec["Fit_Tx"].notna().sum() == 0:
        return (np.nan, np.nan, np.nan)
    return (-float(vec["Fit_Tx"].mean()), -float(vec["Fit_Ty"].mean()), -float(vec["Fit_Rot"].mean()))


def spread(vec: pd.DataFrame) -> dict:
    out = {}
    for c in ("Centroid_X", "Centroid_Y", "Rotation_Angle", "Fit_Tx", "Fit_Ty", "Fit_Rot"):
        s = vec[c].dropna() if c in vec else pd.Series(dtype=float)
        if len(s):
            q75, q25 = np.percentile(s, [75, 25])
            out[c] = {"median": float(s.median()), "mean": float(s.mean()),
                      "std": float(s.std()) if len(s) > 1 else np.nan,
                      "iqr": float(q75 - q25), "mad": float(np.median(np.abs(s - s.median()))), "n": int(len(s))}
    return out


# ---------------------------------------------------------------------------
# Exact evaluation
# ---------------------------------------------------------------------------

def pass_mask(df: pd.DataFrame, comp, limit: float, nominals: Nominals) -> np.ndarray:
    """Which runs pass after applying ``comp`` (incomplete runs never pass)."""
    D = apply_compensation(df, comp, nominals)
    with np.errstate(invalid="ignore"):
        return (np.abs(D) <= limit).all(axis=1) & ~np.isnan(D).any(axis=1)


def evaluate(df: pd.DataFrame, comp, limit: float, nominals: Nominals, margin: float = 0.0) -> dict:
    n = len(df)
    if n == 0:
        return {"n": 0, "pass": 0, "fpy": np.nan, "pass_margin": 0, "fpy_margin": np.nan}
    p = pass_mask(df, comp, limit, nominals)
    pm = pass_mask(df, comp, limit - margin, nominals) if margin > 0 else p
    return {"n": n, "pass": int(p.sum()), "fpy": float(p.mean() * 100),
            "pass_margin": int(pm.sum()), "fpy_margin": float(pm.mean() * 100)}


# ---------------------------------------------------------------------------
# Robust optimum
# ---------------------------------------------------------------------------

def _rel_positions(df: pd.DataFrame, nominals: Nominals):
    """Complete runs of ONE type: actual corner positions relative to the
    nominal center (k, 4) and the nominal offsets (4,)."""
    D = dev_matrix(df)
    ok = ~np.isnan(D).any(axis=1)
    t = normalize_type(df["BatteryType"].iloc[0])
    rx, ry = rel_nominal(t, nominals)
    return rx[None] + D[ok][:, 0::2], ry[None] + D[ok][:, 1::2], rx, ry


def _bounds(ax, ay, rx, ry, yaws_deg, limit):
    """Pass intervals of the translation for each yaw: (n_yaw, k) arrays."""
    t = np.radians(np.asarray(yaws_deg, float))[:, None, None]
    c, s = np.cos(t), np.sin(t)
    ex = ax[None] * c - ay[None] * s - rx[None, None]
    ey = ax[None] * s + ay[None] * c - ry[None, None]
    return (-limit - ex.min(axis=2), limit - ex.max(axis=2),
            -limit - ey.min(axis=2), limit - ey.max(axis=2))


def _pass_prob(lo, hi, grid, sigma):
    """(k, G) probability that grid + N(0, σ) falls inside [lo, hi]."""
    if sigma <= 0:
        g = grid[None, :]
        return ((g >= lo[:, None] - 1e-12) & (g <= hi[:, None] + 1e-12)).astype(float)
    p = ndtr((hi[:, None] - grid[None, :]) / sigma) - ndtr((lo[:, None] - grid[None, :]) / sigma)
    return np.where((hi >= lo)[:, None], np.clip(p, 0.0, 1.0), 0.0)


@dataclass
class Optimum:
    dx: float
    dy: float
    yaw: float
    n: int
    n_complete: int
    expected_fpy: float            # under the noise model
    exact: dict = field(default_factory=dict)
    yaws: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    best_by_yaw: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    landscape: dict | None = field(repr=False, default=None)


def robust_optimum(df: pd.DataFrame, limit: float, nominals: Nominals, sigma_xy: float = 0.10,
                   sigma_yaw: float = 0.01, center: tuple[float, float, float] | None = None,
                   xy_half: float = 2.0, yaw_half: float = 0.25, step_xy: float = 0.02,
                   step_yaw: float = 0.005, keep_landscape: bool = True) -> Optimum | None:
    """Setting (dx, dy, yaw) with the highest expected FPY for ONE battery type."""
    n = len(df)
    if n == 0:
        return None
    ax, ay, rx, ry = _rel_positions(df, nominals)
    k = len(ax)
    if k == 0:
        return None
    if center is None or not np.isfinite(center).all():
        center = (0.0, 0.0, 0.0)
    cx, cy, cyaw = center
    gx = np.round(np.arange(cx - xy_half, cx + xy_half + step_xy / 2, step_xy), 6)
    gy = np.round(np.arange(cy - xy_half, cy + xy_half + step_xy / 2, step_xy), 6)
    yaws = np.round(np.arange(cyaw - yaw_half, cyaw + yaw_half + step_yaw / 2, step_yaw), 6)
    lo_x, hi_x, lo_y, hi_y = _bounds(ax, ay, rx, ry, yaws, limit)
    cube = np.empty((len(yaws), len(gx), len(gy)), dtype=np.float32)
    for i in range(len(yaws)):
        px = _pass_prob(lo_x[i], hi_x[i], gx, sigma_xy)
        py = _pass_prob(lo_y[i], hi_y[i], gy, sigma_xy)
        cube[i] = (px.T @ py) / n
    if sigma_yaw > 0 and len(yaws) > 2:
        cube = gaussian_filter1d(cube, sigma=sigma_yaw / step_yaw, axis=0, mode="nearest")
    move = (np.abs(gx)[:, None] + np.abs(gy)[None, :])[None] + 20 * np.abs(yaws)[:, None, None]
    score = cube - 1e-6 * move  # tie-break: prefer smaller moves
    i, j, l_ = np.unravel_index(int(np.argmax(score)), score.shape)
    dx, dy, yaw = round(float(gx[j]), 3), round(float(gy[l_]), 3), round(float(yaws[i]), 4)
    exact = evaluate(df, (dx, dy, yaw), limit, nominals)
    land = {"x": gx, "y": gy, "fpy": cube[i].T * 100, "yaw": float(yaws[i])} if keep_landscape else None
    return Optimum(dx=dx, dy=dy, yaw=yaw, n=n, n_complete=k, expected_fpy=float(cube[i, j, l_] * 100),
                   exact=exact, yaws=yaws, best_by_yaw=cube.reshape(len(yaws), -1).max(axis=1) * 100,
                   landscape=land)


def landscape(df: pd.DataFrame, yaw: float, limit: float, nominals: Nominals, sigma_xy: float = 0.0,
              half: float = 2.5, step: float = 0.025, center=(0.0, 0.0)) -> dict | None:
    """FPY (%) for every X/Y setting at a fixed yaw (heatmap)."""
    n = len(df)
    if n == 0:
        return None
    ax, ay, rx, ry = _rel_positions(df, nominals)
    if len(ax) == 0:
        return None
    gx = np.round(np.arange(center[0] - half, center[0] + half + step / 2, step), 6)
    gy = np.round(np.arange(center[1] - half, center[1] + half + step / 2, step), 6)
    lo_x, hi_x, lo_y, hi_y = _bounds(ax, ay, rx, ry, [yaw], limit)
    px = _pass_prob(lo_x[0], hi_x[0], gx, sigma_xy)
    py = _pass_prob(lo_y[0], hi_y[0], gy, sigma_xy)
    return {"x": gx, "y": gy, "fpy": (px.T @ py).T / n * 100, "yaw": yaw}


def fpy_vs_yaw(df: pd.DataFrame, dx: float, dy: float, yaws, limit: float, nominals: Nominals) -> np.ndarray:
    n = len(df)
    yaws = np.asarray(yaws, float)
    if n == 0:
        return np.full(len(yaws), np.nan)
    ax, ay, rx, ry = _rel_positions(df, nominals)
    if len(ax) == 0:
        return np.zeros(len(yaws))
    lo_x, hi_x, lo_y, hi_y = _bounds(ax, ay, rx, ry, yaws, limit)
    inside = (lo_x <= dx) & (dx <= hi_x) & (lo_y <= dy) & (dy <= hi_y)
    return inside.sum(axis=1) / n * 100


def best_fpy_by_yaw(df: pd.DataFrame, yaws, limit: float, nominals: Nominals, center=(0.0, 0.0),
                    half: float = 2.5, step: float = 0.05) -> np.ndarray:
    """Exact best FPY (%) reachable at each yaw when X / Y are re-optimised
    on a grid around ``center``."""
    yaws = np.asarray(yaws, float)
    if not len(df):
        return np.full(len(yaws), np.nan)
    ax, ay, rx, ry = _rel_positions(df, nominals)
    if len(ax) == 0:
        return np.zeros(len(yaws))
    n = len(df)
    gx = np.arange(center[0] - half, center[0] + half + step / 2, step)
    gy = np.arange(center[1] - half, center[1] + half + step / 2, step)
    lo_x, hi_x, lo_y, hi_y = _bounds(ax, ay, rx, ry, yaws, limit)
    out = np.empty(len(yaws))
    for i in range(len(yaws)):
        px = _pass_prob(lo_x[i], hi_x[i], gx, 0.0)
        py = _pass_prob(lo_y[i], hi_y[i], gy, 0.0)
        out[i] = float((px.T @ py).max()) / n * 100
    return out


# ---------------------------------------------------------------------------
# All methods at once
# ---------------------------------------------------------------------------

def suggestions(vec: pd.DataFrame, limit: float, nominals: Nominals, sigma_xy: float = 0.10,
                sigma_yaw: float = 0.01, median_method: str = "median", fast: bool = False) -> dict:
    """Settings of every method for ONE battery type."""
    out: dict = {"No compensation": (0.0, 0.0, 0.0)}
    med = median_suggestion(vec, median_method)
    rig = rigid_mean_suggestion(vec)
    if np.isfinite(med).all():
        out["Median-based"] = tuple(round(v, 4) for v in med)
    if np.isfinite(rig).all():
        out["Rigid fit (mean)"] = tuple(round(v, 4) for v in rig)
    center = rig if np.isfinite(rig).all() else (med if np.isfinite(med).all() else (0.0, 0.0, 0.0))
    kw = dict(step_xy=0.04, step_yaw=0.01, xy_half=1.6, yaw_half=0.2, keep_landscape=False) if fast else {}
    opt = robust_optimum(vec, limit, nominals, sigma_xy, sigma_yaw, center=center, **kw) if len(vec) else None
    if opt is not None:
        out["Robust optimum"] = (opt.dx, opt.dy, opt.yaw)
    return {"settings": out, "optimum": opt}


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------

BACKTEST_MODES = ("fixed", "rolling")


def auto_test_weeks(n_weeks: int, window_weeks: int, most: int = 8) -> int:
    """How many of the last weeks to test on when the user does not say: the
    weeks left after a full window, at least 2 and at most ``most``, always
    leaving at least one earlier week to learn from."""
    if n_weeks < 2:
        return 0
    return int(min(n_weeks - 1, max(2, min(most, n_weeks - window_weeks))))


def backtest(vec: pd.DataFrame, limit: float, nominals: Nominals, window_weeks: int = 6,
             test_weeks: int | None = None, min_train: int = 15, sigma_xy: float = 0.10,
             sigma_yaw: float = 0.01, median_method: str = "median", mode: str = "fixed") -> pd.DataFrame:
    """Replay the last ``test_weeks`` weeks as if each method had been in use
    (``None``: :func:`auto_test_weeks`). The settings are learnt from up to
    ``window_weeks`` weeks before them — fewer when the data is short.

    ``mode="fixed"`` (set once and left): every method's setting is computed
    ONCE from the ``window_weeks`` before the first test week and then left
    unchanged for all the test weeks — how a permanent correction is used.
    ``mode="rolling"``: every test week gets a fresh setting computed from the
    ``window_weeks`` just before it.

    In both modes no module ever helps compute the setting it is judged with.
    """
    weeks = sorted(vec["WeekKey"].dropna().unique())
    labels = vec.drop_duplicates("WeekKey").set_index("WeekKey")["CalendarWeek"]
    cols = ["WeekKey", "Week", "Modules", "Trained on", "Train n weeks", "Train weeks", *METHODS,
            *[f"{m} {k}" for m in METHODS for k in ("X", "Y", "Yaw")]]
    k = auto_test_weeks(len(weeks), window_weeks) if not test_weeks else min(int(test_weeks), len(weeks) - 1)
    if k < 1:
        return pd.DataFrame(columns=cols)
    test = weeks[-k:]                          # at least one earlier week is always left to learn from

    def train_on(prior):
        train = vec[vec["WeekKey"].isin(prior)]
        if len(train) < min_train:
            return None, train
        return suggestions(train, limit, nominals, sigma_xy, sigma_yaw, median_method, fast=True)["settings"], train

    fixed = None
    if mode == "fixed" and test:
        prior = [w for w in weeks if w < test[0]][-window_weeks:]
        fixed = (prior,) + train_on(prior)
        if fixed[1] is None:
            return pd.DataFrame(columns=cols)
    rows = []
    for wk in test:
        if fixed is not None:
            prior, sug, train = fixed
        else:
            prior = [w for w in weeks if w < wk][-window_weeks:]
            sug, train = train_on(prior)
        t = vec[vec["WeekKey"] == wk]
        if sug is None or t.empty:
            continue
        row = {"WeekKey": int(wk), "Week": labels.get(wk, str(wk)), "Modules": len(t), "Trained on": len(train),
               "Train n weeks": len(prior),
               "Train weeks": (f"{labels.get(prior[0], prior[0])} – {labels.get(prior[-1], prior[-1])}"
                               if len(prior) > 1 else str(labels.get(prior[0], prior[0])))}
        for m in METHODS:
            st_ = sug.get(m)
            row[m] = evaluate(t, st_, limit, nominals)["pass"] if st_ is not None else np.nan
            for k, val in zip(("X", "Y", "Yaw"), st_ if st_ is not None else (np.nan,) * 3):
                row[f"{m} {k}"] = val
        rows.append(row)
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["mode"] = mode
    return out


def backtest_summary(bt: pd.DataFrame, mode: str | None = None) -> dict:
    if bt is None or bt.empty:
        return {}
    n = int(bt["Modules"].sum())
    mode = mode or bt.attrs.get("mode", "fixed")
    out = {"n": n, "weeks": int(len(bt)), "mode": mode,
           "label": f"{bt['Week'].iloc[0]} – {bt['Week'].iloc[-1]}" if len(bt) > 1 else str(bt["Week"].iloc[0])}
    if "Train weeks" in bt:
        out["train_label"] = str(bt["Train weeks"].iloc[0])
        out["train_weeks"] = int(bt["Train n weeks"].iloc[0])
    for m in METHODS:
        if m in bt and bt[m].notna().all():
            out[m] = float(bt[m].sum() / n * 100)
            if mode == "fixed" and f"{m} X" in bt:
                out.setdefault("settings", {})[m] = tuple(float(bt[f"{m} {k}"].iloc[0]) for k in ("X", "Y", "Yaw"))
    present = [m for m in METHODS if m in out]
    out["best"] = max(present, key=lambda m: out[m]) if present else None
    return out


def backtest_text(bt_sum: dict) -> str:
    """How the backtest was run, in words (for captions and the readiness check)."""
    if not bt_sum:
        return ""
    tested = f"{bt_sum['label']} ({_weeks(bt_sum['weeks'])}, {bt_sum['n']} modules)"
    if bt_sum.get("mode") == "rolling":
        return f"recalculated every week over {tested}"
    return (f"set once from {bt_sum.get('train_label', 'the weeks before')} ({_weeks(bt_sum.get('train_weeks', 0))}) "
            f"and left unchanged over {tested}")


def _weeks(n: int) -> str:
    return f"{n} week" + ("" if n == 1 else "s")


# ---------------------------------------------------------------------------
# Uncertainty of the gain
# ---------------------------------------------------------------------------

def bootstrap_gain(df: pd.DataFrame, comp, limit: float, nominals: Nominals, B: int = 400,
                   seed: int = 0) -> dict:
    """Paired bootstrap of the FPY gain (pp) of ``comp`` vs no compensation."""
    n = len(df)
    if n < 10:
        return {}
    base = pass_mask(df, (0.0, 0.0, 0.0), limit, nominals).astype(float)
    new = pass_mask(df, comp, limit, nominals).astype(float)
    diff = new - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    g = diff[idx].mean(axis=1) * 100
    return {"gain": float(diff.mean() * 100), "lo": float(np.percentile(g, 5)), "hi": float(np.percentile(g, 95)),
            "p_positive": float((g > 0).mean()), "fixed": int(((new == 1) & (base == 0)).sum()),
            "broken": int(((new == 0) & (base == 1)).sum())}


# ---------------------------------------------------------------------------
# Readiness
# ---------------------------------------------------------------------------

def weekly_placement(vec: pd.DataFrame) -> pd.DataFrame:
    if vec.empty:
        return pd.DataFrame(columns=["WeekKey", "CalendarWeek", "N", "X", "Y", "Yaw"])
    g = vec.groupby(["WeekKey", "CalendarWeek"])
    return pd.DataFrame({
        "N": g.size(), "X": g["Fit_Tx"].median(), "Y": g["Fit_Ty"].median(), "Yaw": g["Fit_Rot"].median(),
    }).reset_index().sort_values("WeekKey").reset_index(drop=True)


def readiness(vec: pd.DataFrame, bt_sum: dict | None = None, recent_change: bool = False,
              min_n: int = 40, min_weeks: int = 4, min_week_n: int = 5,
              xy_band: float = 0.5, yaw_band: float = 0.05, min_gain: float = 1.0) -> list[dict]:
    """Is the process stable enough to trust a compensation?"""
    items = []
    n = len(vec)
    items.append({"check": "Enough modules", "ok": n >= min_n,
                  "detail": f"{n} first-run modules in the window (need ≥ {min_n})"})
    wb = weekly_placement(vec)
    wb_ok = wb[wb["N"] >= min_week_n]
    items.append({"check": "Enough weeks", "ok": len(wb_ok) >= min_weeks,
                  "detail": f"{len(wb_ok)} weeks with ≥ {min_week_n} modules (need ≥ {min_weeks})"})
    recent = wb_ok.tail(min_weeks)
    if len(recent) >= 2:
        rx_, ry_, ryaw = (float(recent[c].max() - recent[c].min()) for c in ("X", "Y", "Yaw"))
        items.append({"check": "Stable X/Y placement", "ok": bool(rx_ <= xy_band and ry_ <= xy_band),
                      "detail": f"weekly medians moved {rx_:.2f} mm (X) and {ry_:.2f} mm (Y) over "
                                f"{len(recent)} weeks (limit {xy_band:.2f} mm)"})
        items.append({"check": "Stable yaw", "ok": bool(ryaw <= yaw_band),
                      "detail": f"weekly median yaw moved {ryaw:.3f}° (limit {yaw_band:.3f}°)"})
    else:
        items.append({"check": "Stable X/Y placement", "ok": False, "detail": "not enough weeks to judge"})
        items.append({"check": "Stable yaw", "ok": False, "detail": "not enough weeks to judge"})
    items.append({"check": "No recent process shift", "ok": not recent_change,
                  "detail": "a change point was detected in the last 2 weeks — wait for the new level to settle"
                  if recent_change else "no change point in placement over the last 2 weeks"})
    if bt_sum:
        none = bt_sum.get("No compensation", np.nan)
        best_m = bt_sum.get("best")
        best = bt_sum.get(best_m, np.nan) if best_m else np.nan
        gain = best - none
        items.append({"check": "Helps in the backtest", "ok": bool(np.isfinite(gain) and gain >= min_gain),
                      "detail": f"{backtest_text(bt_sum)}: "
                                f"{none:.1f}% → {best:.1f}% with {best_m}, {gain:+.1f} pp "
                                f"(need ≥ +{min_gain:.1f} pp)"})
    else:
        items.append({"check": "Helps in the backtest", "ok": False,
                      "detail": "needs earlier weeks with enough modules to test on"})
    return items


def simulate_table(df: pd.DataFrame, comp, limit: float, nominals: Nominals) -> pd.DataFrame:
    """Original-style piece table: deviations before/after, both statuses and
    the centroid / yaw of the compensated geometry."""
    D = apply_compensation(df, comp, nominals)
    sim = evaluate_status(with_deviations(df, D, "_Sim"), limit, suffix="_Sim")
    geo = df[["BatteryType"]].copy()
    for j, c in enumerate(DEV_COLS):
        geo[c] = D[:, j] if len(df) else np.array([], float)
    v = add_vectors(geo, nominals)
    sim["Centroid_X_Sim"] = v["Centroid_X"].to_numpy()
    sim["Centroid_Y_Sim"] = v["Centroid_Y"].to_numpy()
    sim["Rotation_Angle_Sim"] = v["Rotation_Angle"].to_numpy()
    return sim


def equivalent_offsets_table(settings: dict, nominals: Nominals) -> pd.DataFrame:
    frames = [corner_offsets(t, *s, nominals) for t, s in settings.items()
              if s is not None and np.isfinite(s).all()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
