"""Statistical process control: control charts, Nelson rules, change points.

* ``imr``: individuals chart (σ from the average moving range).
* ``xbar_s``: weekly means with limits that follow each week's sample size.
* ``nelson_rules``: the run rules most used in automotive SPC.
* ``change_points``: where the process mean shifted (binary segmentation with
  a BIC-type penalty), so drift becomes dated, measurable steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import welch

RULES = {
    1: "1 point beyond 3σ",
    2: "9 points in a row on one side of the center",
    3: "6 points in a row steadily rising or falling",
    5: "2 of 3 points beyond 2σ on one side",
    6: "4 of 5 points beyond 1σ on one side",
    7: "15 points in a row within 1σ (stratification)",
    8: "8 points in a row outside 1σ (mixture)",
}
RULE_SHORT = {1: "R1 >3σ", 2: "R2 shift", 3: "R3 trend", 5: "R5 2of3>2σ", 6: "R6 4of5>1σ",
              7: "R7 hugging", 8: "R8 mixture"}


def _run_lengths(cond: np.ndarray) -> np.ndarray:
    """Length of the current run of True values ending at each index."""
    out = np.zeros(len(cond), dtype=int)
    run = 0
    for i, c in enumerate(cond):
        run = run + 1 if c else 0
        out[i] = run
    return out


def _window_count(cond: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(cond.astype(int))])
    idx = np.arange(len(cond))
    lo = np.maximum(0, idx + 1 - w)
    return c[idx + 1] - c[lo]


def nelson_rules(x, center: float, sigma, rules=(1, 2, 3, 5, 6)) -> dict[int, np.ndarray]:
    """Boolean flags per rule, raised on the point that completes the pattern.
    ``sigma`` may be a scalar or an array (varying limits)."""
    x = np.asarray(x, float)
    n = len(x)
    s = np.broadcast_to(np.asarray(sigma, float), x.shape)
    ok = np.isfinite(x) & np.isfinite(s) & (s > 0)
    z = np.where(ok, (x - center) / np.where(s > 0, s, 1), 0.0)
    out: dict[int, np.ndarray] = {}
    if 1 in rules:
        out[1] = ok & (np.abs(z) > 3)
    if 2 in rules:
        up, dn = _run_lengths(ok & (z > 0)), _run_lengths(ok & (z < 0))
        out[2] = (up >= 9) | (dn >= 9)
    if 3 in rules:
        d = np.diff(x, prepend=np.nan)
        inc, dec = _run_lengths(np.nan_to_num(d) > 0), _run_lengths(np.nan_to_num(d) < 0)
        out[3] = (inc >= 5) | (dec >= 5)
    if 5 in rules:
        out[5] = ok & (((_window_count(z > 2, 3) >= 2) & (z > 2)) | ((_window_count(z < -2, 3) >= 2) & (z < -2)))
    if 6 in rules:
        out[6] = ok & (((_window_count(z > 1, 5) >= 4) & (z > 1)) | ((_window_count(z < -1, 5) >= 4) & (z < -1)))
    if 7 in rules:
        out[7] = _run_lengths(ok & (np.abs(z) < 1)) >= 15
    if 8 in rules:
        out[8] = _run_lengths(ok & (np.abs(z) > 1)) >= 8
    for k in out:
        if len(out[k]) != n:
            out[k] = np.zeros(n, bool)
    return out


@dataclass
class ControlChart:
    x: np.ndarray            # plotted values
    center: float
    sigma: np.ndarray        # σ of each plotted point (array: varying limits)
    ucl: np.ndarray
    lcl: np.ndarray
    flags: dict              # rule -> bool array
    mr: np.ndarray | None = None
    mr_ucl: float | None = None
    labels: list | None = None

    @property
    def any_flag(self) -> np.ndarray:
        f = np.zeros(len(self.x), bool)
        for v in self.flags.values():
            f |= v
        return f

    def flag_text(self) -> list[str]:
        return [", ".join(RULE_SHORT[k] for k, v in self.flags.items() if v[i]) for i in range(len(self.x))]


def imr(values, baseline=None, rules=(1, 2, 3, 5, 6)) -> ControlChart | None:
    x = np.asarray(values, float)
    if np.isfinite(x).sum() < 3:
        return None
    base = x[baseline] if baseline is not None else x
    base = base[np.isfinite(base)]
    if len(base) < 3:
        base = x[np.isfinite(x)]
    center = float(base.mean())
    mr_base = np.abs(np.diff(base))
    mrbar = float(mr_base.mean()) if len(mr_base) else float("nan")
    sigma = mrbar / 1.128 if mrbar > 0 else float(base.std(ddof=1))
    mr = np.abs(np.diff(x, prepend=np.nan))
    sig = np.full(len(x), sigma)
    return ControlChart(x=x, center=center, sigma=sig, ucl=center + 3 * sig, lcl=center - 3 * sig,
                        flags=nelson_rules(x, center, sigma, rules), mr=mr, mr_ucl=3.267 * mrbar)


def xbar_s(df: pd.DataFrame, value: str, group: str = "WeekKey", label: str = "CalendarWeek",
           rules=(1, 2, 3, 5, 6)):
    """Weekly means; limits follow each week's n. Returns (chart, table) or None."""
    d = df[[group, label, value]].dropna()
    if d.empty:
        return None
    g = d.groupby([group, label])[value]
    t = pd.DataFrame({"n": g.size(), "mean": g.mean(), "sd": g.std(ddof=1)}).reset_index().sort_values(group)
    t = t[t["n"] >= 2].reset_index(drop=True)
    if len(t) < 3:
        return None
    center = float((t["mean"] * t["n"]).sum() / t["n"].sum())
    pooled = float(np.sqrt(((t["n"] - 1) * t["sd"] ** 2).sum() / (t["n"] - 1).sum()))
    sig = pooled / np.sqrt(t["n"].to_numpy(float))
    x = t["mean"].to_numpy(float)
    cc = ControlChart(x=x, center=center, sigma=sig, ucl=center + 3 * sig, lcl=center - 3 * sig,
                      flags=nelson_rules(x, center, sig, rules), labels=t[label].tolist())
    return cc, t


# ---------------------------------------------------------------------------
# Change points
# ---------------------------------------------------------------------------

def _robust_sigma(y: np.ndarray) -> float:
    d = np.diff(y)
    if len(d) < 2:
        return float(np.std(y)) or 1.0
    s = float(np.median(np.abs(d - np.median(d))) / 0.6745 / np.sqrt(2))
    return s if s > 0 else (float(np.std(y)) or 1.0)


def change_points(y, min_size: int = 12, max_cp: int = 4, penalty: float | None = None) -> list[int]:
    """Indices where the mean shifts (start of each new segment).

    Binary segmentation on the sum of squared errors with penalty
    ``penalty`` (default 3·σ²·log n, σ from the point-to-point noise).
    """
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 2 * min_size:
        return []
    sigma = _robust_sigma(y)
    pen = penalty if penalty is not None else 3.0 * sigma ** 2 * np.log(n)
    c1 = np.concatenate([[0.0], np.cumsum(y)])
    c2 = np.concatenate([[0.0], np.cumsum(y * y)])

    def cost(a, b):
        m = b - a
        s = c1[b] - c1[a]
        return (c2[b] - c2[a]) - s * s / m

    cps: list[int] = []
    segs = [(0, n)]
    while segs and len(cps) < max_cp:
        best = None
        for a, b in segs:
            if b - a < 2 * min_size:
                continue
            ks = np.arange(a + min_size, b - min_size + 1)
            m1, m2 = ks - a, b - ks
            s1, s2 = c1[ks] - c1[a], c1[b] - c1[ks]
            q1, q2 = c2[ks] - c2[a], c2[b] - c2[ks]
            costs = (q1 - s1 * s1 / m1) + (q2 - s2 * s2 / m2)
            j = int(np.argmin(costs))
            gain = cost(a, b) - costs[j]
            if gain > pen and (best is None or gain > best[0]):
                best = (gain, a, b, int(ks[j]))
        if best is None:
            break
        _, a, b, k = best
        cps.append(k)
        segs.remove((a, b))
        segs += [(a, k), (k, b)]
    return sorted(cps)


CP_COLUMNS = ["Variable", "Date", "Index", "Before", "After", "Shift", "p", "N before", "N after"]


def describe_change_points(dates, y, cps: list[int], variable: str) -> pd.DataFrame:
    """Table of shifts: when, before → after mean, Δ, p-value."""
    y = np.asarray(y, float)
    dates = pd.Series(pd.to_datetime(pd.Series(dates))).reset_index(drop=True)
    bounds = [0] + list(cps) + [len(y)]
    rows = []
    for i, k in enumerate(cps):
        before = y[bounds[i]:k]
        after = y[k:bounds[i + 2]]
        diff, p = welch(before, after)
        rows.append({
            "Variable": variable, "Date": dates.iloc[k], "Index": k,
            "Before": float(np.nanmean(before)), "After": float(np.nanmean(after)),
            "Shift": diff, "p": p, "N before": len(before), "N after": len(after),
        })
    return pd.DataFrame(rows, columns=CP_COLUMNS)


def segment_means(y, cps: list[int]) -> np.ndarray:
    y = np.asarray(y, float)
    out = np.full(len(y), np.nan)
    bounds = [0] + list(cps) + [len(y)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b > a:
            out[a:b] = np.nanmean(y[a:b])
    return out


# ---------------------------------------------------------------------------
# Alarm board
# ---------------------------------------------------------------------------

ALARM_COLUMNS = ["BatteryType", "Variable", "Column", "N recent", "Baseline from", "Baseline", "Recent", "Shift",
                 "Shift σ", "Spread ratio", "R1 recent", "R1 expected", "p shift", "p spread", "p R1", "State",
                 "Reason"]


def _binom_sf(k: int, n: int, p: float) -> float:
    from scipy.stats import binom
    return float(binom.sf(k - 1, n, p)) if k > 0 else 1.0


def alarm_board(first: pd.DataFrame, variables: dict[str, str], recent_days: int = 14,
                min_baseline: int = 30) -> pd.DataFrame:
    """Has anything changed recently? For each battery type and variable the
    last ``recent_days`` are compared with a baseline: the data since the last
    detected level shift before the window (or everything before the window).

    * level: Welch t-test, flagged when also ≥ 0.25 σ (practically relevant),
    * spread: Brown–Forsythe test, flagged when σ grew by ≥ 25 %,
    * points beyond the baseline's 3σ limits vs the 0.27 % expected by chance.

    State: Signal (p < 0.001 on any check), Watch (p < 0.05), In control.
    """
    from .stats import levene

    rows = []
    if first.empty:
        return pd.DataFrame(columns=ALARM_COLUMNS)
    t_end = pd.to_datetime(first["Date"]).max()
    start = t_end - pd.Timedelta(days=recent_days)
    for t, sub in first.sort_values("Date").groupby("BatteryType"):
        dates = pd.to_datetime(sub["Date"])
        for col, label in variables.items():
            if col not in sub:
                continue
            m = sub[col].notna().to_numpy()
            y = sub[col].to_numpy(float)[m]
            d = dates.to_numpy()[m]
            rec = d >= np.datetime64(start)
            if rec.sum() < 5:
                continue
            pre = y[~rec]
            if len(pre) < min_baseline:
                continue
            cps = change_points(pre, min_size=max(12, len(pre) // 40))
            b0 = cps[-1] if cps and len(pre) - cps[-1] >= min_baseline else 0
            base = pre[b0:]
            recent = y[rec]
            mr = np.abs(np.diff(base))
            sigma = float(mr.mean() / 1.128) if len(mr) and mr.mean() > 0 else float(base.std(ddof=1))
            center = float(base.mean())
            diff, p_shift = welch(base, recent)
            ratio, p_spread = levene(base, recent)
            r1 = int((np.abs(recent - center) > 3 * sigma).sum())
            exp_r1 = len(recent) * 0.0027
            p_r1 = _binom_sf(r1, len(recent), 0.0027)
            shift_sig = diff / sigma if sigma > 0 else np.nan
            reasons = []
            level = "In control"

            def bump(p, ok_practical, text):
                nonlocal level
                if not ok_practical or not np.isfinite(p):
                    return
                if p < 0.001:
                    level = "Signal"
                    reasons.append(text)
                elif p < 0.05:
                    if level == "In control":
                        level = "Watch"
                    reasons.append(text + " (weak)")

            unit = "°" if "Rot" in col or "Yaw" in label else ""
            bump(p_shift, np.isfinite(shift_sig) and abs(shift_sig) >= 0.25,
                 f"level moved {diff:+.3f}{unit} ({shift_sig:+.1f}σ)")
            bump(p_spread, np.isfinite(ratio) and ratio >= 1.25, f"scatter ×{ratio:.2f}")
            bump(p_r1, r1 >= 2, f"{r1} points beyond 3σ (≈{exp_r1:.1f} expected)")
            rows.append({
                "BatteryType": t, "Variable": label, "Column": col, "N recent": int(rec.sum()),
                "Baseline from": pd.Timestamp(d[~rec][b0]) if len(pre) else pd.NaT,
                "Baseline": center, "Recent": float(recent.mean()), "Shift": diff, "Shift σ": shift_sig,
                "Spread ratio": ratio, "R1 recent": r1, "R1 expected": exp_r1, "p shift": p_shift,
                "p spread": p_spread, "p R1": p_r1, "State": level, "Reason": "; ".join(reasons) or "—",
            })
    return pd.DataFrame(rows, columns=ALARM_COLUMNS)
