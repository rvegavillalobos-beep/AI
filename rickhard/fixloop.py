"""The fix loop: problem → action → verification → what is left.

A line that is adjusted every week does not have one FPY, it has one FPY per
*set-up* — the stretch of production between two logged changes. Everything
here works on set-ups instead of calendar windows:

* ``setups``        FPY (with its confidence interval) of every set-up,
* ``ledger``        each logged action judged on the modules right before and
                    right after it, never across another logged change, and on
                    the very thing it was meant to fix,
* ``loss_budget``   where the missing FPY points of a set-up go,
* ``bridge``        what it takes to close the gap to the target, simulated on
                    the set-up's own modules,
* ``proof``         what the data can claim about the target, and how many
                    modules it takes to demonstrate it,
* ``story``         the same findings in plain sentences (for the brief).

Nothing here imports Streamlit or Plotly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as _st

from . import events as ev
from . import improvement as imp
from .compensation import rigid_mean_suggestion
from .config import Nominals
from .constants import BATTERY_TYPES, MECHANISMS
from .stats import fisher_2x2, is_significant, levene, significance_label, welch, wilson

MIN_N = 30          # modules needed after a change before it is judged
MIN_BEFORE = 10     # modules needed before a change to have a baseline
SETTLED_N = 150     # with this many modules after it, "not significant" becomes "no effect"
WINDOW = 300        # modules compared on each side of a change, at most
CONFIDENCE = 0.95

# NOK mechanisms (MechanismDetail) an action aimed at a target is meant to remove
TARGET_MECHANISMS = {
    "Shift X": ("Shift X", "Shift X+Y", "Shift + yaw"),
    "Shift Y": ("Shift Y", "Shift X+Y", "Shift + yaw"),
    "Yaw": ("Yaw", "Shift + yaw"),
    "Scatter": ("Shift X", "Shift Y", "Shift X+Y", "Yaw", "Shift + yaw"),
    "Shape": ("Shape", "Placement + shape"),
    "Height (Z)": ("Height (Z)",),
    "Missing corner": ("Missing corner",),
}
TARGET_LEVEL = {"Shift X": ("Fit_Tx", "Shift X level", "mm", 0.10),
                "Shift Y": ("Fit_Ty", "Shift Y level", "mm", 0.10),
                "Yaw": ("Fit_Rot", "Yaw level", "°", 0.010)}
PLANNED_AXES = (("dX", "Fit_Tx", "X", "mm", 2), ("dY", "Fit_Ty", "Y", "mm", 2), ("dYaw", "Fit_Rot", "yaw", "°", 3))

VERDICT_LEVEL = {"Worked": "good", "Target hit": "good", "No effect": "neutral", "Backfired": "bad",
                 "Verifying": "warn", "Superseded": "neutral", "No baseline": "neutral", "Planned": "info"}
VERDICT_HELP = {
    "Worked": "FPY rose after the change, and the rise is statistically significant.",
    "Target hit": "The thing the action aimed at improved significantly; the FPY gain is not significant yet.",
    "No effect": "Enough modules after the change, and neither FPY nor the target moved significantly.",
    "Backfired": "FPY dropped after the change, and the drop is statistically significant.",
    "Verifying": "Too few modules since the change to tell an effect from noise.",
    "Superseded": "Another change followed before enough modules were measured: the effects cannot be separated.",
    "No baseline": "Too few modules before the change to compare with.",
    "Planned": "Not done yet.",
}

# who can remove each loss, in the words used on the shop floor
MECHANISM_FIX = {
    "Shift": "Stop position of the conveyor — compensation in X / Y",
    "Yaw": "Side guides and pushers — yaw compensation",
    "Shift + yaw": "Placement as a whole — compensation in X / Y / yaw",
    "Shape": "Upstream: the module is not square (no placement fixes it)",
    "Placement + shape": "Both: a tighter placement gives the shape its margin back",
    "Height (Z)": "Carrier supports / seating height",
    "Missing corner": "Measurement: a corner was not read (vision, failed reads)",
}
MECHANISM_FAMILY = {"Shift": "Placement", "Yaw": "Placement", "Shift + yaw": "Placement", "Shape": "Shape",
                    "Placement + shape": "Placement + shape", "Height (Z)": "Height", "Missing corner": "Measurement"}


# ---------------------------------------------------------------------------
# Set-ups
# ---------------------------------------------------------------------------

def _types_in(first: pd.DataFrame) -> tuple[str, ...]:
    present = set(first["BatteryType"]) if len(first) else set()
    return tuple(t for t in BATTERY_TYPES if t in present)


def _affects(scope: str, types) -> bool:
    return bool(set(ev.scope_types(scope)) & set(types))


def numbered(events) -> pd.DataFrame:
    """The action log with a stable number (``No``) in date order — the same
    number is used in the timeline, the ledger and the brief."""
    e = ev.normalize_events(events)
    e.insert(0, "No", np.arange(1, len(e) + 1))
    return e


def changes(events, types) -> pd.DataFrame:
    """Done actions that touch at least one of ``types``, in date order."""
    e = numbered(events)
    hit = np.array([_affects(s, types) for s in e["Scope"]], dtype=bool)
    e = e[(e["Status"] != "Planned").to_numpy() & hit]
    return e.reset_index(drop=True)


SETUP_COLUMNS = ["Setup", "Label", "Start", "End", "StartedBy", "EventNo", "N", "Passed", "NOK", "FPY", "CI_lo",
                 "CI_hi", "Days", "PerDay", "Current"]


def _prod_days(d: pd.DataFrame) -> int:
    col = "ProdDate" if "ProdDate" in d else "Date"
    return int(pd.to_datetime(d[col]).dt.normalize().nunique()) if len(d) else 0


def setups(first: pd.DataFrame, events) -> pd.DataFrame:
    """One row per set-up: the production between two logged changes.

    Changes logged before the first module start no set-up of their own (the
    first set-up is simply named after the latest of them); a change logged
    after the last module opens an empty current set-up."""
    if first.empty:
        return pd.DataFrame(columns=SETUP_COLUMNS)
    f = first.sort_values("Date", kind="stable")
    dates = pd.to_datetime(f["Date"])
    t0, t1 = dates.iloc[0], dates.iloc[-1]
    ch = changes(events, _types_in(f))
    earlier = ch[ch["Date"] <= t0]
    inside = ch[ch["Date"] > t0]
    starts = [(t0, earlier.iloc[-1] if len(earlier) else None)]
    for when, grp in inside.groupby("Date", sort=True):   # changes made at the same moment are one boundary
        starts.append((when, grp.iloc[-1]))
    passed = (f["Status"] == "PASS").to_numpy()
    rows = []
    for i, (start, e) in enumerate(starts):
        last = i == len(starts) - 1
        end = t1 if last else starts[i + 1][0]
        m = ((dates >= start) & ((dates <= end) if last else (dates < end))).to_numpy()
        n, k = int(m.sum()), int(passed[m].sum())
        lo, hi = wilson(k, n)
        days = _prod_days(f[m])
        rows.append({
            "Setup": i + 1, "Label": f"Set-up {i + 1}", "Start": start, "End": max(end, start),
            "StartedBy": "Start of the data" if e is None else str(e["Title"]),
            "EventNo": np.nan if e is None else float(e["No"]),
            "N": n, "Passed": k, "NOK": n - k, "FPY": k / n * 100 if n else np.nan, "CI_lo": lo, "CI_hi": hi,
            "Days": days, "PerDay": n / days if days else np.nan, "Current": last,
        })
    return pd.DataFrame(rows, columns=SETUP_COLUMNS)


def setup_modules(first: pd.DataFrame, row) -> pd.DataFrame:
    """The first runs that belong to one row of ``setups``."""
    d = pd.to_datetime(first["Date"])
    m = (d >= row["Start"]) & ((d <= row["End"]) if bool(row["Current"]) else (d < row["End"]))
    return first[m.to_numpy()]


def judged_setup(su: pd.DataFrame):
    """The newest set-up with enough modules to be judged (``MIN_N``); with
    none that large, the largest one. None when there is no module at all."""
    if su.empty or not (su["N"] > 0).any():
        return None
    big = su[su["N"] >= MIN_N]
    return big.iloc[-1] if len(big) else su.sort_values(["N", "Setup"]).iloc[-1]


# ---------------------------------------------------------------------------
# Did it work? One verdict per action
# ---------------------------------------------------------------------------

def _target_loss(d: pd.DataFrame, target: str) -> int:
    """NOK first runs whose mechanism the target is meant to remove."""
    names = TARGET_MECHANISMS.get(target)
    if not names or "MechanismDetail" not in d:
        return 0
    return int(((d["Status"] != "PASS") & d["MechanismDetail"].isin(names)).sum())


def _count(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def _mean(x) -> float:
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def _sd(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if len(x) > 1 else float("nan")


def _target_check(before: pd.DataFrame, after: pd.DataFrame, target: str) -> dict:
    """The check that belongs to the target: its metric before and after, and
    whether it got significantly better or worse."""
    out = {"metric": "", "unit": "", "m_before": np.nan, "m_after": np.nan, "m_p": np.nan,
           "better": False, "worse": False, "loss_before": np.nan, "loss_after": np.nan, "loss_p": np.nan}
    if target not in TARGET_MECHANISMS:
        return out
    nb, na = len(before), len(after)
    lb, la = _target_loss(before, target), _target_loss(after, target)
    out["loss_before"], out["loss_after"] = lb / nb * 100, la / na * 100
    out["loss_p"] = fisher_2x2(lb, nb - lb, la, na - la)
    loss_better = is_significant(out["loss_p"]) and out["loss_after"] < out["loss_before"]
    loss_worse = is_significant(out["loss_p"]) and out["loss_after"] > out["loss_before"]
    better = worse = False
    if target in TARGET_LEVEL:
        col, label, unit, practical = TARGET_LEVEL[target]
        if col in before and col in after:
            _, p = welch(before[col], after[col])
            mb, ma = _mean(before[col]), _mean(after[col])
            out.update({"metric": label, "unit": unit, "m_before": mb, "m_after": ma, "m_p": p})
            better = is_significant(p) and abs(ma) <= abs(mb) - practical
            worse = is_significant(p) and abs(ma) >= abs(mb) + practical
    elif target == "Scatter":
        best = None
        for col, label, unit in (("Fit_Tx", "Shift X scatter (σ)", "mm"), ("Fit_Ty", "Shift Y scatter (σ)", "mm"),
                                 ("Fit_Rot", "Yaw scatter (σ)", "°")):
            if col not in before or col not in after:
                continue
            ratio, p = levene(before[col], after[col])
            if not np.isfinite(p):
                continue
            better |= p < 0.05 and ratio <= 0.85
            worse |= p < 0.05 and ratio >= 1.15
            if best is None or p < best[0]:
                best = (p, label, unit, _sd(before[col]), _sd(after[col]))
        if best:
            out.update({"metric": best[1], "unit": best[2], "m_before": best[3], "m_after": best[4], "m_p": best[0]})
        if better and worse:   # one axis tighter, another looser: not a clear result
            better = worse = False
    elif target == "Shape" and "ShapeMax" in before and "ShapeMax" in after:
        diff, p = welch(before["ShapeMax"], after["ShapeMax"])
        out.update({"metric": "Shape residual", "unit": "mm", "m_before": _mean(before["ShapeMax"]),
                    "m_after": _mean(after["ShapeMax"]), "m_p": p})
        better, worse = is_significant(p) and diff < 0, is_significant(p) and diff > 0
    elif target in ("Height (Z)", "Missing corner"):
        out.update({"metric": f"{target} NOK", "unit": "%", "m_before": out["loss_before"],
                    "m_after": out["loss_after"], "m_p": out["loss_p"]})
    out["better"] = bool(better or loss_better) and not (worse or loss_worse)
    out["worse"] = bool(worse or loss_worse) and not (better or loss_better)
    return out


def _fmt(v: float, unit: str) -> str:
    if not np.isfinite(v):
        return "–"
    if unit == "%":
        return f"{v:.1f}%"
    return f"{v:+.3f}°" if unit == "°" else f"{v:+.2f} mm"


def _planned_vs_observed(e, before: pd.DataFrame, after: pd.DataFrame) -> str:
    bits = []
    for key, col, name, unit, nd in PLANNED_AXES:
        want = e.get(key, np.nan)
        if not np.isfinite(want) or abs(want) < 1e-9 or col not in before or col not in after:
            continue
        got = _mean(after[col]) - _mean(before[col])
        if np.isfinite(got):
            bits.append(f"{name} planned {want:+.{nd}f}, measured {got:+.{nd}f} {unit}")
    return "; ".join(bits)


LEDGER_COLUMNS = ["No", "Date", "Title", "Scope", "Category", "Target", "Owner", "Status", "Verdict", "Level",
                  "Detail", "N_before", "N_after", "FPY_before", "FPY_after", "FPY_delta", "FPY_p", "Metric", "Unit",
                  "Metric_before", "Metric_after", "Metric_p", "Loss_before", "Loss_after", "Loss_p", "Delivered",
                  "Before_from", "After_to", "Notes"]


def windows(first: pd.DataFrame, events, no: int, window: int = WINDOW) -> dict:
    """The modules compared for action ``no``: up to ``window`` first runs right
    before it and right after it, in its scope, never across another logged
    change. Also says whether a later change cut the *after* side short."""
    e_all = numbered(events)
    e = e_all[e_all["No"] == no].iloc[0]
    types = ev.scope_types(e["Scope"])
    d = first[first["BatteryType"].isin(types)].sort_values("Date", kind="stable")
    dates = pd.to_datetime(d["Date"])
    others = changes(events, types)
    others = others[others["Date"] != e["Date"]]
    prev = others.loc[others["Date"] < e["Date"], "Date"].max() if (others["Date"] < e["Date"]).any() else None
    nxt = others.loc[others["Date"] > e["Date"], "Date"].min() if (others["Date"] > e["Date"]).any() else None
    mb = dates < e["Date"]
    if prev is not None:
        mb &= dates >= prev
    ma = dates >= e["Date"]
    if nxt is not None:
        ma &= dates < nxt
    before, after = d[mb.to_numpy()].tail(int(window)), d[ma.to_numpy()].head(int(window))
    cut_short = nxt is not None and int(ma.sum()) < MIN_N
    return {"event": e, "before": before, "after": after, "prev": prev, "next": nxt, "cut_short": cut_short}


def verify(first: pd.DataFrame, events, no: int, window: int = WINDOW) -> dict:
    """Verdict for one action (see ``VERDICT_HELP``), with the evidence."""
    w = windows(first, events, no, window)
    e, before, after = w["event"], w["before"], w["after"]
    nb, na = len(before), len(after)
    out = {**w, "n_before": nb, "n_after": na, "fpy_before": np.nan, "fpy_after": np.nan, "fpy_p": np.nan,
           "check": _target_check(before.iloc[:0], after.iloc[:0], ""), "delivered": "", "compare": None}

    def done(verdict: str, detail: str) -> dict:
        out.update({"verdict": verdict, "level": VERDICT_LEVEL[verdict], "detail": detail})
        return out

    if e["Status"] == "Planned":
        return done("Planned", "Not done yet" + (f" — owner: {e['Owner']}" if e["Owner"].strip() else "") + ".")
    if nb:
        out["fpy_before"] = float((before["Status"] == "PASS").mean() * 100)
    if na:
        out["fpy_after"] = float((after["Status"] == "PASS").mean() * 100)
    if nb < MIN_BEFORE:
        return done("No baseline", f"Only {_count(nb, 'module')} before the change in the current filters — nothing "
                                   "to compare with.")
    if na < MIN_N:
        if w["cut_short"]:
            return done("Superseded", f"The next change came after only {_count(na, 'module')}: the two effects "
                                      "cannot be separated.")
        return done("Verifying", f"{_count(na, 'module')} since the change — {MIN_N - na} more before it can be "
                                 "judged.")
    cmp_ = ev.compare(before, after)
    out["compare"] = cmp_
    out["fpy_p"] = cmp_["fpy_p"]
    gain = out["fpy_after"] - out["fpy_before"]
    sig = is_significant(cmp_["fpy_p"])
    chk = _target_check(before, after, e["Target"])
    out["check"] = chk
    out["delivered"] = _planned_vs_observed(e, before, after)
    fpy_txt = (f"FPY {out['fpy_before']:.1f}% → {out['fpy_after']:.1f}% ({gain:+.1f} pp, "
               f"{significance_label(cmp_['fpy_p'])})")
    tgt_txt = ""
    if chk["metric"] and chk["unit"] != "%":
        tgt_txt = (f"; {chk['metric']} {_fmt(chk['m_before'], chk['unit'])} → "
                   f"{_fmt(chk['m_after'], chk['unit'])}")
    if np.isfinite(chk["loss_before"]):
        tgt_txt += f"; NOK it aimed at {chk['loss_before']:.1f}% → {chk['loss_after']:.1f}% of modules"
    detail = fpy_txt + tgt_txt + "."
    if sig and gain < 0:
        return done("Backfired", detail)
    if sig and gain > 0:
        return done("Worked", detail)
    if chk["better"]:
        return done("Target hit", detail)
    if w["next"] is None and na < SETTLED_N:
        return done("Verifying", detail[:-1] + f" — not significant yet with {na} modules.")
    if chk["worse"]:
        detail = detail[:-1] + " — the target moved the wrong way."
    return done("No effect", detail)


def ledger(first: pd.DataFrame, events, window: int = WINDOW) -> pd.DataFrame:
    """Every logged action with its verdict, in date order."""
    e_all = numbered(events)
    rows = []
    for no in e_all["No"]:
        v = verify(first, events, int(no), window)
        e, chk = v["event"], v["check"]
        rows.append({
            "No": int(no), "Date": e["Date"], "Title": e["Title"], "Scope": e["Scope"], "Category": e["Category"],
            "Target": e["Target"], "Owner": e["Owner"], "Status": e["Status"], "Verdict": v["verdict"],
            "Level": v["level"], "Detail": v["detail"], "N_before": v["n_before"], "N_after": v["n_after"],
            "FPY_before": v["fpy_before"], "FPY_after": v["fpy_after"],
            "FPY_delta": v["fpy_after"] - v["fpy_before"], "FPY_p": v["fpy_p"], "Metric": chk["metric"],
            "Unit": chk["unit"], "Metric_before": chk["m_before"], "Metric_after": chk["m_after"],
            "Metric_p": chk["m_p"], "Loss_before": chk["loss_before"], "Loss_after": chk["loss_after"],
            "Loss_p": chk["loss_p"], "Delivered": v["delivered"],
            "Before_from": pd.to_datetime(v["before"]["Date"]).min() if v["n_before"] else pd.NaT,
            "After_to": pd.to_datetime(v["after"]["Date"]).max() if v["n_after"] else pd.NaT, "Notes": e["Notes"],
        })
    return pd.DataFrame(rows, columns=LEDGER_COLUMNS)


def scorecard(led: pd.DataFrame) -> dict:
    """How the loop is doing: counts per verdict among the done actions."""
    d = led[led["Status"] != "Planned"] if len(led) else led
    vc = d["Verdict"].value_counts() if len(d) else pd.Series(dtype=int)
    judged = int(sum(vc.get(v, 0) for v in ("Worked", "Target hit", "No effect", "Backfired")))
    return {"done": int(len(d)), "planned": int((led["Status"] == "Planned").sum()) if len(led) else 0,
            "judged": judged, "worked": int(vc.get("Worked", 0) + vc.get("Target hit", 0)),
            "no_effect": int(vc.get("No effect", 0)), "backfired": int(vc.get("Backfired", 0)),
            "verifying": int(vc.get("Verifying", 0)), "superseded": int(vc.get("Superseded", 0)),
            "no_baseline": int(vc.get("No baseline", 0))}


def unexplained_shifts(change_points: pd.DataFrame, events, tolerance_days: float = 2.0) -> pd.DataFrame:
    """Detected process shifts with no logged change within ``tolerance_days``:
    the line moved and the log does not say why."""
    if change_points is None or not len(change_points):
        return pd.DataFrame(columns=list(change_points.columns) if change_points is not None else [])
    e = ev.done(ev.normalize_events(events))
    if e.empty:
        return change_points.reset_index(drop=True)
    tol = pd.Timedelta(days=tolerance_days)
    logged = e["Date"].to_numpy()
    keep = [not bool((np.abs(logged - np.datetime64(pd.Timestamp(d))) <= tol.to_timedelta64()).any())
            for d in change_points["Date"]]
    return change_points[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# What can be claimed about the target
# ---------------------------------------------------------------------------

def lower_bound(passed: int, n: int, confidence: float = CONFIDENCE) -> float:
    """One-sided exact (Clopper–Pearson) lower bound of the FPY, in percent:
    'with this confidence the true FPY is at least …'."""
    if n <= 0:
        return float("nan")
    if passed <= 0:
        return 0.0
    return float(_st.beta.ppf(1 - confidence, passed, n - passed + 1) * 100)


def upper_bound(passed: int, n: int, confidence: float = CONFIDENCE) -> float:
    if n <= 0:
        return float("nan")
    if passed >= n:
        return 100.0
    return float(_st.beta.ppf(confidence, passed + 1, n - passed) * 100)


def modules_to_prove(target: float, fails: int = 0, confidence: float = CONFIDENCE, start: int = 0) -> int | None:
    """Smallest number of modules (at least ``start``) that demonstrates
    FPY ≥ ``target`` % when exactly ``fails`` of them are NOK. None when the
    target cannot be demonstrated (100 %)."""
    if not np.isfinite(target) or target >= 100.0:
        return None
    if target <= 0:
        return max(start, fails)
    lo = max(int(start), int(fails) + 1, 1)
    if lower_bound(lo - fails, lo, confidence) >= target:
        return lo
    hi = max(lo * 2, 16)
    while lower_bound(hi - fails, hi, confidence) < target:
        hi *= 2
        if hi > 10 ** 9:
            return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if lower_bound(mid - fails, mid, confidence) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def streaks(first: pd.DataFrame) -> tuple[int, int]:
    """(first-pass OK modules in a row right now, longest such run)."""
    if first.empty:
        return 0, 0
    ok = (first.sort_values("Date", kind="stable")["Status"] == "PASS").to_numpy()
    best = cur = 0
    for v in ok:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return cur, best


def proof(modules: pd.DataFrame, target: float, confidence: float = CONFIDENCE) -> dict:
    """What the modules of one set-up say about the target.

    ``state``: *proven* (even the cautious bound is above the target),
    *below* (even the optimistic bound is under it) or *open*."""
    n = len(modules)
    k = int((modules["Status"] == "PASS").sum()) if n else 0
    fails = n - k
    lo, hi = lower_bound(k, n, confidence), upper_bound(k, n, confidence)
    cur, best = streaks(modules)
    days = _prod_days(modules)
    per_day = n / days if days else float("nan")
    out = {"n": n, "passed": k, "fails": fails, "fpy": k / n * 100 if n else float("nan"), "lower": lo, "upper": hi,
           "target": float(target), "confidence": confidence, "streak": cur, "best_streak": best,
           "per_day": per_day, "zero_fail_n": modules_to_prove(target, 0, confidence)}
    if not n:
        out["state"] = "open"
    else:
        out["state"] = "proven" if lo >= target else ("below" if hi < target else "open")
    total = modules_to_prove(target, fails, confidence, start=n)
    out["more_ok"] = None if total is None else max(0, total - n)
    # a fresh start: the shortest run without a NOK that proves the target, counted from the current streak
    z = out["zero_fail_n"]
    out["streak_to_go"] = None if z is None else max(0, z - cur)
    out["days_to_go"] = (math.ceil(out["streak_to_go"] / per_day)
                         if out["streak_to_go"] is not None and np.isfinite(per_day) and per_day > 0 else None)
    return out


# ---------------------------------------------------------------------------
# What still costs FPY, and what it takes to close the gap
# ---------------------------------------------------------------------------

BUDGET_COLUMNS = ["Mechanism", "Family", "Modules", "Loss", "Share", "Fix"]


def loss_budget(modules: pd.DataFrame) -> pd.DataFrame:
    """FPY points lost per failure mechanism (largest first)."""
    n = len(modules)
    nok = modules[modules["Status"] != "PASS"] if n else modules
    if not len(nok):
        return pd.DataFrame(columns=BUDGET_COLUMNS)
    mech = nok["Mechanism"].where(nok["Mechanism"].isin(MECHANISMS), "Missing corner") if "Mechanism" in nok \
        else pd.Series("Missing corner", index=nok.index)
    vc = mech.value_counts()
    rows = [{"Mechanism": m, "Family": MECHANISM_FAMILY[m], "Modules": int(c), "Loss": c / n * 100,
             "Share": c / len(nok) * 100, "Fix": MECHANISM_FIX[m]} for m, c in vc.items()]
    out = pd.DataFrame(rows, columns=BUDGET_COLUMNS)
    order = {m: i for i, m in enumerate(MECHANISMS)}
    return (out.assign(_o=out["Mechanism"].map(order)).sort_values(["Modules", "_o"], ascending=[False, True])
            .drop(columns="_o").reset_index(drop=True))


def centring(modules: pd.DataFrame, min_n: int = 15) -> dict:
    """Per battery type, the X / Y / yaw move that centres the placement of
    these modules (minus their mean least-squares placement)."""
    out = {}
    for t in _types_in(modules):
        v = modules[modules["BatteryType"] == t]
        if len(v) < min_n:
            continue
        s = rigid_mean_suggestion(v)
        if np.isfinite(s).all():
            out[t] = tuple(float(x) for x in s)
    return out


BRIDGE_COLUMNS = ["Step", "Key", "FPY", "Delta", "How", "Owner"]
BRIDGE_NOUN = {"centre": "centring the placement", "scatter": "halving the placement scatter",
               "shape": "square modules", "measure": "complete corner readings"}


def bridge(modules: pd.DataFrame, limit: float, nominals: Nominals, comps: dict | None = None) -> pd.DataFrame:
    """From the set-up's FPY to its potential, one lever after the other
    (cheapest first), each simulated on the set-up's own modules:

    centre the placement → halve its scatter → square modules → every corner
    measured. The first row is the measured FPY; the gains are the simulated
    differences, so the order matters for how the total is split, not for the
    total itself."""
    if modules.empty:
        return pd.DataFrame(columns=BRIDGE_COLUMNS)
    comps = centring(modules) if comps is None else comps
    start = float((modules["Status"] == "PASS").mean() * 100)

    def sim(levers: imp.Levers) -> float:
        return imp.fpy_after(modules, levers, comps, limit, nominals)

    base = sim(imp.Levers())
    s1 = sim(imp.Levers(compensate=True))
    s2 = sim(imp.Levers(True, 0.5, 0.5, 0.0, False))
    lv3 = imp.Levers(True, 0.5, 0.5, 1.0, True)
    s3 = sim(lv3)
    D = imp.apply_levers(modules, lv3, comps, nominals)
    complete = ~np.isnan(D).any(axis=1)
    with np.errstate(invalid="ignore"):
        s4 = float((np.abs(D[complete]) <= limit).all(axis=1).mean() * 100) if complete.any() else s3
    move = "; ".join(f"{t}: X {s[0]:+.2f}, Y {s[1]:+.2f} mm, yaw {s[2]:+.3f}°" for t, s in comps.items())
    steps = [
        ("Today", "today", base, "Measured on this set-up", ""),
        ("Centre the placement", "centre", s1, move or "Too few modules per type to suggest a move",
         "Process / maintenance"),
        ("Halve the placement scatter", "scatter", s2, "End-stop play, guides, side pushers, gripper", "Maintenance"),
        ("Modules arrive square", "shape", s3, "Deformation removed upstream", "Upstream / supplier"),
        ("Every corner measured", "measure", s4, "No missing or failed corner reads", "Quality / vision"),
    ]
    rows, level, prev = [], start, base
    for i, (label, key, val, how, owner) in enumerate(steps):
        delta = 0.0 if i == 0 else max(0.0, val - prev)
        level = start if i == 0 else min(100.0, level + delta)
        prev = max(prev, val)
        rows.append({"Step": label, "Key": key, "FPY": level, "Delta": delta, "How": how, "Owner": owner})
    return pd.DataFrame(rows, columns=BRIDGE_COLUMNS)


def required_scatter_cut(modules: pd.DataFrame, target: float, limit: float, nominals: Nominals,
                         comps: dict | None = None) -> float | None:
    """How much of the placement scatter must go, after centring, to reach the
    target on these modules (0..1); None when even a perfect placement is not
    enough."""
    if modules.empty:
        return None
    comps = centring(modules) if comps is None else comps
    return imp.required_reduction(modules, target, comps, limit, nominals, which="placement", compensate=True)


# ---------------------------------------------------------------------------
# Everything at once, and the story in sentences
# ---------------------------------------------------------------------------

def run(first: pd.DataFrame, events, target: float, limit: float, nominals: Nominals, window: int = WINDOW,
        setup_no: int | None = None) -> dict:
    """The whole loop for the first runs in scope.

    The loss budget, the bridge and the proof describe one set-up: the newest
    one with enough modules, or ``setup_no`` when given."""
    su = setups(first, events)
    led = ledger(first, events, window)
    pick = judged_setup(su)
    if setup_no is not None and len(su) and (su["Setup"] == setup_no).any():
        pick = su[su["Setup"] == setup_no].iloc[0]
    out = {"setups": su, "ledger": led, "score": scorecard(led), "target": float(target), "setup": pick,
           "current": su.iloc[-1] if len(su) else None, "events": numbered(events)}
    mods = setup_modules(first, pick) if pick is not None else first.iloc[:0]
    comps = centring(mods)
    out["modules"] = mods
    out["comps"] = comps
    out["budget"] = loss_budget(mods)
    out["bridge"] = bridge(mods, limit, nominals, comps)
    out["proof"] = proof(mods, target)
    out["scatter_cut"] = required_scatter_cut(mods, target, limit, nominals, comps)
    out["story"] = story(out)
    return out


def _date(d) -> str:
    return f"{pd.Timestamp(d):%b %d}" if pd.notna(d) else "–"


def story(loop: dict) -> dict:
    """Headline and short paragraphs for the brief. Every sentence carries its
    number; nothing is said that the data does not show."""
    su, led, s, target = loop["setups"], loop["ledger"], loop["setup"], loop["target"]
    out = {"headline": "No modules in the current filters.", "status": "", "done": "", "left": "", "path": "",
           "proof": "", "level": "neutral", "which": "current set-up"}
    if s is None:
        return out
    gap = s["FPY"] - target
    cur = loop["current"]
    which = "current set-up" if cur is None or int(cur["Setup"]) == int(s["Setup"]) else "latest judged set-up"
    out["which"] = which
    since = "the start of the data" if s["Setup"] == 1 and pd.isna(s["EventNo"]) else _date(s["Start"])
    if gap >= 0:
        out["level"] = "good"
        out["headline"] = f"First-pass yield is {s['FPY']:.1f}% in the {which} — at the {target:g}% target."
    else:
        out["level"] = "warn" if gap > -5 else "bad"
        out["headline"] = (f"First-pass yield is {s['FPY']:.1f}% in the {which} — {abs(gap):.1f} points "
                           f"short of the {target:g}% target.")
    out["status"] = (f"{int(s['N']):,} modules since {since}, 95% confidence interval {s['CI_lo']:.1f}–"
                     f"{s['CI_hi']:.1f}%.")
    first_judged = su[su["N"] >= MIN_N]
    if len(first_judged) and int(first_judged.iloc[0]["Setup"]) != int(s["Setup"]):
        f0 = first_judged.iloc[0]
        out["status"] += (f" The first set-up ran at {f0['FPY']:.1f}%: {s['FPY'] - f0['FPY']:+.1f} points since "
                          f"{_date(f0['Start'])}.")
    if cur is not None and int(cur["Setup"]) != int(s["Setup"]):
        out["status"] += (f" The line was changed again on {_date(cur['Start'])}: {_count(int(cur['N']), 'module')} "
                          "since, too few to judge.")

    sc = loop["score"]
    if sc["done"]:
        bits = [f"{sc['worked']} of {sc['judged']} judged line changes improved the process"] if sc["judged"] else []
        if sc["backfired"]:
            bits.append(f"{sc['backfired']} made it worse")
        if sc["verifying"]:
            bits.append(f"{sc['verifying']} still being verified")
        if sc["superseded"]:
            bits.append(f"{sc['superseded']} could not be judged because the next change came too soon")
        out["done"] = (", ".join(bits) + ".") if bits else f"{sc['done']} line change(s) logged, none judged yet."
        won = led[led["Verdict"] == "Worked"].sort_values("FPY_delta", ascending=False)
        if len(won):
            b = won.iloc[0]
            out["done"] += f" Biggest gain: “{b['Title']}” ({b['FPY_delta']:+.1f} points)."
    else:
        out["done"] = "No line change is logged yet — log what is done to the line to see what works."

    bud = loop["budget"]
    if len(bud):
        top = bud.iloc[0]
        total = float(bud["Loss"].sum())
        place = float(bud.loc[bud["Family"] == "Placement", "Loss"].sum())
        out["left"] = (f"{total:.1f} points are lost today; the largest part is {top['Mechanism'].lower()} "
                       f"({top['Loss']:.1f} points, {int(top['Modules'])} modules). Placement, the part an adjustment "
                       f"of the line can remove, accounts for {place:.1f} points.")
    else:
        out["left"] = "No module failed its first measurement in this set-up."

    br = loop["bridge"]
    if len(br) > 1 and gap < 0:
        reach = br[br["FPY"] >= target - 1e-9]
        b = br.set_index("Key")
        if len(reach):
            last = reach.iloc[0]
            names = [BRIDGE_NOUN[k] for k in br["Key"].iloc[1:int(reach.index[0]) + 1]]
            out["path"] = (f"Simulated on these modules, {_join(names)} "
                           f"{'brings' if len(names) == 1 else 'together bring'} FPY to {last['FPY']:.1f}%.")
        else:
            out["path"] = (f"Simulated on these modules, every lever together reaches {br['FPY'].iloc[-1]:.1f}% — "
                           f"below the {target:g}% target: the tolerance itself is tight for this process.")
        if b.loc["centre", "Delta"] >= 0.3:
            out["path"] += f" Centring the placement alone is worth {b.loc['centre', 'Delta']:+.1f} points."
        cut = loop.get("scatter_cut")
        if cut is not None and 0 < cut < 1:
            out["path"] += f" With the placement centred, {cut:.0%} of its scatter must go to reach the target."
        elif cut is None:
            out["path"] += " A perfect placement alone would not be enough: shape and missing corners must improve too."
    elif gap >= 0:
        out["path"] = "The set-up meets the target: hold it, and keep logging changes."

    p = loop["proof"]
    if p["n"]:
        if p["state"] == "proven":
            out["proof"] = (f"The target is demonstrated: with {p['n']:,} modules and {p['fails']} NOK the FPY is at "
                            f"least {p['lower']:.1f}% (95% confidence).")
        elif p["zero_fail_n"] is not None:
            days = f" — about {p['days_to_go']} production days at today's volume" if p["days_to_go"] else ""
            out["proof"] = (f"Demonstrating {target:g}% takes {p['zero_fail_n']:,} first-pass OK modules in a row "
                            f"without touching the line{days}. The current run is {p['streak']:,}; the best in this "
                            f"set-up was {p['best_streak']:,}.")
    return out


def _join(words: list[str]) -> str:
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + " and " + words[-1]
