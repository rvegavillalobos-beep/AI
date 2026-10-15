"""Plain-language findings and a prioritised action plan.

Every finding carries the number behind it, a severity, and — where one
exists — the action it suggests with an estimate of the FPY it is worth,
computed on the user's own modules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import compensation as comp
from . import improvement as imp
from . import patterns
from .analysis import Analysis
from .constants import MODE_LABELS
from .stats import fisher_2x2, significance_label

LEVEL_WEIGHT = {"bad": 3, "warn": 2, "info": 1, "good": 0}


@dataclass
class Insight:
    level: str                 # good | info | warn | bad
    title: str
    text: str
    topic: str
    action: str | None = None
    impact: float | None = None   # FPY points this action is estimated to be worth
    page: str | None = None
    score: float = 0.0


@dataclass
class Action:
    action: str
    why: str
    impact: float | None
    owner: str
    page: str | None


def _pct(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.1f}%"


def placement_suggestions(a: Analysis, weeks: int = 6) -> dict:
    """Rigid-fit compensation per type on the last ``weeks`` weeks (fast)."""
    out = {}
    lim, nom = a.settings.rules.spec_limit, a.settings.nominals
    for t in a.types:
        v = a.first[a.first["BatteryType"] == t]
        wk = sorted(v["WeekKey"].dropna().unique())[-weeks:]
        v = v[v["WeekKey"].isin(wk)]
        s = comp.rigid_mean_suggestion(v)
        if np.isfinite(s).all() and len(v) >= 15:
            before = comp.evaluate(v, (0.0, 0.0, 0.0), lim, nom)
            after = comp.evaluate(v, s, lim, nom)
            out[t] = {"setting": s, "n": len(v), "before": before["fpy"], "after": after["fpy"],
                      "gain": after["fpy"] - before["fpy"], "weeks": len(wk)}
    return out


def shift_effect(f: pd.DataFrame) -> dict:
    """Largest shift-to-shift difference in placement, and FPY by shift."""
    out: dict = {}
    if "Shift" not in f or f["Shift"].nunique() < 2:
        return out
    best = None
    for col, axis in (("Fit_Tx", "X"), ("Fit_Ty", "Y")):
        p = patterns.group_mean_test(f, "Shift", col)
        if not np.isfinite(p):
            continue
        means = f.groupby("Shift")[col].mean()
        worst = (means - means.mean()).abs().idxmax()
        delta = float(means[worst] - f.loc[f["Shift"] != worst, col].mean())
        if best is None or p < best[0]:
            best = (p, worst, delta, axis)
    if best:
        out.update({"p_mean": best[0], "worst": best[1], "delta": best[2], "axis": best[3]})
    bg = patterns.by_group(f, "Shift")
    if len(bg) >= 2:
        out["p_fpy"] = patterns.group_test(f, "Shift")
        out["fpy_min"], out["fpy_max"] = float(bg["FPY"].min()), float(bg["FPY"].max())
    return out


def lever_gains(a: Analysis, comps: dict | None = None) -> dict:
    """FPY gain (pp) of the standard what-if levers, after compensation."""
    f = a.first
    lim, nom = a.settings.rules.spec_limit, a.settings.nominals
    if comps is None:
        comps = {t: s["setting"] for t, s in placement_suggestions(a).items()}
    base = imp.fpy_after(f, imp.Levers(compensate=True), comps, lim, nom)
    return {
        "yaw": imp.fpy_after(f, imp.Levers(True, 0.0, 0.5, 0.0, False), comps, lim, nom) - base,
        "shift": imp.fpy_after(f, imp.Levers(True, 0.5, 0.0, 0.0, False), comps, lim, nom) - base,
        "shape": imp.fpy_after(f, imp.Levers(True, 0.0, 0.0, 1.0, True), comps, lim, nom) - base,
    }


def generate(a: Analysis, extras: dict | None = None) -> tuple[list[Insight], list[Action]]:
    ex = extras or {}
    r = a.settings.rules
    k = a.kpis
    f = a.first
    out: list[Insight] = []
    actions: list[Action] = []
    if f.empty:
        return out, actions
    n = len(f)

    # 1 — FPY vs target (always first)
    gap = k["fpy"] - r.fpy_target
    lo, hi = k["fpy_ci"]
    lvl = "good" if gap >= 0 else ("warn" if gap > -5 else "bad")
    out.append(Insight(
        lvl, "First-pass yield vs target",
        f"FPY is **{_pct(k['fpy'])}** on {n:,} modules (95% CI {lo:.1f}–{hi:.1f}%), "
        f"**{abs(gap):.1f} pp {'above' if gap >= 0 else 'below'}** the {r.fpy_target:.0f}% target.",
        "fpy", page="command", score=100.0))

    # 2 — latest week vs the 4 before
    w = a.weekly[a.weekly["Total"] >= 5] if len(a.weekly) else a.weekly
    if len(w) >= 3:
        last = w.iloc[-1]
        prev = w.iloc[-5:-1]
        pn, pp_ = int(prev["Total"].sum()), int(prev["Passed"].sum())
        if pn:
            prev_fpy = pp_ / pn * 100
            diff = last["PassRate"] - prev_fpy
            p = fisher_2x2(int(last["Total"] - last["Passed"]), int(last["Passed"]), pn - pp_, pp_)
            sig = np.isfinite(p) and p < 0.05
            lvl = ("good" if diff > 0 else "bad") if sig else "info"
            out.append(Insight(
                lvl, f"{last['CalendarWeek']} vs the weeks before",
                f"{last['CalendarWeek']} ran at **{last['PassRate']:.1f}%** ({int(last['Total'])} modules), "
                f"{diff:+.1f} pp vs the previous {len(prev)} weeks ({prev_fpy:.1f}%)"
                + (f" — a significant change ({significance_label(p)})." if sig
                   else " — within normal week-to-week noise."),
                "trend", page="command", score=LEVEL_WEIGHT[lvl] * 10 + abs(diff) / 2))

    # 3 — battery types differ?
    if len(a.types) == 2:
        s_, m_ = (f[f["BatteryType"] == t] for t in a.types)
        if len(s_) >= 20 and len(m_) >= 20:
            ps, pm = int((s_["Status"] == "PASS").sum()), int((m_["Status"] == "PASS").sum())
            p = fisher_2x2(len(s_) - ps, ps, len(m_) - pm, pm)
            fs, fm = ps / len(s_) * 100, pm / len(m_) * 100
            if np.isfinite(p) and p < 0.05:
                worse = a.types[1] if fm < fs else a.types[0]
                out.append(Insight(
                    "warn", f"{worse} is the weaker product",
                    f"{a.types[0]} {fs:.1f}% vs {a.types[1]} {fm:.1f}% FPY ({significance_label(p)}). "
                    f"Look at {worse} first.", "types", page="corners", score=14 + abs(fs - fm) / 2))

    # 4 — mechanisms
    nok = f[f["Status"] != "PASS"]
    if len(nok) >= 5 and "Mechanism" in nok:
        mc = nok["Mechanism"].value_counts(normalize=True) * 100
        placement = float(mc.reindex(["Shift", "Yaw", "Shift + yaw"]).fillna(0).sum())
        shape = float(mc.reindex(["Shape"]).fillna(0).sum())
        top = mc.index[0]
        out.append(Insight(
            "warn" if placement >= 50 else "info", "Why modules fail",
            f"**{placement:.0f}%** of NOK modules fail mainly because of their **placement** (shift / yaw) — "
            f"the part a conveyor adjustment can fix; **{shape:.0f}%** mainly because of their **shape**; "
            f"most common: **{top}** ({mc.iloc[0]:.0f}%).", "mechanism", page="anatomy", score=16 + placement / 10))

    # 5 — systematic placement → compensation
    sugg = ex.get("placement")
    if sugg is None:
        sugg = placement_suggestions(a)
    for t, s in sugg.items():
        dx, dy, yaw = s["setting"]
        if s["gain"] >= 1.0:
            out.append(Insight(
                "warn", f"{t}: systematic placement offset",
                f"Over the last {s['weeks']} weeks {t} modules sit on average **{-dx:+.2f} mm X, {-dy:+.2f} mm Y, "
                f"{-yaw:+.3f}°** off nominal. Undoing that offset would have lifted FPY from "
                f"{s['before']:.1f}% to **{s['after']:.1f}%** ({s['gain']:+.1f} pp).",
                "placement",
                action=f"Compensate {t}: X {dx:+.2f} mm, Y {dy:+.2f} mm, yaw {yaw:+.3f}° (validate in Compensation Studio)",
                impact=s["gain"] * (len(f[f["BatteryType"] == t]) / n), page="compensation",
                score=20 + s["gain"]))

    # 6 — where variation comes from
    bud = ex.get("budget")
    if bud is None:
        bud = imp.budget(f)
    if bud is not None and len(bud):
        for t in a.types:
            b = bud[bud["BatteryType"] == t].sort_values("VarShare", ascending=False)
            if len(b):
                top = b.iloc[0]
                placement_share = b[b["Mode"].isin(["Tx", "Ty", "Rot"])]["VarShare"].sum()
                out.append(Insight(
                    "info", f"{t}: where the scatter comes from",
                    f"**{placement_share:.0f}%** of the module-to-module variation is placement "
                    f"(shift + yaw); the largest single source is **{MODE_LABELS[top['Mode']]}** "
                    f"({top['VarShare']:.0f}%, σ {top['SD']:.2f} mm at the corners).",
                    "variation", page="anatomy", score=8 + top["VarShare"] / 10))

    # 7 — worst capability
    cap = a.capability[a.capability["Axis"].isin(["X", "Y"])].dropna(subset=["Ppk"])
    if len(cap):
        w0 = cap.sort_values("Ppk").iloc[0]
        p0 = w0["Ppk"]
        lvl = "bad" if p0 < 0.67 else ("warn" if p0 < 1.0 else ("info" if p0 < 1.33 else "good"))
        out.append(Insight(
            lvl, "Weakest corner",
            f"**{w0['BatteryType']} {w0['Corner']} {w0['Axis']}**: Ppk {p0:.2f} "
            f"({w0['Diagnosis'].lower()}), mean {w0['Mean']:+.2f} mm, σ {w0['Std']:.2f} mm → "
            f"{w0['OutPct']:.1f}% out of tolerance on its own.", "capability", page="corners",
            score=LEVEL_WEIGHT[lvl] * 9 + max(0, 1.33 - p0) * 5))

    # 8 — near misses
    nmr = k.get("near_miss_rate", np.nan)
    if np.isfinite(nmr) and nmr >= 5:
        out.append(Insight(
            "warn" if nmr >= 10 else "info", "Near misses",
            f"**{nmr:.1f}%** of modules passed with less than {r.near_miss:.1f} mm to spare on "
            f"their worst corner — they are one small disturbance away from NOK.", "risk", page="corners",
            score=7 + nmr / 5))

    # 9 — deformation link
    sl = ex.get("shape_link") or {}
    if sl.get("n_deformed", 0) >= 5 and np.isfinite(sl.get("p_value", np.nan)):
        sig = sl["p_value"] < 0.05
        ratio = sl["fail_rate_deformed"] / sl["fail_rate_square"] if sl.get("fail_rate_square") else np.nan
        out.append(Insight(
            "warn" if sig and np.isfinite(ratio) and ratio > 1.5 else "info", "Deformation and NOK",
            f"Deformed modules fail **{sl['fail_rate_deformed']:.0f}%** of the time vs "
            f"{sl['fail_rate_square']:.0f}% for square ones ({significance_label(sl['p_value'])}). "
            + (f"{sl.get('nok_if_square', 0)} NOK would have passed if the module had been square."
               if sl.get("nok_if_square") is not None else ""),
            "shape", page="shape", score=(12 if sig else 5) + (sl.get("nok_if_square", 0) or 0) / 5))

    # 10 — SPC signals
    board = ex.get("alarms")
    if board is not None and len(board):
        sig = board[board["State"] == "Signal"]
        watch = board[board["State"] == "Watch"]
        if len(sig):
            lst = "; ".join(f"**{r_['BatteryType']} {r_['Variable']}**: {r_['Reason']}" for _, r_ in sig.head(3).iterrows())
            out.append(Insight("bad", "Something changed in the last 2 weeks",
                               f"{len(sig)} variable(s) moved away from their recent baseline — {lst}.",
                               "spc", action="Find what changed on the line in the last 2 weeks (see SPC Monitor)",
                               page="spc", score=24 + len(sig)))
        elif len(watch):
            lst = "; ".join(f"{r_['BatteryType']} {r_['Variable']}: {r_['Reason']}" for _, r_ in watch.head(2).iterrows())
            out.append(Insight("info", "Weak signals in the last 2 weeks", f"Worth watching — {lst}.", "spc",
                               page="spc", score=6))
        else:
            out.append(Insight("good", "Process stable in the last 2 weeks",
                               "No placement variable moved away from its baseline (level, scatter or 3σ outliers).",
                               "spc", page="spc", score=5))

    # 11 — change points
    cps = ex.get("change_points")
    if cps is not None and len(cps):
        recent = cps.sort_values("Date").tail(3)
        bits = []
        for _, c in recent.iterrows():
            unit = "°" if "Yaw" in str(c["Variable"]) else " mm"
            bits.append(f"{c['Variable']} {c['Shift']:+.2f}{unit} on {pd.Timestamp(c['Date']):%b %d}")
        out.append(Insight("info", "Process shifts detected", "Dated level changes: " + "; ".join(bits)
                           + ". Log what was done then in **Interventions** to measure its effect.",
                           "change", page="events", score=9))

    # 12 — shifts
    sh = ex.get("shift_effect") or {}
    if sh.get("p_mean", 1) < 0.01 and sh.get("worst"):
        out.append(Insight("warn", "Shift-to-shift difference",
                           f"Shift **{sh['worst']}** places modules **{sh['delta']:+.2f} mm** in {sh['axis']} "
                           f"vs the other shifts ({significance_label(sh['p_mean'])}).",
                           "shift", action=f"Review set-up and handling differences on shift {sh['worst']}",
                           page="patterns", score=11))
    if np.isfinite(sh.get("p_fpy", np.nan)) and sh["p_fpy"] < 0.05:
        out.append(Insight("warn", "FPY depends on the shift",
                           f"FPY ranges from {sh['fpy_min']:.1f}% to {sh['fpy_max']:.1f}% between shifts "
                           f"({significance_label(sh['p_fpy'])}).", "shift", page="patterns", score=12))

    # 13 — start-up
    su = ex.get("startup") or {}
    if su and np.isfinite(su.get("p_value", np.nan)) and su["p_value"] < 0.05 and su["fpy_startup"] < su["fpy_steady"]:
        share = su["n_startup"] / (su["n_startup"] + su["n_steady"])
        impact = (su["fpy_steady"] - su["fpy_startup"]) * share
        tx_txt = (f"; they land {su['Fit_Tx_shift']:+.2f} mm in X."
                  if np.isfinite(su.get("Fit_Tx_shift", np.nan)) and su.get("Fit_Tx_p", 1) < 0.05 else ".")
        out.append(Insight(
            "warn", "Start-up effect after line stops",
            f"The first {su['k']} modules after a stop of ≥{su['gap_min']:.0f} min pass **{su['fpy_startup']:.1f}%** "
            f"vs {su['fpy_steady']:.1f}% in steady running ({significance_label(su['p_value'])}){tx_txt}",
            "startup", action="Investigate the first cycles after stops (warm-up, end-stop damping, first-part routine)",
            impact=impact, page="patterns", score=13 + impact))

    # 14 — measurement
    rm = ex.get("remeasure") or {}
    if rm.get("max_grr") is not None and np.isfinite(rm["max_grr"]):
        lvl = "bad" if rm["max_grr"] >= 30 else ("warn" if rm["max_grr"] >= 10 else "good")
        nok_ok = rm.get("nok_to_ok_rate", np.nan)
        out.append(Insight(
            lvl, "Measurement repeatability",
            f"Measuring the same module twice moves a corner by σ≈{rm['median_sigma']:.2f} mm; "
            f"%GRR up to **{rm['max_grr']:.0f}%** ({rm['worst_coord']}). "
            + (f"{nok_ok:.0f}% of re-measured NOK modules passed the second time." if np.isfinite(nok_ok) else ""),
            "msa", action="Run a formal gauge R&R study on the station" if lvl == "bad" else None,
            page="remeasure", score=LEVEL_WEIGHT[lvl] * 8))

    # 15 — data quality
    h = ex.get("health") or {}
    inc = int((f["MissingCorners"] > 0).sum()) if "MissingCorners" in f else 0
    if inc / n >= 0.01:
        out.append(Insight("warn", "Missing corners",
                           f"{inc} first runs ({inc / n * 100:.1f}%) lack at least one corner"
                           + (" and count as NOK." if r.incomplete_as_fail else "."),
                           "data", action="Check why corners are not reported (vision misses, failed reads)",
                           impact=inc / n * 100 * (k["fpy"] / 100) if r.incomplete_as_fail else None,
                           page="data", score=10 + inc / n * 100))
    if h.get("similar_part_ids") and not h.get("lookalike_ids_merged"):
        out.append(Insight("info", "Look-alike Part IDs",
                           f"{len(h['similar_part_ids'])} Part ID pair(s) differ only by letter O vs digit 0; their "
                           "run history may be split. Merge them in Data & Settings.", "data", page="data", score=4))

    out.sort(key=lambda i: -i.score)

    # ---- action plan ---------------------------------------------------------------
    owners = {"placement": "Process / maintenance", "startup": "Process", "shift": "Production",
              "msa": "Quality", "data": "Quality / IT"}
    for i in out:
        if i.action:
            actions.append(Action(i.action, i.title, i.impact, owners.get(i.topic, "Process"), i.page))
    lev = ex.get("levers") or {}
    for label, gain, owner, page in (
        ("Halve the yaw scatter (guide rails, side pushers)", lev.get("yaw"), "Maintenance", "target"),
        ("Halve the X/Y placement scatter (end-stop play, gripper)", lev.get("shift"), "Maintenance", "target"),
        ("Eliminate module deformation (upstream process)", lev.get("shape"), "Upstream / supplier", "target"),
    ):
        if gain is not None and np.isfinite(gain) and gain >= 0.5:
            actions.append(Action(label, "What-if simulation on your modules", gain, owner, page))
    actions.sort(key=lambda x: -(x.impact if x.impact is not None and np.isfinite(x.impact) else -1))
    return out, actions
