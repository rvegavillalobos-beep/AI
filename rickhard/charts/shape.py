"""Shape & squareness figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..constants import NOK_CAUSE_HELP, NOK_CAUSES, SQ_CAUSE_ORDER
from . import theme as T


def squareness_status(sq: pd.DataFrame, title: str | None = None, height: int = 280) -> go.Figure:
    P = T.pal()
    if sq.empty:
        return T.empty("No complete modules.")
    t = sq.groupby(["BatteryType", "SquarenessStatus"]).size().unstack(fill_value=0)
    for s in ("SQUARE OK", "DEFORMED"):
        if s not in t:
            t[s] = 0
    tot = t.sum(axis=1)
    pct = t.div(tot, axis=0) * 100
    fig = go.Figure()
    for s, label in (("SQUARE OK", "Square"), ("DEFORMED", "Deformed")):
        last = s == "DEFORMED"  # thin last segment: label outside the bar
        fig.add_trace(go.Bar(
            y=pct.index, x=pct[s], orientation="h", name=label, marker_color=P.status(s),
            marker_line=dict(width=1.5, color=P.ink["surface"]),
            text=[f"{p:.1f}%" if (last or p >= 12) else "" for p in pct[s]],
            textposition="outside" if last else "inside", textangle=0, cliponaxis=False, insidetextanchor="middle",
            textfont=dict(color=P.status(s) if last else "#ffffff"), customdata=np.stack([t[s], tot], axis=1),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}% (%{{customdata[0]}} of %{{customdata[1]}})<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(range=[0, 114], tickvals=[0, 25, 50, 75, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=height, title=title)


def root_causes(sq: pd.DataFrame, title: str | None = None, height: int = 280) -> go.Figure:
    P = T.pal()
    d = sq[sq["SquarenessStatus"] == "DEFORMED"] if len(sq) else sq
    if d.empty:
        return T.empty("No deformed modules in this scope.")
    t = d.groupby(["BatteryType", "CauseCategory"]).size().unstack(fill_value=0)
    tot = t.sum(axis=1)
    pct = t.div(tot, axis=0) * 100
    fig = go.Figure()
    for c in SQ_CAUSE_ORDER:
        if c not in pct:
            continue
        fig.add_trace(go.Bar(
            y=pct.index, x=pct[c], orientation="h", name=c, marker_color=P.sq_cause(c),
            marker_line=dict(width=1.5, color=P.ink["surface"]),
            text=[f"{p:.0f}%" if p >= 9 else "" for p in pct[c]], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#ffffff"), customdata=np.stack([t[c], tot], axis=1),
            hovertemplate=f"<b>%{{y}}</b><br>{c}: %{{x:.1f}}% (%{{customdata[0]}} of %{{customdata[1]}})<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title_text="Share of deformed modules")
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=height, title=title)


def nok_causes(causes: pd.DataFrame | None, title: str | None = None, height: int = 260) -> go.Figure:
    """Why each NOK module failed (placement / shape counterfactuals), share per battery type."""
    P = T.pal()
    if causes is None or causes.empty:
        return T.empty("No NOK modules with all four corners measured.")
    t = causes.copy()
    tot = t.sum(axis=1)
    pct = t.div(tot, axis=0) * 100
    fig = go.Figure()
    for c in NOK_CAUSES:
        if c not in pct or t[c].sum() == 0:
            continue
        fig.add_trace(go.Bar(
            y=[f"<b>{i}</b><br>{int(n)} NOK" for i, n in zip(pct.index, tot)], x=pct[c], orientation="h", name=c,
            marker_color=P.nok_cause(c), marker_line=dict(width=1.5, color=P.ink["surface"]),
            text=[f"{p:.0f}% ({int(n)})" if p >= 14 else "" for p, n in zip(pct[c], t[c])], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#ffffff"), customdata=np.stack([t[c], tot], axis=1),
            hovertemplate=f"{c}: %{{x:.1f}}% (%{{customdata[0]}} of %{{customdata[1]}} NOK)<br>"
                          f"{NOK_CAUSE_HELP.get(c, '')}<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title_text="Share of NOK modules")
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=height, title=title)


def failure_map(first: pd.DataFrame, limit: float, title: str | None = None, height: int = 470) -> go.Figure:
    """Placement error vs shape error of every complete first run."""
    P = T.pal()
    d = first.dropna(subset=["RigidMax", "ShapeMax"])
    if d.empty:
        return T.empty("No complete modules.")
    fig = go.Figure()
    for key, label, color, size, op in (("PASS", "OK", P.ink["muted"], 6, 0.35), ("FAIL", "NOK", P.status("FAIL"), 8, 0.85)):
        m = (d["Status"] == "PASS") if key == "PASS" else (d["Status"] != "PASS")
        s = d[m]
        if s.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=s["RigidMax"], y=s["ShapeMax"], mode="markers", name=label,
            marker=dict(size=size, color=color, opacity=op),
            customdata=np.stack([s["PartID"].astype(object).map(str), s["Mechanism"].astype(object).map(str),
                                 s["WorstAbs"]], axis=1),
            hovertemplate=("<b>%{customdata[0]}</b><br>placement %{x:.2f} mm · shape %{y:.2f} mm"
                           "<br>worst corner %{customdata[2]:.2f} mm · %{customdata[1]}<extra></extra>")))
    xmax = max(float(d["RigidMax"].max()), limit) * 1.08
    ymax = max(float(d["ShapeMax"].max()), 1.0) * 1.1
    fig.add_vline(x=limit, line=dict(color=P.limit, dash="dash", width=1.3),
                  annotation_text=f"placement alone ≥ {limit:g} mm",
                  annotation_font=dict(size=10, color=P.ink["secondary"]))
    fig.add_trace(go.Scatter(x=[0, limit], y=[limit, 0], mode="lines", name="placement + shape = tolerance",
                             line=dict(color=P.ink["secondary"], width=1.2, dash="dot"), hoverinfo="skip"))
    fig.update_xaxes(title_text="Placement error — worst corner if the module were square (mm)", range=[0, xmax])
    fig.update_yaxes(title_text="Shape error — worst corner residual (mm)", range=[0, ymax])
    return T.style(fig, height=height, title=title)


def fail_rate_by_band(by_band: pd.DataFrame, diag_tol: float, overall_fail: float | None = None,
                      title: str | None = None, height: int = 320) -> go.Figure:
    P = T.pal()
    if by_band is None or by_band.empty:
        return T.empty("No data.")
    d = by_band
    colors = [P.status("FAIL") if a else P.accent for a in d["AboveTol"]]
    fig = go.Figure(go.Bar(
        x=d["Label"], y=d["FailRate"], marker_color=colors,
        error_y=dict(type="data", symmetric=False, array=(d["CI_hi"] - d["FailRate"]).clip(lower=0),
                     arrayminus=(d["FailRate"] - d["CI_lo"]).clip(lower=0), color=P.ink["secondary"],
                     thickness=1.2, width=4),
        customdata=d["N"],
        hovertemplate="ΔDiag %{x} mm<br>NOK rate %{y:.1f}% (n=%{customdata})<extra></extra>", showlegend=False))
    if overall_fail is not None:
        T.hline(fig, overall_fail, P.ink["secondary"], "dot", 1.2, f"overall {overall_fail:.1f}%", "top left")
    fig.update_xaxes(title_text=f"Diagonal delta (mm) · red = above {diag_tol:g} mm", type="category",
                     tickmode="array", tickvals=list(d["Label"]),
                     ticktext=[f"{lab}<br><span style='font-size:10px'>n={n}</span>" for lab, n in zip(d["Label"], d["N"])])
    fig.update_yaxes(title_text="NOK rate", ticksuffix="%",
                     range=[0, max(10.0, float(d["CI_hi"].max()) * 1.25)])
    fig.update_layout(bargap=0.35)
    return T.style(fig, height=height, title=title, legend=False)


def weekly_deformation(wd: pd.DataFrame, title: str | None = None, height: int = 300) -> go.Figure:
    P = T.pal()
    if wd is None or wd.empty:
        return T.empty("No data.")
    fig = go.Figure()
    for t in sorted(wd["BatteryType"].unique()):
        s = wd[wd["BatteryType"] == t]
        fig.add_trace(go.Scatter(x=s["CalendarWeek"], y=s["Share"], mode="lines+markers", name=t,
                                 line=dict(color=P.type(t), width=2),
                                 marker=dict(size=8, color=P.type(t), line=dict(color=P.ink["surface"], width=2)),
                                 customdata=np.stack([s["Deformed"], s["N"], s["MedianDiag"]], axis=1),
                                 hovertemplate=("<b>%{x}</b> · " + t + "<br>%{y:.1f}% deformed (%{customdata[0]} of "
                                                "%{customdata[1]})<br>median ΔDiag %{customdata[2]:.2f} mm<extra></extra>")))
    weeks = wd.drop_duplicates("CalendarWeek").sort_values("WeekKey")["CalendarWeek"].tolist() \
        if "WeekKey" in wd else list(dict.fromkeys(wd["CalendarWeek"]))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=weeks)
    T.thin_ticks(fig, weeks, max_ticks=8)
    fig.update_yaxes(title_text="Deformed", ticksuffix="%", rangemode="tozero")
    return T.style(fig, height=height, title=title)


def diag_hist(sq: pd.DataFrame, diag_tol: float, title: str | None = None, height: int = 300) -> go.Figure:
    P = T.pal()
    if sq.empty:
        return T.empty("No data.")
    fig = go.Figure()
    for t in sorted(sq["BatteryType"].unique()):
        v = sq.loc[sq["BatteryType"] == t, "DeltaDiag"]
        fig.add_trace(go.Histogram(x=v, name=t, marker_color=T.rgba(P.type(t), 0.6), nbinsx=50,
                                   hovertemplate="%{x} mm: %{y}<extra>" + t + "</extra>"))
    fig.add_vline(x=diag_tol, line=dict(color=P.limit, dash="dash", width=1.5),
                  annotation_text=f"limit {diag_tol:g} mm", annotation_font=dict(size=10, color=P.ink["secondary"]))
    fig.update_layout(barmode="overlay")
    p99 = float(np.nanpercentile(sq["DeltaDiag"], 99)) if sq["DeltaDiag"].notna().any() else diag_tol
    fig.update_xaxes(title_text="Diagonal delta (mm) — axis cut at the 99th percentile",
                     range=[0, max(diag_tol * 2.0, p99 * 1.15)])
    fig.update_yaxes(title_text="Modules")
    return T.style(fig, height=height, title=title)
