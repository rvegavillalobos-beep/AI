"""Re-measurement / repeatability and height (Z) figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import CORNER_LABELS, CORNERS
from ..geometry import polygon
from . import theme as T


def transition_heatmap(tr: pd.DataFrame, title: str | None = None, height: int = 300) -> go.Figure:
    P = T.pal()
    if tr is None or tr.empty:
        return T.empty("No re-measured modules.")
    z = tr.to_numpy(float)
    rows = z.sum(axis=1, keepdims=True)
    pct = z / np.where(rows == 0, 1, rows) * 100
    text = [[f"<b>{int(v)}</b> · {p:.0f}%" for v, p in zip(row, prow)] for row, prow in zip(z, pct)]
    short = {"INCOMPLETE": "INCOMPL."}
    fig = go.Figure(go.Heatmap(z=pct, x=[short.get(c, c) for c in tr.columns], y=[short.get(r, r) for r in tr.index],
                               text=text, texttemplate="%{text}", textfont=dict(size=13), colorscale=P.seq_light,
                               zmin=0, zmax=100, showscale=False, xgap=3, ygap=3,
                               hovertemplate="run 1 %{y} → next run %{x}: %{z:.0f}% of the row<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title_text="Status at run 1")
    fig.update_xaxes(title_text="Status at the next run", tickangle=0, side="bottom")
    return T.style(fig, height=height, title=title, legend=False)


def grr_bars(rep: pd.DataFrame, title: str | None = None, height: int = 340) -> go.Figure:
    P = T.pal()
    if rep is None or rep.empty:
        return T.empty("Needs modules measured twice within a short time.")
    fig = go.Figure()
    for t in [x for x in ("Type S", "Type M") if (rep["BatteryType"] == x).any()]:
        s = rep[rep["BatteryType"] == t]
        fig.add_trace(go.Bar(x=s["Coord"].str.replace("_", " "), y=s["GRR"], name=t, marker_color=P.type(t),
                             customdata=np.stack([s["SigmaRepeat"], s["N"]], axis=1),
                             hovertemplate="<b>%{x}</b> · " + t + "<br>%GRR %{y:.1f}%<br>σ repeat %{customdata[0]:.3f} mm"
                                           " (n=%{customdata[1]})<extra></extra>"))
    T.hline(fig, 10, P.target, "dot", 1.5, "10% good", "top right")
    T.hline(fig, 30, P.limit, "dash", 1.5, "30% not acceptable", "top right")
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(title_text="%GRR (6σ / tolerance)", ticksuffix="%", rangemode="tozero",
                     range=[0, max(35.0, float(rep["GRR"].max()) * 1.15)])
    return T.style(fig, height=height, title=title)


def bland_altman(pairs: pd.DataFrame, coord: str, max_gap: float, title: str | None = None, height: int = 360) -> go.Figure:
    P = T.pal()
    p = pairs[pairs["GapMin"] <= max_gap]
    a, b = p[f"a_{coord}"].to_numpy(float), p[f"b_{coord}"].to_numpy(float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return T.empty("Not enough pairs.")
    mean, diff = (a[m] + b[m]) / 2, b[m] - a[m]
    md, sd = float(diff.mean()), float(diff.std(ddof=1))
    fig = go.Figure(go.Scatter(x=mean, y=diff, mode="markers", marker=dict(size=7, color=T.rgba(P.accent, 0.6)),
                               customdata=np.stack([p.loc[m, "PartID"].astype(object).map(str), p.loc[m, "GapMin"]], axis=1),
                               hovertemplate="%{customdata[0]}<br>mean %{x:+.2f} · Δ %{y:+.2f} mm<br>"
                                             "%{customdata[1]:.0f} min apart<extra></extra>", showlegend=False))
    T.hline(fig, md, P.ink["primary"], "solid", 1.5, f"bias {md:+.3f}", "top left")
    T.hline(fig, md + 1.96 * sd, P.limit, "dash", 1.2, f"+1.96σ {md + 1.96 * sd:+.2f}", "top left")
    T.hline(fig, md - 1.96 * sd, P.limit, "dash", 1.2, f"−1.96σ {md - 1.96 * sd:+.2f}", "bottom left")
    fig.update_xaxes(title_text=f"Mean of the two runs — {coord.replace('_', ' ')} (mm)")
    fig.update_yaxes(title_text="Second run − first run (mm)")
    return T.style(fig, height=height, title=title, legend=False)


def gap_hist(pairs: pd.DataFrame, max_gap: float, title: str | None = None, height: int = 260) -> go.Figure:
    P = T.pal()
    if pairs is None or pairs.empty:
        return T.empty("No re-measurements.")
    g = pairs["GapMin"].clip(upper=24 * 60)
    fig = go.Figure(go.Histogram(x=g, nbinsx=48, marker_color=P.accent, showlegend=False,
                                 hovertemplate="%{x} min: %{y} pairs<extra></extra>"))
    fig.add_vline(x=max_gap, line=dict(color=P.ink["primary"], dash="dash"),
                  annotation_text=f"{max_gap:.0f} min", annotation_font=dict(size=10, color=P.ink["secondary"]))
    fig.update_xaxes(title_text="Minutes between runs (cap 24 h)")
    fig.update_yaxes(title_text="Pairs")
    return T.style(fig, height=height, title=title, legend=False)


# ---------------------------------------------------------------------------
# Height (Z)
# ---------------------------------------------------------------------------

def z_surface(surf: dict, bat_type: str, title: str | None = None, height: int = 420) -> go.Figure:
    """Top view of the average module surface: colour = fitted height Z,
    with the mean Z of each corner (same orientation as the outline plots)."""
    P = T.pal()
    if not surf:
        return T.empty("No Z data.")
    nom = surf["nominal"]
    Z = np.asarray(surf["z"], float)
    zabs = max(0.05, float(np.nanmax(np.abs(Z))) * 1.1, max(abs(v) for v in surf["corner_z"].values()) * 1.1)
    fig = go.Figure(go.Contour(
        x=surf["x"], y=surf["y"], z=Z, colorscale=P.diverging, zmin=-zabs, zmax=zabs, ncontours=14,
        contours=dict(coloring="heatmap", showlabels=True, labelfont=dict(size=10, color=P.ink["secondary"])),
        line=dict(width=0.6, color=T.rgba(P.ink["secondary"], 0.4)),
        colorbar=dict(title="Z (mm)", thickness=12, len=0.8),
        hovertemplate="X %{x:.0f} · Y %{y:.0f}<br>Z %{z:+.3f} mm<extra></extra>"))
    xs, ys = polygon(nom)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=P.nominal, width=2, dash="dash"),
                             name="Nominal outline", hoverinfo="skip", showlegend=False))
    cx = [nom[c][0] for c in CORNERS]
    cy = [nom[c][1] for c in CORNERS]
    cz = [surf["corner_z"][c] for c in CORNERS]
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers+text", text=[f"<b>{c}</b> {v:+.2f}" for c, v in zip(CORNERS, cz)],
        textposition=["bottom left", "top left", "bottom right", "top right"], cliponaxis=False,
        textfont=dict(size=12, color=P.ink["primary"]), marker=dict(size=10, color=P.type(bat_type),
                                                                   line=dict(color=P.ink["surface"], width=2)),
        hovertemplate="%{text} mm (mean corner Z)<extra></extra>", showlegend=False))
    fig.update_xaxes(title_text="Global X (mm) — front →", showgrid=False)
    fig.update_yaxes(title_text="Global Y (mm)", autorange="reversed", scaleanchor="x", scaleratio=1, showgrid=False)
    return T.style(fig, height=height, title=title, legend=False)


def z_box(first: pd.DataFrame, z_lim: tuple[float, float], title: str | None = None, height: int = 320) -> go.Figure:
    P = T.pal()
    fig = go.Figure()
    for t in [x for x in ("Type S", "Type M") if (first["BatteryType"] == x).any()]:
        s = first[first["BatteryType"] == t]
        xs, ys = [], []
        for c in CORNERS:
            v = s[f"{c}_Z"].dropna().to_numpy(float)
            xs += [CORNER_LABELS[c]] * len(v)
            ys += list(v)
        fig.add_trace(go.Box(x=xs, y=ys, name=t, marker_color=P.type(t), boxpoints=False,
                             hovertemplate="%{x}<br>Z %{y:+.3f} mm<extra>" + t + "</extra>"))
    T.hline(fig, z_lim[0], P.limit, "dash", 1.3, f"LSL {z_lim[0]:+g}", "bottom left")
    T.hline(fig, z_lim[1], P.limit, "dash", 1.3, f"USL {z_lim[1]:+g}", "top left")
    fig.update_layout(boxmode="group")
    fig.update_yaxes(title_text="Z deviation (mm)")
    return T.style(fig, height=height, title=title)


def z_trends(first: pd.DataFrame, title: str | None = None, height: int = 430) -> go.Figure:
    P = T.pal()
    cols = [("Z_Level", "Level (mm)"), ("Z_Pitch", "Pitch (mm/m)"), ("Z_Roll", "Roll (mm/m)"), ("Z_Twist", "Twist (mm)")]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[c[1] for c in cols], horizontal_spacing=0.08, vertical_spacing=0.18)
    for i, (col, _) in enumerate(cols):
        r, c = i // 2 + 1, i % 2 + 1
        for t in [x for x in ("Type S", "Type M") if (first["BatteryType"] == x).any()]:
            s = first[first["BatteryType"] == t]
            w = s.groupby(["WeekKey", "CalendarWeek"])[col].agg(["mean", "std"]).reset_index().sort_values("WeekKey")
            fig.add_trace(go.Scatter(x=w["CalendarWeek"], y=w["mean"], mode="lines+markers", name=t, legendgroup=t,
                                     showlegend=i == 0, line=dict(color=P.type(t), width=2),
                                     marker=dict(size=7, color=P.type(t)),
                                     error_y=dict(type="data", array=w["std"], color=T.rgba(P.type(t), 0.4), thickness=1),
                                     hovertemplate="%{x}: %{y:+.3f}<extra>" + t + "</extra>"), row=r, col=c)
    weeks = first.drop_duplicates("WeekKey").sort_values("WeekKey")["CalendarWeek"].tolist()
    for i in range(4):
        r, c = i // 2 + 1, i % 2 + 1
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=weeks, row=r, col=c)
        T.thin_ticks(fig, weeks, max_ticks=6, row=r, col=c)
    return T.style(fig, height=height, title=title, legend_bottom=True)
