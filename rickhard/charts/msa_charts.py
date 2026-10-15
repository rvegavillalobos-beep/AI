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
    fig = go.Figure(go.Heatmap(z=pct, x=[f"→ {c}" for c in tr.columns], y=[f"{r} at run 1" for r in tr.index],
                               text=text, texttemplate="%{text}", colorscale=P.seq_light, zmin=0, zmax=100,
                               showscale=False, xgap=3, ygap=3,
                               hovertemplate="%{y} %{x}: %{z:.0f}% of the row<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Status at the next run")
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
    T.hline(fig, 10, P.target, "dot", 1.5, "10% good", "top left")
    T.hline(fig, 30, P.limit, "dash", 1.5, "30% not acceptable", "top left")
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
    fig.update_xaxes(title_text="Minutes between the two runs (capped at 24 h)")
    fig.update_yaxes(title_text="Pairs")
    return T.style(fig, height=height, title=title, legend=False)


# ---------------------------------------------------------------------------
# Height (Z)
# ---------------------------------------------------------------------------

def z_surface(surf: dict, bat_type: str, title: str | None = None, height: int = 520) -> go.Figure:
    """3D view of the average module surface with the four corners (Z is
    exaggerated by the aspect ratio, values in the hover are real mm)."""
    P = T.pal()
    if not surf:
        return T.empty("No Z data.")
    nom = surf["nominal"]
    X, Y = np.meshgrid(surf["x"], surf["y"])
    Z = surf["z"]
    zabs = max(0.05, float(np.nanmax(np.abs(Z))) * 1.1, max(abs(v) for v in surf["corner_z"].values()) * 1.1)
    magnify = 1.0
    fig = go.Figure()
    fig.add_trace(go.Surface(x=X, y=Y, z=Z, colorscale=P.diverging, cmin=-zabs, cmax=zabs,
                             colorbar=dict(title="Z (mm)", len=0.6), opacity=0.95,
                             hovertemplate="X %{x:.0f} · Y %{y:.0f}<br>Z %{z:+.3f} mm<extra></extra>"))
    cx = [nom[c][0] for c in CORNERS]
    cy = [nom[c][1] for c in CORNERS]
    cz = [surf["corner_z"][c] for c in CORNERS]
    fig.add_trace(go.Scatter3d(x=cx, y=cy, z=[v * magnify for v in cz], mode="markers+text",
                               text=[f"{c} {v:+.2f}" for c, v in zip(CORNERS, cz)], textposition="top center",
                               marker=dict(size=6, color=P.type(bat_type)), hoverinfo="text", name="Corners (mean Z)"))
    for (x0, y0), z0 in zip(zip(cx, cy), cz):
        fig.add_trace(go.Scatter3d(x=[x0, x0], y=[y0, y0], z=[0, z0 * magnify], mode="lines",
                                   line=dict(color=P.ink["muted"], width=4), hoverinfo="skip", showlegend=False))
    xs, ys = polygon(nom)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=[0] * len(xs), mode="lines", line=dict(color=P.nominal, width=4),
                               name="Nominal (Z = 0)", hoverinfo="skip"))
    fig.update_layout(scene=dict(
        xaxis_title="X (front →)", yaxis_title="Y", zaxis_title="Z (mm)",
        zaxis=dict(range=[-zabs * 1.2, zabs * 1.2]),
        yaxis=dict(autorange="reversed"), aspectmode="manual", aspectratio=dict(x=1.6, y=1.0, z=0.5),
        camera=dict(eye=dict(x=-1.35, y=-1.6, z=0.9))))
    return T.style(fig, height=height, title=title, margin=dict(t=30))


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
        fig.update_xaxes(type="category", row=r, col=c)
    return T.style(fig, height=height, title=title, legend_bottom=True)
