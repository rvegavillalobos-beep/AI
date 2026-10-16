"""Height (Z) figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..constants import CORNER_LABELS, CORNERS
from ..geometry import polygon
from . import theme as T


def z_surface(surf: dict, bat_type: str, title: str | None = None, height: int = 330) -> go.Figure:
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
        colorbar=dict(title="Z (mm)", thickness=10, len=0.9, xpad=6),
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
    # a top view, not a measurement: no axes, so the drawing gets the whole width
    fig.add_annotation(x=max(cx), y=max(cy) + 105, text="front →", showarrow=False, xanchor="right",
                       yanchor="middle", font=dict(size=11, color=P.ink["secondary"]))
    fig.update_xaxes(visible=False, range=[min(cx) - 120, max(cx) + 120], constrain="domain")
    fig.update_yaxes(visible=False, range=[max(cy) + 140, min(cy) - 140],
                     scaleanchor="x", scaleratio=1, constrain="domain")
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
