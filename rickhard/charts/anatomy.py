"""Error-anatomy figures: where the error comes from, mode by mode."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import Nominals
from ..constants import CORNERS, MODE_FAMILY, MODE_LABELS, MODES
from ..geometry import polygon
from ..improvement import mode_shape_points, mode_vector
from . import theme as T


def budget_bars(bud: pd.DataFrame, bat_type: str, title: str | None = None, height: int = 420) -> go.Figure:
    """Per mode: systematic part (signed mean, mm) and random part (σ, mm),
    each bar labelled with its share."""
    P = T.pal()
    b = bud[bud["BatteryType"] == bat_type].set_index("Mode").reindex(list(MODES)).reset_index()
    if b["N"].isna().all():
        return T.empty("Not enough complete modules.")
    labels = [MODE_LABELS[m] for m in b["Mode"]]
    colors = [P.family(MODE_FAMILY[m]) for m in b["Mode"]]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Systematic — mean error (mm)", "Random — module-to-module σ (mm)"))
    fig.add_trace(go.Bar(
        y=labels, x=b["Bias"], orientation="h", marker_color=colors, showlegend=False, cliponaxis=False,
        text=[f"{v:+.2f} ({s:.0f}%)" for v, s in zip(b["Bias"], b["BiasShare"])], textposition="outside",
        textfont=dict(size=11), customdata=b["BiasShare"],
        hovertemplate="<b>%{y}</b><br>mean %{x:+.3f} mm<br>%{customdata:.1f}% of the systematic error<extra></extra>"),
        row=1, col=1)
    fig.add_trace(go.Bar(
        y=labels, x=b["SD"], orientation="h", marker_color=colors, showlegend=False, cliponaxis=False,
        text=[f"{v:.2f} ({s:.0f}%)" for v, s in zip(b["SD"], b["VarShare"])], textposition="outside",
        textfont=dict(size=11), customdata=b["VarShare"],
        hovertemplate="<b>%{y}</b><br>σ %{x:.3f} mm<br>%{customdata:.1f}% of the variance<extra></extra>"),
        row=1, col=2)
    bmax = float(np.nanmax(np.abs(b["Bias"]))) * 1.9 + 0.05
    fig.update_xaxes(range=[-bmax, bmax], zeroline=True, row=1, col=1)
    fig.update_xaxes(range=[0, float(np.nanmax(b["SD"])) * 1.6 + 0.02], row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    for fam in ("Placement", "Size", "Shape"):
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=P.family(fam), name=fam, showlegend=True))
    fig.update_layout(bargap=0.35)
    T.style(fig, height=height, title=title)
    fig.update_layout(legend=dict(y=-0.06, yanchor="top"), margin=dict(t=64, b=40))
    return fig


def mode_gallery(bud: pd.DataFrame, bat_type: str, nominals: Nominals, magnify: float = 120.0,
                 title: str | None = None, height: int = 540) -> go.Figure:
    """Eight small multiples: what each mode does to the module (nominal
    dashed, the mode at |mean| + 2σ solid, magnified)."""
    P = T.pal()
    b = bud[bud["BatteryType"] == bat_type].set_index("Mode")
    if b.empty:
        return T.empty("Not enough complete modules.")
    nom = nominals.get(bat_type)
    nx, ny = polygon(nom)
    titles = []
    for m in MODES:
        r = b.loc[m] if m in b.index else None
        sub = f"{r['VarShare']:.0f}% of scatter · mean {r['Bias']:+.2f} mm" if r is not None else ""
        titles.append(f"<b>{MODE_LABELS[m]}</b><br><span style='font-size:11px'>{sub}</span>")
    fig = make_subplots(rows=2, cols=4, subplot_titles=titles, horizontal_spacing=0.03, vertical_spacing=0.18)
    for i, m in enumerate(MODES):
        rr, cc = i // 4 + 1, i % 4 + 1
        fig.add_trace(go.Scatter(x=nx, y=ny, mode="lines", line=dict(color=P.ink["muted"], width=1.2, dash="dash"),
                                 hoverinfo="skip", showlegend=False), row=rr, col=cc)
        amp = 1.0
        sign = 1.0
        if m in b.index:
            amp = float(abs(b.loc[m, "Bias"]) + 2 * b.loc[m, "SD"]) or 1.0
            sign = 1.0 if b.loc[m, "Bias"] >= 0 else -1.0
        v = mode_vector(bat_type, m, nominals)
        pts = mode_shape_points(bat_type, nominals, v, sign * amp * magnify)
        xs, ys = polygon(pts)
        color = P.family(MODE_FAMILY[m])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color, width=2.5),
                                 marker=dict(size=7, color=color), fill="toself", fillcolor=T.rgba(color, 0.08),
                                 hovertemplate=f"<b>{MODE_LABELS[m]}</b><br>shown at |mean| + 2σ = {amp:.2f} mm "
                                               f"(×{magnify:g})<extra></extra>", showlegend=False), row=rr, col=cc)
        axis_id = "x" if i == 0 else f"x{i + 1}"
        fig.update_xaxes(visible=False, row=rr, col=cc)
        fig.update_yaxes(visible=False, autorange="reversed", scaleanchor=axis_id, scaleratio=1, row=rr, col=cc)
    for a in fig.layout.annotations:
        a.font = dict(size=12)
    fig.add_annotation(x=0.5, y=-0.05, xref="paper", yref="paper", showarrow=False,
                       text=f"Grey dashed: nominal · coloured: the mode at |mean| + 2σ, magnified ×{magnify:g} · front to the right",
                       font=dict(size=10, color=P.ink["muted"]))
    return T.style(fig, height=height, title=title, legend=False, margin=dict(t=72, b=30))


def mode_trends(wm: pd.DataFrame, bat_type: str, modes: list[str], title: str | None = None,
                height: int = 380) -> go.Figure:
    P = T.pal()
    d = wm[wm["BatteryType"] == bat_type] if len(wm) else wm
    if d is None or d.empty:
        return T.empty("No data.")
    fig = go.Figure()
    for i, m in enumerate(modes):
        line_color = P.series(i)
        mean, sd = d[f"A_{m}_mean"], d[f"A_{m}_sd"].fillna(0)
        fig.add_trace(go.Scatter(x=list(d["CalendarWeek"]) + list(d["CalendarWeek"])[::-1],
                                 y=list(mean + sd) + list(mean - sd)[::-1], fill="toself",
                                 fillcolor=T.rgba(line_color, 0.08), line=dict(width=0), hoverinfo="skip",
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=d["CalendarWeek"], y=mean, mode="lines+markers", name=MODE_LABELS[m],
                                 line=dict(color=line_color, width=2),
                                 marker=dict(size=8, color=line_color, line=dict(color=P.ink["surface"], width=2)),
                                 customdata=np.stack([sd, d["N"]], axis=1),
                                 hovertemplate="<b>%{x}</b> · " + MODE_LABELS[m]
                                               + "<br>mean %{y:+.3f} mm · σ %{customdata[0]:.3f}"
                                               + "<br>%{customdata[1]} modules<extra></extra>"))
    T.hline(fig, 0, P.ink["axis"], "solid", 1, None)
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text="Mode amplitude (mm at the corners)")
    return T.style(fig, height=height, title=title)


def pca_components(p: dict, bat_type: str, nominals: Nominals, magnify: float = 120.0, title: str | None = None,
                   height: int = 330) -> go.Figure:
    P = T.pal()
    if not p or not p.get("components"):
        return T.empty("Not enough complete modules.")
    comps = p["components"]
    nom = nominals.get(bat_type)
    nx, ny = polygon(nom)
    titles = [f"<b>PC{c['k']} · {c['share']:.0f}%</b><br><span style='font-size:11px'>{c['label']}</span>" for c in comps]
    fig = make_subplots(rows=1, cols=len(comps), subplot_titles=titles, horizontal_spacing=0.03)
    for i, c in enumerate(comps, start=1):
        fig.add_trace(go.Scatter(x=nx, y=ny, mode="lines", line=dict(color=P.ink["muted"], width=1.2, dash="dash"),
                                 hoverinfo="skip", showlegend=False), row=1, col=i)
        v = c["vector"] * 2 * c["sd"]
        pts = {k: (nom[k][0] + magnify * v[2 * j], nom[k][1] + magnify * v[2 * j + 1]) for j, k in enumerate(CORNERS)}
        xs, ys = polygon(pts)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=P.accent, width=2.5),
                                 marker=dict(size=7, color=P.accent), fill="toself", fillcolor=T.rgba(P.accent, 0.08),
                                 hovertemplate=f"PC{c['k']}: {c['share']:.1f}% of the variance<br>shown at 2σ = "
                                               f"{2 * c['sd']:.2f} mm<extra></extra>", showlegend=False), row=1, col=i)
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, autorange="reversed", scaleanchor="x" if i == 1 else f"x{i}", scaleratio=1,
                         row=1, col=i)
    for a in fig.layout.annotations:
        a.font = dict(size=12)
    return T.style(fig, height=height, title=title, legend=False, margin=dict(t=72))


def mode_link(link: pd.DataFrame, bat_type: str, title: str | None = None, height: int = 340) -> go.Figure:
    P = T.pal()
    d = link[link["BatteryType"] == bat_type] if len(link) else link
    if d is None or d.empty:
        return T.empty("Needs at least 40 complete modules.")
    d = d.set_index("Mode").reindex(list(MODES)).reset_index()
    labels = [MODE_LABELS[m] for m in d["Mode"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=d["FailLow"], orientation="h", name="when the mode is small (bottom 25%)",
                         marker_color=P.c["aqua"],
                         hovertemplate="<b>%{y}</b><br>NOK rate %{x:.1f}% when the mode is small<extra></extra>"))
    fig.add_trace(go.Bar(y=labels, x=d["FailHigh"], orientation="h", name="when the mode is large (top 25%)",
                         marker_color=P.c["orange"],
                         hovertemplate="<b>%{y}</b><br>NOK rate %{x:.1f}% when the mode is large<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="NOK rate", ticksuffix="%")
    return T.style(fig, height=height, title=title)
