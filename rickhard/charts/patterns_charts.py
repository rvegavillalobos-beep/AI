"""Time-pattern figures: shift, weekday, hour of day, start-up."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme as T


def fpy_by_group(t: pd.DataFrame, col: str, target: float | None = None, title: str | None = None,
                 height: int = 320, xlabel: str | None = None) -> go.Figure:
    P = T.pal()
    if t is None or t.empty:
        return T.empty("No data.")
    labels = t[col].astype(object).map(str).tolist()
    few = len(t) <= 4  # wide bars carry their value; narrow ones get it in the hover and n under the label
    fig = go.Figure(go.Bar(
        x=labels, y=t["FPY"], marker_color=P.accent,
        error_y=dict(type="data", symmetric=False, array=(t["CI_hi"] - t["FPY"]).clip(lower=0),
                     arrayminus=(t["FPY"] - t["CI_lo"]).clip(lower=0), color=P.ink["secondary"], thickness=1.2, width=4),
        text=[f"<b>{v:.1f}%</b><br>n={n}" for v, n in zip(t["FPY"], t["N"])] if few else None,
        textposition="inside", insidetextanchor="start", textangle=0, textfont=dict(color="#ffffff"),
        customdata=np.stack([t["N"], t["CI_lo"], t["CI_hi"]], axis=1),
        hovertemplate="<b>%{x}</b><br>FPY %{y:.1f}% (95% CI %{customdata[1]:.0f}–%{customdata[2]:.0f}%)<br>"
                      "%{customdata[0]} modules<extra></extra>", showlegend=False))
    if target is not None:
        T.hline(fig, target, P.target, "dot", 2, f"Target {target:.0f}%", "top right")
    lo = max(0.0, float(np.nanmin(t["CI_lo"])) - 10)
    fig.update_yaxes(range=[lo, 106], ticksuffix="%", title_text="FPY")
    fig.update_xaxes(type="category", title_text=xlabel or col)
    if not few:
        fig.update_xaxes(tickmode="array", tickvals=labels,
                         ticktext=[f"{lab}<br><span style='font-size:10px'>n={n}</span>" for lab, n in zip(labels, t["N"])])
    fig.update_layout(bargap=0.4)
    return T.style(fig, height=height, title=title, legend=False)


def placement_by_group(first: pd.DataFrame, col: str, order: list | None = None, title: str | None = None,
                       height: int = 320, xlabel: str | None = None) -> go.Figure:
    """Mean shift X / shift Y per group with 95 % CI."""
    P = T.pal()
    if first.empty or col not in first:
        return T.empty("No data.")
    groups = [g for g in (order or sorted(first[col].dropna().unique())) if (first[col] == g).any()]
    fig = go.Figure()
    for var, lab, a in (("Fit_Tx", "Shift X", "X"), ("Fit_Ty", "Shift Y", "Y")):
        g = first.groupby(col)[var]
        m, s, n = g.mean().reindex(groups), g.std().reindex(groups), g.count().reindex(groups)
        se = 1.96 * s / np.sqrt(n.clip(lower=1))
        fig.add_trace(go.Bar(x=[str(x) for x in groups], y=m, name=lab, marker_color=P.axis(a),
                             error_y=dict(type="data", array=se, color=P.ink["secondary"], thickness=1.2, width=4),
                             hovertemplate=f"<b>%{{x}}</b><br>{lab} %{{y:+.3f}} mm<extra></extra>"))
    T.hline(fig, 0, P.ink["axis"], "solid", 1, None)
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.1)
    fig.update_xaxes(type="category", title_text=xlabel or col)
    fig.update_yaxes(title_text="Mean placement (mm)")
    return T.style(fig, height=height, title=title)


def heatmap_week_hour(tab: pd.DataFrame, vol: pd.DataFrame, title: str | None = None, height: int = 330) -> go.Figure:
    P = T.pal()
    if tab is None or tab.empty:
        return T.empty("No data.")
    z = tab.to_numpy(float)
    v = vol.reindex(index=tab.index, columns=tab.columns).fillna(0).to_numpy(int)
    text = np.where(v > 0, v.astype(str), "")
    zmax = max(10.0, float(np.nanpercentile(z, 97))) if np.isfinite(z).any() else 10.0
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{h:02d}" for h in tab.columns], y=list(tab.index), colorscale=P.seq_light, zmin=0, zmax=zmax,
        text=text, texttemplate="%{text}", textfont=dict(size=9), xgap=2, ygap=2,
        colorbar=dict(title="NOK %", ticksuffix="%", len=0.9),
        customdata=v, hovertemplate="<b>%{y} %{x}:00</b><br>NOK rate %{z:.1f}%<br>%{customdata} modules<extra></extra>"))
    fig.update_xaxes(title_text="Hour of day (numbers: modules produced)", type="category")
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=height, title=title, legend=False)


def hourly_profile(prof: pd.DataFrame, label: str, unit: str, title: str | None = None, height: int = 300) -> go.Figure:
    P = T.pal()
    d = prof.dropna(subset=["Mean"]) if len(prof) else prof
    if d is None or d.empty:
        return T.empty("No data.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(d["Hour"]) + list(d["Hour"])[::-1], y=list(d["Hi"]) + list(d["Lo"])[::-1],
                             fill="toself", fillcolor=T.rgba(P.accent, 0.12), line=dict(width=0), hoverinfo="skip",
                             name="95% CI"))
    fig.add_trace(go.Scatter(x=d["Hour"], y=d["Mean"], mode="lines+markers", name=label,
                             line=dict(color=P.accent, width=2),
                             marker=dict(size=8, color=P.accent, line=dict(color=P.ink["surface"], width=2)),
                             customdata=d["N"],
                             hovertemplate="%{x:02d}:00 · %{y:+.3f} " + unit + " (n=%{customdata})<extra></extra>"))
    T.hline(fig, 0, P.ink["axis"], "solid", 1, None)
    fig.update_xaxes(title_text="Hour of day", tickmode="linear", dtick=2, range=[-0.5, 23.5])
    fig.update_yaxes(title_text=f"{label} ({unit})")
    return T.style(fig, height=height, title=title)


def startup_compare(first: pd.DataFrame, flags: np.ndarray, var: str, label: str, unit: str,
                    title: str | None = None, height: int = 320) -> go.Figure:
    P = T.pal()
    if first.empty or var not in first:
        return T.empty("No data.")
    v = first[var].to_numpy(float)
    fig = go.Figure()
    for m, name, color in ((~flags, "Steady running", P.c["blue"]), (flags, "First modules after a stop", P.c["orange"])):
        vals = v[m & np.isfinite(v)]
        if len(vals) < 3:
            continue
        fig.add_trace(go.Violin(y=vals, name=name, line_color=color, fillcolor=T.rgba(color, 0.18), box_visible=True,
                                meanline_visible=True, points=False, spanmode="hard",
                                hovertemplate=f"{name}<br>{label} %{{y:+.3f}} {unit}<extra></extra>"))
    fig.update_yaxes(title_text=f"{label} ({unit})")
    return T.style(fig, height=height, title=title, legend=False)
