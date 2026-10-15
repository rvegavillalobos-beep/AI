"""Control charts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..spc import RULES, ControlChart, segment_means
from . import theme as T


def imr_chart(cc: ControlChart, x, labels: list[str] | None, name: str, unit: str, cps: list[int] | None = None,
              spec: tuple[float, float] | None = None, events: pd.DataFrame | None = None, title: str | None = None,
              height: int = 560) -> go.Figure:
    """Individuals (top) and moving range (bottom) with σ zones and run-rule signals."""
    P = T.pal()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    s = float(cc.sigma[0])
    c = cc.center
    for k, alpha in ((3, 0.05), (2, 0.07), (1, 0.09)):
        fig.add_hrect(y0=c - k * s, y1=c + k * s, fillcolor=T.rgba(P.accent, alpha), line_width=0, row=1, col=1)
    flags = cc.any_flag
    txt = cc.flag_text()
    labels = labels or [""] * len(cc.x)
    hov = [f"{lab}<br>{name} {v:+.3f} {unit}" + (f"<br><b>{t}</b>" if t else "")
           for lab, v, t in zip(labels, cc.x, txt)]
    xs = np.asarray(x)
    fig.add_trace(go.Scattergl(x=xs, y=cc.x, mode="lines+markers", name=name,
                               line=dict(color=T.rgba(P.accent, 0.55), width=1),
                               marker=dict(size=5, color=P.accent), hovertext=hov, hoverinfo="text"), row=1, col=1)
    if flags.any():
        fig.add_trace(go.Scatter(x=xs[flags], y=cc.x[flags], mode="markers", name="Signal (run rule)",
                                 marker=dict(size=11, color=P.status("FAIL"), symbol="circle-open", line=dict(width=2.5)),
                                 hovertext=[h for h, f in zip(hov, flags) if f], hoverinfo="text"), row=1, col=1)
    if cps:
        seg = segment_means(cc.x, cps)
        fig.add_trace(go.Scatter(x=xs, y=seg, mode="lines", name="Level between detected shifts",
                                 line=dict(color=P.ink["primary"], width=2.5, shape="hv"), hoverinfo="skip"), row=1, col=1)
    T.hline(fig, c, P.ink["secondary"], "solid", 1.2, f"CL {c:+.3f}", "top left", row=1, col=1)
    T.hline(fig, c + 3 * s, P.limit, "dash", 1.2, f"UCL {c + 3 * s:+.3f}", "top left", row=1, col=1)
    T.hline(fig, c - 3 * s, P.limit, "dash", 1.2, f"LCL {c - 3 * s:+.3f}", "bottom left", row=1, col=1)
    if spec is not None:
        for v, lab in ((spec[0], "LSL"), (spec[1], "USL")):
            T.hline(fig, v, P.ink["primary"], "dot", 1.5, f"{lab} {v:+g}", "top right", row=1, col=1)
    if cc.mr is not None:
        fig.add_trace(go.Scattergl(x=xs, y=cc.mr, mode="lines", name="Moving range",
                                   line=dict(color=P.ink["muted"], width=1),
                                   hovertemplate="MR %{y:.3f}<extra></extra>", showlegend=False), row=2, col=1)
        if cc.mr_ucl and np.isfinite(cc.mr_ucl):
            T.hline(fig, cc.mr_ucl, P.limit, "dash", 1.2, f"UCL {cc.mr_ucl:.3f}", "top left", row=2, col=1)
    if events is not None and len(events):
        T.event_marks(fig, pd.to_datetime(events["Date"]), events["Title"], rows=[1, 2])
    fig.update_yaxes(title_text=f"{name} ({unit})", row=1, col=1)
    fig.update_yaxes(title_text=f"Moving range ({unit})", row=2, col=1, rangemode="tozero")
    return T.style(fig, height=height, title=title, legend_bottom=True)


def xbar_chart(cc: ControlChart, table: pd.DataFrame, name: str, unit: str, title: str | None = None,
               height: int = 380) -> go.Figure:
    P = T.pal()
    x = list(table["CalendarWeek"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(cc.ucl) + list(cc.lcl)[::-1], fill="toself",
                             fillcolor=T.rgba(P.accent, 0.08), line=dict(color=T.rgba(P.limit, 0.6), width=1, dash="dash"),
                             name="±3σ (varies with n)", hoverinfo="skip"))
    flags = cc.any_flag
    fig.add_trace(go.Scatter(x=x, y=cc.x, mode="lines+markers", name="Weekly mean",
                             line=dict(color=P.accent, width=2),
                             marker=dict(size=9, color=[P.status("FAIL") if f else P.accent for f in flags],
                                         line=dict(color=P.ink["surface"], width=2)),
                             customdata=np.stack([table["n"].astype(int), cc.flag_text()], axis=1),
                             hovertemplate="<b>%{x}</b><br>mean %{y:+.3f} " + unit
                                           + " (n=%{customdata[0]})<br>%{customdata[1]}<extra></extra>"))
    T.hline(fig, cc.center, P.ink["secondary"], "solid", 1.2, f"CL {cc.center:+.3f}", "top left")
    fig.update_xaxes(type="category")
    fig.update_yaxes(title_text=unit)
    return T.style(fig, height=height, title=title)


def rules_legend() -> str:
    return " · ".join(f"R{k}: {v}" for k, v in RULES.items() if k in (1, 2, 3, 5, 6))
