"""Command-center figures: weekly FPY, status by type, failure mechanisms,
corner Pareto and the corner hotspot map."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import Nominals
from ..constants import CORNER_LABELS, CORNERS, MECHANISMS
from ..geometry import polygon
from . import theme as T


def weekly_fpy(weekly: pd.DataFrame, target: float, ma_window: int, show_incomplete: bool = False,
               events: pd.DataFrame | None = None, title: str | None = None, height: int = 440) -> go.Figure:
    """FPY per week (line + 95 % CI + moving average + target) above the
    weekly volume (stacked OK / NOK). Two panels, one shared week axis."""
    P = T.pal()
    if weekly is None or weekly.empty:
        return T.empty("No first runs in the current filters.")
    w = weekly.copy()
    x = w["CalendarWeek"].tolist()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(w["CI_hi"]) + list(w["CI_lo"])[::-1], fill="toself",
                             fillcolor=T.rgba(P.accent, 0.10), line=dict(width=0), hoverinfo="skip",
                             name="95% CI"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=w["MA_FPY"], mode="lines", name=f"MA{ma_window}",
                             line=dict(color=P.ma, width=2, dash="dash"),
                             hovertemplate=f"MA{ma_window}: %{{y:.1f}}%<extra></extra>"), row=1, col=1)
    low = w["LowSample"].to_numpy(bool)
    fig.add_trace(go.Scatter(
        x=x, y=w["PassRate"], mode="lines+markers", name="Weekly FPY",
        line=dict(color=P.accent, width=2.5),
        marker=dict(size=9, color=[P.ink["surface"] if lo else P.accent for lo in low],
                    line=dict(color=P.accent, width=2)),
        customdata=np.stack([w["Total"], w["Passed"], w["Failed"], w["CI_lo"], w["CI_hi"]], axis=1),
        hovertemplate=("<b>%{x}</b><br>FPY %{y:.1f}%  (CI %{customdata[3]:.0f}–%{customdata[4]:.0f}%)"
                       "<br>%{customdata[1]} OK · %{customdata[2]} NOK of %{customdata[0]}<extra></extra>")),
        row=1, col=1)
    fig.add_annotation(x=x[-1], y=float(w["PassRate"].iloc[-1]), text=f"<b>{w['PassRate'].iloc[-1]:.1f}%</b>",
                       showarrow=False, xanchor="left", xshift=8, font=dict(size=12), row=1, col=1)
    T.hline(fig, target, P.target, "dot", 2, f"Target {target:.0f}%", "top left", row=1, col=1)
    lo_y = max(0.0, min(float(np.nanmin(w["CI_lo"])), target) - 8)
    fig.update_yaxes(title_text="FPY", range=[lo_y, 103], row=1, col=1, ticksuffix="%")
    for name, col, color in (("OK", "Passed", P.status("PASS")), ("NOK", "Failed", P.status("FAIL")),
                             ("Incomplete", "Incomplete", P.status("INCOMPLETE"))):
        if name == "Incomplete" and (not show_incomplete or w[col].sum() == 0):
            continue
        fig.add_trace(go.Bar(x=x, y=w[col], name=f"{name} modules", marker_color=color,
                             marker_line=dict(width=1, color=P.ink["surface"]), opacity=0.85,
                             hovertemplate=f"%{{x}}: %{{y}} {name}<extra></extra>"), row=2, col=1)
    fig.update_layout(barmode="stack", bargap=0.35)
    fig.update_yaxes(title_text="Modules", row=2, col=1, rangemode="tozero")
    if events is not None and len(events) and "CalendarWeek" in events:
        for _, e in events.iterrows():
            if e["CalendarWeek"] in x:
                fig.add_vline(x=e["CalendarWeek"], line=dict(color=P.event, width=1.2, dash="dot"))
                fig.add_annotation(x=e["CalendarWeek"], y=1.0, yref="paper", text="◆ " + T.short(e["Title"], 24),
                                   showarrow=False, xanchor="left", yanchor="bottom",
                                   font=dict(size=10, color=P.ink["secondary"]))
    if low.any():
        fig.add_annotation(x=1, y=-0.02, xref="paper", yref="paper", xanchor="right", yanchor="top",
                           text="hollow points: weeks with fewer than 5 modules", showarrow=False,
                           font=dict(size=10, color=P.ink["muted"]))
    fig.update_xaxes(type="category", row=1, col=1)
    fig.update_xaxes(type="category", row=2, col=1)
    return T.style(fig, height=height, title=title, margin=dict(b=24))


def status_by_type(first: pd.DataFrame, title: str | None = None, height: int = 300) -> go.Figure:
    """Share of OK / incomplete / NOK per battery type (first runs)."""
    P = T.pal()
    if first.empty:
        return T.empty("No modules.")
    t = first.groupby(["BatteryType", "Status"]).size().unstack(fill_value=0)
    for s in ("PASS", "INCOMPLETE", "FAIL"):
        if s not in t:
            t[s] = 0
    tot = t.sum(axis=1)
    pct = t.div(tot, axis=0) * 100
    fig = go.Figure()
    for s, label in (("PASS", "OK"), ("INCOMPLETE", "Incomplete"), ("FAIL", "NOK")):
        if t[s].sum() == 0 and s == "INCOMPLETE":
            continue
        fig.add_trace(go.Bar(
            y=pct.index, x=pct[s], orientation="h", name=label, marker_color=P.status(s),
            marker_line=dict(width=1.5, color=P.ink["surface"]),
            text=[f"{p:.1f}%" if p >= 7 else "" for p in pct[s]], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#ffffff", size=12),
            customdata=np.stack([t[s], tot], axis=1),
            hovertemplate=f"<b>%{{y}}</b><br>{label}: %{{x:.1f}}% (%{{customdata[0]}} of %{{customdata[1]}})<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title_text="Share of first runs")
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=height, title=title)


def mechanisms(mech: pd.DataFrame, title: str | None = None, height: int = 300) -> go.Figure:
    """Why NOK modules fail: one 100 % bar per battery type, split by mechanism."""
    P = T.pal()
    if mech is None or mech.empty:
        return T.empty("No NOK modules — nothing to explain.")
    fig = go.Figure()
    types = sorted(mech["BatteryType"].unique())
    for m in MECHANISMS:
        sub = mech[mech["Mechanism"] == m].set_index("BatteryType").reindex(types)
        if sub["Modules"].fillna(0).sum() == 0:
            continue
        share = sub["Share"].fillna(0)
        fig.add_trace(go.Bar(
            y=types, x=share, orientation="h", name=m, marker_color=P.mechanism(m),
            marker_line=dict(width=1.5, color=P.ink["surface"]),
            text=[f"{v:.0f}%" if v >= 9 else "" for v in share], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#ffffff", size=12),
            customdata=sub["Modules"].fillna(0).astype(int),
            hovertemplate=f"<b>%{{y}}</b> · {m}<br>%{{x:.1f}}% of NOK (%{{customdata}} modules)<extra></extra>"))
    fig.update_layout(barmode="stack", bargap=0.45)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", title_text="Share of NOK first runs")
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=height, title=title)


def corner_pareto(pareto: pd.DataFrame, n_nok: int, title: str | None = None, height: int = 330) -> go.Figure:
    P = T.pal()
    if pareto is None or pareto.empty:
        return T.empty("No NOK modules.")
    d = pareto.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(
        y=d["Cause"], x=d["Share"], orientation="h", marker_color=P.accent,
        text=[f"{s:.0f}%  ({m})" for s, m in zip(d["Share"], d["Modules"])], textposition="outside",
        textfont=dict(size=11), cliponaxis=False,
        hovertemplate="<b>%{y}</b><br>out of tolerance in %{x:.1f}% of NOK modules<extra></extra>",
        showlegend=False))
    fig.update_xaxes(range=[0, min(100, float(d["Share"].max()) * 1.3 + 5)], ticksuffix="%",
                     title_text=f"Share of {n_nok} NOK modules (a module can count in several bars)")
    fig.update_layout(bargap=0.35)
    return T.style(fig, height=height, title=title, legend=False)


def hotspot_map(first: pd.DataFrame, limit: float, nominals: Nominals, types: list[str],
                exaggeration: float = 150.0, title: str | None = None, height: int = 360) -> go.Figure:
    """Where on the module do NOKs happen? Nominal outline per type; each
    corner's bubble is sized by its out-of-tolerance rate and an arrow shows
    its mean error (magnified)."""
    P = T.pal()
    types = [t for t in types if (first["BatteryType"] == t).any()]
    if not types:
        return T.empty("No modules.")
    fig = make_subplots(rows=1, cols=len(types), subplot_titles=types, horizontal_spacing=0.06)
    all_rates = []
    for t in types:
        sub = first[first["BatteryType"] == t]
        for c in CORNERS:
            dx, dy = sub[f"{c}_X"].to_numpy(float), sub[f"{c}_Y"].to_numpy(float)
            ok = np.isfinite(dx) & np.isfinite(dy)
            with np.errstate(invalid="ignore"):
                all_rates.append(((np.abs(dx) > limit) | (np.abs(dy) > limit))[ok].mean() * 100 if ok.any() else 0.0)
    rmax = max(1.0, max(all_rates) if all_rates else 1.0)
    for k, t in enumerate(types, start=1):
        sub = first[first["BatteryType"] == t]
        nom = nominals.get(t)
        xs, ys = polygon(nom)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=P.nominal, width=1.5, dash="dash"),
                                 hoverinfo="skip", showlegend=False), row=1, col=k)
        rates, labels, cx, cy, sizes, hov = [], [], [], [], [], []
        xref = "x" if k == 1 else f"x{k}"
        yref = "y" if k == 1 else f"y{k}"
        for c in CORNERS:
            dx, dy = sub[f"{c}_X"].to_numpy(float), sub[f"{c}_Y"].to_numpy(float)
            ok = np.isfinite(dx) & np.isfinite(dy)
            with np.errstate(invalid="ignore"):
                out = ((np.abs(dx) > limit) | (np.abs(dy) > limit)) & ok
            rate = out.sum() / max(1, ok.sum()) * 100
            mx = float(np.nanmean(dx)) if ok.any() else 0.0
            my = float(np.nanmean(dy)) if ok.any() else 0.0
            x0, y0 = nom[c]
            rates.append(rate)
            cx.append(x0)
            cy.append(y0)
            sizes.append(14 + 46 * rate / rmax)
            labels.append(f"<b>{c}</b> {rate:.1f}%")
            hov.append(f"<b>{CORNER_LABELS[c]}</b><br>out of tolerance: {rate:.1f}%<br>"
                       f"mean error X {mx:+.2f} · Y {my:+.2f} mm")
            fig.add_annotation(x=x0 + exaggeration * mx, y=y0 + exaggeration * my, ax=x0, ay=y0,
                               xref=xref, yref=yref, axref=xref, ayref=yref,
                               showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
                               arrowcolor=P.ink["secondary"], text="")
        fig.add_trace(go.Scatter(
            x=cx, y=cy, mode="markers+text", text=labels,
            textposition=["top right", "bottom right", "top left", "bottom left"],  # FL, FR, RL, RR on screen
            textfont=dict(size=12),
            marker=dict(size=sizes, color=[T.rgba(P.status("FAIL"), 0.15 + 0.6 * r / rmax) for r in rates],
                        line=dict(color=P.status("FAIL"), width=1.5)),
            hovertext=hov, hoverinfo="text", showlegend=False), row=1, col=k)
        fig.update_yaxes(autorange="reversed", scaleanchor=xref, scaleratio=1,
                         showgrid=False, zeroline=False, showticklabels=False, row=1, col=k)
        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=k,
                         range=[min(xs) - 420, max(xs) + 420])
    fig.add_annotation(x=0.5, y=-0.06, xref="paper", yref="paper", showarrow=False,
                       text=f"Bubble = share of modules out of tolerance at that corner · arrow = mean error ×{exaggeration:g}"
                            " · front of the module to the right",
                       font=dict(size=10, color=P.ink["muted"]))
    return T.style(fig, height=height, title=title, legend=False, margin=dict(b=30))
