"""Compensation, what-if and intervention figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import Nominals
from ..constants import CORNERS
from ..geometry import order_corners_convex, polygon
from . import theme as T

SYMBOLS = {"No compensation": "circle-open", "Median-based": "square", "Rigid fit (mean)": "diamond",
           "Robust optimum": "star", "Manual": "x"}


def landscape(land: dict, points: dict, title: str | None = None, height: int = 520,
              label: str = "Expected FPY") -> go.Figure:
    """FPY for every X/Y setting at the chosen yaw, with each method's setting."""
    P = T.pal()
    if not land:
        return T.empty("Not enough complete modules.")
    z = np.asarray(land["fpy"], float)
    zmax = float(np.nanmax(z))
    zlo = max(0.0, float(np.floor(zmax / 5) * 5) - 25)
    fig = go.Figure(go.Heatmap(
        x=land["x"], y=land["y"], z=z, colorscale=P.seq_light, zmin=zlo, zmax=float(np.ceil(zmax)),
        colorbar=dict(title=label, ticksuffix="%", len=0.85),
        hovertemplate="X %{x:+.2f} mm · Y %{y:+.2f} mm<br>" + label + " %{z:.1f}%<extra></extra>"))
    if zmax > 3:
        top = float(np.floor(zmax))
        fig.add_trace(go.Contour(x=land["x"], y=land["y"], z=z, showscale=False,
                                 contours=dict(coloring="none", start=top - 6, end=top - 1, size=2.5,
                                               showlabels=True, labelformat=".0f",
                                               labelfont=dict(size=10, color=P.ink["primary"])),
                                 line=dict(color=P.ink["primary"], width=1), hoverinfo="skip", showlegend=False))
    for name, setting in points.items():
        if setting is None:
            continue
        dx, dy, yaw = setting
        if not np.isfinite([dx, dy]).all():
            continue
        sym = SYMBOLS.get(name, "circle")
        fig.add_trace(go.Scatter(
            x=[dx], y=[dy], mode="markers", name=name,
            marker=dict(size=16 if name == "Robust optimum" else 12, color=P.method(name), symbol=sym,
                        line=dict(color=P.method(name) if "open" in sym else P.ink["surface"], width=2)),
            hovertemplate=f"<b>{name}</b><br>X {dx:+.2f} · Y {dy:+.2f} mm · yaw {yaw:+.3f}°<extra></extra>"))
    fig.update_xaxes(title_text="X compensation (mm)", range=[float(land["x"][0]), float(land["x"][-1])],
                     constrain="domain")
    fig.update_yaxes(title_text="Y compensation (mm)", range=[float(land["y"][0]), float(land["y"][-1])],
                     scaleanchor="x", scaleratio=1, constrain="domain")
    fig.add_annotation(x=1, y=1.0, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"at yaw {land['yaw']:+.3f}°", font=dict(size=11, color=P.ink["secondary"]))
    return T.style(fig, height=height, title=title)


def yaw_sweep(yaws, fpy, points: dict, title: str | None = None, height: int = 300,
              best: tuple | None = None) -> go.Figure:
    """FPY as only the yaw changes (X / Y fixed) and, optionally, the best FPY
    reachable at each yaw when X / Y are re-optimised."""
    P = T.pal()
    fpy = np.asarray(fpy, float)
    fig = go.Figure(go.Scatter(x=yaws, y=fpy, mode="lines", line=dict(color=P.ink["primary"], width=2),
                               name="Only yaw changes", hovertemplate="yaw %{x:+.3f}° → FPY %{y:.1f}%<extra></extra>"))
    top = float(np.nanmax(fpy)) if np.isfinite(fpy).any() else 100.0
    if best is not None and len(best[0]):
        by, bf = np.asarray(best[0], float), np.asarray(best[1], float)
        fig.add_trace(go.Scatter(x=by, y=bf, mode="lines", name="Best reachable (X/Y re-optimised)",
                                 line=dict(color=P.c["violet"], width=2, dash="dash"),
                                 hovertemplate="yaw %{x:+.3f}° → best FPY %{y:.1f}%<extra></extra>"))
        if np.isfinite(bf).any():
            top = max(top, float(np.nanmax(bf)))
    for name, yaw in points.items():
        if yaw is not None and np.isfinite(yaw):
            fig.add_trace(go.Scatter(x=[yaw, yaw], y=[0, top], mode="lines", name=name,
                                     line=dict(color=P.method(name), width=2, dash="dot"),
                                     hovertemplate=f"{name}: yaw {yaw:+.3f}°<extra></extra>"))
    lo = max(0.0, float(np.nanmin(fpy)) - 5) if np.isfinite(fpy).any() else 0.0
    fig.update_xaxes(title_text="Yaw compensation (°) at the selected X/Y")
    fig.update_yaxes(title_text="FPY", ticksuffix="%", range=[lo, min(100.0, top + 3)])
    return T.style(fig, height=height, title=title)


def method_bars(rows: pd.DataFrame, title: str | None = None, height: int = 300) -> go.Figure:
    """In-window vs backtest FPY per method."""
    P = T.pal()
    fig = go.Figure()
    for col, name, alpha in (("In window", "In window (optimistic)", 0.4), ("Backtest", "Backtest (honest)", 1.0)):
        if col not in rows:
            continue
        fig.add_trace(go.Bar(x=rows["Method"], y=rows[col], name=name,
                             marker_color=[T.rgba(P.method(m), alpha) for m in rows["Method"]],
                             marker_line=dict(color=[P.method(m) for m in rows["Method"]], width=1.5),
                             text=[f"{v:.1f}%" if np.isfinite(v) else "" for v in rows[col].astype(float)],
                             textposition="outside", cliponaxis=False,
                             hovertemplate="%{x}<br>" + name + ": %{y:.1f}%<extra></extra>"))
    vals = pd.concat([rows[c] for c in ("In window", "Backtest") if c in rows]).astype(float)
    lo = max(0.0, float(np.nanmin(vals)) - 8) if vals.notna().any() else 0.0
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    fig.update_yaxes(range=[lo, 104], ticksuffix="%", title_text="FPY")
    return T.style(fig, height=height, title=title)


def backtest_chart(bt: pd.DataFrame, methods: list[str], title: str | None = None, height: int = 320) -> go.Figure:
    P = T.pal()
    if bt is None or bt.empty:
        return T.empty("Needs earlier weeks with enough modules.")
    fig = go.Figure()
    for m in methods:
        if m not in bt:
            continue
        y = bt[m] / bt["Modules"] * 100
        fig.add_trace(go.Scatter(x=bt["Week"], y=y, mode="lines+markers", name=m,
                                 line=dict(color=P.method(m), width=2.5 if m == "Robust optimum" else 2,
                                           dash="dot" if m == "No compensation" else None),
                                 marker=dict(size=8, color=P.method(m), symbol=SYMBOLS.get(m, "circle"),
                                             line=dict(color=P.method(m), width=1.5)),
                                 hovertemplate="%{x} · " + m + ": %{y:.1f}%<extra></extra>"))
    fig.update_xaxes(type="category", title_text="Test week (setting computed from the weeks before it)")
    fig.update_yaxes(ticksuffix="%", title_text="FPY")
    return T.style(fig, height=height, title=title)


def before_after_centroids(sim: pd.DataFrame, title: str | None = None, height: int = 420) -> go.Figure:
    P = T.pal()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color=P.target, size=14, symbol="cross"),
                             name="Nominal center"))
    fig.add_trace(go.Scattergl(x=sim["Centroid_X"], y=sim["Centroid_Y"], mode="markers", name="Before",
                               marker=dict(size=6, color=T.rgba(P.ink["muted"], 0.5))))
    fig.add_trace(go.Scattergl(x=sim["Centroid_X_Sim"], y=sim["Centroid_Y_Sim"], mode="markers", name="After",
                               marker=dict(size=6, color=T.rgba(P.accent, 0.55))))
    fig.update_xaxes(title_text="Centroid X (mm)", zeroline=True)
    fig.update_yaxes(title_text="Centroid Y (mm)", zeroline=True, autorange="reversed", scaleanchor="x", scaleratio=1)
    return T.style(fig, height=height, title=title)


def before_after_yaw(sim: pd.DataFrame, title: str | None = None, height: int = 420) -> go.Figure:
    P = T.pal()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=sim["Rotation_Angle"], name="Before", marker_color=T.rgba(P.ink["muted"], 0.6), nbinsx=40))
    fig.add_trace(go.Histogram(x=sim["Rotation_Angle_Sim"], name="After", marker_color=T.rgba(P.accent, 0.6), nbinsx=40))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text="Yaw (°)")
    fig.update_yaxes(title_text="Modules")
    return T.style(fig, height=height, title=title)


def comp_vectors(rows: list[dict], title: str | None = None, height: int = 400) -> go.Figure:
    """Original view: observed bias (solid) vs applied offset (dashed), per type."""
    P = T.pal()
    fig = go.Figure()
    for r in rows:
        col = P.type(r["BatteryType"])
        fig.add_trace(go.Scatter(x=[0, r["bias_x"]], y=[0, r["bias_y"]], mode="lines+markers",
                                 name=f"Observed bias ({r['BatteryType']})", line=dict(color=col, width=3)))
        fig.add_trace(go.Scatter(x=[0, r["dx"]], y=[0, r["dy"]], mode="lines+markers",
                                 name=f"Applied offset ({r['BatteryType']})", line=dict(color=col, width=3, dash="dash")))
    fig.update_xaxes(title_text="X (mm)", zeroline=True)
    fig.update_yaxes(title_text="Y (mm)", zeroline=True, autorange="reversed", scaleanchor="x", scaleratio=1)
    return T.style(fig, height=height, title=title)


def weekly_real_vs_sim(sim: pd.DataFrame, title: str | None = None, height: int = 360) -> go.Figure:
    P = T.pal()
    if sim.empty:
        return T.empty("No data.")
    g = sim.assign(_r=sim["Status"] == "PASS", _s=sim["Status_Sim"] == "PASS").groupby(["WeekKey", "CalendarWeek"])
    w = (pd.DataFrame({"Real": g["_r"].mean() * 100, "Sim": g["_s"].mean() * 100, "N": g.size()})
         .reset_index().sort_values("WeekKey"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=w["CalendarWeek"], y=w["Real"], mode="lines+markers", name="Real FPY",
                             line=dict(color=P.ink["muted"], width=2), marker=dict(size=8, color=P.ink["muted"])))
    fig.add_trace(go.Scatter(x=w["CalendarWeek"], y=w["Sim"], mode="lines+markers", name="Simulated FPY",
                             line=dict(color=P.accent, width=2.5), marker=dict(size=8, color=P.accent)))
    fig.update_xaxes(type="category")
    T.thin_ticks(fig, w["CalendarWeek"].tolist(), max_ticks=8)
    lo = max(0.0, float(min(w["Real"].min(), w["Sim"].min())) - 8)
    fig.update_yaxes(range=[lo, 103], ticksuffix="%")
    return T.style(fig, height=height, title=title)


def outline_overlay(sim: pd.DataFrame, bat_type: str, nominals: Nominals, magnify: float = 50.0,
                    title: str | None = None, height: int = 440, limit: float | None = None) -> go.Figure:
    """Nominal vs measured (avg) vs compensated (avg), magnified (original
    view), with the tolerance zone of each corner at the same scale."""
    P = T.pal()
    nom = nominals.get(bat_type)
    d = sim[sim["BatteryType"] == bat_type]
    fig = go.Figure()
    span = min(max(p[0] for p in nom.values()) - min(p[0] for p in nom.values()),
               max(p[1] for p in nom.values()) - min(p[1] for p in nom.values()))
    if limit and limit * magnify <= 0.2 * span:  # at high magnification the zones would cover the module
        h = limit * magnify
        bx, by = [], []
        for c in CORNERS:
            x0, y0 = nom[c]
            bx += [x0 - h, x0 + h, x0 + h, x0 - h, x0 - h, None]
            by += [y0 - h, y0 - h, y0 + h, y0 + h, y0 - h, None]
        fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", name=f"Tolerance ±{limit:g} mm",
                                 line=dict(color=T.rgba(T.STATUS["warning"], 0.9), width=1.2, dash="dot"),
                                 hoverinfo="skip"))
    xs, ys = polygon(nom)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Nominal", line=dict(color=P.nominal, width=2, dash="dash"),
                             hoverinfo="skip"))
    if len(d):
        for label, sfx, color in (("Measured (avg)", "", P.ink["muted"]), ("Compensated (avg)", "_Sim", P.accent)):
            mean = {c: (float(d[f"{c}_X{sfx}"].mean()), float(d[f"{c}_Y{sfx}"].mean())) for c in CORNERS}
            if not np.isfinite(np.array(list(mean.values()))).all():
                continue
            pts = {c: (nom[c][0] + magnify * mx_, nom[c][1] + magnify * my_) for c, (mx_, my_) in mean.items()}
            order = list(order_corners_convex(pts))
            order = order + order[:1]           # closed outline
            fig.add_trace(go.Scatter(
                x=[pts[c][0] for c in order], y=[pts[c][1] for c in order], mode="lines+markers", name=label,
                line=dict(color=color, width=2.5), marker=dict(size=7, color=color), hoverinfo="text",
                hovertext=[f"<b>{c}</b> · {label}<br>X {mean[c][0]:+.2f} · Y {mean[c][1]:+.2f} mm" for c in order]))
    fig.update_xaxes(title_text="Global X (mm) — front →")
    fig.update_yaxes(title_text="Global Y (mm)", autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.add_annotation(x=1, y=0, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                       text=f"{bat_type} · deviations ×{magnify:g}", font=dict(size=11, color=P.ink["muted"]))
    return T.style(fig, height=height, title=title)


# ---------------------------------------------------------------------------
# What-if
# ---------------------------------------------------------------------------

def waterfall(wf: pd.DataFrame, target: float, title: str | None = None, height: int = 380) -> go.Figure:
    P = T.pal()
    if wf is None or wf.empty:
        return T.empty("No data.")
    measures = ["absolute"] + ["relative"] * (len(wf) - 1) + ["total"]
    x = [str(v).replace(" −", "<br>−") for v in wf["Step"]] + ["Scenario"]
    y = [wf["FPY"].iloc[0]] + list(wf["Delta"].iloc[1:]) + [wf["FPY"].iloc[-1]]
    text = [f"{wf['FPY'].iloc[0]:.1f}%"] + [f"{d:+.1f}" for d in wf["Delta"].iloc[1:]] + [f"{wf['FPY'].iloc[-1]:.1f}%"]
    fig = go.Figure(go.Waterfall(
        x=x, y=y, measure=measures, text=text, textposition="outside", cliponaxis=False,
        connector=dict(line=dict(color=P.ink["axis"], width=1)),
        increasing=dict(marker=dict(color=T.STATUS["good"])), decreasing=dict(marker=dict(color=T.STATUS["critical"])),
        totals=dict(marker=dict(color=P.accent)),
        hovertemplate="%{x}: %{text}<extra></extra>", showlegend=False))
    T.hline(fig, target, P.target, "dot", 2, f"Target {target:.0f}%", "top right")
    lo = max(0.0, float(wf["FPY"].min()) - 10)
    fig.update_yaxes(range=[lo, 104], ticksuffix="%", title_text="FPY")
    fig.update_xaxes(tickangle=0)
    return T.style(fig, height=height, title=title, legend=False)


def sensitivity(curves: dict, target: float, title: str | None = None, height: int = 330) -> go.Figure:
    """FPY vs spread reduction (0–100 %) for the lever families."""
    P = T.pal()
    fig = go.Figure()
    for i, (name, (xs, ys)) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(x=np.asarray(xs) * 100, y=ys, mode="lines+markers", name=name,
                                 line=dict(color=P.series(i), width=2.5), marker=dict(size=6, color=P.series(i)),
                                 hovertemplate=name + ": −%{x:.0f}% spread → %{y:.1f}%<extra></extra>"))
    T.hline(fig, target, P.target, "dot", 2, f"Target {target:.0f}%", "bottom right")
    fig.update_xaxes(title_text="Spread reduction", ticksuffix="%", range=[0, 100])
    fig.update_yaxes(title_text="Predicted FPY", ticksuffix="%")
    return T.style(fig, height=height, title=title)


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------

def event_timeline(roll: pd.DataFrame, events: pd.DataFrame, label: str, unit: str, highlight=None,
                   cps_dates: list | None = None, title: str | None = None, height: int = 380,
                   target: float | None = None) -> go.Figure:
    P = T.pal()
    if roll is None or roll.empty:
        return T.empty("No data.")
    fig = go.Figure()
    if unit != "%":
        fig.add_trace(go.Scattergl(x=roll["Date"], y=roll["Raw"], mode="markers", name="Modules",
                                   marker=dict(size=4, color=T.rgba(P.ink["muted"], 0.35)), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=roll["Date"], y=roll["Value"], mode="lines", name=f"Rolling {label}",
                             line=dict(color=P.accent, width=2.5),
                             hovertemplate="%{x|%b %d %H:%M}<br>" + label + " %{y:.2f}" + unit + "<extra></extra>"))
    if target is not None:
        T.hline(fig, target, P.target, "dot", 2, f"Target {target:.0f}%", "bottom right")
    if events is not None and len(events):
        T.event_marks(fig, pd.to_datetime(events["Date"]), events["Title"],
                      highlight=pd.Timestamp(highlight) if highlight is not None else None)
    for d in cps_dates or []:
        fig.add_vline(x=d, line=dict(color=P.ink["primary"], width=1, dash="dot"))
    fig.update_yaxes(title_text=f"{label} ({unit})" if unit != "%" else label, ticksuffix="%" if unit == "%" else None)
    return T.style(fig, height=height, title=title)


def before_after_dist(before, after, label: str, unit: str, title: str | None = None, height: int = 280) -> go.Figure:
    P = T.pal()
    fig = go.Figure()
    for vals, name, color in ((before, "Before", P.ink["muted"]), (after, "After", P.accent)):
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        if len(v) < 3:
            continue
        fig.add_trace(go.Violin(x=v, name=name, orientation="h", line_color=color, fillcolor=T.rgba(color, 0.2),
                                box_visible=True, meanline_visible=True, points=False, spanmode="hard",
                                hovertemplate=f"{name}: %{{x:+.3f}} {unit}<extra></extra>"))
    fig.update_xaxes(title_text=f"{label} ({unit})")
    return T.style(fig, height=height, title=title, legend=False)
