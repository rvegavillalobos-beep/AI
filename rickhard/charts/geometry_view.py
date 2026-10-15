"""Geometry figures: module outlines, the drift movie, the correction map and
the single-module drawing used by the passport."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..config import Nominals
from ..constants import CORNERS
from ..geometry import polygon
from . import theme as T
from .corners import ellipse


def _tol_boxes(fig, nom: dict, half: float, color, name="Tolerance zone"):
    xs, ys = [], []
    for c in CORNERS:
        cx, cy = nom[c]
        xs += [cx - half, cx + half, cx + half, cx - half, cx - half, None]
        ys += [cy - half, cy - half, cy + half, cy + half, cy - half, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.2, dash="dot"),
                             name=name, hoverinfo="skip", showlegend=False))


def _scale_note(text: str) -> dict:
    return dict(x=1, y=0, xref="paper", yref="paper", xanchor="right", yanchor="bottom", showarrow=False,
                text=text, font=dict(size=11, color=T.pal().ink["muted"]))


def outlines(df: pd.DataFrame, nominals: Nominals, limit: float, exaggeration: float = 1.0,
             focus_key: str | None = None, types: list[str] | None = None, title: str | None = None,
             height: int = 680, status_filter: tuple = ("PASS", "FAIL", "INCOMPLETE"),
             per_module: bool | None = None) -> go.Figure:
    """Every selected module drawn against the green nominal, deviations
    magnified, tolerance zones around each nominal corner (original view).

    With ``per_module`` (default: up to 60 modules) every module is its own
    legend entry, grouped by status: click an entry to hide it, double-click
    to show only that one."""
    P = T.pal()
    fig = go.Figure()
    if not types:
        types = sorted(df["BatteryType"].unique()) if len(df) else []
    for i, t in enumerate(types):
        nom = nominals.get(t)
        xs, ys = polygon(nom)
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"Nominal {t}", legendgroup="ref",
                                 legendgrouptitle_text="Reference" if i == 0 else None,
                                 line=dict(color=P.nominal, width=2, dash="dash"), hoverinfo="skip"))
        _tol_boxes(fig, nom, limit * exaggeration, T.rgba(T.STATUS["warning"], 0.9))
    groups = {"PASS": [], "FAIL": [], "INCOMPLETE": []}
    focus_row = None
    for _, r in df.iterrows():
        if focus_key and r.get("_mod_key") == focus_key:
            focus_row = r
            continue
        st = r["Status"] if r["Status"] in groups else "FAIL"
        if st not in status_filter:
            continue
        nom = nominals.get(r["BatteryType"])
        pts = {}
        for c in CORNERS:
            dx, dy = r[f"{c}_X"], r[f"{c}_Y"]
            if pd.notna(dx) and pd.notna(dy):
                pts[c] = (nom[c][0] + exaggeration * float(dx), nom[c][1] + exaggeration * float(dy))
        if len(pts) < len(CORNERS):  # a module with a missing corner has no outline to draw
            continue
        xs, ys = polygon(pts)
        groups[st].append((xs, ys, f"{r['PartID']} · Run {int(r['RunNum'])} · {r['Status']}", r["PartID"],
                           int(r["RunNum"])))
    style_ = {"PASS": (T.rgba(P.ink["muted"], 0.55), 1.0, "OK modules"),
              "INCOMPLETE": (T.rgba(T.STATUS["serious"], 0.85), 1.6, "Incomplete modules"),
              "FAIL": (T.rgba(P.status("FAIL"), 0.75), 1.6, "NOK modules")}
    n_mod = sum(len(v) for v in groups.values())
    if per_module is None:
        per_module = 0 < n_mod <= 60
    for st in ("FAIL", "INCOMPLETE", "PASS"):
        items = groups[st]
        if not items:
            continue
        color, width, name = style_[st]
        if per_module:
            for k, (xs, ys, lab, pid, run) in enumerate(items):
                fig.add_trace(go.Scatter(
                    x=xs, y=ys, mode="lines+markers", name=f"{pid} · R{run}", legendgroup=st,
                    legendgrouptitle_text=f"{name} ({len(items)})" if k == 0 else None,
                    line=dict(color=color, width=width), marker=dict(size=4, color=color),
                    hovertext=[lab] * len(xs), hoverinfo="text"))
            continue
        X, Y, txt = [], [], []
        for xs, ys, lab, _, _ in items:
            X += list(xs) + [None]
            Y += list(ys) + [None]
            txt += [lab] * len(xs) + [None]
        fig.add_trace(go.Scatter(x=X, y=Y, mode="lines+markers", name=f"{name} ({len(items)})",
                                 line=dict(color=color, width=width), marker=dict(size=4, color=color),
                                 hovertext=txt, hoverinfo="text"))
    if focus_row is not None:
        nom = nominals.get(focus_row["BatteryType"])
        pts = {c: (nom[c][0] + exaggeration * float(focus_row[f"{c}_X"]),
                   nom[c][1] + exaggeration * float(focus_row[f"{c}_Y"]))
               for c in CORNERS if pd.notna(focus_row[f"{c}_X"]) and pd.notna(focus_row[f"{c}_Y"])}
        if len(pts) == len(CORNERS):
            xs, ys = polygon(pts)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=f"★ {focus_row['PartID']} (focus)",
                                     line=dict(color=P.focus, width=4),
                                     marker=dict(size=9, color=P.focus, line=dict(color=P.ink["surface"], width=2)),
                                     hoverinfo="skip"))
            for c, (px, py) in pts.items():
                fig.add_annotation(x=px, y=py, showarrow=True, arrowhead=0, ax=0, ay=-26 if nom[c][1] < 0 else 26,
                                   text=f"<b>{c}</b> {focus_row[f'{c}_X']:+.2f} / {focus_row[f'{c}_Y']:+.2f}",
                                   font=dict(size=11, color=P.ink["primary"]), bgcolor=T.rgba(P.ink["surface"], 0.85))
    fig.update_xaxes(title_text="Global X (mm) — front →")
    fig.update_yaxes(title_text="Global Y (mm)", autorange="reversed", scaleanchor="x", scaleratio=1)
    T.style(fig, height=height, title=title)
    if per_module:
        # one entry per module with the full Part ID: a NOK column and an OK column under the axis title
        # (the nominal is named in the note instead). The figure grows by the longest column, so the
        # drawing keeps its size and the legend never needs a scrollbar.
        fig.update_traces(showlegend=False, selector=dict(legendgroup="ref"))
        names = [f"{it[3]} · R{it[4]}" for v in groups.values() for it in v]
        entry_w = int(min(380, 44 + 5.9 * max(len(n) for n in names)))
        tallest = max(len(v) for v in groups.values())
        leg_h = 34 + 19 * tallest
        fig.update_layout(height=height + leg_h, margin=dict(t=12, b=72 + leg_h),
                          legend=dict(orientation="h", x=0, xanchor="left", yref="container", y=0, yanchor="bottom",
                                      entrywidth=entry_w, maxheight=leg_h + 20, font=dict(size=10),
                                      groupclick="toggleitem", itemclick="toggle", itemdoubleclick="toggleothers",
                                      traceorder="grouped", tracegroupgap=6))
    if exaggeration != 1:
        fig.add_annotation(**_scale_note(f"deviations ×{exaggeration:g} · boxes = ±{limit:g} mm (scaled)"
                                         + (" · green dashed = nominal" if per_module else "")))
    return fig


def drift_movie(first: pd.DataFrame, nominals: Nominals, bat_type: str, limit: float,
                exaggeration: float = 40.0, title: str | None = None, height: int = 620) -> go.Figure:
    """Animated: the average module outline and the corner clouds, week by
    week (deviations magnified)."""
    P = T.pal()
    d = first[first["BatteryType"] == bat_type].dropna(subset=[f"{c}_{a}" for c in CORNERS for a in "XY"])
    weeks = d[["WeekKey", "CalendarWeek"]].drop_duplicates().sort_values("WeekKey")
    if len(weeks) < 2:
        return T.empty("Needs at least two weeks of complete modules.")
    nom = nominals.get(bat_type)
    nxs, nys = polygon(nom)
    k = exaggeration

    def frame_traces(wk):
        w = d[d["WeekKey"] == wk]
        mean_pts = {c: (nom[c][0] + k * w[f"{c}_X"].mean(), nom[c][1] + k * w[f"{c}_Y"].mean()) for c in CORNERS}
        mx, my = polygon(mean_pts)
        cx, cy = [], []
        for c in CORNERS:
            cx += list(nom[c][0] + k * w[f"{c}_X"].to_numpy(float))
            cy += list(nom[c][1] + k * w[f"{c}_Y"].to_numpy(float))
        fail = np.tile((w["Status"] != "PASS").to_numpy(), 4)
        return [
            go.Scatter(x=cx, y=cy, mode="markers",
                       marker=dict(size=5, color=[P.status("FAIL") if f_ else T.rgba(P.accent, 0.35) for f_ in fail]),
                       name="Corners (each module, red = NOK)", hoverinfo="skip"),
            go.Scatter(x=mx, y=my, mode="lines+markers", name="Average module",
                       line=dict(color=P.type(bat_type), width=3),
                       marker=dict(size=9, color=P.type(bat_type), line=dict(color=P.ink["surface"], width=2)),
                       hoverinfo="skip"),
        ]

    note = _scale_note(f"deviations ×{k:g}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nxs, y=nys, mode="lines", name="Nominal", line=dict(color=P.nominal, width=2, dash="dash"),
                             hoverinfo="skip"))
    _tol_boxes(fig, nom, limit * k, T.rgba(T.STATUS["warning"], 0.9))
    for tr in frame_traces(int(weeks["WeekKey"].iloc[0])):
        fig.add_trace(tr)
    frames, steps = [], []
    for wk, lab in zip(weeks["WeekKey"], weeks["CalendarWeek"]):
        w = d[d["WeekKey"] == wk]
        fpy = (w["Status"] == "PASS").mean() * 100
        head = dict(x=0.01, y=0.99, xref="paper", yref="paper", xanchor="left", yanchor="top", showarrow=False,
                    align="left", text=f"<b>{lab}</b> · {len(w)} modules · FPY {fpy:.0f}%", font=dict(size=14))
        frames.append(go.Frame(data=frame_traces(wk), traces=[2, 3], name=str(lab),
                               layout=go.Layout(annotations=[head, note])))
        steps.append(dict(method="animate", label=str(lab),
                          args=[[str(lab)], dict(mode="immediate", frame=dict(duration=600, redraw=True),
                                                 transition=dict(duration=300))]))
    fig.frames = frames
    xs_all = [nom[c][0] for c in CORNERS]
    ys_all = [nom[c][1] for c in CORNERS]
    pad = (limit + 1.5) * k
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0, y=-0.12, xanchor="left", yanchor="top",
                          direction="right", pad=dict(t=0, r=10),
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True,
                                                         transition=dict(duration=300))]),
                                   dict(label="❚❚ Pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, steps=steps, x=0, y=-0.21, len=1.0, xanchor="left", yanchor="top",
                      currentvalue=dict(visible=False), pad=dict(t=4, b=4))],
        annotations=list(frames[0].layout.annotations),
    )
    fig.update_xaxes(range=[min(xs_all) - pad, max(xs_all) + pad], title_text="Global X (mm) — front →")
    fig.update_yaxes(range=[max(ys_all) + pad, min(ys_all) - pad], scaleanchor="x", scaleratio=1,
                     title_text="Global Y (mm)")
    return T.style(fig, height=height, title=title, margin=dict(b=150))


def correction_map(first: pd.DataFrame, nominals: Nominals, bat_type: str, limit: float,
                   exaggeration: float = 60.0, title: str | None = None, height: int = 520) -> go.Figure:
    """Mean error of each corner (arrow) and its spread (2σ ellipse), magnified."""
    P = T.pal()
    d = first[first["BatteryType"] == bat_type]
    if d.empty:
        return T.empty("No modules.")
    nom = nominals.get(bat_type)
    k = exaggeration
    fig = go.Figure()
    xs, ys = polygon(nom)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Nominal", line=dict(color=P.nominal, width=2, dash="dash"),
                             hoverinfo="skip"))
    _tol_boxes(fig, nom, limit * k, T.rgba(T.STATUS["warning"], 0.9))
    mean_pts = {}
    for c in CORNERS:
        dx, dy = d[f"{c}_X"].to_numpy(float), d[f"{c}_Y"].to_numpy(float)
        if not (np.isfinite(dx).any() and np.isfinite(dy).any()):
            continue
        mx, my = float(np.nanmean(dx)), float(np.nanmean(dy))
        x0, y0 = nom[c]
        mean_pts[c] = (x0 + k * mx, y0 + k * my)
        ell = ellipse(dx, dy)
        if ell is not None:
            fig.add_trace(go.Scatter(x=x0 + k * ell[0], y=y0 + k * ell[1], mode="lines", fill="toself",
                                     fillcolor=T.rgba(P.type(bat_type), 0.10),
                                     line=dict(color=T.rgba(P.type(bat_type), 0.7), width=1.5),
                                     name="2σ spread", legendgroup="spread", showlegend=c == "FL", hoverinfo="skip"))
        fig.add_annotation(x=x0 + k * mx, y=y0 + k * my, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowsize=1.3, arrowwidth=2.5, arrowcolor=P.ink["primary"],
                           text="")
        fig.add_annotation(x=x0, y=y0, text=f"<b>{c}</b> {mx:+.2f} / {my:+.2f} mm", showarrow=False,
                           yshift=22 if y0 < 0 else -22, font=dict(size=12, color=P.ink["primary"]),
                           bgcolor=T.rgba(P.ink["surface"], 0.8))
    if len(mean_pts) >= 2:
        mx_, my_ = polygon(mean_pts)
        fig.add_trace(go.Scatter(x=mx_, y=my_, mode="lines", name="Average module",
                                 line=dict(color=P.type(bat_type), width=2.5), hoverinfo="skip"))
    fig.update_xaxes(title_text="Global X (mm) — front →")
    fig.update_yaxes(title_text="Global Y (mm)", autorange="reversed", scaleanchor="x", scaleratio=1)
    fig.add_annotation(**_scale_note(f"errors ×{k:g}"))
    return T.style(fig, height=height, title=title)


def module_drawing(row: pd.Series, nominals: Nominals, limit: float, exaggeration: float = 80.0,
                   title: str | None = None, height: int = 460) -> go.Figure:
    """One module: nominal outline, measured outline (magnified), arrows and values."""
    P = T.pal()
    nom = nominals.get(row["BatteryType"])
    k = exaggeration
    fig = go.Figure()
    xs, ys = polygon(nom)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Nominal", line=dict(color=P.nominal, width=2, dash="dash"),
                             hoverinfo="skip"))
    _tol_boxes(fig, nom, limit * k, T.rgba(T.STATUS["warning"], 0.9))
    pts = {}
    for c in CORNERS:
        dx, dy = row.get(f"{c}_X"), row.get(f"{c}_Y")
        x0, y0 = nom[c]
        if pd.isna(dx) or pd.isna(dy):
            fig.add_annotation(x=x0, y=y0, text=f"<b>{c}</b> missing", showarrow=False,
                               font=dict(color=P.status("FAIL"), size=12))
            continue
        out = abs(dx) > limit or abs(dy) > limit
        pts[c] = (x0 + k * dx, y0 + k * dy)
        fig.add_annotation(x=pts[c][0], y=pts[c][1], ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowwidth=2.5, text="",
                           arrowcolor=P.status("FAIL") if out else P.ink["primary"])
        fig.add_annotation(x=x0, y=y0, text=f"<b>{c}</b>  X {dx:+.2f} · Y {dy:+.2f}", showarrow=False,
                           yshift=24 if y0 < 0 else -24,
                           font=dict(size=12, color=P.status("FAIL") if out else P.ink["primary"]),
                           bgcolor=T.rgba(P.ink["surface"], 0.85))
    if len(pts) >= 2:
        mx, my = polygon(pts)
        color = P.status("FAIL") if row.get("Status") != "PASS" else P.status("PASS")
        fig.add_trace(go.Scatter(x=mx, y=my, mode="lines+markers", name="Measured", line=dict(color=color, width=3),
                                 marker=dict(size=9, color=color, line=dict(color=P.ink["surface"], width=2)),
                                 hoverinfo="skip"))
    fig.update_xaxes(title_text="Global X (mm) — front →", showgrid=False)
    fig.update_yaxes(title_text="Global Y (mm)", autorange="reversed", scaleanchor="x", scaleratio=1, showgrid=False)
    fig.add_annotation(**_scale_note(f"deviations ×{k:g} · boxes = ±{limit:g} mm (scaled)"))
    return T.style(fig, height=height, title=title)
