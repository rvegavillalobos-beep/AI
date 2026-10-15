"""Corner Lab figures: target plots, capability, trends, correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as _st

from ..constants import CORNER_LABELS, CORNERS, DEV_COLS
from . import theme as T

# Spatial layout used everywhere: front of the module to the right, left side
# (negative Y) at the top — the same view as the geometry plots.
SPATIAL = [["RL", "FL"], ["RR", "FR"]]


def ellipse(x, y, n_std: float = 2.0, n: int = 60):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return None
    cov = np.cov(x[m], y[m])
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    t = np.linspace(0, 2 * np.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])
    pts = vecs @ (np.sqrt(vals)[:, None] * n_std * circle)
    return pts[0] + x[m].mean(), pts[1] + y[m].mean()


def target_plots(df: pd.DataFrame, limit: float, color_by: str = "Status", title: str | None = None,
                 height: int = 660, max_points: int = 4000) -> go.Figure:
    """Four corner 'targets' laid out like the module: every module's corner
    deviation (X, Y), the tolerance box, the mean error and a 2σ ellipse."""
    P = T.pal()
    if df.empty:
        return T.empty("No modules.")
    d = df.sample(max_points, random_state=0) if len(df) > max_points else df
    vals = d[list(DEV_COLS)].to_numpy(float)
    span = limit + 0.8
    if np.isfinite(vals).any():
        span = max(span, float(np.nanpercentile(np.abs(vals), 99.5)) * 1.08)
    titles = [f"{CORNER_LABELS[c]} ({c})" for row in SPATIAL for c in row]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, horizontal_spacing=0.07, vertical_spacing=0.1)
    legend_done = set()
    status = d["Status"].astype(str).to_numpy() if "Status" in d else np.array(["PASS"] * len(d))
    near = d["NearMiss"].to_numpy(bool) if "NearMiss" in d else np.zeros(len(d), bool)
    cd = np.stack([d["PartID"].astype(object).map(str), pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d %H:%M"),
                   status], axis=1)
    for r, row in enumerate(SPATIAL, start=1):
        for k, c in enumerate(row, start=1):
            bx = [-limit, limit, limit, -limit, -limit]
            by = [-limit, -limit, limit, limit, -limit]
            fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", line=dict(color=P.limit, width=1.3, dash="dash"),
                                     hoverinfo="skip", showlegend=False), row=r, col=k)
            x, y = d[f"{c}_X"].to_numpy(float), d[f"{c}_Y"].to_numpy(float)
            hover = (f"<b>{c}</b> X %{{x:+.2f}} · Y %{{y:+.2f}} mm<br>%{{customdata[0]}}<br>%{{customdata[1]}} · "
                     "%{customdata[2]}<extra></extra>")

            def add(mask, name, color, size, op, group=None):
                if not mask.any():
                    return
                g = group or name
                fig.add_trace(go.Scatter(
                    x=x[mask], y=y[mask], mode="markers", name=name, legendgroup=g,
                    showlegend=g not in legend_done,
                    marker=dict(size=size, color=color, opacity=op, line=dict(width=0)),
                    customdata=cd[mask], hovertemplate=hover), row=r, col=k)
                legend_done.add(g)

            if color_by == "Status":
                add((status == "PASS") & ~near, "OK", P.ink["muted"], 5, 0.35)
                add(near, "Near miss", P.status("NEAR MISS"), 6, 0.85)
                add(status != "PASS", "NOK", P.status("FAIL"), 7, 0.9)
            elif color_by == "Battery type":
                for t in sorted(d["BatteryType"].unique()):
                    add((d["BatteryType"] == t).to_numpy(), t, P.type(t), 6, 0.55)
            elif color_by == "Shift" and "Shift" in d:
                for i, s in enumerate(sorted(d["Shift"].dropna().unique())):
                    add((d["Shift"] == s).to_numpy(), f"Shift {s}", P.series(i), 6, 0.55)
            else:  # by week (sequential)
                wk = d["WeekKey"].astype(float).to_numpy()
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="markers", name="Week", showlegend=False,
                    marker=dict(size=6, color=wk, colorscale=P.seq, opacity=0.75,
                                showscale=(r == 1 and k == 2),
                                colorbar=dict(title="Week", tickvals=[np.nanmin(wk), np.nanmax(wk)],
                                              ticktext=[d["CalendarWeek"].iloc[int(np.nanargmin(wk))],
                                                        d["CalendarWeek"].iloc[int(np.nanargmax(wk))]],
                                              len=0.5, y=0.78)),
                    customdata=cd, hovertemplate=hover), row=r, col=k)
            ell = ellipse(x, y)
            if ell is not None:
                fig.add_trace(go.Scatter(x=ell[0], y=ell[1], mode="lines", name="2σ ellipse", legendgroup="ell",
                                         showlegend="ell" not in legend_done,
                                         line=dict(color=P.accent, width=2), hoverinfo="skip"), row=r, col=k)
                legend_done.add("ell")
            if np.isfinite(x).any() and np.isfinite(y).any():
                mx, my = float(np.nanmean(x)), float(np.nanmean(y))
                fig.add_trace(go.Scatter(x=[0, mx], y=[0, my], mode="lines+markers", name="Mean error",
                                         legendgroup="mean", showlegend="mean" not in legend_done,
                                         line=dict(color=P.ink["primary"], width=2),
                                         marker=dict(size=[0, 10], color=P.ink["primary"],
                                                     line=dict(color=P.ink["surface"], width=2)),
                                         hovertemplate=f"<b>{c} mean</b> X {mx:+.2f} · Y {my:+.2f} mm<extra></extra>"),
                              row=r, col=k)
                legend_done.add("mean")
                fig.add_annotation(x=mx, y=my, text=f"<b>{mx:+.2f}, {my:+.2f}</b>", showarrow=False, yshift=-16,
                                   font=dict(size=11, color=P.ink["primary"]), bgcolor=T.rgba(P.ink["surface"], 0.85),
                                   row=r, col=k)
            anchor = "x" if (r, k) == (1, 1) else f"x{(r - 1) * 2 + k}"
            fig.update_xaxes(range=[-span, span], zeroline=True, title_text="dX (mm)" if r == 2 else None,
                             row=r, col=k)
            fig.update_yaxes(range=[span, -span], zeroline=True, scaleanchor=anchor, scaleratio=1,
                             title_text="dY (mm)" if k == 1 else None, row=r, col=k)
    return T.style(fig, height=height, title=title, legend_bottom=True)


def capability_grid(cap: pd.DataFrame, title: str | None = None, height: int = 380) -> go.Figure:
    """Ppk per corner/axis and battery type, coloured by verdict."""
    P = T.pal()
    c = cap[cap["Axis"].isin(["X", "Y"])].copy() if len(cap) else cap
    if c is None or c.empty:
        return T.empty("No capability data.")
    types = [t for t in ("Type S", "Type M") if (c["BatteryType"] == t).any()]
    rows = [f"{k} {a}" for k in CORNERS for a in ("X", "Y")]
    z, txt = [], []
    for rr in rows:
        k, a = rr.split()
        zr, tr = [], []
        for t in types:
            m = c[(c["BatteryType"] == t) & (c["Corner"] == k) & (c["Axis"] == a)]
            if m.empty or not np.isfinite(m["Ppk"].iloc[0]):
                zr.append(np.nan)
                tr.append("")
            else:
                v = float(m["Ppk"].iloc[0])
                zr.append(v)
                tr.append(f"<b>{v:.2f}</b>  {m['Diagnosis'].iloc[0]}")
        z.append(zr)
        txt.append(tr)
    zz = np.array(z, float)
    zc = np.select([zz < 0.67, zz < 1.0, zz < 1.33], [0.12, 0.37, 0.62], default=0.88)
    zc = np.where(np.isnan(zz), np.nan, zc)
    scale = [[0.0, T.STATUS["critical"]], [0.249, T.STATUS["critical"]], [0.25, T.STATUS["serious"]],
             [0.499, T.STATUS["serious"]], [0.5, T.STATUS["warning"]], [0.749, T.STATUS["warning"]],
             [0.75, T.STATUS["good"]], [1.0, T.STATUS["good"]]]
    fig = go.Figure(go.Heatmap(
        z=zc, x=types, y=rows, text=txt, texttemplate="%{text}", colorscale=scale, zmin=0, zmax=1,
        showscale=False, xgap=3, ygap=3, customdata=zz,
        hovertemplate="<b>%{x} · %{y}</b><br>Ppk %{customdata:.2f}<extra></extra>",
        textfont=dict(color="#111111", size=12)))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(side="top")
    fig.add_annotation(x=0.5, y=0, yshift=-8, xref="paper", yref="paper", showarrow=False, yanchor="top",
                       text="Ppk < 0.67 critical · < 1.00 poor<br>< 1.33 marginal · ≥ 1.33 capable",
                       font=dict(size=10, color=P.ink["muted"]))
    return T.style(fig, height=height, title=title, legend=False, margin=dict(b=44, t=30))


def capability_hist(values, lo: float, hi: float, name: str, stats: dict | None = None,
                    title: str | None = None, height: int = 340, color: str | None = None) -> go.Figure:
    P = T.pal()
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return T.empty("Not enough values.")
    color = color or P.accent
    nb = min(60, max(15, int(np.sqrt(len(v)) * 1.5)))
    fig = go.Figure(go.Histogram(x=v, nbinsx=nb, name=name, marker_color=T.rgba(color, 0.55),
                                 marker_line=dict(width=1, color=P.ink["surface"]),
                                 hovertemplate="%{x}: %{y} modules<extra></extra>", showlegend=False))
    mu, sd = float(v.mean()), float(v.std(ddof=1))
    xs = np.linspace(min(v.min(), lo) - 0.5, max(v.max(), hi) + 0.5, 200)
    _, edges = np.histogram(v, bins=nb)
    bw = edges[1] - edges[0]
    fig.add_trace(go.Scatter(x=xs, y=_st.norm.pdf(xs, mu, sd) * len(v) * bw, mode="lines", name="Normal fit",
                             line=dict(color=color, width=2), hoverinfo="skip", showlegend=False))
    for x, lab in ((lo, "LSL"), (hi, "USL")):
        fig.add_vline(x=x, line=dict(color=P.limit, width=1.5, dash="dash"),
                      annotation_text=f"{lab} {x:+g}", annotation_font=dict(size=10, color=P.ink["secondary"]))
    fig.add_vline(x=mu, line=dict(color=P.ink["primary"], width=1.5),
                  annotation_text=f"mean {mu:+.2f}", annotation_position="top left",
                  annotation_font=dict(size=10, color=P.ink["secondary"]))
    if stats:
        txt = (f"N {stats.get('N', len(v))} · σ {stats.get('Std', sd):.2f} · Cp {stats.get('Cp', np.nan):.2f} · "
               f"Cpk {stats.get('Cpk', np.nan):.2f} · Pp {stats.get('Pp', np.nan):.2f} · "
               f"Ppk {stats.get('Ppk', np.nan):.2f} · {stats.get('PPM', np.nan):,.0f} ppm expected")
        fig.add_annotation(x=0, y=1.0, xref="paper", yref="paper", xanchor="left", yanchor="bottom", showarrow=False,
                           text=txt, font=dict(size=11, color=P.ink["secondary"]))
    fig.update_xaxes(title_text=f"{name} (mm)")
    fig.update_yaxes(title_text="Modules")
    fig.update_layout(bargap=0.04)
    return T.style(fig, height=height, title=title, legend=False, margin=dict(t=56))


def corner_trends(df: pd.DataFrame, limit: float, bat_type: str, title: str | None = None,
                  height: int = 560, window: int = 15) -> go.Figure:
    """Original corner trend matrix: X and Y deviation per corner in
    chronological order, with a rolling mean and the tolerance."""
    P = T.pal()
    d = df[df["BatteryType"] == bat_type].sort_values("Date").reset_index(drop=True)
    if d.empty:
        return T.empty(f"No {bat_type} modules.")
    order = [["FL", "FR"], ["RL", "RR"]]
    titles = [f"{CORNER_LABELS[c]} ({c})" for row in order for c in row]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles, shared_xaxes=True, horizontal_spacing=0.06,
                        vertical_spacing=0.12)
    seq = np.arange(1, len(d) + 1)
    cd = np.stack([d["PartID"].astype(object).map(str), pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d %H:%M"),
                   d["CalendarWeek"].astype(object).map(str)], axis=1)
    shown = set()
    for r, row in enumerate(order, start=1):
        for k, c in enumerate(row, start=1):
            for a in ("X", "Y"):
                y = d[f"{c}_{a}"].to_numpy(float)
                fig.add_trace(go.Scattergl(
                    x=seq, y=y, mode="markers", name=f"{a} deviation", legendgroup=a,
                    showlegend=a not in shown, marker=dict(size=4, color=P.axis(a), opacity=0.45),
                    customdata=cd,
                    hovertemplate=(f"<b>Module #%{{x}}</b> · {c} {a} %{{y:+.2f}} mm<br>%{{customdata[0]}}"
                                   "<br>%{customdata[1]} · %{customdata[2]}<extra></extra>")), row=r, col=k)
                roll = pd.Series(y).rolling(window, min_periods=max(3, window // 3), center=True).mean()
                fig.add_trace(go.Scatter(x=seq, y=roll, mode="lines", name=f"{a} rolling mean ({window})",
                                         legendgroup=f"{a}r", showlegend=a not in shown,
                                         line=dict(color=P.axis(a), width=2), hoverinfo="skip"), row=r, col=k)
                shown.add(a)
            T.limit_lines(fig, limit, row=r, col=k, label=(k == 2))
            fig.update_yaxes(range=[-limit - 2.5, limit + 2.5], title_text="mm" if k == 1 else None, row=r, col=k)
            if r == 2:
                fig.update_xaxes(title_text="Module # (chronological)", row=r, col=k)
    return T.style(fig, height=height, title=title, legend_bottom=True)


def correlation(df: pd.DataFrame, bat_type: str, title: str | None = None, height: int = 430) -> go.Figure:
    P = T.pal()
    d = df[df["BatteryType"] == bat_type][list(DEV_COLS)].dropna()
    if len(d) < 10:
        return T.empty("Not enough complete modules.")
    cmat = d.corr().to_numpy()
    labels = [c.replace("_", " ") for c in DEV_COLS]
    fig = go.Figure(go.Heatmap(z=cmat, x=labels, y=labels, zmin=-1, zmax=1, colorscale=P.diverging,
                               text=np.round(cmat, 2), texttemplate="%{text:.2f}", xgap=2, ygap=2,
                               colorbar=dict(title="r", len=0.8),
                               hovertemplate="%{y} vs %{x}: r = %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    return T.style(fig, height=height, title=title, legend=False)
