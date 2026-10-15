"""Drift & yaw figures (the original vector-drift views plus dated shifts)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..spc import segment_means
from . import theme as T


def centroid_path(vec: pd.DataFrame, split_by_type: bool = True, title: str | None = None,
                  height: int = 520) -> go.Figure:
    """Weekly mean centroid offset (path) over the individual module centroids."""
    P = T.pal()
    if vec.empty:
        return T.empty("No modules.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text", marker=dict(color=P.target, size=14, symbol="cross"),
                             text=["nominal"], textposition="bottom left", name="Nominal center",
                             textfont=dict(color=P.ink["secondary"])))
    groups = ([(t, vec[vec["BatteryType"] == t]) for t in sorted(vec["BatteryType"].unique())]
              if split_by_type else [("All", vec)])
    for gi, (t, d) in enumerate(groups):
        color = P.type(t) if t != "All" else P.accent
        fig.add_trace(go.Scattergl(
            x=d["Centroid_X"], y=d["Centroid_Y"], mode="markers", name=f"Modules ({t})",
            marker=dict(size=4, color=T.rgba(color, 0.16)),
            customdata=np.stack([d["PartID"].astype(object).map(str), d["CalendarWeek"].astype(object).map(str)], axis=1),
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>X %{x:+.2f} · Y %{y:+.2f} mm<extra></extra>"))
        w = (d.groupby(["WeekKey", "CalendarWeek"])
             .agg(X=("Centroid_X", "mean"), Y=("Centroid_Y", "mean"), N=("PartID", "count"))
             .reset_index().sort_values("WeekKey"))
        fig.add_trace(go.Scatter(
            x=w["X"], y=w["Y"], mode="lines+markers+text", name=f"Weekly path ({t})",
            line=dict(color=color, width=2.5), marker=dict(size=10, color=color, line=dict(color=P.ink["surface"], width=2)),
            text=[lab if i == len(w) - 1 else "" for i, lab in enumerate(w["CalendarWeek"])],  # where it is now
            textposition="top right" if gi % 2 == 0 else "bottom right", textfont=dict(size=11, color=color),
            customdata=np.stack([w["N"], w["CalendarWeek"]], axis=1),
            hovertemplate="<b>%{customdata[1]}</b> (" + t + ")<br>mean X %{x:+.2f} · Y %{y:+.2f} mm<br>"
                          "%{customdata[0]} modules<extra></extra>"))
    fig.update_xaxes(title_text="Mean X deviation (mm)", zeroline=True)
    fig.update_yaxes(title_text="Mean Y deviation (mm)", zeroline=True, autorange="reversed", scaleanchor="x", scaleratio=1)
    return T.style(fig, height=height, title=title)


def magnitude_yaw(vec: pd.DataFrame, title: str | None = None, height: int = 520) -> go.Figure:
    """Mean drift magnitude per week (bars) and yaw per module (original)."""
    P = T.pal()
    if vec.empty:
        return T.empty("No modules.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    w = (vec.groupby(["WeekKey", "CalendarWeek"]).agg(R=("Vector_Magnitude", "mean"), N=("PartID", "count"))
         .reset_index().sort_values("WeekKey"))
    fig.add_trace(go.Bar(x=w["CalendarWeek"], y=w["R"], marker_color=P.accent, name="Mean R", customdata=w["N"],
                         hovertemplate="%{x}: R %{y:.2f} mm (%{customdata} modules)<extra></extra>", showlegend=False),
                  row=1, col=1)
    v = vec.sort_values("Date")
    for t in sorted(v["BatteryType"].unique()):
        s = v[v["BatteryType"] == t]
        fig.add_trace(go.Scatter(x=s["CalendarWeek"], y=s["Rotation_Angle"], mode="markers", name=t,
                                 marker=dict(size=6, color=T.rgba(P.type(t), 0.5)),
                                 customdata=np.stack([s["PartID"].astype(object).map(str),
                                                      s["Status"].astype(object).map(str)], axis=1),
                                 hovertemplate="%{x} · %{customdata[0]}<br>yaw %{y:+.3f}° · %{customdata[1]}<extra></extra>"),
                      row=2, col=1)
    T.hline(fig, 0, P.ink["axis"], "solid", 1, None, row=2, col=1)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=w["CalendarWeek"].tolist(), row=1, col=1)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=w["CalendarWeek"].tolist(),
                     title_text="Calendar week", row=2, col=1)
    T.thin_ticks(fig, w["CalendarWeek"].tolist(), max_ticks=8, row=2, col=1)
    fig.update_yaxes(title_text="Mean offset R (mm)", row=1, col=1)
    fig.update_yaxes(title_text="Module yaw (°)", row=2, col=1)
    fig.update_layout(bargap=0.4)
    return T.style(fig, height=height, title=title, legend_bottom=True)


def weekly_medians(vec: pd.DataFrame, title: str | None = None, height: int = 520) -> go.Figure:
    """Weekly median centroid X, Y (mm) and yaw (°) per battery type."""
    P = T.pal()
    if vec.empty:
        return T.empty("No modules.")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06)
    weeks = vec.drop_duplicates("WeekKey").sort_values("WeekKey")["CalendarWeek"].tolist()
    for t in sorted(vec["BatteryType"].unique()):
        g = vec[vec["BatteryType"] == t].groupby(["WeekKey", "CalendarWeek"])
        w = pd.DataFrame({"N": g.size(), "X": g["Centroid_X"].median(), "Y": g["Centroid_Y"].median(),
                          "Yaw": g["Rotation_Angle"].median()}).reset_index().sort_values("WeekKey")
        for row, (col, unit) in enumerate((("X", "mm"), ("Y", "mm"), ("Yaw", "°")), start=1):
            fig.add_trace(go.Scatter(
                x=w["CalendarWeek"], y=w[col], mode="lines+markers", name=t, legendgroup=t, showlegend=row == 1,
                line=dict(color=P.type(t), width=2), marker=dict(size=7, color=P.type(t)), customdata=w["N"],
                hovertemplate=f"<b>{t}</b> %{{x}} (n=%{{customdata}})<br>median {col} %{{y:+.3f}} {unit}<extra></extra>"),
                row=row, col=1)
    for row, label in enumerate(("Median centre X (mm)", "Median centre Y (mm)", "Median yaw (°)"), start=1):
        T.hline(fig, 0, P.ink["axis"], "solid", 1, None, row=row, col=1)
        fig.update_yaxes(title_text=label, row=row, col=1)
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=weeks, row=row, col=1)
    T.thin_ticks(fig, weeks, max_ticks=8, row=3, col=1)
    return T.style(fig, height=height, title=title, legend_bottom=True)


PLACEMENT_VARS = [("Fit_Tx", "Shift X", "mm"), ("Fit_Ty", "Shift Y", "mm"), ("Fit_Rot", "Yaw", "°")]


def placement_timeseries(first: pd.DataFrame, bat_type: str, cps: dict | None = None, events: pd.DataFrame | None = None,
                         window: int = 25, title: str | None = None, height: int = 620) -> go.Figure:
    """Least-squares placement of every module over time with a rolling
    median and the levels between detected change points."""
    P = T.pal()
    d = first[first["BatteryType"] == bat_type].sort_values("Date")
    if d.empty:
        return T.empty("No modules.")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05)
    color = P.type(bat_type)
    for i, (col, lab, unit) in enumerate(PLACEMENT_VARS, start=1):
        s = d[["Date", col, "PartID", "Status"]].dropna(subset=[col]).reset_index(drop=True)
        if s.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=s["Date"], y=s[col], mode="markers", name="Modules (red = NOK)", legendgroup="m", showlegend=i == 1,
            marker=dict(size=4, color=[P.status("FAIL") if st != "PASS" else T.rgba(color, 0.35) for st in s["Status"]]),
            customdata=np.stack([s["PartID"].astype(object).map(str), s["Status"].astype(object).map(str)], axis=1),
            hovertemplate=f"%{{customdata[0]}} · %{{customdata[1]}}<br>{lab} %{{y:+.3f}} {unit}<br>%{{x|%b %d %H:%M}}<extra></extra>"),
            row=i, col=1)
        roll = s[col].rolling(window, min_periods=max(5, window // 3), center=True).median()
        fig.add_trace(go.Scatter(x=s["Date"], y=roll, mode="lines", name=f"Rolling median ({window})",
                                 legendgroup="r", showlegend=i == 1, line=dict(color=color, width=2), hoverinfo="skip"),
                      row=i, col=1)
        if cps and cps.get(col):
            seg = segment_means(s[col].to_numpy(float), cps[col])
            fig.add_trace(go.Scatter(x=s["Date"], y=seg, mode="lines", name="Level between detected shifts",
                                     legendgroup="seg", showlegend=i == 1,
                                     line=dict(color=P.ink["primary"], width=2.5, shape="hv"),
                                     hovertemplate=f"level %{{y:+.3f}} {unit}<extra></extra>"), row=i, col=1)
            for kk in cps[col]:
                if kk < len(s):
                    fig.add_vline(x=s["Date"].iloc[kk], line=dict(color=P.ink["primary"], width=1, dash="dot"), row=i, col=1)
        T.hline(fig, 0, P.ink["axis"], "solid", 1, None, row=i, col=1)
        fig.update_yaxes(title_text=f"{lab} ({unit})", row=i, col=1)
    if events is not None and len(events):
        T.event_marks(fig, events["Date"], events["Title"], rows=[1, 2, 3])
    return T.style(fig, height=height, title=title, legend_bottom=True)
