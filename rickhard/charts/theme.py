"""Colour palette and layout helpers shared by every chart.

The categorical palette is the validated reference instance of the data-viz
method (fixed hue order, CVD-checked), with separate steps for light and dark
surfaces. Status colours are reserved for status (OK / NOK / near miss) and
never used for a series. Call ``set_dark(True)`` before building charts for a
dark surface.
"""

from __future__ import annotations

import math
import threading

import numpy as np
import plotly.graph_objects as go

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"

_CAT = {
    "light": {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
              "magenta": "#e87ba4", "green": "#008300", "violet": "#6250d6", "red": "#e34948"},
    "dark": {"blue": "#3987e5", "orange": "#d95926", "aqua": "#199e70", "yellow": "#c98500",
             "magenta": "#d55181", "green": "#008300", "violet": "#9085e9", "red": "#e66767"},
}
_INK = {
    "light": {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "axis": "#c3c2b7", "surface": "#fcfcfb", "neutral": "#f0efec"},
    "dark": {"primary": "#f0efec", "secondary": "#c3c2b7", "muted": "#898781", "grid": "#2c2c2a",
             "axis": "#383835", "surface": "#1a1a19", "neutral": "#383835"},
}
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


class Palette:
    def __init__(self, dark: bool = False):
        self.dark = dark
        self.mode = "dark" if dark else "light"
        self.c = _CAT[self.mode]
        self.ink = _INK[self.mode]

    # identity -----------------------------------------------------------------
    def type(self, t: str) -> str:
        return {"Type S": self.c["blue"], "Type M": self.c["orange"]}.get(t, self.ink["muted"])

    def axis(self, a: str) -> str:
        return {"X": self.c["violet"], "Y": self.c["aqua"], "Z": self.c["orange"]}.get(a, self.ink["muted"])

    def mechanism(self, m: str) -> str:
        return {"Shift": self.c["blue"], "Yaw": self.c["orange"], "Shift + yaw": self.c["violet"],
                "Shape": self.c["magenta"], "Placement + shape": self.c["yellow"], "Height (Z)": self.c["aqua"],
                "Missing corner": self.ink["muted"]}.get(m, self.ink["muted"])

    def method(self, m: str) -> str:
        return {"No compensation": self.ink["muted"], "Median-based": self.c["blue"],
                "Rigid fit (mean)": self.c["orange"], "Robust optimum": self.c["violet"],
                "Manual": self.c["magenta"]}.get(m, self.ink["muted"])

    def family(self, f: str) -> str:
        return {"Placement": self.c["blue"], "Size": self.c["aqua"], "Shape": self.c["orange"]}.get(f, self.ink["muted"])

    def sq_cause(self, c: str) -> str:
        return {"Parallelogram Tilt": self.c["yellow"], "Trapezoidal Width": self.c["aqua"],
                "Trapezoidal Length": self.c["violet"], "Combined Asymmetry": self.c["magenta"]}.get(c, self.ink["muted"])

    def nok_cause(self, c: str) -> str:
        return {"Placement": self.c["blue"], "Shape + placement": self.c["yellow"], "Shape": self.c["magenta"],
                "Both": self.c["violet"], "Height (Z)": self.c["aqua"]}.get(c, self.ink["muted"])

    def series(self, i: int) -> str:
        order = ["blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red"]
        return self.c[order[i % len(order)]]

    # state --------------------------------------------------------------------
    def status(self, s: str) -> str:
        return {"PASS": STATUS["good"], "OK": STATUS["good"], "SQUARE OK": STATUS["good"],
                "FAIL": STATUS["critical"], "NOK": STATUS["critical"], "DEFORMED": STATUS["critical"],
                "NEAR MISS": STATUS["warning"], "INCOMPLETE": self.ink["muted"]}.get(s, self.ink["muted"])

    # references -----------------------------------------------------------------
    @property
    def limit(self) -> str:
        return STATUS["critical"]

    @property
    def target(self) -> str:
        return STATUS["good"]

    @property
    def nominal(self) -> str:
        return self.c["green"]

    @property
    def ma(self) -> str:
        return self.c["yellow"]

    @property
    def focus(self) -> str:
        return self.c["violet"]

    @property
    def event(self) -> str:
        return self.c["magenta"]

    @property
    def accent(self) -> str:
        return self.c["blue"]

    @property
    def seq(self) -> list:
        steps = SEQ_BLUE if not self.dark else SEQ_BLUE[::-1]
        return [[i / (len(steps) - 1), s] for i, s in enumerate(steps)]

    @property
    def seq_light(self) -> list:
        """Sequential blue whose low end recedes into the surface (heatmaps)."""
        if self.dark:  # dark surface: low values dark, high values bright
            steps = [self.ink["surface"]] + SEQ_BLUE[2:-1][::-1]
        else:
            steps = [self.ink["surface"]] + SEQ_BLUE[2:]
        return [[i / (len(steps) - 1), s] for i, s in enumerate(steps)]

    @property
    def diverging(self) -> list:
        return [[0.0, self.c["blue"]], [0.5, self.ink["neutral"]], [1.0, self.c["red"]]]


_LIGHT, _DARK = Palette(False), Palette(True)
_state = threading.local()  # one palette per thread: concurrent sessions may use different themes


def set_dark(dark: bool) -> None:
    _state.palette = _DARK if dark else _LIGHT


def pal() -> Palette:
    return getattr(_state, "palette", _LIGHT)


def rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _top_label_shift(fig: go.Figure) -> float | None:
    """Largest y-shift of labels sitting on top of the plot area (subplot
    titles, event labels), or None when there are none."""
    shift = None
    for a in fig.layout.annotations or ():
        if (a.yref == "paper" and a.y is not None and a.y >= 0.97 and a.yanchor == "bottom"
                and a.showarrow is False):
            shift = max(shift or 0.0, float(a.yshift or 0.0))
    return shift


def style(fig: go.Figure, height: int | None = None, title: str | None = None, legend: bool = True,
          margin: dict | None = None, hovermode: str | None = None, legend_bottom: bool = False) -> go.Figure:
    top_shift = _top_label_shift(fig)
    if top_shift is not None and legend:
        legend_bottom = True  # a legend on top would sit on the labels
    m = dict(l=8, r=8, t=44 if (title or (legend and not legend_bottom)) else 12, b=8)
    if top_shift is not None:
        m["t"] = max(m["t"], (58 if title else 30) + int(top_shift))
    if margin:
        m.update(margin)
    fig.update_layout(
        font=dict(family=FONT, size=12),
        margin=m,
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="",
                    font=dict(size=11), itemsizing="constant", traceorder="normal"),
        hoverlabel=dict(font=dict(family=FONT, size=12), namelength=-1),
        barcornerradius=4,
    )
    if height:
        fig.update_layout(height=height)
    if hovermode:
        fig.update_layout(hovermode=hovermode)
    if title:
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xanchor="left", y=0.98, yanchor="top",
                                     font=dict(size=14)))
        if legend and not legend_bottom:
            fig.update_layout(margin=dict(t=max(m["t"], 64)), legend=dict(y=1.02))
    if legend and legend_bottom:
        fig.update_layout(legend=dict(y=-0.08, yanchor="top"), margin=dict(b=max(m["b"], 64)))
    return fig


def report_style(fig: go.Figure) -> go.Figure:
    """Explicit light styling for figures exported outside Streamlit."""
    ink = _INK["light"]
    fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                      font=dict(color=ink["primary"], family=FONT))
    fig.update_xaxes(gridcolor=ink["grid"], linecolor=ink["axis"], zerolinecolor=ink["axis"])
    fig.update_yaxes(gridcolor=ink["grid"], linecolor=ink["axis"], zerolinecolor=ink["axis"])
    return fig


def empty(message: str, height: int = 260) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=pal().ink["muted"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=height, legend=False)


def hline(fig, y, color, dash="dash", width=1.5, text=None, pos="top right", row=None, col=None):
    kw = {"row": row, "col": col} if row is not None else {}
    if text:
        P = pal()
        kw.update(annotation_text=text, annotation_position=pos,
                  annotation_font=dict(size=10, color=P.ink["secondary"]),
                  annotation_bgcolor=rgba(P.ink["surface"], 0.8))
    fig.add_hline(y=y, line=dict(color=color, dash=dash, width=width), **kw)


def limit_lines(fig, limit: float, row=None, col=None, label=True):
    P = pal()
    hline(fig, limit, P.limit, "dash", 1.2, f"+{limit:g}" if label else None, "top right", row, col)
    hline(fig, -limit, P.limit, "dash", 1.2, f"−{limit:g}" if label else None, "bottom right", row, col)


def event_marks(fig, xs, titles, highlight=None, rows=None, max_chars: int = 22, x_ref: str = "x") -> None:
    """Dashed vertical lines for logged interventions, each with a short label
    above the plot area. Neighbouring labels alternate between two heights so
    they do not run into each other; hovering a label shows the full title."""
    P = pal()
    items = sorted(zip(list(xs), [str(t) for t in titles]), key=lambda it: it[0])
    for i, (x, title) in enumerate(items):
        hl = highlight is not None and x == highlight
        line = dict(color=P.event, width=2.5 if hl else 1.3, dash="solid" if hl else "dash")
        if rows:
            for r in rows:
                fig.add_vline(x=x, line=line, row=r, col=1)
        else:
            fig.add_vline(x=x, line=line)
        text = f"◆ {short(title, max_chars)}"
        fig.add_annotation(x=x, xref=x_ref, y=1.0, yref="paper", yanchor="bottom", xanchor="left", xshift=-4,
                           yshift=14 * (i % 2), showarrow=False, text=f"<b>{text}</b>" if hl else text,
                           hovertext=title, font=dict(size=10, color=P.ink["secondary"]),
                           bgcolor=rgba(P.ink["surface"], 0.85))


def thin_ticks(fig, labels, max_ticks: int = 8, row=None, col=None) -> None:
    """Show every k-th category label, horizontally, so they never overlap."""
    labels = list(labels)
    if not labels:
        return
    k = max(1, math.ceil(len(labels) / max_ticks))
    kw = {"row": row, "col": col} if row is not None else {}
    fig.update_xaxes(tickmode="array", tickvals=labels[::k], ticktext=labels[::k], tickangle=0, **kw)


def short(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def nice_range(values, pad: float = 0.1, min_span: float = 1.0):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return [-1, 1]
    lo, hi = float(v.min()), float(v.max())
    span = max(hi - lo, min_span)
    return [lo - pad * span, hi + pad * span]


def sparkline_svg(values, width: int = 120, height: int = 32, color: str | None = None,
                  target: float | None = None) -> str:
    """A tiny inline SVG trend line (stat tiles)."""
    v = []
    for x in values:
        try:
            fx = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fx):
            v.append(fx)
    if len(v) < 2:
        return ""
    P = pal()
    color = color or P.accent
    lo = min(v + ([target] if target is not None else []))
    hi = max(v + ([target] if target is not None else []))
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    pad = 3

    def xy(i, val):
        x = pad + i * (width - 2 * pad) / (len(v) - 1)
        y = height - pad - (val - lo) / (hi - lo) * (height - 2 * pad)
        return x, y

    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in (xy(i, val) for i, val in enumerate(v)))
    lx, ly = xy(len(v) - 1, v[-1])
    tgt = ""
    if target is not None:
        _, ty = xy(0, target)
        tgt = (f'<line x1="{pad}" y1="{ty:.1f}" x2="{width - pad}" y2="{ty:.1f}" stroke="{STATUS["good"]}" '
               f'stroke-width="1" stroke-dasharray="3 3" opacity="0.9"/>')
    return (f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="trend">{tgt}'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" '
            f'stroke-linecap="round" opacity="0.9"/><circle cx="{lx:.1f}" cy="{ly:.1f}" r="3.5" fill="{color}" '
            f'stroke="{P.ink["surface"]}" stroke-width="1.5"/></svg>')
