"""Hand-built SVG charts for the fix loop and the executive brief.

Static on purpose: they print exactly as they look, need no chart library in
the exported file, and are embedded as images with ``svg.as_img``. Colours
come from the shared palette (``theme.pal()``), so they follow the light and
dark theme; text always wears the ink colours, never a series colour.
"""

from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

from . import theme as T


def _esc(s) -> str:
    return html.escape(str(s), quote=True)


def _tw(s: str, size: float, bold: bool = False) -> float:
    """Approximate rendered width of a label (system sans)."""
    return len(str(s)) * size * (0.6 if bold else 0.55)


def _text(x, y, s, size=11.0, fill="#000", anchor="start", weight=None, extra="") -> str:
    w = f' font-weight="{weight}"' if weight else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:g}" fill="{fill}" text-anchor="{anchor}"{w}{extra}>'
            f'{_esc(s)}</text>')


def _halo() -> str:
    """Attributes that paint a surface-coloured outline under a label, so it
    stays readable where it crosses a line or a dot."""
    return f' paint-order="stroke" stroke="{T.pal().ink["surface"]}" stroke-width="4" stroke-linejoin="round"'


def _pair(x, y, strong: str, rest: str, anchor: str = "start", size: float = 13.0, small: float = 10.5) -> str:
    """A bold value followed by a quieter remark, laid out as one label."""
    P = T.pal()
    tail = (f'<tspan dx="5" font-size="{small:g}" font-weight="400" fill="{P.ink["secondary"]}">{_esc(rest)}</tspan>'
            if rest else "")
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size:g}" font-weight="700" '
            f'fill="{P.ink["primary"]}"{_halo()}>{_esc(strong)}{tail}</text>')


def _on(fill: str) -> str:
    """Near-black or white, whichever reads on ``fill``."""
    h = fill.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#0b0b0b" if lum > 0.36 else "#ffffff"


def _open(width: int, height: int, label: str) -> str:
    return (f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="{_esc(label)}" font-family="{T.FONT}">')


def _nice_step(span: float, max_ticks: int = 7) -> float:
    for step in (0.5, 1, 2, 5, 10, 20, 25, 50):
        if span / step <= max_ticks:
            return step
    return 50


def _bar(x0: float, x1: float, y: float, h: float, fill: str, r: float = 4.0) -> str:
    """Horizontal bar, square at its start and rounded at its data end."""
    w = x1 - x0
    if w <= 0.5:
        return ""
    r = min(r, w, h / 2)
    return (f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} V{y + h - r:.1f} '
            f'Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{fill}"/>')


# ---------------------------------------------------------------------------
# FPY by set-up
# ---------------------------------------------------------------------------

def setup_timeline_svg(setups: pd.DataFrame, daily: pd.DataFrame | None, changes: pd.DataFrame | None,
                       target: float, width: int = 1040, height: int = 340, min_day: int = 10) -> str:
    """The FPY of every set-up as a step with its 95 % interval, the daily FPY
    behind it, the logged changes as numbered marks and the target."""
    P = T.pal()
    sec, muted, grid = P.ink["secondary"], P.ink["muted"], P.ink["grid"]
    su = setups[setups["N"] > 0] if len(setups) else setups
    if su is None or not len(su):
        return ""
    L, R, TOP, BOT = 48, 18, 44, 62
    pw, ph = width - L - R, height - TOP - BOT
    t0 = pd.Timestamp(setups["Start"].min())
    t1 = pd.Timestamp(setups["End"].max())
    if changes is not None and len(changes):
        t1 = max(t1, pd.Timestamp(changes["Date"].max()))
    span = max((t1 - t0) / pd.Timedelta(hours=1), 1.0)
    pad = span * 0.012

    def X(t) -> float:
        return L + ((pd.Timestamp(t) - t0) / pd.Timedelta(hours=1) + pad) / (span + 2 * pad) * pw

    days = daily[daily["Total"] >= min_day] if daily is not None and len(daily) else None
    lows = [float(v) for v in su.loc[su["N"] >= 5, "CI_lo"]] + [float(su["FPY"].min()), float(target)]
    if days is not None and len(days):
        lows.append(float(np.percentile(days["PassRate"], 5)))
    lo = max(0.0, min(lows) - 1.5)
    step = _nice_step(100 - lo, 7)
    y_min = max(0.0, math.floor(lo / step) * step)
    if 100 - y_min < step * 2:
        y_min = max(0.0, 100 - step * 2)

    def Y(v) -> float:
        return TOP + (100.0 - float(v)) / (100.0 - y_min) * ph

    out = [_open(width, height, "First-pass yield of every set-up over time")]
    v = y_min
    while v <= 100 + 1e-9:
        out.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{width - R}" y2="{Y(v):.1f}" stroke="{grid}" stroke-width="1"/>')
        out.append(_text(L - 8, Y(v) + 4, f"{v:g}%", 11, muted, "end"))
        v += step
    # x ticks: Mondays
    monday = (t0 - pd.Timedelta(days=t0.dayofweek)).normalize()
    mondays = [monday + pd.Timedelta(weeks=i) for i in range(int(span // 168) + 3)]
    mondays = [m for m in mondays if t0 - pd.Timedelta(days=1) <= m <= t1]
    k = max(1, math.ceil(len(mondays) / 9))
    for m in mondays[::k]:
        x = X(m)
        out.append(f'<line x1="{x:.1f}" y1="{TOP + ph}" x2="{x:.1f}" y2="{TOP + ph + 4}" stroke="{P.ink["axis"]}" '
                   'stroke-width="1"/>')
        out.append(_text(x, TOP + ph + 18, f"{m:%b %d}", 11, muted, "middle"))
    # daily FPY, in the background
    if days is not None:
        for d, r in zip(days["Day"], days["PassRate"]):
            if r >= y_min:
                out.append(f'<circle cx="{X(pd.Timestamp(d) + pd.Timedelta(hours=12)):.1f}" cy="{Y(r):.1f}" r="2.6" '
                           f'fill="{muted}" opacity="0.5"/>')
    # target
    ty = Y(target)
    out.append(f'<line x1="{L}" y1="{ty:.1f}" x2="{width - R}" y2="{ty:.1f}" stroke="{P.target}" stroke-width="1.6" '
               'stroke-dasharray="7 4"/>')
    # set-ups
    acc = P.accent
    labels = []
    for _, s in su.iterrows():
        x0, x1 = X(s["Start"]) + 1.5, X(s["End"]) - 1.5
        if x1 - x0 < 3:
            x0, x1 = x0 - 1.5, x0 + 3
        y = Y(s["FPY"])
        if s["N"] >= 5:
            ya, yb = Y(min(100.0, s["CI_hi"])), Y(max(y_min, s["CI_lo"]))
            out.append(f'<rect x="{x0:.1f}" y="{ya:.1f}" width="{x1 - x0:.1f}" height="{max(1.0, yb - ya):.1f}" '
                       f'rx="3" fill="{T.rgba(acc, 0.16)}"/>')
        out.append(f'<line x1="{x0 + 1.5:.1f}" y1="{y:.1f}" x2="{x1 - 1.5:.1f}" y2="{y:.1f}" stroke="{acc}" '
                   'stroke-width="3" stroke-linecap="round"/>')
        seg = x1 - x0
        val, cnt = f"{s['FPY']:.1f}%", f"{int(s['N']):,} modules"
        if seg >= 44:
            full = seg >= _tw(val, 13, True) + _tw(cnt, 10.5) + 14
            top = Y(min(100.0, s["CI_hi"])) if s["N"] >= 5 else y
            ly = top - 7
            if ly < TOP + 11:                       # no room above: write under the interval
                ly = (Y(max(y_min, s["CI_lo"])) if s["N"] >= 5 else y) + 16
            labels.append(((x0 + x1) / 2, ly, val, cnt if full else ""))
    for cx, ly, val, cnt in labels:
        out.append(_pair(cx, ly, val, f"· {cnt}" if cnt else "", "middle"))
    # target label: right end, or the left end when the newest set-up's label is in the way
    tl = f"Target {target:g}%"
    right_busy = any(cx + 90 > width - R - _tw(tl, 11) and abs(ly - (ty - 6)) < 16 for cx, ly, _, _ in labels)
    if right_busy:
        out.append(_text(L + 6, ty - 6, tl, 11, sec, extra=_halo()))
    else:
        out.append(_text(width - R - 4, ty - 6, tl, 11, sec, "end", extra=_halo()))
    # logged changes
    if changes is not None and len(changes):
        last_x, row = -99.0, 0
        for _, e in changes.sort_values("Date").iterrows():
            x = X(e["Date"])
            if x < L - 1 or x > width - R + 1:
                continue
            row = (row + 1) % 2 if x - last_x < 22 else 0
            last_x = x
            cy = 14 + row * 19
            out.append(f'<line x1="{x:.1f}" y1="{cy + 9:.1f}" x2="{x:.1f}" y2="{TOP + ph}" stroke="{P.event}" '
                       'stroke-width="1.3" stroke-dasharray="3 3"/>')
            out.append(f'<circle cx="{x:.1f}" cy="{cy}" r="9.5" fill="{P.event}"/>')
            out.append(_text(x, cy + 4, str(int(e["No"])), 11, _on(P.event), "middle", 700))
    # legend
    ly = height - 12
    x = float(L)
    out.append(f'<line x1="{x}" y1="{ly - 4}" x2="{x + 22}" y2="{ly - 4}" stroke="{acc}" stroke-width="3" '
               'stroke-linecap="round"/>')
    out.append(_text(x + 28, ly, "FPY of the set-up", 11, sec))
    x += 28 + _tw("FPY of the set-up", 11) + 22
    out.append(f'<rect x="{x}" y="{ly - 10}" width="22" height="12" rx="3" fill="{T.rgba(acc, 0.16)}"/>')
    out.append(_text(x + 28, ly, "95% confidence interval", 11, sec))
    x += 28 + _tw("95% confidence interval", 11) + 22
    out.append(f'<circle cx="{x + 5}" cy="{ly - 4}" r="2.6" fill="{muted}" opacity="0.5"/>')
    out.append(_text(x + 14, ly, f"one production day (≥ {min_day} modules)", 11, sec))
    x += 14 + _tw(f"one production day (≥ {min_day} modules)", 11) + 22
    if changes is not None and len(changes):
        out.append(f'<circle cx="{x + 8}" cy="{ly - 4}" r="8" fill="{P.event}"/>')
        out.append(_text(x + 8, ly, "n", 10, _on(P.event), "middle", 700))
        out.append(_text(x + 22, ly, "logged change (number in the action log)", 11, sec))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Where the missing points go
# ---------------------------------------------------------------------------

def loss_budget_svg(budget: pd.DataFrame, target: float, width: int = 560) -> str:
    """One bar for the FPY points lost today, split by failure mechanism, over
    the loss the target allows; one row per mechanism with what removes it."""
    P = T.pal()
    ink, sec, muted = P.ink["primary"], P.ink["secondary"], P.ink["muted"]
    if budget is None or not len(budget):
        return ""
    total = float(budget["Loss"].sum())
    allowed = max(0.0, 100.0 - float(target))
    L, R = 4, 4
    pw = width - L - R
    scale = pw / max(total, allowed, 1e-9)
    row_h = 40
    height = 92 + row_h * len(budget) + 4
    out = [_open(width, height, "FPY points lost by failure mechanism")]
    out.append(_text(L, 15, f"Lost today: {total:.1f} points", 13, ink, weight=700))
    out.append(_text(width - R, 15, f"{int(budget['Modules'].sum()):,} NOK modules", 11, sec, "end"))
    # segmented bar
    y, h, x = 26, 26, float(L)
    for _, r in budget.iterrows():
        w = r["Loss"] * scale
        fill = P.mechanism(r["Mechanism"])
        if w > 3:
            out.append(f'<rect x="{x + 1:.1f}" y="{y}" width="{w - 2:.1f}" height="{h}" rx="4" fill="{fill}"/>')
            lab = f"{r['Loss']:.1f}"
            if w - 2 >= _tw(lab, 11, True) + 12:
                out.append(_text(x + 8, y + 17, lab, 11, _on(fill), weight=700))
        x += w
    # the loss the target allows
    ay = y + h + 9
    aw = allowed * scale
    out.append(f'<line x1="{L + 1}" y1="{ay}" x2="{L + max(aw, 2) - 1:.1f}" y2="{ay}" stroke="{P.target}" '
               'stroke-width="4" stroke-linecap="round"/>')
    out.append(_text(L, ay + 17, f"The {target:g}% target allows {allowed:.1f} points", 11, sec))
    # one row per mechanism
    y0 = 92
    for i, (_, r) in enumerate(budget.iterrows()):
        yy = y0 + i * row_h
        if i:
            out.append(f'<line x1="{L}" y1="{yy - 6}" x2="{width - R}" y2="{yy - 6}" stroke="{P.ink["grid"]}" '
                       'stroke-width="1"/>')
        out.append(f'<rect x="{L}" y="{yy + 1}" width="11" height="11" rx="3" fill="{P.mechanism(r["Mechanism"])}"/>')
        out.append(_text(L + 18, yy + 11, r["Mechanism"], 12.5, ink, weight=600))
        cnt = f"{int(r['Modules'])} module" + ("" if int(r["Modules"]) == 1 else "s")
        out.append(_text(width - R, yy + 11, f"{r['Loss']:.1f} points", 12.5, ink, "end", 700))
        out.append(_text(width - R - _tw(f"{r['Loss']:.1f} points", 12.5, True) - 8, yy + 11, cnt, 11, muted, "end"))
        out.append(_text(L + 18, yy + 26, T.short(r["Fix"], int((pw - 18) / 5.9)), 11, sec))
    out.append("</svg>")
    return "".join(out)


# ---------------------------------------------------------------------------
# What it takes to reach the target
# ---------------------------------------------------------------------------

def bridge_svg(bridge: pd.DataFrame, target: float, width: int = 560) -> str:
    """From today's FPY to the potential, lever by lever, on one FPY axis."""
    P = T.pal()
    ink, sec, muted, grid = P.ink["primary"], P.ink["secondary"], P.ink["muted"], P.ink["grid"]
    if bridge is None or not len(bridge):
        return ""
    start = float(bridge["FPY"].iloc[0])
    lo = min(start, float(target))
    step = _nice_step(100 - lo + 0.5, 6)
    x_min = max(0.0, math.floor((lo - step * 0.4) / step) * step)
    if 100 - x_min < step * 2:
        x_min = max(0.0, 100 - step * 2)
    LAB, R, TOP = 178, 64, 30
    row_h = 36
    pw = width - LAB - R
    height = TOP + row_h * len(bridge) + 30

    def X(v) -> float:
        return LAB + (float(v) - x_min) / (100.0 - x_min) * pw

    out = [_open(width, height, "What it takes to reach the FPY target")]
    bottom = TOP + row_h * len(bridge)
    v = x_min
    while v <= 100 + 1e-9:
        out.append(f'<line x1="{X(v):.1f}" y1="{TOP}" x2="{X(v):.1f}" y2="{bottom}" stroke="{grid}" stroke-width="1"/>')
        out.append(_text(X(v), bottom + 16, f"{v:g}%", 11, muted, "middle"))
        v += step
    tx = X(target)
    out.append(f'<line x1="{tx:.1f}" y1="{TOP - 6}" x2="{tx:.1f}" y2="{bottom}" stroke="{P.target}" stroke-width="1.6" '
               'stroke-dasharray="7 4"/>')
    anchor = "end" if tx > width - 70 else "middle"
    out.append(_text(min(tx, width - 4) if anchor == "end" else tx, TOP - 12, f"Target {target:g}%", 11, sec, anchor))
    prev = start
    for i, (_, r) in enumerate(bridge.iterrows()):
        cy = TOP + i * row_h + row_h / 2
        out.append(_text(4, cy + 4, r["Step"], 12.5, ink, weight=600 if i == 0 else None))
        level = float(r["FPY"])
        if i == 0:
            out.append(f'<circle cx="{X(level):.1f}" cy="{cy:.1f}" r="5.5" fill="{P.accent}"/>')
            out.append(_pair(X(level) + 11, cy + 4, f"{level:.1f}%", "", size=12.5))
        else:
            out.append(f'<line x1="{X(prev):.1f}" y1="{cy - row_h + 8:.1f}" x2="{X(prev):.1f}" y2="{cy + 8:.1f}" '
                       f'stroke="{P.ink["axis"]}" stroke-width="1"/>')
            if r["Delta"] >= 0.05:
                out.append(_bar(X(prev), X(level), cy - 8, 16, P.accent))
                lab, rest = f"+{r['Delta']:.1f}", f"→ {level:.1f}%"
                fits = X(level) + 7 + _tw(lab, 12.5, True) + _tw(rest, 11) + 8 <= width - 2
                if fits:
                    out.append(_pair(X(level) + 7, cy + 4, lab, rest, size=12.5, small=11))
                else:                               # no room after the bar: write before it
                    out.append(_pair(X(prev) - 8, cy + 4, lab, rest, "end", size=12.5, small=11))
            else:
                note = "no gain on these modules"
                fits = X(level) + 7 + _tw(note, 11) <= width - 2
                out.append(_text(X(level) + 7 if fits else X(level) - 8, cy + 4, note, 11, muted,
                                 "start" if fits else "end", extra=_halo()))
        prev = level
    out.append("</svg>")
    return "".join(out)
