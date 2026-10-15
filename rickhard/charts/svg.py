"""Hand-built SVG: the compensation work instruction and the app logo.

Arrowheads are drawn as polygons (no <marker> elements). HTML sanitisers
(Streamlit's ``st.html`` among them) drop inline ``<svg>``, so pages embed
these drawings as images with :func:`as_img`.
"""

from __future__ import annotations

import base64
import html
import math

import numpy as np

from ..config import Nominals
from ..constants import CORNERS
from ..geometry import rel_nominal, rigid_transform
from . import theme as T


def as_img(svg: str, alt: str = "", style: str = "") -> str:
    """An ``<img>`` showing the SVG from a data URI."""
    if not svg:
        return ""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return (f'<img src="data:image/svg+xml;base64,{b64}" alt="{html.escape(alt, quote=True)}" '
            f'style="{html.escape(style, quote=True)}">')


def _arrow(x1, y1, x2, y2, color, width=2.5, head=9.0) -> str:
    dx, dy = x2 - x1, y2 - y1
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return ""
    head = min(head, L * 0.6)
    ux, uy = dx / L, dy / L
    hx, hy = x2 - ux * head, y2 - uy * head
    px, py = -uy * head * 0.55, ux * head * 0.55
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{hx:.1f}" y2="{hy:.1f}" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round"/>'
            f'<polygon points="{x2:.1f},{y2:.1f} {hx + px:.1f},{hy + py:.1f} {hx - px:.1f},{hy - py:.1f}" fill="{color}"/>')


MAGNIFY_STEPS = (1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300)


def auto_magnify(largest_mm: float, span_mm: float, share: float = 0.15) -> int:
    """A round magnification that draws the largest deviation at about
    ``share`` of the module size: big enough to see, small enough to fit."""
    if not np.isfinite(largest_mm) or largest_mm <= 1e-9 or not np.isfinite(span_mm) or span_mm <= 0:
        return 150
    k = share * span_mm / largest_mm
    return max([m for m in MAGNIFY_STEPS if m <= k], default=MAGNIFY_STEPS[0])


def correction_offsets(bat_type: str, dx: float, dy: float, yaw: float, nominals: Nominals):
    """Per-corner offsets (mm) of a rigid X / Y / yaw correction."""
    rx, ry = rel_nominal(bat_type, nominals)
    cx, cy = rigid_transform(rx, ry, 0.0, 0.0, dx, dy, yaw)
    return cx - rx, cy - ry


def work_instruction_svg(bat_type: str, dx: float, dy: float, yaw: float, nominals: Nominals,
                         magnify: float | None = None, width: int = 640, height: int = 520) -> str:
    """Top view of the module: where each corner must go (magnified).

    ``magnify=None`` picks a round factor from the size of the correction;
    whatever the factor, the drawing is scaled to stay inside its frame."""
    P = T.pal()
    rx, ry = rel_nominal(bat_type, nominals)
    ox, oy = correction_offsets(bat_type, dx, dy, yaw, nominals)   # per-corner offsets (mm)
    span_x = float(rx.max() - rx.min())
    span_y = float(ry.max() - ry.min())
    if magnify is None:
        magnify = auto_magnify(float(np.max(np.hypot(ox, oy))), min(span_x, span_y))
    k = float(magnify)
    # fit the nominal outline, the moved outline and the centre arrow into the frame
    xs = np.r_[rx, rx + k * ox, 0.0, k * dx]
    ys = np.r_[ry, ry + k * oy, 0.0, k * dy]
    pad_x, pad_y = 48, 78
    s = min((width - 2 * pad_x) / max(float(xs.max() - xs.min()), 1e-6),
            (height - 2 * pad_y) / max(float(ys.max() - ys.min()), 1e-6))
    mx = width / 2 - s * float(xs.max() + xs.min()) / 2
    my = height / 2 - s * float(ys.max() + ys.min()) / 2

    def to_px(x, y):  # front (larger X) to the right, Y downwards (same as the plots)
        return mx + x * s, my + y * s

    ink, muted, acc = P.ink["primary"], P.ink["muted"], P.c["violet"]
    order = [0, 1, 3, 2]  # FL, FR, RR, RL around the outline
    nom_pts = " ".join("{:.1f},{:.1f}".format(*to_px(rx[i], ry[i])) for i in order)
    new_pts = " ".join("{:.1f},{:.1f}".format(*to_px(rx[i] + k * ox[i], ry[i] + k * oy[i])) for i in order)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
        f'role="img" aria-label="Compensation work instruction" font-family="{T.FONT}">',
        f'<polygon points="{nom_pts}" fill="{T.rgba(muted, 0.06)}" stroke="{muted}" stroke-width="1.5" '
        f'stroke-dasharray="6 5"/>',
        f'<polygon points="{new_pts}" fill="{T.rgba(acc, 0.08)}" stroke="{acc}" stroke-width="2.5"/>',
    ]
    c0 = to_px(0, 0)
    c1 = to_px(k * dx, k * dy)
    parts.append(f'<circle cx="{c0[0]:.1f}" cy="{c0[1]:.1f}" r="4" fill="{muted}"/>')
    parts.append(_arrow(c0[0], c0[1], c1[0], c1[1], acc, 3, 11))
    label_y = c0[1] + max(38.0, c1[1] - c0[1] + 22.0)   # under the centre arrow, not across it
    parts.append(f'<text x="{c0[0]:.1f}" y="{label_y:.1f}" font-size="12" fill="{ink}" text-anchor="middle">'
                 f'centre moves X {dx:+.2f} · Y {dy:+.2f} mm</text>')
    if abs(yaw) > 0.002:  # +yaw turns +X towards +Y: clockwise on screen (Y down)
        r_arc = 44
        a0 = -math.pi * 0.35
        sweep = math.copysign(math.pi * 0.55, yaw)
        pts = [(c0[0] + r_arc * math.cos(a0 + t * sweep), c0[1] + r_arc * math.sin(a0 + t * sweep))
               for t in np.linspace(0, 1, 18)]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{acc}" stroke-width="2.5" stroke-linecap="round"/>')
        (xa, ya), (xb, yb) = pts[-3], pts[-1]
        parts.append(_arrow(xa, ya, xb, yb, acc, 2.5, 10))
        parts.append(f'<text x="{c0[0] + r_arc + 8:.1f}" y="{c0[1] - r_arc + 6:.1f}" font-size="13" fill="{ink}" '
                     f'font-weight="600">yaw {yaw:+.3f}°</text>')
    for i, c in enumerate(CORNERS):
        p0 = to_px(rx[i], ry[i])
        p1 = to_px(rx[i] + k * ox[i], ry[i] + k * oy[i])
        parts.append(f'<circle cx="{p0[0]:.1f}" cy="{p0[1]:.1f}" r="4.5" fill="{muted}"/>')
        parts.append(_arrow(p0[0], p0[1], p1[0], p1[1], ink, 2.2, 9))
        parts.append(f'<circle cx="{p1[0]:.1f}" cy="{p1[1]:.1f}" r="5" fill="{acc}" stroke="{P.ink["surface"]}" '
                     f'stroke-width="2"/>')
        # labels sit in the frame corner on the same side as their corner, so they never collide,
        # however small a large magnification makes the drawing
        right = rx[i] > 0
        tx = width - 14 if right else 14
        anchor = "end" if right else "start"
        ty = 28 if ry[i] < 0 else height - 34
        parts.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="13" fill="{ink}" text-anchor="{anchor}">'
                     f'<tspan font-weight="700">{c}</tspan> X {ox[i]:+.2f} · Y {oy[i]:+.2f} mm</text>')
    parts.append(f'<text x="{width - 8}" y="{height - 8}" font-size="11" fill="{muted}" text-anchor="end">'
                 f'grey dashed: today · violet: after the correction (×{k:g}) · front →</text>')
    parts.append("</svg>")
    return "".join(parts)


def logo_svg(size: int = 40) -> str:
    P = T.pal()
    b, o = P.c["blue"], P.c["orange"]
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg" '
            f'aria-hidden="true">'
            f'<rect x="7" y="11" width="34" height="26" rx="3" fill="none" stroke="{b}" stroke-width="3"/>'
            f'<rect x="10.5" y="14.5" width="27" height="19" rx="2" fill="{T.rgba(b, 0.15)}"/>'
            f'<circle cx="41" cy="11" r="4.2" fill="{o}"/><circle cx="7" cy="11" r="4" fill="{b}"/>'
            f'<circle cx="7" cy="37" r="4" fill="{b}"/><circle cx="41" cy="37" r="4" fill="{b}"/>'
            f'<path d="M18 24 h12 M24 18 v12" stroke="{b}" stroke-width="2.5" stroke-linecap="round"/></svg>')
