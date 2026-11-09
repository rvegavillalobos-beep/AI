"""The executive brief: one page that tells the whole story on its own.

Where the first-pass yield stands against the target, what was done to the
line and whether it worked, what still costs yield, what it takes to close
the gap and what happens next. Built from the fix loop (``rickhard.fixloop``),
with static SVG charts: the file needs no chart library, opens offline and
prints to PDF as it looks.

``build`` returns the complete HTML file; ``fragment`` returns the same page
without the document skeleton (the app shows it as a preview, with ``css()``
injected separately).
"""

from __future__ import annotations

import html
import re
from datetime import datetime

import numpy as np
import pandas as pd

from . import APP_NAME, __version__, metrics
from . import fixloop as fx
from .charts import fix_svg, svg
from .charts import theme as T

DEFAULT_TITLE = "ST020 first-pass yield brief"

_LEVEL = {"good": "#0ca30c", "warn": "#c98500", "bad": "#d03b3b", "info": "#2a78d6", "neutral": "#898781"}
_ICON = {"Worked": "✓", "Target hit": "✓", "No effect": "○", "Backfired": "✕", "Verifying": "…",
         "Superseded": "»", "No baseline": "?", "Planned": "▸"}


def css() -> str:
    """Styles of the brief, scoped to ``.xb`` (the page is always light)."""
    return f"""
.xb {{font-family:{T.FONT}; color:#0b0b0b; background:#fff; line-height:1.45; font-size:14px;}}
.xb * {{box-sizing:border-box;}}
.xb-head {{display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap;
           border-bottom:2px solid #0b0b0b; padding-bottom:10px;}}
.xb-eyebrow {{font-size:11.5px; letter-spacing:.08em; text-transform:uppercase; color:#52514e; font-weight:600;}}
.xb-title {{font-size:26px; font-weight:750; line-height:1.15; margin:2px 0 0 0;}}
.xb-meta {{font-size:12px; color:#52514e; text-align:right;}}
.xb-headline {{font-size:21px; font-weight:650; line-height:1.3; margin:18px 0 6px 0; padding-left:14px;
               border-left:6px solid var(--lvl);}}
.xb-lead {{font-size:14px; color:#2c2c2a; margin:0 0 4px 20px; padding:0; max-width:900px;}}
.xb-tiles {{display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:16px 0 4px 0;}}
.xb-tile {{border:1px solid #e1e0d9; border-radius:10px; padding:10px 12px 9px 12px; background:#fcfcfb;}}
.xb-tile .l {{font-size:11.5px; color:#52514e;}}
.xb-tile .v {{font-size:25px; font-weight:700; line-height:1.15; margin:1px 0;}}
.xb-tile .s {{font-size:11.5px; color:#52514e;}}
.xb-sec {{margin-top:20px; break-inside:avoid; page-break-inside:avoid;}}
.xb-h {{font-size:15.5px; font-weight:700; margin:0 0 2px 0; display:flex; gap:9px; align-items:baseline;}}
.xb-h .n {{font-size:11.5px; font-weight:700; color:#fff; background:#0b0b0b; border-radius:999px; padding:1px 8px;}}
.xb-say {{font-size:13.5px; color:#2c2c2a; margin:0 0 8px 0; padding:0; max-width:940px;}}
.xb-note {{font-size:11.5px; color:#6b6a65; margin:4px 0 0 0; padding:0;}}
.xb img.chart {{display:block; width:100%; height:auto;}}
.xb-two {{display:grid; grid-template-columns:1fr 1fr; gap:26px;}}
.xb table {{border-collapse:collapse; width:100%; font-size:12.8px; margin:0;}}
.xb th {{text-align:left; font-size:11px; letter-spacing:.04em; text-transform:uppercase; color:#52514e;
         font-weight:600; padding:5px 8px; border:0; border-bottom:1.5px solid #0b0b0b; background:none;}}
.xb td {{padding:7px 8px; border:0; border-bottom:1px solid #e1e0d9; vertical-align:top; font-size:inherit;}}
.xb td.num {{white-space:nowrap; font-variant-numeric:tabular-nums;}}
.xb .sub {{font-size:11.5px; color:#6b6a65;}}
.xb-no {{display:inline-flex; width:20px; height:20px; border-radius:50%; background:#e87ba4; color:#0b0b0b;
         font-size:11px; font-weight:700; align-items:center; justify-content:center;}}
.xb-chip {{display:inline-flex; gap:5px; align-items:center; white-space:nowrap; font-size:11.5px; font-weight:650;
           border-radius:999px; padding:2px 9px 2px 7px; border:1px solid var(--c); color:#0b0b0b;
           background:color-mix(in srgb, var(--c) 14%, #fff);}}
.xb-chip i {{font-style:normal; color:var(--c); font-weight:800;}}
.xb-list {{margin:0; padding:0; list-style:none;}}
.xb-list li {{margin:0; padding:7px 0; font-size:inherit; border-bottom:1px solid #e1e0d9; display:flex; gap:10px;
              align-items:baseline;}}
.xb-list li:last-child {{border-bottom:0;}}
.xb-list .g {{margin-left:auto; white-space:nowrap; font-weight:700; font-variant-numeric:tabular-nums;}}
.xb-kind {{font-size:11px; letter-spacing:.04em; text-transform:uppercase; color:#52514e; font-weight:600;
           margin:10px 0 0 0;}}
.xb-asks {{border:1px solid #e1e0d9; border-left:5px solid #2a78d6; border-radius:8px; padding:10px 14px;
           background:#fcfcfb;}}
.xb-asks ul {{margin:4px 0 0 0; padding:0 0 0 18px;}}
.xb-asks li {{margin:0 0 2px 0; padding:0; font-size:inherit;}}
.xb-foot {{margin-top:22px; padding-top:8px; border-top:1px solid #e1e0d9; font-size:11px; color:#6b6a65;}}
@media (max-width: 760px) {{ .xb-two {{grid-template-columns:1fr;}} .xb-meta {{text-align:left;}} }}
"""


def _e(s) -> str:
    return html.escape(str(s))


def _b(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", _e(text))


def _chip(verdict: str, level: str) -> str:
    return (f"<span class='xb-chip' style='--c:{_LEVEL.get(level, _LEVEL['neutral'])}' "
            f"title='{_e(fx.VERDICT_HELP.get(verdict, ''))}'><i>{_ICON.get(verdict, '·')}</i>{_e(verdict)}</span>")


def _img(drawing: str, alt: str) -> str:
    if not drawing:
        return ""
    return svg.as_img(drawing, alt).replace("<img ", "<img class='chart' ", 1)


def _fnum(v, fmt: str, dash: str = "–") -> str:
    return dash if v is None or not np.isfinite(v) else format(v, fmt)


def _tiles(loop: dict) -> str:
    s, su, sc, p, target = loop["setup"], loop["setups"], loop["score"], loop["proof"], loop["target"]
    gap = s["FPY"] - target
    lost = float(loop["budget"]["Loss"].sum()) if len(loop["budget"]) else 0.0
    since = "the start of the data" if pd.isna(s["EventNo"]) and s["Setup"] == 1 else f"{pd.Timestamp(s['Start']):%b %d}"
    tiles = [
        (f"FPY, {loop['story']['which']}", f"{s['FPY']:.1f}%", f"{int(s['N']):,} modules since {since} · 95% CI "
                                                    f"{s['CI_lo']:.1f}–{s['CI_hi']:.1f}%"),
        (f"Gap to the {target:g}% target", "none" if gap >= 0 else f"{gap:.1f} pts".replace("-", "−"),
         f"{lost:.1f} points lost today · the target allows {max(0.0, 100 - target):.1f}"),
    ]
    judged = su[su["N"] >= fx.MIN_N]
    if len(judged) and int(judged.iloc[0]["Setup"]) != int(s["Setup"]):
        f0 = judged.iloc[0]
        tiles.append(("Since the first set-up", f"{s['FPY'] - f0['FPY']:+.1f} pts".replace("-", "−"),
                      f"{f0['FPY']:.1f}% → {s['FPY']:.1f}% over {len(su) - 1} logged change"
                      + ("" if len(su) == 2 else "s")))
    else:
        tiles.append(("FPY is at least", f"{_fnum(p['lower'], '.1f')}%", "with 95% confidence (cautious bound)"))
    if sc["done"]:
        rest = [f"{sc[k]} {lab}" for k, lab in (("no_effect", "no effect"), ("backfired", "backfired"),
                                                ("verifying", "being verified"), ("planned", "planned")) if sc[k]]
        tiles.append(("Line changes that worked", f"{sc['worked']} of {sc['judged']}" if sc["judged"] else "–",
                      " · ".join(rest) or "all judged changes improved the process"))
    if p["zero_fail_n"] is not None:
        tiles.append(("First-pass OK in a row", f"{p['streak']:,}",
                      f"best {p['best_streak']:,} · {p['zero_fail_n']:,} in a row demonstrate {target:g}%"))
    return "".join(f"<div class='xb-tile'><div class='l'>{_e(lab)}</div><div class='v'>{_e(v)}</div>"
                   f"<div class='s'>{_e(sub)}</div></div>" for lab, v, sub in tiles)


def _ledger_rows(led: pd.DataFrame) -> str:
    rows = []
    for _, r in led[led["Status"] != "Planned"].iterrows():
        if np.isfinite(r["FPY_before"]) and np.isfinite(r["FPY_after"]) and r["Verdict"] not in ("No baseline",):
            fpy = (f"{r['FPY_before']:.1f}% → {r['FPY_after']:.1f}%<div class='sub'>"
                   f"{r['FPY_delta']:+.1f} pts · {int(r['N_before'])} / {int(r['N_after'])} modules</div>")
        else:
            fpy = f"<span class='sub'>{int(r['N_before'])} before / {int(r['N_after'])} after</span>"
        ev_bits = []
        if r["Metric"] and r["Unit"] != "%" and np.isfinite(r["Metric_before"]):
            nd = 3 if r["Unit"] == "°" else 2
            ev_bits.append(f"{_e(r['Metric'])} {r['Metric_before']:+.{nd}f} → {r['Metric_after']:+.{nd}f} {r['Unit']}")
        if np.isfinite(r["Loss_before"]) and r["Verdict"] in ("Worked", "Target hit", "No effect", "Backfired"):
            ev_bits.append(f"NOK it aimed at {r['Loss_before']:.1f}% → {r['Loss_after']:.1f}% of modules")
        if not ev_bits and r["Verdict"] in ("Verifying", "Superseded", "No baseline"):
            ev_bits.append(_e(r["Detail"]))
        if r["Delivered"]:
            ev_bits.append(_e(r["Delivered"]))
        who = f" · {_e(r['Owner'])}" if str(r["Owner"]).strip() else ""
        rows.append(
            f"<tr><td><span class='xb-no'>{int(r['No'])}</span></td>"
            f"<td class='num'>{pd.Timestamp(r['Date']):%b %d}</td>"
            f"<td><b>{_e(r['Title'])}</b><div class='sub'>{_e(r['Scope'])} · {_e(r['Category'])}{who}</div></td>"
            f"<td>{_e(r['Target'])}</td><td class='num'>{fpy}</td><td>{_chip(r['Verdict'], r['Level'])}</td>"
            f"<td class='sub'>{'<br>'.join(ev_bits)}</td></tr>")
    return "".join(rows)


def _next_steps(loop: dict) -> tuple[str, str]:
    """(planned actions, levers the data suggests) as HTML; either may be empty."""
    led, br = loop["ledger"], loop["bridge"]
    out = []
    plan = led[led["Status"] == "Planned"]
    if len(plan):
        out.append("<div class='xb-kind'>Planned</div><ul class='xb-list'>")
        for _, r in plan.iterrows():
            who = f" · {_e(r['Owner'])}" if str(r["Owner"]).strip() else ""
            aim = "FPY in general" if r["Target"] == "FPY (general)" else _e(r["Target"])
            out.append(f"<li><span class='num sub'>{pd.Timestamp(r['Date']):%b %d}</span><span><b>{_e(r['Title'])}</b>"
                       f"<div class='sub'>aims at {aim} · {_e(r['Scope'])}{who}</div></span></li>")
        out.append("</ul>")
    planned, out = "".join(out), []
    sugg = br.iloc[1:][br.iloc[1:]["Delta"] >= 0.3] if len(br) > 1 else br.iloc[:0]
    if len(sugg):
        out.append("<div class='xb-kind'>Suggested by the data — gain simulated on this set-up</div><ul class='xb-list'>")
        for _, r in sugg.iterrows():
            out.append(f"<li><span><b>{_e(r['Step'])}</b><div class='sub'>{_e(r['How'])} · {_e(r['Owner'])}</div></span>"
                       f"<span class='g'>+{r['Delta']:.1f} pts</span></li>")
        out.append("</ul>")
    return planned, "".join(out)


def fragment(a, loop: dict, title: str = DEFAULT_TITLE, author: str = "", asks: str = "", note: str = "") -> str:
    """The brief without the document skeleton (charts in the light palette)."""
    prev = T.pal().dark
    T.set_dark(False)
    try:
        return _fragment(a, loop, title, author, asks, note)
    finally:
        T.set_dark(prev)


def _fragment(a, loop: dict, title: str, author: str, asks: str, note: str) -> str:
    r, f = a.settings.rules, a.first
    target = loop["target"]
    span = f"{f['Date'].min():%b %d} – {f['Date'].max():%b %d, %Y}" if len(f) else "–"
    by = f"{_e(author)} · " if author.strip() else ""
    head = (f"<div class='xb-head'><div><div class='xb-eyebrow'>Station ST020 · module corner deviation · first "
            f"measurement</div><div class='xb-title'>{_e(title)}</div></div>"
            f"<div class='xb-meta'>{span} · {_e(a.scope.label_types())} · tolerance ±{r.spec_limit:g} mm<br>"
            f"{by}{datetime.now():%Y-%m-%d}{(' · ' + _e(note)) if note.strip() else ''}</div></div>")
    if loop["setup"] is None:
        return f"<div class='xb'>{head}<p class='xb-say'>No modules in the current filters.</p></div>"
    st = loop["story"]
    changes = fx.changes(loop["events"], a.types)
    timeline = fix_svg.setup_timeline_svg(loop["setups"], metrics.daily_fpy(f), changes, target)
    rows = _ledger_rows(loop["ledger"])
    s = loop["setup"]
    since = f"{pd.Timestamp(s['Start']):%b %d}"
    parts = [
        f"<div class='xb'>{head}",
        f"<div class='xb-headline' style='--lvl:{_LEVEL[st['level']]}'>{_e(st['headline'])}</div>",
        f"<p class='xb-lead'>{_e(st['status'])}</p>",
        f"<div class='xb-tiles'>{_tiles(loop)}</div>",
        "<div class='xb-sec'><div class='xb-h'><span class='n'>1</span>Where we are</div>"
        "<p class='xb-say'>The line is not one process but a sequence of set-ups: each logged change starts a new one. "
        "The yield of a set-up is the yield of the line as it was run then.</p>"
        f"{_img(timeline, 'First-pass yield of every set-up over time')}</div>",
    ]
    if rows:
        parts.append(
            "<div class='xb-sec'><div class='xb-h'><span class='n'>2</span>What we did, and whether it worked</div>"
            f"<p class='xb-say'>{_e(st['done'])}</p>"
            "<table><tr><th>#</th><th>When</th><th>Change</th><th>Aimed at</th><th>FPY before → after</th>"
            f"<th>Verdict</th><th>Evidence</th></tr>{rows}</table>"
            f"<p class='xb-note'>Each change is judged on up to {fx.WINDOW} modules right before and right after it, "
            "never across another logged change. Worked / backfired: the FPY difference is statistically "
            "significant (Fisher's exact test, p &lt; 0.05).</p></div>")
    else:
        parts.append("<div class='xb-sec'><div class='xb-h'><span class='n'>2</span>What we did, and whether it "
                     f"worked</div><p class='xb-say'>{_e(st['done'])}</p></div>")
    loss = fix_svg.loss_budget_svg(loop["budget"], target)
    path = fix_svg.bridge_svg(loop["bridge"], target)
    parts.append(
        "<div class='xb-sec xb-two'>"
        "<div><div class='xb-h'><span class='n'>3</span>What still costs yield</div>"
        f"<p class='xb-say'>{_e(st['left'])}</p>{_img(loss, 'FPY points lost by failure mechanism')}"
        f"<p class='xb-note'>Set-up since {since}, {int(s['N']):,} modules. Every NOK first measurement is assigned "
        "the mechanism that pushes its failing corner out of tolerance.</p></div>"
        f"<div><div class='xb-h'><span class='n'>4</span>What it takes to reach {target:g}%</div>"
        f"<p class='xb-say'>{_e(st['path'])}</p>{_img(path, 'What it takes to reach the FPY target')}"
        "<p class='xb-note'>Each lever is applied to the measured modules of this set-up and every module is judged "
        "again; no distribution is assumed. The split between levers depends on their order.</p></div></div>")
    ask_lines = [x.strip(" -•\t") for x in asks.splitlines() if x.strip(" -•\t")]
    planned, suggested = _next_steps(loop)
    head5 = "<div class='xb-h'><span class='n'>5</span>Next steps</div>"
    proof = f"<p class='xb-note'>{_e(st['proof'])}</p>" if st["proof"] else ""
    nothing = "<p class='xb-say'>No next step is planned, and the data suggests none on this set-up.</p>"
    if ask_lines:
        ask_html = ("<div><div class='xb-h'><span class='n'>6</span>What we need</div><div class='xb-asks'><ul>"
                    + "".join(f"<li>{_b(x)}</li>" for x in ask_lines) + "</ul></div></div>")
        parts.append(f"<div class='xb-sec xb-two'><div>{head5}{planned}{suggested or (nothing if not planned else '')}"
                     f"{proof}</div>{ask_html}</div>")
    elif planned and suggested:      # no requests: the two kinds of next steps side by side
        parts.append(f"<div class='xb-sec'>{head5}<div class='xb-two'><div>{planned}</div><div>{suggested}</div></div>"
                     f"{proof}</div>")
    else:
        parts.append(f"<div class='xb-sec'>{head5}{planned or suggested or nothing}{proof}</div>")
    parts.append(
        f"<div class='xb-foot'>FPY counts the first measurement of each module; a module passes when every corner is "
        f"within ±{r.spec_limit:g} mm in X and Y"
        + (" and Z is within its limits" if r.z_in_status and a.has_z else "")
        + ("; a first measurement with a missing corner counts as NOK" if r.incomplete_as_fail else "")
        + f". Confidence intervals: Wilson 95%. Generated with {APP_NAME} v{__version__}.</div></div>")
    return "".join(parts)


def build(a, loop: dict, title: str = DEFAULT_TITLE, author: str = "", asks: str = "", note: str = "") -> str:
    """The complete, self-contained HTML file."""
    return wrap(fragment(a, loop, title, author, asks, note), title)


def wrap(body: str, title: str = DEFAULT_TITLE) -> str:
    """A ``fragment`` as a complete HTML file (screen: a sheet on a grey desk;
    print: A4 landscape, three pages that read like slides)."""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_e(title)}</title>
<style>
 html {{background:#f0efec;}}
 body {{margin:0; padding:24px 16px;}}
 .xb-page {{max-width:1120px; margin:0 auto; background:#fff; padding:30px 36px 26px 36px; border-radius:6px;
            box-shadow:0 1px 3px rgba(0,0,0,.12);}}
 @page {{size:A4 landscape; margin:10mm;}}
 @media print {{
   html {{background:#fff;}} body {{padding:0;}}
   .xb-page {{max-width:none; box-shadow:none; padding:0; border-radius:0;}}
   .xb-chip, .xb-no, .xb-h .n {{-webkit-print-color-adjust:exact; print-color-adjust:exact;}}
 }}
{css()}
</style></head><body><div class="xb-page">{body}</div></body></html>"""
