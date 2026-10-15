"""Self-contained HTML management report: KPIs, findings, action plan and the
key interactive charts — opens in any browser, prints cleanly."""

from __future__ import annotations

import html
import re
from datetime import datetime

import numpy as np

from . import APP_NAME, __version__
from .charts import anatomy as ca
from .charts import corners as cc
from .charts import overview as co
from .charts import theme as T
from .improvement import budget


def _b(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(str(text)))


def _fig_html(fig, first: bool, include: str) -> str:
    T.report_style(fig)
    return fig.to_html(full_html=False, include_plotlyjs=(include if first else False),
                       config={"displaylogo": False, "responsive": True})


def build_html(a, insights, actions, events=None, comp_rows=None, title: str = "ST020 corner deviation report",
               offline: bool = False, note: str = "") -> str:
    """Report HTML; charts are always drawn with the light palette."""
    prev = T.pal().dark
    T.set_dark(False)
    try:
        return _build(a, insights, actions, events, comp_rows, title, offline, note)
    finally:
        T.set_dark(prev)


def _build(a, insights, actions, events, comp_rows, title, offline, note) -> str:
    r = a.settings.rules
    k = a.kpis
    f = a.first
    span = f"{f['Date'].min():%b %d} – {f['Date'].max():%b %d, %Y}" if len(f) else "–"
    lo, hi = k["fpy_ci"]
    gap = k["gap"]
    color = "#0ca30c" if gap >= 0 else ("#c98500" if gap > -5 else "#d03b3b")
    tiles = [
        ("First-pass yield", f"{k['fpy']:.1f}%", f"{gap:+.1f} pp vs {r.fpy_target:g}% target · CI {lo:.1f}–{hi:.1f}%", color),
        ("Modules", f"{k['modules']:,}", f"{k['runs']:,} runs · {k['remeasured']:,} re-measured", None),
        ("NOK at first measurement", f"{k['fail']:,}", f"{k['near_miss']} near misses", None),
    ]
    if "last_week" in k:
        tiles.append((f"Latest week {k['last_week']}", f"{k['last_week_fpy']:.1f}%", f"{k['last_week_n']} modules", None))
    if "worst_ppk" in k:
        tiles.append(("Weakest corner", f"Ppk {k['worst_ppk']:.2f}", k["worst_coord"], None))
    tile_html = "".join(
        f"<div class='tile' style='border-top-color:{c or '#2a78d6'}'><div class='l'>{html.escape(lab)}</div>"
        f"<div class='v'>{html.escape(v)}</div><div class='s'>{html.escape(s)}</div></div>" for lab, v, s, c in tiles)
    ins_html = "".join(
        f"<div class='ins {i.level}'><b>{_b(i.title)}</b><br>{_b(i.text)}</div>" for i in insights[:8])
    act_rows = "".join(
        f"<tr><td>{n}</td><td>{_b(x.action)}</td><td>{_b(x.why)}</td>"
        f"<td class='num'>{'' if x.impact is None or not np.isfinite(x.impact) else f'+{x.impact:.1f}'}</td>"
        f"<td>{html.escape(x.owner)}</td></tr>" for n, x in enumerate(actions[:10], start=1))
    include = "cdn" if not offline else True
    figs = [
        ("Weekly first-pass yield", co.weekly_fpy(a.weekly, r.fpy_target, 3, not r.incomplete_as_fail, events, height=420)),
        ("Why NOK modules fail", co.mechanisms(a.mechanisms, height=280)),
        ("Where on the module NOK happens", co.hotspot_map(f, r.spec_limit, a.settings.nominals, a.types, height=340),
         "Bubble: share of modules out of tolerance at that corner · arrow: the corner's mean error ×150 · "
         "front of the module to the right."),
        ("Corner and axis out of tolerance", co.corner_pareto(a.pareto, k["fail"], height=330)),
        ("Capability per corner and axis (Ppk)", cc.capability_grid(a.capability, height=360)),
    ]
    bud = budget(f)
    for t in a.types:
        figs.append((f"Error budget · {t}", ca.budget_bars(bud, t, height=380)))
    chart_html = ""
    for i, (h, fig, *fig_note) in enumerate(figs):
        cap = f"<div class='meta'>{html.escape(fig_note[0])}</div>" if fig_note else ""
        chart_html += (f"<section class='chart'><h3>{html.escape(h)}</h3>{_fig_html(fig, i == 0, include)}{cap}"
                       "</section>")
    comp_html = ""
    if comp_rows is not None and len(comp_rows):
        comp_html = "<h2>Compensation</h2>" + comp_rows.to_html(index=False, float_format=lambda v: f"{v:+.3f}",
                                                                 classes="tbl", border=0)
    ev_html = ""
    if events is not None and len(events):
        e = events[["Date", "Scope", "Category", "Title"]].copy()
        e["Date"] = e["Date"].dt.strftime("%Y-%m-%d %H:%M")
        ev_html = "<h2>Interventions</h2>" + e.to_html(index=False, classes="tbl", border=0)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
 body {{font-family: {T.FONT}; color:#111; background:#fff; margin:0; padding:28px 36px; max-width:1180px;}}
 h1 {{margin:0 0 4px 0; font-size:28px;}} h2 {{margin:28px 0 10px 0; font-size:19px;}}
 h3 {{margin:0 0 4px 0; font-size:15px;}}
 .meta {{color:#666; font-size:13px; margin-bottom:18px;}}
 .tiles {{display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px;}}
 .tile {{border:1px solid #e3e3e3; border-top:4px solid #2a78d6; border-radius:10px; padding:12px 14px;}}
 .tile .l {{font-size:12px; color:#666;}} .tile .v {{font-size:26px; font-weight:700; margin:2px 0;}}
 .tile .s {{font-size:12px; color:#666;}}
 .ins {{border:1px solid #e6e6e6; border-left:4px solid #2a78d6; border-radius:8px; padding:8px 12px; margin:0 0 8px 0;
        font-size:14px; line-height:1.45;}}
 .ins.good {{border-left-color:#0ca30c;}} .ins.warn {{border-left-color:#fab219;}} .ins.bad {{border-left-color:#d03b3b;}}
 table.tbl, table.act {{border-collapse:collapse; width:100%; font-size:13px;}}
 table.tbl th, table.act th {{background:#1f3a5f; color:#fff; text-align:left; padding:6px 8px;}}
 table.tbl td, table.act td {{border-bottom:1px solid #eee; padding:6px 8px; vertical-align:top;}}
 td.num {{text-align:right; font-weight:700; color:#0b7a0b; white-space:nowrap;}}
 .chart {{margin:18px 0; page-break-inside:avoid;}}
 .foot {{color:#888; font-size:12px; margin-top:30px;}}
 @media print {{ body {{padding:0;}} .chart {{page-break-inside:avoid;}} }}
</style></head><body>
<h1>{html.escape(title)}</h1>
<div class="meta">{span} · {html.escape(a.scope.label_types())} · tolerance ±{r.spec_limit:g} mm · FPY target
{r.fpy_target:g}% · generated {datetime.now():%Y-%m-%d %H:%M} with {APP_NAME} v{__version__}
{(' · ' + html.escape(note)) if note else ''}</div>
<div class="tiles">{tile_html}</div>
<h2>What the data says</h2>{ins_html}
<h2>Action plan</h2>
<table class="act"><tr><th>#</th><th>Action</th><th>Why</th><th>Est. FPY gain (pp)</th><th>Owner</th></tr>{act_rows}</table>
{chart_html}{comp_html}{ev_html}
<div class="foot">FPY uses the first measurement of each module. Confidence intervals: Wilson 95%. Estimated gains
are simulated on the modules in this report.</div>
</body></html>"""
