"""Small rendering helpers shared by the pages."""

from __future__ import annotations

import html
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from rickhard.charts import svg
from rickhard.charts import theme as T
from rickhard.constants import DEV_COLS

PAGE_ICONS = {
    "command": ":material/space_dashboard:", "corners": ":material/crop_free:", "geometry": ":material/view_in_ar:",
    "shape": ":material/square_foot:", "drift": ":material/moving:", "anatomy": ":material/biotech:",
    "spc": ":material/monitor_heart:", "patterns": ":material/schedule:", "remeasure": ":material/replay:",
    "height": ":material/height:", "target": ":material/flag:", "compensation": ":material/tune:",
    "events": ":material/event_note:", "passport": ":material/badge:", "data": ":material/dataset:",
    "reports": ":material/summarize:",
}
PAGE_TITLES = {
    "command": "Command Center", "corners": "Corner Lab", "geometry": "Geometry", "shape": "Shape & Squareness",
    "drift": "Drift & Yaw", "anatomy": "Error Anatomy", "spc": "SPC Monitor", "patterns": "Time Patterns",
    "remeasure": "Re-measurement & MSA", "height": "Height (Z)", "target": "Path to Target",
    "compensation": "Compensation Studio", "events": "Interventions", "passport": "Module Passport",
    "data": "Data & Settings", "reports": "Reports",
}
LEVEL_COLOR = {"good": T.STATUS["good"], "warn": T.STATUS["warning"], "bad": T.STATUS["critical"],
               "info": "#2a78d6", "neutral": "#898781"}
LEVEL_ICON = {"good": "✓", "warn": "!", "bad": "✕", "info": "i", "neutral": "·"}

CSS = """
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1500px;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  .rx-scope {opacity: .68; font-size: .88rem; margin: -.55rem 0 1rem 0;}
  .rx-kpiwrap {container-type: inline-size;}
  .rx-kpis {display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); gap: 12px; margin: .2rem 0 1rem 0;}
  @container (max-width: 1020px) {.rx-kpis.n6, .rx-kpis.n7, .rx-kpis.n8 {grid-template-columns: repeat(var(--n2), minmax(0, 1fr));}}
  @container (max-width: 860px) {.rx-kpis.n5 {grid-template-columns: repeat(3, minmax(0, 1fr));}}
  @container (max-width: 700px) {.rx-kpis.n4 {grid-template-columns: repeat(2, minmax(0, 1fr));}}
  @container (max-width: 520px) {.rx-kpis {grid-template-columns: repeat(2, minmax(0, 1fr));}}
  @container (max-width: 300px) {.rx-kpis {grid-template-columns: minmax(0, 1fr);}}
  .rx-card {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 14px 16px 12px 16px;
            background: rgba(128,128,128,.04); position: relative; overflow: hidden; min-width: 0;}
  .rx-card .lab {font-size: .78rem; opacity: .72; letter-spacing: .01em;}
  .rx-card .row {display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 2px 10px;}
  .rx-card .val {font-size: 1.85rem; font-weight: 650; line-height: 1.15; margin-top: 2px; overflow-wrap: anywhere;}
  .rx-card .val.m {font-size: 1.5rem;} .rx-card .val.s {font-size: 1.22rem;}
  .rx-card .val small {font-size: .9rem; font-weight: 500; opacity: .7;}
  .rx-card .dlt {font-size: .82rem; font-weight: 600; margin-top: 2px;}
  .rx-card .sub {font-size: .76rem; opacity: .66; margin-top: 2px;}
  .rx-card .spk img {display: block; opacity: .95;}
  .rx-card .bar {position: absolute; left: 0; top: 0; bottom: 0; width: 4px;}
  .rx-ins {border: 1px solid rgba(128,128,128,.20); border-left-width: 4px; border-radius: 10px;
           padding: 9px 12px; margin: 0 0 8px 0; background: rgba(128,128,128,.03);}
  .rx-ins .t {font-weight: 650; font-size: .92rem; margin-bottom: 2px; display: flex; gap: 8px; align-items: center;}
  .rx-ins .t .ic {display: inline-flex; width: 18px; height: 18px; border-radius: 50%; color: #fff; font-size: 11px;
                  font-weight: 700; align-items: center; justify-content: center; flex: 0 0 18px;}
  .rx-ins .x {font-size: .88rem; line-height: 1.45; opacity: .92;}
  .rx-ins .a {font-size: .82rem; margin-top: 4px; opacity: .85;}
  .rx-act {display: flex; gap: 10px; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid rgba(128,128,128,.15);}
  .rx-act:last-child {border-bottom: 0;}
  .rx-act .n {flex: 0 0 26px; height: 26px; border-radius: 50%; background: rgba(42,120,214,.14); color: #2a78d6;
              font-weight: 700; display: flex; align-items: center; justify-content: center; font-size: .85rem;}
  .rx-act .b {flex: 1 1 auto; font-size: .9rem; line-height: 1.4;}
  .rx-act .b .w {font-size: .78rem; opacity: .65;}
  .rx-act .g {flex: 0 0 auto; font-weight: 700; font-size: .86rem; color: #0ca30c; white-space: nowrap;}
  .rx-tag {display: inline-block; font-size: .72rem; padding: 1px 8px; border-radius: 999px;
           border: 1px solid rgba(128,128,128,.3); margin-left: 6px; opacity: .85;}
  .rx-hero h1 {font-size: 2.6rem; margin-bottom: .2rem;}
  .rx-hero p {font-size: 1.05rem; opacity: .8; max-width: 760px;}
  .rx-feat {display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 12px; margin-top: .6rem;}
  .rx-feat div {border: 1px solid rgba(128,128,128,.2); border-radius: 12px; padding: 12px 14px; font-size: .88rem;
                line-height: 1.45; background: rgba(128,128,128,.03);}
  .rx-feat b {display: block; margin-bottom: 3px; font-size: .95rem;}
  .rx-chk {display: flex; gap: 10px; padding: 6px 0; font-size: .9rem; border-bottom: 1px solid rgba(128,128,128,.12);}
  .rx-chk .ok {color: #0ca30c; font-weight: 700;} .rx-chk .no {color: #d03b3b; font-weight: 700;}
  .rx-chk .d {opacity: .7; font-size: .82rem;}
  .rx-svg {border: 1px solid rgba(128,128,128,.2); border-radius: 12px; padding: 8px; background: rgba(128,128,128,.03);}
  .rx-svg img {display: block; width: 100%; height: auto; margin: 0 auto;}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def md_bold(text: str) -> str:
    """Escape and turn **bold** markdown into <b>."""
    t = html.escape(str(text))
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


def fmt_pct(v, nd: int = 1) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    if 0 < abs(v) < 10 ** -nd:  # a few modules must not read as 0.0 %
        return f"<{10 ** -nd:g}%"
    return f"{v:.{nd}f}%"


def plural(n: int, word: str, words: str | None = None) -> str:
    return f"{n:,} {word if n == 1 else (words or word + 's')}"


def fmt_mm(v, nd: int = 2, signed: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v:+.{nd}f} mm" if signed else f"{v:.{nd}f} mm"


# ---------------------------------------------------------------------------
# Headers
# ---------------------------------------------------------------------------

def page_header(c, key: str, subtitle: str | None = None) -> None:
    st.title(PAGE_TITLES[key])
    f = c.a.first
    d0, d1 = (f["Date"].min(), f["Date"].max()) if len(f) else (pd.NaT, pd.NaT)
    span = f"{d0:%b %d} – {d1:%b %d, %Y}" if pd.notna(d0) else "no dates"
    demo = " · <b>demo data</b>" if c.is_demo else ""
    st.markdown(f"<div class='rx-scope'>{html.escape(c.file_label)}{demo} · {len(f):,} modules (first run) · "
                f"{span} · {html.escape(c.week_label)} · {html.escape(c.scope.label_types())} · "
                f"tolerance ±{c.rules.spec_limit:g} mm</div>", unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def require(df, what: str = "modules") -> None:
    if df is None or len(df) == 0:
        st.info(f"No {what} in the current filters. Widen the calendar weeks or the battery type in the sidebar.",
                icon=":material/filter_alt_off:")
        st.stop()


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

def kpi_cards(cards: list[dict]) -> None:
    """cards: {label, value, unit?, delta?, good? (True/False/None), sub?, spark? (SVG), accent?}

    One row when there is room, otherwise balanced rows (6 → 3 + 3, 5 → 3 + 2)."""
    n = max(1, len(cards))
    # each card needs ~170 px: the row splits into balanced rows below n × 170 px (CSS container queries)
    out = [f"<div class='rx-kpiwrap'><div class='rx-kpis n{min(n, 8)}' style='--n:{n};--n2:{math.ceil(n / 2)}'>"]
    for c in cards:
        color = {True: T.STATUS["good"], False: T.STATUS["critical"]}.get(c.get("good"), "inherit")
        arrow = ""
        if c.get("delta"):
            d = str(c["delta"]).strip()
            arrow = "▲ " if d.startswith("+") else ("▼ " if d.startswith(("−", "-")) else "")
        bar = f"<div class='bar' style='background:{c['accent']}'></div>" if c.get("accent") else ""
        unit = f" <small>{html.escape(str(c['unit']))}</small>" if c.get("unit") else ""
        value = str(c["value"])
        size = "" if len(value) <= 9 else (" m" if len(value) <= 13 else " s")
        spark = (f"<div class='spk'>{svg.as_img(c['spark'], 'trend', 'width:104px;height:30px')}</div>"
                 if c.get("spark") else "")
        out.append(
            "<div class='rx-card'>" + bar
            + f"<div class='lab'>{html.escape(str(c['label']))}</div>"
            + f"<div class='row'><div class='val{size}'>{html.escape(value)}{unit}</div>{spark}</div>"
            + (f"<div class='dlt' style='color:{color}'>{arrow}{html.escape(str(c['delta']))}</div>" if c.get("delta") else "")
            + (f"<div class='sub'>{md_bold(c['sub'])}</div>" if c.get("sub") else "")
            + "</div>")
    out.append("</div></div>")
    st.html("".join(out))


def insight_cards(items, limit: int | None = None) -> None:
    rows = []
    for i in (items[:limit] if limit else items):
        col = LEVEL_COLOR.get(i.level, "#898781")
        action = f"<div class='a'>→ {md_bold(i.action)}</div>" if getattr(i, "action", None) else ""
        rows.append(f"<div class='rx-ins' style='border-left-color:{col}'>"
                    f"<div class='t'><span class='ic' style='background:{col}'>{LEVEL_ICON.get(i.level, '·')}</span>"
                    f"{md_bold(i.title)}</div><div class='x'>{md_bold(i.text)}</div>{action}</div>")
    if rows:
        st.html("".join(rows))
    else:
        st.caption("Not enough data for findings.")


def action_list(actions, limit: int | None = None) -> None:
    rows = []
    for n, a in enumerate(actions[:limit] if limit else actions, start=1):
        gain = f"+{a.impact:.1f} pp" if a.impact is not None and np.isfinite(a.impact) and a.impact > 0 else ""
        rows.append(f"<div class='rx-act'><div class='n'>{n}</div><div class='b'>{md_bold(a.action)}"
                    f"<div class='w'>{md_bold(a.why)}<span class='rx-tag'>{html.escape(a.owner)}</span></div></div>"
                    f"<div class='g' title='Estimated FPY gain'>{gain}</div></div>")
    if rows:
        st.html("<div>" + "".join(rows) + "</div>")
    else:
        st.caption("No actions — the data does not point at anything specific yet.")


def checklist(items: list[dict]) -> None:
    rows = []
    for it in items:
        mark = "<span class='ok'>✓</span>" if it["ok"] else "<span class='no'>✕</span>"
        rows.append(f"<div class='rx-chk'>{mark}<div><b>{html.escape(it['check'])}</b><br>"
                    f"<span class='d'>{html.escape(it['detail'])}</span></div></div>")
    st.html("".join(rows))


def svg_box(drawing: str, alt: str = "", max_width: int = 640) -> None:
    st.html(f"<div class='rx-svg'>{svg.as_img(drawing, alt, f'max-width:{max_width}px')}</div>")


# ---------------------------------------------------------------------------
# Charts and tables
# ---------------------------------------------------------------------------

PLOT_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"],
               "toImageButtonOptions": {"format": "png", "scale": 2}}


def chart(fig, key: str, **kw):
    return st.plotly_chart(fig, width="stretch", key=key, config=PLOT_CONFIG, **kw)


def section(title: str, help_text: str | None = None) -> None:
    st.markdown(f"##### {title}", help=help_text)


STATUS_CSS = {"PASS": "background-color:rgba(12,163,12,.16);color:#0b7a0b;font-weight:600",
              "FAIL": "background-color:rgba(208,59,59,.16);color:#b02a2a;font-weight:600",
              "INCOMPLETE": "background-color:rgba(137,135,129,.22);font-weight:600",
              "DEFORMED": "background-color:rgba(208,59,59,.16);color:#b02a2a;font-weight:600",
              "SQUARE OK": "color:#0b7a0b"}


def status_styler(df: pd.DataFrame, limit: float, max_rows: int = 3000, precision: int = 2):
    """Red cells for out-of-tolerance deviations, coloured status cells.
    Plain dataframe above ``max_rows`` (styling large tables is slow)."""
    if len(df) > max_rows:
        return df
    dev_cols = [c for c in df.columns if c in DEV_COLS or (c.endswith("_Sim") and c[:-4] in DEV_COLS)]

    def dev_style(v):
        try:
            return "background-color:rgba(208,59,59,.18);color:#b02a2a;font-weight:600" if abs(float(v)) > limit else ""
        except (TypeError, ValueError):
            return ""

    sty = df.style
    if dev_cols:
        sty = sty.map(dev_style, subset=dev_cols)
    for col in ("Status", "Status_Sim", "Status_a", "Status_b", "SquarenessStatus"):
        if col in df.columns:
            sty = sty.map(lambda v: STATUS_CSS.get(str(v), ""), subset=[col])
    return sty.format(precision=precision, na_rep="–")


def run_table_config() -> dict:
    cc = st.column_config
    cfg = {
        "Date": cc.DatetimeColumn("Date", format="YYYY-MM-DD HH:mm"),
        "CalendarWeek": cc.TextColumn("Week"),
        "PartID": cc.TextColumn("Part ID", width="medium"),
        "BatteryType": cc.TextColumn("Type"),
        "RunNum": cc.NumberColumn("Run", format="%d"),
        "CornersOutOfSpec": cc.NumberColumn("Corners out", format="%d"),
        "MissingCorners": cc.NumberColumn("Missing", format="%d"),
        "WorstAbs": cc.NumberColumn("Worst |dev|", format="%.2f"),
        "Margin": cc.NumberColumn("Margin", format="%.2f"),
    }
    for c in DEV_COLS:
        cfg[c] = cc.NumberColumn(c.replace("_", " "), format="%.2f")
    return cfg


def selected_row(event) -> int | None:
    """Index of the selected row from an st.dataframe selection event."""
    try:
        rows = event.selection.rows
    except AttributeError:
        try:
            rows = event["selection"]["rows"]
        except (KeyError, TypeError):
            return None
    return rows[0] if rows else None


def focus_note(c) -> None:
    from . import state
    key = state.focus()
    if not key:
        return
    page = st.session_state.get("_page", "")
    links = [(p, lab, ic) for p, lab, ic in (("geometry", "Geometry", ":material/view_in_ar:"),
                                             ("passport", "Passport", ":material/badge:")) if p != page]
    cols = st.columns([5] + [1.3] * len(links) + [1.2], vertical_alignment="center")
    cols[0].caption(f"Focused module: **{key}**")
    for col, (p, lab, ic) in zip(cols[1:], links):
        col.page_link(f"app/pages/{p}.py", label=f"Open in {lab}", icon=ic)
    cols[-1].button("Clear focus", icon=":material/close:", key=f"clear_focus_{page}",
                    on_click=state.set_focus, args=(None,), type="tertiary")


def download_bytes(label: str, data: bytes, file_name: str, mime: str, key: str, icon: str = ":material/download:",
                   primary: bool = False) -> None:
    st.download_button(label, data=data, file_name=file_name, mime=mime, key=key, icon=icon,
                       type="primary" if primary else "secondary", width="stretch")
