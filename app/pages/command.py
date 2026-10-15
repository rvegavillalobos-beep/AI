import numpy as np
import streamlit as st

from app import data, ui
from rickhard.charts import overview as co
from rickhard.charts import theme as T
from rickhard.constants import MECHANISM_HELP

c = data.ctx()
a, r, k = c.a, c.rules, c.a.kpis
ui.page_header(c, "command")
ui.require(a.first)

bundle = data.insight_bundle(c.key, a, c.ds)
ins, acts = bundle["insights"], bundle["actions"]
P = T.pal()

# ---- KPI cards ----------------------------------------------------------------------------
n = k["modules"]
lo, hi = k["fpy_ci"]
cards = [{
    "label": "First-pass yield", "value": ui.fmt_pct(k["fpy"]),
    "delta": f"{k['gap']:+.1f} pp vs {r.fpy_target:g}% target", "good": bool(k["gap"] >= 0),
    "sub": f"95% CI {lo:.1f}–{hi:.1f}%",
    "spark": T.sparkline_svg(k.get("spark", []), 104, 30, P.accent, target=r.fpy_target),
    "accent": T.STATUS["good"] if k["gap"] >= 0 else (T.STATUS["warning"] if k["gap"] > -5 else T.STATUS["critical"]),
}]
if "last_week" in k:
    diff = k["last_week_fpy"] - k.get("prev_fpy", np.nan)
    cards.append({"label": f"Latest week · {k['last_week']}", "value": ui.fmt_pct(k["last_week_fpy"]),
                  "delta": f"{diff:+.1f} pp vs the 4 weeks before" if np.isfinite(diff) else None,
                  "good": bool(diff >= 0) if np.isfinite(diff) else None, "sub": f"{k['last_week_n']} modules"})
cards.append({"label": "Modules (first run)", "value": f"{n:,}",
              "sub": f"{k['runs']:,} runs · {k['remeasured']:,} re-measured"})
nok = k["fail"] + (0 if r.incomplete_as_fail else k["incomplete"])
cards.append({"label": "NOK at first measurement", "value": f"{nok:,}",
              "sub": f"{nok / n * 100:.1f}% of modules · **{k['near_miss']}** near misses"})
if "worst_ppk" in k:
    cards.append({"label": "Weakest corner (Ppk)", "value": f"{k['worst_ppk']:.2f}", "sub": k["worst_coord"],
                  "accent": T.STATUS["critical"] if k["worst_ppk"] < 1 else (T.STATUS["warning"] if k["worst_ppk"] < 1.33
                                                                              else T.STATUS["good"])})
if "deformed_rate" in k:
    cards.append({"label": "Deformed modules", "value": ui.fmt_pct(k["deformed_rate"]),
                  "sub": f"diagonal delta > {r.diag_tol:g} mm"})
ui.kpi_cards(cards)

# ---- Findings and action plan -------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")
with left:
    ui.section("What the data says")
    ui.insight_cards(ins, limit=6)
    if len(ins) > 6:
        with st.expander(f"{len(ins) - 6} more findings", icon=":material/expand_more:"):
            ui.insight_cards(ins[6:])
with right:
    ui.section("Action plan", "Ranked by the FPY each action is estimated to be worth, simulated on your own modules.")
    with st.container(border=True):
        ui.action_list(acts, limit=8)
    st.page_link("app/pages/target.py", label="Simulate the levers — Path to Target", icon=":material/flag:")
    st.page_link("app/pages/compensation.py", label="Validate a correction — Compensation Studio", icon=":material/tune:")
    st.page_link("app/pages/reports.py", label="Report for management — Reports", icon=":material/summarize:")

# ---- Weekly FPY ------------------------------------------------------------------------------
ui.section("Weekly first-pass yield", "Line: FPY of the first measurement per week, band: 95% confidence interval, "
                                     "dashed: moving average. Bars: modules measured. Diamonds: logged interventions.")
ui.chart(co.weekly_fpy(a.weekly, r.fpy_target, c.ma, not r.incomplete_as_fail, c.events), "cc_weekly")

# ---- Why and where ---------------------------------------------------------------------------
help_mech = "  \n".join(f"**{m}** — {t}" for m, t in MECHANISM_HELP.items())
col1, col2 = st.columns(2, gap="large")
with col1:
    ui.section("Why NOK modules fail", help_mech)
    ui.chart(co.mechanisms(a.mechanisms), "cc_mech")
    st.caption("Each NOK is split into placement (shift, yaw) and shape; the label names what pushes the failing "
               "corner out. Placement failures are the ones a conveyor correction can fix.")
with col2:
    ui.section("Where on the module", "Bubble size: share of modules out of tolerance at each corner. Arrow: the "
                                      "corner's mean error, magnified.")
    ui.chart(co.hotspot_map(a.first, r.spec_limit, c.nominals, c.types), "cc_hot")

col3, col4 = st.columns([3, 2], gap="large")
with col3:
    ui.section("Corner and axis out of tolerance")
    ui.chart(co.corner_pareto(a.pareto, nok), "cc_pareto")
with col4:
    ui.section("First run by battery type")
    ui.chart(co.status_by_type(a.first), "cc_type")

with st.expander("Week by week", icon=":material/table_rows:"):
    wk = a.weekly.iloc[::-1]
    cols = ["CalendarWeek", "Total", "Passed", "Failed"] + (["Incomplete"] if not r.incomplete_as_fail else []) \
        + ["PassRate", "CI_lo", "CI_hi", "MA_FPY", "NearMissRate"]
    cc = st.column_config
    st.dataframe(wk[cols], hide_index=True, width="stretch", height=min(460, 40 + 35 * len(wk)), column_config={
        "CalendarWeek": cc.TextColumn("Week"), "Total": cc.NumberColumn("Modules"), "Passed": cc.NumberColumn("OK"),
        "Failed": cc.NumberColumn("NOK"),
        "PassRate": cc.ProgressColumn("FPY", format="%.1f%%", min_value=0, max_value=100),
        "CI_lo": cc.NumberColumn("CI low", format="%.1f%%"), "CI_hi": cc.NumberColumn("CI high", format="%.1f%%"),
        "MA_FPY": cc.NumberColumn(f"MA{c.ma}", format="%.1f%%"),
        "NearMissRate": cc.NumberColumn("Near misses", format="%.1f%%"),
    })
