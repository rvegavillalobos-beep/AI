import numpy as np
import streamlit as st

from app import data, state, ui
from rickhard import metrics
from rickhard.charts import shape as cs
from rickhard.stats import significance_label

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "shape", "Is the module itself deformed, and does that make it fail? Each module is split into its "
                           "placement (a rigid move) and its shape (what no move can fix).")
ui.require(a.sq_first, "complete modules")
ui.focus_note(c)

sq = a.sq_first
sl = metrics.shape_link(sq, r.diag_tol)
if sl.get("n", 0) == 0:
    st.info("No complete first runs with PASS/FAIL status in the current filters.")
    st.stop()

ratio = sl["fail_rate_deformed"] / sl["fail_rate_square"] if sl.get("fail_rate_square") else np.nan
ui.kpi_cards([
    {"label": "Deformed modules", "value": ui.fmt_pct(sl["n_deformed"] / sl["n"] * 100),
     "sub": f"{sl['n_deformed']} of {sl['n']} · ΔDiag > {r.diag_tol:g} mm"},
    {"label": "NOK rate · deformed vs square", "value": f"{sl['fail_rate_deformed']:.0f}% vs {sl['fail_rate_square']:.0f}%",
     "sub": (f"×{ratio:.1f} · " if np.isfinite(ratio) else "") + significance_label(sl["p_value"])},
    {"label": "NOK that would pass if square", "value": f"{sl.get('nok_if_square', 0)}",
     "sub": f"of {sl['n_fail']} NOK (same placement, perfect shape)"},
    {"label": "FPY if every module were square", "value": ui.fmt_pct(sl.get("fpy_if_square", np.nan)),
     "delta": f"{sl.get('fpy_if_square', np.nan) - sl['fpy']:+.1f} pp vs today", "good": True},
    {"label": "FPY with perfect placement", "value": ui.fmt_pct(sl.get("fpy_if_placed", np.nan)),
     "delta": f"{sl.get('fpy_if_placed', np.nan) - sl['fpy']:+.1f} pp vs today", "good": True},
])
st.caption("Compare the last two cards: they tell you whether placement or shape costs more FPY on this line.")

left, right = st.columns([3, 2], gap="large")
with left:
    ui.section("Placement error vs shape error", "x: the worst corner deviation the module would have if it were "
                                                 "perfectly square (pure placement). y: the worst shape residual. NOK "
                                                 "modules to the right of the red line fail on placement alone.")
    ui.chart(cs.failure_map(a.first, r.spec_limit), "sq_map")
with right:
    ui.section("NOK rate by amount of deformation", "Bars: NOK rate per diagonal-delta band with 95% CI. If the "
                                                    "deformation drives NOK, the bars rise to the right.")
    ui.chart(cs.fail_rate_by_band(sl["by_band"], r.diag_tol, 100 - sl["fpy"]), "sq_band")

c1, c2 = st.columns(2, gap="large")
with c1:
    ui.section("Squareness by battery type")
    ui.chart(cs.squareness_status(sq), "sq_status")
with c2:
    ui.section("Deformation pattern", "Original root-cause rules: parallelogram tilt (FL angle > 0.15°, width delta "
                                      "< 0.8 mm), trapezoidal width, trapezoidal length, combined asymmetry.")
    ui.chart(cs.root_causes(sq), "sq_cause")

c3, c4 = st.columns(2, gap="large")
with c3:
    ui.section("Deformed share per week")
    ui.chart(cs.weekly_deformation(metrics.weekly_deformation(sq)), "sq_week")
with c4:
    ui.section("Diagonal delta distribution")
    ui.chart(cs.diag_hist(sq, r.diag_tol), "sq_hist")

ui.section("Squareness report", "Original table. Select a row to focus that module in Geometry and Module Passport.")
scopes = ["First runs", "All runs"]
state.ensure_option("p_sq_scope", scopes, "First runs")
which = st.segmented_control("Rows", scopes, key="p_sq_scope") or "First runs"
src = sq if which == "First runs" else a.sq_runs
only_def = st.toggle("Only deformed modules", key=state.init("p_sq_only", False))
if only_def:
    src = src[src["SquarenessStatus"] == "DEFORMED"]
tbl = src.sort_values("Date", ascending=False)[["_mod_key", "Date", "CalendarWeek", "PartID", "RunNum", "BatteryType",
                                                "Diag1", "Diag2", "DeltaDiag", "WidthDelta", "LengthDelta",
                                                "AngleDevFL", "SquarenessStatus", "RootCause", "Status"]].reset_index(drop=True)
cfg = st.column_config
ev = st.dataframe(ui.status_styler(tbl.drop(columns="_mod_key"), r.spec_limit), hide_index=True, width="stretch",
                  height=360, on_select="rerun", selection_mode="single-row", key="sq_table",
                  column_config={**ui.run_table_config(),
                                 "Diag1": cfg.NumberColumn("Diag FL–RR", format="%.2f"),
                                 "Diag2": cfg.NumberColumn("Diag FR–RL", format="%.2f"),
                                 "DeltaDiag": cfg.NumberColumn("Δ diag", format="%.2f"),
                                 "WidthDelta": cfg.NumberColumn("Width Δ", format="%+.2f"),
                                 "LengthDelta": cfg.NumberColumn("Length Δ", format="%+.2f"),
                                 "AngleDevFL": cfg.NumberColumn("FL angle Δ °", format="%+.3f"),
                                 "SquarenessStatus": cfg.TextColumn("Squareness"),
                                 "RootCause": cfg.TextColumn("Root cause", width="large")})
i = ui.selected_row(ev)
if i is not None and i < len(tbl) and tbl.loc[i, "_mod_key"] != state.focus():
    state.set_focus(tbl.loc[i, "_mod_key"])
    st.rerun()
