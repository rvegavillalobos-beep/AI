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
n_fail, k_sq = sl.get("n_fail", 0), sl.get("nok_if_square", 0)
ui.kpi_cards([
    {"label": "Deformed modules", "value": ui.fmt_pct(sl["n_deformed"] / sl["n"] * 100),
     "sub": f"{sl['n_deformed']} of {sl['n']} · ΔDiag > {r.diag_tol:g} mm"},
    {"label": "NOK rate · deformed vs square", "value": f"{sl['fail_rate_deformed']:.0f}% vs {sl['fail_rate_square']:.0f}%",
     "sub": (f"×{ratio:.1f} · " if np.isfinite(ratio) else "") + significance_label(sl["p_value"])},
    {"label": "NOK caused by shape", "value": f"{k_sq} of {n_fail}" if n_fail else "–",
     "sub": (f"{k_sq / n_fail * 100:.0f}% of NOK would pass if square" if n_fail else "no NOK modules")},
    {"label": "FPY if every module were square", "value": ui.fmt_pct(sl.get("fpy_if_square", np.nan)),
     "delta": f"{sl.get('fpy_if_square', np.nan) - sl['fpy']:+.1f} pp vs today", "good": True},
    {"label": "FPY with perfect placement", "value": ui.fmt_pct(sl.get("fpy_if_placed", np.nan)),
     "delta": f"{sl.get('fpy_if_placed', np.nan) - sl['fpy']:+.1f} pp vs today", "good": True},
    {"label": "Tolerance used by shape", "value": f"{sl.get('shape_median', np.nan):.2f}", "unit": "mm",
     "sub": f"median, of ±{r.spec_limit:g} mm · NOK {sl.get('shape_median_fail', np.nan):.2f} · "
            f"OK {sl.get('shape_median_pass', np.nan):.2f}"},
])
st.caption("Compare *FPY if square* with *FPY with perfect placement*: they tell you whether shape or placement costs "
           "more FPY on this line. Complete first runs only (all four corners measured).")

# ---- Is deformation behind the NOK? -------------------------------------------------------------
ui.section("Is deformation behind the NOK?")
if n_fail == 0:
    st.success("No NOK modules among the complete first runs in this selection.", icon=":material/check_circle:")
else:
    if k_sq == 0:
        line1 = ("**Shape did not cost any NOK here.** Every NOK module would still fail if it were perfectly "
                 "square: they fail on placement (shift and rotation).")
    elif k_sq == n_fail:
        line1 = (f"**All {n_fail} NOK modules would have passed with the same placement and a perfectly square shape**, "
                 f"lifting FPY of complete modules from {sl['fpy']:.1f}% to {sl['fpy_if_square']:.1f}%.")
    else:
        line1 = (f"**Shape cost {k_sq} of {n_fail} NOK modules** ({k_sq / n_fail * 100:.0f}%): with the same placement "
                 f"and a perfectly square shape they would have passed, lifting FPY of complete modules from "
                 f"{sl['fpy']:.1f}% to {sl['fpy_if_square']:.1f}%. The other {n_fail - k_sq} fail on placement alone.")
    alone = sl.get("shape_alone", 0)
    if k_sq:
        line1 += (" No module fails from its shape alone." if alone == 0 else
                  f" {alone} fail from their shape alone, whatever the placement.")
    p = sl["p_value"]
    line2 = ""
    if np.isfinite(sl["fail_rate_deformed"]) and np.isfinite(sl["fail_rate_square"]):
        line2 = (f"Modules flagged *deformed* by the diagonal rule were NOK {sl['fail_rate_deformed']:.0f}% of the time "
                 f"vs {sl['fail_rate_square']:.0f}% for square ones"
                 + (f": a real difference (p = {p:.3g})." if np.isfinite(p) and p < 0.05 else
                    f": no clear difference (p = {p:.2f}), so the diagonal flag on its own does not predict NOK."))
    line3 = ""
    if np.isfinite(sl.get("shape_median_pass", np.nan)) and np.isfinite(sl.get("shape_median_fail", np.nan)):
        line3 = (f"Shape uses a median {sl['shape_median_fail']:.2f} mm of the ±{r.spec_limit:g} mm tolerance in NOK "
                 f"modules and {sl['shape_median_pass']:.2f} mm in OK ones.")
    box = st.warning if k_sq / n_fail >= 0.25 else st.info
    box("\n\n".join(x for x in (line1, line2, line3) if x), icon=":material/troubleshoot:")

left, right = st.columns(2, gap="large")
with left:
    ui.section("Why each NOK module failed", "Placement: fails even if perfectly square · Shape + placement: would "
                                            "pass if square (the shape ate the margin) · Shape: fails from its shape "
                                            "even if perfectly placed · Both: each alone is out of tolerance.")
    ui.chart(cs.nok_causes(sl.get("causes")), "sq_nokcause")
with right:
    ui.section("NOK rate by amount of deformation", "Bars: NOK rate per diagonal-delta band with 95% CI. If the "
                                                    "deformation drives NOK, the bars rise to the right.")
    ui.chart(cs.fail_rate_by_band(sl["by_band"], r.diag_tol, 100 - sl["fpy"]), "sq_band")

left, right = st.columns([3, 2], gap="large")
with left:
    ui.section("Placement error vs shape error", "x: the worst corner deviation the module would have if it were "
                                                 "perfectly square (pure placement). y: the worst shape residual. NOK "
                                                 "modules to the right of the red line fail on placement alone.")
    ui.chart(cs.failure_map(a.first, r.spec_limit), "sq_map")
with right:
    ui.section("Squareness by battery type")
    ui.chart(cs.squareness_status(sq), "sq_status")
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

with st.expander("How this is worked out", icon=":material/help:"):
    st.markdown(
        "- Each module's four corner deviations are split in two: **placement**, the X/Y shift and rotation that "
        "best explains the four corners together (least squares), and **shape**, what is left over.\n"
        "- *Would pass if square*: keep the module's placement, drop its shape, check the tolerance again. "
        "*Perfect placement*: keep its shape, drop the placement.\n"
        "- *Shape uses*: the largest corner deviation left by the shape alone. A module whose shape uses 1 mm has "
        "only the remaining ±(tolerance − 1) mm for placement.\n"
        "- The deformed/square comparison uses Fisher's exact test; p < 0.05 means the difference is unlikely to be "
        "chance.\n"
        "- Only modules with all four corners measured are included.")

ui.section("Squareness report", "Original table plus why each NOK failed. Select a row to focus that module in "
                                "Geometry and Module Passport.")
scopes = ["First runs", "All runs"]
state.ensure_option("p_sq_scope", scopes, "First runs")
which = st.segmented_control("Rows", scopes, key="p_sq_scope") or "First runs"
src = sq if which == "First runs" else a.sq_runs
only_def = st.toggle("Only deformed modules", key=state.init("p_sq_only", False))
if only_def:
    src = src[src["SquarenessStatus"] == "DEFORMED"]
tbl = src.sort_values("Date", ascending=False)[["_mod_key", "Date", "CalendarWeek", "PartID", "RunNum", "BatteryType",
                                                "Status", "NokCause", "ShapeMax", "Diag1", "Diag2", "DeltaDiag",
                                                "WidthDelta", "LengthDelta", "AngleDevFL", "SquarenessStatus",
                                                "RootCause"]].reset_index(drop=True)
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
                                 "NokCause": cfg.TextColumn("Why NOK", help="Placement / Shape + placement / Shape / "
                                                                            "Both (see the chart above)"),
                                 "ShapeMax": cfg.NumberColumn("Shape uses (mm)", format="%.2f"),
                                 "RootCause": cfg.TextColumn("Root cause", width="large")})
i = ui.selected_row(ev)
if i is not None and i < len(tbl) and tbl.loc[i, "_mod_key"] != state.focus():
    state.set_focus(tbl.loc[i, "_mod_key"])
    st.rerun()
