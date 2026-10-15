import numpy as np
import streamlit as st

from app import data, state, ui
from rickhard.charts import msa_charts as cm
from rickhard.constants import DEV_COLS

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "remeasure", "What re-measuring the same module tells you: how repeatable the station is (an MSA "
                               "estimate from production data) and how often a NOK turns OK on the next run.")
ui.require(a.runs, "runs")

state.init("p_rm_gap", 120.0)
gap = float(st.session_state["p_rm_gap"])
b = data.remeasure_bundle(c.key, gap, a)
pairs, rep, summ = b["pairs"], b["rep"], b["summary"]
if pairs is None or len(pairs) == 0:
    st.info("No module was measured more than once in the current filters.")
    st.stop()

ui.kpi_cards([
    {"label": "Modules re-measured", "value": f"{summ.get('modules_remeasured', 0):,}", "sub": f"{summ.get('pairs', 0):,} run pairs"},
    {"label": "NOK at run 1 re-measured", "value": f"{summ.get('nok_remeasured', 0):,}"},
    {"label": "NOK → OK at the next run", "value": ui.fmt_pct(summ.get("nok_to_ok_rate", np.nan)),
     "sub": ui.plural(int(summ.get('nok_to_ok', 0)), "module")},
    {"label": "Repeatability σ (median)", "value": ui.fmt_mm(summ.get("median_sigma", np.nan), 3, signed=False),
     "sub": f"from pairs ≤ {gap:.0f} min apart"},
    {"label": "%GRR (worst)", "value": ui.fmt_pct(summ.get("max_grr", np.nan), 0), "sub": summ.get("worst_coord", ""),
     "accent": "#d03b3b" if summ.get("max_grr", 0) >= 30 else ("#fab219" if summ.get("max_grr", 0) >= 10 else "#0ca30c")},
])

left, right = st.columns([2, 3], gap="large")
with left:
    ui.section("What happens at the next run")
    ui.chart(cm.transition_heatmap(b["transitions"]), "rm_trans")
    st.caption("A high NOK → OK share means many NOK verdicts are measurement or seating noise near the limit — "
               "or that re-measuring is used to 'test into compliance'. FPY always uses the first run.")
with right:
    ui.section("%GRR per corner coordinate", "6 × repeatability σ as a share of the tolerance width (±limit). "
                                             "< 10 % good, 10–30 % acceptable, > 30 % not acceptable (AIAG).")
    ui.chart(cm.grr_bars(rep), "rm_grr")

left, right = st.columns([3, 2], gap="large")
with left:
    coords = [c_.replace("_", " ") for c_ in DEV_COLS]
    state.ensure_option("p_rm_coord", coords, "FR X")
    coord = st.selectbox("Agreement between runs (Bland–Altman) for", coords, key="p_rm_coord")
    ui.chart(cm.bland_altman(pairs, coord.replace(" ", "_"), gap), "rm_ba")
with right:
    st.slider("Pairs used for repeatability: at most N minutes apart", 10.0, 1440.0, step=10.0, key="p_rm_gap")
    ui.chart(cm.gap_hist(pairs, gap), "rm_gap")
    st.caption("Long gaps usually mean rework between the runs, so they are excluded from the repeatability estimate.")

with st.expander("Repeatability table", icon=":material/table_rows:"):
    cfg = st.column_config
    st.dataframe(rep, hide_index=True, width="stretch", column_config={
        "BatteryType": "Type", "MeanDelta": cfg.NumberColumn("Mean Δ (bias)", format="%+.3f"),
        "SDDelta": cfg.NumberColumn("σ of Δ", format="%.3f"), "SigmaRepeat": cfg.NumberColumn("σ repeat", format="%.3f"),
        "GRR": cfg.NumberColumn("%GRR", format="%.1f%%")})
    st.caption("σ repeat = robust σ of the run-to-run difference / √2. A non-zero mean Δ means the second run is "
               "systematically different (re-seating, temperature, a fixture that settles).")

with st.expander("All run pairs", icon=":material/table_rows:"):
    show = pairs[["PartID", "BatteryType", "Run_a", "Run_b", "Date_a", "GapMin", "Status_a", "Status_b", "Worst_a",
                  "Worst_b"]].sort_values("Date_a", ascending=False)
    cfg = st.column_config
    st.dataframe(ui.status_styler(show, r.spec_limit), hide_index=True, width="stretch", height=360, column_config={
        "Date_a": cfg.DatetimeColumn("First run", format="YYYY-MM-DD HH:mm"),
        "GapMin": cfg.NumberColumn("Gap (min)", format="%.0f"),
        "Worst_a": cfg.NumberColumn("Worst |dev| a", format="%.2f"),
        "Worst_b": cfg.NumberColumn("Worst |dev| b", format="%.2f")})
