import numpy as np
import streamlit as st

from app import data, state, ui
from rickhard import improvement as imp
from rickhard.charts import comp_charts as ccmp

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "target", "What would it take to reach the FPY target? Pull the levers and the app re-judges your "
                            "own modules — no distribution assumptions.")
ui.require(a.first)

bundle = data.insight_bundle(c.key, a, c.ds)
comps = dict(bundle["comps"])
chosen = data.chosen_compensation()
src = {t: "rigid fit, last 6 weeks" for t in comps}
for t, v in chosen.items():
    comps[t] = v["setting"]
    src[t] = f"{v['method']} (Compensation Studio)"

left, right = st.columns([1, 2], gap="large")
with left:
    ui.section("Levers")
    with st.container(border=True):
        comp_on = st.toggle("Apply the placement correction", key=state.init("p_tg_comp", True))
        if comp_on:
            for t, s in comps.items():
                st.caption(f"{t}: X {s[0]:+.2f} mm · Y {s[1]:+.2f} mm · yaw {s[2]:+.3f}° — {src[t]}")
            if not comps:
                st.caption("Not enough modules to estimate a correction.")
        yaw_r = st.slider("Reduce the yaw scatter by", 0, 100, step=5, format="%d%%", key=state.init("p_tg_yaw", 50),
                          help="Guides, side pushers, belt tracking.")
        shift_r = st.slider("Reduce the X/Y placement scatter by", 0, 100, step=5, format="%d%%",
                            key=state.init("p_tg_shift", 30), help="End-stop play, gripper, conveyor stop accuracy.")
        shape_r = st.slider("Reduce the shape scatter by", 0, 100, step=5, format="%d%%",
                            key=state.init("p_tg_shape", 0), help="Module build quality upstream.")
        no_bias = st.toggle("Remove the systematic shape error", key=state.init("p_tg_bias", False),
                            help="E.g. a fixture that makes every module slightly trapezoidal.")
levers = imp.Levers(bool(comp_on), shift_r / 100, yaw_r / 100, shape_r / 100, bool(no_bias))
wf = imp.waterfall(a.first, levers, comps, r.spec_limit, c.nominals)
today, scen = float(wf["FPY"].iloc[0]), float(wf["FPY"].iloc[-1])
with right:
    ui.kpi_cards([
        {"label": "FPY today", "value": ui.fmt_pct(today), "sub": f"{len(a.first):,} modules"},
        {"label": "FPY in this scenario", "value": ui.fmt_pct(scen), "delta": f"{scen - today:+.1f} pp",
         "good": scen >= today},
        {"label": f"Gap to the {r.fpy_target:g}% target", "value": f"{scen - r.fpy_target:+.1f} pp",
         "good": scen >= r.fpy_target, "accent": "#0ca30c" if scen >= r.fpy_target else "#d03b3b"},
    ])
    ui.chart(ccmp.waterfall(wf, r.fpy_target), "tg_wf")

need = imp.required_reduction(a.first, r.fpy_target, comps, r.spec_limit, c.nominals, "placement", comp_on)
ceil = imp.ceilings(a.first, comps, r.spec_limit, c.nominals)
ui.section("What it takes")
cols = st.columns(3)
with cols[0]:
    with st.container(border=True):
        if need is None:
            st.markdown(f"**Placement alone cannot reach {r.fpy_target:g}%** — even a perfect placement gives "
                        f"{ceil['perfect_placement']:.1f}%. The shape of the modules must improve too.")
        elif need == 0:
            st.markdown(f"**The target is reached {'with the correction alone' if comp_on else 'already'}.**")
        else:
            st.markdown(f"Reduce the placement scatter (shift and yaw together) by **{need * 100:.0f}%**"
                        f"{' on top of the correction' if comp_on else ''} to reach **{r.fpy_target:g}%**.")
with cols[1]:
    with st.container(border=True):
        st.markdown(f"**Ceiling with perfect placement:** {ceil['perfect_placement']:.1f}%  \n"
                    f"Every module placed exactly at nominal, with its own shape.")
with cols[2]:
    with st.container(border=True):
        st.markdown(f"**Ceiling with perfect modules:** {ceil['perfect_shape']:.1f}%  \n"
                    f"Every module perfectly square, placed as it was.")

ui.section("Sensitivity", "Predicted FPY as each kind of scatter is reduced on its own"
                          + (" (with the correction applied)." if comp_on else "."))
sig = repr(sorted((k, tuple(np.round(v, 4))) for k, v in comps.items()))
curves = data.sensitivity(c.key, sig, bool(comp_on), a, comps)
ui.chart(ccmp.sensitivity(curves, r.fpy_target), "tg_sens")
st.caption("How it works: every complete module's corner errors are split into shift, yaw and shape; a lever "
           "shrinks one of them towards its mean (or moves the mean, for the correction); the modified module is "
           "judged again against the tolerance. Incomplete first runs stay NOK.")
