import numpy as np
import pandas as pd
import streamlit as st

from app import data, state, ui
from rickhard import zheight
from rickhard.charts import msa_charts as cm
from rickhard.constants import Z_COLS

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "height", "The height (Z) of the four corners: level, pitch, roll and twist of every module. "
                            "Z is reported for insight; it decides PASS/FAIL only if you switch it on in the sidebar.")
ui.require(a.first)
f = a.first
if not a.has_z or "Z_Level" not in f:
    st.info("This export has no Z deviation column.")
    st.stop()

zl, zh = r.z_lo, r.z_hi
Z = f[list(Z_COLS)].to_numpy(float)
with np.errstate(invalid="ignore"):
    out = ((Z < zl) | (Z > zh)) & np.isfinite(Z)
mod_out = out.any(axis=1)
ui.kpi_cards([
    {"label": "Modules with a corner out of Z limits", "value": ui.fmt_pct(mod_out.mean() * 100),
     "sub": f"{ui.plural(int(mod_out.sum()), 'module')} · limits {zl:+g} / {zh:+g} mm"},
    {"label": "Mean level", "value": ui.fmt_mm(np.nanmean(f["Z_Level"]), 2), "sub": f"σ {np.nanstd(f['Z_Level']):.2f} mm"},
    {"label": "Mean pitch (front ↔ rear)", "value": f"{np.nanmean(f['Z_Pitch']):+.2f}", "unit": "mm/m",
     "sub": f"σ {np.nanstd(f['Z_Pitch']):.2f} mm/m"},
    {"label": "Mean roll (left ↔ right)", "value": f"{np.nanmean(f['Z_Roll']):+.2f}", "unit": "mm/m",
     "sub": f"σ {np.nanstd(f['Z_Roll']):.2f} mm/m"},
    {"label": "Twist", "value": f"{np.nanmean(np.abs(f['Z_Twist'])):.2f}", "unit": "mm", "sub": "mean |twist| at the corners"},
])

left, right = st.columns([2, 3], gap="large")
with left:
    types = c.types
    if len(types) > 1:
        state.ensure_option("p_z_type", types, types[0])
        t = st.segmented_control("Battery type", types, key="p_z_type") or types[0]
    else:
        t = types[0]
    ui.section(f"Average surface · {t}", "Fitted through the mean corner heights (exact for four corners); the "
                                        "vertical scale is exaggerated.")
    ui.chart(cm.z_surface(zheight.mean_surface(f, t, c.nominals), t), "z_surf")
with right:
    ui.section("Z per corner")
    ui.chart(cm.z_box(f, (zl, zh)), "z_box")

ui.section("Level, pitch, roll and twist per week", "Mean ± 1σ per week. Pitch and roll in mm per metre.")
ui.chart(cm.z_trends(f), "z_trends")

with st.expander("Does Z relate to X/Y failures?", icon=":material/insights:"):
    d = f[f["Status"].isin(["PASS", "FAIL"])]
    rows = []
    for col, lab in (("Z_Level", "Level"), ("Z_Pitch", "Pitch"), ("Z_Roll", "Roll"), ("Z_Twist", "Twist")):
        ok, nok = d.loc[d["Status"] == "PASS", col], d.loc[d["Status"] == "FAIL", col]
        rows.append({"Measure": lab, "OK modules (mean)": ok.mean(), "NOK modules (mean)": nok.mean(),
                     "Difference": nok.mean() - ok.mean()})
    cfg = st.column_config
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "OK modules (mean)": cfg.NumberColumn(format="%+.3f"), "NOK modules (mean)": cfg.NumberColumn(format="%+.3f"),
        "Difference": cfg.NumberColumn(format="%+.3f")})
    st.caption("A module that sits tilted or raised can be measured shifted in X/Y; a clear difference here points "
               "to seating (a chip, a worn support) rather than conveyor placement.")
