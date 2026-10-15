import streamlit as st

from app import data, state, ui
from rickhard import improvement as imp
from rickhard.charts import anatomy as ca
from rickhard.constants import MODE_LABELS, MODES

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "anatomy", "Every module's eight corner deviations are written exactly as the sum of eight physical "
                             "modes. Their sizes tell you where the error comes from — and which part a placement "
                             "correction can remove.")
ui.require(a.first)
ui.focus_note(c)

types = c.types
if len(types) > 1:
    state.ensure_option("p_an_type", types, types[0])
    t = st.segmented_control("Battery type", types, key="p_an_type") or types[0]
else:
    t = types[0]

bud = imp.budget(a.first)
b = bud[bud["BatteryType"] == t]
if b.empty:
    st.info("Not enough complete modules of this type.")
    st.stop()
place_var = b[b["Mode"].isin(["Tx", "Ty", "Rot"])]["VarShare"].sum()
shape_var = b[b["Mode"].isin(["Shear", "TrapX", "TrapY"])]["VarShare"].sum()
size_var = b[b["Mode"].isin(["ScaleX", "ScaleY"])]["VarShare"].sum()
top = b.sort_values("VarShare", ascending=False).iloc[0]
place_bias = b[b["Mode"].isin(["Tx", "Ty", "Rot"])]["BiasShare"].sum()
ui.kpi_cards([
    {"label": "Scatter from placement", "value": f"{place_var:.0f}%", "sub": "shift X + shift Y + yaw",
     "accent": "#2a78d6"},
    {"label": "Scatter from size", "value": f"{size_var:.0f}%", "sub": "length and width stretch", "accent": "#1baf7a"},
    {"label": "Scatter from shape", "value": f"{shape_var:.0f}%", "sub": "parallelogram + trapezoids", "accent": "#eb6834"},
    {"label": "Largest single source", "value": MODE_LABELS[top["Mode"]], "sub": f"{top['VarShare']:.0f}% of the scatter"},
    {"label": "Systematic error from placement", "value": f"{place_bias:.0f}%",
     "sub": "the part a fixed X / Y / yaw correction removes"},
])

ui.section("Error budget", "Left: the systematic part of each mode (the mean over modules) — a fixed correction "
                           "removes it. Right: the random part (module-to-module σ) — only reducing variation "
                           "removes it. Amplitudes are RMS mm at the four corners, so they add up as variances.")
ui.chart(ca.budget_bars(bud, t, height=400), "an_budget")

left, right = st.columns(2, gap="large")
with left:
    ui.section("Data-driven modes (PCA)", "The directions in which this process actually varies, and which physical "
                                          "modes they are made of.")
    ui.chart(ca.pca_components(imp.pca(a.first, t, c.nominals), t, c.nominals, title=None, height=340), "an_pca")
with right:
    ui.section("Does the mode drive NOK?", "NOK rate of the quarter of modules where a mode deviates least from "
                                           "its average vs the quarter where it deviates most. A big gap means "
                                           "that mode's variation pushes modules out.")
    ui.chart(ca.mode_link(imp.mode_fail_link(a.first), t, height=340), "an_link")

ui.section("What each mode does to the module", "Every mode drawn at |mean| + 2σ of this process, magnified.")
state.init("p_an_mag", 120.0)
mag = st.slider("Magnify ×", 20.0, 400.0, step=10.0, key="p_an_mag")
ui.chart(ca.mode_gallery(bud, t, c.nominals, mag), "an_gallery")

ui.section("Mode trends per week")
labels = {MODE_LABELS[m]: m for m in MODES}
state.init("p_an_modes", ["Shift X", "Shift Y", "Yaw rotation"])
sel = st.multiselect("Modes", list(labels), key="p_an_modes", max_selections=4)
if sel:
    ui.chart(ca.mode_trends(imp.weekly_modes(a.first), t, [labels[s] for s in sel]), "an_trends")

with st.expander("NOK modules and their mechanism", icon=":material/table_rows:"):
    nok = a.first[(a.first["BatteryType"] == t) & (a.first["Status"] != "PASS")]
    tbl = nok.sort_values("Date", ascending=False)[["_mod_key", "Date", "PartID", "Mechanism", "MechanismDetail",
                                                    "ShareShift", "ShareYaw", "ShareShape", "WorstCoord", "WorstAbs",
                                                    "Fit_Tx", "Fit_Ty", "Fit_Rot", "ShapeMax"]].reset_index(drop=True)
    for col in ("ShareShift", "ShareYaw", "ShareShape"):
        tbl[col] = tbl[col] * 100
    cfg = st.column_config
    ev = st.dataframe(tbl.drop(columns="_mod_key"), hide_index=True, width="stretch", height=360, on_select="rerun",
                      selection_mode="single-row", key="an_table", column_config={
                          "Date": cfg.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
                          "MechanismDetail": cfg.TextColumn("Detail"),
                          "ShareShift": cfg.ProgressColumn("Shift share", format="%.0f%%", min_value=0, max_value=100),
                          "ShareYaw": cfg.ProgressColumn("Yaw share", format="%.0f%%", min_value=0, max_value=100),
                          "ShareShape": cfg.ProgressColumn("Shape share", format="%.0f%%", min_value=0, max_value=100),
                          "WorstCoord": cfg.TextColumn("Worst coord."), "WorstAbs": cfg.NumberColumn("Worst |dev|", format="%.2f"),
                          "Fit_Tx": cfg.NumberColumn("Shift X", format="%+.2f"), "Fit_Ty": cfg.NumberColumn("Shift Y", format="%+.2f"),
                          "Fit_Rot": cfg.NumberColumn("Yaw °", format="%+.3f"), "ShapeMax": cfg.NumberColumn("Shape max", format="%.2f")})
    i = ui.selected_row(ev)
    if i is not None and i < len(tbl) and tbl.loc[i, "_mod_key"] != state.focus():
        state.set_focus(tbl.loc[i, "_mod_key"])
        st.rerun()
    st.caption("Shares: how much of the excess on the failing coordinates each part pushes outwards.")

with st.expander("How the decomposition works", icon=":material/school:"):
    st.markdown(
        "For every battery type the eight corner coordinates (FL X, FL Y, … RR Y) are projected on an orthonormal "
        "basis built from the nominal geometry, in this order: **shift X, shift Y, yaw** (a rigid move), **length and "
        "width stretch** (size), **parallelogram** and the two **trapezoids** (shape). Because the basis is "
        "orthonormal, the decomposition is exact (the eight modes rebuild the measurement to the micron) and the "
        "variances of the modes add up to the total — that is what makes the shares above meaningful. Shift and yaw "
        "are exactly the least-squares rigid placement of the module; everything else is shape that no conveyor "
        "correction can remove.")
