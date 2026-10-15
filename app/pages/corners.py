import streamlit as st

from app import data, state, ui
from rickhard.charts import corners as cc_charts
from rickhard.charts import theme as T
from rickhard.constants import CORNERS
from rickhard.metrics import capability_series

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "corners", "Where every corner lands, how capable each corner/axis is, and whether its problem is an "
                             "offset (fixable by a correction) or scatter (find the source of variation).")
ui.require(a.first)

types = c.types
ctl = st.columns([1.2, 1.6, 1.2], gap="medium", vertical_alignment="bottom")
with ctl[0]:
    if len(types) > 1:
        state.ensure_option("p_cn_type", types, types[0])
        t = st.segmented_control("Battery type", types, key="p_cn_type") or types[0]
    else:
        t = types[0]
        st.markdown(f"**{t}**")
with ctl[1]:
    opts = ["Status", "Week", "Shift"]
    state.ensure_option("p_cn_color", opts, "Status")
    color_by = st.segmented_control("Colour points by", opts, key="p_cn_color") or "Status"
with ctl[2]:
    state.init("p_cn_last", 0)
    last_n = st.number_input("Only the last N modules (0 = all)", 0, 100000, step=50, key="p_cn_last")

d = a.first[a.first["BatteryType"] == t]
if last_n:
    d = d.tail(int(last_n))

ui.section(f"Corner targets · {t}", "Each panel sits where its corner is on the module (front to the right, left side "
                                     "on top). Dashed box: tolerance. Black arrow: mean error. Blue: 2σ ellipse — its "
                                     "size is the scatter, its centre the offset.")
ui.chart(cc_charts.target_plots(d, r.spec_limit, color_by=color_by), "cn_target")

# ---- Capability -------------------------------------------------------------------------------
cap = a.capability
left, right = st.columns([2, 3], gap="large")
with left:
    ui.section("Capability per corner and axis", "Ppk = distance from the mean to the nearer limit in units of 3σ "
                                                 "(long-term σ). 'Offset': the mean is ≥ ¼ of the tolerance off nominal — "
                                                 "a correction helps. 'Scatter': Pp < 1 — the spread itself is too wide.")
    ui.chart(cc_charts.capability_grid(cap), "cn_capgrid")
with right:
    coords = [f"{k} {ax}" for k in CORNERS for ax in ("X", "Y")]
    worst = cap[(cap["BatteryType"] == t) & cap["Axis"].isin(["X", "Y"])].sort_values("Ppk")
    default = f"{worst['Corner'].iloc[0]} {worst['Axis'].iloc[0]}" if len(worst) else coords[0]
    state.ensure_option("p_cn_coord", coords, default)
    coord = st.selectbox("Distribution of", coords, key="p_cn_coord")
    col = coord.replace(" ", "_")
    stats = capability_series(d.sort_values("Date")[col].to_numpy(float), -r.spec_limit, r.spec_limit)
    ui.chart(cc_charts.capability_hist(d[col], -r.spec_limit, r.spec_limit, f"{t} {coord}", stats,
                                       color=T.pal().type(t)), "cn_hist")

with st.expander("Capability table", icon=":material/table_chart:"):
    show = cap[["BatteryType", "Corner", "Axis", "N", "Mean", "Std", "StdWithin", "Cp", "Cpk", "Pp", "Ppk", "PpkLo",
                "PpkHi", "PPM", "OutPct", "Diagnosis"]]
    cfg = st.column_config
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "BatteryType": "Type", "Mean": cfg.NumberColumn("Mean", format="%+.2f"),
        "Std": cfg.NumberColumn("σ overall", format="%.2f"), "StdWithin": cfg.NumberColumn("σ within", format="%.2f"),
        "Cp": cfg.NumberColumn(format="%.2f"), "Cpk": cfg.NumberColumn(format="%.2f"),
        "Pp": cfg.NumberColumn(format="%.2f"), "Ppk": cfg.NumberColumn(format="%.2f"),
        "PpkLo": cfg.NumberColumn("Ppk 95% low", format="%.2f"), "PpkHi": cfg.NumberColumn("Ppk 95% high", format="%.2f"),
        "PPM": cfg.NumberColumn("Expected PPM", format="%.0f"), "OutPct": cfg.NumberColumn("Out of tol.", format="%.1f%%"),
    })
    st.caption("Cp/Cpk use the short-term σ from consecutive modules (moving range); Pp/Ppk the overall σ. "
               "A big gap between them means the process moves over time (drift, shifts between weeks).")

# ---- Trends -------------------------------------------------------------------------------------
ui.section(f"Corner deviation trend · {t}", "The original trend matrix: every module in time order, X and Y, with a "
                                             "rolling mean and the tolerance.")
state.init("p_cn_win", 15)
win = st.slider("Rolling mean window (modules)", 5, 101, step=2, key="p_cn_win")
ui.chart(cc_charts.corner_trends(d, r.spec_limit, t, window=int(win)), "cn_trends")

with st.expander("Which corners move together?", icon=":material/hub:"):
    st.caption("Correlation between the eight corner coordinates. Corners that move together point to a placement "
               "cause (the whole module moves); independent corners point to the corners themselves or to the shape.")
    ui.chart(cc_charts.correlation(d, t), "cn_corr")
