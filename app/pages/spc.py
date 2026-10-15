import streamlit as st

from app import data, state, ui
from rickhard import spc
from rickhard.charts import spc_charts as csp
from rickhard.constants import DEV_COLS

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "spc", "Is the placement in statistical control? Individuals and weekly charts with the run rules "
                         "used in automotive SPC, and the dates where the process level shifted.")
ui.require(a.first)

# ---- Alarm board ------------------------------------------------------------------------------
state.init("p_spc_days", 14)
board = data.alarms(c.key, int(st.session_state["p_spc_days"]), a)
ui.section("Alarm board", "Did anything change recently? The last N days are compared with a baseline (the data "
                          "since the last detected shift before the window): the level (Welch t-test, ≥ 0.25σ), the "
                          "scatter (Brown–Forsythe, ≥ ×1.25) and points beyond 3σ (vs 0.27 % by chance). "
                          "Signal: p < 0.001 · Watch: p < 0.05.")
cfg = st.column_config
if len(board):
    show = board.copy()
    show["State"] = show["State"].map({"Signal": "🔴 Signal", "Watch": "🟡 Watch", "In control": "🟢 In control"})
    st.dataframe(show[["State", "BatteryType", "Variable", "Reason", "N recent", "Baseline from", "Baseline", "Recent",
                       "Shift σ", "Spread ratio", "R1 recent", "R1 expected"]], hide_index=True, width="stretch",
                 column_config={
                     "BatteryType": "Type", "Reason": cfg.TextColumn("What changed", width="large"),
                     "Baseline from": cfg.DatetimeColumn(format="YYYY-MM-DD"),
                     "Baseline": cfg.NumberColumn("Baseline level", format="%+.3f"),
                     "Recent": cfg.NumberColumn("Recent level", format="%+.3f"),
                     "Shift σ": cfg.NumberColumn("Shift (σ)", format="%+.2f"),
                     "Spread ratio": cfg.NumberColumn("Scatter ratio", format="%.2f"),
                     "R1 recent": cfg.NumberColumn("Beyond 3σ"), "R1 expected": cfg.NumberColumn("expected", format="%.1f")})
else:
    st.caption("Not enough data for the alarm board (needs a baseline of 30 modules and 5 recent ones per type).")
st.slider("Recent window (days)", 3, 60, key="p_spc_days")

# ---- Chart controls ---------------------------------------------------------------------------
var_opts = {"Shift X (fit)": ("Fit_Tx", "mm"), "Shift Y (fit)": ("Fit_Ty", "mm"), "Yaw (fit)": ("Fit_Rot", "°"),
            "Worst corner |dev|": ("WorstAbs", "mm"), "Shape residual": ("ShapeMax", "mm")}
var_opts.update({c_.replace("_", " "): (c_, "mm") for c_ in DEV_COLS})
if a.has_z and "Z_Level" in a.first:
    var_opts.update({"Z level": ("Z_Level", "mm"), "Z pitch": ("Z_Pitch", "mm/m"), "Z roll": ("Z_Roll", "mm/m")})
types = c.types
ctl = st.columns([1.1, 1.6, 1.6], gap="medium", vertical_alignment="bottom")
with ctl[0]:
    if len(types) > 1:
        state.ensure_option("p_spc_type", types, types[0])
        t = st.segmented_control("Battery type", types, key="p_spc_type") or types[0]
    else:
        t = types[0]
with ctl[1]:
    state.ensure_option("p_spc_var", list(var_opts), "Shift X (fit)")
    var = st.selectbox("Variable", list(var_opts), key="p_spc_var")
with ctl[2]:
    bases = ["All data", "Since the last detected shift", "First 50 modules"]
    state.ensure_option("p_spc_base", bases, "Since the last detected shift")
    base_mode = st.selectbox("Control limits from", bases, key="p_spc_base",
                             help="Phase I (all data) shows every past excursion. Limits from the current level "
                                  "(since the last shift) show whether today's process is stable.")

col, unit = var_opts[var]
s = a.first[a.first["BatteryType"] == t].sort_values("Date")[["Date", "PartID", "Status", col]].dropna()
if len(s) < 10:
    st.info("Not enough values for a control chart.")
    st.stop()
y = s[col].to_numpy(float)
cps = spc.change_points(y, min_size=max(12, len(y) // 40))
if base_mode == "Since the last detected shift" and cps:
    baseline = slice(cps[-1], None)
    week_base = (s["Date"].iloc[cps[-1]], None)
elif base_mode == "First 50 modules":
    baseline = slice(0, min(50, len(y)))
    week_base = (None, s["Date"].iloc[min(50, len(y)) - 1])
else:
    baseline = week_base = None
chart = spc.imr(y, baseline=baseline)
spec = (-r.spec_limit, r.spec_limit) if col in DEV_COLS else None
labels = [f"{p} · {d:%Y-%m-%d %H:%M} · {st_}" for p, d, st_ in zip(s["PartID"], s["Date"], s["Status"])]
ev = c.events
ev_t = ev[ev["Scope"].isin(["All", t])] if len(ev) else ev
ui.section(f"I-MR chart · {t} · {var}")
ui.chart(csp.imr_chart(chart, s["Date"], labels, var, unit, cps, spec, ev_t), "spc_imr")
flags = chart.flags
counts = {spc.RULE_SHORT[k]: int(v.sum()) for k, v in flags.items()}
m = st.columns(len(counts) + 1)
m[0].metric("Points", f"{len(y):,}", border=True)
for i, (k, v) in enumerate(counts.items(), start=1):
    m[i].metric(k, v, border=True)
st.caption(csp.rules_legend())

xb = spc.xbar_s(a.first[a.first["BatteryType"] == t], col, baseline=week_base)
if xb is not None:
    ui.section("Weekly means (X̄ chart)", "Each point is a week's mean; the limits narrow with the number of modules. "
                                         "Centre and σ come from the same modules as the I-MR limits (Control "
                                         "limits from).")
    ui.chart(csp.xbar_chart(xb[0], xb[1], var, unit), "spc_xbar")

if cps:
    tab = spc.describe_change_points(s["Date"], y, cps, var)
    st.dataframe(tab[["Date", "Before", "After", "Shift", "p", "N before", "N after"]], hide_index=True,
                 width="stretch", column_config={"Date": cfg.DatetimeColumn("Shift from", format="YYYY-MM-DD HH:mm"),
                                                 "Before": cfg.NumberColumn(format="%+.3f"),
                                                 "After": cfg.NumberColumn(format="%+.3f"),
                                                 "Shift": cfg.NumberColumn(format="%+.3f"),
                                                 "p": cfg.NumberColumn("p-value", format="%.1e")})
