import numpy as np
import streamlit as st

from app import data, state, ui
from rickhard import patterns
from rickhard.charts import patterns_charts as cpt
from rickhard.stats import significance_label, welch

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "patterns", "When do modules fail? Differences between shifts, weekdays and hours, and the first "
                              "modules after the line stood still.")
ui.require(a.first)
f = a.first
shift_names = list(c.settings.shifts.names)

# ---- Shifts -------------------------------------------------------------------------------------
p_fpy = patterns.group_test(f, "Shift")
p_x = patterns.group_mean_test(f, "Shift", "Fit_Tx")
p_y = patterns.group_mean_test(f, "Shift", "Fit_Ty")
ui.section("Shifts", "Shift plan: " + " · ".join(c.settings.shifts.describe()) + " (editable in Data & Settings).")
left, right = st.columns(2, gap="large")
with left:
    ui.chart(cpt.fpy_by_group(patterns.by_group(f, "Shift", shift_names), "Shift", r.fpy_target, xlabel="Shift"),
             "pt_shift")
    st.caption(f"Does FPY depend on the shift? Chi-square {significance_label(p_fpy)}.")
with right:
    ui.chart(cpt.placement_by_group(f, "Shift", shift_names, xlabel="Shift"), "pt_shift_pl")
    st.caption(f"Does the placement differ between shifts? Kruskal–Wallis: shift X {significance_label(p_x)}, "
               f"shift Y {significance_label(p_y)}.")

# ---- Weekday x hour -----------------------------------------------------------------------------
ui.section("Weekday × hour", "Colour: NOK rate; numbers: modules produced in that hour (summed over the period).")
left, right = st.columns([3, 2], gap="large")
with left:
    ui.chart(cpt.heatmap_week_hour(patterns.heatmap(f), patterns.heatmap(f, "volume")), "pt_heat")
with right:
    wd = patterns.by_group(f, "WeekdayName", patterns.WEEKDAYS)
    ui.chart(cpt.fpy_by_group(wd, "WeekdayName", r.fpy_target, xlabel="Weekday", height=300), "pt_wd")
    st.caption(f"Weekday effect on FPY: chi-square {significance_label(patterns.group_test(f, 'WeekdayName'))}.")

var_opts = {"Shift X (fit)": ("Fit_Tx", "mm"), "Shift Y (fit)": ("Fit_Ty", "mm"), "Yaw (fit)": ("Fit_Rot", "°"),
            "Worst corner |dev|": ("WorstAbs", "mm")}
state.ensure_option("p_pt_var", list(var_opts), "Shift Y (fit)")
var = st.selectbox("Hour-of-day profile of", list(var_opts), key="p_pt_var")
col, unit = var_opts[var]
ui.chart(cpt.hourly_profile(patterns.hourly_profile(f, col), var, unit), "pt_hour")

# ---- Start-up -----------------------------------------------------------------------------------
ui.section("Start-up after line stops", "Are the first modules after a stop worse? A stop is a gap without any "
                                        "measured module.")
ctl = st.columns(3, vertical_alignment="bottom")
with ctl[0]:
    state.init("p_pt_gap", float(patterns.suggest_stop_gap(f)))
    gap = st.slider("A stop is a gap of at least (min)", 20.0, 480.0, step=10.0, key="p_pt_gap")
with ctl[1]:
    state.init("p_pt_k", 3)
    k = st.slider("First N modules after a stop", 1, 10, key="p_pt_k")
with ctl[2]:
    state.ensure_option("p_pt_svar", list(var_opts), "Shift X (fit)")
    svar = st.selectbox("Compare", list(var_opts), key="p_pt_svar")
eff = patterns.startup_effect(f, gap, int(k))
flags = patterns.startup_flags(f, gap, int(k))
scol, sunit = var_opts[svar]
if eff:
    d = eff["fpy_startup"] - eff["fpy_steady"]
    vals = f[scol].to_numpy(float)
    vdiff, vp = welch(vals[~flags], vals[flags]) if flags.any() and (~flags).any() else (np.nan, np.nan)
    ui.kpi_cards([
        {"label": "FPY · first modules after a stop", "value": ui.fmt_pct(eff["fpy_startup"]),
         "sub": f"{eff['n_startup']:,} modules"},
        {"label": "FPY · steady running", "value": ui.fmt_pct(eff["fpy_steady"]), "sub": f"{eff['n_steady']:,} modules"},
        {"label": "Start-up vs steady (FPY)", "value": f"{d:+.1f} pp", "sub": significance_label(eff["p_value"]),
         "good": None},
        {"label": f"Start-up vs steady ({svar})",
         "value": f"{vdiff:+.2f}" if np.isfinite(vdiff) else "—", "unit": sunit if np.isfinite(vdiff) else "",
         "sub": significance_label(vp)},
    ])
ui.chart(cpt.startup_compare(f, flags, scol, svar, sunit), "pt_startup")
st.caption("If start-up modules land differently, look at what changes after a stop: end-stop damping or "
           "temperature, conveyor acceleration, the first-part routine.")
