import numpy as np
import pandas as pd
import streamlit as st

from app import data, state, ui
from rickhard import compensation as comp
from rickhard.charts import comp_charts as ccmp
from rickhard.charts import svg

c = data.ctx()
a, r = c.a, c.rules
lim, nom = r.spec_limit, c.nominals
ui.page_header(c, "compensation", "A rigid X / Y / yaw correction of the module placement (rotation about the nominal "
                                  "module centre), simulated on every first measurement. Nothing here changes the data.")
ui.require(a.first)
ui.focus_note(c)

# ---- Controls ---------------------------------------------------------------------------------
types = c.types
ctl = st.columns([1.2, 1.0, 1.6, 0.9], gap="medium", vertical_alignment="bottom")
with ctl[0]:
    if len(types) > 1:
        state.ensure_option("p_cp_type", types, types[0])
        t = st.segmented_control("Battery type", types, key="p_cp_type") or types[0]
    else:
        t = types[0]
        st.markdown(f"**{t}**")
vt_all = a.first[a.first["BatteryType"] == t]
n_weeks_avail = max(1, int(vt_all["WeekKey"].nunique()))
with ctl[1]:
    state.init("p_cp_weeks", min(6, n_weeks_avail))
    state.clamp("p_cp_weeks", 1, n_weeks_avail, min(6, n_weeks_avail))
    weeks = st.number_input("Last N weeks", 1, n_weeks_avail, step=1, key="p_cp_weeks",
                            help="The analysis window, counted back from the last week of the sidebar range.")
with ctl[2]:
    state.init("p_cp_sigma", 0.10)
    sigma = st.slider("Robustness: placement noise to tolerate (mm)", 0.0, 0.40, step=0.02, key="p_cp_sigma",
                      help="The robust optimum maximises the FPY you would still get if every module landed this much "
                           "off (normally distributed). 0 = the exact best setting on these modules, which tends to sit "
                           "on a fragile edge.")
with ctl[3]:
    with st.popover("More", icon=":material/settings:", width="stretch"):
        state.init("p_cp_sigyaw", 0.01)
        sigma_yaw = st.slider("Yaw noise to tolerate (°)", 0.0, 0.05, step=0.005, format="%.3f", key="p_cp_sigyaw")
        state.ensure_option("p_cp_method", ["Median", "Mean"], "Median")
        mname = st.radio("Median-based method uses", ["Median", "Mean"], horizontal=True, key="p_cp_method")
        state.init("p_cp_minn", 40)
        min_n = st.number_input("Minimum modules to trust a result", 5, 1000, step=5, key="p_cp_minn")
        state.init("p_cp_margin", 0.2)
        margin = st.slider("Safety margin to report (mm)", 0.0, 1.0, step=0.05, key="p_cp_margin",
                           help="The methods table also shows how many modules would pass with at least this much "
                                "room on every corner — a setting that only passes modules at the edge will not "
                                "hold up.")
method_key = "mean" if mname == "Mean" else "median"

v = data.window(a, t, int(weeks))
ui.require(v)
sug = data.suggestions(c.key, t, int(weeks), float(sigma), float(sigma_yaw), method_key, a)
settings = dict(sug["settings"])
opt = sug["optimum"]
st.caption(f"Window: **{sug['label']}** · {len(v):,} first-run modules of {t}"
           + (f" · :orange[fewer than {min_n} modules — treat the result with caution]" if len(v) < min_n else ""))

# ---- Manual ------------------------------------------------------------------------------------
kx, ky, kyaw = f"p_man_x_{t}", f"p_man_y_{t}", f"p_man_yaw_{t}"
base = settings.get("Robust optimum", settings.get("Rigid fit (mean)", (0.0, 0.0, 0.0)))
state.init(kx, round(float(base[0]), 2))
state.init(ky, round(float(base[1]), 2))
state.init(kyaw, round(float(base[2]), 3))


def _best_yaw():
    yaws = np.round(np.arange(-0.4, 0.4001, 0.001), 4)
    f_ = comp.fpy_vs_yaw(v, st.session_state[kx], st.session_state[ky], yaws, lim, nom)
    best = yaws[np.flatnonzero(f_ == np.nanmax(f_))]
    st.session_state[kyaw] = float(best[np.argmin(np.abs(best - st.session_state[kyaw]))])


def _copy_from(m):
    s = settings.get(m)
    if s:
        st.session_state[kx], st.session_state[ky], st.session_state[kyaw] = round(s[0], 2), round(s[1], 2), round(s[2], 3)


with st.expander("Manual values — try your own X / Y / yaw", icon=":material/edit:"):
    m1, m2, m3, m4 = st.columns([1, 1, 1, 1.4], vertical_alignment="bottom")
    m1.number_input("X (mm)", step=0.01, format="%.2f", key=kx)
    m2.number_input("Y (mm)", step=0.01, format="%.2f", key=ky)
    m3.number_input("Yaw (°)", step=0.001, format="%.3f", key=kyaw)
    with m4:
        st.button("Best yaw for this X / Y", icon=":material/rotate_right:", on_click=_best_yaw, width="stretch",
                  help="Scans the yaw at fixed X/Y and picks the value with the highest FPY (closest to the current one).")
    b1, b2, b3 = st.columns(3)
    b1.button("Start from the robust optimum", on_click=_copy_from, args=("Robust optimum",), width="stretch")
    b2.button("Start from median-based", on_click=_copy_from, args=("Median-based",), width="stretch")
    b3.button("Start from rigid fit", on_click=_copy_from, args=("Rigid fit (mean)",), width="stretch")
settings["Manual"] = (float(st.session_state[kx]), float(st.session_state[ky]), float(st.session_state[kyaw]))

# ---- Methods table -------------------------------------------------------------------------------
bt = data.backtest(c.key, t, int(weeks), float(sigma), float(sigma_yaw), method_key, a)
bts = comp.backtest_summary(bt)
rows = []
none_fpy = comp.evaluate(v, (0.0, 0.0, 0.0), lim, nom)["fpy"]
for m, s in settings.items():
    ev_ = comp.evaluate(v, s, lim, nom, margin=float(margin))
    rows.append({"Method": m, "X [mm]": s[0], "Y [mm]": s[1], "Yaw [°]": s[2], "In window": ev_["fpy"],
                 "With margin": ev_["fpy_margin"], "Δ vs none": ev_["fpy"] - none_fpy, "Backtest": bts.get(m, np.nan)})
mt = pd.DataFrame(rows)
ui.section("Methods compared", "In window: FPY on the same modules the setting was computed from (optimistic). "
                               "Backtest: each recent week gets the setting computed from the weeks before it — the "
                               "honest estimate of what the line would see.")
cfg = st.column_config
st.dataframe(mt, hide_index=True, width="stretch", column_config={
    "X [mm]": cfg.NumberColumn(format="%+.3f"), "Y [mm]": cfg.NumberColumn(format="%+.3f"),
    "Yaw [°]": cfg.NumberColumn(format="%+.4f"),
    "In window": cfg.ProgressColumn("FPY in window", format="%.1f%%", min_value=0, max_value=100),
    "With margin": cfg.ProgressColumn(f"Pass with {margin:.2f} mm room", format="%.1f%%", min_value=0,
                                      max_value=100,
                                      help="Modules that would still pass with every corner limit tightened by the "
                                           "safety margin (set it in More)."),
    "Δ vs none": cfg.NumberColumn("Δ vs none (pp)", format="%+.1f"),
    "Backtest": cfg.NumberColumn("FPY backtest", format="%.1f%%",
                                 help="Each recent week judged with the setting this method computes from the "
                                      "weeks before it. Manual values are not computed from data, so they have "
                                      "no backtest (empty).")})

choices = [m for m in ("Robust optimum", "Rigid fit (mean)", "Median-based", "Manual") if m in settings]
state.ensure_option(f"p_cp_apply_{t}", choices, choices[0])
applied_m = st.segmented_control("Show results for", choices, key=f"p_cp_apply_{t}") or choices[0]
dx, dy, yaw = settings[applied_m]

ev_none = comp.evaluate(v, (0.0, 0.0, 0.0), lim, nom)
ev_app = comp.evaluate(v, (dx, dy, yaw), lim, nom)
boot = comp.bootstrap_gain(v, (dx, dy, yaw), lim, nom)
gain = ev_app["fpy"] - ev_none["fpy"]
ui.kpi_cards([
    {"label": f"{applied_m}", "value": f"X {dx:+.2f} · Y {dy:+.2f}", "unit": "mm", "sub": f"yaw **{yaw:+.3f}°**",
     "accent": "#6250d6"},
    {"label": "FPY in window", "value": f"{ev_none['fpy']:.1f}% → {ev_app['fpy']:.1f}%", "delta": f"{gain:+.1f} pp",
     "good": gain >= 0},
    {"label": "Gain · 90% bootstrap interval", "value": f"{boot.get('lo', np.nan):+.1f} … {boot.get('hi', np.nan):+.1f}",
     "unit": "pp", "sub": f"P(gain > 0) = {boot.get('p_positive', np.nan) * 100:.0f}%" if boot else "too few modules"},
    {"label": "Modules fixed / broken", "value": f"{boot.get('fixed', 0)} / {boot.get('broken', 0)}",
     "sub": "NOK → OK vs OK → NOK in the window"},
    {"label": "Backtest of this method", "value": ui.fmt_pct(bts.get(applied_m, np.nan)) if applied_m != "Manual" else "–",
     "sub": (f"vs {bts.get('No compensation', np.nan):.1f}% without · {bts.get('label', '')}" if bts else "needs more weeks")},
])

# ---- Work instruction --------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")
with left:
    ui.section("Work instruction for the line", "Where each corner must move, magnified. The arrows are the "
                                                "equivalent per-corner offsets of the rigid correction.")
    ui.svg_box(svg.work_instruction_svg(t, dx, dy, yaw, nom), "Work instruction: where each corner must move")
with right:
    ui.section("Equivalent corner offsets")
    st.dataframe(comp.equivalent_offsets_table({t: (dx, dy, yaw)}, nom), hide_index=True, width="stretch",
                 column_config={"Offset X [mm]": cfg.NumberColumn(format="%+.3f"),
                                "Offset Y [mm]": cfg.NumberColumn(format="%+.3f")})
    st.caption("Convention: positive X moves the module towards the front, positive Y towards the right side, "
               "positive yaw turns +X towards +Y. Check the sign convention of your conveyor before applying.")
    if st.button("Use in Path to Target", icon=":material/flag:", width="stretch"):
        data.set_chosen_compensation(t, (dx, dy, yaw), applied_m, int(weeks))
        st.toast(f"{t}: {applied_m} correction will be used in Path to Target.", icon=":material/check:")
    if st.button("Log as intervention", icon=":material/event_note:", width="stretch",
                 help="Adds an entry to the Interventions log at the end of the data, so you can measure the effect "
                      "once new data arrives."):
        state.add_event({"Date": pd.Timestamp(a.first["Date"].max()).isoformat(), "Scope": t, "Category": "Adjustment",
                         "Title": f"Compensation {applied_m}: X {dx:+.2f}, Y {dy:+.2f}, yaw {yaw:+.3f}°",
                         "Notes": f"From Compensation Studio, window {sug['label']}", "dX": dx, "dY": dy, "dYaw": yaw})
        st.toast("Added to the Interventions log.", icon=":material/check:")

# ---- Landscape and yaw ----------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")
with left:
    ui.section("FPY for every X / Y setting", "At the yaw of the selected method. Darker = more modules pass. The "
                                               "robust optimum sits in the middle of the dark region, not on its edge.")
    half = max(2.0, abs(dx) / 2 + 1.5, abs(dy) / 2 + 1.5)
    land = comp.landscape(v, yaw, lim, nom, sigma_xy=float(sigma), half=half, step=max(0.02, half / 100),
                          center=(dx / 2, dy / 2))
    ui.chart(ccmp.landscape(land, settings, label="FPY" if sigma == 0 else "Expected FPY"), "cp_land")
with right:
    ui.section("FPY vs yaw", "Solid: only the yaw changes, X / Y kept at the selected values. Dashed: the best FPY "
                             "reachable at each yaw when X / Y are re-optimised. The peak is often not at the median "
                             "yaw, because what matters is each module's worst corner, not the average.")
    yaws = np.round(np.arange(yaw - 0.3, yaw + 0.3001, 0.002), 4)
    yb = np.round(np.arange(yaw - 0.3, yaw + 0.3001, 0.015), 4)
    ui.chart(ccmp.yaw_sweep(yaws, comp.fpy_vs_yaw(v, dx, dy, yaws, lim, nom),
                            {m: s[2] for m, s in settings.items() if m != "No compensation"},
                            best=(yb, data.best_by_yaw(c.key, t, int(weeks), round(dx, 3), round(dy, 3),
                                                       tuple(yb), a))), "cp_yaw")
    cp = data.change_points(c.key, a)
    recent_change = False
    t_end = a.first["Date"].max()
    for (tt, col), idx in cp["idx"].items():
        if tt != t or not idx:
            continue
        dates = vt_all.sort_values("Date").dropna(subset=[col])["Date"].to_numpy()
        if len(dates) and pd.Timestamp(dates[min(idx[-1], len(dates) - 1)]) >= t_end - pd.Timedelta(days=14):
            recent_change = True
    checks = comp.readiness(v, bts, recent_change, min_n=int(min_n))
    ready = all(ck["ok"] for ck in checks)
    h1, h2 = st.columns([3, 1.3], vertical_alignment="center")
    with h1:
        ui.section("Is it ready?", "Is the process stable enough to trust a compensation on the line?")
    h2.badge("Ready" if ready else "Not yet", color="green" if ready else "orange",
             icon=":material/check:" if ready else ":material/hourglass_top:")
    ui.checklist(checks)

ui.section("Backtest week by week")
ui.chart(ccmp.backtest_chart(bt, list(comp.METHODS)), "cp_bt")
if bts:
    best = bts.get("best")
    st.caption(f"Applied week by week over {bts['label']} ({bts['n']} modules), **{best}** did best: "
               f"{bts.get(best, np.nan):.1f}% vs {bts.get('No compensation', np.nan):.1f}% without correction.")

# ---- Original diagnostics -------------------------------------------------------------------------
sim = comp.simulate_table(v, (dx, dy, yaw), lim, nom)
with st.expander("Before / after diagnostics (original views)", icon=":material/compare:"):
    c1, c2 = st.columns(2)
    with c1:
        ui.chart(ccmp.before_after_centroids(sim, title="Centroids: original vs compensated"), "cp_bac")
    with c2:
        ui.chart(ccmp.before_after_yaw(sim, title="Yaw before vs after"), "cp_bay")
    c3, c4 = st.columns(2)
    with c3:
        ui.chart(ccmp.weekly_real_vs_sim(sim, title="FPY per week: real vs simulated"), "cp_wrs")
    with c4:
        ui.chart(ccmp.outline_overlay(sim, t, nom, title="Average outline: nominal · measured · compensated"),
                 "cp_overlay")
    c5, c6 = st.columns(2)
    with c5:
        ui.chart(ccmp.comp_vectors([{"BatteryType": t, "bias_x": float(v["Centroid_X"].median()),
                                     "bias_y": float(v["Centroid_Y"].median()), "dx": dx, "dy": dy}],
                                   title="Compensation vector: observed bias vs applied offset"), "cp_vec")
    with c6:
        ui.section("Transition matrix (original status → simulated status)")
        st.dataframe(pd.crosstab(sim["Status"], sim["Status_Sim"]), width="stretch")
    st.caption("A good correction points opposite to the observed bias; the outline overlay shows whether the "
               "average module ends up on nominal.")
    ui.section("Piece-level simulation")
    cols = ["_mod_key", "CalendarWeek", "PartID", *[f"{k}" for k in ("FL_X", "FL_Y", "FR_X", "FR_Y", "RL_X", "RL_Y",
                                                                       "RR_X", "RR_Y")], "Status",
            *[f"{k}_Sim" for k in ("FL_X", "FL_Y", "FR_X", "FR_Y", "RL_X", "RL_Y", "RR_X", "RR_Y")], "Status_Sim"]
    tbl = sim[cols].reset_index(drop=True)
    evs = st.dataframe(ui.status_styler(tbl.drop(columns="_mod_key"), lim), hide_index=True, width="stretch",
                       height=360, on_select="rerun", selection_mode="single-row", key="cp_table",
                       column_config=ui.run_table_config())
    i = ui.selected_row(evs)
    if i is not None and i < len(tbl) and tbl.loc[i, "_mod_key"] != state.focus():
        state.set_focus(tbl.loc[i, "_mod_key"])
        st.rerun()

with st.expander("How the robust optimum is found", icon=":material/school:"):
    st.markdown(
        "A module passes only when **all four corners** are inside the tolerance, so what matters is each module's "
        "worst corner — not the average module. For a given yaw, the X/Y settings that make one module pass form a "
        "rectangle. If every setting is blurred by placement noise σ, the chance that a module passes is a product "
        "of two normal-distribution differences, and the expected FPY over all modules on the whole X × Y grid is "
        "one matrix product. The search scans yaw too and picks the setting with the highest expected FPY; with "
        "σ > 0 it prefers the centre of a good region over a lucky edge, which is what makes it hold up on new "
        "modules. That is also why small manual tweaks on top of the median — especially in yaw — often beat it: "
        "the median centres the average module, while the corners are biased differently.")
    if opt is not None:
        st.caption(f"Robust optimum: expected FPY {opt.expected_fpy:.1f}% under σ = {sigma:.2f} mm / {sigma_yaw:.3f}°, "
                   f"exact FPY on the window {opt.exact.get('fpy', np.nan):.1f}%.")
