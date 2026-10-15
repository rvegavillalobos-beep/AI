import json

import numpy as np
import pandas as pd
import streamlit as st

from app import data, state, ui
from rickhard import events as ev_mod
from rickhard.charts import comp_charts as ccmp
from rickhard.stats import significance_label

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "events", "Log what was done to the line — an end-stop moved, a guide realigned, a tool changed — and "
                            "get a before / after verdict with statistics.")
ui.require(a.first)

# ---- Editable log (the editor's input only changes when events are added elsewhere) ---------------
records = state.events()
sig = json.dumps(records, sort_keys=True, default=str)
if st.session_state.get("_ev_sig") != sig or "_ev_base" not in st.session_state:
    base = ev_mod.normalize_events(records)
    st.session_state["_ev_base"] = base
    st.session_state["_ev_ver"] = st.session_state.get("_ev_ver", 0) + 1

ui.section("Intervention log", "Add a row with the + button below the table. Scope says which battery type the change "
                               "affects. dX / dY / dYaw are optional (a correction you applied).")
cfg = st.column_config
edited = st.data_editor(
    st.session_state["_ev_base"], key=f"ev_editor_{st.session_state['_ev_ver']}", num_rows="dynamic",
    hide_index=True, width="stretch",
    column_config={
        "Date": cfg.DatetimeColumn("Date / time", format="YYYY-MM-DD HH:mm", required=True),
        "Scope": cfg.SelectboxColumn("Scope", options=ev_mod.SCOPES, default="All", required=True),
        "Category": cfg.SelectboxColumn("Category", options=ev_mod.CATEGORIES, default="Adjustment"),
        "Title": cfg.TextColumn("What was done", width="large", required=True),
        "Notes": cfg.TextColumn("Notes", width="medium"),
        "dX": cfg.NumberColumn("dX mm", format="%+.2f"), "dY": cfg.NumberColumn("dY mm", format="%+.2f"),
        "dYaw": cfg.NumberColumn("dYaw °", format="%+.3f"),
    })
new_records = ev_mod.events_to_records(edited)
state.set_events(new_records)
st.session_state["_ev_sig"] = json.dumps(new_records, sort_keys=True, default=str)
evdf = ev_mod.normalize_events(new_records)

cp = data.change_points(c.key, a)["table"]
if len(cp):
    with st.expander(f"{len(cp)} process shift(s) detected in the data — add them to the log?", icon=":material/bolt:"):
        show = cp[["Date", "Variable", "Before", "After", "Shift", "p"]]
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "Date": cfg.DatetimeColumn(format="YYYY-MM-DD HH:mm"), "Before": cfg.NumberColumn(format="%+.3f"),
            "After": cfg.NumberColumn(format="%+.3f"), "Shift": cfg.NumberColumn(format="%+.3f"),
            "p": cfg.NumberColumn("p-value", format="%.1e")})

        def _add_detected():
            for _, row in cp.iterrows():
                unit = "°" if "Yaw" in row["Variable"] else " mm"
                scope = "Type S" if row["Variable"].startswith("Type S") else ("Type M" if row["Variable"].startswith("Type M") else "All")
                state.add_event({"Date": pd.Timestamp(row["Date"]).isoformat(), "Scope": scope, "Category": "Other",
                                 "Title": f"Detected shift: {row['Variable']} {row['Shift']:+.3f}{unit}",
                                 "Notes": "Found by change-point detection — replace with what was done.",
                                 "dX": None, "dY": None, "dYaw": None})

        st.button("Add them as placeholders", icon=":material/add:", on_click=_add_detected)

if evdf.empty:
    st.info("No interventions logged yet. Add one above to measure its effect.", icon=":material/event_note:")
    st.stop()

# ---- Impact analysis -------------------------------------------------------------------------
ui.section("Impact of an intervention")
labels = [f"{d:%Y-%m-%d %H:%M} · {t_} ({s_})" for d, t_, s_ in zip(evdf["Date"], evdf["Title"], evdf["Scope"])]
state.ensure_option("p_ev_sel", labels, labels[-1])
ctl = st.columns([3, 1.2, 1.4], vertical_alignment="bottom")
with ctl[0]:
    sel = st.selectbox("Intervention", labels, key="p_ev_sel")
with ctl[1]:
    state.init("p_ev_days", 14)
    days = st.number_input("Days before / after", 1, 90, step=1, key="p_ev_days")
var_opts = {"Shift X (fit)": ("Fit_Tx", "mm"), "Shift Y (fit)": ("Fit_Ty", "mm"), "Yaw (fit)": ("Fit_Rot", "°"),
            "Worst corner |dev|": ("WorstAbs", "mm"), "Shape residual": ("ShapeMax", "mm")}
with ctl[2]:
    state.ensure_option("p_ev_var", list(var_opts), "Shift X (fit)")
    var = st.selectbox("Variable to plot", list(var_opts), key="p_ev_var")
e = evdf.iloc[labels.index(sel)]
ba = ev_mod.before_after(a.first, e["Date"], e["Scope"], float(days))
if not ba.get("enough"):
    st.warning(f"Not enough modules around this date in the current filters ({ba['n_before']} before, "
               f"{ba['n_after']} after; at least 5 each are needed). Widen the weeks in the sidebar or the window.")
else:
    lvl, txt = ba["verdict"]
    {"good": st.success, "bad": st.error, "info": st.info, "neutral": st.info}[lvl](txt)
    tb = ba["table"].set_index("Column")
    cards = [
        {"label": "FPY before → after", "value": f"{ba['fpy_before']:.1f}% → {ba['fpy_after']:.1f}%",
         "delta": f"{ba['fpy_after'] - ba['fpy_before']:+.1f} pp", "good": ba["fpy_after"] >= ba["fpy_before"],
         "sub": significance_label(ba["fpy_p"])},
        {"label": "Modules", "value": f"{ba['n_before']} / {ba['n_after']}", "sub": f"{days} days before / after"},
    ]
    for col, lab, unit in (("Fit_Tx", "Shift X", "mm"), ("Fit_Ty", "Shift Y", "mm"), ("Fit_Rot", "Yaw", "°")):
        if col in tb.index:
            rr = tb.loc[col]
            nd = 3 if unit == "°" else 2
            cards.append({"label": f"{lab} level", "value": f"{rr['Before']:+.{nd}f} → {rr['After']:+.{nd}f}",
                          "unit": unit, "sub": f"Δ {rr['Change']:+.{nd}f} · {significance_label(rr['p (mean)'])} · "
                                               f"σ ×{rr['SD ratio']:.2f}"})
    ui.kpi_cards(cards)
    if np.isfinite(e.get("dX", np.nan)) or np.isfinite(e.get("dYaw", np.nan)):
        planned = [f"{n} {e[k]:+.3f}" for k, n in (("dX", "dX"), ("dY", "dY"), ("dYaw", "dYaw")) if np.isfinite(e[k])]
        st.caption("Planned correction: " + ", ".join(planned) + " — compare with the observed level change above.")

col, unit = var_opts[var]
ev_scope = evdf[evdf["Scope"].isin(["All", e["Scope"]]) | (e["Scope"] == "All")]
f_scope = a.first[a.first["BatteryType"].isin(ev_mod.scope_types(e["Scope"]))]
left, right = st.columns([3, 2], gap="large")
with left:
    ui.chart(ccmp.event_timeline(ev_mod.rolling_series(f_scope, "PASS", 40), ev_scope, "FPY (rolling 40)", "%",
                                 e["Date"], target=r.fpy_target), "ev_fpy")
    ui.chart(ccmp.event_timeline(ev_mod.rolling_series(f_scope, col, 25), ev_scope, var, unit, e["Date"]), "ev_var")
with right:
    if ba.get("enough"):
        ui.chart(ccmp.before_after_dist(ba["before"][col], ba["after"][col], var, unit,
                                        title=f"{var}: {days} days before vs after"), "ev_dist")
        tab = ba["table"][["Variable", "Before", "After", "Change", "p (mean)", "SD before", "SD after", "p (spread)"]]
        st.dataframe(tab, hide_index=True, width="stretch", height=420, column_config={
            "Before": cfg.NumberColumn(format="%+.3f"), "After": cfg.NumberColumn(format="%+.3f"),
            "Change": cfg.NumberColumn(format="%+.3f"), "p (mean)": cfg.NumberColumn(format="%.3f"),
            "SD before": cfg.NumberColumn(format="%.3f"), "SD after": cfg.NumberColumn(format="%.3f"),
            "p (spread)": cfg.NumberColumn(format="%.3f")})
        st.caption("p (mean): Welch t-test. p (spread): Brown–Forsythe test on the variances. Small p (< 0.05) "
                   "= a real change, not noise.")
