import numpy as np
import pandas as pd
import streamlit as st

from app import data, state, ui
from rickhard.charts import geometry_view as gv
from rickhard.constants import DEV_COLS, MECHANISM_HELP

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "passport", "Everything about one module: every run, its geometry, why it failed, and what was "
                              "produced around it.")
ui.require(a.runs, "runs")

runs = a.runs
ids = runs.sort_values("Date", ascending=False)["PartID"].drop_duplicates().tolist()
focus = state.focus()
focus_id = runs.loc[runs["_mod_key"] == focus, "PartID"].iloc[0] if focus and (runs["_mod_key"] == focus).any() else None

ctl = st.columns([2, 3], vertical_alignment="bottom")
with ctl[0]:
    q = st.text_input("Search Part ID", key=state.init("p_pp_q", ""), placeholder="type any part of the ID")
matches = [i for i in ids if q.strip().upper() in i.upper()] if q.strip() else ids
if focus_id and focus_id not in matches:
    matches = [focus_id] + matches
if not matches:
    st.warning("No Part ID matches the search.")
    st.stop()
default = focus_id or matches[0]
state.ensure_option("p_pp_id", matches, default)
if focus_id and st.session_state.get("_pp_last_focus") != focus:
    st.session_state["p_pp_id"] = focus_id
    st.session_state["_pp_last_focus"] = focus
with ctl[1]:
    pid = st.selectbox(f"Module ({len(matches):,} found)", matches, key="p_pp_id")

mr = runs[runs["PartID"] == pid].sort_values("RunNum")
fr = a.first[a.first["PartID"] == pid]
f0 = fr.iloc[0] if len(fr) else mr.iloc[0]
status = f0["Status"]
mech = f0.get("Mechanism", "") if len(fr) else ""
sqs = f0.get("SquarenessStatus", None)
ui.kpi_cards([
    {"label": "First measurement", "value": "OK" if status == "PASS" else ("NOK" if status == "FAIL" else status),
     "sub": f"{pd.Timestamp(f0['Date']):%Y-%m-%d %H:%M} · {f0['BatteryType']} · shift {f0.get('Shift', '–')}",
     "accent": "#0ca30c" if status == "PASS" else "#d03b3b"},
    {"label": "Runs", "value": f"{len(mr)}", "sub": " → ".join(("OK" if s == "PASS" else "NOK" if s == "FAIL" else "INC")
                                                                for s in mr["Status"])},
    {"label": "Worst corner (run 1)", "value": ui.fmt_mm(f0["WorstAbs"], 2, signed=False),
     "sub": f"{str(f0['WorstCoord'] or '').replace('_', ' ')} · margin {f0['Margin']:+.2f} mm"
     if np.isfinite(f0["Margin"]) else "incomplete"},
    {"label": "Why it failed" if status != "PASS" else "Placement (fit)",
     "value": mech if status != "PASS" and mech else f"{f0.get('Fit_Tx', np.nan):+.2f} / {f0.get('Fit_Ty', np.nan):+.2f}",
     "unit": None if status != "PASS" else "mm", "sub": MECHANISM_HELP.get(mech, f"yaw {f0.get('Fit_Rot', np.nan):+.3f}°")},
    {"label": "Squareness", "value": sqs or "–",
     "sub": f"Δ diag {f0['DeltaDiag']:.2f} mm" if sqs else "needs all four corners"},
])

left, right = st.columns([3, 2], gap="large")
with left:
    run_opts = [f"Run {int(x)} · {pd.Timestamp(d):%Y-%m-%d %H:%M} · {s}" for x, d, s in
                zip(mr["RunNum"], mr["Date"], mr["Status"])]
    state.ensure_option("p_pp_run", run_opts, run_opts[0])
    run_lab = st.selectbox("Run", run_opts, key="p_pp_run")
    row = mr.iloc[run_opts.index(run_lab)]
    state.init("p_pp_mag", 80.0)
    mag = st.slider("Magnify ×", 10.0, 250.0, step=10.0, key="p_pp_mag")
    ui.chart(gv.module_drawing(row, c.nominals, r.spec_limit, mag), "pp_draw")
with right:
    ui.section("Corner values per run")
    tbl = mr[["RunNum", "Date", *DEV_COLS, "WorstAbs", "Status"]].copy()
    st.dataframe(ui.status_styler(tbl, r.spec_limit), hide_index=True, width="stretch",
                 column_config=ui.run_table_config())
    if len(fr):
        ui.section("Compared with all modules of its type")
        pop = a.first[a.first["BatteryType"] == f0["BatteryType"]]
        rows = []
        for col, lab in (("Fit_Tx", "Shift X"), ("Fit_Ty", "Shift Y"), ("Fit_Rot", "Yaw"), ("ShapeMax", "Shape residual"),
                         ("WorstAbs", "Worst corner")):
            v = pop[col].dropna()
            x = f0.get(col, np.nan)
            if len(v) and np.isfinite(x):
                rows.append({"Measure": lab, "This module": x, "Typical (median)": float(v.median()),
                             "Percentile": float((v < x).mean() * 100)})
        cfg = st.column_config
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "This module": cfg.NumberColumn(format="%+.3f"), "Typical (median)": cfg.NumberColumn(format="%+.3f"),
            "Percentile": cfg.ProgressColumn(format="%.0f", min_value=0, max_value=100)})

ui.section("Produced around it", "First runs within ±2 hours of this module — a cluster of NOK around it points to an "
                                 "event on the line rather than to the module itself.")
t0 = pd.Timestamp(f0["Date"])
near = a.first[(a.first["Date"] >= t0 - pd.Timedelta(hours=2)) & (a.first["Date"] <= t0 + pd.Timedelta(hours=2))]
near = near.sort_values("Date")[["_mod_key", "Date", "PartID", "BatteryType", "Shift", "Fit_Tx", "Fit_Ty", "Fit_Rot",
                                 "WorstAbs", "Mechanism", "Status"]].reset_index(drop=True)
if len(near):
    nok = int((near["Status"] != "PASS").sum())
    st.caption(f"{len(near)} modules in the window, {nok} NOK ({nok / len(near) * 100:.0f}%) vs "
               f"{100 - a.fpy:.1f}% overall.")
    cfg = st.column_config
    ev = st.dataframe(ui.status_styler(near.drop(columns="_mod_key"), r.spec_limit), hide_index=True, width="stretch",
                      height=300, on_select="rerun", selection_mode="single-row", key="pp_near",
                      column_config={**ui.run_table_config(), "Fit_Tx": cfg.NumberColumn("Shift X", format="%+.2f"),
                                     "Fit_Ty": cfg.NumberColumn("Shift Y", format="%+.2f"),
                                     "Fit_Rot": cfg.NumberColumn("Yaw °", format="%+.3f")})
    i = ui.selected_row(ev)
    if i is not None and i < len(near) and near.loc[i, "_mod_key"] != focus:
        state.set_focus(near.loc[i, "_mod_key"])
        st.rerun()
