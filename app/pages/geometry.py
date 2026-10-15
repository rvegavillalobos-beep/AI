import numpy as np
import pandas as pd
import streamlit as st

from app import data, state, ui
from rickhard.charts import geometry_view as gv
from rickhard.constants import CORNERS

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "geometry", "Modules drawn against the nominal outline with magnified deviations; the week-by-week "
                              "drift movie; and the correction map of every corner.")
ui.require(a.first)
ui.focus_note(c)

tab1, tab2, tab3 = st.tabs(["Module outlines", "Drift movie", "Correction map"])

with tab1:
    ctl = st.columns([1.3, 1.4, 1.6, 1.6], gap="medium", vertical_alignment="bottom")
    with ctl[0]:
        scopes = ["First runs", "All runs"]
        state.ensure_option("p_geo_scope", scopes, "First runs")
        scope = st.segmented_control("Runs", scopes, key="p_geo_scope") or "First runs"
    with ctl[1]:
        sts = ["OK", "NOK", "Incomplete"]
        state.init("p_geo_status", ["OK", "NOK", "Incomplete"])
        shown = st.pills("Show", sts, selection_mode="multi", key="p_geo_status") or []
    with ctl[2]:
        state.init("p_geo_exag", 20.0)
        exag = st.slider("Magnify deviations ×", 1.0, 60.0, step=1.0, key="p_geo_exag")
    with ctl[3]:
        types = c.types
        if len(types) > 1:
            state.ensure_option("p_geo_types", ["All types"] + types, "All types")
            tsel = st.segmented_control("Battery type", ["All types"] + types, key="p_geo_types") or "All types"
        else:
            tsel = types[0]

    base = a.first if scope == "First runs" else a.runs
    base = base.sort_values("Date").reset_index(drop=True)
    if tsel != "All types":
        base = base[base["BatteryType"] == tsel].reset_index(drop=True)
        types = [tsel]
    n_all = len(base)
    rng = st.session_state.get("p_geo_range")
    if (not isinstance(rng, (tuple, list)) or len(rng) != 2 or not (0 <= rng[0] <= rng[1] <= max(0, n_all - 1))
            or st.session_state.get("_geo_n") != n_all):
        st.session_state["p_geo_range"] = (max(0, n_all - 40), max(0, n_all - 1))
        st.session_state["_geo_n"] = n_all
    if n_all > 1:
        i0, i1 = st.slider("Modules shown (chronological index — drag to go back in time)", 0, n_all - 1,
                           key="p_geo_range")
    else:
        i0, i1 = 0, 0
    plot_df = base.iloc[int(i0): int(i1) + 1]
    if len(plot_df):
        st.caption(f"{len(plot_df)} modules · {plot_df['Date'].min():%Y-%m-%d %H:%M} – {plot_df['Date'].max():%Y-%m-%d %H:%M}")
    focus = state.focus()
    if focus and focus in set(base["_mod_key"]) and focus not in set(plot_df["_mod_key"]):
        plot_df = pd.concat([plot_df, base[base["_mod_key"] == focus]])
    status_map = {"OK": "PASS", "NOK": "FAIL", "Incomplete": "INCOMPLETE"}
    status_filter = tuple(status_map[s] for s in shown)
    ui.chart(gv.outlines(plot_df, c.nominals, r.spec_limit, exag, focus_key=focus, types=types,
                         status_filter=status_filter), "geo_outlines")
    st.caption("Click a legend entry to hide a group, double-click to isolate it. Select a row below to focus one "
               "module (it is drawn in violet with its corner values).")

    if focus and focus in set(a.runs["_mod_key"]):
        row = a.runs[a.runs["_mod_key"] == focus].iloc[0]
        sq = a.sq_runs[a.sq_runs["_mod_key"] == focus]
        m = st.columns(5)
        m[0].metric("Status", row["Status"], border=True)
        m[1].metric("Worst corner", f"{row['WorstAbs']:.2f} mm" if np.isfinite(row["WorstAbs"]) else "–",
                    str(row["WorstCoord"] or "").replace("_", " "), delta_color="off", border=True)
        m[2].metric("Margin to limit", f"{row['Margin']:+.2f} mm" if np.isfinite(row["Margin"]) else "–", border=True)
        if len(sq):
            s0 = sq.iloc[0]
            m[3].metric("Diagonal delta", f"{s0['DeltaDiag']:.2f} mm", s0["SquarenessStatus"], delta_color="off",
                        border=True)
            m[4].metric("FL angle deviation", f"{s0['AngleDevFL']:+.3f}°", border=True)

    cols = ["_mod_key", "Date", "PartID", "BatteryType", "RunNum", *[f"{k}_{ax}" for k in CORNERS for ax in "XY"],
            "WorstAbs", "Status"]
    tbl = plot_df.sort_values("Date", ascending=False)[cols].reset_index(drop=True)
    ev = st.dataframe(ui.status_styler(tbl.drop(columns="_mod_key"), r.spec_limit), hide_index=True, width="stretch",
                      height=300, on_select="rerun", selection_mode="single-row", key="geo_table",
                      column_config=ui.run_table_config())
    i = ui.selected_row(ev)
    if i is not None and i < len(tbl) and tbl.loc[i, "_mod_key"] != focus:
        state.set_focus(tbl.loc[i, "_mod_key"])
        st.rerun()

with tab2:
    types = c.types
    ctl = st.columns([1.2, 2], vertical_alignment="bottom")
    with ctl[0]:
        if len(types) > 1:
            state.ensure_option("p_mov_type", types, types[0])
            tm = st.segmented_control("Battery type", types, key="p_mov_type") or types[0]
        else:
            tm = types[0]
    with ctl[1]:
        state.init("p_mov_exag", 40.0)
        mk = st.slider("Magnify ×", 5.0, 100.0, step=5.0, key="p_mov_exag")
    ui.chart(gv.drift_movie(a.first, c.nominals, tm, r.spec_limit, mk), "geo_movie")
    st.caption("Press ▶ to play the weeks. The thick outline is the average module of the week; dots are the corners of "
               "every module (red = NOK). A drifting average outline is a placement trend; growing clouds are more scatter.")

with tab3:
    types = c.types
    ctl = st.columns([1.2, 2], vertical_alignment="bottom")
    with ctl[0]:
        if len(types) > 1:
            state.ensure_option("p_cm_type", types, types[0])
            tc = st.segmented_control("Battery type", types, key="p_cm_type") or types[0]
        else:
            tc = types[0]
    with ctl[1]:
        state.init("p_cm_exag", 60.0)
        ck = st.slider("Magnify ×", 10.0, 200.0, step=10.0, key="p_cm_exag")
    ui.chart(gv.correction_map(a.first, c.nominals, tc, r.spec_limit, ck), "geo_cmap")
    d = a.first[a.first["BatteryType"] == tc]
    rows = []
    for k in CORNERS:
        for ax in "XY":
            v = d[f"{k}_{ax}"].dropna()
            rows.append({"Corner": k, "Axis": ax, "Mean": v.mean(), "Median": v.median(), "σ": v.std(),
                         "Correction (−mean)": -v.mean(), "N": len(v)})
    cfg = st.column_config
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Mean": cfg.NumberColumn(format="%+.2f"), "Median": cfg.NumberColumn(format="%+.2f"),
        "σ": cfg.NumberColumn(format="%.2f"), "Correction (−mean)": cfg.NumberColumn(format="%+.2f")})
    st.caption("Per-corner corrections are only achievable if they are consistent with a rigid move of the whole "
               "module — Compensation Studio finds the best rigid X / Y / yaw.")
