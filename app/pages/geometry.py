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
        sts = ["OK", "NOK"]
        state.init("p_geo_status", list(sts))
        if not isinstance(st.session_state["p_geo_status"], list) or not set(st.session_state["p_geo_status"]) <= set(sts):
            st.session_state["p_geo_status"] = [s for s in (st.session_state["p_geo_status"] or []) if s in sts]
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
    if tsel != "All types":
        base = base[base["BatteryType"] == tsel]
        types = [tsel]
    # a module with a missing corner has no outline: it is left out of the drawing, the slider and the table
    complete = base[[f"{k}_{ax}" for k in CORNERS for ax in "XY"]].notna().all(axis=1)
    n_missing = int((~complete).sum())
    base = base[complete].sort_values("Date").reset_index(drop=True)
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
        st.caption(f"{ui.plural(len(plot_df), 'module')} · {plot_df['Date'].min():%Y-%m-%d %H:%M} – "
                   f"{plot_df['Date'].max():%Y-%m-%d %H:%M}"
                   + (f" · {ui.plural(n_missing, 'module')} with a missing corner not drawn" if n_missing else ""))

    # focus: pick any module by (part of) its ID, or select a row in the table below
    focus = state.focus()
    keys = base["_mod_key"].iloc[::-1].tolist()
    options = ["(none)"] + keys
    if focus not in keys:
        focus = None
    sel = st.selectbox("Focus on a module", options, index=options.index(focus) if focus else 0,
                       help="Type part of a Part ID to search. Selecting a row in a table (here, Shape & Squareness, "
                            "Compensation Studio…) also sets the focus.")
    picked = None if sel == "(none)" else sel
    if picked != focus:
        state.set_focus(picked)
        focus = picked
    if focus and focus in set(base["_mod_key"]) and focus not in set(plot_df["_mod_key"]):
        plot_df = pd.concat([plot_df, base[base["_mod_key"] == focus]])
    status_map = {"OK": "PASS", "NOK": "FAIL"}
    status_filter = tuple(status_map[s] for s in shown)

    left, right = st.columns([3.2, 1], gap="large")
    with left:
        ui.chart(gv.outlines(plot_df, c.nominals, r.spec_limit, exag, focus_key=focus, types=types,
                             status_filter=status_filter), "geo_outlines")
        st.caption("Up to 60 modules, each is its own legend entry: click one to hide it, double-click to show only "
                   "that one (double-click again to bring all back). Larger windows are grouped by status.")
    with right:
        if focus and focus in set(a.runs["_mod_key"]):
            row = a.runs[a.runs["_mod_key"] == focus].iloc[0]
            st.markdown(f"**{row['PartID']}**")
            st.caption(f"Run {int(row['RunNum'])} · {row['Date']:%Y-%m-%d %H:%M} · {row['BatteryType']} · "
                       f"{row['CalendarWeek']}")
            st.badge({"PASS": "OK", "FAIL": "NOK"}.get(row["Status"], row["Status"].title()),
                     color={"PASS": "green", "FAIL": "red"}.get(row["Status"], "gray"))
            dev = pd.DataFrame({"Corner": list(CORNERS),
                                "X (mm)": [row[f"{k}_X"] for k in CORNERS],
                                "Y (mm)": [row[f"{k}_Y"] for k in CORNERS]})
            st.dataframe(dev.style.map(lambda v: "background-color:rgba(208,59,59,.18);color:#b02a2a;font-weight:600"
                                       if pd.notna(v) and abs(v) > r.spec_limit else "", subset=["X (mm)", "Y (mm)"])
                         .format("{:+.2f}", subset=["X (mm)", "Y (mm)"], na_rep="missing"),
                         hide_index=True, width="stretch")
            sq = a.sq_runs[a.sq_runs["_mod_key"] == focus]
            if len(sq):
                q = sq.iloc[0]
                geo = pd.DataFrame({
                    "Measure": ["Diagonal Δ", "Width Δ (front − rear)", "Length Δ (left − right)", "FL angle Δ",
                                "Placement uses", "Shape uses"],
                    "Value": [f"{q['DeltaDiag']:.2f} mm", f"{q['WidthDelta']:+.2f} mm", f"{q['LengthDelta']:+.2f} mm",
                              f"{q['AngleDevFL']:+.3f}°", f"{q['RigidMax']:.2f} of ±{r.spec_limit:g} mm",
                              f"{q['ShapeMax']:.2f} of ±{r.spec_limit:g} mm"]})
                st.dataframe(geo, hide_index=True, width="stretch")
                if q["SquarenessStatus"] == "DEFORMED":
                    st.markdown(f":red[**Deformed**] · {str(q['RootCause']).capitalize()}")
                else:
                    st.markdown(f":green[**Square**] (diagonal Δ ≤ {r.diag_tol:g} mm)")
                if row["Status"] != "PASS":
                    why = {"Placement": "Fails on placement alone: it would fail even if perfectly square.",
                           "Shape + placement": "Would pass if it were perfectly square (same placement): the shape "
                                                "ate the margin.",
                           "Shape": "Its shape alone is out of tolerance, whatever the placement.",
                           "Both": "Placement alone and shape alone would each fail.",
                           "Height (Z)": "X and Y are in tolerance; the height (Z) is not."}
                    st.caption(why.get(str(row.get("NokCause", "")), ""))
            else:
                st.caption("Not all four corners were measured: no geometry metrics.")
            hist = a.runs[a.runs["PartID"] == row["PartID"]].sort_values("RunNum")
            if len(hist) > 1:
                st.caption("All runs: " + " → ".join(
                    f"R{int(rn)} {'OK' if s_ == 'PASS' else ('NOK' if s_ == 'FAIL' else s_.title())}"
                    for rn, s_ in zip(hist["RunNum"], hist["Status"])))
            st.page_link("app/pages/passport.py", label="Open in Module Passport", icon=":material/badge:")
        else:
            st.info("Pick a module above, or select a row below, to see its deviations and geometry.",
                    icon=":material/ads_click:")
        in_view = plot_df[plot_df["Status"].isin(status_filter)]
        n_ok = int((in_view["Status"] == "PASS").sum())
        st.caption(f"In view: {ui.plural(len(in_view), 'module')} · {n_ok} OK · {len(in_view) - n_ok} not OK.")

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
