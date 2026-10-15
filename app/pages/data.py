import dataclasses

import pandas as pd
import streamlit as st

from app import data, state, ui
from rickhard import loading
from rickhard.config import Nominals, ShiftPlan, project_from_json, project_to_json
from rickhard.constants import BATTERY_TYPES, CORNERS, DEFAULT_NOMINALS

c = data.ctx()
a, r, ds = c.a, c.rules, c.ds
h = ds.health
ui.page_header(c, "data", "What was read and how: the data-quality report, how times, runs and corners are "
                          "interpreted, the nominal geometry, the shift plan and your project file.")

TIMEZONES = ["Europe/Berlin", "America/Mexico_City", "UTC", "America/Chicago", "America/New_York",
             "America/Los_Angeles", "Europe/London", "Europe/Madrid", "Asia/Shanghai", "Asia/Tokyo"]

# ---- Health -------------------------------------------------------------------------------------
ui.kpi_cards([
    {"label": "Rows read", "value": f"{h['rows_read']:,}", "sub": f"{h['files']} file(s)"},
    {"label": "Modules", "value": f"{h['modules']:,}", "sub": f"{h['module_runs']:,} runs · {h['repeat_runs']:,} re-measurements"},
    {"label": "Runs with a missing corner", "value": f"{h['incomplete_runs']:,}"},
    {"label": "Duplicate rows removed", "value": f"{h['duplicate_rows_removed']:,}"},
    {"label": "Failed reads (all zero)", "value": f"{h['zero_readings']:,}",
     "sub": "treated as missing" if h["zero_as_missing"] else "used as measurements"},
])

notes = []
if h.get("duplicate_rows_removed"):
    notes.append(("warn", f"**{h['duplicate_rows_removed']} exact duplicate row(s)** were dropped (repeated rows or "
                          "overlapping exports). Left in, a duplicated corner looks like the start of a fake re-measurement."))
for pair in h.get("similar_part_ids", []):
    merged = h.get("lookalike_ids_merged", 0) > 0
    notes.append(("info" if merged else "warn",
                  f"Part IDs **{pair[0]}** and **{pair[1]}** differ only by letter O vs digit 0. "
                  + ("They are merged." if merged else "They are counted as two modules — merge them below.")))
if h.get("zero_readings"):
    notes.append(("warn", f"**{h['zero_readings']} reading(s) with X, Y (and Z) all exactly 0** — almost certainly "
                          "failed reads. " + ("Treated as missing corners." if h["zero_as_missing"]
                                              else "Currently used as real measurements.")))
if h.get("dst_ambiguous_rows") or h.get("dst_nonexistent_rows"):
    notes.append(("info", f"Daylight-saving change in the export time zone: {h['dst_ambiguous_rows']} row(s) in the "
                          f"repeated autumn hour (read as summer time), {h['dst_nonexistent_rows']} in the skipped "
                          "spring hour (moved forward). None were dropped."))
if h.get("rows_without_valid_time"):
    ex = ", ".join(f"`{x}`" for x in h.get("examples_bad_time", []))
    notes.append(("bad", f"**{h['rows_without_valid_time']} row(s) without a readable time** were left out ({ex})."))
if h.get("rows_unmapped_feature"):
    feats = ", ".join(f"`{f}` ({n})" for f, n in h.get("unmapped_features", {}).items())
    notes.append(("warn", f"**{h['rows_unmapped_feature']} row(s)** have a feature that is not one of the four corners "
                          f"and were ignored: {feats}. Map them below if they are corners."))
if h.get("non_numeric_deviation_values"):
    notes.append(("warn", f"{h['non_numeric_deviation_values']} deviation value(s) were not numbers and count as missing."))
if h.get("corner_measured_twice_in_a_run"):
    notes.append(("info", f"{h['corner_measured_twice_in_a_run']} time(s) a corner appeared twice within one run; "
                          "the last value was used."))
if ds.limits:
    txt = " · ".join(f"{ax} {lo:+g} / {hi:+g} mm" for ax, (lo, hi) in ds.limits.items())
    notes.append(("info", f"Limits found in the export: {txt}. The X/Y tolerance in the sidebar is ±{r.spec_limit:g} mm."))
opts = state.load_options()
notes.append(("info", f"Times are read as **{opts.source_tz}** and shown in **{opts.display_tz}**; calendar weeks are "
                      f"ISO weeks of the shown time. Time format detected: {', '.join(h.get('time_formats', [])) or '–'}."))


class _N:
    def __init__(self, level, title, text):
        self.level, self.title, self.text, self.action = level, title, text, None


ui.section("Data-quality report")
ui.insight_cards([_N(lv, {"warn": "Check", "bad": "Problem", "info": "Note", "good": "OK"}[lv], t) for lv, t in notes])

with st.expander("Files", icon=":material/description:"):
    st.dataframe(pd.DataFrame([{k: v for k, v in f.items() if k != "columns"} for f in ds.files]), hide_index=True,
                 width="stretch")
    for f in ds.files:
        st.caption(f"**{f['name']}** — columns used: " + ", ".join(f"{k} → `{v}`" for k, v in f["columns"].items()))

# ---- Reading options ------------------------------------------------------------------------------
ui.section("How the export is read")
with st.form("load_form"):
    c1, c2, c3 = st.columns(3)
    src = c1.selectbox("Time zone of the export", TIMEZONES, index=TIMEZONES.index(opts.source_tz)
                       if opts.source_tz in TIMEZONES else 0)
    disp = c2.selectbox("Show times in", TIMEZONES, index=TIMEZONES.index(opts.display_tz)
                        if opts.display_tz in TIMEZONES else 1)
    gap = c3.number_input("New run if the part returns after (min, 0 = only when a corner repeats)", 0.0, 1440.0,
                          value=float(opts.run_gap_min), step=5.0)
    c4, c5 = st.columns(2)
    zero = c4.toggle("All-zero readings are failed reads (missing)", value=opts.zero_as_missing)
    merge = c5.toggle(f"Merge look-alike Part IDs (O vs 0) · {len(h.get('similar_part_ids', []))} found",
                      value=opts.merge_lookalikes)
    if st.form_submit_button("Apply and re-read", icon=":material/refresh:", type="primary"):
        state.set_load_options(dataclasses.replace(opts, source_tz=src, display_tz=disp, run_gap_min=float(gap),
                                                   zero_as_missing=bool(zero), merge_lookalikes=bool(merge)))
        st.rerun()

with st.expander("Feature → corner mapping", icon=":material/alt_route:"):
    fm = pd.DataFrame(h.get("feature_mapping", []))
    st.dataframe(fm, hide_index=True, width="stretch")
    st.caption("Add rules if your features are named differently: a text contained in the feature name → the corner. "
               "Your rules are checked before the built-in ones.")
    rules_df = pd.DataFrame(list(opts.feature_map) or [], columns=["Contains", "Corner"])
    ed = st.data_editor(rules_df, num_rows="dynamic", hide_index=True, width="stretch", key="fm_editor",
                        column_config={"Contains": st.column_config.TextColumn(required=True),
                                       "Corner": st.column_config.SelectboxColumn(options=list(CORNERS), required=True)})
    if st.button("Apply mapping rules", icon=":material/check:"):
        fmap = tuple((str(x), str(y)) for x, y in ed.dropna().itertuples(index=False) if str(x).strip())
        state.set_load_options(dataclasses.replace(opts, feature_map=fmap))
        st.rerun()

with st.expander("Column mapping", icon=":material/view_column:"):
    first = state.files()[0]
    info = loading.inspect_file(first["data"], first["name"])
    cols = ["(automatic)"] + info["columns"]
    ov = dict(opts.column_overrides)
    picks = {}
    grid = st.columns(3)
    for i, canon in enumerate(["time", "part", "feature", "x", "y", "z"]):
        cur = ov.get(canon, "(automatic)")
        picks[canon] = grid[i % 3].selectbox(loading.CANONICAL_LABELS[canon], cols,
                                             index=cols.index(cur) if cur in cols else 0, key=f"colmap_{canon}",
                                             help=f"Detected: {info['mapping'].get(canon) or '—'}")
    if st.button("Apply column mapping", icon=":material/check:"):
        state.set_load_options(dataclasses.replace(
            opts, column_overrides=tuple((k, v) for k, v in picks.items() if v != "(automatic)")))
        st.rerun()

# ---- Geometry and shifts -------------------------------------------------------------------------------
left, right = st.columns([3, 2], gap="large")
with left:
    ui.section("Nominal corner coordinates", "CAD positions of the corners (mm). Left corners carry negative Y. Used "
                                             "for geometry, yaw, squareness and compensation.")
    nd = c.nominals.as_dict()
    rows = [{"Type": t, "Corner": k, "X": nd[t][k][0], "Y": nd[t][k][1]} for t in BATTERY_TYPES for k in CORNERS]
    ned = st.data_editor(pd.DataFrame(rows), hide_index=True, width="stretch", disabled=["Type", "Corner"],
                         key="nom_editor", column_config={"X": st.column_config.NumberColumn(format="%.2f"),
                                                          "Y": st.column_config.NumberColumn(format="%.2f")})
    b1, b2 = st.columns(2)
    if b1.button("Apply coordinates", icon=":material/check:", width="stretch"):
        new = {t: {} for t in BATTERY_TYPES}
        for _, row in ned.iterrows():
            new[row["Type"]][row["Corner"]] = (float(row["X"]), float(row["Y"]))
        state.set_nominals(Nominals.from_dict(new))
        st.rerun()
    if b2.button("Reset to the original values", icon=":material/restart_alt:", width="stretch"):
        state.set_nominals(Nominals.from_dict(DEFAULT_NOMINALS))
        st.rerun()
with right:
    ui.section("Shift plan")
    sp = c.settings.shifts
    with st.form("shift_form"):
        cols3 = st.columns(len(sp.starts))
        starts, names = [], []
        for i, (stt, nm) in enumerate(zip(sp.starts, sp.names)):
            names.append(cols3[i].text_input(f"Shift {i + 1} name", value=nm))
            starts.append(cols3[i].number_input("starts at (h)", 0, 23, value=int(stt), key=f"shift_start_{i}"))
        if st.form_submit_button("Apply shift plan", icon=":material/check:"):
            if len(set(starts)) == len(starts):
                state.set_shifts(ShiftPlan(starts=tuple(int(x) for x in starts),
                                           names=tuple(n.strip() or f"S{i + 1}" for i, n in enumerate(names))))
                st.rerun()
            else:
                st.error("Each shift needs a different start hour.")
    st.caption(" · ".join(sp.describe()))

# ---- Project file -------------------------------------------------------------------------------------------
ui.section("Project file", "Keeps your rules, reading options, nominal geometry, shift plan and intervention log. "
                           "Load it next time together with new exports.")
p1, p2 = st.columns(2)
with p1:
    ui.download_bytes("Download project file (.json)", project_to_json(opts, c.settings, state.events()).encode(),
                      "rickhard_project.json", "application/json", key="dl_project")
with p2:
    up = st.file_uploader("Load a project file", type=["json"], key="up_project", label_visibility="collapsed")
    if up is not None and st.session_state.get("_project_loaded") != up.file_id:
        try:
            lo, se, evs = project_from_json(up.getvalue().decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read the project file: {e}")
        else:
            state.set_load_options(lo)
            state.set_nominals(se.nominals)
            state.set_shifts(se.shifts)
            state.set_events(evs)
            rr = se.rules
            # sidebar widgets already exist in this run: apply their values at the start of the next one
            st.session_state["_pending_rules"] = {
                "sb_spec": rr.spec_limit, "sb_target": rr.fpy_target, "sb_diag": rr.diag_tol,
                "sb_excl": rr.incomplete_as_fail, "sb_near": rr.near_miss, "sb_zon": rr.z_in_status}
            st.session_state["_project_loaded"] = up.file_id
            st.toast("Project loaded.", icon=":material/check:")
            st.rerun()

# ---- Raw tables -----------------------------------------------------------------------------------------
with st.expander("All runs in the current filters", icon=":material/table_rows:"):
    cols = ["Date", "CalendarWeek", "PartID", "BatteryType", "RunNum", *[f"{k}_{ax}" for k in CORNERS for ax in "XY"],
            "CornersOutOfSpec", "MissingCorners", "WorstAbs", "Status"]
    st.dataframe(ui.status_styler(a.runs[cols], r.spec_limit), hide_index=True, width="stretch", height=420,
                 column_config=ui.run_table_config())
with st.expander("Readings as read from the file(s) — first 2,000", icon=":material/table_rows:"):
    st.dataframe(ds.rows.head(2000), hide_index=True, width="stretch", height=380)
