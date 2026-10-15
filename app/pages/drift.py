import streamlit as st

from app import data, state, ui
from rickhard.charts import drift as cd

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "drift", "How the placement of the modules moves over time: the original centroid path and yaw "
                           "views, plus the least-squares placement with automatically dated process shifts.")
ui.require(a.first)
vec = a.first

weeks = vec[["WeekKey", "CalendarWeek"]].drop_duplicates().sort_values("WeekKey")
n_weeks = len(weeks)
ctl = st.columns([1.4, 1.4, 1.2], vertical_alignment="bottom")
with ctl[0]:
    state.init("p_dr_n", min(6, max(1, n_weeks)))
    state.clamp("p_dr_n", 1, max(1, n_weeks), min(6, n_weeks))
    n = st.slider("Centroid path: last N weeks", 1, max(2, n_weeks), key="p_dr_n", disabled=n_weeks < 2)
with ctl[1]:
    counts = vec.groupby("WeekKey").size()
    state.init("p_dr_minvol", 1)
    state.clamp("p_dr_minvol", 1, max(2, int(counts.max())), 1)
    min_vol = st.slider("Bottom charts: minimum weekly volume", 1, max(2, int(counts.max())), key="p_dr_minvol")
with ctl[2]:
    split = st.toggle("Split by battery type", key=state.init("p_dr_split", True))

recent = vec[vec["WeekKey"].isin(weeks["WeekKey"].tail(int(n)))]
valid_weeks = counts[counts >= min_vol].index
left, right = st.columns(2, gap="large")
with left:
    ui.section("Weekly centroid drift (X–Y offset)", "Mean of the four corner deviations per module (small dots) and "
                                                     "its weekly average (path). Y points down, as on the line.")
    ui.chart(cd.centroid_path(recent, split), "dr_path")
with right:
    ui.section("Weekly median offset and yaw", "Median of the module centre offset (X, Y) and of the yaw, week by "
                                               "week and per battery type — what a median-based correction would "
                                               "have chased each week.")
    ui.chart(cd.weekly_medians(vec[vec["WeekKey"].isin(valid_weeks)]), "dr_medians")

ui.section("Drift magnitude and yaw of every module")
ui.chart(cd.magnitude_yaw(vec[vec["WeekKey"].isin(valid_weeks)], height=480), "dr_magyaw")

# ---- Placement over time with change points ---------------------------------------------------
ui.section("Placement over time and detected process shifts",
           "Shift X / Y and yaw from a least-squares fit of the whole module (all four corners vote). Black steps: the "
           "level between automatically detected shifts. Pink dashed: interventions you logged.")
types = c.types
if len(types) > 1:
    state.ensure_option("p_dr_type", types, types[0])
    t = st.segmented_control("Battery type", types, key="p_dr_type") or types[0]
else:
    t = types[0]
cp = data.change_points(c.key, a)
cps = {col: cp["idx"].get((t, col), []) for col in data.PLACEMENT_VARS}
ev = c.events
ev_t = ev[ev["Scope"].isin(["All", t])] if len(ev) else ev
state.init("p_dr_win", 25)
win = st.slider("Rolling median window (modules)", 5, 151, step=2, key="p_dr_win")
ui.chart(cd.placement_timeseries(vec, t, cps, ev_t, window=int(win)), "dr_ts")

tab = cp["table"]
if len(tab):
    ui.section("Detected shifts")
    show = tab.copy()
    show["Unit"] = ["°" if "Yaw" in v else "mm" for v in show["Variable"]]
    cfg = st.column_config
    st.dataframe(show[["Date", "Variable", "Before", "After", "Shift", "Unit", "p", "N before", "N after"]],
                 hide_index=True, width="stretch", column_config={
                     "Date": cfg.DatetimeColumn("From", format="YYYY-MM-DD HH:mm"),
                     "Before": cfg.NumberColumn("Level before", format="%+.3f"),
                     "After": cfg.NumberColumn("Level after", format="%+.3f"),
                     "Shift": cfg.NumberColumn("Shift", format="%+.3f"), "p": cfg.NumberColumn("p-value", format="%.1e")})
    st.caption("Log what was done at those dates in **Interventions** to measure each change; a shift you cannot "
               "explain is worth an investigation.")
else:
    st.caption("No significant level shift detected in the placement.")

with st.expander("Vector & rotation summary table (original)", icon=":material/table_rows:"):
    tbl = vec[vec["WeekKey"].isin(valid_weeks)][["Date", "CalendarWeek", "PartID", "BatteryType", "Centroid_X",
                                                 "Centroid_Y", "Vector_Magnitude", "Rotation_Angle", "Fit_Tx", "Fit_Ty",
                                                 "Fit_Rot", "Status"]]
    cfg = st.column_config
    st.dataframe(ui.status_styler(tbl, r.spec_limit, precision=3), hide_index=True, width="stretch", height=380,
                 column_config={"Date": cfg.DatetimeColumn("Date (MX)", format="YYYY-MM-DD HH:mm"),
                                "Centroid_X": "Centroid X", "Centroid_Y": "Centroid Y",
                                "Vector_Magnitude": "Magnitude R", "Rotation_Angle": "Yaw (original) °",
                                "Fit_Tx": "Shift X (fit)", "Fit_Ty": "Shift Y (fit)", "Fit_Rot": "Yaw (fit) °"})
    st.caption("'Yaw (original)' is the angle of the rear-to-front midpoint line, as in the original app; "
               "'(fit)' values come from a least-squares rigid fit of all four corners.")
