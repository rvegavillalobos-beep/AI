import streamlit as st

from app import sidebar

st.html("""
<div class='rx-hero'>
  <h1>RICKHARD X</h1>
  <p>ST020 corner intelligence. Load the <i>Corner Cell Deviations</i> export and get first-pass yield, why
  modules fail, where the error comes from, whether the process is in control, and the X / Y / yaw correction
  that would help — with the evidence behind every number.</p>
</div>
""")

left, right = st.columns([3, 2], gap="large")
with left:
    with st.container(border=True):
        st.markdown("##### Load measurement exports")
        sidebar.uploader("up_main")
        st.caption("Two title rows, then columns for Time, Part ID, Featurename and X / Y (and Z) deviation — the "
                   "format is detected automatically, so other header positions, CSV with ';' and several files "
                   "covering overlapping periods also work.")
    st.button("Explore with demo data", icon=":material/science:", on_click=sidebar.load_demo, type="primary",
              help="A synthetic export in the same format, with a realistic process story (an end-stop adjustment, "
                   "guide wear, a night-shift offset, deformed modules, re-measurements). The numbers are invented.")
with right:
    st.markdown("##### Privacy")
    st.caption("Files stay in this browser session: nothing is stored on a server or sent anywhere. Download a "
               "project file in *Data & Settings* to keep your rules and intervention log.")

st.html("""
<div class='rx-feat'>
  <div><b>Command Center</b>FPY against target with confidence intervals, what the data says in plain language,
  and a ranked action plan with the FPY each action is worth.</div>
  <div><b>Error Anatomy</b>Every module's corner errors split into 8 physical modes — shift, yaw, stretch,
  parallelogram, trapezoid — so you know which ones a conveyor adjustment can fix.</div>
  <div><b>Compensation Studio</b>Median-based, rigid-fit and a robust optimum, with how sure the gain is and a
  printable work instruction for the line.</div>
  <div><b>Path to Target</b>What-if levers on your own modules: how much scatter you must remove to reach the target.</div>
  <div><b>SPC Monitor</b>I-MR and weekly charts with run rules, and dated process shifts found automatically.</div>
  <div><b>Interventions</b>Log what was changed on the line and get a before / after verdict with statistics.</div>
  <div><b>Time Patterns</b>Shift, weekday × hour and start-up after line stops.</div>
  <div><b>Re-measurement & MSA</b>What re-measuring tells you about repeatability (%GRR) from production data.</div>
  <div><b>Corner Lab · Geometry · Shape</b>Targets per corner, capability (Cp/Cpk/Pp/Ppk), module outlines, the
  drift movie, squareness and its link to NOK — everything the original app had, deeper.</div>
</div>
""")
