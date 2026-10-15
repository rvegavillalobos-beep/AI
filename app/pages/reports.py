from datetime import datetime

import pandas as pd
import streamlit as st

from app import data, state, ui
from rickhard import report, report_html

c = data.ctx()
a, r = c.a, c.rules
ui.page_header(c, "reports", "One click to the files you send: a formatted Excel workbook for the team and a "
                             "management report that opens in any browser.")
ui.require(a.first)

bundle = data.insight_bundle(c.key, a, c.ds)
rb = data.remeasure_bundle(c.key, 120.0, a)
extras = dict(bundle["extras"])
extras["repeatability"] = rb["rep"]
chosen = data.chosen_compensation()
comp_rows = pd.DataFrame([{"BatteryType": t, "Method": v["method"], "X [mm]": v["setting"][0],
                           "Y [mm]": v["setting"][1], "Yaw [°]": v["setting"][2]} for t, v in chosen.items()]) \
    if chosen else None
stamp = datetime.now().strftime("%Y%m%d_%H%M")
sig = c.key + "|" + repr(sorted(chosen.items())) + "|" + repr(state.events())

left, right = st.columns(2, gap="large")
with left:
    with st.container(border=True):
        st.markdown("##### Excel workbook")
        st.caption("Summary with key figures, findings and the action plan; weekly FPY; first runs and all runs with "
                   "out-of-tolerance cells in red; capability; failure mechanisms; squareness; vector drift; error "
                   "budget; process shifts; SPC alarms; repeatability; interventions; data health.")
        title = st.text_input("Report title", key=state.init("p_rep_title", "ST020 corner deviation report"))
        if st.button("Prepare Excel", icon=":material/table_view:", key="rep_xlsx_go", width="stretch"):
            with st.spinner("Writing the workbook…"):
                st.session_state["_rep_xlsx"] = (sig, report.build_excel(
                    a, bundle["insights"], bundle["actions"], extras, c.ds.health, c.events, comp_rows, title=title))
        got = st.session_state.get("_rep_xlsx")
        if got and got[0] == sig:
            ui.download_bytes("Download Excel report", got[1], f"RICKHARD_X_report_{stamp}.xlsx",
                              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="rep_xlsx_dl",
                              primary=True)
with right:
    with st.container(border=True):
        st.markdown("##### Management report (HTML)")
        st.caption("KPIs, what the data says, the ranked action plan and the key interactive charts on one page. "
                   "Print it to PDF from the browser if you need a PDF.")
        note = st.text_input("Note on the cover line (optional)", key=state.init("p_rep_note", ""))
        offline = st.toggle("Works offline (embeds the chart library, ~5 MB)", key=state.init("p_rep_off", False))
        if st.button("Prepare HTML report", icon=":material/summarize:", key="rep_html_go", width="stretch"):
            with st.spinner("Building the report…"):
                html_doc = report_html.build_html(a, bundle["insights"], bundle["actions"], c.events, comp_rows,
                                                  title=st.session_state.get("p_rep_title", "ST020 corner deviation report"),
                                                  offline=offline, note=note)
                st.session_state["_rep_html"] = (sig + f"|{offline}|{note}", html_doc.encode("utf-8"))
        got = st.session_state.get("_rep_html")
        if got and got[0] == sig + f"|{offline}|{note}":
            ui.download_bytes("Download HTML report", got[1], f"RICKHARD_X_report_{stamp}.html", "text/html",
                              key="rep_html_dl", primary=True)

with st.container(border=True):
    st.markdown("##### Raw tables (CSV)")
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.download_bytes("First runs (CSV)", a.first.drop(columns=["_mod_key"], errors="ignore").to_csv(index=False)
                          .encode("utf-8"), f"first_runs_{stamp}.csv", "text/csv", key="dl_first")
    with c2:
        ui.download_bytes("All runs (CSV)", a.runs.drop(columns=["_mod_key"], errors="ignore").to_csv(index=False)
                          .encode("utf-8"), f"all_runs_{stamp}.csv", "text/csv", key="dl_runs")
    with c3:
        ui.download_bytes("Weekly FPY (CSV)", a.weekly.to_csv(index=False).encode("utf-8"), f"weekly_fpy_{stamp}.csv",
                          "text/csv", key="dl_weekly")

if not chosen:
    st.caption("Tip: pick a correction in **Compensation Studio** and press *Use in Path to Target* — it is then "
               "included in both reports.")
