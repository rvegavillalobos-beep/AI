"""Run every page of the app with the Streamlit stub (no browser, no server).

Usage:  python tests/run_pages.py [--json]
Prints one line per scenario and exits non-zero if any page raised.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tests", "stub_streamlit"))
sys.path.insert(1, ROOT)
os.chdir(ROOT)
warnings.simplefilter("ignore")

import streamlit as st  # noqa: E402  (the stub)

assert hasattr(st, "run_app"), "the Streamlit stub was not picked up"

PAGES = ["command", "corners", "geometry", "shape", "drift", "anatomy", "spc", "patterns", "remeasure", "height",
         "target", "compensation", "events", "passport", "data", "reports"]


def scenario(name, page=None, click=(), select=None, uploads=None, before=None):
    st.CONFIG["click"] = set(click)
    st.CONFIG["select"] = dict(select or {})
    st.CONFIG["uploads"] = dict(uploads or {})
    if before:
        st._reset_run()  # between runs the user may change any widget
        before()
    t = time.time()
    try:
        status = st.run_app("streamlit_app.py", page)
        err = None
    except Exception as e:  # noqa: BLE001
        status, err = "ERROR", "".join(traceback.format_exception(type(e), e, e.__traceback__)[-6:])
    dt = time.time() - t
    errors = [x for x in st.LOG if x[0] in ("error", "exception")]
    st.LOG.clear()
    return {"scenario": name, "status": status, "seconds": round(dt, 2), "error": err,
            "st_errors": [str(e[1])[:300] for e in errors]}


def main() -> int:
    results = []
    # start page, then the demo button
    results.append(scenario("start page", None))
    results.append(scenario("load demo", None, click={"Explore with demo data"}))
    for p in PAGES:
        results.append(scenario(p, p))
    # interactions
    results.append(scenario("geometry: select a row → focus", "geometry", select={"geo_table": [0]}))
    results.append(scenario("passport with focus", "passport"))
    results.append(scenario("compensation: best yaw + use + log", "compensation",
                            click={"Best yaw for this X / Y", "Use in Path to Target", "Log as intervention"}))
    results.append(scenario("target with chosen correction", "target"))
    results.append(scenario("events after logging", "events"))
    results.append(scenario("reports: build both", "reports", click={"rep_xlsx_go", "rep_html_go"}))
    results.append(scenario("data: apply reading options", "data", click={"Apply and re-read"}))

    def type_m_only():
        st.session_state["sb_types"] = "Type M"
    results.append(scenario("command, Type M only", "command", before=type_m_only))
    results.append(scenario("compensation, Type M only", "compensation"))

    def narrow_weeks():
        st.session_state["sb_types"] = "All types"
        labels = st.session_state["sb_weeks"]
        st.session_state["sb_weeks"] = (labels[1], labels[1])
    results.append(scenario("all pages, one week", "command", before=narrow_weeks))
    for p in PAGES[1:]:
        results.append(scenario(f"{p} (one week)", p))

    # a fresh session opened with "?demo" in the address lands on the demo data
    def fresh_with_demo_param():
        st.session_state.clear()
        st.query_params["demo"] = ""
    r = scenario("fresh session with ?demo", "corners", before=fresh_with_demo_param)
    if st.session_state.get("_ctx") is None or st.session_state.get("_page") != "corners":
        r["status"], r["error"] = "ERROR", "?demo did not open the demo data on the requested page"
    st.query_params.clear()
    results.append(r)

    bad = [r for r in results if r["status"] == "ERROR" or r["st_errors"]]
    if "--json" in sys.argv:
        print(json.dumps(results, indent=1, default=str))
    else:
        for r in results:
            mark = "OK " if r["status"] != "ERROR" and not r["st_errors"] else "ERR"
            print(f"{mark} {r['scenario']:<40} {r['status']:<8} {r['seconds']:>6.2f}s")
            if r["error"]:
                print(r["error"])
            for e in r["st_errors"]:
                print("    st.error:", e)
    print(f"\n{len(results) - len(bad)}/{len(results)} scenarios clean")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
