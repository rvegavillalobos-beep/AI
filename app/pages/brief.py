from datetime import datetime

import streamlit as st

from app import data, state, ui
from rickhard import brief

c = data.ctx()
a = c.a
ui.page_header(c, "brief", "One page that tells the story on its own: where FPY stands against the target, what was done "
                           "to the line and whether it worked, what still costs yield and what happens next.")
ui.require(a.first)

loop = data.fix_loop(c)

left, right = st.columns([3, 2], gap="large")
with left:
    title = st.text_input("Title", key=state.init("p_brief_title", brief.DEFAULT_TITLE))
    c1, c2 = st.columns(2)
    with c1:
        author = st.text_input("Prepared by (optional)", key=state.init("p_brief_author", ""))
    with c2:
        note = st.text_input("Note next to the date (optional)", key=state.init("p_brief_note", ""),
                             placeholder="e.g. weekly review, CW45")
with right:
    asks = st.text_area("What we need — one request per line (optional)", key=state.init("p_brief_asks", ""), height=122,
                        placeholder="e.g. 4 h maintenance window to shim the end-stop\n"
                                    "Supplier review of module squareness",
                        help="Shown as its own block at the end of the brief. Leave empty to leave the block out.")

body = brief.fragment(a, loop, title=title or brief.DEFAULT_TITLE, author=author, asks=asks, note=note,
                      demo=state.is_demo())
doc = brief.wrap(body, title or brief.DEFAULT_TITLE)

b1, b2 = st.columns([2, 3], vertical_alignment="center")
with b1:
    ui.download_bytes("Download the brief (HTML)", doc.encode("utf-8"),
                      f"ST020_FPY_brief_{datetime.now():%Y%m%d_%H%M}.html", "text/html", key="brief_dl", primary=True)
with b2:
    st.caption("One self-contained file: it opens in any browser, also offline, and can be mailed as it is. "
               "Print it from the browser (landscape) for a PDF — three pages that read like slides.")
if state.is_demo():
    st.caption(":orange[Demo data: the story below is invented, and the brief is stamped as demo.]")
if loop["score"]["done"] == 0:
    st.info("No change to the line is logged yet, so the brief has one set-up and no verdicts. Log what was done in "
            "the **Fix Loop** — that is what turns the numbers into a story.", icon=":material/autorenew:")
    st.page_link("app/pages/events.py", label="Open the Fix Loop", icon=":material/autorenew:")

st.html(f"<style>{brief.css()}\n.xb-prev {{background:#fff; border:1px solid rgba(128,128,128,.3); border-radius:12px; "
        "padding:28px 32px 24px 32px; margin-top:.4rem;}</style>")
st.html(f"<div class='xb-prev'>{body}</div>")
