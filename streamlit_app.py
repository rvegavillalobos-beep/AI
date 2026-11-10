"""ST020 Corner Analytics.

Run with:  streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="ST020 Corner Analytics", page_icon=":material/crop_free:",
                   layout="wide", initial_sidebar_state="auto")

_ROOT = Path(__file__).resolve().parent
_PACKAGES = ("app", "rickhard")


def _source_signature() -> tuple:
    files = [p for pkg in _PACKAGES for p in (_ROOT / pkg).rglob("*.py")]
    stats = [p.stat() for p in files]
    return len(files), max((s.st_mtime_ns for s in stats), default=0), sum(s.st_size for s in stats)


def refresh_code() -> bool:
    """Drop the app's own modules when their source changed under a running app.

    A host that pulls an update into a running app (Streamlit Community Cloud) re-reads this script but can keep
    the packages it imported before, so new code meets old modules — a page that exists here and not in the old
    ``app.ui``, for one. Comparing a signature of the source files on every run and re-importing on a change keeps
    the two in step. True when modules were dropped."""
    sig = _source_signature()
    if getattr(sys, "_st020_sources", None) == sig:
        return False
    sys._st020_sources = sig
    stale = [m for m in list(sys.modules) if m.split(".")[0] in _PACKAGES]
    for m in stale:
        del sys.modules[m]
    if stale:                       # results computed by the old code may not fit the new one
        st.cache_data.clear()
        st.cache_resource.clear()
    return bool(stale)


refresh_code()

from app import data, sidebar, state, ui  # noqa: E402  (page config and the refresh must come first)
from rickhard.charts import theme  # noqa: E402

state.renew_settings_objects()
state.keep_widget_state()
for _k, _v in (st.session_state.pop("_pending_rules", None) or {}).items():
    st.session_state[_k] = _v
ui.inject_css()
try:
    theme.set_dark(st.context.theme.type == "dark")
except Exception:  # noqa: BLE001 - older Streamlit without st.context.theme
    theme.set_dark(False)

sidebar.brand()

# "?demo" in the address opens the app straight on the demo data (handy for showing it to others)
if "demo" in st.query_params and not st.session_state.get("_qp_demo"):
    st.session_state["_qp_demo"] = True
    if not state.files():
        sidebar.load_demo()

context = None
if state.files():
    try:
        ds, ds_key = data.get_dataset()
    except Exception as e:  # noqa: BLE001 - show any reading problem to the user
        st.error(sidebar.describe_load_error(e), icon=":material/error:")
        sidebar.file_card(None)
        ds = None
    if ds is not None:
        if ds.runs.empty:
            st.warning("The file was read, but it contains no corner measurements with a valid time.")
            sidebar.file_card(ds)
        else:
            sidebar.file_card(ds)
            settings, scope, week_label, ma = sidebar.filters_and_rules(ds)
            context = data.make_context(ds, ds_key, settings, scope, ma, week_label)
st.session_state["_ctx"] = context


def page(key: str, default: bool = False) -> st.Page:
    return st.Page(f"app/pages/{key}.py", title=ui.PAGE_TITLES[key], icon=ui.PAGE_ICONS[key], url_path=key,
                   default=default)


if context is None:
    nav = st.navigation([st.Page("app/pages/home.py", title="Start", icon=":material/upload_file:", default=True)],
                        position="hidden")
else:
    sections = {
        "Overview": [page("command", default=True), page("events"), page("brief")],
        "Analyze": [page("corners"), page("geometry"), page("shape"), page("drift"), page("anatomy")],
        "Monitor": [page("spc"), page("patterns")] + ([page("height")] if context.ds.has_z else []),
        "Improve": [page("target"), page("compensation")],
        "Tools": [page("passport"), page("data"), page("reports")],
    }
    nav = st.navigation(sections, position="top")

st.session_state["_page"] = getattr(nav, "url_path", "") or ""
nav.run()
