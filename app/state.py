"""Session state: data source, load options, events, focus and page widgets.

Streamlit forgets the state of widgets that are not rendered on the current
page. Page widgets therefore use keys starting with ``p_`` and are re-assigned
at the top of every run (``keep_widget_state``), so filters and settings
survive navigating between pages.
"""

from __future__ import annotations

import hashlib

import streamlit as st

from rickhard.config import LoadOptions, Nominals, ShiftPlan

PERSIST = "p_"


# ---------------------------------------------------------------------------
# Widget persistence
# ---------------------------------------------------------------------------

def keep_widget_state() -> None:
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(PERSIST):
            st.session_state[k] = st.session_state[k]


def init(key: str, value) -> str:
    """Give a persisted widget its default once (the widget must then not
    pass value/index/default itself)."""
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def ensure_option(key: str, options: list, fallback) -> str:
    """Reset a persisted selection that is no longer a valid option."""
    if key not in st.session_state or st.session_state[key] not in options:
        st.session_state[key] = fallback
    return key


def clamp(key: str, lo, hi, fallback=None) -> str:
    v = st.session_state.get(key, fallback)
    if v is None:
        v = fallback if fallback is not None else lo
    try:
        st.session_state[key] = type(lo)(min(max(v, lo), hi))
    except TypeError:
        st.session_state[key] = fallback if fallback is not None else lo
    return key


def reset_page_widgets() -> None:
    """Forget page settings and data-dependent filters (new data loaded)."""
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (k.startswith(PERSIST) or k in ("sb_types", "sb_weeks", "focus")):
            del st.session_state[k]


# ---------------------------------------------------------------------------
# Data source
# ---------------------------------------------------------------------------

def files() -> list[dict]:
    return st.session_state.get("_files") or []


def files_hash() -> str:
    return st.session_state.get("_files_hash", "")


def set_files(items: list[dict], demo: bool = False, events: list[dict] | None = None) -> None:
    h = hashlib.sha1()
    for it in items:
        h.update(it["name"].encode())
        h.update(it["data"])
    st.session_state["_files"] = items
    st.session_state["_files_hash"] = h.hexdigest()[:16]
    st.session_state["_demo"] = bool(demo)
    if events is not None:
        st.session_state["events"] = events
    reset_page_widgets()


def is_demo() -> bool:
    return bool(st.session_state.get("_demo"))


def clear_files() -> None:
    for k in ("_files", "_files_hash", "_demo"):
        st.session_state.pop(k, None)
    reset_page_widgets()


def load_options() -> LoadOptions:
    return st.session_state.get("load_opts") or LoadOptions()


def set_load_options(opts: LoadOptions) -> None:
    st.session_state["load_opts"] = opts


def nominals() -> Nominals:
    return st.session_state.get("nominals") or Nominals()


def set_nominals(n: Nominals) -> None:
    st.session_state["nominals"] = n


def shifts() -> ShiftPlan:
    return st.session_state.get("shifts") or ShiftPlan()


def set_shifts(s: ShiftPlan) -> None:
    st.session_state["shifts"] = s


# ---------------------------------------------------------------------------
# Events and focus
# ---------------------------------------------------------------------------

def events() -> list[dict]:
    return list(st.session_state.get("events") or [])


def set_events(records: list[dict]) -> None:
    st.session_state["events"] = list(records)


def add_event(record: dict) -> None:
    ev = events()
    ev.append(record)
    set_events(ev)


def focus() -> str | None:
    return st.session_state.get("focus")


def set_focus(key: str | None) -> None:
    if key:
        st.session_state["focus"] = key
    else:
        st.session_state.pop("focus", None)
