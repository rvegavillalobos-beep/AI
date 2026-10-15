"""Sidebar: data source, filters and quality rules (rendered on every page)."""

from __future__ import annotations

import html

import streamlit as st

from rickhard import __version__, demo, loading
from rickhard.charts import svg
from rickhard.config import Rules, Scope, Settings
from rickhard.constants import BATTERY_TYPES

from . import data, state


def _store_uploads(widget_key: str) -> None:
    files = st.session_state.get(widget_key) or []
    if not isinstance(files, list):
        files = [files]
    items = [{"name": f.name, "data": f.getvalue()} for f in files if f is not None]
    if items:
        state.set_files(items, demo=False)


def load_demo() -> None:
    state.set_files([{"name": "demo_export.xlsx", "data": data.demo_bytes(data.CODE_VERSION)}], demo=True,
                    events=demo.demo_events())


def uploader(key: str, label: str = "Measurement export(s) — Excel or CSV") -> None:
    st.file_uploader(label, type=["xlsx", "xlsm", "xls", "csv", "txt"], accept_multiple_files=True, key=key,
                     on_change=_store_uploads, args=(key,),
                     help="One or more 'Corner Cell Deviations' exports. Overlapping periods are fine: exact "
                          "duplicate rows are removed.")


def brand() -> None:
    st.sidebar.html(f"<div style='display:flex;gap:10px;align-items:center;margin:-.4rem 0 .2rem 0'>"
                    f"{svg.logo_svg(34)}<div><div style='font-weight:750;font-size:1.05rem;line-height:1.1'>RICKHARD X</div>"
                    f"<div style='font-size:.74rem;opacity:.65'>ST020 corner intelligence · v{__version__}</div></div></div>")


def file_card(ds) -> None:
    sb = st.sidebar
    names = [f["name"] for f in state.files()]
    with sb.container(border=True):
        st.markdown("**" + html.escape(names[0] if len(names) == 1 else f"{len(names)} files") + "**")
        if ds is not None:
            h = ds.health
            st.caption(f"{h['modules']:,} modules · {h['module_runs']:,} runs · "
                       f"{h['date_min']:%b %d} – {h['date_max']:%b %d, %Y}")
        if state.is_demo():
            st.caption(":orange[Synthetic demo data — the numbers are invented.]")
        with st.popover("Replace / add files", icon=":material/swap_horiz:", width="stretch"):
            uploader("up_sidebar", "Upload new export(s)")
            st.button("Clear data", icon=":material/close:", on_click=state.clear_files, width="stretch",
                      key="sb_clear")


def _v(key: str, value, name: str = "value") -> dict:
    """Pass a default only when the widget has no state yet (avoids Streamlit's
    'default value but also set via Session State' warning)."""
    return {} if key in st.session_state else {name: value}


def filters_and_rules(ds) -> tuple[Settings, Scope, str, int]:
    sb = st.sidebar
    ss = st.session_state
    runs = ds.runs
    present = [t for t in BATTERY_TYPES if (runs["BatteryType"] == t).any()]
    type_opts = (["All types"] if len(present) > 1 else []) + present

    with sb:
        st.subheader("Filters", divider="gray")
        if ss.get("sb_types") not in type_opts:
            ss.pop("sb_types", None)
        choice = st.segmented_control("Battery type", type_opts, key="sb_types", **_v("sb_types", type_opts[0], "default"))
        if choice is None:
            choice = type_opts[0]
        types = tuple(present) if choice == "All types" else (choice,)

        weeks = runs[["WeekKey", "CalendarWeek"]].dropna().drop_duplicates().sort_values("WeekKey")
        labels, keys = weeks["CalendarWeek"].tolist(), [int(k) for k in weeks["WeekKey"].tolist()]
        if len(labels) > 1:
            cur = ss.get("sb_weeks")
            if cur is not None and (not isinstance(cur, (tuple, list)) or any(w not in labels for w in cur)):
                ss.pop("sb_weeks")
            w0, w1 = st.select_slider("Calendar weeks", options=labels, key="sb_weeks",
                                      **_v("sb_weeks", (labels[0], labels[-1])))
            wf, wt = keys[labels.index(w0)], keys[labels.index(w1)]
        else:
            w0 = w1 = labels[0] if labels else "–"
            wf = wt = keys[0] if keys else None
            st.caption(f"Calendar week: {w0}")
        week_label = w0 if w0 == w1 else f"{w0} – {w1}"

        st.subheader("Quality rules", divider="gray")
        file_lim = ds.limits.get("X") if ds.limits.get("X") == ds.limits.get("Y") else None
        default_spec = float(file_lim[1]) if file_lim and abs(file_lim[0] + file_lim[1]) < 1e-9 else 3.0
        spec = st.slider("X/Y tolerance (± mm)", 0.5, 6.0, step=0.1, key="sb_spec", **_v("sb_spec", default_spec),
                         help="A module passes when every corner's X and Y deviation is within ± this value.")
        if file_lim and abs(file_lim[1] - spec) > 1e-9:
            st.caption(f":orange[The export's own limit is ±{file_lim[1]:g} mm.]")
        target = st.slider("FPY target (%)", 50.0, 100.0, step=0.5, key="sb_target", **_v("sb_target", 90.0))
        excl = st.toggle("Count incomplete first runs as NOK", key="sb_excl", **_v("sb_excl", True),
                         help="On (original rule): a first measurement with a missing corner is a FAIL. "
                              "Off: it is kept apart as INCOMPLETE (it still lowers FPY).")
        with st.expander("More rules", icon=":material/rule:"):
            diag = st.slider("Max. diagonal delta (mm)", 0.5, 10.0, step=0.1, key="sb_diag", **_v("sb_diag", 1.5),
                             help="Above this a module is DEFORMED in the squareness analysis.")
            near = st.slider("Near-miss margin (mm)", 0.0, 1.5, step=0.05, key="sb_near", **_v("sb_near", 0.5),
                             help="A PASS whose worst corner has less than this left to the limit is a near miss.")
            z_on, z_lo, z_hi = False, -1.25, 1.21
            if ds.has_z:
                zl = ds.limits.get("Z", (-1.25, 1.21))
                z_on = st.toggle("Also judge Z (height)", key="sb_zon", **_v("sb_zon", False),
                                 help="Off (original rule): only X and Y decide PASS/FAIL.")
                z_lo, z_hi = float(zl[0]), float(zl[1])
            ma = st.slider("Moving average (weeks)", 2, 10, key="sb_ma", **_v("sb_ma", 3))
        st.caption(f"RICKHARD X v{__version__} · analytics run in your browser session only")

    rules = Rules(spec_limit=float(spec), fpy_target=float(target), diag_tol=float(diag),
                  incomplete_as_fail=bool(excl), near_miss=float(near), z_in_status=bool(z_on),
                  z_lo=z_lo, z_hi=z_hi)
    settings = Settings(rules=rules, nominals=state.nominals(), shifts=state.shifts())
    scope = Scope(types=types, week_from=wf, week_to=wt)
    return settings, scope, week_label, int(ma)


def describe_load_error(e: Exception) -> str:
    if isinstance(e, loading.DataFormatError):
        return f"**This doesn't look like a corner-deviation export.** {e}"
    return f"**The file could not be read.** {type(e).__name__}: {e}"
