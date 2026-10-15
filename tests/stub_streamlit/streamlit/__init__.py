"""A minimal stand-in for Streamlit used by the smoke tests when Streamlit is
not installed. It runs the real app scripts top to bottom and checks:

* every call uses parameters the real API accepts (explicit signatures),
* widget values follow session-state rules (key wins over default),
* a widget's session-state value is not modified after the widget was
  created in the same run (Streamlit raises in that case),
* every Plotly figure serialises and every pandas Styler renders,
* download buttons receive bytes/str.

It is not a renderer: it only executes and validates.
"""

from __future__ import annotations

import contextlib
import runpy
import types
from collections.abc import MutableMapping

# ---------------------------------------------------------------------------
# Control-flow exceptions
# ---------------------------------------------------------------------------


class StopException(Exception):
    pass


class RerunException(Exception):
    pass


class StreamlitAPIException(Exception):
    pass


LOG: list[tuple] = []          # (function, info)
FIGURES: list = []
_INSTANTIATED: set = set()     # widget keys created in the current run
_IN_CALLBACK = [False]
CONFIG = {"page": None, "select": {}, "click": set(), "uploads": {}}


class SessionState(MutableMapping):
    def __init__(self):
        object.__setattr__(self, "_d", {})

    def __getitem__(self, k):
        return self._d[k]

    def __setitem__(self, k, v):
        if k in _INSTANTIATED and not _IN_CALLBACK[0]:
            raise StreamlitAPIException(f"st.session_state.{k} cannot be modified after the widget with key {k} "
                                        "is instantiated.")
        self._d[k] = v

    def __delitem__(self, k):
        del self._d[k]

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


session_state = SessionState()


def _reset_run():
    _INSTANTIATED.clear()


def _widget(key, default):
    """Session-state semantics of a widget."""
    if key is None:
        return default
    if key in session_state:
        val = session_state[key]
    else:
        val = default
        session_state._d[key] = val
    _INSTANTIATED.add(key)
    return val


def _callback(fn, args=None, kwargs=None):
    if fn is None:
        return
    _IN_CALLBACK[0] = True
    try:
        fn(*(args or ()), **(kwargs or {}))
    finally:
        _IN_CALLBACK[0] = False


def _check_label_visibility(v):
    assert v in ("visible", "hidden", "collapsed"), v


def _check_width(width):
    assert width is None or width in ("stretch", "content") or isinstance(width, int), width


# ---------------------------------------------------------------------------
# Elements (methods shared by st, st.sidebar and every container)
# ---------------------------------------------------------------------------

class _Selection:
    def __init__(self, rows):
        self.rows = rows
        self.columns = []
        self.cells = []


class _Event(dict):
    def __init__(self, rows):
        super().__init__(selection={"rows": rows, "columns": []})
        self.selection = _Selection(rows)


class DeltaGenerator:
    # ---- text ------------------------------------------------------------------------
    def title(self, body, anchor=None, *, help=None, width="stretch"):
        LOG.append(("title", str(body)))

    def header(self, body, anchor=None, *, help=None, divider=False, width="stretch"):
        LOG.append(("header", str(body)))

    def subheader(self, body, anchor=None, *, help=None, divider=False, width="stretch"):
        assert divider in (False, True, "blue", "green", "orange", "red", "violet", "gray", "grey", "rainbow")
        LOG.append(("subheader", str(body)))

    def markdown(self, body, unsafe_allow_html=False, *, help=None, width="stretch"):
        assert isinstance(body, str)
        LOG.append(("markdown", body))

    def badge(self, label, *, icon=None, color="blue", width="content", help=None):
        assert color in ("red", "orange", "yellow", "blue", "green", "violet", "gray", "grey", "rainbow", "primary"), color
        LOG.append(("badge", str(label)))

    def caption(self, body, unsafe_allow_html=False, *, help=None, width="stretch"):
        assert isinstance(body, str)
        LOG.append(("caption", body))

    def html(self, body, *, width="stretch"):
        assert isinstance(body, str)
        LOG.append(("html", body))

    def write(self, *args, unsafe_allow_html=False):
        LOG.append(("write", args))

    def json(self, body, *, expanded=True, width="stretch"):
        LOG.append(("json", body))

    def code(self, body, language="python", *, line_numbers=False, wrap_lines=False, height=None, width="stretch"):
        LOG.append(("code", body))

    def divider(self, *, width="stretch"):
        LOG.append(("divider", None))

    def _msg(self, kind, body, icon=None, width="stretch"):
        assert isinstance(body, str), body
        LOG.append((kind, body))

    def info(self, body, *, icon=None, width="stretch"):
        self._msg("info", body, icon)

    def warning(self, body, *, icon=None, width="stretch"):
        self._msg("warning", body, icon)

    def error(self, body, *, icon=None, width="stretch"):
        self._msg("error", body, icon)

    def success(self, body, *, icon=None, width="stretch"):
        self._msg("success", body, icon)

    def exception(self, exception, width="stretch"):
        LOG.append(("exception", repr(exception)))

    # ---- data --------------------------------------------------------------------------
    def metric(self, label, value, delta=None, delta_color="normal", *, help=None, label_visibility="visible",
               border=False, width="stretch", height="content", chart_data=None, chart_type="line"):
        assert delta_color in ("normal", "inverse", "off")
        _check_label_visibility(label_visibility)
        LOG.append(("metric", (label, value, delta)))

    def dataframe(self, data=None, width="stretch", height="auto", *, use_container_width=None, hide_index=None,
                  column_order=None, column_config=None, key=None, on_select="ignore", selection_mode="multi-row",
                  row_height=None, placeholder=None):
        _check_width(width)
        _render_table(data)
        if column_config:
            _check_column_config(column_config, data)
        if on_select != "ignore":
            assert selection_mode in ("single-row", "multi-row", "single-column", "multi-column", "single-cell",
                                      "multi-cell")
            rows = CONFIG["select"].get(key, [])
            return _Event(rows)
        return None

    def data_editor(self, data, *, width="stretch", height="auto", use_container_width=None, hide_index=None,
                    column_order=None, column_config=None, num_rows="fixed", disabled=False, key=None,
                    on_change=None, args=None, kwargs=None, row_height=None, placeholder=None):
        import pandas as pd
        assert num_rows in ("fixed", "dynamic", "add", "delete")
        assert isinstance(data, pd.DataFrame)
        if column_config:
            _check_column_config(column_config, data)
        if key is not None:
            _INSTANTIATED.add(key)
        return data.copy()

    def table(self, data=None):
        _render_table(data)

    def plotly_chart(self, figure_or_data, use_container_width=None, *, width="stretch", height="content",
                     theme="streamlit", key=None, on_select="ignore", selection_mode=("points", "box", "lasso"),
                     config=None):
        import plotly.graph_objects as go
        assert isinstance(figure_or_data, go.Figure), type(figure_or_data)
        _check_width(width)
        assert theme in ("streamlit", None)
        figure_or_data.to_json()
        FIGURES.append((key, figure_or_data))
        if key is not None:
            assert key not in _CHART_KEYS, f"duplicate chart key {key}"
            _CHART_KEYS.add(key)
        LOG.append(("plotly_chart", key))
        return None

    def image(self, image, caption=None, width="content", *, use_container_width=None, clamp=False, channels="RGB",
              output_format="auto"):
        LOG.append(("image", caption))

    # ---- widgets ---------------------------------------------------------------------------
    def button(self, label, key=None, help=None, on_click=None, args=None, kwargs=None, *, type="secondary",
               icon=None, disabled=False, use_container_width=None, width="content", shortcut=None):
        assert type in ("primary", "secondary", "tertiary")
        _check_width(width)
        k = key or label
        clicked = k in CONFIG["click"]
        if clicked:
            CONFIG["click"].discard(k)  # a button is True only in the run right after the click
            _callback(on_click, args, kwargs)
        return clicked

    def download_button(self, label, data, file_name=None, mime=None, key=None, help=None, on_click="rerun",
                        args=None, kwargs=None, *, type="secondary", icon=None, disabled=False,
                        use_container_width=None, width="content", shortcut=None):
        assert isinstance(data, (bytes, str)), type(data)
        assert len(data) > 0
        assert type in ("primary", "secondary", "tertiary")
        _check_width(width)
        LOG.append(("download_button", (label, file_name, len(data))))
        return False

    def form_submit_button(self, label="Submit", help=None, on_click=None, args=None, kwargs=None, *,
                           key=None, type="secondary", icon=None, disabled=False, use_container_width=None,
                           width="content", shortcut=None):
        assert type in ("primary", "secondary", "tertiary")
        k = key or label
        clicked = k in CONFIG["click"]
        if clicked:
            CONFIG["click"].discard(k)
            _callback(on_click, args, kwargs)
        return clicked

    def page_link(self, page, *, label=None, icon=None, help=None, disabled=False, use_container_width=None,
                  width="content", query_params=None):
        if isinstance(page, str):
            assert page in _PAGE_PATHS, f"page_link to unregistered page {page}"
        LOG.append(("page_link", page))

    def toggle(self, label, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, *,
               disabled=False, label_visibility="visible", width="content"):
        _check_label_visibility(label_visibility)
        return bool(_widget(key, value))

    def checkbox(self, label, value=False, key=None, help=None, on_change=None, args=None, kwargs=None, *,
                 disabled=False, label_visibility="visible", width="content"):
        return bool(_widget(key, value))

    def slider(self, label, min_value=None, max_value=None, value=None, step=None, format=None, key=None,
               help=None, on_change=None, args=None, kwargs=None, *, disabled=False, label_visibility="visible",
               width="stretch"):
        _check_label_visibility(label_visibility)
        default = value if value is not None else min_value
        v = _widget(key, default)
        vals = v if isinstance(v, (tuple, list)) else (v,)
        for x in vals:
            if min_value is not None and max_value is not None:
                assert min_value <= x <= max_value, f"slider {label}: {x} outside [{min_value}, {max_value}]"
        if min_value is not None and step is not None and isinstance(step, int) and isinstance(min_value, int):
            assert all(isinstance(x, int) for x in vals), f"slider {label}: int slider with {vals}"
        return v

    def select_slider(self, label, options=(), value=None, format_func=str, key=None, help=None, on_change=None,
                      args=None, kwargs=None, *, disabled=False, label_visibility="visible", width="stretch"):
        options = list(options)
        default = value if value is not None else options[0]
        v = _widget(key, default)
        vals = v if isinstance(v, (tuple, list)) else (v,)
        for x in vals:
            assert x in options, f"select_slider {label}: {x!r} not in options"
        return tuple(v) if isinstance(v, list) else v

    def selectbox(self, label, options, index=0, format_func=str, key=None, help=None, on_change=None, args=None,
                  kwargs=None, *, placeholder=None, disabled=False, label_visibility="visible", accept_new_options=False,
                  width="stretch"):
        options = list(options)
        default = options[index] if (index is not None and options) else None
        v = _widget(key, default)
        assert v is None or v in options, f"selectbox {label}: {v!r} not in options"
        return v

    def radio(self, label, options, index=0, format_func=str, key=None, help=None, on_change=None, args=None,
              kwargs=None, *, disabled=False, horizontal=False, captions=None, label_visibility="visible",
              width="content"):
        options = list(options)
        v = _widget(key, options[index] if index is not None else None)
        assert v is None or v in options
        return v

    def multiselect(self, label, options, default=None, format_func=str, key=None, help=None, on_change=None,
                    args=None, kwargs=None, *, max_selections=None, placeholder=None, disabled=False,
                    label_visibility="visible", accept_new_options=False, width="stretch"):
        options = list(options)
        v = _widget(key, list(default or []))
        assert all(x in options for x in v), f"multiselect {label}: {v} not all in options"
        if max_selections:
            assert len(v) <= max_selections
        return list(v)

    def segmented_control(self, label, options, *, selection_mode="single", default=None, format_func=None,
                          key=None, help=None, on_change=None, args=None, kwargs=None, disabled=False,
                          label_visibility="visible", width="content"):
        assert selection_mode in ("single", "multi")
        options = list(options)
        v = _widget(key, default)
        if selection_mode == "single":
            assert v is None or v in options, f"segmented_control {label}: {v!r} not in {options}"
        return v

    def pills(self, label, options, *, selection_mode="single", default=None, format_func=None, key=None,
              help=None, on_change=None, args=None, kwargs=None, disabled=False, label_visibility="visible",
              width="content"):
        assert selection_mode in ("single", "multi")
        options = list(options)
        v = _widget(key, default)
        if selection_mode == "multi":
            v = list(v or [])
            assert all(x in options for x in v)
        return v

    def number_input(self, label, min_value=None, max_value=None, value="min", step=None, format=None, key=None,
                     help=None, on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False,
                     label_visibility="visible", icon=None, width="stretch"):
        default = (min_value if min_value is not None else 0) if value == "min" else value
        v = _widget(key, default)
        if min_value is not None:
            assert v >= min_value, f"number_input {label}: {v} < {min_value}"
        if max_value is not None:
            assert v <= max_value, f"number_input {label}: {v} > {max_value}"
        return v

    def text_input(self, label, value="", max_chars=None, key=None, type="default", help=None, autocomplete=None,
                   on_change=None, args=None, kwargs=None, *, placeholder=None, disabled=False,
                   label_visibility="visible", icon=None, width="stretch"):
        return _widget(key, value)

    def text_area(self, label, value="", height=None, max_chars=None, key=None, help=None, on_change=None,
                  args=None, kwargs=None, *, placeholder=None, disabled=False, label_visibility="visible",
                  width="stretch"):
        return _widget(key, value)

    def date_input(self, label, value="today", min_value=None, max_value=None, key=None, help=None,
                   on_change=None, args=None, kwargs=None, *, format="YYYY/MM/DD", disabled=False,
                   label_visibility="visible", width="stretch"):
        return _widget(key, value)

    def file_uploader(self, label, type=None, accept_multiple_files=False, key=None, help=None, on_change=None,
                      args=None, kwargs=None, *, disabled=False, label_visibility="visible", width="stretch"):
        _check_label_visibility(label_visibility)
        up = CONFIG["uploads"].get(key)
        if up is not None:
            session_state._d[key] = up
            _callback(on_change, args, kwargs)
            CONFIG["uploads"].pop(key)
        if key is not None:
            _INSTANTIATED.add(key)
        return up

    # ---- layout --------------------------------------------------------------------------------
    def columns(self, spec, *, gap="small", vertical_alignment="top", border=False, width="stretch"):
        assert gap in ("xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", None)
        assert vertical_alignment in ("top", "center", "bottom")
        n = spec if isinstance(spec, int) else len(spec)
        return [DeltaGenerator() for _ in range(n)]

    def container(self, *, border=None, key=None, height=None, width="stretch", horizontal=False,
                  horizontal_alignment="left", vertical_alignment="top", gap="small"):
        return DeltaGenerator()

    def expander(self, label, expanded=False, *, icon=None, width="stretch"):
        assert isinstance(label, str)
        return DeltaGenerator()

    def popover(self, label, *, type="secondary", help=None, icon=None, disabled=False, use_container_width=None,
                width="content"):
        _check_width(width)
        return DeltaGenerator()

    def tabs(self, tabs, *, width="stretch", default=None):
        return [DeltaGenerator() for _ in tabs]

    def form(self, key, clear_on_submit=False, *, enter_to_submit=True, border=True, width="stretch",
             height="content"):
        return DeltaGenerator()

    def empty(self):
        return DeltaGenerator()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _render_table(data):
    import pandas as pd
    if data is None:
        return
    from pandas.io.formats.style import Styler
    if isinstance(data, Styler):
        data.to_html()  # runs every styling function
        return
    assert isinstance(data, (pd.DataFrame, pd.Series)), type(data)


def _check_column_config(cfg, data):
    assert isinstance(cfg, dict)
    for k, v in cfg.items():
        assert isinstance(v, (str, dict, _Column)) or v is None, (k, v)


_main = DeltaGenerator()
sidebar = DeltaGenerator()
_CHART_KEYS: set = set()

for _name in [n for n in dir(DeltaGenerator) if not n.startswith("_")]:
    globals()[_name] = getattr(_main, _name)


# ---------------------------------------------------------------------------
# Column config
# ---------------------------------------------------------------------------

class _Column(dict):
    pass


def _col(allowed):
    def make(label=None, **kw):
        bad = set(kw) - set(allowed)
        assert not bad, f"unknown column_config parameters {bad}"
        if "width" in kw:
            assert kw["width"] in ("small", "medium", "large") or isinstance(kw["width"], int)
        return _Column(label=label, **kw)
    return make


_COMMON = ("width", "help", "disabled", "required", "default", "pinned")
column_config = types.SimpleNamespace(
    Column=_col(_COMMON),
    TextColumn=_col(_COMMON + ("max_chars", "validate")),
    NumberColumn=_col(_COMMON + ("format", "min_value", "max_value", "step")),
    ProgressColumn=_col(("width", "help", "format", "min_value", "max_value", "pinned", "color")),
    DatetimeColumn=_col(_COMMON + ("format", "min_value", "max_value", "step", "timezone")),
    DateColumn=_col(_COMMON + ("format", "min_value", "max_value", "step")),
    SelectboxColumn=_col(_COMMON + ("options", "format_func")),
    CheckboxColumn=_col(_COMMON),
    LineChartColumn=_col(("width", "help", "y_min", "y_max", "pinned", "color")),
    BarChartColumn=_col(("width", "help", "y_min", "y_max", "pinned", "color")),
    LinkColumn=_col(_COMMON + ("max_chars", "validate", "display_text")),
    ListColumn=_col(("width", "help", "pinned")),
)


# ---------------------------------------------------------------------------
# App-level functions
# ---------------------------------------------------------------------------

def set_page_config(page_title=None, page_icon=None, layout=None, initial_sidebar_state=None, menu_items=None):
    assert layout in (None, "centered", "wide")
    assert initial_sidebar_state in (None, "auto", "expanded", "collapsed")


def stop():
    raise StopException()


def rerun(*, scope="app"):
    raise RerunException()


def toast(body, *, icon=None, duration="short"):
    LOG.append(("toast", body))


@contextlib.contextmanager
def spinner(text="In progress...", *, show_time=False, width="content"):
    yield


def _cache(func=None, **kw):
    allowed = {"ttl", "max_entries", "show_spinner", "persist", "hash_funcs", "validate", "scope"}
    bad = set(kw) - allowed
    assert not bad, f"unknown cache parameters {bad}"

    def deco(f):
        store = {}

        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(f)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            key = tuple((k, repr(v)) for k, v in bound.arguments.items() if not k.startswith("_"))
            if key not in store:
                store[key] = f(*args, **kwargs)
            import copy
            return copy.deepcopy(store[key]) if CACHE_COPY[0] else store[key]

        wrapper.clear = store.clear
        return wrapper

    return deco(func) if callable(func) else deco


CACHE_COPY = [True]
cache_data = _cache
cache_resource = _cache


def fragment(func=None, *, run_every=None):
    return func if callable(func) else (lambda f: f)


def dialog(title, *, width="small", dismissible=True, on_dismiss="ignore"):
    return lambda f: f


class _Theme:
    type = "light"


context = types.SimpleNamespace(theme=_Theme())
query_params: dict = {}


# ---------------------------------------------------------------------------
# Multipage
# ---------------------------------------------------------------------------

_PAGE_PATHS: set = set()


class Page:
    def __init__(self, page, *, title=None, icon=None, url_path=None, default=False, visibility="visible"):
        assert isinstance(page, str)
        self.page = page
        self.title = title
        self.icon = icon
        self.url_path = url_path or page.rsplit("/", 1)[-1].removesuffix(".py")
        self.default = default
        _PAGE_PATHS.add(page)

    def run(self):
        runpy.run_path(self.page, run_name="__main__")


class _Nav:
    def __init__(self, pages):
        self.pages = pages
        want = CONFIG["page"]
        chosen = next((p for p in pages if p.url_path == want), None)
        if chosen is None:
            chosen = next((p for p in pages if p.default), pages[0])
        self.page = chosen
        self.url_path = chosen.url_path
        self.title = chosen.title

    def run(self):
        self.page.run()


def navigation(pages, *, position="sidebar", expanded=False):
    assert position in ("sidebar", "top", "hidden")
    flat = [p for group in pages.values() for p in group] if isinstance(pages, dict) else list(pages)
    assert flat and all(isinstance(p, Page) for p in flat)
    return _Nav(flat)


def switch_page(page):
    raise RerunException()


# ---------------------------------------------------------------------------
# Driver used by the tests
# ---------------------------------------------------------------------------

def run_app(main_script: str, page: str | None = None, max_reruns: int = 5):
    """Execute the app like one Streamlit script run (plus reruns)."""
    CONFIG["page"] = page
    for _ in range(max_reruns):
        _reset_run()
        _CHART_KEYS.clear()
        _PAGE_PATHS.clear()
        try:
            runpy.run_path(main_script, run_name="__main__")
            return "ok"
        except StopException:
            return "stopped"
        except RerunException:
            continue
    raise RuntimeError("too many reruns")
