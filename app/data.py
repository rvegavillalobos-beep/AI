"""Cached computations and the context every page works from.

Heavy objects are passed to cached functions with a leading underscore (not
hashed); a string ``key`` that fingerprints the data, the settings and the
code version decides what is reused.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from rickhard import analysis, compensation, demo, events as ev_mod, improvement, insights, loading
from rickhard import metrics, patterns, remeasure, spc
from rickhard.analysis import Analysis
from rickhard.config import LoadOptions, Scope, Settings
from rickhard.loading import Dataset

from . import state


def _code_version() -> str:
    h = hashlib.sha1()
    root = Path(__file__).resolve().parent.parent / "rickhard"
    for f in sorted(root.rglob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


CODE_VERSION = _code_version()
PLACEMENT_VARS = {"Fit_Tx": "Shift X", "Fit_Ty": "Shift Y", "Fit_Rot": "Yaw"}


# ---------------------------------------------------------------------------
# Loading and analysis
# ---------------------------------------------------------------------------

@st.cache_resource(max_entries=4, show_spinner="Reading measurements…")
def _dataset(key: str, _files: tuple, _opts: LoadOptions) -> Dataset:
    return loading.build_dataset(list(_files), _opts)


def dataset_key(opts: LoadOptions) -> str:
    return f"{state.files_hash()}|{opts!r}|{CODE_VERSION}"


def get_dataset() -> tuple[Dataset, str]:
    opts = state.load_options()
    key = dataset_key(opts)
    files = tuple((f["name"], f["data"]) for f in state.files())
    return _dataset(key, files, opts), key


@st.cache_data(max_entries=24, show_spinner="Analysing…")
def _analysis(key: str, _ds: Dataset, _settings: Settings, _scope: Scope, ma: int) -> Analysis:
    return analysis.analyze(_ds.runs, _settings, _scope, ma, _ds.has_z, _ds.limits)


@st.cache_data(max_entries=4, show_spinner=False)
def demo_bytes(code_version: str) -> bytes:
    return demo.demo_file("xlsx")


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class Context:
    ds: Dataset
    a: Analysis
    key: str                 # data + settings + scope fingerprint
    settings: Settings
    scope: Scope
    week_label: str
    ma: int
    file_label: str
    is_demo: bool

    @property
    def rules(self):
        return self.settings.rules

    @property
    def nominals(self):
        return self.settings.nominals

    @property
    def types(self) -> list[str]:
        return self.a.types

    @property
    def events(self) -> pd.DataFrame:
        return events_frame()


def make_context(ds: Dataset, ds_key: str, settings: Settings, scope: Scope, ma: int, week_label: str) -> Context:
    key = f"{ds_key}|{settings!r}|{scope!r}|{ma}"
    a = _analysis(key, ds, settings, scope, ma)
    names = [f["name"] for f in state.files()]
    label = names[0] if len(names) == 1 else f"{len(names)} files"
    return Context(ds=ds, a=a, key=key, settings=settings, scope=scope, week_label=week_label, ma=ma,
                   file_label=label, is_demo=state.is_demo())


def ctx() -> Context:
    c = st.session_state.get("_ctx")
    if c is None:
        st.info("Load a measurement export first.", icon=":material/upload_file:")
        st.stop()
    return c


def events_frame() -> pd.DataFrame:
    df = ev_mod.normalize_events(state.events())
    if len(df):
        loading.add_week_columns(df, "Date")
    else:
        df["WeekKey"] = pd.Series(dtype="Int64")
        df["CalendarWeek"] = pd.Series(dtype=object)
    return df


# ---------------------------------------------------------------------------
# Derived, cached on the context key
# ---------------------------------------------------------------------------

@st.cache_data(max_entries=24, show_spinner="Detecting process shifts…")
def change_points(key: str, _a: Analysis) -> dict:
    """Change points of the placement per type: {(type, col): [idx]}, the
    dates and a table."""
    out: dict = {"idx": {}, "table": []}
    for t, sub in _a.first.sort_values("Date").groupby("BatteryType"):
        for col, label in PLACEMENT_VARS.items():
            s = sub[["Date", col]].dropna()
            if len(s) < 30:
                continue
            y = s[col].to_numpy(float)
            cps = spc.change_points(y, min_size=max(12, len(y) // 40))
            out["idx"][(t, col)] = cps
            if cps:
                out["table"].append(spc.describe_change_points(s["Date"], y, cps, f"{t} {label}"))
    tab = pd.concat(out["table"], ignore_index=True) if out["table"] else pd.DataFrame(columns=spc.CP_COLUMNS)
    out["table"] = tab
    return out


@st.cache_data(max_entries=24, show_spinner=False)
def alarms(key: str, recent_days: int, _a: Analysis) -> pd.DataFrame:
    vars_ = dict(PLACEMENT_VARS)
    vars_["WorstAbs"] = "Worst corner |dev|"
    return spc.alarm_board(_a.first, vars_, recent_days=recent_days)


@st.cache_data(max_entries=24, show_spinner=False)
def remeasure_bundle(key: str, max_gap: float, _a: Analysis) -> dict:
    pairs = remeasure.run_pairs(_a.runs)
    rep = remeasure.repeatability(pairs, _a.settings.rules.spec_limit, max_gap)
    return {"pairs": pairs, "rep": rep, "transitions": remeasure.transitions(pairs),
            "summary": remeasure.summary(pairs, rep)}


@st.cache_data(max_entries=24, show_spinner="Reading the data for findings…")
def insight_bundle(key: str, _a: Analysis, _ds: Dataset) -> dict:
    f = _a.first
    rules = _a.settings.rules
    extras: dict = {"health": _ds.health}
    extras["placement"] = insights.placement_suggestions(_a)
    extras["budget"] = improvement.budget(f)
    extras["shape_link"] = metrics.shape_link(_a.sq_first, rules.diag_tol)
    extras["alarms"] = alarms(key, 14, _a)
    cp = change_points(key, _a)
    extras["change_points"] = cp["table"]
    extras["shift_effect"] = insights.shift_effect(f)
    gap = patterns.suggest_stop_gap(f)
    extras["startup"] = patterns.startup_effect(f, gap)
    rb = remeasure_bundle(key, 120.0, _a)
    extras["remeasure"] = rb["summary"]
    comps = {t: s["setting"] for t, s in extras["placement"].items()}
    extras["levers"] = insights.lever_gains(_a, comps)
    ins, acts = insights.generate(_a, extras)
    return {"insights": ins, "actions": acts, "extras": extras, "comps": comps}


@st.cache_data(max_entries=48, show_spinner="Searching the best compensation…")
def suggestions(key: str, bat_type: str, weeks: int, sigma_xy: float, sigma_yaw: float, method: str,
                _a: Analysis) -> dict:
    v = window(_a, bat_type, weeks)
    if v.empty:
        return {"settings": {"No compensation": (0.0, 0.0, 0.0)}, "optimum": None, "n": 0, "label": ""}
    out = compensation.suggestions(v, _a.settings.rules.spec_limit, _a.settings.nominals, sigma_xy, sigma_yaw, method)
    out["n"] = len(v)
    out["label"] = window_label(v)
    return out


@st.cache_data(max_entries=48, show_spinner="Replaying past weeks…")
def backtest(key: str, bat_type: str, weeks: int, sigma_xy: float, sigma_yaw: float, method: str,
             _a: Analysis) -> pd.DataFrame:
    v = _a.first[_a.first["BatteryType"] == bat_type]
    return compensation.backtest(v, _a.settings.rules.spec_limit, _a.settings.nominals, window_weeks=weeks,
                                 sigma_xy=sigma_xy, sigma_yaw=sigma_yaw, median_method=method)


@st.cache_data(max_entries=24, show_spinner="Simulating levers…")
def sensitivity(key: str, comps_sig: str, compensate: bool, _a: Analysis, _comps: dict) -> dict:
    return improvement.sensitivity_curves(_a.first, _comps, _a.settings.rules.spec_limit, _a.settings.nominals,
                                          compensate=compensate)


def window(a: Analysis, bat_type: str, weeks: int) -> pd.DataFrame:
    v = a.first[a.first["BatteryType"] == bat_type]
    wk = sorted(v["WeekKey"].dropna().unique())[-int(weeks):]
    return v[v["WeekKey"].isin(wk)]


def window_label(v: pd.DataFrame) -> str:
    if v.empty:
        return ""
    w = v.sort_values("WeekKey")["CalendarWeek"]
    return w.iloc[0] if w.iloc[0] == w.iloc[-1] else f"{w.iloc[0]} – {w.iloc[-1]}"


def chosen_compensation() -> dict:
    """Compensation chosen in the Compensation Studio, per type (if any)."""
    return dict(st.session_state.get("_chosen_comp") or {})


def set_chosen_compensation(bat_type: str, setting: tuple, method: str) -> None:
    d = chosen_compensation()
    d[bat_type] = {"setting": tuple(float(x) for x in setting), "method": method}
    st.session_state["_chosen_comp"] = d


def nice_float(v, nd=2) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:+.{nd}f}"
