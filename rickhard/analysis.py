"""One function that turns the run table + settings + scope into every table
the pages show, so all pages agree on the same numbers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import geometry, metrics, patterns, zheight
from .config import Scope, Settings
from .constants import DEV_COLS


@dataclass
class Analysis:
    settings: Settings
    scope: Scope
    runs: pd.DataFrame          # every run in scope, with status + vectors
    first: pd.DataFrame         # first run of every module in scope (+ vectors, mechanism, squareness, time)
    weekly: pd.DataFrame
    pareto: pd.DataFrame
    mechanisms: pd.DataFrame
    capability: pd.DataFrame
    sq_first: pd.DataFrame      # complete first runs with squareness metrics
    sq_runs: pd.DataFrame       # complete runs (all) with squareness metrics (original table)
    kpis: dict
    has_z: bool = False
    limits: dict = field(default_factory=dict)

    @property
    def types(self) -> list[str]:
        present = set(self.first["BatteryType"]) if len(self.first) else set()
        return [t for t in self.scope.types if t in present]

    @property
    def fpy(self) -> float:
        return metrics.fpy(self.first)


def module_key(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object, index=df.index)
    return (df["PartID"].astype(object).map(str) + " | Run " + df["RunNum"].astype(int).astype(str)
            + " | " + pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d"))


def in_scope(df: pd.DataFrame, scope: Scope) -> pd.DataFrame:
    if df.empty:
        return df
    m = df["BatteryType"].isin(scope.types)
    if scope.week_from is not None:
        m &= df["WeekKey"] >= scope.week_from
    if scope.week_to is not None:
        m &= df["WeekKey"] <= scope.week_to
    return df[m.fillna(False).astype(bool)]


def analyze(runs_all: pd.DataFrame, settings: Settings, scope: Scope, ma_window: int = 3,
            has_z: bool = False, limits: dict | None = None) -> Analysis:
    r = settings.rules
    nom = settings.nominals
    zlim = (r.z_lo, r.z_hi) if (r.z_in_status and has_z) else None
    runs_st = geometry.evaluate_status(runs_all, r.spec_limit, z_limits=zlim)
    runs_st["_mod_key"] = module_key(runs_st)
    # A module's Run 1 is its Run 1 whatever the filters; filter afterwards.
    first_all = metrics.first_runs(runs_st, r.incomplete_as_fail)
    runs = in_scope(runs_st, scope).reset_index(drop=True)
    first = in_scope(first_all, scope).reset_index(drop=True)

    first = geometry.add_vectors(first, nom)
    mech = geometry.failure_mechanism(first, r.spec_limit, nom)
    for c in mech.columns:
        first[c] = mech[c].to_numpy()
    sq = geometry.squareness(first, r.diag_tol, nom)
    for c in geometry.SQ_COLUMNS:
        first[c] = sq[c].reindex(first.index) if len(sq) else np.nan
    first = patterns.add_time_features(first, settings.shifts)
    if has_z:
        zm = zheight.z_metrics(first, nom)
        for c in zm.columns:
            first[c] = zm[c].to_numpy()
    first["NearMiss"] = (first["Status"] == "PASS") & (first["Margin"] < r.near_miss)

    runs = geometry.add_vectors(runs, nom)
    mech_r = geometry.failure_mechanism(runs, r.spec_limit, nom)
    for c in ("Mechanism", "NokCause", "PassIfSquare", "PassIfPlaced"):
        runs[c] = mech_r[c].to_numpy()
    sq_runs = geometry.squareness(runs, r.diag_tol, nom)
    sq_first = first[first["SquarenessStatus"].notna()].copy() if len(first) else first.copy()

    weekly = metrics.weekly_fpy(first, ma_window, near_miss=r.near_miss)
    pareto = metrics.corner_pareto(first, r.spec_limit)
    mechs = metrics.mechanism_summary(first)
    cap = metrics.capability(first, r.spec_limit, z_limits=(r.z_lo, r.z_hi) if has_z else None)
    k = metrics.kpis(first, runs, weekly, cap, sq_first, r.near_miss, r.fpy_target)
    return Analysis(settings=settings, scope=scope, runs=runs, first=first, weekly=weekly, pareto=pareto,
                    mechanisms=mechs, capability=cap, sq_first=sq_first, sq_runs=sq_runs, kpis=k,
                    has_z=has_z, limits=limits or {})


def complete_first(a: Analysis, bat_type: str | None = None) -> pd.DataFrame:
    f = a.first
    if bat_type:
        f = f[f["BatteryType"] == bat_type]
    return f[f[list(DEV_COLS)].notna().all(axis=1)]
