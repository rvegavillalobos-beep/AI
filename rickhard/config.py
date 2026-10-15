"""Settings objects.

Everything that changes a number lives in one of these frozen (hashable)
dataclasses, so the app can cache results on them and a project file can
save and restore them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from .constants import BATTERY_TYPES, CORNERS, DEFAULT_NOMINALS

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadOptions:
    """How raw export rows become module runs."""

    source_tz: str = "Europe/Berlin"
    display_tz: str = "America/Mexico_City"
    zero_as_missing: bool = True
    merge_lookalikes: bool = False
    # A new run also starts when the same part is measured again after this
    # many minutes (0 = only when a corner repeats, the original rule).
    run_gap_min: float = 30.0
    # Extra feature -> corner rules checked before the built-in ones:
    # ((substring, "FL"|"FR"|"RL"|"RR"), ...), case-insensitive.
    feature_map: tuple[tuple[str, str], ...] = ()
    # Manual column mapping ((canonical, column in file), ...) used when the
    # automatic detection picks the wrong column.
    column_overrides: tuple[tuple[str, str], ...] = ()


# ---------------------------------------------------------------------------
# Quality rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rules:
    spec_limit: float = 3.0       # ± mm, X and Y, every corner
    fpy_target: float = 90.0      # %
    diag_tol: float = 1.5         # mm, squareness (diagonal delta)
    incomplete_as_fail: bool = True
    near_miss: float = 0.5        # mm: a PASS with less margin than this is a near miss
    z_in_status: bool = False     # also judge Z against its limits
    z_lo: float = -1.25
    z_hi: float = 1.21


# ---------------------------------------------------------------------------
# Shifts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShiftPlan:
    starts: tuple[int, ...] = (6, 14, 22)
    names: tuple[str, ...] = ("A", "B", "C")

    def assign(self, hours) -> np.ndarray:
        """Shift name for each hour of day (0-23)."""
        h = np.asarray(hours, dtype=float)
        starts = np.asarray(self.starts, dtype=float)
        order = np.argsort(starts)
        starts = starts[order]
        names = np.asarray(self.names, dtype=object)[order]
        idx = np.searchsorted(starts, np.nan_to_num(h, nan=0.0), side="right") - 1
        idx = np.where(idx < 0, len(starts) - 1, idx)  # before the first start: last shift
        out = names[idx]
        return np.where(np.isnan(h), None, out)

    def describe(self) -> list[str]:
        s = list(self.starts)
        out = []
        for i, (st, nm) in enumerate(zip(s, self.names)):
            en = s[(i + 1) % len(s)]
            out.append(f"{nm}: {st:02d}:00–{en:02d}:00")
        return out


# ---------------------------------------------------------------------------
# Nominal geometry
# ---------------------------------------------------------------------------

def _freeze_nominals(d: dict) -> tuple:
    return tuple(
        (t, tuple((c, float(d[t][c][0]), float(d[t][c][1])) for c in CORNERS))
        for t in BATTERY_TYPES
        if t in d
    )


def normalize_type(bat_type) -> str:
    """Same fallback as the original app: anything that is not Type S is Type M."""
    return "Type S" if str(bat_type).strip().upper() == "TYPE S" else "Type M"


@dataclass(frozen=True)
class Nominals:
    table: tuple = field(default_factory=lambda: _freeze_nominals(DEFAULT_NOMINALS))

    @classmethod
    def from_dict(cls, d: dict) -> Nominals:
        merged = {t: dict(DEFAULT_NOMINALS[t]) for t in BATTERY_TYPES}
        for t, corners in (d or {}).items():
            if t in merged:
                for c, xy in corners.items():
                    if c in merged[t]:
                        merged[t][c] = (float(xy[0]), float(xy[1]))
        return cls(_freeze_nominals(merged))

    def as_dict(self) -> dict[str, dict[str, tuple[float, float]]]:
        return {t: {c: (x, y) for c, x, y in rows} for t, rows in self.table}

    def get(self, bat_type: str) -> dict[str, tuple[float, float]]:
        d = self.as_dict()
        return d.get(normalize_type(bat_type), d["Type M"])

    def center(self, bat_type: str) -> tuple[float, float]:
        n = self.get(bat_type)
        return (sum(n[c][0] for c in CORNERS) / 4.0, sum(n[c][1] for c in CORNERS) / 4.0)

    def is_default(self) -> bool:
        return self == Nominals()


# ---------------------------------------------------------------------------
# Everything that defines an analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    rules: Rules = field(default_factory=Rules)
    nominals: Nominals = field(default_factory=Nominals)
    shifts: ShiftPlan = field(default_factory=ShiftPlan)

    def with_rules(self, **kw) -> Settings:
        return replace(self, rules=replace(self.rules, **kw))


@dataclass(frozen=True)
class Scope:
    types: tuple[str, ...] = BATTERY_TYPES
    week_from: int | None = None   # WeekKey (YYYYWW), inclusive
    week_to: int | None = None

    def label_types(self) -> str:
        return "All types" if len(self.types) > 1 else self.types[0]


# ---------------------------------------------------------------------------
# Project file (settings + event log) as JSON
# ---------------------------------------------------------------------------

PROJECT_VERSION = 1


def project_to_json(load: LoadOptions, settings: Settings, events: list[dict] | None = None) -> str:
    payload = {
        "app": "RICKHARD X",
        "project_version": PROJECT_VERSION,
        "load": asdict(load),
        "rules": asdict(settings.rules),
        "shifts": asdict(settings.shifts),
        "nominals": {t: {c: list(xy) for c, xy in corners.items()}
                     for t, corners in settings.nominals.as_dict().items()},
        "events": events or [],
    }
    return json.dumps(payload, indent=2, default=str)


def project_from_json(text: str) -> tuple[LoadOptions, Settings, list[dict]]:
    p = json.loads(text)
    if not isinstance(p, dict) or "rules" not in p:
        raise ValueError("This is not a RICKHARD X project file.")
    ld = p.get("load", {})
    load = LoadOptions(
        source_tz=str(ld.get("source_tz", "Europe/Berlin")),
        display_tz=str(ld.get("display_tz", "America/Mexico_City")),
        zero_as_missing=bool(ld.get("zero_as_missing", True)),
        merge_lookalikes=bool(ld.get("merge_lookalikes", False)),
        run_gap_min=float(ld.get("run_gap_min", 30.0)),
        feature_map=tuple(tuple(x) for x in ld.get("feature_map", ())),
        column_overrides=tuple(tuple(x) for x in ld.get("column_overrides", ())),
    )
    r = p.get("rules", {})
    defaults = Rules()
    rules = Rules(**{k: type(getattr(defaults, k))(r[k]) for k in asdict(defaults) if k in r})
    sh = p.get("shifts", {})
    shifts = ShiftPlan(
        starts=tuple(int(x) for x in sh.get("starts", (6, 14, 22))),
        names=tuple(str(x) for x in sh.get("names", ("A", "B", "C"))),
    )
    if len(shifts.starts) != len(shifts.names) or not shifts.starts:
        shifts = ShiftPlan()
    nominals = Nominals.from_dict(p.get("nominals", {}))
    events = list(p.get("events", []))
    return load, Settings(rules=rules, nominals=nominals, shifts=shifts), events
