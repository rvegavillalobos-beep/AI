"""Reading corner-deviation exports into one row per module run.

Pipeline::

    files ─► header & column detection ─► timestamps (source tz → display tz)
          ─► clean-up (duplicates, failed reads, look-alike IDs)
          ─► battery type & corner mapping ─► run inference ─► run summary

Robust to what real exports do: title rows above the header, several files
covering overlapping periods, newest-first ordering, daylight-saving hours,
mixed time formats, all-zero failed reads and IDs mistyped with O instead of 0.
"""

from __future__ import annotations

import csv
import datetime
import io
import re
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import LoadOptions
from .constants import CORNER_INDEX, CORNERS, DEV_COLS, Z_COLS


class DataFormatError(ValueError):
    """The file does not look like a corner-deviation export."""


# ---------------------------------------------------------------------------
# Domain rules (identical to the original RICKHARDV app)
# ---------------------------------------------------------------------------

def determine_battery_type(part_id, feature_names) -> str:
    """Type M or Type S from the PartID and the feature name(s)."""
    p_id = str(part_id).upper().strip()
    if isinstance(feature_names, (list, set, tuple, pd.Series, np.ndarray)):
        f_combined = " ".join(str(f).upper().strip() for f in feature_names)
    else:
        f_combined = str(feature_names).upper().strip()

    if "_DJ" in p_id or "_DJ" in f_combined or "_DI" in p_id or "_DI" in f_combined:
        return "Type M"
    if (
        "_M" in p_id or " M" in p_id or "TYPE M" in p_id
        or "TYPEM" in p_id or p_id.endswith("M")
    ):
        return "Type M"
    if "_DA" in p_id or "_DA" in f_combined:
        return "Type S"
    if "_S" in p_id or " S" in p_id or "TYPE S" in p_id or "TYPES" in p_id:
        return "Type S"
    return "Type S"


def extract_corner_index(feature_name, part_id, custom: Sequence[tuple[str, str]] = ()) -> int:
    """1=FL, 2=FR, 3=RL, 4=RR, 0=not a corner feature.

    ``custom`` rules ((substring, corner), ...) are checked first, then the
    original app's rules.
    """
    f = str(feature_name).lower().strip()
    if not f:
        return 0
    for sub, corner in custom:
        if sub and str(sub).lower() in f and corner in CORNERS:
            return CORNERS.index(corner) + 1

    is_type_m = determine_battery_type(part_id, feature_name) == "Type M"

    if "l0324_aa" in f or "l324_aa" in f:
        return 1
    if "r0301_aa" in f or "r301_aa" in f:
        return 2

    if is_type_m:
        if any(k in f for k in ["l0324_dj", "l324_dj", "l0324_di", "l324_di"]):
            return 3
        if any(k in f for k in ["r0301_dj", "r301_dj", "r0301_di", "r301_di"]):
            return 4
    else:
        if "l0324_da" in f or "l324_da" in f:
            return 3
        if "r0302_da" in f or "r302_da" in f:
            return 4

    if "fl" in f or "c1" in f:
        return 1
    elif "fr" in f or "c2" in f:
        return 2
    elif "rl" in f or "c3" in f:
        return 3
    elif "rr" in f or "c4" in f or "r302" in f or "r301" in f or "r00301" in f:
        return 4
    return 0


# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

CANONICAL = ("time", "part", "feature", "x", "y", "z", "x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi")
REQUIRED = ("time", "part", "feature", "x", "y")
CANONICAL_LABELS = {
    "time": "Timestamp", "part": "Part ID", "feature": "Feature name",
    "x": "X deviation", "y": "Y deviation", "z": "Z deviation (optional)",
    "x_lo": "X lower limit", "x_hi": "X upper limit", "y_lo": "Y lower limit",
    "y_hi": "Y upper limit", "z_lo": "Z lower limit", "z_hi": "Z upper limit",
}


def _tokens(name: str) -> list[str]:
    return re.findall(r"[a-z]+", str(name).lower())


def _is_axis_col(name: str, axis: str, kind: str) -> bool:
    toks = _tokens(name)
    low = str(name).lower().strip()
    has_axis = (axis in toks or low.startswith(axis + " ") or low.startswith(axis + "_")
                or ("d" + axis) in toks or low.startswith(axis + "-"))
    if not has_axis:
        return False
    if kind == "dev":
        return any(t.startswith("dev") for t in toks) or ("d" + axis) in toks or "delta" in toks
    if kind == "lo":
        return any(t in ("lower", "low", "min", "lsl", "lo") for t in toks)
    if kind == "hi":
        return any(t in ("upper", "up", "max", "usl", "hi") for t in toks)
    return False


def detect_columns(columns: Iterable) -> dict[str, str | None]:
    """Map canonical names to the file's columns (None when absent)."""
    cols = [str(c).strip() for c in columns if c is not None and str(c).strip() and str(c) != "nan"]
    low = {c: c.lower() for c in cols}
    found: dict[str, str | None] = {k: None for k in CANONICAL}

    def first(pred):
        for c in cols:
            if c in found.values():
                continue
            if pred(low[c]):
                return c
        return None

    found["time"] = (first(lambda c: c in ("time", "timestamp", "date", "datetime", "date time",
                                          "measurement time", "measuring time"))
                     or first(lambda c: "time" in c and "limit" not in c)
                     or first(lambda c: "date" in c))
    found["part"] = (first(lambda c: "part" in c)
                     or first(lambda c: any(k in c for k in ("serial", "dmc", "module id", "barcode"))))
    found["feature"] = (first(lambda c: "feature" in c)
                        or first(lambda c: any(k in c for k in ("characteristic", "merkmal", "point name"))))
    for ax in ("x", "y", "z"):
        found[ax] = first(lambda c, ax=ax: _is_axis_col(c, ax, "dev"))
        found[f"{ax}_lo"] = first(lambda c, ax=ax: _is_axis_col(c, ax, "lo"))
        found[f"{ax}_hi"] = first(lambda c, ax=ax: _is_axis_col(c, ax, "hi"))
    return found


def _missing_required(found: dict) -> list[str]:
    return [k for k in REQUIRED if not found.get(k)]


# ---------------------------------------------------------------------------
# Reading files
# ---------------------------------------------------------------------------

@dataclass
class FileRead:
    name: str
    frame: pd.DataFrame            # raw table under the detected header
    columns: dict[str, str | None]
    header_row: int
    sheet: str | None = None
    delimiter: str | None = None
    notes: list[str] = field(default_factory=list)


def _decode(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace")


def _sniff_delimiter(lines: list[str]) -> str:
    best, best_score = ",", -1
    for d in (",", ";", "\t", "|"):
        counts = [ln.count(d) for ln in lines if ln.strip()]
        if not counts:
            continue
        mx = max(counts)  # the header and the data rows share the largest count
        score = sum(1 for c in counts if c == mx and c > 0)
        if mx > 0 and score > best_score:
            best, best_score = d, score
    return best


def _apply_overrides(found: dict, header: list[str], overrides: Sequence[tuple[str, str]]) -> dict:
    out = dict(found)
    hdr = {str(h).strip() for h in header}
    for canon, col in overrides:
        if canon in out and str(col).strip() in hdr:
            out[canon] = str(col).strip()
    return out


def _read_csv(data: bytes, name: str, overrides) -> FileRead:
    text = _decode(data)
    lines = text.splitlines()
    head = lines[:60]
    delim = _sniff_delimiter(head[2:] if len(head) > 4 else head)
    rows = list(csv.reader(head, delimiter=delim))
    hrow = None
    found: dict = {}
    for i, r in enumerate(rows):
        f = _apply_overrides(detect_columns(r), r, overrides)
        if not _missing_required(f):
            hrow, found = i, f
            break
    if hrow is None:
        best = max(rows, key=lambda r: len([c for c in r if c.strip()]), default=[])
        miss = _missing_required(_apply_overrides(detect_columns(best), best, overrides))
        raise DataFormatError(
            f"{name}: no header row with " + ", ".join(CANONICAL_LABELS[m] for m in miss)
            + f" in the first {len(rows)} lines. Columns seen: " + ", ".join(best[:14])
        )
    sample = "\n".join(lines[hrow + 1: hrow + 30])
    decimal = "," if delim == ";" and re.search(r"\d,\d", sample) else "."
    df = pd.read_csv(io.StringIO(text), sep=delim, skiprows=hrow, header=0, decimal=decimal,
                     dtype=str, keep_default_na=True, skip_blank_lines=True, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return FileRead(name=name, frame=df, columns=found, header_row=hrow, delimiter=delim,
                    notes=[f"decimal separator '{decimal}'"] if decimal == "," else [])


def _read_excel(data: bytes, name: str, overrides) -> FileRead:
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, nrows=60, dtype=object)
    except ImportError as e:  # .xls without xlrd
        raise DataFormatError(f"{name}: this Excel format needs an extra package ({e}). "
                              "Save the file as .xlsx or .csv and try again.") from e
    except Exception as e:  # noqa: BLE001
        raise DataFormatError(f"{name}: the file could not be opened as Excel ({type(e).__name__}: {e}).") from e
    last_cols: list[str] = []
    for sheet, top in sheets.items():
        for i in range(len(top)):
            row = ["" if pd.isna(v) else str(v) for v in top.iloc[i].tolist()]
            f = _apply_overrides(detect_columns(row), row, overrides)
            if not _missing_required(f):
                df = pd.read_excel(io.BytesIO(data), sheet_name=sheet, header=i)
                df.columns = [str(c).strip() for c in df.columns]
                return FileRead(name=name, frame=df, columns=f, header_row=i, sheet=str(sheet))
            if len([c for c in row if c.strip()]) > len(last_cols):
                last_cols = [c for c in row if c.strip()]
    miss = _missing_required(_apply_overrides(detect_columns(last_cols), last_cols, overrides))
    raise DataFormatError(
        f"{name}: no header row with " + ", ".join(CANONICAL_LABELS[m] for m in miss)
        + ". Columns seen: " + ", ".join(last_cols[:14])
    )


def read_file(data: bytes, name: str, overrides: Sequence[tuple[str, str]] = ()) -> FileRead:
    if not data:
        raise DataFormatError(f"{name}: the file is empty.")
    is_excel = (name.lower().endswith((".xlsx", ".xlsm", ".xls"))
                or data[:4] in (b"PK\x03\x04", b"\xd0\xcf\x11\xe0"))
    return _read_excel(data, name, overrides) if is_excel else _read_csv(data, name, overrides)


def inspect_file(data: bytes, name: str) -> dict:
    """Header row, detected mapping and the column list (for the mapping UI)."""
    try:
        fr = read_file(data, name)
        return {"ok": True, "columns": list(fr.frame.columns), "mapping": fr.columns,
                "header_row": fr.header_row, "sheet": fr.sheet, "error": None}
    except DataFormatError as e:
        return {"ok": False, "columns": [], "mapping": {}, "header_row": None, "sheet": None, "error": str(e)}


# ---------------------------------------------------------------------------
# Time handling
# ---------------------------------------------------------------------------

KNOWN_TIME_FORMATS = (
    "%b %d, %Y %I:%M%p",       # Aug 19, 2026 7:16pm  (measurement export)
    "%b %d, %Y %I:%M:%S%p",
    "%b %d, %Y %I:%M %p",
    "%b %d, %Y %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%d.%m.%Y %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %I:%M %p",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
)


def parse_timestamps(raw: pd.Series) -> tuple[pd.Series, dict]:
    """Parse timestamps robustly. Returns (naive datetimes, info).

    Known export formats are tried first on a sample (fast and unambiguous);
    otherwise every value is parsed on its own — letting pandas guess one
    format from the first row is unsafe (an "am" first row makes it reject
    every "pm" time).
    """
    info = {"format": None, "recovered": 0, "excel_serial": False}
    if pd.api.types.is_datetime64_any_dtype(raw):
        out = pd.to_datetime(raw, errors="coerce")
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_localize(None)
        info["format"] = "datetime cells"
        return out, info

    obj = raw.astype(object)
    num = pd.to_numeric(obj, errors="coerce")
    if num.notna().mean() > 0.9 and num.dropna().between(20000, 80000).all():
        info["excel_serial"] = True
        info["format"] = "Excel serial"
        return pd.to_datetime(num, unit="D", origin="1899-12-30", errors="coerce"), info

    is_dt = obj.map(lambda v: hasattr(v, "year") and hasattr(v, "hour"))
    s = obj.map(lambda v: None if (v is None or (isinstance(v, float) and np.isnan(v)))
                else re.sub(r"\s+", " ", str(v).strip()))
    parsed = None
    sample = s.dropna()
    sample = sample[sample != ""].head(300)
    if len(sample):
        for fmt in KNOWN_TIME_FORMATS:
            ok = pd.to_datetime(sample, format=fmt, errors="coerce")
            if ok.notna().all():
                parsed = pd.to_datetime(s, format=fmt, errors="coerce")
                info["format"] = fmt
                break
    if parsed is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", format="mixed")
        info["format"] = "mixed (parsed value by value)"
    if is_dt.any():
        parsed = parsed.copy()
        parsed[is_dt] = pd.to_datetime(obj[is_dt].tolist(), errors="coerce")
    failed = parsed.isna() & s.notna() & (s != "")
    if failed.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retry = pd.to_datetime(s[failed], errors="coerce", format="mixed")
        info["recovered"] = int(retry.notna().sum())
        parsed = parsed.copy()
        parsed[failed] = retry
    parsed = pd.to_datetime(parsed, errors="coerce")
    if getattr(parsed.dt, "tz", None) is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed, info


def convert_timezone(naive: pd.Series, source_tz: str | None, display_tz: str | None):
    """Naive source-local → naive display-local, never dropping a reading.

    Ambiguous times (repeated autumn hour) are read as summer time;
    non-existent times (skipped spring hour) are shifted forward.
    Returns (converted, n_ambiguous, n_nonexistent).
    """
    if not source_tz or not display_tz or source_tz == display_tz or naive.empty:
        return naive, 0, 0
    valid = naive.notna()
    strict = naive.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
    problems = strict.isna() & valid
    n_nonexistent = 0
    if problems.any():
        shifted = naive[problems].dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="shift_forward")
        n_nonexistent = int(shifted.notna().sum())
    n_ambiguous = int(problems.sum()) - n_nonexistent
    localized = naive.dt.tz_localize(
        source_tz, ambiguous=np.ones(len(naive), dtype=bool), nonexistent="shift_forward"
    )
    converted = localized.dt.tz_convert(display_tz).dt.tz_localize(None)
    return converted, n_ambiguous, n_nonexistent


def add_week_columns(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """ISO weeks that survive the year boundary.

    ``WeekKey`` (202643) sorts chronologically. ``CalendarWeek`` is "CW43"
    when all data falls in one ISO year, "2026-CW43" otherwise.
    """
    d = pd.to_datetime(df[date_col])
    iso = d.dt.isocalendar()
    year = iso["year"].astype("Int64")
    week = iso["week"].astype("Int64")
    df["WeekKey"] = (year * 100 + week).astype("Int64")
    multi_year = year.dropna().nunique() > 1
    labels = []
    for y, w in zip(year, week):
        if pd.isna(y) or pd.isna(w):
            labels.append(None)
        else:
            labels.append(f"{int(y)}-CW{int(w):02d}" if multi_year else f"CW{int(w):02d}")
    df["CalendarWeek"] = labels
    return df


def week_start(week_key: int) -> pd.Timestamp:
    """Monday of an ISO WeekKey (YYYYWW)."""
    y, w = divmod(int(week_key), 100)
    return pd.Timestamp(datetime.date.fromisocalendar(y, w, 1))


# ---------------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------------

def infer_runs(part_keys, features, times_ns=None, gap_minutes: float = 0.0) -> np.ndarray:
    """Sequential run counter per part.

    Walking measurements in time order, a part's run increases when a feature
    already measured in the current run appears again (original rule) or —
    when ``gap_minutes`` > 0 — when the part comes back after that long.
    One continuous run history per part across all days.
    """
    gap_ns = float(gap_minutes) * 60e9 if gap_minutes and gap_minutes > 0 else 0.0
    run: dict = {}
    seen: dict = {}
    last: dict = {}
    keys = list(part_keys)
    feats = list(features)
    tms = list(times_ns) if times_ns is not None else [0] * len(keys)
    out = np.empty(len(keys), dtype=np.int64)
    for i, (k, f, t) in enumerate(zip(keys, feats, tms)):
        if k not in run:
            run[k] = 1
            seen[k] = {f}
        else:
            new = f in seen[k] or (gap_ns and (t - last[k]) > gap_ns)
            if new:
                run[k] += 1
                seen[k] = {f}
            else:
                seen[k].add(f)
        last[k] = t
        out[i] = run[k]
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    rows: pd.DataFrame            # one row per corner reading (standardized)
    runs: pd.DataFrame            # one row per module run
    health: dict
    has_z: bool
    limits: dict                  # limits exported in the file: {"X": (lo, hi), ...}
    files: list[dict]


def _similar_part_ids(part_ids) -> list[tuple[str, ...]]:
    groups: dict[str, list[str]] = {}
    for pid in part_ids:
        groups.setdefault(re.sub("[Oo]", "0", str(pid)), []).append(str(pid))
    return [tuple(sorted(v)) for v in groups.values() if len(v) > 1]


def _standardize(fr: FileRead) -> pd.DataFrame:
    m = fr.columns
    df = fr.frame
    out = pd.DataFrame(index=df.index)
    out["TimeRaw"] = df[m["time"]]
    out["PartID"] = df[m["part"]]
    out["Feature"] = df[m["feature"]]
    for k, col in (("X", "x"), ("Y", "y"), ("Z", "z"), ("X_lo", "x_lo"), ("X_hi", "x_hi"),
                   ("Y_lo", "y_lo"), ("Y_hi", "y_hi"), ("Z_lo", "z_lo"), ("Z_hi", "z_hi")):
        out[k + "_raw"] = df[m[col]] if m.get(col) else np.nan
    out["SourceFile"] = fr.name
    out["SourceRow"] = np.arange(len(df)) + fr.header_row + 2
    return out


def _to_number(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)

    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        if isinstance(v, (int, float, np.number)):
            return float(v)
        t = str(v).strip().replace("−", "-")
        if re.fullmatch(r"-?\d+,\d+", t):
            t = t.replace(",", ".")
        try:
            return float(t)
        except ValueError:
            return np.nan

    return s.astype(object).map(conv).astype(float)


def _mode_value(s: pd.Series):
    s = s.dropna()
    return float(s.mode().iloc[0]) if len(s) else None


def build_dataset(files: Sequence[tuple[str, bytes]], opts: LoadOptions = LoadOptions()) -> Dataset:
    """Read one or more exports and build the per-run module table."""
    if not files:
        raise DataFormatError("No file given.")
    frames, file_info = [], []
    health: dict = {"files": len(files)}
    time_formats = set()
    tz_amb = tz_nonexist = recovered = 0
    for name, data in files:
        fr = read_file(data, name, opts.column_overrides)
        std = _standardize(fr)
        std = std[~(std["PartID"].isna() & std["Feature"].isna())]
        parsed, tinfo = parse_timestamps(std["TimeRaw"])
        time_formats.add(tinfo["format"])
        recovered += tinfo["recovered"]
        conv, a, n = convert_timezone(parsed, opts.source_tz, opts.display_tz)
        tz_amb += a
        tz_nonexist += n
        std["Date"] = conv
        # Exports are often newest first: make ties (same minute) chronological.
        valid = std["Date"].dropna()
        newest_first = len(valid) > 2 and valid.iloc[0] > valid.iloc[-1]
        if newest_first:
            std = std.iloc[::-1]
        frames.append(std)
        file_info.append({
            "name": name, "rows": int(len(fr.frame)), "header_row": fr.header_row + 1,
            "sheet": fr.sheet, "delimiter": fr.delimiter, "time_format": tinfo["format"],
            "newest_first": bool(newest_first),
            "columns": {k: v for k, v in fr.columns.items() if v},
            "notes": fr.notes,
        })
    df = pd.concat(frames, ignore_index=True)
    health["rows_read"] = int(len(df))
    health["time_formats"] = sorted(x for x in time_formats if x)
    health["timestamps_recovered"] = int(recovered)
    health["dst_ambiguous_rows"] = int(tz_amb)
    health["dst_nonexistent_rows"] = int(tz_nonexist)

    no_id = (df["PartID"].isna() | df["Feature"].isna()
             | (df["PartID"].astype(object).map(lambda v: str(v).strip()) == ""))
    health["rows_missing_part_or_feature"] = int(no_id.sum())
    df = df[~no_id].copy()

    # Exact duplicates (repeated rows, overlapping exports)
    key_cols = ["TimeRaw", "PartID", "Feature", "X_raw", "Y_raw", "Z_raw"]
    dup = df[key_cols].astype(object).map(lambda v: "" if pd.isna(v) else str(v)).duplicated(keep="first")
    health["duplicate_rows_removed"] = int(dup.sum())
    df = df[~dup].copy()

    bad_time = df["Date"].isna()
    health["rows_without_valid_time"] = int(bad_time.sum())
    health["examples_bad_time"] = df.loc[bad_time, "TimeRaw"].astype(object).map(str).head(5).tolist()
    df = df[~bad_time].copy()
    if df.empty:
        raise DataFormatError("No row has a readable timestamp.")

    df["PartKey"] = df["PartID"].astype(object).map(lambda v: str(v).strip())
    lookalikes = _similar_part_ids(df["PartKey"].unique())
    health["similar_part_ids"] = lookalikes
    health["lookalike_ids_merged"] = 0
    if opts.merge_lookalikes and lookalikes:
        counts = df["PartKey"].value_counts()
        mapping = {}
        for group in lookalikes:
            keep = max(group, key=lambda g: (counts.get(g, 0), g.count("0")))
            mapping.update({g: keep for g in group if g != keep})
        df["PartKey"] = df["PartKey"].replace(mapping)
        health["lookalike_ids_merged"] = len(mapping)
    df["Feature"] = df["Feature"].astype(object).map(lambda v: str(v).strip())

    for k in ("X", "Y", "Z", "X_lo", "X_hi", "Y_lo", "Y_hi", "Z_lo", "Z_hi"):
        df[k] = _to_number(df[k + "_raw"])
    has_z = bool(df["Z"].notna().mean() > 0.5)
    health["non_numeric_deviation_values"] = int(
        (df["X"].isna() & df["X_raw"].notna()).sum() + (df["Y"].isna() & df["Y_raw"].notna()).sum()
    )
    zero = (df["X"] == 0) & (df["Y"] == 0)
    if has_z:
        zero &= df["Z"].fillna(0) == 0
    health["zero_readings"] = int(zero.sum())
    health["zero_as_missing"] = bool(opts.zero_as_missing)
    if opts.zero_as_missing and zero.any():
        df.loc[zero, ["X", "Y", "Z"]] = np.nan

    limits = {}
    for ax in ("X", "Y", "Z"):
        lo, hi = _mode_value(df[f"{ax}_lo"]), _mode_value(df[f"{ax}_hi"])
        if lo is not None and hi is not None and hi > lo:
            limits[ax] = (lo, hi)
    health["limits_in_file"] = limits

    # Battery type per part, decided from every feature ever measured on it.
    feats = df.groupby("PartKey", sort=False)["Feature"].agg(lambda s: list(pd.unique(s)))
    pids = df.groupby("PartKey", sort=False)["PartID"].first()
    part_type = {k: determine_battery_type(pids[k], feats[k]) for k in feats.index}
    df["BatteryType"] = df["PartKey"].map(part_type)

    # Corner mapping, once per (part, feature) pair.
    pairs = df[["PartKey", "PartID", "Feature"]].drop_duplicates(["PartKey", "Feature"])
    pairs = pairs.assign(CornerIndex=[
        extract_corner_index(f, p, opts.feature_map) for p, f in zip(pairs["PartID"], pairs["Feature"])
    ])
    df = df.merge(pairs[["PartKey", "Feature", "CornerIndex"]], on=["PartKey", "Feature"], how="left")
    unmapped = df[df["CornerIndex"] == 0]
    health["rows_unmapped_feature"] = int(len(unmapped))
    health["unmapped_features"] = unmapped["Feature"].value_counts().head(20).to_dict()
    feat_map = (df.groupby(["Feature", "CornerIndex"]).size().rename("Rows").reset_index()
                .sort_values(["CornerIndex", "Feature"]))
    feat_map["Corner"] = feat_map["CornerIndex"].map(CORNER_INDEX).fillna("— not a corner")
    health["feature_mapping"] = feat_map[["Feature", "Corner", "Rows"]].to_dict("records")

    df = df.sort_values("Date", kind="stable").reset_index(drop=True)
    times_ns = df["Date"].to_numpy().astype("datetime64[ns]").astype("int64")
    df["RunNum"] = infer_runs(df["PartKey"].tolist(), df["Feature"].tolist(),
                              times_ns.tolist(), opts.run_gap_min)
    runs = _summarize_runs(df, has_z)
    runs = add_week_columns(runs, "Date")
    df = add_week_columns(df, "Date")

    dup_c = df[df["CornerIndex"] > 0].groupby(["PartKey", "RunNum", "CornerIndex"]).size()
    health["corner_measured_twice_in_a_run"] = int((dup_c > 1).sum())
    health["modules"] = int(runs["BaseKey"].nunique())
    health["module_runs"] = int(len(runs))
    health["repeat_runs"] = int((runs["RunNum"] > 1).sum())
    health["incomplete_runs"] = int(runs[list(DEV_COLS)].isna().any(axis=1).sum())
    health["date_min"] = runs["Date"].min()
    health["date_max"] = runs["Date"].max()
    health["has_z"] = has_z
    health["types"] = runs["BatteryType"].value_counts().to_dict()

    keep = ["Date", "WeekKey", "CalendarWeek", "PartKey", "PartID", "Feature", "CornerIndex",
            "BatteryType", "RunNum", "X", "Y", "Z", "SourceFile", "SourceRow"]
    return Dataset(rows=df[keep], runs=runs, health=health, has_z=has_z, limits=limits, files=file_info)


def _summarize_runs(df: pd.DataFrame, has_z: bool) -> pd.DataFrame:
    keys = ["PartKey", "RunNum"]
    g = df.groupby(keys, sort=False)
    base = pd.DataFrame({
        "Date": g["Date"].first(),
        "PartIDRaw": g["PartID"].first(),
        "BatteryType": g["BatteryType"].first(),
        "Readings": g.size(),
        "SourceFile": g["SourceFile"].first(),
    })
    corners = df[df["CornerIndex"].between(1, 4)].drop_duplicates(keys + ["CornerIndex"], keep="last")
    for idx, name in CORNER_INDEX.items():
        sub = corners[corners["CornerIndex"] == idx].set_index(keys)
        base[f"{name}_X"] = sub["X"].reindex(base.index).astype(float)
        base[f"{name}_Y"] = sub["Y"].reindex(base.index).astype(float)
        if has_z:
            base[f"{name}_Z"] = sub["Z"].reindex(base.index).astype(float)
    base = base.reset_index().rename(columns={"PartKey": "BaseKey"})
    base["PartID"] = base["BaseKey"]  # merged look-alike IDs share one PartID
    base = base.sort_values(["Date", "RunNum"], kind="stable").reset_index(drop=True)
    cols = ["Date", "PartID", "BaseKey", "BatteryType", "RunNum", *DEV_COLS]
    if has_z:
        cols += list(Z_COLS)
    return base[cols + ["Readings", "SourceFile"]]
