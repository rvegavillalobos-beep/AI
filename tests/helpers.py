"""Deterministic synthetic exports in the station's format, for the tests."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

from rickhard.demo import FEATURES

COLUMNS = ["Time", "Part ID", "Featurename", "X deviation", "X lower limit", "X upper limit",
           "Y deviation", "Y lower limit", "Y upper limit"]


def make_export(n_modules: int = 220, seed: int = 7, start: str = "2026-07-06 06:00:00",
                p_remeasure: float = 0.2, p_late: float = 0.04, p_missing: float = 0.03,
                sd: float = 1.1, newest_first: bool = True) -> pd.DataFrame:
    """A corner-deviation export with unique timestamps (one second apart at
    least), re-measurements right away and on a later day, missing corners
    and modules out of tolerance. Times are datetimes in German time."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    runs = []  # (intended start, part id, type, offsets, skip)
    t = t0
    for i in range(n_modules):
        btype = "Type S" if rng.random() < 0.6 else "Type M"
        pid = f"FO889952{'3' if btype == 'Type S' else '4'}0126B{i:05d}"
        t = t + pd.Timedelta(seconds=int(rng.integers(90, 420)))
        off = rng.normal(0, 0.8, 2)
        skip = int(rng.integers(0, 4)) if rng.random() < p_missing else -1
        runs.append((t, pid, btype, off, skip))
        if rng.random() < p_remeasure:
            runs.append((t + pd.Timedelta(seconds=int(rng.integers(60, 600))), pid, btype, off, -1))
        if rng.random() < p_late:
            runs.append((t + pd.Timedelta(seconds=int(rng.uniform(14, 40) * 3600)), pid, btype, off, -1))
    runs.sort(key=lambda r: r[0])

    rows = []
    last = t0 - pd.Timedelta(seconds=1)
    for start_t, pid, btype, off, skip in runs:
        tt = max(start_t, last + pd.Timedelta(seconds=1))
        for j, feat in enumerate(FEATURES[btype]):
            if j == skip:
                continue
            tt = max(tt + pd.Timedelta(seconds=int(rng.integers(1, 4))), last + pd.Timedelta(seconds=1))
            last = tt
            x = round(float(off[0] + rng.normal(0, sd)), 3)
            y = round(float(off[1] + rng.normal(0, sd)), 3)
            rows.append([tt.to_pydatetime(), pid, feat, x, -3.0, 3.0, y, -3.0, 3.0])
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.iloc[::-1].reset_index(drop=True) if newest_first else df


def to_xlsx(df: pd.DataFrame, title_rows: int = 2) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        if title_rows:
            pd.DataFrame([["Corner Cell Deviations for All"]] + [[None]] * (title_rows - 1)).to_excel(
                w, index=False, header=False, startrow=0, sheet_name="Sheet1")
        df.to_excel(w, index=False, startrow=title_rows, sheet_name="Sheet1")
    return buf.getvalue()


def to_csv(df: pd.DataFrame, sep: str = ",", decimal: str = ".", title_rows: int = 2,
           time_format: str = "%Y-%m-%d %H:%M:%S") -> bytes:
    out = df.copy()
    out["Time"] = pd.to_datetime(out["Time"]).dt.strftime(time_format)
    body = out.to_csv(index=False, sep=sep, decimal=decimal)
    head = "Corner Cell Deviations for All\n" + "\n" * (title_rows - 1) if title_rows else ""
    return (head + body).encode()


class NamedBytes(io.BytesIO):
    """A file-like object with a name, like Streamlit's UploadedFile."""

    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name
