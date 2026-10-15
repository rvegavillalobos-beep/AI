"""Synthetic corner-deviation exports in the same layout as the real one.

The numbers are invented, but the process behind them is realistic enough to
exercise every analysis in the app:

* systematic corner offsets that differ per corner (so centering the average
  module is not the same as maximising FPY),
* a conveyor end-stop adjustment and a guide-rail realignment (logged as
  events) that change the process,
* a slow yaw drift (guide wear), a night-shift offset and a start-up effect
  after production stops,
* a deformed sub-population (parallelograms and trapezoids),
* re-measurements in place and after re-seating, Z heights,
* and the usual data glitches: missing corners, all-zero failed reads,
  duplicate rows and Part IDs mistyped with the letter O.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import CORNERS, DEFAULT_NOMINALS

FEATURES = {
    "Type S": ["72_L0324_aa", "72_R0301_aa", "72_L0324_da", "72_R0302_da"],
    "Type M": ["72_L0324_aa", "72_R0301_aa", "72_L0324_dj", "72_R0301_dj"],
}
PREFIX = {"Type S": "FO88995230126B", "Type M": "FO88995310126B"}
LIMIT = 3.0


@dataclass(frozen=True)
class DemoSpec:
    seed: int = 11
    start: str = "2026-06-29"     # Monday, plant local time
    weeks: int = 15
    plant_tz: str = "America/Mexico_City"
    export_tz: str = "Europe/Berlin"


def _rel(bat_type: str) -> tuple[np.ndarray, np.ndarray]:
    nom = DEFAULT_NOMINALS[bat_type]
    xs = np.array([nom[c][0] for c in CORNERS])
    ys = np.array([nom[c][1] for c in CORNERS])
    return xs - xs.mean(), ys - ys.mean()


def demo_events(spec: DemoSpec = DemoSpec()) -> list[dict]:
    """The interventions injected into the demo process, as an event log."""
    t0 = pd.Timestamp(spec.start)
    return [
        {"Date": (t0 + pd.Timedelta(days=36, hours=10, minutes=30)).isoformat(),
         "Scope": "Type S", "Category": "Adjustment",
         "Title": "Conveyor end-stop moved −0.9 mm in X",
         "Notes": "Type S modules were landing long (front corners +X).",
         "dX": -0.9, "dY": 0.0, "dYaw": 0.0},
        {"Date": (t0 + pd.Timedelta(days=56, hours=7)).isoformat(),
         "Scope": "All", "Category": "Training",
         "Title": "Night-shift operator refresher",
         "Notes": "No machine change.", "dX": 0.0, "dY": 0.0, "dYaw": 0.0},
        {"Date": (t0 + pd.Timedelta(days=73, hours=15)).isoformat(),
         "Scope": "All", "Category": "Maintenance",
         "Title": "Side guide rail realigned, worn pads replaced",
         "Notes": "Yaw had been drifting for several weeks.", "dX": 0.0, "dY": 0.0, "dYaw": 0.0},
    ]


def _production_times(rng, spec: DemoSpec) -> list[pd.Timestamp]:
    """When each module is first measured: three shifts, batches with a
    takt of ~7 min and idle gaps between batches."""
    t0 = pd.Timestamp(spec.start)
    times = []
    for day in range(spec.weeks * 7):
        d = t0 + pd.Timedelta(days=day)
        wk = day // 7
        dow = d.dayofweek
        if dow == 6 or (dow == 5 and wk < 6):
            continue
        per_day = 12 if wk < 3 else (24 if wk < 7 else 36)
        if dow == 5:
            per_day = per_day // 3
        for s_start, s_factor in ((6, 1.05), (14, 1.0), (22, 0.85)):
            if dow == 5 and s_start != 6:
                continue
            n = rng.poisson(per_day / 3 * s_factor)
            if n == 0:
                continue
            n_batches = 1 if rng.random() < 0.65 else 2
            sizes = rng.multinomial(n, np.ones(n_batches) / n_batches)
            clock = rng.uniform(5, 60)
            for size in sizes:
                for _ in range(size):
                    tt = d + pd.Timedelta(hours=s_start) + pd.Timedelta(minutes=float(clock))
                    if clock < 470:
                        times.append(tt)
                    clock += max(3.0, rng.normal(7.0, 1.8))
                clock += rng.uniform(70, 160)
    return sorted(times)


def make_demo(spec: DemoSpec = DemoSpec()) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (export_rows, truth). ``export_rows`` has the real export's columns."""
    rng = np.random.default_rng(spec.seed)
    t0 = pd.Timestamp(spec.start)
    ev = demo_events(spec)
    e1, e3 = pd.Timestamp(ev[0]["Date"]), pd.Timestamp(ev[2]["Date"])

    times = _production_times(rng, spec)
    n_mod = len(times)

    # start-up effect: first 3 modules after >= 60 min without production
    gaps = np.diff(np.array([t.value for t in times])) / 6e10 if n_mod > 1 else np.array([])
    since_gap = np.zeros(n_mod, dtype=int)
    k = 99
    for i in range(n_mod):
        k = 0 if (i == 0 or gaps[i - 1] >= 60) else k + 1
        since_gap[i] = k

    rows, truth = [], []
    slips = [0]
    serial = {"Type S": 30000, "Type M": 60000}
    for i, tt in enumerate(times):
        days = (tt - t0) / pd.Timedelta(days=1)
        btype = "Type S" if rng.random() < (0.68 if days < 50 else 0.6) else "Type M"
        serial[btype] += int(rng.integers(1, 3))
        pid = f"{PREFIX[btype]}{serial[btype]:05d}N{int(rng.integers(1, 9)):07d}11296112"
        hour = tt.hour
        shift = "A" if 6 <= hour < 14 else ("B" if 14 <= hour < 22 else "C")
        rx, ry = _rel(btype)

        # ---- systematic placement (changes with the events) ----------------------
        if btype == "Type S":
            tx_b = 1.15 if tt < e1 else 0.22
            ty_b = -0.30
        else:
            tx_b = 0.45
            ty_b = -0.75
        weeks_in = days / 7.0
        if tt < e3:
            rot_b = (0.012 if btype == "Type S" else -0.025) + 0.011 * max(0.0, weeks_in - 5)
            rot_sd = 0.055 + 0.002 * max(0.0, weeks_in - 5)
        else:
            rot_b = -0.004 if btype == "Type S" else -0.02
            rot_sd = 0.038
        if shift == "C":
            ty_b += 0.45
        startup = 0.75 * (0.5 ** since_gap[i]) if since_gap[i] < 3 else 0.0

        sd_x = 0.85 * (1.2 if shift == "C" else 1.0)
        sd_y = 0.72

        def placement():
            return (tx_b + startup + rng.normal(0, sd_x), ty_b + rng.normal(0, sd_y),
                    np.radians(rot_b + rng.normal(0, rot_sd)))

        tx, ty, th = placement()

        # ---- shape -----------------------------------------------------------------
        shape_x = rng.normal(0, 0.12, 4)
        shape_y = rng.normal(0, 0.12, 4)
        kind = "square"
        u = rng.random()
        if u < 0.035:
            kind = "parallelogram"
            s = rng.normal(0.0016, 0.0004) * rng.choice([-1, 1], p=[0.25, 0.75])
            shape_x += s * ry
            shape_y += s * rx
        elif u < 0.065:
            kind = "trapezoid width"
            dw = rng.normal(1.9, 0.4)
            shape_y += np.where(rx > 0, 1.0, 0.0) * np.sign(ry) * dw / 2
        elif u < 0.08:
            kind = "trapezoid length"
            dl = rng.normal(1.8, 0.4)
            shape_x += np.where(ry < 0, 1.0, 0.0) * np.sign(rx) * dl / 2
        # fixture / measurement offsets that differ per corner
        fixture_x = np.array([0.0, 0.2, -0.22, 0.05])
        fixture_y = np.array([-0.05, 0.4, -0.1, -0.3 if btype == "Type M" else 0.1])

        z0, pitch = rng.normal(-0.12, 0.18), rng.normal(0, 0.22)
        roll, twist = rng.normal(0.08, 0.2), rng.normal(0, 0.12)
        pitch += 0.12 * (tx - tx_b)

        def corners(tx_, ty_, th_, noise=0.15):
            dx = tx_ - th_ * ry + shape_x + fixture_x + rng.normal(0, noise, 4)
            dy = ty_ + th_ * rx + shape_y + fixture_y + rng.normal(0, noise, 4)
            dz = (z0 + pitch * rx / 1000 + roll * ry / 1000 + twist * np.sign(rx * ry) / 2
                  + rng.normal(0, 0.06, 4))
            return dx, dy, dz

        dx, dy, dz = corners(tx, ty, th)
        runs = [(tt, dx, dy, dz)]
        fail = bool((np.abs(dx) > LIMIT).any() or (np.abs(dy) > LIMIT).any())
        if (fail and rng.random() < 0.75) or (not fail and rng.random() < 0.04):
            t2 = tt + pd.Timedelta(minutes=float(rng.uniform(4, 70)))
            if rng.random() < 0.6:   # measured again in place
                dx2, dy2, dz2 = corners(tx, ty, th)
            else:                    # taken out and re-seated
                dx2, dy2, dz2 = corners(*placement())
            runs.append((t2, dx2, dy2, dz2))
            if fail and rng.random() < 0.15:
                t3 = t2 + pd.Timedelta(hours=float(rng.uniform(16, 40)))
                dx3, dy3, dz3 = corners(tx_b * 0.2 + rng.normal(0, 0.4), ty_b * 0.2 + rng.normal(0, 0.4), 0.0)
                runs.append((t3, dx3, dy3, dz3))

        truth.append({"PartID": pid, "BatteryType": btype, "Date": tt, "Shift": shift, "Tx": tx, "Ty": ty,
                      "RotDeg": np.degrees(th), "Shape": kind, "StartUp": bool(since_gap[i] < 3)})

        miss_corner = int(rng.integers(0, 4)) if rng.random() < 0.006 else -1
        slip = len(runs) > 1 and i > 40 and slips[0] < 2
        if slip:
            slips[0] += 1
        for r_i, (when, cx, cy, cz) in enumerate(runs):
            pid_r = pid.replace("N0", "NO", 1) if (slip and r_i == 0) else pid  # scanner slip: O for 0
            for j, feat in enumerate(FEATURES[btype]):
                if r_i == 0 and j == miss_corner:
                    continue
                x, y, z = float(cx[j]), float(cy[j]), float(cz[j])
                if rng.random() < 0.0015:
                    x = y = z = 0.0
                elif rng.random() < 0.001:
                    x = y = np.nan
                rows.append({
                    "When": when, "Part ID": pid_r, "Featurename": feat,
                    "X deviation": round(x, 3), "X lower limit": -LIMIT, "X upper limit": LIMIT,
                    "Y deviation": round(y, 3), "Y lower limit": -LIMIT, "Y upper limit": LIMIT,
                    "Z deviation": round(z, 3), "Z lower limit": -1.25, "Z upper limit": 1.21,
                })

    df = pd.DataFrame(rows)
    # plant time -> export time (Germany), formatted like the export
    local = pd.to_datetime(df["When"]).dt.tz_localize(spec.plant_tz).dt.tz_convert(spec.export_tz)
    df["Time"] = [t.strftime("%b %d, %Y %I:%M%p").replace(" 0", " ").replace("AM", "am").replace("PM", "pm")
                  for t in local]
    df = df.sort_values("When", kind="stable")
    dups = df.sample(frac=0.003, random_state=spec.seed)
    df = pd.concat([df, dups]).sort_values("When", kind="stable")
    cols = ["Time", "Part ID", "Featurename", "X deviation", "X lower limit", "X upper limit",
            "Y deviation", "Y lower limit", "Y upper limit", "Z deviation", "Z lower limit", "Z upper limit"]
    export = df[cols].iloc[::-1].reset_index(drop=True)  # newest first, like the real export
    return export, pd.DataFrame(truth)


def to_export_bytes(df: pd.DataFrame, fmt: str = "xlsx") -> bytes:
    """Write with the two title rows above the header, like the real export."""
    if fmt == "csv":
        return ("Corner Cell Deviations for All\n\n" + df.to_csv(index=False)).encode()
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame([["Corner Cell Deviations for All"], [None]]).to_excel(
            w, index=False, header=False, startrow=0, sheet_name="Sheet1")
        df.to_excel(w, index=False, startrow=2, sheet_name="Sheet1")
    return buf.getvalue()


def demo_file(fmt: str = "xlsx", spec: DemoSpec = DemoSpec()) -> bytes:
    export, _ = make_demo(spec)
    return to_export_bytes(export, fmt)


if __name__ == "__main__":  # pragma: no cover
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "demo_export.xlsx"
    ex, tr = make_demo()
    open(out, "wb").write(to_export_bytes(ex, "csv" if out.endswith(".csv") else "xlsx"))
    print(f"wrote {out}: {len(ex)} rows, {tr.shape[0]} modules")
