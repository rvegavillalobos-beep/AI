"""Height (Z) of the four corners: level, pitch, roll and twist.

With four corners the surface z = a + b·x + c·y + d·x·y fits exactly
(x, y relative to the nominal module center):

* level  a — height at the module center (mm)
* pitch  b — slope front↔rear (mm per m)
* roll   c — slope left↔right (mm per m)
* twist  d — the part no plane can follow (mm at the corners)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Nominals, normalize_type
from .constants import CORNERS, Z_COLS
from .geometry import rel_nominal

Z_METRICS = ("Z_Level", "Z_Pitch", "Z_Roll", "Z_Twist", "Z_Range", "Z_Worst")


def z_metrics(df: pd.DataFrame, nominals: Nominals) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in Z_METRICS:
        out[c] = np.nan
    if df.empty or not all(c in df for c in Z_COLS):
        return out
    Z = df[list(Z_COLS)].to_numpy(float)
    ok = np.isfinite(Z).all(axis=1)
    types = df["BatteryType"].map(normalize_type).to_numpy()
    level = np.full(len(df), np.nan)
    pitch = np.full(len(df), np.nan)
    roll = np.full(len(df), np.nan)
    twist = np.full(len(df), np.nan)
    for t in np.unique(types):
        m = ok & (types == t)
        if not m.any():
            continue
        rx, ry = rel_nominal(t, nominals)
        A = np.column_stack([np.ones(4), rx, ry, rx * ry])
        coef = np.linalg.solve(A, Z[m].T).T            # (k, 4)
        level[m] = coef[:, 0]
        pitch[m] = coef[:, 1] * 1000.0
        roll[m] = coef[:, 2] * 1000.0
        twist[m] = coef[:, 3] * float(np.mean(np.abs(rx * ry)))
    out["Z_Level"], out["Z_Pitch"], out["Z_Roll"], out["Z_Twist"] = level, pitch, roll, twist
    any_z = np.isfinite(Z).any(axis=1)
    with np.errstate(invalid="ignore"):
        zmax = np.where(any_z, np.nanmax(np.where(np.isfinite(Z), Z, -np.inf), axis=1), np.nan)
        zmin = np.where(any_z, np.nanmin(np.where(np.isfinite(Z), Z, np.inf), axis=1), np.nan)
    out["Z_Range"] = zmax - zmin
    absz = np.where(np.isfinite(Z), np.abs(Z), -1.0)
    out["Z_Worst"] = np.where(any_z, Z[np.arange(len(Z)), absz.argmax(axis=1)], np.nan)
    return out


def mean_surface(df: pd.DataFrame, bat_type: str, nominals: Nominals, n: int = 15) -> dict | None:
    """Average fitted surface of one battery type on an n×n grid (for 3D plots)."""
    sub = df[df["BatteryType"] == bat_type]
    if sub.empty or "Z_Level" not in sub or sub["Z_Level"].notna().sum() == 0:
        return None
    rx, ry = rel_nominal(bat_type, nominals)
    nom = nominals.get(bat_type)
    cx = np.mean([nom[c][0] for c in CORNERS])
    cy = np.mean([nom[c][1] for c in CORNERS])
    a = sub["Z_Level"].mean()
    b = sub["Z_Pitch"].mean() / 1000.0
    c = sub["Z_Roll"].mean() / 1000.0
    d = sub["Z_Twist"].mean() / float(np.mean(np.abs(rx * ry)))
    xs = np.linspace(rx.min(), rx.max(), n)
    ys = np.linspace(ry.min(), ry.max(), n)
    X, Y = np.meshgrid(xs, ys)
    Zs = a + b * X + c * Y + d * X * Y
    corner_z = {k: float(sub[f"{k}_Z"].mean()) for k in CORNERS}
    return {"x": xs + cx, "y": ys + cy, "z": Zs, "corner_z": corner_z, "nominal": nom}
