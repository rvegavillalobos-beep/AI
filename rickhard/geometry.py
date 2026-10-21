"""Geometry on module corners.

* pass / fail status of a run,
* centroid and yaw (exactly as in the original app),
* the **error anatomy**: every module's eight corner deviations written as
  the sum of eight physical modes — shift X, shift Y, yaw, length stretch,
  width stretch, parallelogram, trapezoid (length), trapezoid (width).
  The decomposition is exact and orthonormal, so variances add up,
* squareness (original formulas) and the failure mechanism of every NOK,
* rigid roto-translation used by the compensation tools.

All functions are vectorised over modules and never modify their input.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import Nominals, normalize_type
from .constants import ANGULAR_DEV_TOL, CORNERS, DEV_COLS, DIM_DELTA_TOL, MODES, Z_COLS

# ---------------------------------------------------------------------------
# Arrays
# ---------------------------------------------------------------------------


def dev_matrix(df: pd.DataFrame, suffix: str = "") -> np.ndarray:
    """(n, 8) deviations ordered FL_X, FL_Y, FR_X, FR_Y, RL_X, RL_Y, RR_X, RR_Y."""
    if df.empty:
        return np.zeros((0, 8))
    return df[[f"{c}{suffix}" for c in DEV_COLS]].to_numpy(dtype=float)


def nominal_xy(types: pd.Series, nominals: Nominals) -> tuple[np.ndarray, np.ndarray]:
    """(n, 4) absolute nominal X and Y per row."""
    norm = types.map(normalize_type).to_numpy()
    nx = np.empty((len(norm), 4))
    ny = np.empty((len(norm), 4))
    for t in (np.unique(norm) if len(norm) else []):
        m = norm == t
        nom = nominals.get(t)
        nx[m] = [nom[c][0] for c in CORNERS]
        ny[m] = [nom[c][1] for c in CORNERS]
    return nx, ny


def rel_nominal(bat_type: str, nominals: Nominals) -> tuple[np.ndarray, np.ndarray]:
    """(4,) nominal corner positions relative to the nominal module center."""
    nom = nominals.get(bat_type)
    xs = np.array([nom[c][0] for c in CORNERS], float)
    ys = np.array([nom[c][1] for c in CORNERS], float)
    return xs - xs.mean(), ys - ys.mean()


def is_complete(df: pd.DataFrame, suffix: str = "") -> pd.Series:
    if df.empty:
        return pd.Series(dtype=bool, index=df.index)
    return df[[f"{c}{suffix}" for c in DEV_COLS]].notna().all(axis=1)


# ---------------------------------------------------------------------------
# PASS / FAIL / INCOMPLETE
# ---------------------------------------------------------------------------

def evaluate_status(df: pd.DataFrame, limit: float, suffix: str = "",
                    z_limits: tuple[float, float] | None = None) -> pd.DataFrame:
    """Adds CornersOutOfSpec, MissingCorners, IsComplete, Status, WorstAbs,
    Margin and WorstCoord (with ``suffix`` appended to every new column).

    A corner is missing when its X or Y is missing. A complete run FAILs when
    any corner has |X| or |Y| above the limit (and, with ``z_limits``, Z
    outside its limits).
    """
    out = df.copy()
    n = len(df)
    D = dev_matrix(df, suffix) if n else np.zeros((0, 8))
    absd = np.abs(D)
    miss = np.isnan(D[:, 0::2]) | np.isnan(D[:, 1::2])          # (n, 4)
    with np.errstate(invalid="ignore"):
        oos = (absd[:, 0::2] > limit) | (absd[:, 1::2] > limit)  # NaN -> False
    if z_limits is not None and all(f"{c}{suffix}" in df.columns for c in Z_COLS):
        Z = df[[f"{c}{suffix}" for c in Z_COLS]].to_numpy(float)
        with np.errstate(invalid="ignore"):
            oos |= (Z < z_limits[0]) | (Z > z_limits[1])
    oos &= ~miss
    missing = miss.sum(axis=1)
    n_oos = oos.sum(axis=1)
    if n:
        filled = np.where(np.isnan(absd), -1.0, absd)
        idx = filled.argmax(axis=1)
        worst = filled[np.arange(n), idx]
        worst = np.where(worst < 0, np.nan, worst)
        labels = np.array(DEV_COLS, dtype=object)[idx]
        worst_coord = np.where(np.isfinite(worst), labels, None)
    else:
        worst = np.array([], float)
        worst_coord = np.array([], dtype=object)
    out[f"CornersOutOfSpec{suffix}"] = n_oos
    out[f"MissingCorners{suffix}"] = missing
    out[f"IsComplete{suffix}"] = missing == 0
    out[f"Status{suffix}"] = np.where(missing > 0, "INCOMPLETE", np.where(n_oos > 0, "FAIL", "PASS"))
    out[f"WorstAbs{suffix}"] = worst
    out[f"Margin{suffix}"] = limit - worst
    out[f"WorstCoord{suffix}"] = worst_coord
    return out


# ---------------------------------------------------------------------------
# Centroid and yaw (original definitions)
# ---------------------------------------------------------------------------

def yaw_angles(df: pd.DataFrame, nominals: Nominals, suffix: str = "") -> np.ndarray:
    """Angle change of the rear-mid → front-mid line vs nominal, degrees,
    wrapped to (-180, 180] (original app's 'Rotation_Angle')."""
    if df.empty:
        return np.array([], dtype=float)
    nx, ny = nominal_xy(df["BatteryType"], nominals)
    D = dev_matrix(df, suffix)
    ax, ay = nx + D[:, 0::2], ny + D[:, 1::2]

    def ang(x, y):
        fx, fy = (x[:, 0] + x[:, 1]) / 2, (y[:, 0] + y[:, 1]) / 2
        rx, ry = (x[:, 2] + x[:, 3]) / 2, (y[:, 2] + y[:, 3]) / 2
        return np.degrees(np.arctan2(fy - ry, fx - rx))

    return (ang(ax, ay) - ang(nx, ny) + 180) % 360 - 180


# ---------------------------------------------------------------------------
# Error anatomy: an orthonormal basis of 8 physical modes per battery type
# ---------------------------------------------------------------------------

def _raw_basis(rx: np.ndarray, ry: np.ndarray) -> np.ndarray:
    """(8, 8) raw mode vectors as columns, in MODES order."""
    z = np.zeros(4)
    one = np.ones(4)
    s = float(np.sqrt(np.mean(rx ** 2) * np.mean(ry ** 2))) or 1.0
    vecs = {
        "Tx": (one, z),
        "Ty": (z, one),
        "Rot": (-ry, rx),
        "ScaleX": (rx, z),
        "ScaleY": (z, ry),
        "Shear": (ry, rx),
        "TrapX": (rx * ry / s, z),
        "TrapY": (z, rx * ry / s),
    }
    B = np.zeros((8, 8))
    for j, m in enumerate(MODES):
        vx, vy = vecs[m]
        B[0::2, j] = vx
        B[1::2, j] = vy
    return B


@lru_cache(maxsize=32)
def _basis_cached(rx_t: tuple, ry_t: tuple) -> tuple[np.ndarray, float]:
    rx, ry = np.array(rx_t), np.array(ry_t)
    B = _raw_basis(rx, ry)
    E = np.zeros_like(B)
    for j in range(8):  # Gram–Schmidt in MODES order (shift and yaw stay exact)
        v = B[:, j].copy()
        for k in range(j):
            v -= (E[:, k] @ v) * E[:, k]
        nrm = np.linalg.norm(v)
        E[:, j] = v / nrm if nrm > 1e-12 else 0.0
    return E, float(np.linalg.norm(B[:, 2]))


def mode_basis(bat_type: str, nominals: Nominals) -> tuple[np.ndarray, float]:
    """Orthonormal (8, 8) basis (columns = MODES) and the norm of the raw
    rotation vector (to convert the yaw coefficient to radians)."""
    rx, ry = rel_nominal(bat_type, nominals)
    E, rot_norm = _basis_cached(tuple(np.round(rx, 6)), tuple(np.round(ry, 6)))
    return E, rot_norm


def mode_coefficients(df: pd.DataFrame, nominals: Nominals, suffix: str = "") -> pd.DataFrame:
    """Mode amplitudes per module (complete modules only, NaN otherwise).

    ``A_<mode>``: amplitude in mm = RMS displacement the mode gives the four
    corners. ``Fit_Tx`` / ``Fit_Ty`` (mm) and ``Fit_Rot`` (deg) are the
    least-squares placement of the module.
    """
    n = len(df)
    out = pd.DataFrame(index=df.index)
    cols = {f"A_{m}": np.full(n, np.nan) for m in MODES}
    fit_rot = np.full(n, np.nan)
    if n:
        D = dev_matrix(df, suffix)
        ok = ~np.isnan(D).any(axis=1)
        types = df["BatteryType"].map(normalize_type).to_numpy()
        for t in np.unique(types):
            m = ok & (types == t)
            if not m.any():
                continue
            E, rot_norm = mode_basis(t, nominals)
            C = D[m] @ E                       # (k, 8) orthonormal coefficients
            for j, mode in enumerate(MODES):
                cols[f"A_{mode}"][m] = C[:, j] / 2.0
            fit_rot[m] = np.degrees(C[:, 2] / rot_norm)
    for k, v in cols.items():
        out[k] = v
    out["Fit_Tx"] = out["A_Tx"]
    out["Fit_Ty"] = out["A_Ty"]
    out["Fit_Rot"] = fit_rot
    return out


def reconstruct(coeffs: np.ndarray, bat_type: str, nominals: Nominals) -> np.ndarray:
    """(n, 8) deviations from (n, 8) mode amplitudes (inverse of the above)."""
    E, _ = mode_basis(bat_type, nominals)
    return (2.0 * coeffs) @ E.T


def split_rigid_shape(df: pd.DataFrame, nominals: Nominals) -> dict[str, np.ndarray]:
    """For complete modules: deviations D and their parts — shift T, yaw R,
    shape S (everything a rigid move cannot explain). D = T + R + S exactly."""
    D = dev_matrix(df)
    n = len(df)
    T = np.full((n, 8), np.nan)
    R = np.full((n, 8), np.nan)
    if n:
        ok = ~np.isnan(D).any(axis=1)
        types = df["BatteryType"].map(normalize_type).to_numpy()
        for t in np.unique(types):
            m = ok & (types == t)
            if not m.any():
                continue
            E, _ = mode_basis(t, nominals)
            C = D[m] @ E
            T[m] = C[:, :2] @ E[:, :2].T
            R[m] = C[:, 2:3] @ E[:, 2:3].T
    return {"D": D, "T": T, "R": R, "S": D - T - R}


# ---------------------------------------------------------------------------
# Vectors: original centroid / yaw + error anatomy
# ---------------------------------------------------------------------------

def add_vectors(df: pd.DataFrame, nominals: Nominals) -> pd.DataFrame:
    """Centroid, magnitude and yaw (original) plus mode amplitudes, the
    least-squares placement and the rigid / shape split maxima."""
    out = df.copy()
    D = dev_matrix(df)
    n = len(df)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["Centroid_X"] = np.nanmean(D[:, 0::2], axis=1) if n else np.array([], float)
        out["Centroid_Y"] = np.nanmean(D[:, 1::2], axis=1) if n else np.array([], float)
    out["Vector_Magnitude"] = np.hypot(out["Centroid_X"], out["Centroid_Y"])
    out["Rotation_Angle"] = yaw_angles(df, nominals)
    coef = mode_coefficients(df, nominals)
    for c in coef.columns:
        out[c] = coef[c].to_numpy()
    parts = split_rigid_shape(df, nominals)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out["RigidMax"] = np.nanmax(np.abs(parts["T"] + parts["R"]), axis=1) if n else np.array([], float)
        out["ShapeMax"] = np.nanmax(np.abs(parts["S"]), axis=1) if n else np.array([], float)
    return out


# ---------------------------------------------------------------------------
# Failure mechanism of every NOK
# ---------------------------------------------------------------------------

def failure_mechanism(df: pd.DataFrame, limit: float, nominals: Nominals) -> pd.DataFrame:
    """One mechanism per NOK run (empty string for PASS).

    The deviations are split into shift (T), yaw (R) and shape (S): D = T+R+S.
    On every out-of-tolerance coordinate each part either pushes the corner
    outwards or holds it back; the mechanism names the part(s) doing the
    pushing: Shift, Yaw or Shape when one part carries ≥ 60 % of it,
    "Shift + yaw" when placement as a whole carries ≥ 75 %, otherwise
    "Placement + shape". Runs with a missing corner are "Missing corner".

    Two counterfactuals are added for every complete run:
    PassIfPlaced (perfect placement, own shape) and PassIfSquare (own
    placement, perfectly square module).
    """
    parts = split_rigid_shape(df, nominals)
    D, T, R, S = parts["D"], parts["T"], parts["R"], parts["S"]
    n = len(df)
    out = pd.DataFrame(index=df.index)
    if not n:
        for c in ("Mechanism", "MechanismDetail", "NokCause"):
            out[c] = pd.Series(dtype=object)
        for c in ("ShareShift", "ShareYaw", "ShareShape"):
            out[c] = pd.Series(dtype=float)
        out["PassIfPlaced"] = pd.Series(dtype=bool)
        out["PassIfSquare"] = pd.Series(dtype=bool)
        return out
    status = df["Status"].to_numpy() if "Status" in df else np.array(["PASS"] * n)
    complete = ~np.isnan(D).any(axis=1)

    Dn = np.nan_to_num(D)
    sign = np.sign(Dn)
    excess = np.abs(Dn) > limit
    Tx = np.zeros_like(T)
    Tx[:, 0::2] = T[:, 0::2]
    Ty = T - Tx
    contrib = np.stack([
        (np.nan_to_num(Tx) * sign * excess).sum(axis=1),
        (np.nan_to_num(Ty) * sign * excess).sum(axis=1),
        (np.nan_to_num(R) * sign * excess).sum(axis=1),
        (np.nan_to_num(S) * sign * excess).sum(axis=1),
    ], axis=1)
    pos = np.clip(contrib, 0, None)
    tot = pos.sum(axis=1)
    share = np.divide(pos, tot[:, None], out=np.zeros_like(pos), where=tot[:, None] > 0)
    sh_shift = share[:, 0] + share[:, 1]
    sh_yaw, sh_shape = share[:, 2], share[:, 3]
    nok = status != "PASS"
    has_excess = tot > 0
    mech = np.select(
        [~nok, ~complete, ~has_excess, sh_shift >= 0.6, sh_yaw >= 0.6, sh_shape >= 0.6,
         sh_shift + sh_yaw >= 0.75],
        ["", "Missing corner", "Height (Z)", "Shift", "Yaw", "Shape", "Shift + yaw"],
        default="Placement + shape",
    )
    detail = np.where(
        mech == "Shift",
        np.where(share[:, 0] >= 2 * share[:, 1], "Shift X",
                 np.where(share[:, 1] >= 2 * share[:, 0], "Shift Y", "Shift X+Y")),
        mech,
    )

    def ok(M):
        with np.errstate(invalid="ignore"):
            return (np.abs(M) <= limit + 1e-12).all(axis=1)

    out["Mechanism"] = mech
    out["MechanismDetail"] = detail
    out["ShareShift"] = np.where(nok & complete, sh_shift, np.nan)
    out["ShareYaw"] = np.where(nok & complete, sh_yaw, np.nan)
    out["ShareShape"] = np.where(nok & complete, sh_shape, np.nan)
    pip = np.where(complete, ok(S), False)        # perfect placement, own shape
    pis = np.where(complete, ok(D - S), False)    # own placement, perfect shape
    out["PassIfPlaced"] = pip
    out["PassIfSquare"] = pis
    out["NokCause"] = np.select(
        [~nok | ~complete, mech == "Height (Z)", pis & pip, pis & ~pip, ~pis & pip],
        ["", "Height (Z)", "Shape + placement", "Shape", "Placement"], default="Both")
    return out


# ---------------------------------------------------------------------------
# Squareness / deformation (original formulas)
# ---------------------------------------------------------------------------

def _dist(ax, ay, bx, by):
    return np.hypot(bx - ax, by - ay)


def _corner_angle(a, b, c):
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - a[0], c[1] - a[1]
    dot = v1x * v2x + v1y * v2y
    mag = np.hypot(v1x, v1y) * np.hypot(v2x, v2y)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.clip(dot / mag, -1.0, 1.0)
    return np.where(mag == 0, 0.0, np.degrees(np.arccos(cos)))


SQ_COLUMNS = ["Diag1", "Diag2", "DeltaDiag", "WidthDelta", "LengthDelta", "AngleDevFL",
              "SquarenessStatus", "RootCause", "CauseCategory"]

# the six distances of a module, as pairs of corner indices into CORNERS (FL, FR, RL, RR)
DIMENSIONS: dict[str, tuple[int, int]] = {
    "Diag FL–RR": (0, 3), "Diag FR–RL": (1, 2),
    "Front FL–FR": (0, 1), "Rear RL–RR": (2, 3),
    "Left FL–RL": (0, 2), "Right FR–RR": (1, 3),
}


def dimension_deviations(df: pd.DataFrame, nominals: Nominals) -> pd.DataFrame:
    """Measured minus nominal length (mm) of both diagonals and the four edges
    of every complete run, plus the corner angles' deviations (°).

    These distances do not depend on where the module was placed, only on its
    shape: a perfectly square module reads 0 everywhere."""
    cols = [*DIMENSIONS, "Diagonal difference", *[f"Angle {c}" for c in CORNERS]]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    if df.empty:
        return out
    nx, ny = nominal_xy(df["BatteryType"], nominals)
    D = dev_matrix(df)
    ax, ay = nx + D[:, 0::2], ny + D[:, 1::2]
    for name, (i, j) in DIMENSIONS.items():
        out[name] = _dist(ax[:, i], ay[:, i], ax[:, j], ay[:, j]) - _dist(nx[:, i], ny[:, i], nx[:, j], ny[:, j])
    out["Diagonal difference"] = out["Diag FL–RR"] - out["Diag FR–RL"]
    # each corner's angle between its two edges (FL: to FR and RL, FR: to FL and RR, …)
    nb = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}
    for k, c in enumerate(CORNERS):
        p, q = nb[k]

        def ang(x, y, k=k, p=p, q=q):
            return _corner_angle((x[:, k], y[:, k]), (x[:, p], y[:, p]), (x[:, q], y[:, q]))
        out[f"Angle {c}"] = ang(ax, ay) - ang(nx, ny)
    out.loc[np.isnan(D).any(axis=1), :] = np.nan     # a module with a missing corner has no shape to measure
    return out


def squareness(df: pd.DataFrame, diag_tol: float, nominals: Nominals) -> pd.DataFrame:
    """Deformation metrics for complete runs (incomplete runs are dropped).

    Diagonal delta = change of (diag FL–RR minus diag FR–RL) vs nominal.
    Above ``diag_tol`` the run is DEFORMED and the dominant pattern is named in
    this order: parallelogram tilt, trapezoidal width, trapezoidal length,
    combined asymmetry (thresholds 0.15° and 0.8 mm, as in the original).
    """
    d = df[is_complete(df)].copy() if len(df) else df.copy()
    if d.empty:
        for c in SQ_COLUMNS:
            d[c] = pd.Series(dtype=object if c in ("SquarenessStatus", "RootCause", "CauseCategory") else float)
        return d
    nx, ny = nominal_xy(d["BatteryType"], nominals)
    D = dev_matrix(d)
    ax, ay = nx + D[:, 0::2], ny + D[:, 1::2]

    def metrics(x, y):
        fl, fr, rl, rr = (x[:, 0], y[:, 0]), (x[:, 1], y[:, 1]), (x[:, 2], y[:, 2]), (x[:, 3], y[:, 3])
        d1 = _dist(*fl, *rr)
        d2 = _dist(*fr, *rl)
        w_top = _dist(*fl, *fr)
        w_bot = _dist(*rl, *rr)
        l_left = _dist(*fl, *rl)
        l_right = _dist(*fr, *rr)
        ang = _corner_angle(fl, fr, rl)
        return d1, d2, w_top, w_bot, l_left, l_right, ang

    d1n, d2n, wtn, wbn, lln, lrn, angn = metrics(nx, ny)
    d1a, d2a, wta, wba, lla, lra, anga = metrics(ax, ay)
    delta = np.abs((d1a - d2a) - (d1n - d2n))
    width_delta = (wta - wtn) - (wba - wbn)
    length_delta = (lla - lln) - (lra - lrn)
    ang_dev = anga - angn

    deformed = delta > diag_tol
    is_par = deformed & (np.abs(ang_dev) > ANGULAR_DEV_TOL) & (np.abs(width_delta) < DIM_DELTA_TOL)
    is_w = deformed & ~is_par & (np.abs(width_delta) >= DIM_DELTA_TOL)
    is_l = deformed & ~is_par & ~is_w & (np.abs(length_delta) >= DIM_DELTA_TOL)
    is_c = deformed & ~is_par & ~is_w & ~is_l
    category = np.select([is_par, is_w, is_l, is_c],
                         ["Parallelogram Tilt", "Trapezoidal Width", "Trapezoidal Length", "Combined Asymmetry"],
                         default="")
    detail = np.select(
        [is_par, is_w, is_l, is_c],
        [
            [f"PARALLELOGRAM DISTORTION (Tilt: {v:+.2f}°)" for v in ang_dev],
            [f"TRAPEZOIDAL WIDTH VARIATION (Delta: {v:+.2f} mm)" for v in width_delta],
            [f"TRAPEZOIDAL LENGTH VARIATION (Delta: {v:+.2f} mm)" for v in length_delta],
            [f"COMBINED ASYMMETRY (Diagonal Delta: {v:.2f} mm)" for v in delta],
        ],
        default="Geometry within acceptable tolerance",
    )
    d["Diag1"] = d1a
    d["Diag2"] = d2a
    d["DeltaDiag"] = delta
    d["WidthDelta"] = width_delta
    d["LengthDelta"] = length_delta
    d["AngleDevFL"] = ang_dev
    d["SquarenessStatus"] = np.where(deformed, "DEFORMED", "SQUARE OK")
    d["RootCause"] = detail
    d["CauseCategory"] = np.where(category == "", None, category)
    return d


# ---------------------------------------------------------------------------
# Rigid roto-translation (compensation)
# ---------------------------------------------------------------------------

def rigid_transform(x, y, pivot_x, pivot_y, dx, dy, theta_deg):
    """Rotate (x, y) by theta about the pivot, then translate by (dx, dy)."""
    t = np.radians(theta_deg)
    c, s = np.cos(t), np.sin(t)
    xr, yr = x - pivot_x, y - pivot_y
    return xr * c - yr * s + pivot_x + dx, xr * s + yr * c + pivot_y + dy


def apply_compensation(df: pd.DataFrame, comp, nominals: Nominals) -> np.ndarray:
    """(n, 8) deviations after a rigid compensation (rotation about the
    nominal module center). ``comp`` is (dx, dy, yaw) or a dict per type."""
    D = dev_matrix(df)
    if not len(df):
        return D
    out = np.full_like(D, np.nan)
    types = df["BatteryType"].map(normalize_type).to_numpy()
    for t in np.unique(types):
        m = types == t
        setting = comp.get(t) if isinstance(comp, dict) else comp
        if setting is None or not np.isfinite(setting).all():
            out[m] = D[m]
            continue
        dx, dy, yaw = setting
        rx, ry = rel_nominal(t, nominals)
        ax, ay = rx + D[m][:, 0::2], ry + D[m][:, 1::2]
        cx, cy = rigid_transform(ax, ay, 0.0, 0.0, dx, dy, yaw)
        sub = np.empty((int(m.sum()), 8))
        sub[:, 0::2] = cx - rx
        sub[:, 1::2] = cy - ry
        out[m] = sub
    return out


def with_deviations(df: pd.DataFrame, D: np.ndarray, suffix: str = "_Sim") -> pd.DataFrame:
    out = df.copy()
    for j, c in enumerate(DEV_COLS):
        out[f"{c}{suffix}"] = D[:, j] if len(df) else np.array([], float)
    return out


def corner_offsets(bat_type: str, dx: float, dy: float, yaw: float, nominals: Nominals) -> pd.DataFrame:
    """Per-corner displacement produced by a rigid compensation."""
    rx, ry = rel_nominal(bat_type, nominals)
    cx, cy = rigid_transform(rx, ry, 0.0, 0.0, dx, dy, yaw)
    return pd.DataFrame({
        "BatteryType": normalize_type(bat_type),
        "Corner": list(CORNERS),
        "Offset X [mm]": np.round(cx - rx, 3),
        "Offset Y [mm]": np.round(cy - ry, 3),
    })


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def order_corners_convex(points: dict) -> list:
    """Corner names ordered by angle around the centroid, so the drawn polygon
    never self-intersects even with a large exaggeration factor."""
    cx = np.mean([p[0] for p in points.values()])
    cy = np.mean([p[1] for p in points.values()])
    return sorted(points, key=lambda k: np.arctan2(points[k][1] - cy, points[k][0] - cx))


def polygon(points: dict) -> tuple[list, list]:
    order = order_corners_convex(points)
    xs = [points[c][0] for c in order] + [points[order[0]][0]]
    ys = [points[c][1] for c in order] + [points[order[0]][1]]
    return xs, ys
