"""Error anatomy, failure mechanisms, squareness and the rigid compensation."""

import numpy as np
import pandas as pd
import pytest

from rickhard import geometry
from rickhard.config import Nominals
from rickhard.constants import DEV_COLS, MODES

NOM = Nominals()
TYPES = ["Type S", "Type M"]


def frame(bat_type: str, D, status: str = "FAIL") -> pd.DataFrame:
    D = np.atleast_2d(np.asarray(D, float))
    df = pd.DataFrame(D, columns=list(DEV_COLS))
    df.insert(0, "BatteryType", bat_type)
    df["Status"] = status
    return df


def rigid_deviations(bat_type: str, dx: float, dy: float, theta_deg: float) -> np.ndarray:
    """Exact deviations of a module rotated about its nominal centre and shifted."""
    rx, ry = geometry.rel_nominal(bat_type, NOM)
    x, y = geometry.rigid_transform(rx, ry, 0.0, 0.0, dx, dy, theta_deg)
    D = np.empty(8)
    D[0::2], D[1::2] = x - rx, y - ry
    return D


@pytest.mark.parametrize("t", TYPES)
def test_basis_is_orthonormal_and_keeps_shift_and_yaw_exact(t):
    E, rot_norm = geometry.mode_basis(t, NOM)
    np.testing.assert_allclose(E.T @ E, np.eye(8), atol=1e-12)
    np.testing.assert_allclose(E[:, 0], np.tile([1, 0], 4) / 2, atol=1e-12)
    np.testing.assert_allclose(E[:, 1], np.tile([0, 1], 4) / 2, atol=1e-12)
    rx, ry = geometry.rel_nominal(t, NOM)
    raw_rot = np.empty(8)
    raw_rot[0::2], raw_rot[1::2] = -ry, rx
    np.testing.assert_allclose(E[:, 2], raw_rot / np.linalg.norm(raw_rot), atol=1e-12)
    assert rot_norm == pytest.approx(np.linalg.norm(raw_rot))


@pytest.mark.parametrize("t", TYPES)
def test_decomposition_is_exact(t):
    rng = np.random.default_rng(1)
    D = rng.normal(0, 1.5, (50, 8))
    df = frame(t, D)
    coef = geometry.mode_coefficients(df, NOM)
    A = coef[[f"A_{m}" for m in MODES]].to_numpy()
    np.testing.assert_allclose(geometry.reconstruct(A, t, NOM), D, atol=1e-10)
    parts = geometry.split_rigid_shape(df, NOM)
    np.testing.assert_allclose(parts["T"] + parts["R"] + parts["S"], D, atol=1e-12)
    # energy is conserved: sum of squared amplitudes = sum of squared deviations / 4
    np.testing.assert_allclose((A ** 2).sum(axis=1), (D ** 2).sum(axis=1) / 4, rtol=1e-12)


@pytest.mark.parametrize("t", TYPES)
def test_rigid_motion_is_read_as_placement(t):
    D = rigid_deviations(t, 0.7, -0.4, 0.05)
    v = geometry.add_vectors(frame(t, D), NOM).iloc[0]
    assert v["Fit_Tx"] == pytest.approx(0.7, abs=2e-3)
    assert v["Fit_Ty"] == pytest.approx(-0.4, abs=2e-3)
    assert v["Fit_Rot"] == pytest.approx(0.05, rel=1e-3)
    assert v["Rotation_Angle"] == pytest.approx(0.05, rel=1e-3)   # the original yaw agrees
    assert v["ShapeMax"] < 5e-3                                    # a rigid move has (almost) no shape
    for m in ("ScaleX", "ScaleY", "Shear", "TrapX", "TrapY"):
        assert abs(v[f"A_{m}"]) < 5e-3


def test_failure_mechanisms():
    t = "Type S"
    E, _ = geometry.mode_basis(t, NOM)
    cases = {
        "Shift": np.tile([3.6, 0.2], 4),
        "Yaw": rigid_deviations(t, 0, 0, 0.4),
        "Shape": 9.0 * E[:, 5],               # a pure parallelogram
        "Pass": np.tile([0.5, -0.5], 4),
    }
    df = pd.concat([frame(t, d, "PASS" if k == "Pass" else "FAIL") for k, d in cases.items()], ignore_index=True)
    missing = frame(t, np.r_[np.nan, np.nan, np.zeros(6)])
    df = pd.concat([df, missing], ignore_index=True)
    assert (np.abs(df.loc[2, list(DEV_COLS)].to_numpy(float)) > 3).any()
    mech = geometry.failure_mechanism(df, 3.0, NOM)
    assert list(mech["Mechanism"]) == ["Shift", "Yaw", "Shape", "", "Missing corner"]
    assert mech.loc[0, "MechanismDetail"] == "Shift X"
    assert bool(mech.loc[0, "PassIfPlaced"]) and not bool(mech.loc[0, "PassIfSquare"])
    assert bool(mech.loc[2, "PassIfSquare"]) and not bool(mech.loc[2, "PassIfPlaced"])


def test_status_rules():
    df = pd.DataFrame({"BatteryType": ["Type S"] * 4, **{c: [0.0] * 4 for c in DEV_COLS}})
    df.loc[0, "FL_X"] = 3.0          # on the limit: still in tolerance (original rule |x| > limit)
    df.loc[1, "RR_Y"] = -3.001
    df.loc[2, "FR_X"] = np.nan
    df["FL_Z"] = [0.0, 0.0, 0.0, 1.5]
    for c in ("FR_Z", "RL_Z", "RR_Z"):
        df[c] = 0.0
    st = geometry.evaluate_status(df, 3.0)
    assert list(st["Status"]) == ["PASS", "FAIL", "INCOMPLETE", "PASS"]
    assert st.loc[1, "WorstCoord"] == "RR_Y" and st.loc[1, "Margin"] == pytest.approx(-0.001)
    stz = geometry.evaluate_status(df, 3.0, z_limits=(-1.25, 1.21))
    assert stz.loc[3, "Status"] == "FAIL"


@pytest.mark.parametrize("t", TYPES)
def test_rigid_move_is_square(t):
    sq = geometry.squareness(frame(t, rigid_deviations(t, 1.2, -0.8, 0.12)), 1.5, NOM).iloc[0]
    assert sq["DeltaDiag"] < 1e-6
    assert sq["SquarenessStatus"] == "SQUARE OK"


@pytest.mark.parametrize("t", TYPES)
def test_exact_inverse_compensation_puts_the_module_on_nominal(t):
    dx, dy, th = 1.1, -0.6, 0.15
    df = frame(t, rigid_deviations(t, dx, dy, th))
    # undo the rotation first, then the translation rotated back
    c, s = np.cos(np.radians(-th)), np.sin(np.radians(-th))
    tx, ty = c * dx - s * dy, s * dx + c * dy
    out = geometry.apply_compensation(df, (-tx, -ty, -th), NOM)
    np.testing.assert_allclose(out, 0.0, atol=1e-9)
    # per-type settings and "no setting" leave other types untouched
    both = pd.concat([df, frame("Type M" if t == "Type S" else "Type S", np.ones(8))], ignore_index=True)
    out2 = geometry.apply_compensation(both, {t: (-tx, -ty, -th)}, NOM)
    np.testing.assert_allclose(out2[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(out2[1], 1.0)


def test_why_each_nok_failed():
    """The counterfactual split used on Shape & Squareness (as in Corner QC)."""
    t = "Type S"
    E, _ = geometry.mode_basis(t, NOM)
    shear = E[:, 5] / np.abs(E[:, 5]).max()          # a pure parallelogram, 1 mm at its largest corner
    cases = {
        "Placement": np.tile([3.6, 0.0], 4),                       # shifted out, perfectly square
        "Shape + placement": np.tile([2.2, 0.0], 4) + 1.5 * shear,  # each alone fits, together they don't
        "Shape": 3.6 * shear,                                      # square would pass, placement can't help
        "Both": np.tile([3.6, 0.0], 4) + 3.6 * shear,
    }
    df = pd.concat([frame(t, d) for d in cases.values()], ignore_index=True)
    st = geometry.evaluate_status(df, 3.0)
    assert (st["Status"] == "FAIL").all()
    mech = geometry.failure_mechanism(st, 3.0, NOM)
    assert list(mech["NokCause"]) == list(cases)
    ok = geometry.failure_mechanism(frame(t, np.zeros(8), "PASS"), 3.0, NOM)
    assert ok["NokCause"].iloc[0] == ""


@pytest.mark.parametrize("t", TYPES)
def test_dimension_deviations(t):
    E, _ = geometry.mode_basis(t, NOM)
    df = pd.concat([frame(t, rigid_deviations(t, 1.3, -0.7, 0.2)),            # a rigid move: nothing changes
                    frame(t, 2.0 * E[:, 5]),                                   # a parallelogram
                    frame(t, np.r_[np.nan, np.nan, np.zeros(6)])], ignore_index=True)
    dims = geometry.dimension_deviations(df, NOM)
    np.testing.assert_allclose(dims.iloc[0].to_numpy(float), 0.0, atol=2e-3)
    par = dims.iloc[1]
    assert np.sign(par["Diag FL–RR"]) == -np.sign(par["Diag FR–RL"]) and abs(par["Diag FL–RR"]) > 0.5
    assert np.sign(par["Angle FL"]) == -np.sign(par["Angle FR"])
    assert dims.iloc[2].isna().all()                                            # incomplete: no geometry
    sq = geometry.squareness(df, 1.5, NOM)                                      # same numbers as the report
    assert sq.loc[1, "DeltaDiag"] == pytest.approx(abs(par["Diagonal difference"]))
    assert sq.loc[1, "AngleDevFL"] == pytest.approx(par["Angle FL"])


def test_nominal_dimensions_match_the_deviations():
    nd = geometry.nominal_dimensions(NOM, TYPES).set_index("BatteryType")
    for t in TYPES:
        nom = NOM.get(t)
        assert nd.loc[t, "Diag FL–RR"] == pytest.approx(np.hypot(nom["RR"][0] - nom["FL"][0], nom["RR"][1] - nom["FL"][1]))
        assert nd.loc[t, "Front FL–FR"] == pytest.approx(abs(nom["FR"][1] - nom["FL"][1]))
        # a module stretched 1 mm along X reads +1 mm on both long edges and nothing on the front edge
        df = frame(t, np.r_[0.5, 0, 0.5, 0, -0.5, 0, -0.5, 0])
        dev = geometry.dimension_deviations(df, NOM).iloc[0]
        assert dev["Left FL–RL"] == pytest.approx(1.0, abs=1e-3) and abs(dev["Front FL–FR"]) < 1e-6
