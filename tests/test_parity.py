"""The new code reproduces the original RICKHARDV numbers on the same export."""

import numpy as np
import pandas as pd
import pytest

import original_reference as orig
from helpers import NamedBytes
from rickhard import compensation, geometry, loading, metrics
from rickhard.config import Nominals
from rickhard.constants import CORNERS, DEV_COLS

LIMIT = 3.0
NOM = Nominals()


@pytest.fixture(scope="module")
def both(synth_xlsx, original_opts):
    _, summary, _, first_valid = orig.original_pipeline(NamedBytes(synth_xlsx, "synthetic.xlsx"), LIMIT, True)
    ds = loading.build_dataset([("synthetic.xlsx", synth_xlsx)], original_opts)
    runs = geometry.evaluate_status(ds.runs, LIMIT)
    first = metrics.first_runs(runs, incomplete_as_fail=True)
    return summary, first_valid, runs, first


def _merge(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    old = old.assign(PartID=old["PartID"].astype(str), RunNum=old["RunNum"].astype(int))
    new = new.assign(PartID=new["PartID"].astype(str), RunNum=new["RunNum"].astype(int))
    m = old.merge(new, on=["PartID", "RunNum"], how="outer", suffixes=("_old", "_new"), indicator=True)
    assert (m["_merge"] == "both").all(), m.loc[m["_merge"] != "both", ["PartID", "RunNum", "_merge"]]
    return m


def test_same_runs(both):
    summary, _, runs, _ = both
    assert len(summary) == len(runs)
    assert summary["RunNum"].max() >= 2  # the export contains re-measurements
    m = _merge(summary, runs)
    assert (pd.to_datetime(m["Date_old"]) == pd.to_datetime(m["Date_new"])).all()
    assert (m["BatteryType_old"] == m["BatteryType_new"]).all()
    for c in DEV_COLS:
        np.testing.assert_allclose(m[f"{c}_old"].astype(float), m[f"{c}_new"].astype(float), equal_nan=True)


def test_same_status(both):
    summary, _, runs, _ = both
    m = _merge(summary, runs)
    assert set(m["Status_old"]) >= {"PASS", "FAIL", "INCOMPLETE"}
    assert (m["Status_old"] == m["Status_new"]).all()
    assert (m["CornersOutOfSpec_old"] == m["CornersOutOfSpec_new"]).all()
    assert (m["MissingCorners_old"] == m["MissingCorners_new"]).all()


def test_same_first_runs_and_fpy(both):
    _, first_valid, _, first = both
    old = first_valid.set_index(first_valid["PartID"].astype(str))["Status"]
    new = first.set_index(first["PartID"].astype(str))["Status"]
    assert set(old.index) == set(new.index)
    assert (old.sort_index() == new.sort_index()).all()
    fpy_old = (first_valid["Status"] == "PASS").mean() * 100
    assert metrics.fpy(first) == pytest.approx(fpy_old)


def test_same_centroid_and_yaw(both):
    summary, _, runs, _ = both
    old = orig.add_centroid_and_rotation(summary)
    new = geometry.add_vectors(runs, NOM)
    m = _merge(old, new)
    for c in ("Centroid_X", "Centroid_Y", "Vector_Magnitude", "Rotation_Angle"):
        np.testing.assert_allclose(m[f"{c}_old"].astype(float), m[f"{c}_new"].astype(float),
                                   rtol=0, atol=1e-9, equal_nan=True, err_msg=c)


def test_same_squareness(both):
    summary, _, runs, _ = both
    old = orig.original_squareness(summary, 1.5)
    new = geometry.squareness(runs, 1.5, NOM)
    m = _merge(old, new)
    pairs = {"Diag 1 [mm]": "Diag1", "Diag 2 [mm]": "Diag2", "Delta Diag [mm]": "DeltaDiag",
             "Width Delta [mm]": "WidthDelta", "Length Delta [mm]": "LengthDelta", "FL Angular Dev [°]": "AngleDevFL"}
    for o, n in pairs.items():
        np.testing.assert_allclose(m[o].astype(float), m[n].astype(float).round(2), atol=0.0100001, err_msg=o)
    assert (m["Squareness Status"] == m["SquarenessStatus"]).all()
    assert (m["Root Cause Details"] == m["RootCause"]).all()


@pytest.mark.parametrize("bat_type", ["Type S", "Type M"])
def test_same_compensation(both, bat_type):
    summary, first_valid, runs, first = both
    # original: robust bias of the first runs of one type -> minus the medians
    old_first = orig.add_centroid_and_rotation(first_valid[first_valid["BatteryType"] == bat_type])
    bias = orig.compute_robust_bias(old_first, "median")
    rec = (-bias["Centroid_X"]["value"], -bias["Centroid_Y"]["value"], -bias["Rotation_Angle"]["value"])
    new_first = geometry.add_vectors(first[first["BatteryType"] == bat_type], NOM)
    got = compensation.median_suggestion(new_first)
    np.testing.assert_allclose(got, rec, atol=1e-12)

    # the simulated deviations and statuses of every run of that type
    sel_old = summary[summary["BatteryType"] == bat_type]
    sim_old = orig.simulate_compensation(sel_old, *rec, LIMIT)
    geo_old = orig.build_sim_geometry(sim_old)
    sim_new = compensation.simulate_table(runs[runs["BatteryType"] == bat_type], rec, LIMIT, NOM)
    m = _merge(sim_old.assign(**{f"G_{c}": geo_old[c] for c in ("Centroid_X", "Centroid_Y", "Rotation_Angle")}),
               sim_new)
    for c in DEV_COLS:
        np.testing.assert_allclose(m[f"{c}_Sim_old"].astype(float), m[f"{c}_Sim_new"].astype(float),
                                   atol=1e-9, equal_nan=True)
    assert (m["Status_Sim_old"] == m["Status_Sim_new"]).all()
    for c in ("Centroid_X", "Centroid_Y", "Rotation_Angle"):
        np.testing.assert_allclose(m[f"G_{c}"].astype(float), m[f"{c}_Sim"].astype(float), atol=1e-9,
                                   equal_nan=True)


@pytest.mark.parametrize("bat_type", ["Type S", "Type M"])
def test_same_corner_offsets(bat_type):
    old = orig.compute_corner_offsets(bat_type, 0.8, -0.35, 0.07)
    new = geometry.corner_offsets(bat_type, 0.8, -0.35, 0.07, NOM)
    assert list(new["Corner"]) == list(CORNERS)
    np.testing.assert_allclose(new["Offset X [mm]"], old["Equivalent_Offset_X_mm"], atol=1e-12)
    np.testing.assert_allclose(new["Offset Y [mm]"], old["Equivalent_Offset_Y_mm"], atol=1e-12)


@pytest.mark.parametrize("feature,part", [
    ("72_L0324_aa", "FO88995230126B00001"), ("72_R0301_aa", "X"), ("72_L0324_da", "X"), ("72_R0302_da", "X"),
    ("72_L0324_dj", "X"), ("72_R0301_dj", "X"), ("C1 corner", "P_M"), ("RR_x", "P TYPE S"), ("something", "P"),
    ("l00324_aa", "A_DI1"), ("r00301_dj", "TYPEM"),
])
def test_same_domain_rules(feature, part):
    assert loading.determine_battery_type(part, feature) == orig.determine_battery_type(part, feature)
    assert loading.extract_corner_index(feature, part) == orig.extract_corner_index(feature, part)
