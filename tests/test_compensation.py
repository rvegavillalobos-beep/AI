"""X / Y / yaw compensation: methods, exact evaluation, backtest and readiness."""

import numpy as np
import pandas as pd
import pytest

from rickhard import compensation as comp
from rickhard import geometry, loading
from rickhard.config import Nominals
from rickhard.constants import DEV_COLS

NOM = Nominals()
LIMIT = 3.0
BIAS = (1.3, -0.9, 0.12)   # the line places Type S modules 1.3 mm +X, 0.9 mm −Y, 0.12° yaw


def biased_modules(n: int = 480, bias=BIAS, sd_xy: float = 0.6, sd_yaw: float = 0.03, noise: float = 0.25,
                   seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rx, ry = geometry.rel_nominal("Type S", NOM)
    rows = []
    start = pd.Timestamp("2026-07-06 06:00")
    for i in range(n):
        dx, dy = bias[0] + rng.normal(0, sd_xy), bias[1] + rng.normal(0, sd_xy)
        th = bias[2] + rng.normal(0, sd_yaw)
        x, y = geometry.rigid_transform(rx, ry, 0.0, 0.0, dx, dy, th)
        D = np.empty(8)
        D[0::2], D[1::2] = x - rx + rng.normal(0, noise, 4), y - ry + rng.normal(0, noise, 4)
        rows.append({"Date": start + pd.Timedelta(hours=i * 8 * 7 * 24 / n), "BatteryType": "Type S",
                     **dict(zip(DEV_COLS, D))})
    df = loading.add_week_columns(pd.DataFrame(rows))
    df = geometry.evaluate_status(df, LIMIT)
    return geometry.add_vectors(df, NOM)


@pytest.fixture(scope="module")
def vec():
    return biased_modules()


def test_median_and_rigid_methods_point_against_the_bias(vec):
    med = comp.median_suggestion(vec)
    rig = comp.rigid_mean_suggestion(vec)
    for got in (med, rig):
        assert got[0] == pytest.approx(-BIAS[0], abs=0.12)
        assert got[1] == pytest.approx(-BIAS[1], abs=0.12)
        assert got[2] == pytest.approx(-BIAS[2], abs=0.012)


def test_robust_optimum_is_at_least_as_good(vec):
    sug = comp.suggestions(vec, LIMIT, NOM)
    opt = sug["optimum"]
    assert set(sug["settings"]) == {"No compensation", "Median-based", "Rigid fit (mean)", "Robust optimum"}
    fpy = {m: comp.evaluate(vec, s, LIMIT, NOM)["fpy"] for m, s in sug["settings"].items()}
    assert fpy["Median-based"] > fpy["No compensation"] + 5
    assert fpy["Robust optimum"] >= max(fpy["Median-based"], fpy["Rigid fit (mean)"]) - 1.0
    assert opt.exact["fpy"] == pytest.approx(fpy["Robust optimum"])
    assert abs(opt.dx + BIAS[0]) < 0.4 and abs(opt.dy + BIAS[1]) < 0.4 and abs(opt.yaw + BIAS[2]) < 0.06


def test_unbiased_process_needs_no_move():
    v = biased_modules(n=300, bias=(0.0, 0.0, 0.0), sd_xy=0.3, sd_yaw=0.01, seed=8)
    opt = comp.robust_optimum(v, LIMIT, NOM)
    base = comp.evaluate(v, (0.0, 0.0, 0.0), LIMIT, NOM)["fpy"]
    assert base == pytest.approx(100.0)
    assert opt.exact["fpy"] == pytest.approx(100.0)
    assert abs(opt.dx) < 0.05 and abs(opt.dy) < 0.05 and abs(opt.yaw) < 0.01   # smallest move wins ties


def test_exact_evaluation_is_consistent(vec):
    setting = (-1.2, 0.85, -0.1)
    ev = comp.evaluate(vec, setting, LIMIT, NOM)
    sim = comp.simulate_table(vec, setting, LIMIT, NOM)
    assert ev["pass"] == int((sim["Status_Sim"] == "PASS").sum())
    yaws = [-0.2, -0.1, 0.0]
    curve = comp.fpy_vs_yaw(vec, setting[0], setting[1], yaws, LIMIT, NOM)
    for y, f in zip(yaws, curve):
        assert f == pytest.approx(comp.evaluate(vec, (setting[0], setting[1], y), LIMIT, NOM)["fpy"])
    land = comp.landscape(vec, -0.1, LIMIT, NOM, half=0.2, step=0.05, center=(-1.2, 0.85))
    i = int(np.argmin(np.abs(land["x"] + 1.2)))
    j = int(np.argmin(np.abs(land["y"] - 0.85)))
    assert land["fpy"][j, i] == pytest.approx(ev["fpy"])


def test_backtest_and_readiness(vec):
    bt = comp.backtest(vec, LIMIT, NOM, window_weeks=3, test_weeks=4)
    assert len(bt) == 4 and (bt["Trained on"] > 0).all()
    assert bt["WeekKey"].is_monotonic_increasing and bt["WeekKey"].iloc[0] > vec["WeekKey"].min()
    s = comp.backtest_summary(bt)
    assert s["best"] != "No compensation" and s[s["best"]] > s["No compensation"] + 5
    checks = {c["check"]: c["ok"] for c in comp.readiness(vec, s)}
    assert all(checks.values()), checks
    checks = {c["check"]: c["ok"] for c in comp.readiness(vec, s, recent_change=True)}
    assert not checks["No recent process shift"]


def test_backtest_fits_short_data():
    assert comp.auto_test_weeks(15, 6) == 8      # enough data: a full window and 8 test weeks
    assert comp.auto_test_weeks(7, 6) == 2       # 7 weeks: test on 2, learn from the 5 before
    assert comp.auto_test_weeks(7, 3) == 4       # a shorter window leaves more weeks to test on
    assert comp.auto_test_weeks(2, 6) == 1 and comp.auto_test_weeks(1, 6) == 0


def test_backtest_on_seven_weeks(vec):
    weeks = sorted(vec["WeekKey"].unique())
    assert len(weeks) >= 7
    v7 = vec[vec["WeekKey"].isin(weeks[-7:])]
    bt = comp.backtest(v7, LIMIT, NOM, window_weeks=6, mode="fixed")
    s = comp.backtest_summary(bt, "fixed")
    assert s["weeks"] == 2 and s["train_weeks"] == 5
    assert "(5 weeks)" in comp.backtest_text(s) and "(2 weeks," in comp.backtest_text(s)
    bt6 = comp.backtest(v7, LIMIT, NOM, window_weeks=6, test_weeks=6, mode="fixed")
    assert len(bt6) == 6 and (bt6["Train n weeks"] == 1).all()


def test_set_once_backtest_never_updates_the_setting(vec):
    fixed = comp.backtest(vec, LIMIT, NOM, window_weeks=3, test_weeks=4, mode="fixed")
    rolling = comp.backtest(vec, LIMIT, NOM, window_weeks=3, test_weeks=4, mode="rolling")
    assert list(fixed["WeekKey"]) == list(rolling["WeekKey"])                 # same test weeks
    # set once: one setting, learnt from the 3 weeks before the first test week, for every test week
    for m in comp.METHODS:
        assert fixed[[f"{m} X", f"{m} Y", f"{m} Yaw"]].nunique().max() == 1
    weeks = sorted(vec["WeekKey"].unique())
    first = weeks.index(fixed["WeekKey"].iloc[0])
    train = vec[vec["WeekKey"].isin(weeks[first - 3:first])]
    s = comp.suggestions(train, LIMIT, NOM, fast=True)["settings"]["Median-based"]
    assert tuple(fixed[[f"Median-based {k}" for k in ("X", "Y", "Yaw")]].iloc[0]) == pytest.approx(s)
    for _, row in fixed.iterrows():
        test = vec[vec["WeekKey"] == row["WeekKey"]]
        assert row["Median-based"] == comp.evaluate(test, s, LIMIT, NOM)["pass"]
    # recalculated: the setting follows the weeks
    assert rolling[["Median-based X"]].nunique().iloc[0] > 1
    summ = comp.backtest_summary(fixed, "fixed")
    assert summ["mode"] == "fixed" and summ["settings"]["Median-based"] == pytest.approx(s)
    assert comp.backtest_text(summ).startswith("set once from ")
    assert comp.backtest_text(comp.backtest_summary(rolling, "rolling")).startswith("recalculated every week")


def test_bootstrap_gain(vec):
    g = comp.bootstrap_gain(vec, comp.median_suggestion(vec), LIMIT, NOM, B=300)
    assert g["lo"] <= g["gain"] <= g["hi"] and g["gain"] > 5 and g["p_positive"] > 0.99
    assert g["fixed"] > g["broken"]


def test_equivalent_offsets_table():
    tab = comp.equivalent_offsets_table({"Type S": (0.5, -0.2, 0.05)}, NOM)
    assert len(tab) == 4
