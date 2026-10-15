"""Statistics, SPC, patterns, re-measurement, what-if and the demo's process story."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from rickhard import demo, events, improvement, insights, metrics, patterns, remeasure, spc, stats
from rickhard.config import Nominals
from rickhard.constants import DEV_COLS

NOM = Nominals()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def test_wilson_interval():
    lo, hi = stats.wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(27.75, abs=0.01)
    lo, hi = stats.wilson(5, 10)
    assert 50 - lo == pytest.approx(hi - 50)
    assert np.isnan(stats.wilson(0, 0)[0])


def test_tests_match_scipy():
    assert stats.fisher_2x2(3, 7, 9, 1) == pytest.approx(sps.fisher_exact([[3, 7], [9, 1]])[1])
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 40), rng.normal(0.8, 2, 60)
    diff, p = stats.welch(a, b)
    assert diff == pytest.approx(b.mean() - a.mean())
    assert p == pytest.approx(sps.ttest_ind(a, b, equal_var=False).pvalue)
    ratio, p2 = stats.levene(a, b)
    assert ratio == pytest.approx(b.std(ddof=1) / a.std(ddof=1))
    assert p2 == pytest.approx(sps.levene(a, b, center="median").pvalue)


def test_capability_of_a_centred_normal_process():
    v = np.random.default_rng(2).normal(0, 1, 20000)
    c = metrics.capability_series(v, -3, 3)
    assert c["Pp"] == pytest.approx(1.0, abs=0.03)
    assert c["Ppk"] == pytest.approx(1.0, abs=0.03)
    assert c["Cp"] == pytest.approx(1.0, abs=0.03)        # independent data: within σ = overall σ
    assert c["PPM"] == pytest.approx(2700, rel=0.15)
    assert c["PpkLo"] < c["Ppk"] < c["PpkHi"]


# ---------------------------------------------------------------------------
# SPC
# ---------------------------------------------------------------------------

def test_imr_limits_and_rule_1():
    x = np.array([0.0, 1.0] * 20 + [9.0])
    ch = spc.imr(x)
    mrbar = np.abs(np.diff(x)).mean()
    assert ch.sigma[0] == pytest.approx(mrbar / 1.128)
    assert ch.ucl[0] == pytest.approx(x.mean() + 3 * mrbar / 1.128)
    assert ch.flags[1][-1] and not ch.flags[1][:-1].any()


def test_run_rules():
    x = np.r_[np.zeros(5), np.full(9, 0.5), np.zeros(5)]
    f = spc.nelson_rules(x, center=0.2, sigma=1.0)
    assert f[2][13] and not f[2][:13].any()                     # 9 in a row on one side
    trend = np.arange(8, dtype=float)
    assert spc.nelson_rules(trend, 3.5, 10.0)[3][6]             # 6 increasing points (5 increases)


def test_change_point_finds_a_step():
    rng = np.random.default_rng(5)
    y = np.r_[rng.normal(0, 1, 150), rng.normal(1.5, 1, 150)]
    cps = spc.change_points(y)
    assert len(cps) >= 1 and abs(cps[0] - 150) <= 6
    assert spc.change_points(rng.normal(0, 1, 300)) == []


def _first_runs(values, start="2026-07-01", per_day=20):
    n = len(values)
    dates = pd.Timestamp(start) + pd.to_timedelta(np.arange(n) * (24 / per_day), unit="h")
    return pd.DataFrame({"Date": dates, "BatteryType": "Type S", "Fit_Tx": values})


def test_alarm_board_is_quiet_on_a_stable_process_and_fires_on_a_recent_shift():
    rng = np.random.default_rng(9)
    stable = _first_runs(rng.normal(0, 0.3, 1200))
    board = spc.alarm_board(stable, {"Fit_Tx": "Shift X"})
    assert list(board["State"]) == ["In control"]
    y = rng.normal(0, 0.3, 1200)
    y[-200:] += 0.3                                              # +1σ in the last 10 days
    board = spc.alarm_board(_first_runs(y), {"Fit_Tx": "Shift X"})
    assert board.iloc[0]["State"] == "Signal"
    assert "level moved" in board.iloc[0]["Reason"]


# ---------------------------------------------------------------------------
# Patterns, re-measurement, what-if, interventions
# ---------------------------------------------------------------------------

def test_startup_flags():
    t = pd.to_datetime(["2026-07-01 06:00", "2026-07-01 06:05", "2026-07-01 06:10", "2026-07-01 06:15",
                        "2026-07-01 09:00", "2026-07-01 09:04", "2026-07-01 09:08"])
    f = patterns.startup_flags(pd.DataFrame({"Date": t}), gap_min=60, k=2)
    assert list(f) == [True, True, False, False, True, True, False]


def test_repeatability_recovers_the_measurement_noise():
    rng = np.random.default_rng(4)
    n, sigma_e = 400, 0.2
    truth = rng.normal(0, 1, (n, 8))
    rows = []
    for i in range(n):
        for run in (1, 2):
            d = truth[i] + rng.normal(0, sigma_e, 8)
            rows.append({"PartID": f"P{i}", "RunNum": run, "BatteryType": "Type S",
                         "Date": pd.Timestamp("2026-07-01") + pd.Timedelta(minutes=10 * i + 5 * run),
                         "Status": "PASS", "WorstAbs": 0.0, **dict(zip(DEV_COLS, d))})
    pairs = remeasure.run_pairs(pd.DataFrame(rows))
    rep = remeasure.repeatability(pairs, 3.0)
    assert len(rep) == 8
    assert rep["SigmaRepeat"].median() == pytest.approx(sigma_e, rel=0.12)
    assert rep["GRR"].median() == pytest.approx(6 * sigma_e / 6 * 100, rel=0.12)


def test_levers(demo_analysis):
    f = demo_analysis.first
    today = improvement.fpy_after(f, improvement.Levers(), None, 3.0, NOM)
    assert today == pytest.approx(demo_analysis.fpy)
    better = improvement.fpy_after(f, improvement.Levers(shift_reduction=0.5, yaw_reduction=0.5), None, 3.0, NOM)
    assert better > today
    wf = improvement.waterfall(f, improvement.Levers(yaw_reduction=0.5, shape_reduction=0.5), None, 3.0, NOM)
    assert wf.iloc[0]["FPY"] == pytest.approx(today)
    assert wf["Delta"].iloc[1:].sum() == pytest.approx(wf.iloc[-1]["FPY"] - today)
    r = improvement.required_reduction(f, today + 3, None, 3.0, NOM, which="placement")
    assert r is None or 0 < r <= 1


def test_intervention_before_after(demo_analysis):
    ev = demo.demo_events()[0]                                   # end stop moved −0.9 mm in X (Type S)
    out = events.before_after(demo_analysis.first, ev["Date"], ev["Scope"], 14)
    assert out["enough"]
    row = out["table"].set_index("Column").loc["Fit_Tx"]
    assert row["Change"] == pytest.approx(-0.9, abs=0.35)
    assert row["p (mean)"] < 1e-6


# ---------------------------------------------------------------------------
# The demo's process story is found
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bundle(demo_analysis, demo_ds):
    return insights.bundle(demo_analysis, demo_ds.health)


def test_demo_change_points(bundle):
    cp = bundle["extras"]["change_points"]
    sx = cp[cp["Variable"] == "Type S Shift X"]
    assert len(sx) and abs(sx.iloc[0]["Date"] - pd.Timestamp("2026-08-04 10:30")) < pd.Timedelta(days=1)
    assert sx.iloc[0]["Shift"] == pytest.approx(-0.9, abs=0.35)
    yaw = cp[cp["Variable"].str.endswith("Yaw")]
    near = (yaw["Date"] - pd.Timestamp("2026-09-10 15:00")).abs() < pd.Timedelta(days=2)
    assert near.any() and (yaw.loc[near, "Shift"] < 0).all()      # realignment brings yaw back


def test_demo_patterns(bundle):
    se = bundle["extras"]["shift_effect"]
    assert se["worst"] == "C" and se["axis"] == "Y" and se["p_mean"] < 1e-6
    su = bundle["extras"]["startup"]
    assert su["fpy_startup"] < su["fpy_steady"] and su["p_value"] < 0.01
    rm = bundle["extras"]["remeasure"]
    assert 15 < rm["max_grr"] < 50 and rm["nok_to_ok"] > 0


def test_demo_findings_and_actions(bundle):
    ins, acts = bundle["insights"], bundle["actions"]
    assert {i.level for i in ins} <= {"good", "info", "warn", "bad"}
    topics = {i.topic for i in ins}
    assert {"fpy", "mechanism", "msa", "startup", "shift", "change"} <= topics
    scores = [i.score for i in ins]
    assert scores == sorted(scores, reverse=True)
    impacts = [a.impact for a in acts if a.impact is not None]
    assert impacts == sorted(impacts, reverse=True) and all(i > 0 for i in impacts)
    assert all(a.page is None or isinstance(a.page, str) for a in acts)
