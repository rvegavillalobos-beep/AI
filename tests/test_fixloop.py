"""The fix loop: set-ups, verdicts, loss budget, bridge, proof — and the brief built from them."""

import re

import numpy as np
import pandas as pd
import pytest

from rickhard import analysis, brief, demo, events, fixloop, metrics
from rickhard.charts import fix_svg, theme
from rickhard.config import Nominals, Scope, Settings

NOM = Nominals()


@pytest.fixture(scope="module")
def loop(demo_analysis):
    return fixloop.run(demo_analysis.first, demo.demo_events(), 99.0, 3.0, NOM)


# ---------------------------------------------------------------------------
# Action log
# ---------------------------------------------------------------------------

def test_old_logs_still_load():
    """A project file saved before Status / Target / Owner existed."""
    old = [{"Date": "2026-08-04T10:30:00", "Scope": "Type S", "Category": "Adjustment", "Title": "End-stop",
            "Notes": "", "dX": -0.9, "dY": 0.0, "dYaw": 0.0}]
    e = events.normalize_events(old)
    assert list(e.columns) == events.EVENT_COLUMNS
    assert e.loc[0, "Status"] == "Done" and e.loc[0, "Target"] == events.TARGET_GENERAL and e.loc[0, "Owner"] == ""
    again = events.normalize_events(events.events_to_records(e))
    pd.testing.assert_frame_equal(e, again)


def test_planned_actions_change_nothing(demo_analysis):
    ev = demo.demo_events()
    assert len(events.planned(events.normalize_events(ev))) == 2
    with_plan = fixloop.setups(demo_analysis.first, ev)
    without = fixloop.setups(demo_analysis.first, [e for e in ev if e["Status"] != "Planned"])
    pd.testing.assert_frame_equal(with_plan, without)


# ---------------------------------------------------------------------------
# Set-ups
# ---------------------------------------------------------------------------

def test_setups_partition_the_modules(demo_analysis, loop):
    f, su = demo_analysis.first, loop["setups"]
    assert len(su) == 4                                         # three changes made → four set-ups
    assert int(su["N"].sum()) == len(f)
    assert int(su["Passed"].sum()) == int((f["Status"] == "PASS").sum())
    assert list(su["Current"]) == [False, False, False, True]
    assert su.iloc[0]["StartedBy"] == "Start of the data" and np.isnan(su.iloc[0]["EventNo"])
    assert list(su["EventNo"].iloc[1:]) == [1.0, 2.0, 3.0]
    parts = [fixloop.setup_modules(f, row) for _, row in su.iterrows()]
    assert [len(p) for p in parts] == list(su["N"])
    assert len(set().union(*[set(p.index) for p in parts])) == len(f)
    assert su.iloc[-1]["FPY"] > su.iloc[0]["FPY"] + 5          # the demo line got better
    assert (su["CI_lo"] <= su["FPY"]).all() and (su["FPY"] <= su["CI_hi"]).all()


def test_no_log_is_one_setup(demo_analysis):
    su = fixloop.setups(demo_analysis.first, [])
    assert len(su) == 1 and su.iloc[0]["FPY"] == pytest.approx(demo_analysis.fpy)
    out = fixloop.run(demo_analysis.first, [], 99.0, 3.0, NOM)
    assert out["ledger"].empty and out["score"]["done"] == 0
    assert "No line change is logged yet" in out["story"]["done"]


def test_change_outside_the_data(demo_analysis):
    f = demo_analysis.first
    before = {"Date": "2026-01-05T08:00:00", "Scope": "All", "Category": "Tooling", "Title": "New gripper"}
    after = {"Date": (f["Date"].max() + pd.Timedelta(days=3)).isoformat(), "Scope": "All", "Category": "Adjustment",
             "Title": "Compensation applied"}
    su = fixloop.setups(f, [before, after])
    assert len(su) == 2
    assert su.iloc[0]["StartedBy"] == "New gripper" and su.iloc[0]["N"] == len(f)
    assert su.iloc[1]["N"] == 0 and bool(su.iloc[1]["Current"])
    led = fixloop.ledger(f, [before, after]).set_index("Title")
    assert led.loc["New gripper", "Verdict"] == "No baseline"
    assert led.loc["Compensation applied", "Verdict"] == "Verifying"
    out = fixloop.run(f, [before, after], 99.0, 3.0, NOM)
    assert int(out["setup"]["Setup"]) == 1                       # the empty set-up is not the one that is judged
    assert "0 modules since, too few to judge" in out["story"]["status"]
    assert out["story"]["which"] == "latest judged set-up" and "latest judged set-up" in out["story"]["headline"]
    assert fix_svg.setup_timeline_svg(su, None, fixloop.changes([before, after], ("Type S", "Type M")), 99.0)


def test_scope_keeps_the_other_type_out(demo_analysis):
    """A change to Type S alone is judged on Type S modules, and does not split Type M's set-ups."""
    f = demo_analysis.first
    w = fixloop.windows(f, demo.demo_events(), 1)
    assert set(w["before"]["BatteryType"]) == {"Type S"} == set(w["after"]["BatteryType"])
    only_m = f[f["BatteryType"] == "Type M"]
    assert len(fixloop.setups(only_m, demo.demo_events())) == 3  # the Type S end-stop is no boundary here


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def test_demo_verdicts(loop):
    led = loop["ledger"].set_index("No")
    assert list(led["Verdict"]) == ["Worked", "No effect", "Worked", "Planned", "Planned"]
    stop = led.loc[1]                                            # end-stop moved −0.9 mm in X, aimed at Shift X
    assert stop["FPY_delta"] > 5 and stop["FPY_p"] < 0.001
    assert stop["Metric"] == "Shift X level" and abs(stop["Metric_after"]) < abs(stop["Metric_before"]) - 0.5
    assert stop["Loss_after"] < stop["Loss_before"]
    assert "planned -0.90" in stop["Delivered"]
    rail = led.loc[3]                                            # guide rail realigned, aimed at yaw
    assert rail["Metric"] == "Yaw level" and abs(rail["Metric_after"]) < abs(rail["Metric_before"])
    assert loop["score"] == {"done": 3, "planned": 2, "judged": 3, "worked": 2, "no_effect": 1, "backfired": 0,
                             "verifying": 0, "superseded": 0, "no_baseline": 0}


def test_windows_never_cross_another_change(demo_analysis):
    f, ev = demo_analysis.first, demo.demo_events()
    dates = events.normalize_events(ev)["Date"]
    w = fixloop.windows(f, ev, 2)                                # between the end-stop and the guide rail
    assert w["before"]["Date"].min() >= dates[0] and w["before"]["Date"].max() < dates[1]
    assert w["after"]["Date"].min() >= dates[1] and w["after"]["Date"].max() < dates[2]
    assert len(w["before"]) <= fixloop.WINDOW and len(w["after"]) <= fixloop.WINDOW
    small = fixloop.windows(f, ev, 2, window=50)
    assert len(small["before"]) == 50 == len(small["after"])
    assert small["before"]["Date"].max() == w["before"]["Date"].max()     # the modules nearest to the change


def test_superseded_and_backfired(demo_analysis):
    f = demo_analysis.first.sort_values("Date").reset_index(drop=True)
    t = f["Date"]
    a_, b_ = t.iloc[1000], t.iloc[1012]                          # two changes twelve modules apart
    ev = [{"Date": a_.isoformat(), "Scope": "All", "Title": "First"},
          {"Date": b_.isoformat(), "Scope": "All", "Title": "Second"}]
    led = fixloop.ledger(f, ev).set_index("Title")
    assert led.loc["First", "Verdict"] == "Superseded" and led.loc["First", "N_after"] < fixloop.MIN_N
    # a change after which everything fails
    bad = f.copy()
    bad.loc[bad["Date"] >= a_, "Status"] = "FAIL"
    v = fixloop.verify(bad, [ev[0]], 1)
    assert v["verdict"] == "Backfired" and v["level"] == "bad"


def test_target_hit_without_fpy_gain(demo_analysis):
    """The thing the action aimed at improved, the FPY did not (yet)."""
    f = demo_analysis.first.sort_values("Date").reset_index(drop=True).copy()
    when = f["Date"].iloc[len(f) // 2]
    f["Status"] = "PASS"                                         # same FPY on both sides
    f["Mechanism"] = f["MechanismDetail"] = ""
    after = f["Date"] >= when
    f.loc[~after, "Fit_Rot"] = f.loc[~after, "Fit_Rot"] + 0.2    # yaw was off before, fine after
    v = fixloop.verify(f, [{"Date": when.isoformat(), "Scope": "All", "Title": "Guides", "Target": "Yaw"}], 1)
    assert v["verdict"] == "Target hit" and v["check"]["better"]
    same = fixloop.verify(f, [{"Date": when.isoformat(), "Scope": "All", "Title": "Guides", "Target": "Shift X"}], 1)
    assert same["verdict"] in ("No effect", "Verifying")


def test_unexplained_shifts():
    cp = pd.DataFrame({"Variable": ["Type S Shift X", "Type M Yaw"],
                       "Date": pd.to_datetime(["2026-08-04 12:00", "2026-09-01 08:00"]), "Shift": [-0.9, 0.02]})
    ev = [{"Date": "2026-08-04T10:30:00", "Scope": "Type S", "Title": "End-stop"},
          {"Date": "2026-09-01T07:00:00", "Scope": "All", "Title": "Planned only", "Status": "Planned"}]
    left = fixloop.unexplained_shifts(cp, ev)
    assert list(left["Variable"]) == ["Type M Yaw"]              # a planned action explains nothing
    assert len(fixloop.unexplained_shifts(cp, [])) == 2
    assert fixloop.unexplained_shifts(cp.iloc[:0], ev).empty


# ---------------------------------------------------------------------------
# Proof of the target
# ---------------------------------------------------------------------------

def test_modules_to_prove():
    assert fixloop.modules_to_prove(99.0) == 299                 # the classic "rule of three" number
    assert fixloop.modules_to_prove(99.0, fails=1) == 473
    assert fixloop.modules_to_prove(95.0) == 59
    assert fixloop.modules_to_prove(100.0) is None
    n = fixloop.modules_to_prove(99.0, fails=2)
    assert fixloop.lower_bound(n - 2, n) >= 99.0 > fixloop.lower_bound(n - 3, n - 1)
    assert fixloop.modules_to_prove(99.0, fails=0, start=500) == 500


def test_bounds_and_streaks():
    assert fixloop.lower_bound(299, 299) == pytest.approx(99.0, abs=0.01)
    assert fixloop.upper_bound(299, 299) == 100.0 and fixloop.lower_bound(0, 10) == 0.0
    lo, hi = fixloop.lower_bound(90, 100), fixloop.upper_bound(90, 100)
    assert lo < 90 < hi
    d = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=8, freq="h"),
                      "Status": ["PASS", "PASS", "PASS", "FAIL", "PASS", "INCOMPLETE", "PASS", "PASS"]})
    assert fixloop.streaks(d) == (2, 3)
    assert fixloop.streaks(d.iloc[:0]) == (0, 0)


def test_proof_states(demo_analysis, loop):
    p = loop["proof"]
    assert p["state"] == "below" and p["lower"] < p["fpy"] < p["upper"] < 99.0
    assert p["zero_fail_n"] == 299 and p["streak_to_go"] == 299 - p["streak"] and p["days_to_go"] > 0
    ok = pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=400, freq="h"), "Status": "PASS"})
    q = fixloop.proof(ok, 99.0)
    assert q["state"] == "proven" and q["more_ok"] == 0 and q["streak"] == 400
    few = fixloop.proof(ok.iloc[:50], 99.0)
    assert few["state"] == "open" and few["more_ok"] == 299 - 50


# ---------------------------------------------------------------------------
# Loss budget and bridge
# ---------------------------------------------------------------------------

def test_loss_budget_adds_up(loop):
    mods, bud = loop["modules"], loop["budget"]
    nok = int((mods["Status"] != "PASS").sum())
    assert int(bud["Modules"].sum()) == nok
    assert bud["Loss"].sum() == pytest.approx(100 - loop["setup"]["FPY"])
    assert bud["Share"].sum() == pytest.approx(100.0)
    assert list(bud["Modules"]) == sorted(bud["Modules"], reverse=True)
    assert set(bud["Mechanism"]) <= set(fixloop.MECHANISM_FIX)
    assert fixloop.loss_budget(mods[mods["Status"] == "PASS"]).empty


def test_bridge(loop):
    br = loop["bridge"]
    assert list(br["Key"]) == ["today", "centre", "scatter", "shape", "measure"]
    assert br.iloc[0]["FPY"] == pytest.approx(loop["setup"]["FPY"]) and br.iloc[0]["Delta"] == 0
    assert (br["Delta"] >= 0).all() and br["FPY"].is_monotonic_increasing and br["FPY"].max() <= 100.0
    assert br["FPY"].iloc[-1] == pytest.approx(br.iloc[0]["FPY"] + br["Delta"].sum())
    assert br.set_index("Key").loc["centre", "Delta"] > 0.3     # the demo set-up is off-centre
    assert set(loop["comps"]) == {"Type S", "Type M"}
    assert fixloop.bridge(loop["modules"].iloc[:0], 3.0, NOM).empty


def test_another_setup_can_be_picked(demo_analysis, loop):
    first = fixloop.run(demo_analysis.first, demo.demo_events(), 99.0, 3.0, NOM, setup_no=1)
    assert int(first["setup"]["Setup"]) == 1 and int(loop["setup"]["Setup"]) == 4
    assert len(first["modules"]) == int(loop["setups"].iloc[0]["N"])
    assert first["budget"]["Loss"].sum() > loop["budget"]["Loss"].sum()


def test_filters_narrowed_to_one_week(demo_ds):
    wk = sorted(demo_ds.runs["WeekKey"].dropna().unique())[7]
    a = analysis.analyze(demo_ds.runs, Settings(), Scope(week_from=int(wk), week_to=int(wk)), 3, demo_ds.has_z)
    out = fixloop.run(a.first, demo.demo_events(), 99.0, 3.0, NOM)
    assert int(out["setups"]["N"].sum()) == len(a.first) and out["story"]["headline"]
    assert brief.build(a, out).startswith("<!doctype html")
    empty = fixloop.run(a.first.iloc[:0], demo.demo_events(), 99.0, 3.0, NOM)
    assert empty["setup"] is None and empty["story"]["headline"] == "No modules in the current filters."
    assert "No modules in the current filters" in brief.build(a, empty)


# ---------------------------------------------------------------------------
# Charts and brief
# ---------------------------------------------------------------------------

def test_svg_charts(demo_analysis, loop):
    ch = fixloop.changes(demo.demo_events(), demo_analysis.types)
    for dark in (False, True):
        theme.set_dark(dark)
        try:
            drawings = [fix_svg.setup_timeline_svg(loop["setups"], metrics.daily_fpy(demo_analysis.first), ch, 99.0),
                        fix_svg.loss_budget_svg(loop["budget"], 99.0), fix_svg.bridge_svg(loop["bridge"], 99.0)]
        finally:
            theme.set_dark(False)
        for d in drawings:
            assert d.startswith("<svg") and d.endswith("</svg>")
            assert "nan" not in d.lower() and "<marker" not in d
    timeline = drawings[0]
    for label in ("83.0%", "94.6%", "Target 99%"):
        assert label in timeline
    assert fix_svg.loss_budget_svg(loop["budget"].iloc[:0], 99.0) == ""
    assert fix_svg.bridge_svg(loop["bridge"].iloc[:0], 99.0) == ""


def test_brief(demo_analysis, loop):
    doc = brief.build(demo_analysis, loop, title="Line 3 <review>", author="R. Vega", note="CW45",
                      asks="4 h maintenance window\n- Supplier review of **Type M** squareness\n\n")
    assert doc.lstrip().lower().startswith("<!doctype html")
    assert "Line 3 &lt;review&gt;" in doc and "<review>" not in doc
    assert "R. Vega" in doc and "CW45" in doc
    assert "<script" not in doc and "http://" not in doc.replace("http://www.w3.org", "") and "https://" not in doc
    assert doc.count("data:image/svg+xml;base64,") == 3          # timeline, loss budget, bridge
    for section in ("Where we are", "What we did, and whether it worked", "What still costs yield",
                    "What it takes to reach 99%", "Next steps", "What we need"):
        assert section in doc, section
    assert doc.count("<li>4 h maintenance window</li>") == 1 and "<b>Type M</b>" in doc
    text = re.sub(r"<[^>]+>", " ", doc)
    assert not re.search(r"\bnan\b", text, flags=re.I)
    assert loop["story"]["headline"] in text.replace("&#x27;", "'")
    assert "Conveyor end-stop moved −0.9 mm in X" in text and "Re-teach the vision job" in text
    plain = brief.build(demo_analysis, loop)                    # no requests: the block is left out
    assert "What we need" not in plain and "Demo data" not in plain
    assert "Demo data · invented numbers" in brief.build(demo_analysis, loop, demo=True)
    assert brief.fragment(demo_analysis, loop).startswith("<div class='xb'>")


def test_story_at_target(demo_analysis):
    out = fixloop.run(demo_analysis.first, demo.demo_events(), 60.0, 3.0, NOM)
    assert out["story"]["level"] == "good" and "at the 60% target" in out["story"]["headline"]
    assert out["proof"]["state"] == "proven"
    assert brief.build(demo_analysis, out)
