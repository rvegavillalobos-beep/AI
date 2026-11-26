"""Drawing rules that matter to the reader: what is drawn, how it is labelled, that it fits."""

import re

import numpy as np
import pandas as pd
import pytest

from rickhard.charts import geometry_view as gv
from rickhard.charts import svg
from rickhard.config import Nominals
from rickhard.constants import DEV_COLS

NOM = Nominals()


def _modules(n: int, part="FO88995230126B31966N0000002112961", status="PASS") -> pd.DataFrame:
    df = pd.DataFrame(np.random.default_rng(1).normal(0, 0.8, (n, 8)), columns=list(DEV_COLS))
    df.insert(0, "PartID", [f"{part}{i:02d}" for i in range(n)])
    df["RunNum"] = 1
    df["BatteryType"] = "Type S"
    df["Status"] = status
    df["_mod_key"] = df["PartID"] + " | Run 1"
    return df


def test_outlines_skip_modules_with_a_missing_corner_and_show_full_part_ids():
    df = _modules(6)
    df.loc[2, ["FL_X", "FL_Y", "FR_X", "FR_Y"]] = np.nan       # only two corners measured
    df.loc[3, "Status"] = "FAIL"
    fig = gv.outlines(df, NOM, 3.0, 20, types=["Type S"], status_filter=("PASS", "FAIL"))
    names = [t.name for t in fig.data if t.showlegend is not False and t.name and "·" in t.name]
    assert len(names) == 5                                        # the incomplete module is not drawn
    assert all(re.fullmatch(r"FO\d+B\d+N\d+ · R1", n) for n in names)  # the whole Part ID, never shortened
    assert df.loc[2, "PartID"] + " · R1" not in names
    # every drawn module is a closed four-corner outline
    assert all(len(t.x) == 5 for t in fig.data if t.name in names)


def test_auto_magnify_keeps_large_corrections_on_the_page():
    assert svg.auto_magnify(0.6, 1117) == 200                    # small correction: strong magnification
    assert svg.auto_magnify(3.0, 1117) == 50                     # ~3 mm at the corners: much less
    assert svg.auto_magnify(0.0, 1117) == 150
    drawing = svg.work_instruction_svg("Type S", 1.16, 0.99, -0.139, NOM, magnify=300)
    nums = [float(v) for v in re.findall(r'points="([^"]+)"', drawing)[1].replace(",", " ").split()]
    xs, ys = nums[0::2], nums[1::2]
    assert min(xs) >= 0 and max(xs) <= 640 and min(ys) >= 0 and max(ys) <= 520   # fits its frame at ×300


def test_diagonal_trend_is_one_point_per_module_in_order():
    from rickhard.charts import shape as cs
    from rickhard.geometry import dimension_deviations

    df = _modules(30)
    df["Date"] = pd.date_range("2026-10-01", periods=30, freq="h")
    df["CalendarWeek"] = "CW40"
    dims = dimension_deviations(df, NOM)
    ev = pd.DataFrame({"Date": [pd.Timestamp("2026-10-01 10:30")], "Title": ["Guide realigned"]})
    fig = cs.diagonal_trend(df, dims, 1.5, 7, ev)
    pts = {t.name: t for t in fig.data if t.mode == "markers"}
    assert list(pts["Diag FL–RR"].x) == list(range(1, 31))
    assert list(pts["Diag FR–RL"].y) == pytest.approx(dims["Diag FR–RL"].to_numpy())
    assert list(pts["Difference (square)"].y) == pytest.approx(dims["Diagonal difference"].to_numpy())
    assert any(s.x0 == 11.5 for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1)  # after module 11


def test_fpy_trend_per_day_sits_on_a_date_axis_with_interventions_on_their_day(demo_analysis):
    from rickhard.charts import overview as co

    d = demo_analysis.daily
    day = d["Day"].iloc[5]
    ev = pd.DataFrame({"Date": [day + pd.Timedelta(hours=10), d["Day"].iloc[-1] + pd.Timedelta(days=30)],
                       "Title": ["Shim the end-stop", "Outside the data"]})
    fig = co.daily_fpy(d, 99.0, 3, events=ev)
    line = next(t for t in fig.data if t.name == "Daily FPY")
    assert list(line.x) == list(d["Day"].dt.strftime("%Y-%m-%d"))
    assert fig.layout.xaxis.type == fig.layout.xaxis2.type == "date"
    marks = {s.x0 for s in fig.layout.shapes if s.type == "line" and s.x0 == s.x1}
    assert marks == {day.strftime("%Y-%m-%d")}                    # the intervention is on its day, the other dropped
    assert all(b.width for b in fig.data if b.type == "bar")      # a bar a day wide, even for a single day
    assert ("text" in line.mode) == (len(d) <= co.DAY_LABELS_MAX)  # many days: no % label on every point
    short = co.daily_fpy(d.head(5), 99.0, 3)
    assert "text" in next(t for t in short.data if t.name == "Daily FPY").mode
    # the weekly view stays one point per calendar week on a category axis
    wk = co.weekly_fpy(demo_analysis.weekly, 99.0, 3)
    assert list(next(t for t in wk.data if t.name == "Weekly FPY").x) == list(demo_analysis.weekly["CalendarWeek"])
    assert wk.layout.xaxis2.type == "category"
