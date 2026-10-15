"""Drawing rules that matter to the reader: what is drawn, how it is labelled, that it fits."""

import re

import numpy as np
import pandas as pd

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
