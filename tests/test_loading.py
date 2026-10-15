"""Reading real-world exports: headers, CSV dialects, times, duplicates, IDs."""

import json

import numpy as np
import pandas as pd
import pytest

from helpers import make_export, to_csv, to_xlsx
from rickhard import config, loading
from rickhard.config import LoadOptions
from rickhard.constants import DEV_COLS

OPTS = LoadOptions(run_gap_min=0, zero_as_missing=False)


def _runs(files, opts=OPTS):
    return loading.build_dataset(files, opts)


def _key(runs: pd.DataFrame) -> pd.DataFrame:
    return runs.sort_values(["PartID", "RunNum"]).reset_index(drop=True)[["PartID", "RunNum", "Date", *DEV_COLS]]


@pytest.fixture(scope="module")
def small():
    return make_export(n_modules=60, seed=21)


@pytest.fixture(scope="module")
def baseline(small):
    return _key(_runs([("a.xlsx", to_xlsx(small))]).runs)


@pytest.mark.parametrize("title_rows", [0, 1, 2, 5])
def test_header_row_is_found(small, baseline, title_rows):
    ds = _runs([("a.xlsx", to_xlsx(small, title_rows=title_rows))])
    pd.testing.assert_frame_equal(_key(ds.runs), baseline)
    assert ds.files[0]["header_row"] == title_rows + 1


@pytest.mark.parametrize("sep,decimal", [(",", "."), (";", ","), ("\t", ".")])
def test_csv_dialects(small, baseline, sep, decimal):
    ds = _runs([("a.csv", to_csv(small, sep=sep, decimal=decimal))])
    assert ds.files[0]["delimiter"] == sep
    pd.testing.assert_frame_equal(_key(ds.runs), baseline, check_dtype=False)


def test_export_time_format_with_am_and_pm():
    raw = pd.Series(["Aug 19, 2026 7:16am", "Aug 19, 2026 7:16pm", "Aug 19, 2026 12:05am", "Aug 9, 2026 11:59pm"])
    parsed, info = loading.parse_timestamps(raw)
    assert info["format"] == "%b %d, %Y %I:%M%p"
    assert list(parsed) == [pd.Timestamp("2026-08-19 07:16"), pd.Timestamp("2026-08-19 19:16"),
                            pd.Timestamp("2026-08-19 00:05"), pd.Timestamp("2026-08-09 23:59")]


@pytest.mark.parametrize("raw,expected", [
    ("2026-08-19 19:16:05", "2026-08-19 19:16:05"),
    ("19.08.2026 19:16:05", "2026-08-19 19:16:05"),
    ("08/19/2026 7:16:05 PM", "2026-08-19 19:16:05"),
    ("2026-08-19T19:16:05", "2026-08-19 19:16:05"),
])
def test_other_time_formats(raw, expected):
    parsed, _ = loading.parse_timestamps(pd.Series([raw, raw]))
    assert parsed.iloc[0] == pd.Timestamp(expected)


def test_excel_serial_times():
    parsed, info = loading.parse_timestamps(pd.Series([46253.5, 46253.75]))
    assert info["excel_serial"]
    assert parsed.iloc[0] == pd.Timestamp("2026-08-19 12:00")


def test_daylight_saving_never_drops_a_reading():
    naive = pd.Series(pd.to_datetime(["2026-10-25 02:30:00", "2026-03-29 02:30:00", "2026-07-01 12:00:00"]))
    out, n_amb, n_non = loading.convert_timezone(naive, "Europe/Berlin", "America/Mexico_City")
    assert (n_amb, n_non) == (1, 1)
    assert out.notna().all()
    assert out.iloc[0] == pd.Timestamp("2026-10-24 18:30")   # ambiguous hour read as summer time
    assert out.iloc[1] == pd.Timestamp("2026-03-28 19:00")   # skipped hour shifted forward
    assert out.iloc[2] == pd.Timestamp("2026-07-01 04:00")


def test_same_file_twice_and_overlapping_exports(small, baseline):
    data = to_xlsx(small)
    ds = _runs([("a.xlsx", data), ("b.xlsx", data)])
    assert ds.health["duplicate_rows_removed"] == len(small)
    pd.testing.assert_frame_equal(_key(ds.runs), baseline)
    n = len(small)
    first, second = small.iloc[: int(n * 0.6)], small.iloc[int(n * 0.4):]
    ds2 = _runs([("new.xlsx", to_xlsx(first)), ("old.xlsx", to_xlsx(second))])
    pd.testing.assert_frame_equal(_key(ds2.runs), baseline)


def test_minute_resolution_newest_first_keeps_runs_apart():
    """Exports carry minutes only and list the newest row first: corners of a
    module measured twice in the same minute must still form two runs."""
    feats = ["72_L0324_aa", "72_R0301_aa", "72_L0324_da", "72_R0302_da"]
    rows = [["Aug 19, 2026 7:10pm", "P0", f, 0.0, -3, 3, 0.0, -3, 3] for f in feats]  # an older module
    for k, feat in enumerate(feats * 2):
        rows.append(["Aug 19, 2026 7:16pm", "P1", feat, 0.1 * k, -3, 3, -0.1 * k, -3, 3])
    df = pd.DataFrame(rows, columns=["Time", "Part ID", "Featurename", "X deviation", "X lower limit",
                                     "X upper limit", "Y deviation", "Y lower limit", "Y upper limit"])
    ds = _runs([("a.xlsx", to_xlsx(df.iloc[::-1]))])
    assert ds.files[0]["newest_first"]
    runs = ds.runs[ds.runs["PartID"] == "P1"].sort_values("RunNum")
    assert list(runs["RunNum"]) == [1, 2]
    assert runs.iloc[0]["FL_X"] == pytest.approx(0.0) and runs.iloc[1]["FL_X"] == pytest.approx(0.4)


def test_lookalike_part_ids(small):
    df = small.copy()
    pid = df["Part ID"].iloc[-1]          # the oldest module (export is newest first)
    twin = pid.replace("0", "O", 1)
    second_part = df["Part ID"] == df["Part ID"].iloc[-5]
    df.loc[second_part, "Part ID"] = pid  # pretend it is the same module measured again ...
    df.loc[df.index[-1], "Part ID"] = twin  # ... with one row scanned with the letter O
    ds = _runs([("a.xlsx", to_xlsx(df))])
    assert any(twin in g and pid in g for g in ds.health["similar_part_ids"])
    merged = _runs([("a.xlsx", to_xlsx(df))], LoadOptions(run_gap_min=0, zero_as_missing=False,
                                                         merge_lookalikes=True))
    assert merged.health["lookalike_ids_merged"] == 1
    assert twin not in set(merged.runs["PartID"])


def test_zero_reading_as_missing(small):
    df = small.copy()
    i = df.index[-1]
    df.loc[i, ["X deviation", "Y deviation"]] = 0.0
    pid = df.loc[i, "Part ID"]
    keep = loading.build_dataset([("a.xlsx", to_xlsx(df))], LoadOptions(run_gap_min=0, zero_as_missing=False))
    drop = loading.build_dataset([("a.xlsx", to_xlsx(df))], LoadOptions(run_gap_min=0, zero_as_missing=True))
    assert drop.health["zero_readings"] == 1
    r_keep = keep.runs[(keep.runs["PartID"] == pid) & (keep.runs["RunNum"] == 1)].iloc[0]
    r_drop = drop.runs[(drop.runs["PartID"] == pid) & (drop.runs["RunNum"] == 1)].iloc[0]
    assert r_keep[list(DEV_COLS)].notna().sum() > r_drop[list(DEV_COLS)].notna().sum()


def test_unusual_column_names_need_a_mapping(small):
    df = small.rename(columns={"Part ID": "Module", "Featurename": "Point"})
    with pytest.raises(loading.DataFormatError) as e:
        loading.build_dataset([("a.xlsx", to_xlsx(df))])
    assert "Part ID" in str(e.value)
    info = loading.inspect_file(to_xlsx(df), "a.xlsx")
    assert not info["ok"]
    opts = LoadOptions(run_gap_min=0, zero_as_missing=False,
                       column_overrides=(("part", "Module"), ("feature", "Point")))
    ds = loading.build_dataset([("a.xlsx", to_xlsx(df))], opts)
    assert len(ds.runs) == len(_runs([("a.xlsx", to_xlsx(small))]).runs)


def test_run_gap_splits_a_late_second_measurement():
    rows = [["2026-08-19 10:00:00", "P1", "72_L0324_aa", 0.1], ["2026-08-19 10:00:05", "P1", "72_R0301_aa", 0.2],
            ["2026-08-19 12:00:00", "P1", "72_L0324_da", 0.3], ["2026-08-19 12:00:05", "P1", "72_R0302_da", 0.4]]
    df = pd.DataFrame([[*r, -3, 3, r[3], -3, 3] for r in rows],
                      columns=["Time", "Part ID", "Featurename", "X deviation", "X lower limit", "X upper limit",
                               "Y deviation", "Y lower limit", "Y upper limit"])
    data = to_xlsx(df)
    assert len(loading.build_dataset([("a.xlsx", data)], LoadOptions(run_gap_min=0)).runs) == 1
    assert len(loading.build_dataset([("a.xlsx", data)], LoadOptions(run_gap_min=30)).runs) == 2


def test_quality_report_counts_bad_rows(small):
    df = small.astype({"Time": object, "X deviation": object}).copy()
    df.iloc[0, 0] = "not a time"
    df.iloc[1, df.columns.get_loc("X deviation")] = "ERR"
    ds = _runs([("a.xlsx", to_xlsx(df))])
    assert ds.health["rows_without_valid_time"] == 1
    assert ds.health["non_numeric_deviation_values"] == 1
    assert ds.health["examples_bad_time"] == ["not a time"]


def test_weeks_across_the_year_boundary():
    df = pd.DataFrame({"Date": pd.to_datetime(["2026-12-30", "2027-01-05"])})
    loading.add_week_columns(df)
    assert list(df["WeekKey"]) == [202653, 202701]
    assert list(df["CalendarWeek"]) == ["2026-CW53", "2027-CW01"]
    assert loading.week_start(202701) == pd.Timestamp("2027-01-04")


def test_project_file_round_trip():
    load = LoadOptions(source_tz="UTC", run_gap_min=12.5, feature_map=(("abc", "FL"),),
                       column_overrides=(("part", "Module"),))
    settings = config.Settings(rules=config.Rules(spec_limit=2.5, fpy_target=93.0, z_in_status=True),
                               shifts=config.ShiftPlan(starts=(7, 19), names=("Day", "Night")))
    events = [{"Date": "2026-08-04 10:00", "What": "End stop moved", "Scope": "Type S"}]
    text = config.project_to_json(load, settings, events)
    json.loads(text)
    load2, settings2, events2 = config.project_from_json(text)
    assert load2 == load
    assert settings2.rules == settings.rules
    assert settings2.shifts == settings.shifts
    assert settings2.nominals == settings.nominals
    assert events2 == events
    with pytest.raises(ValueError):
        config.project_from_json('{"something": 1}')


def test_demo_story_is_read(demo_ds):
    h = demo_ds.health
    assert demo_ds.has_z
    assert h["duplicate_rows_removed"] > 0
    assert h["similar_part_ids"]
    assert h["zero_readings"] > 0
    assert set(demo_ds.runs["BatteryType"]) == {"Type S", "Type M"}
    assert np.isfinite(demo_ds.runs[list(DEV_COLS)].to_numpy(float)).mean() > 0.99
