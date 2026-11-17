import numpy as np
import pandas as pd
import pytest

from race_engineer.data.fastf1_loader import fastf1_laps_to_canonical
from race_engineer.data.schema import SchemaError, validate_laps
from race_engineer.data.synthetic import generate_dataset


def test_validation_drops_and_reports_bad_rows(raw_laps):
    clean, report = validate_laps(raw_laps)
    assert report.dropped.get("missing lap time", 0) == raw_laps["lap_time_s"].isna().sum() > 0
    assert report.rows_out == len(clean) == report.rows_in - sum(report.dropped.values())
    assert clean["lap_time_s"].notna().all()
    assert not clean.duplicated(["session_id", "driver", "lap_number"]).any()


def test_validation_rejects_missing_columns_and_bad_values(laps):
    with pytest.raises(SchemaError, match="missing required columns"):
        validate_laps(laps.drop(columns=["compound"]))
    bad = laps.head(50).copy()
    bad.loc[bad.index[0], "compound"] = "HYPERSOFT"
    bad.loc[bad.index[1], "tyre_age"] = -1
    bad = pd.concat([bad, bad.iloc[[5]]])
    _, report = validate_laps(bad)
    assert report.dropped == {"unsupported compound": 1, "invalid tyre age": 1, "duplicate lap": 1}
    with pytest.raises(SchemaError, match="drive modes"):
        validate_laps(laps.head(5).assign(drive_mode="qualifying"))


def test_synthetic_races_are_complete_and_legal(laps, raw_laps):
    per_car = raw_laps.groupby(["session_id", "driver"])
    assert (per_car["lap_number"].count() == per_car["total_laps"].first()).all()
    assert (per_car["compound"].nunique() >= 2).all()            # two-compound rule
    assert (per_car["fuel_kg"].min() > 0).all()
    fresh = laps[laps["is_pit_out_lap"]]
    assert (fresh["tyre_age"] == 0).all()


def test_generation_is_reproducible():
    a = generate_dataset(seasons=(2024,), seed=11, with_telemetry=False)
    b = generate_dataset(seasons=(2024,), seed=11, with_telemetry=False)
    pd.testing.assert_frame_equal(a, b)


def test_fastf1_mapping_to_canonical_schema():
    td = lambda s: pd.Timedelta(seconds=float(s))  # noqa: E731
    ff1 = pd.DataFrame({
        "Driver": ["VER", "VER", "VER", "HAM"], "LapNumber": [1.0, 2.0, 3.0, 1.0],
        "LapTime": [td(95.1), td(91.2), pd.NaT, td(96.0)],
        "Sector1Time": [td(30)] * 4, "Sector2Time": [td(31)] * 4, "Sector3Time": [td(30.2)] * 4,
        "Stint": [1.0, 1.0, 2.0, 1.0], "Compound": ["SOFT", "SOFT", "HARD", "INTERMEDIATE"],
        "TyreLife": [1.0, 2.0, 1.0, 1.0],
        "PitInTime": [pd.NaT, td(200), pd.NaT, pd.NaT], "PitOutTime": [pd.NaT, pd.NaT, td(222), pd.NaT],
    })
    canon = fastf1_laps_to_canonical(ff1, "Sakhir", 2024, total_laps=57)
    assert list(canon["driver"]) == ["VER"] * 3                  # wet-compound lap removed
    assert list(canon["tyre_age"]) == [0, 1, 0]                  # FastF1 TyreLife starts at 1
    assert canon["fuel_is_estimated"].all() and canon["fuel_kg"].is_monotonic_decreasing
    assert list(canon["is_pit_in_lap"]) == [False, True, False]
    clean, report = validate_laps(canon)
    assert len(clean) == 2 and report.dropped == {"missing lap time": 1}
    assert np.isclose(clean["lap_time_s"].iloc[1], 91.2)
