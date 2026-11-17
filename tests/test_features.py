import numpy as np
import pandas as pd
import pytest

from race_engineer.data.telemetry_synth import synth_lap_telemetry
from race_engineer.features.laps import (
    POST_LAP_COLUMNS, PRE_LAP_FEATURES, assert_no_leakage, clean_laps, make_pre_lap_features,
)
from race_engineer.features.telemetry import aggregate_lap


def _tel(time_s, speed, throttle=None, brake=None, gear=None):
    n = len(time_s)
    return pd.DataFrame({
        "time_s": time_s, "speed": speed, "throttle": throttle if throttle is not None else [100.0] * n,
        "brake": brake if brake is not None else [False] * n, "rpm": [10000.0] * n,
        "gear": gear if gear is not None else [5] * n, "drs": [False] * n,
    })


def test_averages_are_time_weighted_not_row_weighted():
    # 100 km/h held for 9 s, then 300 km/h sampled densely for 1 s.
    tel = _tel([0, 9, 9.25, 9.5, 9.75, 10], [100, 300, 300, 300, 300, 300],
               throttle=[0, 100, 100, 100, 100, 100], brake=[True, False, False, False, False, False])
    f = aggregate_lap(tel)
    assert f["tel_avg_speed"] == pytest.approx(120.0)            # naive row mean would be 266.7
    assert f["tel_full_throttle_frac"] == pytest.approx(0.1)
    assert f["tel_brake_time_s"] == pytest.approx(9.0)
    assert f["tel_max_speed"] == 300


def test_gear_changes_and_messy_input():
    tel = _tel([0, 1, 2, 3, 4], [100, 120, 140, 120, 100], gear=[3, 4, 5, 4, 4])
    assert aggregate_lap(tel)["tel_gear_changes"] == 3
    messy = pd.concat([tel.iloc[::-1], tel.iloc[[2]]])            # unsorted + duplicate timestamp
    messy.loc[messy["time_s"] == 1, "speed"] = np.nan             # gap is forward-filled
    f = aggregate_lap(messy)
    assert f["tel_gear_changes"] == 3
    assert f["tel_avg_speed"] == pytest.approx((100 + 100 + 140 + 120) / 4)
    with pytest.raises(ValueError):
        aggregate_lap(tel.head(1))


def test_synthetic_telemetry_matches_lap_time_and_circuit_length():
    tel = synth_lap_telemetry("Monza", 82.0, "normal", seed=1)
    f = aggregate_lap(tel)
    assert tel["time_s"].iloc[-1] == pytest.approx(82.0)
    assert np.diff(tel["time_s"]).std() > 0.05                    # irregular sampling
    assert f["tel_avg_speed"] / 3.6 * 82.0 == pytest.approx(5793, rel=0.02)
    conserve = aggregate_lap(synth_lap_telemetry("Monza", 82.0, "conserve", seed=1))
    assert conserve["tel_full_throttle_frac"] < f["tel_full_throttle_frac"]


def test_pre_lap_features_exclude_everything_observed_during_the_lap(laps):
    x = make_pre_lap_features(laps)
    assert list(x.columns) == PRE_LAP_FEATURES
    assert not set(x.columns) & POST_LAP_COLUMNS
    with pytest.raises(ValueError, match="post-lap"):
        assert_no_leakage(PRE_LAP_FEATURES + ["tel_avg_speed"])
    # Features must not change when the outcome of the lap changes.
    shuffled = laps.assign(lap_time_s=laps["lap_time_s"].sample(frac=1, random_state=0).to_numpy(),
                           tel_avg_speed=0.0, fuel_used_kg=0.0)
    pd.testing.assert_frame_equal(x, make_pre_lap_features(shuffled))


def test_clean_laps_removes_pit_and_start_laps(laps):
    c = clean_laps(laps)
    assert not c["is_pit_in_lap"].any() and not c["is_pit_out_lap"].any() and c["lap_number"].min() == 2
