"""Lap-level features from raw car telemetry.

Telemetry is sampled irregularly, so averages are weighted by the time each
sample is held (until the next sample) instead of naive row means.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FULL_THROTTLE_PCT = 98.0
TELEMETRY_COLUMNS = ("time_s", "speed", "throttle", "brake", "rpm", "gear", "drs")


def aggregate_arrays(time_s, speed, throttle, brake, rpm, gear, drs) -> dict[str, float]:
    t = np.asarray(time_s, dtype=float)
    if len(t) < 2:
        raise ValueError("need at least two telemetry samples")
    dt = np.diff(t)                      # sample i is held for dt[i]; last sample has no duration
    if (dt <= 0).any():
        raise ValueError("telemetry timestamps must be strictly increasing")
    total = dt.sum()
    speed, throttle, rpm = (np.asarray(a, dtype=float) for a in (speed, throttle, rpm))
    brake, drs, gear = np.asarray(brake, dtype=bool), np.asarray(drs, dtype=bool), np.asarray(gear)
    return {
        "tel_avg_speed": float((speed[:-1] * dt).sum() / total),
        "tel_max_speed": float(speed.max()),
        "tel_full_throttle_frac": float(dt[throttle[:-1] >= FULL_THROTTLE_PCT].sum() / total),
        "tel_brake_time_s": float(dt[brake[:-1]].sum()),
        "tel_gear_changes": int((np.diff(gear) != 0).sum()),
        "tel_mean_rpm": float((rpm[:-1] * dt).sum() / total),
        "tel_drs_frac": float(dt[drs[:-1]].sum() / total),
    }


def clean_telemetry(tel: pd.DataFrame) -> pd.DataFrame:
    """Sort by time, drop unusable/duplicate timestamps, forward-fill channel gaps."""
    missing = sorted(set(TELEMETRY_COLUMNS) - set(tel.columns))
    if missing:
        raise ValueError(f"telemetry missing columns: {missing}")
    out = tel.dropna(subset=["time_s"]).sort_values("time_s").drop_duplicates("time_s")
    channels = [c for c in TELEMETRY_COLUMNS if c != "time_s"]
    out[channels] = out[channels].ffill().bfill()
    return out.reset_index(drop=True)


def aggregate_lap(tel: pd.DataFrame) -> dict[str, float]:
    tel = clean_telemetry(tel)
    return aggregate_arrays(*(tel[c].to_numpy() for c in TELEMETRY_COLUMNS))
