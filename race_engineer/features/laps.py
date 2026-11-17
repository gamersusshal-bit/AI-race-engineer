"""Lap-level modelling features, with an explicit pre-/post-lap split.

A lap-time prediction is made *before* the lap is driven, so it may only use
information known at that point. Anything measured during or after the lap
(lap/sector times, that lap's telemetry aggregates, fuel actually burned) is
listed in ``POST_LAP_COLUMNS`` and must never be a model input.
"""
from __future__ import annotations

import pandas as pd

from race_engineer.config import COMPOUND_NAMES

CATEGORICAL_FEATURES = ["circuit", "driver", "compound", "drive_mode"]
NUMERIC_FEATURES = ["tyre_age", "fuel_kg", "lap_number", "lap_frac", "track_temp"]
DERIVED_FEATURES = ["tyre_age_sq"] + [f"age_{c.lower()}" for c in COMPOUND_NAMES]
PRE_LAP_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES + DERIVED_FEATURES

POST_LAP_COLUMNS = frozenset({
    "lap_time_s", "sector1_s", "sector2_s", "sector3_s", "fuel_used_kg",
    "tel_avg_speed", "tel_max_speed", "tel_full_throttle_frac", "tel_brake_time_s",
    "tel_gear_changes", "tel_mean_rpm", "tel_drs_frac",
})


def assert_no_leakage(features) -> None:
    leaked = sorted(set(features) & POST_LAP_COLUMNS)
    if leaked:
        raise ValueError(f"post-lap information used as a pre-lap feature: {leaked}")


def make_pre_lap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for lap-time prediction (works for observed laps and simulated plans)."""
    x = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    x["tyre_age_sq"] = x["tyre_age"] ** 2
    for c in COMPOUND_NAMES:
        x[f"age_{c.lower()}"] = x["tyre_age"].where(x["compound"] == c, 0.0)
    assert_no_leakage(x.columns)
    return x[PRE_LAP_FEATURES]


def clean_laps(df: pd.DataFrame) -> pd.DataFrame:
    """Representative racing laps: no pit in/out laps and no standing-start lap."""
    mask = ~df["is_pit_in_lap"] & ~df["is_pit_out_lap"] & (df["lap_number"] > 1)
    return df[mask]


def add_stint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stint trends for analysis/dashboard (uses observed lap times)."""
    out = df.sort_values(["session_id", "driver", "lap_number"]).copy()
    grp = out.groupby(["session_id", "driver", "stint"])["lap_time_s"]
    out["stint_delta_s"] = out["lap_time_s"] - grp.transform("min")
    out["lap_delta_prev_s"] = grp.diff()
    return out
