"""Canonical lap-table schema and validation.

Both the synthetic generator and the FastF1 loader produce this table, so every
downstream stage (features, models, simulation) is source-agnostic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from race_engineer.config import COMPOUND_NAMES, MODE_NAMES

REQUIRED_COLUMNS = {
    "session_id": "object", "season": "int", "circuit": "object", "driver": "object",
    "lap_number": "int", "total_laps": "int", "lap_time_s": "float", "stint": "int",
    "compound": "object", "tyre_age": "float", "fuel_kg": "float", "track_temp": "float",
    "drive_mode": "object", "is_pit_in_lap": "bool", "is_pit_out_lap": "bool",
}
OPTIONAL_COLUMNS = (
    "sector1_s", "sector2_s", "sector3_s", "fuel_used_kg", "fuel_is_estimated",
    "tel_avg_speed", "tel_max_speed", "tel_full_throttle_frac", "tel_brake_time_s",
    "tel_gear_changes", "tel_mean_rpm", "tel_drs_frac",
)
KEY = ["session_id", "driver", "lap_number"]


class SchemaError(ValueError):
    pass


@dataclass
class ValidationReport:
    rows_in: int = 0
    rows_out: int = 0
    dropped: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"rows_in": self.rows_in, "rows_out": self.rows_out, "dropped": self.dropped}


def validate_laps(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Check structure (raises ``SchemaError``) and drop unusable rows (reported)."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    report = ValidationReport(rows_in=len(df))
    out = df.copy()

    for col, kind in REQUIRED_COLUMNS.items():
        if kind in ("int", "float"):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    def drop(mask: pd.Series, reason: str) -> None:
        nonlocal out
        if mask.any():
            report.dropped[reason] = int(mask.sum())
            out = out[~mask]

    drop(out[["session_id", "driver", "lap_number", "compound"]].isna().any(axis=1), "missing key fields")
    drop(out["lap_time_s"].isna(), "missing lap time")
    drop(out["lap_time_s"] <= 0, "non-positive lap time")
    drop(~out["compound"].isin(COMPOUND_NAMES), "unsupported compound")
    drop(out["tyre_age"].isna() | (out["tyre_age"] < 0), "invalid tyre age")
    drop(out["fuel_kg"].isna() | out["track_temp"].isna(), "missing fuel or track temperature")
    drop(out.duplicated(KEY, keep="first"), "duplicate lap")

    bad_modes = set(out["drive_mode"].unique()) - set(MODE_NAMES)
    if bad_modes:
        raise SchemaError(f"unknown drive modes: {sorted(bad_modes)}")
    if (out["lap_number"] > out["total_laps"]).any() or (out["lap_number"] < 1).any():
        raise SchemaError("lap_number outside 1..total_laps")

    for col, kind in REQUIRED_COLUMNS.items():
        if kind == "int":
            out[col] = out[col].astype(int)
        elif kind == "bool":
            out[col] = out[col].astype(bool)
    out = out.sort_values(KEY).reset_index(drop=True)
    report.rows_out = len(out)
    return out, report
