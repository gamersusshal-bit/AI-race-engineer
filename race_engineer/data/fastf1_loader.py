"""FastF1 ingestion into the canonical lap schema.

Public timing data has no fuel quantity or drive mode. ``fuel_kg`` is therefore
an *estimate* (linear burn-down from an assumed per-lap consumption, flagged in
``fuel_is_estimated``) and ``drive_mode`` is set to "normal".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from race_engineer.config import COMPOUND_NAMES, FUEL_MARGIN_KG, REFERENCE_TRACK_TEMP_C

DEFAULT_FUEL_PER_LAP_KG = 1.8


def fastf1_laps_to_canonical(laps: pd.DataFrame, circuit: str, season: int, total_laps: int,
                             fuel_per_lap_kg: float = DEFAULT_FUEL_PER_LAP_KG) -> pd.DataFrame:
    """Map a FastF1 ``session.laps`` table (pure function, no network)."""
    secs = lambda col: pd.to_timedelta(laps[col]).dt.total_seconds()  # noqa: E731
    lap_no = pd.to_numeric(laps["LapNumber"], errors="coerce")
    out = pd.DataFrame({
        "session_id": f"{season}-{circuit}",
        "season": season,
        "circuit": circuit,
        "driver": laps["Driver"].to_numpy(),
        "lap_number": lap_no.to_numpy(),
        "total_laps": total_laps,
        "lap_time_s": secs("LapTime").to_numpy(),
        "sector1_s": secs("Sector1Time").to_numpy(),
        "sector2_s": secs("Sector2Time").to_numpy(),
        "sector3_s": secs("Sector3Time").to_numpy(),
        "stint": pd.to_numeric(laps["Stint"], errors="coerce").to_numpy(),
        "compound": laps["Compound"].astype(str).str.upper().to_numpy(),
        "tyre_age": pd.to_numeric(laps["TyreLife"], errors="coerce").to_numpy() - 1,  # FastF1 counts from 1
        "fuel_kg": ((total_laps - lap_no + 1) * fuel_per_lap_kg + FUEL_MARGIN_KG).to_numpy(),
        "fuel_is_estimated": True,
        "track_temp": (laps["TrackTemp"].to_numpy() if "TrackTemp" in laps else REFERENCE_TRACK_TEMP_C),
        "drive_mode": "normal",
        "is_pit_in_lap": laps["PitInTime"].notna().to_numpy(),
        "is_pit_out_lap": laps["PitOutTime"].notna().to_numpy(),
    })
    out = out[out["compound"].isin(COMPOUND_NAMES)]          # dry compounds only
    return out.dropna(subset=["lap_number", "stint"]).reset_index(drop=True)


def load_fastf1_session(year: int, grand_prix: str, session: str = "R",
                        cache_dir: str = "fastf1_cache") -> pd.DataFrame:
    """Download a session with FastF1 and return canonical laps (needs network access)."""
    import os

    import fastf1

    os.makedirs(cache_dir, exist_ok=True)
    fastf1.Cache.enable_cache(cache_dir)
    ses = fastf1.get_session(year, grand_prix, session)
    ses.load(telemetry=False, weather=True, messages=False)
    laps = ses.laps.reset_index(drop=True)
    try:
        laps["TrackTemp"] = laps.get_weather_data()["TrackTemp"].to_numpy()
    except Exception:  # weather is optional; fall back to the reference temperature
        pass
    total = int(getattr(ses, "total_laps", None) or np.nanmax(laps["LapNumber"]))
    return fastf1_laps_to_canonical(laps, str(ses.event["Location"]), year, total)


def fastf1_telemetry_to_canonical(car_data: pd.DataFrame) -> pd.DataFrame:
    """Map ``lap.get_car_data()`` to the telemetry columns used by the feature code."""
    return pd.DataFrame({
        "time_s": pd.to_timedelta(car_data["Time"]).dt.total_seconds().to_numpy(),
        "speed": car_data["Speed"].to_numpy(), "throttle": car_data["Throttle"].to_numpy(),
        "brake": car_data["Brake"].astype(bool).to_numpy(), "rpm": car_data["RPM"].to_numpy(),
        "gear": car_data["nGear"].to_numpy(), "drs": (car_data["DRS"].to_numpy() >= 10),
    })
