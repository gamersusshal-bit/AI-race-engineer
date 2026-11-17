"""Synthetic car telemetry with irregular sampling (simulation assumption)."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from race_engineer.config import CIRCUITS, MODES

_GRID = 500


def _circuit_profile(circuit: str) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic speed-vs-distance shape (km/h) for a circuit."""
    c = CIRCUITS[circuit]
    rng = np.random.default_rng(zlib.crc32(circuit.encode()))
    d = np.linspace(0.0, c.length_m, _GRID)
    v = np.full(_GRID, 325.0)
    for pos in np.sort(rng.uniform(0.04, 0.96, c.n_corners)) * c.length_m:
        depth, width = rng.uniform(60, 240), rng.uniform(60, 160)
        v = np.minimum(v, 325.0 - depth * np.exp(-(((d - pos) / width) ** 2)))
    return d, v


def lap_seed(session_id: str, driver: str, lap_number: int) -> int:
    return zlib.crc32(f"{session_id}|{driver}|{lap_number}".encode())


def synth_lap_telemetry(circuit: str, lap_time_s: float, mode: str, seed: int,
                        lift_quantile: float | None = None) -> pd.DataFrame:
    """One lap of telemetry whose duration matches ``lap_time_s``."""
    rng = np.random.default_rng(seed)
    d, v = _circuit_profile(circuit)
    seg_t = np.diff(d) / (0.5 * (v[1:] + v[:-1]) / 3.6)
    scale = seg_t.sum() / lap_time_s                 # stretch speed so the lap takes lap_time_s
    v, t_grid = v * scale, np.concatenate([[0.0], np.cumsum(seg_t / scale)])

    dt = rng.uniform(0.08, 0.40, int(lap_time_s / 0.08) + 2)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    t = np.append(t[t < lap_time_s], lap_time_s)
    speed = np.interp(t, t_grid, v)
    accel = np.gradient(speed / 3.6, t)
    brake = accel < -4.0
    throttle = np.where(brake, 0.0, np.where(accel > -0.5, 100.0, 30.0))
    q = MODES[mode].lift_quantile if lift_quantile is None else lift_quantile
    if q < 1.0:                                      # lift-and-coast at the end of straights
        throttle[(speed >= np.quantile(speed, min(q, 1.0))) & (accel <= 0.3)] = 0.0
    gear = np.clip(1 + (speed // 42).astype(int), 1, 8)
    rpm = 6500 + 5500 * ((speed % 42) / 42) + rng.normal(0, 60, len(t))
    drs = (speed > 0.9 * speed.max()) & (throttle == 100.0)
    return pd.DataFrame({
        "time_s": t, "distance_m": np.interp(t, t_grid, d), "speed": speed + rng.normal(0, 0.5, len(t)),
        "throttle": throttle, "brake": brake, "rpm": rpm, "gear": gear, "drs": drs,
    })
