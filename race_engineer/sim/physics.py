"""Reference ("ground truth") vehicle model.

Used to generate synthetic races and as an oracle when judging strategies chosen
with learned models. Every function is vectorised over numpy arrays.
"""
from __future__ import annotations

import numpy as np

from race_engineer.config import (
    CIRCUITS, COMPOUNDS, DRIVERS, MODES, REFERENCE_TRACK_TEMP_C, Circuit,
)


def _lookup(names, table: dict, attr: str) -> np.ndarray:
    names = np.asarray(names)
    out = np.empty(names.shape, dtype=float)
    for key, obj in table.items():
        out[names == key] = getattr(obj, attr)
    return out


def tyre_delta(compound, wear, circuit: Circuit, track_temp=REFERENCE_TRACK_TEMP_C,
               deg_scale: float = 1.0) -> np.ndarray:
    """Lap-time cost of the tyre relative to a fresh MEDIUM."""
    wear = np.asarray(wear, dtype=float)
    k = circuit.deg_mult * deg_scale * (1.0 + 0.012 * (np.asarray(track_temp) - REFERENCE_TRACK_TEMP_C))
    over = np.clip(wear - _lookup(compound, COMPOUNDS, "cliff_wear"), 0.0, None)
    deg = _lookup(compound, COMPOUNDS, "deg_s_per_lap") * wear + _lookup(compound, COMPOUNDS, "cliff_quad") * over**2
    return _lookup(compound, COMPOUNDS, "pace_offset_s") + k * deg


def track_evolution(lap_number, circuit: Circuit) -> np.ndarray:
    return -circuit.track_evo_s * (1.0 - np.exp(-np.asarray(lap_number, dtype=float) / 15.0))


def clean_lap_time(circuit: Circuit, driver_offset, compound, wear, fuel_kg, lap_number,
                   mode, track_temp=REFERENCE_TRACK_TEMP_C, deg_scale: float = 1.0) -> np.ndarray:
    return (
        circuit.base_lap_s
        + np.asarray(driver_offset, dtype=float)
        + tyre_delta(compound, wear, circuit, track_temp, deg_scale)
        + np.asarray(fuel_kg, dtype=float) * circuit.fuel_effect_s_per_kg
        + track_evolution(lap_number, circuit)
        + _lookup(mode, MODES, "pace_delta_s")
    )


class PhysicsPaceModel:
    """Pace-model interface backed by the reference physics."""

    name = "physics"

    def __init__(self, deg_scale: float = 1.0, driver_offsets: dict[str, float] | None = None):
        self.deg_scale = deg_scale
        self.driver_offsets = DRIVERS if driver_offsets is None else driver_offsets

    def predict(self, frame) -> np.ndarray:
        circuit = CIRCUITS[frame["circuit"].iloc[0]]
        offsets = frame["driver"].map(self.driver_offsets).fillna(0.0).to_numpy()
        return clean_lap_time(
            circuit, offsets, frame["compound"].to_numpy(), frame["wear"].to_numpy(),
            frame["fuel_kg"].to_numpy(), frame["lap_number"].to_numpy(),
            frame["drive_mode"].to_numpy(), frame["track_temp"].to_numpy(), self.deg_scale,
        )

    def fuel_table(self, circuit: str) -> dict[str, float]:
        c = CIRCUITS[circuit]
        return {m.name: c.fuel_per_lap_kg * m.fuel_mult for m in MODES.values()}

    def pit_loss(self, circuit: str) -> float:
        return CIRCUITS[circuit].pit_loss_s
