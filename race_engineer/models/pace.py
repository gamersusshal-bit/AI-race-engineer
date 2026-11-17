"""Learned pace model: the bridge from the prediction models to the simulator."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from race_engineer.config import CIRCUITS, MODES
from race_engineer.models.fuel import FuelModel
from race_engineer.models.laptime import LapTimeModel


def estimate_pit_loss(laps: pd.DataFrame, model: LapTimeModel) -> dict[str, float]:
    """Median time lost per stop: (in-lap + out-lap) minus the predicted clean laps."""
    pit = laps[laps["is_pit_in_lap"] | laps["is_pit_out_lap"]].copy()
    pit["excess"] = pit["lap_time_s"] - model.predict(pit)
    key = ["session_id", "driver"]
    ins = pit[pit["is_pit_in_lap"]][key + ["circuit", "lap_number", "excess"]]
    outs = pit[pit["is_pit_out_lap"]][key + ["lap_number", "excess"]].assign(lap_number=lambda d: d["lap_number"] - 1)
    stops = ins.merge(outs, on=key + ["lap_number"], suffixes=("_in", "_out"))
    return (stops["excess_in"] + stops["excess_out"]).groupby(stops["circuit"]).median().to_dict()


@dataclass
class LearnedPaceModel:
    laptime: LapTimeModel
    pit_losses: dict[str, float]
    fuel: FuelModel | None = None
    name: str = "learned"

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.laptime.predict(frame)

    def fuel_table(self, circuit: str) -> dict[str, float]:
        if self.fuel is not None and circuit in self.fuel.table:
            return {m: self.fuel.table[circuit].get(m, np.mean(list(self.fuel.table[circuit].values()))) for m in MODES}
        c = CIRCUITS[circuit]   # no fuel labels: fall back to the configured assumption
        return {m.name: c.fuel_per_lap_kg * m.fuel_mult for m in MODES.values()}

    def pit_loss(self, circuit: str) -> float:
        return self.pit_losses[circuit]
