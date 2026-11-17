"""Fuel consumption.

Two estimators with different uses:
  * ``FuelModel.predict``  - fuel burned on a *completed* lap from its telemetry
    (driving behaviour); a measurement-style estimate, not a forecast.
  * ``FuelModel.plan_table`` - expected burn per lap by circuit and drive mode,
    which is what the race simulator needs to project fuel forward.

Requires fuel labels (``fuel_used_kg``), which public timing data does not
have; they come from the simulator or a team data source.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from race_engineer.features.laps import clean_laps
from race_engineer.models.laptime import split_by_season
from race_engineer.models.metrics import regression_metrics

FUEL_FEATURES = ["circuit", "tel_full_throttle_frac", "tel_mean_rpm", "tel_avg_speed", "tel_brake_time_s"]
TARGET = "fuel_used_kg"


@dataclass
class FuelModel:
    pipeline: Pipeline
    table: dict[str, dict[str, float]]      # circuit -> mode -> kg per lap

    def predict(self, laps: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(laps[FUEL_FEATURES])

    def plan_table(self, circuit: str) -> dict[str, float]:
        return self.table[circuit]


def train_fuel_model(laps: pd.DataFrame) -> dict:
    data = clean_laps(laps).dropna(subset=FUEL_FEATURES + [TARGET])
    train, test = split_by_season(data)
    # Per-circuit slopes: throttle use scales fuel differently on each track.
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["circuit"]),
                             ("num", "passthrough", FUEL_FEATURES[1:])])
    pipe = Pipeline([("pre", pre), ("model", LinearRegression())]).fit(train[FUEL_FEATURES], train[TARGET])

    circuit_mean = train.groupby("circuit")[TARGET].mean()
    mode_mean = train.groupby(["circuit", "drive_mode"])[TARGET].mean()
    idx = pd.MultiIndex.from_frame(test[["circuit", "drive_mode"]])
    results = {
        "baseline_circuit_mean": {"test": regression_metrics(test[TARGET], test["circuit"].map(circuit_mean))},
        "baseline_circuit_mode_mean": {"test": regression_metrics(test[TARGET], mode_mean.reindex(idx).to_numpy())},
        "telemetry_linear": {"test": regression_metrics(test[TARGET], pipe.predict(test[FUEL_FEATURES]))},
    }
    table = {c: mode_mean.loc[c].to_dict() for c in mode_mean.index.get_level_values(0).unique()}
    return {"model": FuelModel(pipe, table), "results": results}
