"""Does optimising on the learned models produce good decisions?

Every strategy is scored on the reference physics ("ground truth"), never on
the model that chose it:
  * regret          = true time of the learned-model choice - true time of the oracle optimum
  * gain vs baseline = true time of a rule-based baseline   - true time of the learned-model choice
"""
from __future__ import annotations

import pandas as pd

from race_engineer.config import CIRCUITS
from race_engineer.sim.physics import PhysicsPaceModel
from race_engineer.sim.race import RaceState, simulate
from race_engineer.strategy.baselines import BASELINES, start_state
from race_engineer.strategy.optimizer import optimize


def evaluate_strategies(pace_model, driver: str = "CHR", track_temp: float = 35.0,
                        deg_scales=(0.85, 1.0, 1.15)) -> pd.DataFrame:
    rows = []
    for circuit in CIRCUITS:
        state = RaceState(circuit, driver, track_temp=track_temp)
        learned = optimize(state, pace_model, keep_laps=1)
        for scale in deg_scales:
            truth = PhysicsPaceModel(deg_scale=scale)
            true_time = lambda plan, st: simulate([plan], st, truth)[0]["total_time_s"].iloc[0]  # noqa: E731
            oracle = optimize(state, truth, keep_laps=1)
            chosen = true_time(learned.best, learned.best_state)
            row = {
                "circuit": circuit, "deg_scale": scale,
                "learned_strategy": learned.summary["strategy"].iloc[0],
                "oracle_strategy": oracle.summary["strategy"].iloc[0],
                "predicted_time_s": learned.summary["total_time_s"].iloc[0],
                "true_time_s": chosen,
                "regret_s": chosen - oracle.summary["total_time_s"].iloc[0],
            }
            for name, make in BASELINES.items():
                plan = make(circuit)
                row[f"gain_vs_{name}"] = true_time(plan, start_state(circuit, driver, plan, track_temp)) - chosen
            rows.append(row)
    return pd.DataFrame(rows)
