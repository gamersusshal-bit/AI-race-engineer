"""Race simulation: turn strategy plans into lap-by-lap state and total race time.

A plan is a sequence of stints. The first stint continues on the tyres described
by the ``RaceState``; every later stint starts with a pit stop for a fresh set.
State transitions are deterministic given a plan, so many plans are simulated in
one vectorised batch.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from race_engineer.config import (
    CIRCUITS, COMPOUND_NAMES, COMPOUNDS, FUEL_MARGIN_KG, MIN_DISTINCT_COMPOUNDS,
    MODE_NAMES, MODES, REFERENCE_TRACK_TEMP_C,
)


@dataclass(frozen=True)
class Stint:
    compound: str
    laps: int
    mode: str = "normal"


@dataclass(frozen=True)
class Plan:
    stints: tuple[Stint, ...]

    @property
    def n_stops(self) -> int:
        return len(self.stints) - 1

    @property
    def pit_laps_relative(self) -> list[int]:
        """Laps (counted from the first simulated lap = 1) at whose end the car pits."""
        return list(np.cumsum([s.laps for s in self.stints])[:-1])

    def label(self, first_lap: int = 1) -> str:
        parts, lap = [], first_lap - 1
        for i, s in enumerate(self.stints):
            lap += s.laps
            mode = "" if s.mode == "normal" else f"/{s.mode}"
            tail = f" → pit L{lap}" if i < len(self.stints) - 1 else ""
            parts.append(f"{s.compound[0]}{mode} ×{s.laps}{tail}")
        return " | ".join(parts)


@dataclass(frozen=True)
class RaceState:
    """Situation at the start of ``lap`` (the next lap to be driven)."""
    circuit: str
    driver: str
    lap: int = 1
    compound: str = "MEDIUM"
    tyre_age: int = 0                       # laps already done on the current set
    fuel_kg: float | None = None            # None -> fuel load sized to the plan
    compounds_used: tuple[str, ...] = field(default_factory=tuple)
    track_temp: float = REFERENCE_TRACK_TEMP_C

    @property
    def laps_remaining(self) -> int:
        return CIRCUITS[self.circuit].total_laps - self.lap + 1


def check_plan(plan: Plan, state: RaceState) -> list[str]:
    """Return rule/constraint violations (empty list means the plan is legal)."""
    problems = []
    if not plan.stints:
        return ["plan has no stints"]
    if any(s.laps < 1 for s in plan.stints):
        problems.append("every stint needs at least one lap")
    if sum(s.laps for s in plan.stints) != state.laps_remaining:
        problems.append("stint lengths do not add up to the remaining race distance")
    if plan.stints[0].compound != state.compound:
        problems.append("first stint must continue on the fitted compound")
    for s in plan.stints:
        if s.compound not in COMPOUNDS:
            problems.append(f"unknown compound {s.compound}")
        if s.mode not in MODES:
            problems.append(f"unknown drive mode {s.mode}")
    if problems:
        return problems
    used = set(state.compounds_used) | {s.compound for s in plan.stints}
    if len(used) < MIN_DISTINCT_COMPOUNDS:
        problems.append(f"must use at least {MIN_DISTINCT_COMPOUNDS} different compounds")
    for i, s in enumerate(plan.stints):
        life = s.laps + (state.tyre_age if i == 0 else 0)
        if life > COMPOUNDS[s.compound].max_stint_laps:
            problems.append(f"stint {i + 1}: {life} laps exceeds {s.compound} tyre life")
    if state.fuel_kg is not None:
        table = {m: CIRCUITS[state.circuit].fuel_per_lap_kg * MODES[m].fuel_mult for m in MODES}
        need = sum(table[s.mode] * s.laps for s in plan.stints)
        if need > state.fuel_kg + 1e-9:
            problems.append(f"needs {need:.1f} kg of fuel but only {state.fuel_kg:.1f} kg on board")
    return problems


def build_frame(plans: list[Plan], state: RaceState, fuel_table: dict[str, float]) -> pd.DataFrame:
    """Lap-by-lap state for each plan (one row per plan per lap)."""
    circuit = CIRCUITS[state.circuit]
    comp_idx = {c: i for i, c in enumerate(COMPOUND_NAMES)}
    mode_idx = {m: i for i, m in enumerate(MODE_NAMES)}
    wear_mult = np.array([MODES[m].wear_mult for m in MODE_NAMES])
    burn = np.array([fuel_table[m] for m in MODE_NAMES])

    cols = {k: [] for k in ("plan_id", "lap_number", "stint", "comp", "mode", "tyre_age", "wear", "pit_in", "fuel_kg")}
    for pid, plan in enumerate(plans):
        lens = np.array([s.laps for s in plan.stints])
        n, k = int(lens.sum()), len(lens)
        ends = np.cumsum(lens)
        stint = np.repeat(np.arange(k), lens)
        in_stint = np.arange(n) - np.repeat(ends - lens, lens)
        age0 = np.zeros(k)
        age0[0] = state.tyre_age
        mode = np.repeat([mode_idx[s.mode] for s in plan.stints], lens)
        pit_in = np.zeros(n, dtype=bool)
        pit_in[ends[:-1] - 1] = True
        lap_burn = burn[mode]
        start_fuel = state.fuel_kg if state.fuel_kg is not None else lap_burn.sum() + FUEL_MARGIN_KG
        cols["plan_id"].append(np.full(n, pid))
        cols["lap_number"].append(state.lap + np.arange(n))
        cols["stint"].append(stint + 1)
        cols["comp"].append(np.repeat([comp_idx[s.compound] for s in plan.stints], lens))
        cols["mode"].append(mode)
        cols["tyre_age"].append(in_stint + np.repeat(age0, lens))
        cols["wear"].append(np.repeat(age0, lens) + in_stint * wear_mult[mode])
        cols["pit_in"].append(pit_in)
        cols["fuel_kg"].append(start_fuel - (np.cumsum(lap_burn) - lap_burn))

    c = {k: np.concatenate(v) for k, v in cols.items()}
    return pd.DataFrame({
        "plan_id": c["plan_id"],
        "circuit": state.circuit,
        "driver": state.driver,
        "lap_number": c["lap_number"],
        "total_laps": circuit.total_laps,
        "lap_frac": c["lap_number"] / circuit.total_laps,
        "stint": c["stint"],
        "compound": np.array(COMPOUND_NAMES)[c["comp"]],
        "drive_mode": np.array(MODE_NAMES)[c["mode"]],
        "tyre_age": c["tyre_age"],
        "wear": c["wear"],
        "fuel_kg": c["fuel_kg"],
        "track_temp": state.track_temp,
        "pit_in": c["pit_in"],
    })


def simulate(plans: list[Plan], state: RaceState, pace_model) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate plans from ``state`` to the flag.

    Returns ``(summary, laps)``: one row per plan (sorted fastest first) and the
    lap-by-lap frame with predicted lap times.
    """
    for plan in plans:
        problems = check_plan(plan, state)
        if problems:
            raise ValueError(f"illegal plan {plan.label(state.lap)}: {'; '.join(problems)}")
    frame = build_frame(plans, state, pace_model.fuel_table(state.circuit))
    frame["clean_lap_s"] = pace_model.predict(frame)
    frame["lap_time_s"] = frame["clean_lap_s"] + frame["pit_in"] * pace_model.pit_loss(state.circuit)
    totals = frame.groupby("plan_id", sort=True)["lap_time_s"].sum().to_numpy()
    summary = pd.DataFrame({
        "plan_id": np.arange(len(plans)),
        "strategy": [p.label(state.lap) for p in plans],
        "stops": [p.n_stops for p in plans],
        "total_time_s": totals,
    }).sort_values("total_time_s", kind="stable").reset_index(drop=True)
    summary["gap_to_best_s"] = summary["total_time_s"] - summary["total_time_s"].iloc[0]
    return summary, frame
