"""Gymnasium environment for lap-by-lap strategy decisions (long-term RL extension).

Dynamics come from the reference physics, the same model the simulator is
validated against, so RL policies can be compared with the optimizer and the
rule-based baselines on equal terms.

Actions (taken before each lap):
  0 push · 1 normal · 2 conserve · 3 pit→SOFT · 4 pit→MEDIUM · 5 pit→HARD
A pit action fits fresh tyres (pit loss added) and drives the lap in normal mode.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from race_engineer.config import (
    CIRCUITS, COMPOUND_NAMES, COMPOUNDS, DRIVERS, FUEL_MARGIN_KG, MIN_DISTINCT_COMPOUNDS,
    MODE_NAMES, MODES,
)
from race_engineer.sim.physics import clean_lap_time
from race_engineer.sim.race import Plan

RULE_PENALTY_S = 30.0
FUEL_OUT_PENALTY_S = 60.0
REWARD_SCALE = 10.0


class RaceStrategyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, circuit: str = "Sakhir", driver: str = "CHR", start_compound: str | None = None,
                 start_fuel_kg: float | None = None, lap_noise_s: float = 0.0,
                 deg_scale_sigma: float = 0.0, track_temp: float = 35.0):
        self.circuit, self.driver = CIRCUITS[circuit], driver
        self.start_compound, self.start_fuel_kg = start_compound, start_fuel_kg
        self.lap_noise_s, self.deg_scale_sigma, self.track_temp = lap_noise_s, deg_scale_sigma, track_temp
        self.action_space = gym.spaces.Discrete(len(MODE_NAMES) + len(COMPOUND_NAMES))
        # lap fraction, compound one-hot (3), wear, fuel fraction, compounds-used one-hot (3)
        self.observation_space = gym.spaces.Box(0.0, 2.0, shape=(9,), dtype=np.float32)

    def _obs(self) -> np.ndarray:
        onehot = [float(self.compound == c) for c in COMPOUND_NAMES]
        used = [float(c in self.used) for c in COMPOUND_NAMES]
        return np.array([(self.lap - 1) / self.circuit.total_laps, *onehot,
                         min(self.wear / COMPOUNDS[self.compound].max_stint_laps, 2.0),
                         max(self.fuel, 0.0) / self.fuel0, *used], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        c = self.circuit
        self.compound = self.start_compound or str(self.np_random.choice(COMPOUND_NAMES))
        self.used, self.lap, self.wear, self.total_time = {self.compound}, 1, 0.0, 0.0
        self.fuel0 = self.start_fuel_kg or c.fuel_per_lap_kg * c.total_laps + FUEL_MARGIN_KG
        self.fuel = self.fuel0
        self.deg_scale = float(self.np_random.lognormal(0.0, self.deg_scale_sigma)) if self.deg_scale_sigma else 1.0
        return self._obs(), {}

    def step(self, action: int):
        c, lap_time = self.circuit, 0.0
        action = int(action)
        if action >= len(MODE_NAMES):
            self.compound = COMPOUND_NAMES[action - len(MODE_NAMES)]
            self.used.add(self.compound)
            self.wear, lap_time, mode = 0.0, c.pit_loss_s, MODES["normal"]
        else:
            mode = MODES[MODE_NAMES[action]]
        lap_time += float(clean_lap_time(c, DRIVERS.get(self.driver, 0.0), self.compound, self.wear, self.fuel,
                                         self.lap, mode.name, self.track_temp, self.deg_scale))
        if self.lap_noise_s:
            lap_time += float(self.np_random.normal(0, self.lap_noise_s))
        self.fuel -= c.fuel_per_lap_kg * mode.fuel_mult
        self.wear += mode.wear_mult
        self.lap += 1
        penalty, info = 0.0, {}
        terminated = self.lap > c.total_laps
        if self.fuel < 0:                                   # ran dry: race over
            penalty, terminated, info["violation"] = FUEL_OUT_PENALTY_S, True, "out_of_fuel"
        elif terminated and len(self.used) < MIN_DISTINCT_COMPOUNDS:
            penalty, info["violation"] = RULE_PENALTY_S, "single_compound"
        self.total_time += lap_time + penalty
        info["total_time_s"] = self.total_time
        reward = -(lap_time + penalty - c.base_lap_s) / REWARD_SCALE
        return self._obs(), reward, terminated, False, info


def plan_to_actions(plan: Plan) -> list[int]:
    """Action sequence that drives ``plan`` (pit stops happen at the start of the next stint)."""
    actions = []
    for i, s in enumerate(plan.stints):
        first = len(MODE_NAMES) + COMPOUND_NAMES.index(s.compound) if i else MODE_NAMES.index(s.mode)
        actions += [first] + [MODE_NAMES.index(s.mode)] * (s.laps - 1)
    return actions


def rollout(env: RaceStrategyEnv, policy, seed: int | None = None) -> float:
    """Run one race with ``policy(obs) -> action`` and return the total race time."""
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy(obs))
    return info["total_time_s"]
