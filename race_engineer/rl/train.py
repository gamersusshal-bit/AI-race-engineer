"""Optional PPO experiment (needs ``pip install stable-baselines3``).

Reports the trained policy against the optimizer's plan and a rule-based
baseline on the same environment; RL is not assumed to be better.
"""
from __future__ import annotations

import argparse

from race_engineer.rl.env import RaceStrategyEnv, plan_to_actions, rollout
from race_engineer.sim.physics import PhysicsPaceModel
from race_engineer.sim.race import RaceState
from race_engineer.strategy.baselines import one_stop_midpoint
from race_engineer.strategy.optimizer import optimize


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--circuit", default="Sakhir")
    p.add_argument("--timesteps", type=int, default=300_000)
    a = p.parse_args()
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise SystemExit("stable-baselines3 is not installed: pip install stable-baselines3") from e

    model = PPO("MlpPolicy", RaceStrategyEnv(a.circuit), verbose=1, gamma=1.0, n_steps=2048)
    model.learn(a.timesteps)

    def fixed(plan):
        actions = iter(plan_to_actions(plan))
        return lambda obs: next(actions)

    best = optimize(RaceState(a.circuit, "CHR"), PhysicsPaceModel(), keep_laps=1)
    base = one_stop_midpoint(a.circuit)
    print("optimizer :", rollout(RaceStrategyEnv(a.circuit, start_compound=best.best.stints[0].compound), fixed(best.best)))
    print("baseline  :", rollout(RaceStrategyEnv(a.circuit, start_compound=base.stints[0].compound), fixed(base)))
    for comp in ("SOFT", "MEDIUM", "HARD"):
        env = RaceStrategyEnv(a.circuit, start_compound=comp)
        print(f"PPO ({comp} start):", rollout(env, lambda obs: int(model.predict(obs, deterministic=True)[0])))


if __name__ == "__main__":
    main()
