"""Simple rule-based strategies used as yardsticks for the optimizer."""
from __future__ import annotations

from race_engineer.config import CIRCUITS
from race_engineer.sim.race import Plan, RaceState, Stint


def one_stop_midpoint(circuit: str) -> Plan:
    """MEDIUM to half distance, then HARD."""
    n = CIRCUITS[circuit].total_laps
    return Plan((Stint("MEDIUM", n // 2), Stint("HARD", n - n // 2)))


def two_stop_thirds(circuit: str) -> Plan:
    """SOFT / MEDIUM / MEDIUM in equal thirds."""
    n = CIRCUITS[circuit].total_laps
    a = n // 3
    return Plan((Stint("SOFT", a), Stint("MEDIUM", a), Stint("MEDIUM", n - 2 * a)))


def pit_at_fixed_age(circuit: str, soft_age: int = 15, medium_age: int = 24) -> Plan:
    """Rule of thumb: start SOFT, pit at a fixed tyre age, then MEDIUMs changed at a fixed age, HARD to finish."""
    n = CIRCUITS[circuit].total_laps
    stints, left = [Stint("SOFT", soft_age)], n - soft_age
    if left > medium_age + 12:
        stints.append(Stint("MEDIUM", medium_age)); left -= medium_age
    stints.append(Stint("HARD", left))
    return Plan(tuple(stints))


BASELINES = {
    "1-stop M→H at half distance": one_stop_midpoint,
    "2-stop S→M→M in thirds": two_stop_thirds,
    "Pit at fixed tyre age": pit_at_fixed_age,
}


def start_state(circuit: str, driver: str, plan: Plan, track_temp: float = 35.0) -> RaceState:
    return RaceState(circuit, driver, compound=plan.stints[0].compound, track_temp=track_temp)
