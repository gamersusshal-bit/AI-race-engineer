"""Strategy search: enumerate legal plans and minimise predicted total race time."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import pandas as pd

from race_engineer.config import COMPOUND_NAMES, MODE_NAMES
from race_engineer.sim.race import Plan, RaceState, Stint, check_plan, simulate


def enumerate_plans(state: RaceState, max_stops: int = 2, two_stop_step: int = 2,
                    first_compounds: tuple[str, ...] | None = None) -> list[tuple[Plan, RaceState]]:
    """All legal plans (normal mode) from ``state``.

    At the start of the race the starting compound is itself a choice, so each
    plan is paired with the state it starts from.
    """
    n = state.laps_remaining
    free_start = state.lap == 1 and state.tyre_age == 0
    starts = first_compounds or (COMPOUND_NAMES if free_start else (state.compound,))
    out = []
    for start in starts:
        st = RaceState(state.circuit, state.driver, state.lap, start, state.tyre_age,
                       state.fuel_kg, state.compounds_used, state.track_temp)
        for stops in range(0, max_stops + 1):
            step = 1 if stops <= 1 else two_stop_step
            for cuts in combinations(range(1, n, step), stops):
                lens = [b - a for a, b in zip((0, *cuts), (*cuts, n))]
                for comps in product(COMPOUND_NAMES, repeat=stops):
                    plan = Plan(tuple(Stint(c, l) for c, l in zip((start, *comps), lens)))
                    if not check_plan(plan, st):
                        out.append((plan, st))
    return out


def _with_modes(plan: Plan) -> list[Plan]:
    return [Plan(tuple(Stint(s.compound, s.laps, m) for s, m in zip(plan.stints, modes)))
            for modes in product(MODE_NAMES, repeat=len(plan.stints))]


@dataclass
class StrategyResult:
    state: RaceState
    summary: pd.DataFrame           # one row per plan, fastest first
    plans: list[Plan]               # aligned with summary rows
    states: list[RaceState]
    laps: pd.DataFrame              # lap-by-lap frames for the top plans (plan_rank column)

    @property
    def best(self) -> Plan:
        return self.plans[0]

    @property
    def best_state(self) -> RaceState:
        return self.states[0]


def _run(pairs: list[tuple[Plan, RaceState]], pace_model) -> tuple[pd.DataFrame, list, list, dict]:
    """Simulate (plan, state) pairs, grouping by start state for vectorised batches."""
    groups: dict[RaceState, list[Plan]] = {}
    for plan, st in pairs:
        groups.setdefault(st, []).append(plan)
    rows, frames = [], {}
    for st, plans in groups.items():
        summary, frame = simulate(plans, st, pace_model)
        for r in summary.itertuples(index=False):
            rows.append((r.total_time_s, r.strategy, r.stops, plans[r.plan_id], st, (st, r.plan_id)))
        frames[st] = frame
    rows.sort(key=lambda r: r[0])
    summary = pd.DataFrame({"strategy": [r[1] for r in rows], "stops": [r[2] for r in rows],
                            "total_time_s": [r[0] for r in rows]})
    summary["gap_to_best_s"] = summary["total_time_s"] - summary["total_time_s"].iloc[0]
    return summary, [r[3] for r in rows], [r[4] for r in rows], {"frames": frames, "keys": [r[5] for r in rows]}


def optimize(state: RaceState, pace_model, max_stops: int = 2, two_stop_step: int = 2,
             optimize_modes: bool = False, refine_top: int = 15, keep_laps: int = 5) -> StrategyResult:
    pairs = enumerate_plans(state, max_stops, two_stop_step)
    if not pairs:
        raise ValueError("no legal strategy from this state (check tyre life and compound rule)")
    summary, plans, states, aux = _run(pairs, pace_model)
    if optimize_modes:
        # Second stage: expand drive modes only for the most promising pit/compound plans.
        expanded = [(p, st) for plan, st in zip(plans[:refine_top], states[:refine_top])
                    for p in _with_modes(plan) if not check_plan(p, st)]
        summary, plans, states, aux = _run(expanded, pace_model)
    tops = []
    for rank, (st, pid) in enumerate(aux["keys"][:keep_laps]):
        f = aux["frames"][st]
        tops.append(f[f["plan_id"] == pid].assign(plan_rank=rank + 1, strategy=summary["strategy"].iloc[rank]))
    return StrategyResult(state, summary, plans, states, pd.concat(tops, ignore_index=True))


def pit_now_or_stay_out(state: RaceState, pace_model, max_stops: int = 2) -> dict:
    """In-race call: best plan that pits at the end of this lap vs best plan that stays out."""
    res = optimize(state, pace_model, max_stops=max_stops, two_stop_step=1)
    first = pd.Series([p.stints[0].laps if p.n_stops else None for p in res.plans])
    now, later = first == 1, first != 1
    out = {"result": res, "pit_now": None, "stay_out": None}
    if now.any():
        i = int(now.idxmax()); out["pit_now"] = {"plan": res.plans[i], **res.summary.iloc[i].to_dict()}
    if later.any():
        i = int(later.idxmax()); out["stay_out"] = {"plan": res.plans[i], **res.summary.iloc[i].to_dict()}
    if out["pit_now"] and out["stay_out"]:
        out["delta_s"] = out["pit_now"]["total_time_s"] - out["stay_out"]["total_time_s"]
        out["call"] = "PIT NOW" if out["delta_s"] < 0 else "STAY OUT"
    else:
        out["delta_s"] = None
        out["call"] = "PIT NOW" if out["pit_now"] else "STAY OUT"
    return out
