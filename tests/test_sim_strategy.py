import numpy as np
import pytest

from race_engineer.config import CIRCUITS, FUEL_MARGIN_KG
from race_engineer.rl.env import RaceStrategyEnv, plan_to_actions, rollout
from race_engineer.sim.physics import PhysicsPaceModel
from race_engineer.sim.race import Plan, RaceState, Stint, build_frame, check_plan, simulate
from race_engineer.strategy.baselines import BASELINES, start_state
from race_engineer.strategy.optimizer import enumerate_plans, optimize, pit_now_or_stay_out

PHYS = PhysicsPaceModel()
STATE = RaceState("Sakhir", "CHR")


def test_state_transitions():
    plan = Plan((Stint("MEDIUM", 20, "push"), Stint("HARD", 37)))
    f = build_frame([plan], STATE, PHYS.fuel_table("Sakhir"))
    assert list(f["lap_number"]) == list(range(1, 58))
    assert list(f["tyre_age"][:3]) == [0, 1, 2] and f["tyre_age"].iloc[20] == 0      # reset at the stop
    assert f["wear"].iloc[19] == pytest.approx(19 * 1.35)                            # pushing wears faster
    assert f["pit_in"].sum() == 1 and f["pit_in"].iloc[19]
    assert (np.diff(f["fuel_kg"]) < 0).all()
    assert f["fuel_kg"].iloc[-1] - PHYS.fuel_table("Sakhir")["normal"] == pytest.approx(FUEL_MARGIN_KG)


def test_mid_race_state_continues_current_tyres():
    state = RaceState("Sakhir", "CHR", lap=30, compound="MEDIUM", tyre_age=12, compounds_used=("SOFT", "MEDIUM"))
    f = build_frame([Plan((Stint("MEDIUM", 28),))], state, PHYS.fuel_table("Sakhir"))
    assert f["lap_number"].iloc[0] == 30 and f["tyre_age"].iloc[0] == 12 and len(f) == 28


def test_total_time_is_lap_times_plus_pit_loss():
    plan = Plan((Stint("MEDIUM", 28), Stint("HARD", 29)))
    summary, laps = simulate([plan], STATE, PHYS)
    assert summary["total_time_s"].iloc[0] == pytest.approx(laps["clean_lap_s"].sum() + CIRCUITS["Sakhir"].pit_loss_s)


def test_physics_sanity():
    one = lambda **kw: simulate([Plan((Stint("MEDIUM", 28), Stint("HARD", 29)))],  # noqa: E731
                                RaceState("Sakhir", **kw), PHYS)[0]["total_time_s"].iloc[0]
    assert one(driver="ALF") < one(driver="HTL")                                     # faster driver
    assert one(driver="CHR", track_temp=45.0) > one(driver="CHR", track_temp=30.0)   # hotter = more deg
    _, laps = simulate([Plan((Stint("SOFT", 25), Stint("HARD", 32)))], RaceState("Sakhir", "CHR", compound="SOFT"), PHYS)
    soft = laps[laps["stint"] == 1]["clean_lap_s"].to_numpy()
    assert soft[-1] - soft[-2] > soft[2] - soft[1]                                   # past the cliff


@pytest.mark.parametrize("plan,match", [
    (Plan((Stint("MEDIUM", 57),)), "different compounds"),
    (Plan((Stint("MEDIUM", 28), Stint("MEDIUM", 29))), "different compounds"),
    (Plan((Stint("MEDIUM", 45), Stint("HARD", 12))), "tyre life"),
    (Plan((Stint("MEDIUM", 20), Stint("HARD", 30))), "remaining race distance"),
    (Plan((Stint("HARD", 20), Stint("MEDIUM", 37))), "fitted compound"),
])
def test_constraints_reject_illegal_plans(plan, match):
    assert any(match in p for p in check_plan(plan, STATE))
    with pytest.raises(ValueError, match=match):
        simulate([plan], STATE, PHYS)


def test_fuel_constraint():
    state = RaceState("Sakhir", "CHR", fuel_kg=100.0)
    assert any("fuel" in p for p in check_plan(Plan((Stint("MEDIUM", 28, "push"), Stint("HARD", 29, "push"))), state))
    assert not check_plan(Plan((Stint("MEDIUM", 28, "conserve"), Stint("HARD", 29, "conserve"))), state)


def test_enumerated_plans_are_all_legal_and_cover_start_compounds():
    pairs = enumerate_plans(RaceState("Spa", "CHR"))
    assert all(not check_plan(p, st) for p, st in pairs)
    assert {p.stints[0].compound for p, _ in pairs} == {"SOFT", "MEDIUM", "HARD"}
    assert {p.n_stops for p, _ in pairs} == {1, 2}


@pytest.mark.parametrize("circuit", list(CIRCUITS))
def test_optimizer_beats_every_baseline_on_ground_truth(circuit):
    best = optimize(RaceState(circuit, "CHR"), PHYS).summary["total_time_s"].iloc[0]
    for make in BASELINES.values():
        plan = make(circuit)
        base = simulate([plan], start_state(circuit, "CHR", plan), PHYS)[0]["total_time_s"].iloc[0]
        assert best <= base + 1e-9


def test_optimizer_matches_brute_force_on_one_stop():
    res = optimize(STATE, PHYS, max_stops=1)
    pairs = enumerate_plans(STATE, max_stops=1)
    brute = min(simulate([p], st, PHYS)[0]["total_time_s"].iloc[0] for p, st in pairs)
    assert res.summary["total_time_s"].iloc[0] == pytest.approx(brute)
    assert res.summary["total_time_s"].is_monotonic_increasing


def test_mode_search_never_worse_than_normal_mode():
    assert (optimize(STATE, PHYS, optimize_modes=True).summary["total_time_s"].iloc[0]
            <= optimize(STATE, PHYS).summary["total_time_s"].iloc[0] + 1e-9)


def test_pit_call_reacts_to_tyre_state():
    worn = RaceState("Sakhir", "CHR", lap=24, compound="SOFT", tyre_age=23, compounds_used=("SOFT",))
    fresh = RaceState("Sakhir", "CHR", lap=24, compound="MEDIUM", tyre_age=2, compounds_used=("SOFT", "MEDIUM"))
    assert pit_now_or_stay_out(worn, PHYS)["call"] == "PIT NOW"
    d = pit_now_or_stay_out(fresh, PHYS)
    assert d["call"] == "STAY OUT" and d["delta_s"] > 0


def test_rl_env_agrees_with_simulator_and_enforces_rules():
    # The env drives each pit lap in normal mode, so compare on a plan with normal-mode later stints.
    plan = Plan((Stint("MEDIUM", 20, "push"), Stint("SOFT", 17), Stint("HARD", 20)))
    summary, laps = simulate([plan], STATE, PHYS)
    env = RaceStrategyEnv("Sakhir", "CHR", start_compound="MEDIUM", start_fuel_kg=float(laps["fuel_kg"].iloc[0]))
    actions = iter(plan_to_actions(plan))
    assert rollout(env, lambda obs: next(actions), seed=0) == pytest.approx(summary["total_time_s"].iloc[0])

    env = RaceStrategyEnv("Sakhir", start_compound="HARD")
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(57):
        obs, reward, done, _, info = env.step(1)                  # never pits
    assert done and info["violation"] == "single_compound"
    env.reset(seed=0)
    done = False
    while not done:
        _, _, done, _, info = env.step(0)                         # pushing all race runs out of fuel
    assert info["violation"] == "out_of_fuel"
