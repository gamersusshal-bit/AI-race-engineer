import json

import numpy as np
import pytest

from race_engineer.config import CIRCUITS, COMPOUNDS
from race_engineer.models.degradation import fit_degradation
from race_engineer.models.laptime import LapTimeModel, split_by_season, train_laptime_models
from race_engineer.models.pace import LearnedPaceModel
from race_engineer.models.registry import load_model
from race_engineer.sim.race import RaceState
from race_engineer.strategy.explain import explain
from race_engineer.strategy.optimizer import optimize


def test_split_keeps_sessions_whole(laps):
    train, test = split_by_season(laps)
    assert not set(train["session_id"]) & set(test["session_id"])
    assert set(test["season"]) == {laps["season"].max()} and len(train) and len(test)


def test_models_beat_baselines_on_unseen_sessions(pipeline_run):
    m = pipeline_run[1]["metrics"]["laptime"]
    best, base = m["models"][m["selected"]]["test"], m["models"]["baseline_circuit_median"]["test"]
    assert best["mae"] < 0.5 * base["mae"]
    assert best["r2"] > 0.7
    assert m["selected"] == min((k for k in m["models"] if "cv_mae" in m["models"][k]),
                                key=lambda k: m["models"][k]["cv_mae"])
    fuel = pipeline_run[1]["metrics"]["fuel"]
    assert fuel["telemetry_linear"]["test"]["mae"] < fuel["baseline_circuit_mean"]["test"]["mae"]


def test_training_is_reproducible(laps):
    a, b = (train_laptime_models(laps, seed=1) for _ in range(2))
    assert a["best"] == b["best"]
    np.testing.assert_allclose(a["test_predictions"]["predicted_s"], b["test_predictions"]["predicted_s"])


def test_degradation_recovers_true_tyre_behaviour(laps):
    deg = fit_degradation(laps)
    for circuit, c in CIRCUITS.items():
        for comp in ("SOFT", "MEDIUM"):
            ages = np.arange(0, 13)                                # pre-cliff region
            est = np.polyfit(ages, deg.curve(circuit, comp, ages), 1)[0]
            assert est == pytest.approx(COMPOUNDS[comp].deg_s_per_lap * c.deg_mult, abs=0.035)
        wear15 = lambda comp: np.ptp(deg.curve(circuit, comp, [0, 15]))  # noqa: E731
        assert wear15("SOFT") > wear15("HARD")
        assert deg.offsets[circuit]["SOFT"] < 0 < deg.offsets[circuit]["HARD"]
        assert deg.progression_s_per_lap[circuit] < 0              # fuel burn + track evolution


def test_pit_loss_estimates_are_close(pipeline_run):
    for circuit, est in pipeline_run[1]["metrics"]["pit_loss_s"].items():
        assert est == pytest.approx(CIRCUITS[circuit].pit_loss_s, abs=1.5)


def test_end_to_end_artifacts_and_strategy_quality(pipeline_run):
    out, run = pipeline_run
    for name in ("laps.parquet", "metrics.json", "strategy_eval.csv", "degradation.csv",
                 "laptime_test_predictions.parquet"):
        assert (out / name).exists()
    json.loads((out / "metrics.json").read_text())
    pace, meta = load_model("pace", out / "models")
    assert meta["features"] and meta["data_fingerprint"] and meta["test_sessions"]

    nominal = run["strategy"][run["strategy"]["deg_scale"] == 1.0]
    assert nominal["regret_s"].max() < 8.0                         # close to the oracle optimum
    gains = nominal.filter(like="gain_vs_")
    assert (gains.mean() > 0).all()                                # beats each baseline on average

    res = optimize(RaceState("Monza", "CHR"), pace)
    text = explain(res, pace)
    assert text[0].startswith("Recommended") and len(text) >= 4
    unknown = LearnedPaceModel(LapTimeModel("x", pace.laptime.pipeline, {}), pace.pit_losses)
    with pytest.raises(ValueError, match="no training data"):
        optimize(RaceState("Sakhir", "CHR"), unknown)
