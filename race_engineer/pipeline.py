"""End-to-end pipeline: data -> validation -> models -> strategy evaluation -> artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from race_engineer.data.schema import validate_laps
from race_engineer.data.synthetic import generate_dataset
from race_engineer.features.laps import PRE_LAP_FEATURES
from race_engineer.models.degradation import fit_degradation
from race_engineer.models.fuel import FUEL_FEATURES, train_fuel_model
from race_engineer.models.laptime import split_by_season, train_laptime_models
from race_engineer.models.pace import LearnedPaceModel, estimate_pit_loss
from race_engineer.models.registry import data_fingerprint, load_model, save_model
from race_engineer.strategy.evaluate import evaluate_strategies

ARTIFACTS = Path("artifacts")


def run(out: Path = ARTIFACTS, seed: int = 7, laps: pd.DataFrame | None = None) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = "synthetic" if laps is None else "provided"
    raw = generate_dataset(seed=seed) if laps is None else laps
    laps, report = validate_laps(raw)
    laps.to_parquet(out / "laps.parquet", index=False)
    fingerprint = data_fingerprint(laps)

    lap = train_laptime_models(laps, seed=seed)
    train, _ = split_by_season(laps)
    has_fuel = "fuel_used_kg" in laps and laps["fuel_used_kg"].notna().any()
    fuel = train_fuel_model(laps) if has_fuel else None
    deg = fit_degradation(train)
    pace = LearnedPaceModel(lap["model"], estimate_pit_loss(train, lap["model"]), fuel["model"] if fuel else None)

    meta = {"data_fingerprint": fingerprint, "data_source": source, "seed": seed,
            "train_sessions": lap["train_sessions"], "test_sessions": lap["test_sessions"]}
    save_model(pace, "pace", out / "models", {**meta, "estimator": lap["best"], "features": PRE_LAP_FEATURES,
                                              "metrics": lap["results"][lap["best"]]})
    save_model(deg, "degradation", out / "models", meta)
    if fuel:
        save_model(fuel["model"], "fuel", out / "models", {**meta, "features": FUEL_FEATURES,
                                                           "metrics": fuel["results"]["telemetry_linear"]})

    lap["test_predictions"].to_parquet(out / "laptime_test_predictions.parquet", index=False)
    deg.table().to_csv(out / "degradation.csv", index=False)
    strategy = evaluate_strategies(pace) if source == "synthetic" else None
    if strategy is not None:
        strategy.to_csv(out / "strategy_eval.csv", index=False)

    metrics = {
        "data": {"source": source, "fingerprint": fingerprint, "validation": report.as_dict()},
        "laptime": {"selected": lap["best"], "selection_rule": "lowest grouped-CV MAE on training sessions",
                    "models": lap["results"], "error_breakdown": lap["error_breakdown"]},
        "fuel": fuel["results"] if fuel else None,
        "pit_loss_s": pace.pit_losses,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"metrics": metrics, "strategy": strategy, "pace": pace}


def load_pace(out: Path = ARTIFACTS) -> LearnedPaceModel:
    return load_model("pace", Path(out) / "models")[0]
