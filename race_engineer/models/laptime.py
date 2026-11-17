"""Lap-time prediction from pre-lap state.

Models predict the *delta* to a per-circuit reference lap (median clean lap in
the training data), so R² measures skill at explaining within-circuit variation
instead of being inflated by the trivial difference between circuits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from race_engineer.features.laps import (
    CATEGORICAL_FEATURES, PRE_LAP_FEATURES, clean_laps, make_pre_lap_features,
)
from race_engineer.models.metrics import regression_metrics

TARGET = "lap_time_s"


def _estimators(seed: int) -> dict:
    est = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, random_state=seed),
    }
    try:
        from xgboost import XGBRegressor
        est["xgboost"] = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8,
                                      colsample_bytree=0.8, random_state=seed, n_jobs=-1)
    except ImportError:  # optional dependency
        pass
    return est


def _pipeline(estimator) -> Pipeline:
    numeric = [f for f in PRE_LAP_FEATURES if f not in CATEGORICAL_FEATURES]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", "passthrough", numeric),
    ])
    return Pipeline([("pre", pre), ("model", estimator)])


@dataclass
class LapTimeModel:
    name: str
    pipeline: Pipeline
    circuit_ref: dict[str, float]
    resid_std: float = 0.0

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        ref = df["circuit"].map(self.circuit_ref)
        if ref.isna().any():
            unknown = sorted(df.loc[ref.isna(), "circuit"].unique())
            raise ValueError(f"no training data for circuit(s) {unknown}")
        return ref.to_numpy() + self.pipeline.predict(make_pre_lap_features(df))


def split_by_season(laps: pd.DataFrame, test_seasons=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Session-aware holdout: whole sessions of the latest season(s) are unseen in training."""
    test_seasons = [laps["season"].max()] if test_seasons is None else list(test_seasons)
    is_test = laps["season"].isin(test_seasons)
    train, test = laps[~is_test], laps[is_test]
    assert not set(train["session_id"]) & set(test["session_id"]), "session leaked across the split"
    return train, test


def _fit(name: str, estimator, train: pd.DataFrame, ref: dict[str, float]) -> LapTimeModel:
    y = train[TARGET] - train["circuit"].map(ref)
    pipe = _pipeline(estimator).fit(make_pre_lap_features(train), y)
    return LapTimeModel(name, pipe, ref)


def train_laptime_models(laps: pd.DataFrame, seed: int = 0, cv_folds: int = 4) -> dict:
    """Train baselines and candidate models; select by grouped CV on the training set only."""
    data = clean_laps(laps)
    train, test = split_by_season(data)
    ref = train.groupby("circuit")[TARGET].median().to_dict()
    y_test_delta = test[TARGET] - test["circuit"].map(ref)

    results = {}
    base = train.assign(d=train[TARGET] - train["circuit"].map(ref))
    by_compound = base.groupby(["circuit", "compound"])["d"].median()
    idx = pd.MultiIndex.from_frame(test[["circuit", "compound"]])
    results["baseline_circuit_median"] = {"test": regression_metrics(y_test_delta, np.zeros(len(test)))}
    results["baseline_circuit_compound_median"] = {
        "test": regression_metrics(y_test_delta, by_compound.reindex(idx).fillna(0.0).to_numpy())}

    models = {}
    folds = list(GroupKFold(n_splits=cv_folds).split(train, groups=train["session_id"]))
    for name, est in _estimators(seed).items():
        cv_mae = []
        for tr, va in folds:
            fold_train, fold_val = train.iloc[tr], train.iloc[va]
            fold_ref = fold_train.groupby("circuit")[TARGET].median().to_dict()
            m = _fit(name, est, fold_train, fold_ref)
            cv_mae.append(np.abs(m.predict(fold_val) - fold_val[TARGET]).mean())
        model = _fit(name, est, train, ref)
        pred = model.predict(test)
        model.resid_std = float(np.std(pred - test[TARGET]))
        models[name] = model
        results[name] = {"cv_mae": float(np.mean(cv_mae)),
                         "test": regression_metrics(y_test_delta, pred - test["circuit"].map(ref))}

    best = min(models, key=lambda n: results[n]["cv_mae"])
    preds = test[["session_id", "circuit", "driver", "lap_number", "compound", "tyre_age", "drive_mode", TARGET]].copy()
    preds["predicted_s"] = models[best].predict(test)
    preds["error_s"] = preds["predicted_s"] - preds[TARGET]
    breakdown = {
        dim: preds.groupby(dim)["error_s"].agg(mae=lambda e: e.abs().mean(), bias="mean", n="size")
             .round(4).reset_index().to_dict("records")
        for dim in ("circuit", "compound", "drive_mode")
    }
    return {
        "best": best, "model": models[best], "models": models, "results": results,
        "test_predictions": preds, "error_breakdown": breakdown, "features": PRE_LAP_FEATURES,
        "train_sessions": sorted(train["session_id"].unique()),
        "test_sessions": sorted(test["session_id"].unique()),
    }
