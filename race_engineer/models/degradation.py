"""Tyre degradation curves estimated from race laps.

Per circuit, an OLS model explains lap time with:
  * a fixed effect per (session, driver)        -> car/driver/day pace
  * a piecewise-linear curve in lap number      -> "race progression"
  * drive-mode offsets
  * compound offsets and a piecewise-linear curve in tyre age per compound
    (knots every 8 laps, so a late "cliff" does not distort the early slope)

Fuel burn-off and track evolution both make the car faster as the race goes on
and are nearly collinear with lap number, so they are *not* separated: the
lap-number coefficient is their combined effect. Tyre age is identifiable
because it resets at every pit stop while race progression does not.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from race_engineer.config import COMPOUND_NAMES
from race_engineer.features.laps import clean_laps


KNOTS = (0.0, 8.0, 16.0, 24.0, 32.0)
LAP_KNOTS = (0.0, 8.0, 16.0, 30.0)


def _basis(age: np.ndarray, knots=KNOTS) -> np.ndarray:
    """Hinge functions max(age - knot, 0): a continuous piecewise-linear curve through 0 at age 0."""
    return np.clip(np.asarray(age, dtype=float)[:, None] - np.array(knots)[None, :], 0.0, None)


@dataclass
class DegradationModel:
    offsets: dict[str, dict[str, float]]            # circuit -> compound -> fresh pace vs MEDIUM
    hinges: dict[str, dict[str, list[float]]]       # circuit -> compound -> hinge coefficients
    max_age: dict[str, dict[str, float]]            # oldest tyre observed (curve is not valid beyond it)
    progression_s_per_lap: dict[str, float]         # average over the race distance

    def curve(self, circuit: str, compound: str, ages) -> np.ndarray:
        """Lap-time delta vs a fresh MEDIUM at each tyre age."""
        ages = np.atleast_1d(np.asarray(ages, dtype=float))
        return self.offsets[circuit][compound] + _basis(ages) @ np.array(self.hinges[circuit][compound])

    def table(self) -> pd.DataFrame:
        rows = []
        for c, comps in self.offsets.items():
            for k, off in comps.items():
                wear = self.curve(c, k, [0, 10, 20]) - off
                rows.append({"circuit": c, "compound": k, "fresh_offset_s": off, "deg_s_per_lap_first10": wear[1] / 10,
                             "loss_at_10_laps_s": wear[1], "loss_at_20_laps_s": wear[2],
                             "max_observed_age": self.max_age[c][k]})
        return pd.DataFrame(rows)


def fit_degradation(laps: pd.DataFrame) -> DegradationModel:
    data = clean_laps(laps)
    offsets, hinges, max_age, progression = {}, {}, {}, {}
    for circuit, g in data.groupby("circuit"):
        present = [c for c in COMPOUND_NAMES if (g["compound"] == c).sum() >= 20]
        age = g["tyre_age"].to_numpy(dtype=float)
        fe = pd.get_dummies(g["session_id"] + "|" + g["driver"], dtype=float)
        lap_basis = _basis(g["lap_number"].to_numpy(), LAP_KNOTS)
        cols, names = [fe.to_numpy(), lap_basis], [*fe.columns, *(f"lap{i}" for i in range(len(LAP_KNOTS)))]
        for mode in ("push", "conserve"):
            cols.append((g["drive_mode"] == mode).to_numpy(dtype=float)[:, None]); names.append(f"mode_{mode}")
        for c in present:
            is_c = (g["compound"] == c).to_numpy(dtype=float)[:, None]
            if c != "MEDIUM" and "MEDIUM" in present:
                cols.append(is_c); names.append(f"off_{c}")
            cols.append(is_c * _basis(age)); names += [f"h{i}_{c}" for i in range(len(KNOTS))]
        beta, *_ = np.linalg.lstsq(np.hstack(cols), g["lap_time_s"].to_numpy(), rcond=None)
        b = dict(zip(names, beta))
        offsets[circuit] = {c: float(b.get(f"off_{c}", 0.0)) for c in present}
        hinges[circuit] = {c: [float(b[f"h{i}_{c}"]) for i in range(len(KNOTS))] for c in present}
        max_age[circuit] = {c: float(age[(g["compound"] == c).to_numpy()].max()) for c in present}
        ends = _basis(np.array([1.0, g["total_laps"].iloc[0]]), LAP_KNOTS) @ np.array([b[f"lap{i}"] for i in range(len(LAP_KNOTS))])
        progression[circuit] = float((ends[1] - ends[0]) / (g["total_laps"].iloc[0] - 1))
    return DegradationModel(offsets, hinges, max_age, progression)
