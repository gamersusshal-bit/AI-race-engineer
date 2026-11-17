"""Static domain configuration.

The circuit, compound and driver parameters below are *simulation assumptions*
used by the synthetic data generator and the physics reference model. They are
loosely inspired by real venues but are not measured values.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Compound:
    name: str
    pace_offset_s: float   # pace vs MEDIUM on fresh tyres
    deg_s_per_lap: float   # linear degradation per unit of wear
    cliff_wear: float      # wear at which the tyre starts to fall off
    cliff_quad: float      # quadratic penalty past the cliff
    max_stint_laps: int    # rule-of-thumb tyre life limit used as a constraint


@dataclass(frozen=True)
class DriveMode:
    name: str
    pace_delta_s: float
    wear_mult: float
    fuel_mult: float
    lift_quantile: float   # telemetry: speed quantile above which the driver lifts


@dataclass(frozen=True)
class Circuit:
    name: str
    total_laps: int
    base_lap_s: float            # fresh MEDIUM, zero fuel, rubbered-in track
    pit_loss_s: float
    fuel_per_lap_kg: float
    fuel_effect_s_per_kg: float
    deg_mult: float              # tyre abrasiveness
    track_evo_s: float           # total lap-time gain from track evolution
    length_m: float
    n_corners: int


COMPOUNDS: dict[str, Compound] = {
    "SOFT": Compound("SOFT", -0.60, 0.080, 16.0, 0.012, 28),
    "MEDIUM": Compound("MEDIUM", 0.00, 0.045, 26.0, 0.009, 40),
    "HARD": Compound("HARD", 0.50, 0.025, 38.0, 0.007, 55),
}
COMPOUND_NAMES = tuple(COMPOUNDS)

MODES: dict[str, DriveMode] = {
    "push": DriveMode("push", -0.25, 1.35, 1.05, 1.00),
    "normal": DriveMode("normal", 0.00, 1.00, 1.00, 0.97),
    "conserve": DriveMode("conserve", 0.35, 0.70, 0.94, 0.86),
}
MODE_NAMES = tuple(MODES)

CIRCUITS: dict[str, Circuit] = {
    c.name: c
    for c in [
        Circuit("Sakhir", 57, 91.5, 22.5, 1.80, 0.032, 1.50, 1.2, 5412, 15),
        Circuit("Monza", 53, 80.8, 24.0, 1.90, 0.028, 0.70, 0.8, 5793, 11),
        Circuit("Silverstone", 52, 87.6, 20.5, 2.00, 0.031, 1.20, 1.0, 5891, 18),
        Circuit("Barcelona", 66, 75.2, 21.5, 1.55, 0.033, 1.30, 1.1, 4657, 14),
        Circuit("Spa", 44, 104.9, 19.0, 2.40, 0.030, 1.00, 1.4, 7004, 19),
        Circuit("Suzuka", 53, 90.3, 22.0, 1.95, 0.031, 1.15, 0.9, 5807, 18),
    ]
}

# Fictional drivers: pace offset in seconds per lap relative to the field reference.
DRIVERS: dict[str, float] = {
    "ALF": -0.30, "BRV": -0.18, "CHR": -0.05, "DLT": 0.04,
    "ECO": 0.12, "FOX": 0.22, "GLF": 0.35, "HTL": 0.50,
}

REFERENCE_TRACK_TEMP_C = 35.0
FUEL_MARGIN_KG = 1.0
MIN_DISTINCT_COMPOUNDS = 2   # dry-race rule: at least two different compounds
