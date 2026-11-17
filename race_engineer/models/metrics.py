from __future__ import annotations

import numpy as np


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": float(1 - (err**2).sum() / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(err)),
    }
