"""Plain-language reasons behind a strategy recommendation."""
from __future__ import annotations

from race_engineer.strategy.optimizer import StrategyResult


def explain(result: StrategyResult, pace_model) -> list[str]:
    s, best, state = result.summary, result.best, result.best_state
    loss = pace_model.pit_loss(state.circuit)
    laps = result.laps[result.laps["plan_rank"] == 1]
    lines = [f"Recommended: {s['strategy'].iloc[0]} — predicted race time "
             f"{s['total_time_s'].iloc[0]:.1f} s over {len(laps)} laps."]
    if best.n_stops:
        lines.append(f"{best.n_stops} stop(s) cost about {best.n_stops * loss:.1f} s in the pit lane "
                     f"({loss:.1f} s each at {state.circuit}).")
    for k, g in laps.groupby("stint"):
        clean = g["clean_lap_s"]
        lines.append(f"Stint {k} ({g['compound'].iloc[0]}, {len(g)} laps, {g['drive_mode'].iloc[0]}): pace goes from "
                     f"{clean.iloc[0]:.2f} s to {clean.iloc[-1]:.2f} s; tyres end at {int(g['tyre_age'].iloc[-1]) + 1} laps old.")
    for stops in sorted(set(s["stops"]) - {best.n_stops}):
        alt = s[s["stops"] == stops].iloc[0]
        lines.append(f"Best {stops}-stop alternative ({alt['strategy']}) is {alt['gap_to_best_s']:.1f} s slower.")
    if len(s) > 1:
        near = int((s["gap_to_best_s"] <= 1.0).sum()) - 1
        lines.append(f"{near} other plan(s) are within 1.0 s of the best — "
                     + ("the call is robust to small timing changes." if near >= 3 else "the window is narrow."))
    resid = getattr(getattr(pace_model, "laptime", None), "resid_std", None)
    if resid:
        lines.append(f"Model lap-time error is about ±{resid:.2f} s per lap (1σ on unseen sessions), so gaps "
                     f"smaller than ~{resid * len(laps) ** 0.5:.1f} s over the race are not decisive.")
    return lines
