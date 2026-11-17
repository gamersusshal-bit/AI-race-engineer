"""Command line: ``python -m race_engineer.cli {pipeline,strategy,call}``."""
from __future__ import annotations

import argparse

import pandas as pd

from race_engineer import pipeline
from race_engineer.config import CIRCUITS, COMPOUND_NAMES, DRIVERS
from race_engineer.sim.race import RaceState
from race_engineer.strategy.explain import explain
from race_engineer.strategy.optimizer import optimize, pit_now_or_stay_out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="race_engineer")
    p.add_argument("--artifacts", default=str(pipeline.ARTIFACTS))
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pipeline", help="generate data, train models, evaluate strategies").add_argument("--seed", type=int, default=7)
    for name in ("strategy", "call"):
        s = sub.add_parser(name, help="pre-race plan" if name == "strategy" else "pit now vs stay out")
        s.add_argument("--circuit", required=True, choices=list(CIRCUITS))
        s.add_argument("--driver", default="CHR", choices=list(DRIVERS))
        s.add_argument("--temp", type=float, default=35.0)
        if name == "strategy":
            s.add_argument("--modes", action="store_true", help="also choose push/normal/conserve per stint")
        else:
            s.add_argument("--lap", type=int, required=True)
            s.add_argument("--compound", required=True, choices=COMPOUND_NAMES)
            s.add_argument("--tyre-age", type=int, required=True)
            s.add_argument("--used", nargs="*", default=[], choices=COMPOUND_NAMES)
    a = p.parse_args(argv)
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60)

    if a.cmd == "pipeline":
        r = pipeline.run(a.artifacts, a.seed)
        m = r["metrics"]["laptime"]
        print(f"Selected lap-time model: {m['selected']}")
        print(pd.DataFrame({k: {"cv_mae": v.get("cv_mae"), **v["test"]} for k, v in m["models"].items()}).T.round(3))
        print("\nFuel per lap (test):")
        print(pd.DataFrame({k: v["test"] for k, v in r["metrics"]["fuel"].items()}).T.round(4))
        print("\nStrategy evaluation on ground truth (s):")
        print(r["strategy"].drop(columns=["oracle_strategy", "predicted_time_s"]).round(2).to_string(index=False))
        return

    pace = pipeline.load_pace(a.artifacts)
    if a.cmd == "strategy":
        res = optimize(RaceState(a.circuit, a.driver, track_temp=a.temp), pace, optimize_modes=a.modes)
        print(res.summary.head(8).round(2).to_string(index=False), "\n")
        print("\n".join("- " + line for line in explain(res, pace)))
    else:
        state = RaceState(a.circuit, a.driver, a.lap, a.compound, a.tyre_age,
                          compounds_used=tuple({*a.used, a.compound}), track_temp=a.temp)
        d = pit_now_or_stay_out(state, pace)
        print(f"CALL: {d['call']}" + (f"  (pit-now minus stay-out = {d['delta_s']:+.2f} s)" if d["delta_s"] is not None else ""))
        for k in ("pit_now", "stay_out"):
            if d[k]:
                print(f"  best {k.replace('_', ' ')}: {d[k]['strategy']}  -> {d[k]['total_time_s']:.1f} s")


if __name__ == "__main__":
    main()
