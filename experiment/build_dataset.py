#!/usr/bin/env python3
"""Step 1a: run the literal calendar-day simulation and write the raw table.

One row per scenario per calendar day, covering `ACTION_FEATURES`,
`OBSERVABLE_FEATURES`, `HIDDEN_FIELDS`, and `RESUME_FIELDS` from
`experiment/calendar_sim.py`. This is the raw material `experiment/windows.py`
samples observation windows from -- see that module and the experiment
report for the windowing and target-flattening step.

Scenario axes are sampled the same way `sweep_dataset.py`'s continuous sweep
was (uniform draws of trip length, layup length, layup gap, and parasitic
draw, crossed against a discrete ambient grid), with an independent sampling
seed, so this dataset is a fresh, self-contained draw rather than a re-read
of `runs/sweep_cont/`.

Run: `python3 -m experiment.build_dataset runs/telemetry.csv --out runs/experiment`
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from cell.aging import AgingRates
from experiment.calendar_sim import (
    ACTION_FEATURES,
    HIDDEN_FIELDS,
    OBSERVABLE_FEATURES,
    RESUME_FIELDS,
    run_calendar_scenario,
)
from sweep_dataset import _git_commit, load_driving

#: Same discrete grid as `sweep_dataset.py`: the dominant ageing axis
#: (ambient) stays gridded, the sulfation-relevant axes stay continuous.
AMBIENTS_C = (-10.0, 10.0, 25.0, 42.0)

#: 20 scenarios per ambient x 4 ambients = 80 scenarios total. At ~7 ms/day
#: (benchmarked) x 2000 days that is ~14 s/scenario, ~19 min total -- see the
#: experiment report for the actual measured elapsed time.
SCENARIOS_PER_AMBIENT = 20

#: Long enough to comfortably cover the measured 540-1170 day (median 630)
#: crystal transition band with margin on both sides, short enough to keep
#: the sweep to the order of ten minutes rather than hours.
HORIZON_DAYS = 2000

#: Independent of `sweep_dataset.py`'s own `SCENARIO_SAMPLING_SEED` -- this is
#: a fresh, self-contained draw, not a re-read of the existing sweep.
SAMPLING_SEED = 42

TRIP_MINUTES_RANGE = (1.5, 20.0)
LAYUP_DAYS_RANGE = (0.0, 30.0)
LAYUP_GAP_DAYS_RANGE = (20.0, 120.0)
PARASITIC_A_RANGE = (0.02, 0.10)
TRIPS_PER_DAY = 2.0
SOAK_H = 8.0

def draw_scenario_params(rng: random.Random) -> dict[str, float]:
    """One scenario's sampled axes, in the fixed draw order the dataset depends on.

    Shared with `experiment/add_ambient.py`, which replays these draws to
    append scenarios for a new climate without disturbing existing ones --
    so the ORDER of the four uniforms here must never change.
    """
    return {
        "trip_minutes": rng.uniform(*TRIP_MINUTES_RANGE),
        "layup_days": rng.uniform(*LAYUP_DAYS_RANGE),
        "layup_gap_days": rng.uniform(*LAYUP_GAP_DAYS_RANGE),
        "parasitic_a": rng.uniform(*PARASITIC_A_RANGE),
    }


DAY_COLUMNS = (
    ("scenario", "day")
    + ACTION_FEATURES
    + OBSERVABLE_FEATURES
    + HIDDEN_FIELDS
    + RESUME_FIELDS
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--out", default="runs/experiment")
    parser.add_argument("--scenarios-per-ambient", type=int, default=SCENARIOS_PER_AMBIENT)
    parser.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    args = parser.parse_args(argv)

    driving, dropouts = load_driving(Path(args.trajectory))
    rates = AgingRates()
    rng = random.Random(SAMPLING_SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    scenarios_meta: list[dict] = []
    index = 0
    total = len(AMBIENTS_C) * args.scenarios_per_ambient
    started = time.time()

    for ambient_c in AMBIENTS_C:
        for _ in range(args.scenarios_per_ambient):
            params = draw_scenario_params(rng)
            trip_minutes = params["trip_minutes"]
            layup_days = params["layup_days"]
            layup_gap_days = params["layup_gap_days"]
            parasitic_a = params["parasitic_a"]
            name = f"amb{ambient_c:g}_s{index:04d}"

            t0 = time.time()
            day_rows = run_calendar_scenario(
                driving, ambient_c, trip_minutes, layup_days, layup_gap_days,
                parasitic_a, TRIPS_PER_DAY, SOAK_H, args.horizon_days, rates,
                seed=index,
            )
            for r in day_rows:
                rows.append({"scenario": name, **r})
            scenarios_meta.append({
                "scenario": name, "ambient_c": ambient_c,
                "trip_minutes": trip_minutes, "layup_days": layup_days,
                "layup_gap_days": layup_gap_days, "parasitic_a": parasitic_a,
                "seed": index,
            })
            index += 1
            print(
                f"[{index:3d}/{total}] {name:20s} amb={ambient_c:g} "
                f"trip={trip_minutes:.2f}m layup={layup_days:.1f}/{layup_gap_days:.1f}d "
                f"par={parasitic_a:.3f}A took {time.time() - t0:.1f}s "
                f"final_crystal={day_rows[-1]['crystal']:.4f}",
                flush=True,
            )

    with (out / "daily_trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(DAY_COLUMNS)
        for row in rows:
            writer.writerow([row[c] for c in DAY_COLUMNS])

    with (out / "scenarios.json").open("w") as handle:
        json.dump(scenarios_meta, handle, indent=2)

    sidecar = {
        "git_commit": _git_commit(),
        "trajectory": str(args.trajectory),
        "horizon_days": args.horizon_days,
        "scenarios": len(scenarios_meta),
        "day_rows": len(rows),
        "sampling_seed": SAMPLING_SEED,
        "axes": {
            "ambient_c": list(AMBIENTS_C),
            "scenarios_per_ambient": args.scenarios_per_ambient,
            "trip_minutes_range": list(TRIP_MINUTES_RANGE),
            "layup_days_range": list(LAYUP_DAYS_RANGE),
            "layup_gap_days_range": list(LAYUP_GAP_DAYS_RANGE),
            "parasitic_a_range": list(PARASITIC_A_RANGE),
            "trips_per_day": TRIPS_PER_DAY, "soak_h": SOAK_H,
        },
        "action_features": list(ACTION_FEATURES),
        "observable_features": list(OBSERVABLE_FEATURES),
        "hidden_fields": list(HIDDEN_FIELDS),
        "elapsed_s": time.time() - started,
        "note": (
            "Literal day-by-day calendar simulation -- NOT cell.life.project()'s "
            "period-averaged damage. See experiment/calendar_sim.py's module "
            "docstring for why: this dataset needs the real daily observable "
            "pattern (trip count, driving minutes, voltage/current/temperature), "
            "which the period-average abstraction destroys."
        ),
    }
    (out / "daily_dataset.json").write_text(json.dumps(sidecar, indent=2))

    print(f"\nwrote {out}/daily_trajectory.csv  ({len(rows)} day-rows)")
    print(f"wrote {out}/scenarios.json        ({len(scenarios_meta)} scenarios)")
    print(f"wrote {out}/daily_dataset.json")
    print(f"elapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
