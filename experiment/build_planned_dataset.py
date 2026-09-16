#!/usr/bin/env python3
"""Round 12 dataset: every day drawn from the battery's habit, not repeated.

`experiment/build_dataset.py` holds trip length, ambient and accessory load
fixed for a battery's whole life, which is why rounds 4-11 could only ever ask
"drive or park". This builds the same kind of table from the same recording,
but with a per-day `DayPlan` behind every row (`experiment/schedule.py`), so a
battery's own life varies: median within-battery ambient span is ~28 C, and
driving minutes take hundreds of distinct values instead of two.

Between-battery identity is kept -- each battery draws a habit (its habitual
trip length, trip count, accessory load, climate and seasonal amplitude) and
every day is drawn from it.

Run: `python3 -m experiment.build_planned_dataset runs/telemetry.csv --out runs/experiment_planned`
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
from experiment.build_dataset import (
    AMBIENTS_C,
    DAY_COLUMNS,
    HORIZON_DAYS,
    LAYUP_DAYS_RANGE,
    LAYUP_GAP_DAYS_RANGE,
    PARASITIC_A_RANGE,
    SAMPLING_SEED,
    SCENARIOS_PER_AMBIENT,
    SOAK_H,
    _git_commit,
)
from experiment.calendar_sim import (
    ACTION_FEATURES,
    HIDDEN_FIELDS,
    OBSERVABLE_FEATURES,
    build_schedule,
    run_planned_scenario,
)
from experiment.schedule import (
    ACCESSORY_RANGE_A,
    AMBIENT_CLAMP_C,
    AMPLITUDE_RANGE_C,
    MAX_TRIPS,
    TRIP_MINUTES_RANGE,
    draw_habit,
    plan_days,
)
from sweep_dataset import load_driving


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--out", default="runs/experiment_planned")
    parser.add_argument("--scenarios-per-ambient", type=int, default=SCENARIOS_PER_AMBIENT)
    parser.add_argument("--horizon-days", type=int, default=HORIZON_DAYS)
    args = parser.parse_args(argv)

    driving, _dropouts = load_driving(Path(args.trajectory))
    rates = AgingRates()
    rng = random.Random(SAMPLING_SEED)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    scenarios_meta: list[dict] = []
    index = 0
    total = len(AMBIENTS_C) * args.scenarios_per_ambient
    started = time.time()

    for annual_mean_c in AMBIENTS_C:
        for _ in range(args.scenarios_per_ambient):
            # The layup pattern still comes from the round 4-11 generator, so
            # long parked stretches survive; everything else is now per-day.
            layup_days = rng.uniform(*LAYUP_DAYS_RANGE)
            layup_gap_days = rng.uniform(*LAYUP_GAP_DAYS_RANGE)
            parasitic_a = rng.uniform(*PARASITIC_A_RANGE)
            habit = draw_habit(rng, annual_mean_c=annual_mean_c)
            pattern = build_schedule(layup_gap_days, layup_days, args.horizon_days)
            plans = plan_days(habit, pattern, rng)
            name = f"amb{annual_mean_c:g}_s{index:04d}"

            t0 = time.time()
            day_rows = run_planned_scenario(
                driving, plans, rates, SOAK_H, seed=index, parasitic_a=parasitic_a,
            )
            for row in day_rows:
                rows.append({"scenario": name, **row})

            driving_days = [p for p in plans if p.trips]
            scenarios_meta.append({
                "scenario": name, "ambient_c": annual_mean_c,
                "annual_mean_c": annual_mean_c, "amplitude_c": habit.amplitude_c,
                "phase_day": habit.phase_day, "daily_noise_c": habit.daily_noise_c,
                "trip_minutes": habit.trip_minutes_mean,
                "trip_minutes_spread": habit.trip_minutes_spread,
                "trips_mean": habit.trips_mean,
                "accessory_mean_a": habit.accessory_mean_a,
                "accessory_spread_a": habit.accessory_spread_a,
                "layup_days": layup_days, "layup_gap_days": layup_gap_days,
                "parasitic_a": parasitic_a, "seed": index,
            })
            index += 1
            print(
                f"[{index:3d}/{total}] {name:20s} mean={annual_mean_c:g}C "
                f"±{habit.amplitude_c:.1f} trip~{habit.trip_minutes_mean:.1f}m "
                f"trips~{habit.trips_mean:.1f} acc~{habit.accessory_mean_a:.0f}A "
                f"drivedays={len(driving_days)} took {time.time() - t0:.1f}s "
                f"final_crystal={day_rows[-1]['crystal']:.4f}",
                flush=True,
            )

    with (out / "daily_trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(DAY_COLUMNS)
        for row in rows:
            writer.writerow([row[c] for c in DAY_COLUMNS])

    (out / "scenarios.json").write_text(json.dumps(scenarios_meta, indent=2))

    sidecar = {
        "git_commit": _git_commit(),
        "trajectory": str(args.trajectory),
        "horizon_days": args.horizon_days,
        "scenarios": len(scenarios_meta),
        "day_rows": len(rows),
        "sampling_seed": SAMPLING_SEED,
        "per_day_actions": True,
        "axes": {
            "annual_mean_c": list(AMBIENTS_C),
            "scenarios_per_ambient": args.scenarios_per_ambient,
            "seasonal_amplitude_c": list(AMPLITUDE_RANGE_C),
            "ambient_clamp_c": list(AMBIENT_CLAMP_C),
            "trip_minutes_range": list(TRIP_MINUTES_RANGE),
            "accessory_range_a": list(ACCESSORY_RANGE_A),
            "max_trips_per_day": MAX_TRIPS,
            "layup_days_range": list(LAYUP_DAYS_RANGE),
            "layup_gap_days_range": list(LAYUP_GAP_DAYS_RANGE),
            "parasitic_a_range": list(PARASITIC_A_RANGE),
            "soak_h": SOAK_H,
        },
        "action_features": list(ACTION_FEATURES),
        "observable_features": list(OBSERVABLE_FEATURES),
        "hidden_fields": list(HIDDEN_FIELDS),
        "elapsed_s": time.time() - started,
        "note": (
            "Round 12. Every calendar day carries its own trip count, trip "
            "length, ambient and accessory load, drawn from a per-battery "
            "habit. Rounds 4-11 held all four fixed for life, so the only "
            "expressible counterfactual was drive-or-park. Same single "
            "recording as every other round: trip length truncates it, "
            "nothing new was recorded."
        ),
    }
    (out / "daily_dataset.json").write_text(json.dumps(sidecar, indent=2))

    print(f"\nwrote {out}/daily_trajectory.csv  ({len(rows)} day-rows)")
    print(f"wrote {out}/scenarios.json        ({len(scenarios_meta)} scenarios)")
    print(f"elapsed {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
