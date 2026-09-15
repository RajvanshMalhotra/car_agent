#!/usr/bin/env python3
"""Round 9: append batteries at a new climate to an existing calendar dataset.

`experiment/build_dataset.py` drew every scenario's axes from one seeded
generator, ambient by ambient. This script replays those draws for the
existing `len(AMBIENTS_C) * SCENARIOS_PER_AMBIENT` scenarios and continues the
sequence for `--count` new ones at `--ambient`, so:

- the existing scenarios, and every row already written for them, are
  untouched (the source CSV is copied byte for byte), and
- the new scenarios are exactly what build_dataset.py would have produced had
  the new ambient been appended to `AMBIENTS_C`.

New rows are simulated with the same `run_calendar_scenario` call, trip
recording and constants, and appended in `DAY_COLUMNS` order.

Run: `python3 -m experiment.add_ambient runs/telemetry.csv --src runs/experiment
      --out runs/experiment_48_clean --ambient 48 --count 20`
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from pathlib import Path

from cell.aging import AgingRates
from experiment.build_dataset import (
    AMBIENTS_C,
    DAY_COLUMNS,
    HORIZON_DAYS,
    SAMPLING_SEED,
    SCENARIOS_PER_AMBIENT,
    SOAK_H,
    TRIPS_PER_DAY,
    draw_scenario_params,
)
from experiment.calendar_sim import run_calendar_scenario
from sweep_dataset import load_driving


def extension_params(ambient_c: float, count: int) -> list[dict]:
    """Scenario metadata for `count` new batteries at `ambient_c`, continuing the seeded draws."""
    if ambient_c in AMBIENTS_C:
        raise ValueError(f"the dataset already has ambient {ambient_c:g} C")
    rng = random.Random(SAMPLING_SEED)
    existing = len(AMBIENTS_C) * SCENARIOS_PER_AMBIENT
    for _ in range(existing):
        draw_scenario_params(rng)
    out = []
    for offset in range(count):
        index = existing + offset
        out.append({
            "scenario": f"amb{ambient_c:g}_s{index:04d}", "ambient_c": ambient_c,
            **draw_scenario_params(rng), "seed": index,
        })
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--src", default="runs/experiment")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ambient", type=float, required=True)
    parser.add_argument("--count", type=int, default=SCENARIOS_PER_AMBIENT)
    args = parser.parse_args(argv)
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((src / "scenarios.json").read_text())
    expected = len(AMBIENTS_C) * SCENARIOS_PER_AMBIENT
    if len(meta) != expected:
        raise ValueError(f"{src} has {len(meta)} scenarios; the replay assumes {expected}")

    new = extension_params(args.ambient, args.count)
    driving, _dropouts = load_driving(Path(args.trajectory))
    rates = AgingRates()

    shutil.copyfile(src / "daily_trajectory.csv", out / "daily_trajectory.csv")
    started = time.time()
    with (out / "daily_trajectory.csv").open("a", newline="") as handle:
        writer = csv.writer(handle)
        for n, scenario in enumerate(new, start=1):
            t0 = time.time()
            rows = run_calendar_scenario(
                driving, scenario["ambient_c"], scenario["trip_minutes"], scenario["layup_days"],
                scenario["layup_gap_days"], scenario["parasitic_a"], TRIPS_PER_DAY, SOAK_H,
                HORIZON_DAYS, rates, seed=scenario["seed"],
            )
            for row in rows:
                writer.writerow([scenario["scenario"] if c == "scenario" else row[c] for c in DAY_COLUMNS])
            handle.flush()
            print(f"[{n:2d}/{len(new)}] {scenario['scenario']} took {time.time() - t0:.1f}s "
                  f"final_crystal={rows[-1]['crystal']:.4f}", flush=True)

    (out / "scenarios.json").write_text(json.dumps(meta + new, indent=2))
    sidecar = json.loads((src / "daily_dataset.json").read_text())
    sidecar["scenarios"] = len(meta) + len(new)
    sidecar["day_rows"] = sidecar["scenarios"] * HORIZON_DAYS
    sidecar["axes"]["ambient_c"] = [*AMBIENTS_C, args.ambient]
    sidecar["extended"] = {
        "from": str(src), "ambient_c": args.ambient, "count": len(new),
        "note": "existing rows copied byte for byte; new scenarios continue the seeded draw sequence",
        "elapsed_s": time.time() - started,
    }
    (out / "daily_dataset.json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out} ({sidecar['scenarios']} scenarios)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
