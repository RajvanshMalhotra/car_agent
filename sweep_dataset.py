#!/usr/bin/env python3
"""Sweep duty cycles over a recorded trajectory and emit one combined dataset.

`battery_run.py` answers "what does THIS scenario do to the battery". This
answers "what does the space of scenarios do", which is what a model needs if
it is to generalise rather than memorise one operating point.

Two tables come out, for the same reason `cell/dataset.py` emits two: the 1 Hz
rows and the per-scenario life are different granularities, and broadcasting a
per-scenario constant across thousands of near-identical rows is recoverable
from the row index.

    within_trip.csv    1 Hz, every (ambient, accessories) combination, tagged
                       with `scenario`. The driving channels are identical
                       across scenarios by construction -- the same recording
                       is replayed -- so what varies is the battery response.
    scenario_life.csv  one row per full scenario: the declared schedule, the
                       damage it accumulated, and where end of life landed.

Absolute days remain unfitted. Ratios between rows are the usable output.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

from cell.aging import AgingRates, AgingState, soh
from cell.dataset import FEATURES, TARGETS
from cell.integrate import CellState, run_soak, run_trip
from cell.life import TripSchedule, ensemble
from load.legacy import LEGACY_PROVENANCE, read_legacy
from load.spec import BatteryScenario
from load.trips import segment

#: The dominant ageing axis, per the project's own measurements.
AMBIENTS_C = (10.0, 25.0, 35.0, 42.0, 50.0)

#: Trip frequency and the soak between trips. Each pair is one day's pattern.
SCHEDULES = ((1.0, 16.0), (2.0, 8.0), (4.0, 4.0), (8.0, 2.0))

#: Accessory state: nothing, versus blower and lights.
ACCESSORIES = ((0.0, False), (1.0, True))

WITHIN_COLUMNS = ("scenario", "ambient_c", "accessories", "trip", "t_s") + FEATURES + TARGETS

LIFE_COLUMNS = (
    "scenario", "ambient_c", "accessories", "trips_per_day", "soak_h",
    "corrosion_equivalent_h_per_day", "ah_throughput_per_day",
    "low_soc_hours_per_day", "vibration_dose_per_day",
    "soc_after_repeats", "soh_after_repeats",
    "eol_days", "eol_years", "interval_low_days", "interval_high_days",
    "scale_unfitted",
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def load_driving(path: Path):
    """The driving-phase samples at 1 Hz, dropouts already excluded."""
    trajectory = read_legacy(path)
    samples = list(trajectory.at_hz(1.0))
    segments = segment(iter(samples))
    driving = [
        s for s in samples
        if any(g.kind == "trip" and g.start_s <= s["t_s"] <= g.end_s for g in segments)
    ]
    return driving, trajectory.dropouts.spans


def run_scenario(driving, ambient_c, hvac, lights, trips_per_day, soak_h,
                 repeats, rates, draws, seed):
    """One scenario: integrate `repeats` trips, then project a life."""
    scenario = BatteryScenario(
        name=f"a{ambient_c:g}", ambient_c=ambient_c, hvac=hvac, lights=lights,
        seed=seed,
    )
    schedule = TripSchedule(trips_per_day=trips_per_day, soak_s=soak_h * 3600.0)
    last_coolant = float(driving[-1]["coolant_c"])

    state = CellState(temp_c=ambient_c)
    trips = []
    for index in range(repeats):
        result = run_trip(iter(driving), scenario, state, rates, cranked=index > 0)
        trips.append(result)
        if index < repeats - 1:
            run_soak(schedule.soak_s, scenario, state, rates,
                     initial_coolant_c=last_coolant)

    def day_damage(health):
        probe = CellState(soc=state.soc, temp_c=ambient_c, aging=state.aging.copy())
        trip = run_trip(iter(driving), scenario, probe, rates, cranked=True,
                        health=health).damage
        soak = run_soak(schedule.soak_s, scenario, probe, rates,
                        initial_coolant_c=last_coolant, health=health).damage
        return trip.scaled(trips_per_day) + soak.scaled(trips_per_day)

    life = ensemble(day_damage, rates, schedule, draws=draws, seed=seed)
    per_day = day_damage(soh(state.aging, rates))
    return scenario, schedule, trips, life, per_day, state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory")
    parser.add_argument("--out", default="runs/sweep")
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--one", action="store_true",
                        help="run a single scenario and report timing only")
    args = parser.parse_args(argv)

    driving, dropouts = load_driving(Path(args.trajectory))
    rates = AgingRates()
    print(f"driving samples at 1 Hz: {len(driving)}", flush=True)

    if args.one:
        start = time.time()
        run_scenario(driving, 25.0, 0.0, False, 2.0, 8.0, args.repeats, rates,
                     args.draws, 0)
        print(f"one scenario took {time.time() - start:.1f} s", flush=True)
        total = len(AMBIENTS_C) * len(SCHEDULES) * len(ACCESSORIES)
        print(f"full sweep is {total} scenarios "
              f"~= {(time.time() - start) * total / 60:.1f} min", flush=True)
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    within_rows, life_rows = [], []
    seen_within = set()
    index = 0
    started = time.time()

    for ambient_c in AMBIENTS_C:
        for hvac, lights in ACCESSORIES:
            for trips_per_day, soak_h in SCHEDULES:
                name = f"amb{ambient_c:g}_{'acc' if lights else 'bare'}_{trips_per_day:g}x{soak_h:g}h"
                scenario, schedule, trips, life, per_day, state = run_scenario(
                    driving, ambient_c, hvac, lights, trips_per_day, soak_h,
                    args.repeats, rates, args.draws, index,
                )
                health = soh(state.aging, rates)
                low, high = (life.interval_days or (None, None))
                life_rows.append({
                    "scenario": name, "ambient_c": ambient_c,
                    "accessories": int(lights), "trips_per_day": trips_per_day,
                    "soak_h": soak_h,
                    "corrosion_equivalent_h_per_day": per_day.corrosion_equivalent_h,
                    "ah_throughput_per_day": per_day.ah_throughput,
                    "low_soc_hours_per_day": per_day.low_soc_hours,
                    "vibration_dose_per_day": per_day.vibration_dose,
                    "soc_after_repeats": state.soc,
                    "soh_after_repeats": health,
                    "eol_days": "" if life.eol_days is None else life.eol_days,
                    "eol_years": "" if life.eol_days is None else round(life.eol_days / 365.0, 3),
                    "interval_low_days": "" if low is None else low,
                    "interval_high_days": "" if high is None else high,
                    "scale_unfitted": int(life.scale_unfitted),
                })

                # The 1 Hz rows depend only on ambient and accessories, so emit
                # them once per such pair rather than once per schedule.
                key = (ambient_c, lights)
                if key not in seen_within:
                    seen_within.add(key)
                    wname = f"amb{ambient_c:g}_{'acc' if lights else 'bare'}"
                    for trip_index, result in enumerate(trips):
                        for sample, step in zip(driving, result.steps):
                            row = {"scenario": wname, "ambient_c": ambient_c,
                                   "accessories": int(lights),
                                   "trip": trip_index, "t_s": step.t_s}
                            for column in FEATURES:
                                row[column] = (step.t_bay_c if column == "t_bay_c"
                                               else float(sample[column]))
                            row.update({
                                "i_bat_a": step.i_bat_a, "v_bat_v": step.v_bat_v,
                                "t_bat_c": step.t_bat_c, "soc": step.soc,
                                "r_int_ohm": step.r_int_ohm,
                            })
                            within_rows.append(row)
                index += 1
                print(f"  [{index:2d}] {name:34s} eol={life.eol_days} days", flush=True)

    for path, columns, rows in (
        (out / "within_trip.csv", WITHIN_COLUMNS, within_rows),
        (out / "scenario_life.csv", LIFE_COLUMNS, life_rows),
    ):
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                writer.writerow([row[c] for c in columns])

    reached = [r for r in life_rows if r["eol_days"] != ""]
    sidecar = {
        "git_commit": _git_commit(),
        "trajectory": str(args.trajectory),
        "repeats_per_scenario": args.repeats,
        "ensemble_draws": args.draws,
        "scenarios": len(life_rows),
        "within_trip_rows": len(within_rows),
        "dropouts_excluded": [
            {"start_s": s.start_s, "end_s": s.end_s, "rows": s.rows} for s in dropouts
        ],
        "axes": {
            "ambient_c": list(AMBIENTS_C),
            "schedules_trips_per_day_soak_h": [list(s) for s in SCHEDULES],
            "accessories": ["none", "blower+lights"],
        },
        "features": list(FEATURES),
        "targets": list(TARGETS),
        "life_spread": (
            {"min_days": min(r["eol_days"] for r in reached),
             "max_days": max(r["eol_days"] for r in reached),
             "ratio": round(max(r["eol_days"] for r in reached)
                            / min(r["eol_days"] for r in reached), 3)}
            if reached else None
        ),
        "ageing_rates": {
            "corrosion_eol_h": rates.corrosion_eol_h,
            "corrosion_exponent": rates.corrosion_exponent,
            "fitted": rates.fitted,
        },
        "channels": dict(LEGACY_PROVENANCE),
        "caveats": [
            "Absolute days are NOT validated. No battery in this project reached "
            "end of life and no full-life dataset has been fitted against, so "
            "nothing sets the absolute rate. Ratios between scenarios cancel the "
            "unfitted constant and are the usable output.",
            "The ageing rate constant was derived from engine-BAY temperatures "
            "while corrosion is evaluated at BATTERY temperature, which lags by "
            "C_th/h ~ 7500 s. Measured exposure is 0.32x what the constant "
            "assumed, so absolute lives read roughly 3x long.",
            "Driving channels are FEATURES and battery channels are TARGETS. "
            "Within a trip soc = soc_0 - cumsum(i_bat_a*dt)/Q by construction, so "
            "offering current alongside soc hands a model the identity.",
            "t_bat_c tracks t_bay_c with no other forcing of consequence; "
            "predicting one from the other is close to trivial and the "
            "exact-match leakage detector will not flag it.",
            "The driving channels are IDENTICAL across scenarios -- one recording "
            "replayed under different ambient and accessory conditions. This is "
            "an ambient/duty-cycle sweep, not a driving-style sweep.",
        ],
    }
    (out / "dataset.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))

    print(f"\nwrote {out}/within_trip.csv     ({len(within_rows)} rows)")
    print(f"wrote {out}/scenario_life.csv  ({len(life_rows)} scenarios)")
    print(f"wrote {out}/dataset.json")
    if reached:
        print(f"life spread across scenarios: {sidecar['life_spread']['ratio']}x "
              f"({sidecar['life_spread']['min_days']:.0f} to "
              f"{sidecar['life_spread']['max_days']:.0f} days)")
    print(f"elapsed {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
