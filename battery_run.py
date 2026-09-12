#!/usr/bin/env python3
"""Recorded driving in, battery dataset out.

    python3 battery_run.py runs/telemetry.csv --name baseline --out runs/dataset

One recorded trip contains one crank at most, and the recorded file contains
none at all -- it begins with the engine already running. Repeating the trip
under a declared schedule is what gives the sulfation and parasitic pathways
any data, and every repeat's soak duration is a scenario input recorded in the
sidecar, never a measurement.

This script refuses to fabricate two things. It will not report an absolute
remaining life as if it were measured -- `AgingRates.fitted` is False because
no battery in this project has reached end of life, so `cell.life.LifeEstimate`
carries that flag through to the sidecar and this script prints the caveat
next to every number it derives from it, rather than calling
`LifeEstimate.headline_days()`, which would raise. And it will not paper over
the telemetry dropout in the recording: `load.legacy.read_legacy` excludes it,
and this script reports exactly what was excluded rather than staying silent
about the gap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cell.aging import AgingRates, Damage
from cell.dataset import write_dataset
from cell.integrate import CellState, run_soak, run_trip
from cell.life import TripSchedule, ensemble
from load.legacy import LEGACY_PROVENANCE, read_legacy
from load.spec import BatteryScenario
from load.trips import segment


def build(
    trajectory_path: Path | str,
    scenario: BatteryScenario,
    schedule: TripSchedule,
    rates: AgingRates,
    repeats: int,
    out_dir: Path | str,
) -> dict:
    """Integrate the trajectory `repeats` times and write the dataset."""
    trajectory = read_legacy(trajectory_path)

    # Downsampled to 1 Hz once and held: the cell integrates at 1 Hz, and the
    # recorded file is 94 Hz, so this is about 1,780 samples rather than 167k.
    samples = list(trajectory.at_hz(1.0))
    dropouts = trajectory.dropouts.spans
    segments = segment(iter(samples))
    driving = [
        s for s in samples
        if any(
            seg.kind == "trip" and seg.start_s <= s["t_s"] <= seg.end_s
            for seg in segments
        )
    ]

    state = CellState(temp_c=scenario.ambient_c)
    trips = []
    for index in range(repeats):
        result = run_trip(
            iter(driving), scenario, state, rates, cranked=index > 0
        )
        trips.append((driving, result))
        if index < repeats - 1:
            run_soak(
                schedule.soak_s, scenario, state, rates,
                initial_coolant_c=float(driving[-1]["coolant_c"]),
            )

    # A day's damage at a given health: the trip damage scaled by trips per
    # day, plus the soaks between them. Recomputed as resistance rises.
    def day_damage(health: float) -> Damage:
        probe = CellState(
            soc=state.soc, temp_c=scenario.ambient_c, aging=state.aging.copy()
        )
        trip = run_trip(iter(driving), scenario, probe, rates, cranked=True).damage
        soak = run_soak(
            schedule.soak_s, scenario, probe, rates,
            initial_coolant_c=float(driving[-1]["coolant_c"]),
        ).damage
        return trip.scaled(schedule.trips_per_day) + soak.scaled(
            schedule.trips_per_day
        )

    life = ensemble(day_damage, rates, schedule, draws=16, seed=scenario.seed)

    return write_dataset(
        out_dir=out_dir, scenario=scenario, trips=trips, life=life,
        provenance=LEGACY_PROVENANCE, dropouts=dropouts, rates=rates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="a recorded OutGauge-shaped CSV")
    parser.add_argument("--name", default="run")
    parser.add_argument("--out", default="runs/dataset")
    parser.add_argument("--ambient", type=float, default=25.0)
    parser.add_argument("--hvac", type=float, default=0.0)
    parser.add_argument("--lights", action="store_true")
    parser.add_argument("--repeats", type=int, default=4,
                        help="how many times the recorded trip is driven")
    parser.add_argument("--trips-per-day", type=float, default=2.0)
    parser.add_argument("--soak-hours", type=float, default=8.0)
    args = parser.parse_args(argv)

    scenario = BatteryScenario(
        name=args.name, ambient_c=args.ambient, hvac=args.hvac,
        lights=args.lights,
    )
    schedule = TripSchedule(
        trips_per_day=args.trips_per_day, soak_s=args.soak_hours * 3600.0
    )
    sidecar = build(
        args.trajectory, scenario, schedule, AgingRates(), args.repeats, args.out
    )

    life = sidecar["life"]
    print(f"wrote {args.out}/within_trip.csv "
          f"({sidecar['tables']['within_trip']['rows']} rows at 1 Hz)")
    print(f"wrote {args.out}/trip_damage.csv "
          f"({sidecar['tables']['trip_damage']['rows']} trips)")
    if sidecar["dropouts"]:
        for span in sidecar["dropouts"]:
            print(f"  excluded a telemetry dropout: {span['start_s']:.1f}s to "
                  f"{span['end_s']:.1f}s, {span['rows']} rows")
    if life["eol_days"] is not None:
        interval = life["interval_days"]
        print(f"  end of life at about {life['eol_days']:.0f} days"
              + (f" (90% interval {interval[0]:.0f} to {interval[1]:.0f})"
                 if interval else ""))
    print("  scale is UNFITTED: no battery in this project reached end of life, "
          "so absolute days are indicative. Ratios between scenarios are the "
          "usable output.")
    findings = sidecar["leakage"]["findings"]
    print(f"  leakage findings: {len(findings)}")
    for finding in findings:
        print(f"    {finding['relation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
