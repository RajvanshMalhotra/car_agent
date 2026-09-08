#!/usr/bin/env python3
"""Is the simulator earning its place?

    python3 does_the_game_matter.py runs/20260908-074853_26a0d77a6f3f8aba.csv

Takes one real game run and computes its corrosion twice: once from the coolant
temperature BeamNG measured, and once from the coolant our own engine model
predicts given the same speed and throttle. Everything else -- route, idle,
ambient, behaviour -- is identical, because it is the same run.

If the two agree, the game is not contributing on that channel: the driving
patterns could be sampled statistically and the whole simulator dropped. If they
disagree, it is, and by how much.

This does not test everything the game gives. How much idling actually happens
on a real road network with traffic is emergent and has no counterfactual to
compare against -- that one has to be argued, not measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as FilePath

sys.path.insert(0, str(FilePath(__file__).parent))

from analysis.game_value import compare_thermal_sources, read_run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", help="a run from runs/")
    parser.add_argument("--ambient", type=float, default=None,
                        help="ambient temperature; read from the sidecar if omitted")
    args = parser.parse_args()

    csv_path = FilePath(args.csv)
    if not csv_path.exists():
        print(f"no such run: {csv_path}", file=sys.stderr)
        return 1

    ambient = args.ambient
    cold_start = True
    sidecar = csv_path.with_suffix(".json")
    if sidecar.exists():
        stored = json.loads(sidecar.read_text())
        ambient = ambient if ambient is not None else stored.get("ambient_temp_c")
        cold_start = stored.get("cold_start", True)
    if ambient is None:
        print("no ambient temperature: pass --ambient", file=sys.stderr)
        return 1

    samples = read_run(csv_path)
    try:
        result = compare_thermal_sources(samples, ambient_c=ambient,
                                         cold_start=cold_start)
    except ValueError as error:
        print(f"  {error}", file=sys.stderr)
        return 1

    print("=" * 70)
    print(f"  {csv_path.name}")
    print(f"  {len(samples)} rows, {result.seconds:.0f}s, ambient {ambient:.0f} C")
    print("=" * 70)
    print(f"\n  {'':32}{'coolant':>10}{'bay mean':>11}{'bay max':>10}{'eq.hours':>11}")
    for side in (result.measured, result.modelled):
        print(f"  {side.label:32}{side.mean_coolant_c:10.1f}{side.mean_bay_c:11.1f}"
              f"{side.max_bay_c:10.1f}{side.equivalent_hours:11.4f}")

    print(f"\n  the game's coolant ran {result.coolant_gap_c:+.1f} C against our model")
    print(f"  corrosion answer differs by x{result.ratio:.3f}")

    print("\n" + "=" * 70)
    if result.game_matters:
        print("  THE GAME IS CONTRIBUTING")
        print(f"  Our engine model would have been wrong by {abs(result.ratio - 1) * 100:.0f}%")
        print("  on this run, systematically, in every result downstream.")
    else:
        print("  THE GAME IS NOT CONTRIBUTING -- on this channel")
        print("  Our own engine model predicts the same corrosion from speed and")
        print("  throttle alone. The measured coolant is not buying anything, and")
        print("  driving patterns could be sampled instead of simulated.")
    print("=" * 70)
    print("\n  Not tested here: how much idling a real road network with traffic")
    print("  produces. That is emergent, has no counterfactual, and is the")
    print("  strongest remaining argument for using the simulator at all.")
    if result.seconds < 300:
        print(f"\n  NOTE: only {result.seconds:.0f}s of driving. Run 10+ minutes")
        print("  before trusting this -- a cold start dominates a short run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
