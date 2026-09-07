#!/usr/bin/env python3
"""Teach the agent to drive, once. Then collect data with it, many times.

    python3 learn.py                       # train, ~90 s, no game needed
    python3 learn.py --iterations 40       # train harder
    python3 learn.py --evaluate            # test the saved policy and stop

Training runs entirely against the kinematic model, on randomised vehicles and
randomised target speeds. Nothing about any particular car is configured, so
the policy that comes out drives a hatchback and a 6x6 truck without being
measured or retuned for either.

The result is `policies/default.json` -- fourteen numbers. After that:

    py windows_drive.py "Delhi Courier" --mcp --policy

What this learns is how to work the controls to follow a line at a speed. It
does not learn to avoid anything: nothing in this project perceives obstacles.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path as FilePath

sys.path.insert(0, str(FilePath(__file__).parent))

from control.learning import drive_episode, learn_with_restarts  # noqa: E402
from control.learning import SELECTION_SEEDS  # noqa: E402
from control.policy import DrivingPolicy  # noqa: E402

POLICY_DIR = FilePath(__file__).parent / "policies"
DEFAULT_POLICY = POLICY_DIR / "default.json"

#: Seeds never used in training, so the report is on unseen vehicles.
HELD_OUT = range(90_000, 90_032)


def evaluate(policy: DrivingPolicy, seconds: float = 25.0) -> dict[str, float]:
    results = [drive_episode(policy, seed=s, seconds=seconds) for s in HELD_OUT]
    return {
        "episodes": len(results),
        "left_the_road": sum(r.left_the_road for r in results),
        "cross_track_m": statistics.mean(r.mean_cross_track_m for r in results),
        "speed_error_mps": statistics.mean(r.mean_speed_error_mps for r in results),
    }


def report(policy: DrivingPolicy) -> None:
    summary = evaluate(policy)
    print("\n  on vehicles and courses it has never seen:")
    print(f"    left the road   {summary['left_the_road']} of {summary['episodes']}")
    print(f"    tracking error  {summary['cross_track_m']:.2f} m")
    print(f"    speed error     {summary['speed_error_mps']:.2f} m/s")

    print("\n  by target speed (the mean includes accelerating from rest, so it"
          "\n  reads below target even when steady-state tracking is exact):")
    for target in (4.0, 8.0, 12.0, 18.0, 22.0):
        runs = [
            drive_episode(policy, seed=s, seconds=30.0, target_speed_mps=target)
            for s in range(91_000, 91_010)
        ]
        print(
            f"    asked {target:5.1f} m/s   held {statistics.mean(r.mean_driving_speed_mps for r in runs):5.2f} m/s"
            f"   off-road {sum(r.left_the_road for r in runs)}/10"
            f"   tracking {statistics.mean(r.mean_cross_track_m for r in runs):5.2f} m"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=6,
                        help="vehicles each candidate is scored on")
    parser.add_argument("--seconds", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--restarts", type=int, default=4,
                        help="train this many times and keep the best; a single "
                             "run is unreliable, see control/learning.py")
    parser.add_argument("--out", default=str(DEFAULT_POLICY))
    parser.add_argument("--evaluate", action="store_true",
                        help="report on the saved policy without training")
    args = parser.parse_args()

    if args.evaluate:
        path = FilePath(args.out)
        if not path.exists():
            print(f"no policy at {path}; run without --evaluate first", file=sys.stderr)
            return 1
        print(f"policy: {path}")
        report(DrivingPolicy.load(path))
        return 0

    print(f"Training on randomised vehicles: {args.restarts} restarts x "
          f"{args.iterations} iterations x {args.population} candidates x "
          f"{args.episodes} episodes")
    print("Nothing about any particular car is configured.\n")

    started = time.monotonic()

    def show(entry):
        if entry.get("iteration", 0) < 0:
            print(f"  -- restart {entry['restart']}: held-out score "
                  f"{entry['held_out_score']:7.3f}, left the road "
                  f"{entry['off_road']} of {len(SELECTION_SEEDS)}\n")
            return
        print(f"  iteration {entry['iteration']:3d}   best {entry['best']:8.3f}"
              f"   population mean {entry['mean']:8.3f}")

    def save_progress(policy, score, off_road, attempt):
        policy.save(args.out, metadata={
            "restarts_completed": attempt + 1,
            "held_out_score": score,
            "left_the_road": off_road,
            "iterations": args.iterations,
            "population": args.population,
            "seed": args.seed,
            "complete": False,
        })
        print(f"     saved to {args.out} (best so far -- safe to stop here)\n")

    policy, history = learn_with_restarts(
        restarts=args.restarts,
        on_improvement=save_progress,
        iterations=args.iterations,
        population=args.population,
        episodes_per_candidate=args.episodes,
        seconds=args.seconds,
        seed=args.seed,
        progress=show,
    )
    elapsed = time.monotonic() - started

    summary = evaluate(policy)
    policy.save(
        args.out,
        metadata={
            "iterations": args.iterations,
            "population": args.population,
            "episodes_per_candidate": args.episodes,
            "seed": args.seed,
            "training_seconds": round(elapsed, 1),
            "final_training_score": history[-1]["best"],
            "held_out": summary,
        },
    )
    print(f"\ntrained in {elapsed:.0f}s -> {args.out}")
    report(policy)
    print("\nNow collect data with it:")
    print('  py windows_drive.py "Delhi Courier" --mcp --policy')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
