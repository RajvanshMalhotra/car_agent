#!/usr/bin/env python3
"""Behaviour agent CLI.

    export DEEPSEEK_API_KEY=...
    ./agent.py generate "a delivery courier in Delhi in summer"
    ./agent.py list
    ./agent.py drive <spec_hash|substring of name>

Generation is offline and cached: `generate` calls the LLM only on a cache
miss, and `list`/`drive` never touch the network.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path as FilePath

from behaviour.generator import BehaviourGenerator
from behaviour.spec import BehaviourSpec
from control.driver import Driver
from control.path import Path
from sim.fake import FakeBackend

CACHE_DIR = FilePath(__file__).parent / "behaviour" / "cache"
DT = 0.02


def winding_route(length_m=1250.0, amplitude_m=25.0, wavelength_m=377.0):
    """A stand-in route until real BeamNG map data is available."""
    step = 0.5
    n = int(length_m / step)
    k = 2 * math.pi / wavelength_m
    return Path([(i * step, amplitude_m * math.sin(i * step * k)) for i in range(n)])


def load_cached() -> list[tuple[BehaviourSpec, dict]]:
    entries = []
    for path in sorted(CACHE_DIR.glob("*.json")):
        record = json.loads(path.read_text())
        entries.append((BehaviourSpec.from_dict(record["spec"]), record["provenance"]))
    return entries


def cmd_generate(args) -> int:
    from behaviour.deepseek import DeepSeekClient

    generator = BehaviourGenerator(DeepSeekClient(model=args.model), cache_dir=CACHE_DIR)
    for description in args.description:
        spec = generator.generate(description)
        print(f"{spec.spec_hash}  {spec.name}")
    return 0


def cmd_list(args) -> int:
    entries = load_cached()
    if not entries:
        print(f"no cached behaviours in {CACHE_DIR}")
        return 1
    print(f"{'spec_hash':18}{'name':34}{'speed':>7}{'accel':>7}{'decel':>7}"
          f"{'lag':>6}{'erratic':>8}{'idle':>6}{'hvac':>6}{'ambient':>8}")
    for spec, _ in entries:
        print(f"{spec.spec_hash:18}{spec.name[:32]:34}{spec.target_speed_factor:7.2f}"
              f"{spec.accel_limit_mps2:7.2f}{spec.decel_limit_mps2:7.2f}"
              f"{spec.reaction_lag_s:6.2f}{spec.erraticness:8.2f}"
              f"{spec.idle_fraction:6.2f}{spec.hvac_setting:6.2f}"
              f"{spec.ambient_temp_c:8.1f}")
    return 0


def find_spec(needle: str) -> BehaviourSpec | None:
    for spec, _ in load_cached():
        if needle == spec.spec_hash or needle.lower() in spec.name.lower():
            return spec
    return None


def cmd_drive(args) -> int:
    spec = find_spec(args.behaviour)
    if spec is None:
        print(f"no cached behaviour matching {args.behaviour!r}", file=sys.stderr)
        return 1

    route = winding_route()
    backend = FakeBackend(dt=DT)
    backend.reset()
    driver = Driver(spec, route, dt=DT, speed_limit_mps=args.speed_limit, seed=args.seed)

    speeds, accels, errors, progress = [], [], [], 0.0
    while backend.read_state().sim_time_s < args.max_seconds:
        before = backend.read_state()
        backend.apply_control(driver.step(before))
        state = backend.read_state()
        speeds.append(state.speed_mps)
        accels.append((state.speed_mps - before.speed_mps) / DT)
        progress = route.closest_arc_length((state.x_m, state.y_m), progress)
        if state.sim_time_s > 10.0 and progress < route.length_m - 5.0:
            near_x, near_y = route.point_at(progress)
            errors.append(math.hypot(near_x - state.x_m, near_y - state.y_m))
        if driver.is_finished(state):
            break

    jerk = max(abs(b - a) / DT for a, b in zip(accels, accels[1:]))
    print(f"{spec.name}  [{spec.spec_hash}]")
    print(f"  route          {route.length_m:.0f} m, limit {args.speed_limit} m/s")
    print(f"  duration       {backend.read_state().sim_time_s:.1f} s")
    print(f"  speed          mean {statistics.mean(speeds):.2f}  max {max(speeds):.2f} m/s")
    print(f"  acceleration   max {max(accels):.2f} / limit {spec.accel_limit_mps2} m/s^2")
    print(f"  jerk           max {jerk:.2f} / limit {spec.jerk_limit_mps3} m/s^3")
    print(f"  tracking error max {max(errors):.3f} m")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="turn descriptions into cached specs")
    generate.add_argument("description", nargs="+")
    generate.add_argument("--model", default="deepseek-v4-pro")
    generate.set_defaults(func=cmd_generate)

    listing = sub.add_parser("list", help="show cached behaviours")
    listing.set_defaults(func=cmd_list)

    drive = sub.add_parser("drive", help="run a behaviour through the fake backend")
    drive.add_argument("behaviour", help="spec_hash or part of the name")
    drive.add_argument("--speed-limit", type=float, default=22.0)
    drive.add_argument("--seed", type=int, default=0)
    drive.add_argument("--max-seconds", type=float, default=900.0)
    drive.set_defaults(func=cmd_drive)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
