#!/usr/bin/env python3
"""Drive a cached behaviour in BeamNG on the Windows machine, and log it.

Run `windows_probe.py` FIRST and make sure it reports a working gamepad and
both telemetry streams. Then:

    py windows_drive.py --list
    py windows_drive.py "Delhi Courier" --seconds 300

What it does: reads the behaviour spec from disk (no network), drives the car
with the same controller the fake backend uses, and writes a 1 Hz CSV log with
the channels the battery layer needs.

Before you start:
  * spawn a vehicle and put it on a road
  * press Ctrl+R in game to reset it to a clean state
  * the script drives a straight-ahead route by default -- see --route

SAFETY: the script releases throttle and brake on exit, including on Ctrl+C.
If it ever loses control of the car, Ctrl+C then press Ctrl+R in game.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path as FilePath

sys.path.insert(0, str(FilePath(__file__).parent))

from behaviour.spec import BehaviourSpec  # noqa: E402
from datalog.writer import RunLog  # noqa: E402
from control.driver import Driver  # noqa: E402
from control.path import Path  # noqa: E402
from control.alignment import measure_heading, straight_route_from  # noqa: E402
from control.route import load_route  # noqa: E402
from sim.backend import ControlInput  # noqa: E402
from sim.gamepad_udp import GamepadUDPBackend  # noqa: E402
from sim.mcp_backend import MCPBackend  # noqa: E402

CACHE_DIR = FilePath(__file__).parent / "behaviour" / "cache"
LOG_DIR = FilePath(__file__).parent / "runs"


def load_specs() -> list[BehaviourSpec]:
    return [
        BehaviourSpec.from_dict(json.loads(p.read_text())["spec"])
        for p in sorted(CACHE_DIR.glob("*.json"))
    ]


def straight_route(length_m: float = 3000.0) -> Path:
    return Path([(float(i), 0.0) for i in range(int(length_m) + 1)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("behaviour", nargs="?", help="spec_hash or part of the name")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--speed-limit", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=50.0,
                        help="control loop Hz (use ~20 with --mcp: each step is "
                             "several HTTP round trips)")
    parser.add_argument("--log-hz", type=float, default=1.0)
    parser.add_argument("--ports", type=int, nargs="+", default=[4444, 4445],
                        help="UDP ports to listen on; both streams may share one")
    parser.add_argument("--route-length", type=float, default=3000.0)
    parser.add_argument("--route", help="a route recorded with windows_record.py")
    parser.add_argument("--max-deviation", type=float, default=8.0,
                        help="abandon the run once this far off the route")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcp", action="store_true",
                        help="drive through BeamNG's built-in MCP server instead "
                             "of the virtual gamepad (0.39+; no ViGEmBus, no UDP)")
    parser.add_argument("--mcp-endpoint", default=None)
    args = parser.parse_args()

    specs = load_specs()
    if args.list or not args.behaviour:
        for spec in specs:
            print(f"{spec.spec_hash}  {spec.name}")
        return 0

    matches = [
        s for s in specs
        if args.behaviour == s.spec_hash or args.behaviour.lower() in s.name.lower()
    ]
    if not matches:
        print(f"no cached behaviour matching {args.behaviour!r}", file=sys.stderr)
        return 1
    spec = matches[0]

    if args.route:
        route_file = FilePath(args.route)
        if not route_file.exists():
            route_file = FilePath(__file__).parent / "routes" / args.route
        route = load_route(route_file)
        scenario = route_file.stem
        route = None  # built after the backend exists, see below
        recorded = route_file
    else:
        recorded = None
        scenario = f"aligned-straight-{args.route_length:.0f}m"
    dt = 1.0 / args.rate

    if args.mcp:
        backend = MCPBackend(
            endpoint=args.mcp_endpoint,
            ambient_temp_c=spec.ambient_temp_c,
            cold_start=spec.cold_start,
        )
    else:
        backend = GamepadUDPBackend(
            ambient_temp_c=spec.ambient_temp_c,
            cold_start=spec.cold_start,
            ports=args.ports,
        )
    # The route has to start where the car is and run the way it faces. Rather
    # than trusting a reported orientation whose conventions are undocumented,
    # roll the car forward briefly and measure which way it actually went.
    backend.reset()
    if recorded is not None:
        route = load_route(recorded)
        print(f"  route: {recorded.name}, {route.length_m:.0f} m")
    else:
        print("  aligning: rolling forward to find which way the car points...")
        try:
            x, y, heading = measure_heading(backend, dt=dt)
        except RuntimeError as error:
            print(f"\n  {error}", file=sys.stderr)
            backend.apply_control(ControlInput(0.0, 0.0, 0.0))
            backend.close()
            return 1
        route = straight_route_from((x, y), heading, args.route_length)
        print(f"  aligned: heading {math.degrees(heading):.1f} deg, "
              f"{args.route_length:.0f} m straight ahead")

    if args.mcp and args.rate > 25.0:
        print(f"  NOTE: --rate {args.rate:.0f} Hz is optimistic over HTTP; "
              f"20 Hz is a safer start with --mcp.\n")

    driver = Driver(
        spec, route, dt=dt, speed_limit_mps=args.speed_limit, seed=args.seed,
        max_deviation_m=args.max_deviation,
    )

    log_path = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{spec.spec_hash}.csv"
    print(f"behaviour : {spec.name}  [{spec.spec_hash}]")
    print(f"log       : {log_path}")
    print(f"route     : {route.length_m:.0f} m, limit {args.speed_limit} m/s")
    print("Ctrl+C to stop. Controls are released on exit.\n")

    started = time.monotonic()
    next_log = 0.0
    log_interval = 1.0 / args.log_hz
    log = RunLog(
        log_path,
        spec=spec,
        scenario=f"{scenario}@{args.speed_limit:.0f}mps",
        seed=args.seed,
        log_hz=args.log_hz,
    )

    try:
        with log:
            while True:
                loop_start = time.monotonic()
                elapsed = loop_start - started
                if elapsed >= args.seconds:
                    break

                state = backend.read_state()
                control = driver.step(state)
                backend.apply_control(control)

                if getattr(backend, "has_crashed", lambda: False)():
                    print(f"\n  CRASHED at t={elapsed:.1f}s. Releasing controls.")
                    break

                if driver.is_lost(state):
                    print(f"\n  ABANDONED at t={elapsed:.1f}s: "
                          f"{driver.deviation_m(state):.1f} m off the route "
                          f"(limit {args.max_deviation:.0f} m). Releasing controls.")
                    break

                if elapsed >= next_log:
                    next_log += log_interval
                    log.record(state, control)
                    if log.rows % 10 == 0:
                        print(f"  t={elapsed:6.1f}s  v={state.speed_mps:5.2f} m/s  "
                              f"rpm={state.rpm:6.0f}  coolant={state.coolant_temp_c:5.1f}C  "
                              f"bay={state.underbonnet_temp_c:5.1f}C")

                sleep_for = dt - (time.monotonic() - loop_start)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        backend.apply_control(ControlInput(0.0, 0.0, 0.0))
        backend.close()

    summary = log.summary()
    if summary["rows"]:
        print(f"\n{summary['rows']} rows -> {log_path}")
        print(f"  metadata         -> {log.sidecar_path}")
        print(f"  idle fraction    {summary['idle_fraction']:.2f} "
              f"(behaviour asked for {spec.idle_fraction:.2f})")
        print(f"  bay temperature  mean {summary['mean_underbonnet_c']:.1f} C  "
              f"max {summary['max_underbonnet_c']:.1f} C")
        print(f"  corrosion        {summary['equivalent_hours']:.3f} equivalent-hours "
              f"at {summary['equivalent_hours_reference_c']:.0f} C")
    else:
        print("\nNo telemetry received - run windows_probe.py to diagnose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
