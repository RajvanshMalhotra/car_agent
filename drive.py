#!/usr/bin/env python3
"""Let BeamNG's AI drive, and record what happens to the battery.

    py drive.py "Delhi Courier"                 15 minutes of free driving
    py drive.py "Delhi Courier" --minutes 60    an hour
    py drive.py --list                          what styles are available

That is the whole tool. Open a map with roads, spawn a car on one, run this.
The game's own AI does the driving -- it follows roads and avoids traffic,
which nothing in this project can -- and the behaviour you name sets how hard
it drives. Everything else is measurement.

The car is always handed back when this exits, whatever happens: normally, on
Ctrl+C, or on a crash. Leaving the AI engaged means the game keeps driving
after the script has gone.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from behaviour.spec import BehaviourSpec  # noqa: E402
from control.ai_driver import aggression_for  # noqa: E402
from control.health import HealthMonitor  # noqa: E402
from control.places import load_place, place_names  # noqa: E402
from datalog.writer import RunLog  # noqa: E402
from sim.mcp_backend import MCPBackend  # noqa: E402
from sim.mcp_client import DEFAULT_ENDPOINT, MCPError  # noqa: E402

HERE = Path(__file__).parent
BEHAVIOURS = HERE / "behaviour" / "cache"
PLACES = HERE / "places"
RUNS = HERE / "runs"

#: One sample a second. The battery ages over months; a finer log would only be
#: more rows saying the same thing.
LOG_HZ = 1.0

#: Every step is several HTTP round trips, so there is no point asking for more.
CONTROL_HZ = 5.0


def behaviours() -> list[BehaviourSpec]:
    return [
        BehaviourSpec.from_dict(json.loads(p.read_text())["spec"])
        for p in sorted(BEHAVIOURS.glob("*.json"))
    ]


def find_behaviour(needle: str) -> BehaviourSpec | None:
    for spec in behaviours():
        if needle == spec.spec_hash or needle.lower() in spec.name.lower():
            return spec
    return None


def drive(spec: BehaviourSpec, args) -> int:
    """Connect, hand the car to the AI, log until the time is up, hand it back."""
    refuge = load_place(PLACES, args.refuge) if args.refuge else None
    if args.refuge and refuge is None:
        print(f"No place called {args.refuge!r}. Known: "
              f"{', '.join(place_names(PLACES)) or 'none'}", file=sys.stderr)
        print("Drive somewhere easy, then:  py windows_place.py <name>",
              file=sys.stderr)
        return 1

    try:
        backend = MCPBackend(
            endpoint=args.endpoint,
            ambient_temp_c=spec.ambient_temp_c,
            cold_start=spec.cold_start,
            hvac_setting=spec.hvac_setting,
        )
    except (MCPError, RuntimeError) as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  In BeamNG: Options > Advanced > 'Enable MCP server'.",
              file=sys.stderr)
        return 1

    aggression = aggression_for(spec)
    seconds = args.minutes * 60.0
    log_path = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{spec.spec_hash}.csv"

    try:
        if not backend.has_road_network():
            print("\n  This map has no roads, so the AI has nowhere to drive.",
                  file=sys.stderr)
            print("  Load a real map -- West Coast USA, Italy, East Coast USA.",
                  file=sys.stderr)
            return 1

        print(f"  behaviour : {spec.name}  [{spec.spec_hash}]")
        print(f"  driving   : BeamNG's AI, aggression {aggression:.2f}, "
              f"avoiding traffic")
        print(f"  conditions: {spec.ambient_temp_c:.0f} C ambient, "
              f"climate control {spec.hvac_setting:.0%}")
        if refuge:
            print(f"  refuge    : '{refuge.name}' if it keeps crashing")
        print(f"  log       : {log_path}")
        print(f"  running   : {args.minutes:.0f} minutes. Ctrl+C to stop early.\n")

        backend.set_ai(mode=args.mode, aggression=aggression, avoidCars=True)
        return collect(backend, spec, args, log_path, refuge, seconds)
    finally:
        # Whatever happened, the game gets its car back.
        hand_back(backend)


def collect(backend, spec, args, log_path, refuge, seconds) -> int:
    health = HealthMonitor(
        repair_above=args.repair_above,
        stuck_after_s=args.stuck_after,
        crashes_per_window=args.crashes_before_moving,
    )
    interval = 1.0 / CONTROL_HZ
    log_every = 1.0 / LOG_HZ
    started = time.monotonic()
    next_log = 0.0
    repairs = 0

    log = RunLog(log_path, spec=spec, scenario=f"roam-{args.mode}",
                 seed=args.seed, log_hz=LOG_HZ)
    try:
        with log:
            while True:
                tick = time.monotonic()
                elapsed = tick - started
                if elapsed >= seconds:
                    print(f"\n  done: {args.minutes:.0f} minutes")
                    break

                state = backend.read_state()
                health.update(state)

                if health.beyond_help:
                    print(f"\n  stopping at {elapsed / 60:.1f} min: recovered "
                          f"{health.futile_repairs} times and it still will not "
                          f"move.")
                    break

                action = health.recommended_action()
                if action:
                    repairs += 1
                    put_right(backend, health, action, refuge, elapsed)
                    log.end_trip()
                    backend.set_ai(mode=args.mode,
                                   aggression=aggression_for(spec),
                                   avoidCars=True)
                    continue

                if elapsed >= next_log:
                    next_log += log_every
                    log.record(state, None)
                    if log.rows % 60 == 0:
                        print(f"  {elapsed / 60:5.1f} min   "
                              f"{state.speed_mps:5.1f} m/s   "
                              f"{state.rpm:6.0f} rpm   "
                              f"coolant {state.coolant_temp_c:5.1f} C   "
                              f"bay {state.underbonnet_temp_c:5.1f} C   "
                              f"{state.current_a:+6.1f} A")

                pause = interval - (time.monotonic() - tick)
                if pause > 0:
                    time.sleep(pause)
    except KeyboardInterrupt:
        print("\n  stopped")

    report(log, repairs)
    return 0


def put_right(backend, health, action, refuge, elapsed) -> None:
    """Repair, recover, or move the car somewhere easier."""
    if action == "relocate" and refuge is not None:
        print(f"\n  {health.recent_crashes} crashes in a row at "
              f"{elapsed / 60:.1f} min -- moving to '{refuge.name}'")
        backend.teleport_to(refuge.x_m, refuge.y_m, refuge.z_m)
        health.after_relocation(backend.read_state())
        return

    if action == "recover" or action == "relocate":
        print(f"\n  stuck at {elapsed / 60:.1f} min -- back on the road")
        backend.recover()
    else:
        print(f"\n  damage {health.damage_taken:.0f} at {elapsed / 60:.1f} min "
              f"-- repaired")
        backend.repair()
    health.after_repair(backend.read_state())


def report(log: RunLog, repairs: int) -> None:
    summary = log.summary()
    if not summary["rows"]:
        print("\n  No data. Was the car actually moving?")
        return
    print(f"\n  {summary['rows']} rows over {summary['duration_s'] / 60:.1f} min"
          f" -> {log.csv_path}")
    print(f"  metadata        {log.sidecar_path.name}")
    print(f"  idle            {summary['idle_fraction']:.0%} of the time")
    print(f"  engine bay      {summary['mean_underbonnet_c']:.1f} C mean, "
          f"{summary['max_underbonnet_c']:.1f} C peak")
    print(f"  charge balance  {summary['amp_hours_net']:+.2f} Ah "
          f"({'drained' if summary['amp_hours_net'] > 0 else 'charged'})")
    print(f"  corrosion       {summary['equivalent_hours']:.3f} equivalent-hours "
          f"at {summary['equivalent_hours_reference_c']:.0f} C")
    if repairs:
        print(f"  repairs         {repairs}")
    print(f"\n  Was the simulator worth it?")
    print(f"    python3 does_the_game_matter.py {log.csv_path}")


def hand_back(backend) -> None:
    """Disengage the AI and release the controls, ignoring any failure.

    This runs on every exit path. An early return that left the AI engaged once
    sent the car off around the map after the script had already finished.
    """
    for attempt in (
        lambda: backend.set_ai(mode="disabled"),
        lambda: backend.close(),
    ):
        try:
            attempt()
        except Exception:
            pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("behaviour", nargs="?",
                        help="a style: part of its name, or its hash")
    parser.add_argument("--list", action="store_true", help="show the styles")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--mode", default="span", choices=("span", "random"),
                        help="'span' covers the map, 'random' wanders")
    parser.add_argument("--refuge", help="a place saved with windows_place.py, "
                                         "to move to if it keeps crashing")
    parser.add_argument("--repair-above", type=float, default=150.0)
    parser.add_argument("--stuck-after", type=float, default=45.0)
    parser.add_argument("--crashes-before-moving", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args(argv)

    available = behaviours()
    if args.list or not args.behaviour:
        if not available:
            print(f"No styles in {BEHAVIOURS}.")
            print('Make one:  ./agent.py generate "a delivery courier in Delhi"')
            return 1
        for spec in available:
            print(f"  {spec.spec_hash}  {spec.name:36} "
                  f"speed x{spec.target_speed_factor:.2f}  "
                  f"{spec.ambient_temp_c:5.1f} C  "
                  f"climate {spec.hvac_setting:.0%}")
        return 0

    spec = find_behaviour(args.behaviour)
    if spec is None:
        print(f"No style matching {args.behaviour!r}. Available:", file=sys.stderr)
        for other in available:
            print(f"  {other.name}", file=sys.stderr)
        return 1

    return drive(spec, args)


if __name__ == "__main__":
    raise SystemExit(main())
