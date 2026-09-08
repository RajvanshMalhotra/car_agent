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
from control.calibration import (  # noqa: E402
    CalibrationError, VehicleLimits, calibrate, load_limits, save_limits,
)
from control.ai_driver import AIDriver, aggression_for  # noqa: E402
from control.roam_driver import RoamDriver  # noqa: E402
from control.demonstration import load_demonstration  # noqa: E402
from control.health import HealthMonitor  # noqa: E402
from control.places import load_place, place_names  # noqa: E402
from control.route import load_route  # noqa: E402
from control.policy import DrivingPolicy  # noqa: E402
from control.policy_driver import PolicyDriver  # noqa: E402
from control.tuner import ControllerGains, tune  # noqa: E402
from sim.backend import ControlInput  # noqa: E402
from sim.gamepad_udp import GamepadUDPBackend  # noqa: E402
from sim.mcp_backend import MCPBackend  # noqa: E402

CACHE_DIR = FilePath(__file__).parent / "behaviour" / "cache"
LOG_DIR = FilePath(__file__).parent / "runs"
LIMITS_DIR = FilePath(__file__).parent / "vehicles"
DEMO_DIR = FilePath(__file__).parent / "demonstrations"
PLACES_DIR = FilePath(__file__).parent / "places"
ROUTE_SPEC_DIR = FilePath(__file__).parent / "behaviour" / "routes"


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
    parser.add_argument("--manoeuvres", help="an LLM-designed route: part of its name")
    parser.add_argument("--demo", help="a drive recorded with windows_record.py -- "
                                       "its route AND the speed you drove it at")
    parser.add_argument("--crash-damage", type=float, default=100.0,
                        help="damage above which a run is treated as a crash")
    parser.add_argument("--max-deviation", type=float, default=None,
                        help="how far off the route is too far (default: 8 m on "
                             "a synthetic route, 30 m following a recorded one)")
    parser.add_argument("--off-route-seconds", type=float, default=4.0,
                        help="how long it may stay off the route before giving up")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mcp", action="store_true",
                        help="drive through BeamNG's built-in MCP server instead "
                             "of the virtual gamepad (0.39+; no ViGEmBus, no UDP)")
    parser.add_argument("--mcp-endpoint", default=None)
    parser.add_argument("--calibrate", action="store_true",
                        help="measure this vehicle's acceleration, braking and "
                             "steering response before driving, and cache them")
    parser.add_argument("--tune", action="store_true",
                        help="learn controller gains for the measured vehicle "
                             "(trains against the fake backend, seconds not days)")
    parser.add_argument("--roam", action="store_true",
                        help="let the AI drive the road network freely -- no "
                             "route at all. The simplest way to collect real "
                             "driving data, and it cannot get lost.")
    parser.add_argument("--roam-mode", default="span", choices=("span", "random"),
                        help="'span' covers the network, 'random' wanders")
    parser.add_argument("--beamng-ai", "--beamngai", "--ai", action="store_true",
                        help="let BeamNG's own AI drive: it follows roads and "
                             "avoids traffic, which nothing here can. Needs a "
                             "real map -- an empty level has no road network.")
    parser.add_argument("--policy", nargs="?", const="policies/default.json",
                        default=None,
                        help="drive with a learned policy (default "
                             "policies/default.json). Needs no calibration.")
    parser.add_argument("--repair-above", type=float, default=150.0,
                        help="repair the car once it has taken this much damage; "
                             "a pranged car drives differently and quietly "
                             "corrupts the rest of the run")
    parser.add_argument("--relocate-to",
                        help="a spot saved with windows_place.py. When the car "
                             "keeps crashing in one place it is moved here "
                             "instead of being patched up on the spot.")
    parser.add_argument("--crashes-per-window", type=int, default=3,
                        help="crashes close together before the car is moved")
    parser.add_argument("--crash-window", type=float, default=180.0,
                        help="seconds over which those crashes are counted")
    parser.add_argument("--give-up-after", type=int, default=3,
                        help="stop the run after this many recoveries that do "
                             "not get the car moving again")
    parser.add_argument("--stuck-after", type=float, default=45.0,
                        help="seconds stationary before the car counts as stuck "
                             "and is recovered to the road")
    parser.add_argument("--recoveries", type=int, default=5,
                        help="how many times to repair and carry on after a "
                             "crash or losing the route")
    parser.add_argument("--vehicle", default=None,
                        help="name for the cached calibration (default: ask the game)")
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
        print(f"no cached behaviour matching {args.behaviour!r}. Available:",
              file=sys.stderr)
        for spec in specs:
            print(f"  {spec.spec_hash}  {spec.name}", file=sys.stderr)
        print("\nGenerate a new one with:  ./agent.py generate \"<description>\"",
              file=sys.stderr)
        return 1
    spec = matches[0]

    stops = []
    manoeuvre_route = None
    demonstration = None
    if args.demo:
        for candidate in (FilePath(args.demo), DEMO_DIR / args.demo,
                          DEMO_DIR / f"{args.demo}.json"):
            if candidate.exists():
                demonstration = load_demonstration(candidate)
                break
        if demonstration is None:
            print(f"no recorded drive matching {args.demo!r} in {DEMO_DIR}",
                  file=sys.stderr)
            return 1

    if args.manoeuvres:
        from behaviour.route_spec import RouteSpec

        for candidate in sorted(ROUTE_SPEC_DIR.glob("*.json")):
            stored = json.loads(candidate.read_text())["spec"]
            if args.manoeuvres.lower() in stored["name"].lower():
                manoeuvre_route = RouteSpec.from_dict(stored)
                break
        if manoeuvre_route is None:
            print(f"no cached route matching {args.manoeuvres!r}", file=sys.stderr)
            return 1

    if demonstration is not None:
        recorded = None
        scenario = f"demo-{demonstration.name}"
    elif args.route:
        route_file = FilePath(args.route)
        if not route_file.exists():
            route_file = FilePath(__file__).parent / "routes" / args.route
        route = load_route(route_file)
        scenario = route_file.stem
        route = None  # built after the backend exists, see below
        recorded = route_file
    else:
        recorded = None
        scenario = (
            f"manoeuvres-{manoeuvre_route.route_hash}" if manoeuvre_route
            else f"aligned-straight-{args.route_length:.0f}m"
        )
    dt = 1.0 / args.rate

    if args.mcp:
        backend = MCPBackend(
            endpoint=args.mcp_endpoint,
            ambient_temp_c=spec.ambient_temp_c,
            cold_start=spec.cold_start,
            crash_damage=args.crash_damage,
            # The behaviour's accessory state drives the electrical load, which
            # is what makes the recharge-deficit pathway measurable at all.
            hvac_setting=spec.hvac_setting,
            # A recorded drive is in world coordinates; everything else starts
            # wherever the car happens to be.
            rebase_origin=demonstration is None,
        )
    else:
        backend = GamepadUDPBackend(
            ambient_temp_c=spec.ambient_temp_c,
            cold_start=spec.cold_start,
            ports=args.ports,
        )
    # A learned policy needs nothing measured: it was trained across randomised
    # vehicles, so there is no calibration step and nothing to reject.
    policy = DrivingPolicy.load(args.policy) if args.policy else None
    if policy is not None:
        print(f"  policy    : {args.policy} (no calibration needed)")

    vehicle = args.vehicle
    if vehicle is None and args.mcp:
        try:
            vehicle = (backend.client.call("get_vehicle") or {}).get("jbeam", "unknown")
        except Exception:
            vehicle = "unknown"
    vehicle = vehicle or "unknown"

    limits = load_limits(LIMITS_DIR, vehicle)
    if policy is not None or args.beamng_ai or args.roam:
        # None of these needs the vehicle measured: the policy was trained
        # across vehicles, and when the game drives it is driving its own car.
        limits = limits or VehicleLimits(vehicle, 3.0, 8.0, 2.7)
    elif args.calibrate or limits is None:
        print(f"  calibrating '{vehicle}' (about 20 s of driving)...")
        backend.reset()
        try:
            limits = calibrate(backend, dt=dt, vehicle=vehicle).validate()
            save_limits(LIMITS_DIR, limits)
        except (CalibrationError, RuntimeError) as error:
            # Driving on a bad measurement mis-scales every command, which is
            # worse than driving on the passenger-car defaults.
            print(f"  calibration REJECTED: {error}")
            print("  falling back to passenger-car defaults; the run is still "
                  "usable but the limits are assumed, not measured.")
            limits = VehicleLimits(vehicle, 3.0, 8.0, 2.7)
    if policy is None:
        print(f"  vehicle   : {vehicle}  accel {limits.max_accel_mps2:.2f} "
              f"decel {limits.max_decel_mps2:.2f} wheelbase {limits.wheelbase_m:.2f} m")

    gains = ControllerGains.default()
    if args.tune and policy is None and not (args.beamng_ai or args.roam):
        print("  tuning gains against the fake backend...")
        gains = tune(spec, limits, seed=args.seed)
        print(f"  gains     : kp={gains.kp:.2f} ki={gains.ki:.2f} kd={gains.kd:.3f} "
              f"lookahead={gains.lookahead_gain_s:.2f}s")

    # The route has to start where the car is and run the way it faces. Rather
    # than trusting a reported orientation whose conventions are undocumented,
    # roll the car forward briefly and measure which way it actually went.
    backend.reset()
    if args.roam:
        route = straight_route_from((0.0, 0.0), 0.0, 10.0)  # unused; kept for the log
        stops = []
        print("  route  : none -- the AI picks its own way around the map")
    elif demonstration is not None:
        # A recorded drive is already in world coordinates, so there is nothing
        # to align: the route is where the human actually drove.
        route = demonstration.to_path()
        stops = demonstration.stops()
        print(f"  demo   : '{demonstration.name}', {route.length_m:.0f} m, "
              f"you drove it at {demonstration.mean_speed_mps:.1f} m/s "
              f"with {len(stops)} stop(s)")
        print(f"  style  : {spec.name} -- speed x{spec.target_speed_factor:.2f} "
              f"of what you drove")
        # The recording ends at B, so the car is parked at the end of its own
        # route. Put it back at A before asking it to drive there.
        start = route.points[0]
        if hasattr(backend, "teleport_to"):
            state = backend.read_state()
            if math.dist((state.x_m, state.y_m), start) > 5.0:
                print(f"  moving the car back to the start of the route...")
                backend.teleport_to(start[0], start[1])
        else:
            print("  NOTE: drive the car back to the start of the route first.")
    elif recorded is not None:
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
        if manoeuvre_route is not None:
            route = manoeuvre_route.to_path(origin=(x, y), heading_rad=heading)
            stops = manoeuvre_route.stops()
            print(f"  aligned: heading {math.degrees(heading):.1f} deg")
            print(f"  route  : '{manoeuvre_route.name}', {route.length_m:.0f} m, "
                  f"{len(stops)} stops, {manoeuvre_route.total_stop_time_s():.0f}s idle")
        else:
            route = straight_route_from((x, y), heading, args.route_length)
            print(f"  aligned: heading {math.degrees(heading):.1f} deg, "
                  f"{args.route_length:.0f} m straight ahead")

    if args.mcp and args.rate > 25.0:
        print(f"  NOTE: --rate {args.rate:.0f} Hz is optimistic over HTTP; "
              f"20 Hz is a safer start with --mcp.\n")

    def build_route():
        """Lay a route from wherever the car is now, facing wherever it faces."""
        if demonstration is not None:
            # After a recovery the car rejoins the recorded drive wherever it
            # is nearest; the route itself does not move.
            return demonstration.to_path()
        x, y, heading = measure_heading(backend, dt=dt)
        if manoeuvre_route is not None:
            return manoeuvre_route.to_path(origin=(x, y), heading_rad=heading)
        return straight_route_from((x, y), heading, args.route_length)

    # A recorded route is a spatial path to follow, not a line to hug: driving
    # it in a different style takes a different line through a corner, and
    # teleporting the car over that is far more disruptive than the excursion.
    max_deviation = args.max_deviation
    if max_deviation is None:
        max_deviation = 30.0 if demonstration is not None else 8.0

    if args.roam or args.beamng_ai:
        if not getattr(backend, "has_road_network", lambda: True)():
            print("\n  This level has no road network, so BeamNG's AI has "
                  "nowhere to drive.", file=sys.stderr)
            print("  Load a real map (list_levels shows them) and try again.",
                  file=sys.stderr)
            backend.close()
            return 1
        if args.roam:
            driver = RoamDriver(backend, spec, mode=args.roam_mode)
            print(f"  driver : BeamNG's AI roaming ({args.roam_mode}), aggression "
                  f"{driver.aggression:.2f} (from '{spec.name}') -- no route")
        else:
            driver = AIDriver(
                backend, spec, route, dt=dt, speed_limit_mps=args.speed_limit,
                demonstration=demonstration,
            )
            print(f"  driver : BeamNG's own AI, aggression "
                  f"{driver.aggression:.2f} (from '{spec.name}')")
        driver.start()
    elif policy is not None:
        driver = PolicyDriver(
            policy, spec, route, dt=dt, speed_limit_mps=args.speed_limit,
            seed=args.seed, stops=stops, max_deviation_m=max_deviation,
            demonstration=demonstration,
            lost_persistence_s=args.off_route_seconds,
        )
    else:
        driver = Driver(
            spec, route, dt=dt, speed_limit_mps=args.speed_limit, seed=args.seed,
            max_deviation_m=max_deviation,
            wheelbase_m=limits.wheelbase_m,
            max_steer_rad=limits.max_steer_rad,
            max_accel_mps2=limits.max_accel_mps2,
            max_decel_mps2=limits.max_decel_mps2,
            gains=gains,
            stops=stops,
        )

    log_path = LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{spec.spec_hash}.csv"
    print(f"behaviour : {spec.name}  [{spec.spec_hash}]")
    print(f"log       : {log_path}")
    if args.roam:
        print(f"route     : none, {args.seconds:.0f}s of free driving")
    else:
        print(f"route     : {route.length_m:.0f} m, limit {args.speed_limit} m/s")
    print("Ctrl+C to stop. Controls are released on exit.\n")

    refuge = None
    if args.relocate_to:
        refuge = load_place(PLACES_DIR, args.relocate_to)
        if refuge is None:
            print(f"no saved place called {args.relocate_to!r}. "
                  f"Known: {', '.join(place_names(PLACES_DIR)) or 'none'}",
                  file=sys.stderr)
            print("Drive somewhere easy and run:  py windows_place.py <name>",
                  file=sys.stderr)
            backend.close()
            return 1
        print(f"  refuge : '{refuge.name}' at ({refuge.x_m:.0f}, {refuge.y_m:.0f}) "
              f"-- the car goes here if it keeps crashing")

    health = HealthMonitor(repair_above=args.repair_above,
                           stuck_after_s=args.stuck_after,
                           give_up_after=args.give_up_after,
                           crashes_per_window=args.crashes_per_window,
                           crash_window_s=args.crash_window)
    repairs = 0
    relocations = 0
    started = time.monotonic()
    recoveries = 0
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
                if control is not None:
                    # The AI driver returns nothing: BeamNG is working the
                    # controls, and sending pedal commands would fight it.
                    backend.apply_control(control)

                # A damaged or wedged car is put right without ending the run:
                # unattended runs are the point of roaming, and a run that stops
                # at the first kerb strike collects nothing.
                health.update(state)
                action = health.recommended_action()
                if health.beyond_help:
                    print(f"\n  STOPPING at t={elapsed:.1f}s: {health.futile_repairs} "
                          f"recoveries in a row and the car still will not move. "
                          f"It is somewhere it cannot be rescued from.")
                    break

                # Without a refuge there is nowhere better to send it, so a
                # troubled car is simply patched up where it is.
                if action == "relocate" and (refuge is None
                                             or not hasattr(backend, "teleport_to")):
                    action = "repair"

                if action and hasattr(backend, "repair"):
                    repairs += 1
                    detail = (f"{health.recent_crashes} crashes in "
                              f"{args.crash_window:.0f}s" if action == "relocate"
                              else f"damage {health.damage_taken:.0f}, stationary "
                                   f"{health.stationary_for_s:.0f}s")
                    print(f"\n  {action} at t={elapsed:.1f}s ({detail})"
                          f" -- run continues")
                    if action == "relocate":
                        print(f"      moving to '{refuge.name}' -- it cannot cope "
                              f"where it is")
                        backend.teleport_to(refuge.x_m, refuge.y_m, refuge.z_m)
                        relocations += 1
                        health.after_relocation(backend.read_state())
                    elif action == "recover":
                        backend.recover()
                        health.after_repair(backend.read_state())
                    else:
                        backend.repair()
                        health.after_repair(backend.read_state())
                    if hasattr(driver, "restart"):
                        driver.restart(route)
                    # A repair is a discontinuity, not something to smooth over.
                    log.end_trip()
                    continue

                if driver.is_finished(state):
                    # Without this the run sat at the end of a completed route
                    # for the rest of its time budget, padding the log with
                    # parked minutes and inflating the idle fraction -- which is
                    # one of the channels the corrosion figure is computed from.
                    print(f"\n  route completed at t={elapsed:.1f}s")
                    break

                crashed = getattr(backend, "has_crashed", lambda: False)()
                lost = driver.is_lost(state)
                if crashed or lost:
                    why = (
                        f"crashed (damage {getattr(backend, 'damage_since_start', 0.0):.0f})"
                        if crashed else
                        f"{driver.deviation_m(state):.1f} m off the route "
                        f"for {getattr(driver, 'off_route_s', 0.0):.0f}s"
                    )
                    if recoveries >= args.recoveries or not hasattr(backend, "recover"):
                        print(f"\n  STOPPING at t={elapsed:.1f}s: {why}. "
                              f"Releasing controls.")
                        if not crashed:
                            print(f"  (--max-deviation {max_deviation:.0f} was the "
                                  f"limit; raise it if the route is fine)")
                        break
                    recoveries += 1
                    print(f"\n  {why} at t={elapsed:.1f}s -- repairing and "
                          f"carrying on ({recoveries}/{args.recoveries})")
                    if demonstration is not None and hasattr(backend, "teleport_to"):
                        # Rejoin the recorded route at the point already
                        # reached, rather than letting the game drop the car on
                        # whatever road is nearest -- which for a recorded drive
                        # may be nowhere near it.
                        rejoin = route.point_at(
                            max(0.0, driver.progress_m - 10.0)
                        )
                        backend.teleport_to(rejoin[0], rejoin[1])
                    else:
                        backend.recover()
                    # A recovered car is put back on the nearest road, which is
                    # somewhere else entirely, so the route is re-laid from
                    # there rather than the old one being chased across the map.
                    try:
                        driver.restart(build_route())
                    except (RuntimeError, AttributeError) as error:
                        print(f"  could not resume: {error}")
                        break
                    # Each recovery starts a fresh trip, which is what the
                    # sulfation pathway is defined over.
                    log.end_trip()
                    continue

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
        # Hand the car back before letting go of it, or the AI keeps driving.
        if hasattr(driver, "stop"):
            driver.stop()
        backend.apply_control(ControlInput(0.0, 0.0, 0.0))
        backend.close()

    summary = log.summary()
    if summary["rows"]:
        print(f"\n{summary['rows']} rows -> {log_path}")
        print(f"  metadata         -> {log.sidecar_path}")
        if recoveries:
            print(f"  recoveries       {recoveries}")
        if repairs:
            print(f"  repairs          {repairs} "
                  f"(each starts a new trip in the log)")
        if relocations:
            print(f"  relocations      {relocations} to '{refuge.name}'")
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
