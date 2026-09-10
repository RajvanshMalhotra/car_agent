#!/usr/bin/env python3
"""Extract driving trajectories from BeamNG.

    py collect_trajectories.py "Hot commute" --minutes 15
    py collect_trajectories.py "Hot commute" --minutes 15 --aggression 1.0
    python3 collect_trajectories.py "Smoke test" --fake --minutes 1

BeamNG's own AI drives -- it follows roads and avoids traffic, which nothing
here can. This project chooses the style and keeps every measurement.

Samples are taken *inside the vehicle's physics VM* at 100 Hz and drained once
a second, so the HTTP transport is not the sample rate. Longitudinal
acceleration and road grade are therefore measured rather than differenced out
of a 1 Hz speed trace.

**Needs a real map.** The AI drives the navgraph; `smallgrid` has none, and
`gridmap_v2` has one but no realistic roads. Use West Coast USA or Italy.

Two columns you will not find in the output: `current_a` and `voltage_v`.
BeamNG simulates no 12 V system, so they are modelled downstream from these
channels and would read as measurements if they appeared here.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collect.run import collect_run  # noqa: E402
from collect.scenario import ScenarioSpec  # noqa: E402
from collect.source import FakeTrajectorySource  # noqa: E402

RUNS = Path(__file__).parent / "runs"

#: 100 Hz. Fast enough for braking and grade transients, and a hundredth of the
#: physics rate, so the hook costs almost nothing.
SAMPLE_INTERVAL_S = 0.01


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="a label for this run")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--aggression", type=float, default=0.6,
                        help="0.25 crawls, 1.2 drives like it is being chased")
    parser.add_argument("--route", default="span",
                        choices=("span", "random"),
                        help="span covers the road network, random wanders")
    parser.add_argument("--accessory-load", type=float, default=35.0,
                        help="amps. BeamNG models no accessories, so this is an "
                             "assumption the sidecar records")
    parser.add_argument("--ambient", type=float, default=25.0,
                        help="celsius. Readable from the game but not settable "
                             "in it, so this is an assumption too")
    parser.add_argument("--hz", type=float, default=1.0 / SAMPLE_INTERVAL_S)
    parser.add_argument("--no-drive", action="store_true",
                        help="record without engaging the AI")
    parser.add_argument("--fake", action="store_true",
                        help="run against the fake source, with no game")
    parser.add_argument("--endpoint", default=None)
    args = parser.parse_args(argv)

    interval_s = 1.0 / args.hz if args.hz > 0 else SAMPLE_INTERVAL_S

    try:
        spec = ScenarioSpec(
            name=args.name, aggression=args.aggression, minutes=args.minutes,
            accessory_load_a=args.accessory_load, ambient_temp_c=args.ambient,
            route=args.route,
        )
    except ValueError as error:
        print(f"  {error}", file=sys.stderr)
        return 1

    csv_path = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{spec.scenario_hash}.csv"

    if args.fake:
        source = FakeTrajectorySource(interval_s=interval_s)
        beamng = {"backend": "fake"}
        client = None
    else:
        source, beamng, client = connect(args, interval_s)
        if source is None:
            return 1

    print(f"  scenario : {spec.name}  [{spec.scenario_hash}]")
    print(f"  sampling : {1 / interval_s:.0f} Hz inside the vehicle VM, "
          f"drained every second")
    if client is not None:
        print(f"  level    : {beamng.get('level', '?')}")
        print(f"  vehicle  : {beamng.get('vehicle', '?')}  "
              f"{source.mass_kg:.0f} kg")
        how = "off" if args.no_drive else (
            f"BeamNG AI, {args.route}, aggression {args.aggression:.2f}")
        print(f"  driving  : {how}")
    print(f"  log      : {csv_path}")
    print(f"  running  : {args.minutes:.0f} minutes. Ctrl+C to stop early.\n")

    if client is not None and not args.no_drive:
        engage(client, args)

    summary = collect_run(source, spec, csv_path, seconds=args.minutes * 60.0,
                          beamng=beamng, on_progress=progress)

    if client is not None and not args.no_drive:
        # Whatever happened, the game gets its car back. Leaving the AI engaged
        # means it keeps driving after this script has gone.
        hand_back(client)

    print(f"\n\n  {summary['rows']} rows over {summary['duration_s']:.0f} s, "
          f"{summary['dropped']} dropped")
    if summary["capture_error"]:
        print(f"  capture error: {summary['capture_error']}")
    print(f"  {csv_path}")
    print(f"  {csv_path.with_suffix('.json')}  (channels, provenance, scenario)")
    return 0


def connect(args, interval_s):
    """Open the MCP session and install the sampler."""
    from collect.mcp_source import MCPTrajectorySource
    from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError

    client = MCPClient(args.endpoint or DEFAULT_ENDPOINT)
    try:
        client.connect()
        status = client.call("get_status")
    except MCPError as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  In BeamNG: Options > Advanced > 'Enable MCP server'.",
              file=sys.stderr)
        return None, {}, None

    status = status if isinstance(status, dict) else {}
    vehicle = status.get("vehicle", {})
    level = str(status.get("level", "unknown"))

    source = MCPTrajectorySource(client, interval_s=interval_s)
    try:
        source.start()
    except RuntimeError as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  Spawn an undamaged car on a real map and try again.",
              file=sys.stderr)
        return None, {}, None

    beamng = {
        "backend": "mcp",
        "endpoint": client.endpoint,
        "level": level,
        "vehicle": vehicle.get("jbeam", "unknown"),
        "config": vehicle.get("configKey", ""),
        "mass_kg": source.mass_kg,
        "ambient_temp_c_read": read_ambient(client),
    }
    return source, beamng, client


def read_ambient(client) -> float | None:
    """What the game says the ambient is. Readable, not settable."""
    from sim.mcp_client import MCPError

    try:
        result = client.call(
            "run_lua",
            {"code": "return tostring(core_environment.getState().temperatureC)"},
        )
        return float(str(result).strip())
    except (MCPError, TypeError, ValueError):
        return None


def engage(client, args) -> None:
    from sim.mcp_client import MCPError

    try:
        client.call("set_ai", {"mode": args.route, "aggression": args.aggression,
                               "avoidCars": True})
    except MCPError as error:
        print(f"  warning: the AI would not engage ({error}). "
              f"Recording a parked car.")


def hand_back(client) -> None:
    from sim.mcp_client import MCPError

    try:
        client.call("set_ai", {"mode": "disabled"})
    except MCPError:
        pass


def progress(rows: int, elapsed: float, total: float) -> None:
    print(f"\r  {elapsed:5.0f}s / {total:.0f}s   {rows:7d} rows", end="", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
