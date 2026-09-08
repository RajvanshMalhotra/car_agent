#!/usr/bin/env python3
"""Record a route by driving it yourself.

The controller follows a path very accurately, which is worth nothing if the
path is not on the road. Drive the road once by hand; the line you drive becomes
the route the behaviours follow.

    py windows_record.py town-loop
    py windows_record.py town-loop --ports 4444 --seconds 600

Drive normally. Stop with Ctrl+C. The route is written to `routes/<name>.json`.

Tips for a route that behaves:
  * stay in the middle of your lane -- every behaviour will follow this line
  * avoid reversing; the route is assumed to go forwards
  * finish somewhere the car can safely come to a stop
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path as FilePath

sys.path.insert(0, str(FilePath(__file__).parent))

from control.demonstration import Demonstration, save_demonstration  # noqa: E402
from control.route import DEFAULT_SPACING_M, save_route  # noqa: E402
from sim.telemetry import MotionSimPacket, OutSimPacket, TelemetryError, identify  # noqa: E402

ROUTE_DIR = FilePath(__file__).parent / "demonstrations"


def bind(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", port))
    sock.settimeout(0.2)
    return sock


def record_over_mcp(args) -> int:
    """Sample position and speed from the game's own MCP server.

    The same source the driving loop uses, so if driving works, recording works
    -- no separate question about whether Motion Sim is enabled or whether its
    packet layout decodes on this build.
    """
    from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError

    client = MCPClient(args.mcp_endpoint or DEFAULT_ENDPOINT)
    try:
        client.connect()
    except MCPError as error:
        print(f"  {error}", file=sys.stderr)
        return 1

    print(f"Recording '{args.name}' over MCP. Drive from A to B now -- normally,")
    print("the way you want the route driven. Stops are kept. Ctrl+C to finish.\n")

    samples: list[tuple[float, float, float]] = []
    interval = 1.0 / args.rate
    deadline = time.time() + args.seconds
    last_report = 0.0
    try:
        while time.time() < deadline:
            loop_start = time.monotonic()
            try:
                status = client.call("get_status")
            except MCPError as error:
                print(f"\n  lost the connection: {error}", file=sys.stderr)
                break
            vehicle = (status or {}).get("vehicle", {}) if isinstance(status, dict) else {}
            position = vehicle.get("pos")
            if position:
                samples.append(
                    (float(position["x"]), float(position["y"]),
                     float(vehicle.get("speed", 0.0)))
                )
            now = time.time()
            if now - last_report > 1.0 and samples:
                last_report = now
                first, last = samples[0], samples[-1]
                print(f"  {len(samples)} samples, {math.dist(first[:2], last[:2]):7.0f} m "
                      f"from the start, {last[2]:5.1f} m/s", end="\r")
            pause = interval - (time.monotonic() - loop_start)
            if pause > 0:
                time.sleep(pause)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()

    print()
    return finish(samples, args)


def finish(samples, args) -> int:
    """Turn raw samples into a saved demonstration, or say why not."""
    if len(samples) < 2:
        print("No position data arrived. Check windows_probe.py (UDP) or "
              "windows_mcp_probe.py (MCP).", file=sys.stderr)
        return 1

    travelled = sum(
        math.dist(a[:2], b[:2]) for a, b in zip(samples, samples[1:])
    )
    speeds = [speed for _, _, speed in samples]
    print(f"  {len(samples)} samples, {travelled:.0f} m travelled, "
          f"speed {min(speeds):.1f}-{max(speeds):.1f} m/s")
    if travelled < 5.0:
        print(f"\nThe recorded positions barely changed ({travelled:.2f} m), so "
              "there is no route to save.", file=sys.stderr)
        print("The car moved on screen but the telemetry did not follow it: the "
              "pose source is not reporting.", file=sys.stderr)
        print("Try --mcp, which uses the same source the driving loop uses.",
              file=sys.stderr)
        return 1

    out = ROUTE_DIR / f"{args.name}.json"
    try:
        demo = Demonstration.from_samples(
            samples, name=args.name, sample_hz=args.rate,
            min_spacing_m=args.spacing,
        )
    except ValueError as error:
        print(f"Could not save the drive: {error}", file=sys.stderr)
        return 1
    save_demonstration(out, demo, metadata={"mcp": bool(getattr(args, "mcp", False))})

    print(f"\nSaved {out}")
    print(f"  {len(samples)} samples -> {len(demo.points)} points, "
          f"{demo.length_m:.0f} m")
    print(f"  you drove it at {demo.mean_speed_mps:.1f} m/s on average, "
          f"with {len(demo.stops())} stop(s)")
    print(f"\nNow have the agent drive it, in whatever style you ask for:")
    print(f"  py windows_drive.py \"Delhi Courier\" --demo {args.name} --mcp")
    print(f"  py windows_drive.py \"calm commuter\" --demo {args.name} --mcp")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="what to call this route")
    parser.add_argument("--ports", type=int, nargs="+", default=[4444, 4445])
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument("--rate", type=float, default=10.0,
                        help="samples per second")
    parser.add_argument("--mcp", action="store_true",
                        help="record through BeamNG's built-in MCP server "
                             "instead of UDP -- use this if you drive with --mcp")
    parser.add_argument("--mcp-endpoint", default=None)
    args = parser.parse_args()

    if args.mcp:
        return record_over_mcp(args)

    sockets = []
    for port in dict.fromkeys(args.ports):
        try:
            sockets.append(bind(port))
        except OSError as error:
            print(f"  port {port}: could not bind ({error})")
    if not sockets:
        print("no ports available", file=sys.stderr)
        return 1

    print(f"Recording '{args.name}'. Drive from A to B now -- normally, the way")
    print("you want the route driven. Stops are kept. Ctrl+C to finish.\n")
    samples: list[tuple[float, float, float]] = []
    deadline = time.time() + args.seconds
    last_report = 0.0

    try:
        while time.time() < deadline:
            for sock in sockets:
                try:
                    data = sock.recv(4096)
                except socket.timeout:
                    continue
                kind = identify(data)
                parser_for = {"motionsim": MotionSimPacket, "outsim": OutSimPacket}.get(kind)
                if parser_for is None:
                    continue
                try:
                    packet = parser_for.parse(data)
                except TelemetryError:
                    continue
                # Speed as well as position: how fast the drive was done is
                # what a behaviour later scales to make it brisk or gentle.
                samples.append((packet.x_m, packet.y_m, packet.speed_mps))

                now = time.time()
                if now - last_report > 1.0 and samples:
                    last_report = now
                    positions = [(x, y) for x, y, _ in samples]
                    travelled = sum(
                        math.dist(a, b)
                        for a, b in zip(positions[::25], positions[25::25])
                    )
                    print(f"  {len(samples)} samples, roughly {travelled:.0f} m, "
                          f"{samples[-1][2]:5.1f} m/s", end="\r")
    except KeyboardInterrupt:
        pass
    finally:
        for sock in sockets:
            sock.close()

    print()
    return finish(samples, args)


if __name__ == "__main__":
    raise SystemExit(main())
