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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", help="what to call this route")
    parser.add_argument("--ports", type=int, nargs="+", default=[4444, 4445])
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--spacing", type=float, default=DEFAULT_SPACING_M)
    parser.add_argument("--rate", type=float, default=10.0,
                        help="assumed telemetry rate, used to time stops")
    args = parser.parse_args()

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
    if len(samples) < 2:
        print("No pose telemetry received. Enable Motion Sim and check with "
              "windows_probe.py.", file=sys.stderr)
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
    save_demonstration(out, demo, metadata={"ports": args.ports})

    print(f"\nSaved {out}")
    print(f"  {len(samples)} samples -> {len(demo.points)} points, "
          f"{demo.length_m:.0f} m")
    print(f"  you drove it at {demo.mean_speed_mps:.1f} m/s on average, "
          f"with {len(demo.stops())} stop(s)")
    print(f"\nNow have the agent drive it, in whatever style you ask for:")
    print(f"  py windows_drive.py \"Delhi Courier\" --demo {args.name} --mcp")
    print(f"  py windows_drive.py \"calm commuter\" --demo {args.name} --mcp")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
