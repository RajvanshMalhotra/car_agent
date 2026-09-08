#!/usr/bin/env python3
"""Find out whether BeamNG's own AI will drive the car, and if not, why.

    py windows_ai_probe.py

Nothing else is involved: no route, no behaviour, no logging, no policy. It
calls the game's AI tools one at a time, in increasing order of complexity,
prints exactly what each returned, and watches whether the car actually moved.
Whichever step first fails to move it is the answer.

Spawn a vehicle on a road on a real map first. `smallgrid` has no road network
at all and the AI has nowhere to drive.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path as FilePath

sys.path.insert(0, str(FilePath(__file__).parent))

from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError  # noqa: E402

WATCH_SECONDS = 8.0


def position_of(client) -> tuple[float, float, float, float]:
    """(x, y, z, speed) right now."""
    status = client.call("get_status")
    vehicle = (status or {}).get("vehicle", {}) if isinstance(status, dict) else {}
    pos = vehicle.get("pos") or {}
    return (
        float(pos.get("x", 0.0)),
        float(pos.get("y", 0.0)),
        float(pos.get("z", 0.0)),
        float(vehicle.get("speed", 0.0)),
    )


def watch(client, label: str, seconds: float = WATCH_SECONDS) -> float:
    """Watch for movement. Returns metres travelled."""
    start = position_of(client)
    moved = 0.0
    previous = start
    deadline = time.time() + seconds
    peak_speed = 0.0
    while time.time() < deadline:
        time.sleep(0.25)
        now = position_of(client)
        moved += math.dist(previous[:2], now[:2])
        peak_speed = max(peak_speed, now[3])
        previous = now
    verdict = "MOVED" if moved > 3.0 else "did not move"
    print(f"      -> {verdict}: {moved:.1f} m in {seconds:.0f}s, "
          f"peak speed {peak_speed:.1f} m/s")
    return moved


def attempt(client, label: str, tool: str, arguments: dict) -> bool:
    """Call one tool, print exactly what came back, then watch for movement."""
    print(f"\n  {label}")
    print(f"      {tool}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})")
    try:
        result = client.call(tool, arguments)
    except MCPError as error:
        print(f"      -> REFUSED: {error}")
        return False
    text = str(result)
    print(f"      -> returned: {text[:200]}")
    if "fail" in text.lower() or "error" in text.lower():
        print("      -> the game reported a problem with that call")
    return watch(client, label) > 3.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--distance", type=float, default=120.0,
                        help="how far ahead to aim the drive_to target")
    args = parser.parse_args()

    client = MCPClient(args.endpoint)
    try:
        client.connect()
    except MCPError as error:
        print(f"  {error}", file=sys.stderr)
        return 1

    print("=" * 68)
    print("1. IS THERE A ROAD NETWORK?")
    print("=" * 68)
    try:
        navgraph = client.call("get_navgraph", {})
        print(f"  {str(navgraph)[:200]}")
        nodes = navgraph.get("nodeCount", 0) if isinstance(navgraph, dict) else 0
        if not nodes:
            print("\n  No navgraph. BeamNG's AI drives between road nodes, so on a")
            print("  level without roads it cannot move the car at all.")
            print("  Load a real map (list_levels) and run this again.")
            return 1
        print(f"  OK - {nodes} road nodes.")
    except MCPError as error:
        print(f"  get_navgraph refused: {error}")
        return 1

    vehicle_id = client.call("get_player_vehicle_id")
    x, y, z, _ = position_of(client)
    print(f"\n  vehicle {vehicle_id} at ({x:.1f}, {y:.1f}, {z:.1f})")

    print("\n" + "=" * 68)
    print("2. WILL THE AI DRIVE AT ALL?")
    print("=" * 68)
    print("  Each attempt is watched for movement. The first one that moves the")
    print("  car is the call this project should be using.")

    def target_ahead():
        """A point ahead of where the car is *now*.

        Computed per attempt: the earlier free-driving attempts move the car
        hundreds of metres, and aiming a later attempt at the original position
        tests whether the car will drive somewhere it has already left. The
        first version of this probe did exactly that and its last two results
        were meaningless.
        """
        here_x, here_y, here_z, _ = position_of(client)
        return {"x": here_x + args.distance, "y": here_y, "z": here_z}

    attempts = [
        ("A. span mode -- drive the roads, no target at all",
         "set_ai", {"id": vehicle_id, "mode": "span"}),
        ("B. random mode -- the other free-driving mode",
         "set_ai", {"id": vehicle_id, "mode": "random"}),
        ("C. drive_to a point ahead, nothing else specified",
         "drive_to", {"id": vehicle_id, "pos": "AHEAD"}),
        ("D. drive_to with aggression",
         "drive_to", {"id": vehicle_id, "pos": "AHEAD", "aggression": 0.8}),
        ("E. drive_to with everything this project sends",
         "drive_to", {"id": vehicle_id, "pos": "AHEAD", "aggression": 0.8,
                      "avoidCars": True, "driveInLane": True,
                      "routeSpeed": 12.0, "routeSpeedMode": "limit"}),
    ]

    worked = []
    for label, tool, arguments in attempts:
        client.call("set_ai", {"id": vehicle_id, "mode": "disabled"})
        time.sleep(0.5)
        arguments = {
            key: (target_ahead() if value == "AHEAD" else value)
            for key, value in arguments.items()
        }
        if attempt(client, label, tool, arguments):
            worked.append(label)

    client.call("set_ai", {"id": vehicle_id, "mode": "disabled"})

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    if worked:
        for label in worked:
            print(f"  moved the car: {label}")
        print("\n  Send me this output and I will make the AI driver use the")
        print("  call that works.")
    else:
        print("  Nothing moved the car.")
        print("  Check: is the vehicle in gear, on a road, and not up against")
        print("  something? Is the game paused? Does driving it by hand work?")
        print("  Send me this whole output either way.")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
