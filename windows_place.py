#!/usr/bin/env python3
"""Save where the car is now, under a name.

Drive somewhere easy -- a stretch of highway, a quiet road -- and save it:

    py windows_place.py highway
    py windows_place.py --list

Then a run can be told to send the car there when it keeps crashing where it is:

    py windows_drive.py "Aggressive" --mcp --roam --relocate-to highway

Only a person looking at the map knows which road is the easy one, which is why
this is a thing you drive to rather than something picked automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path as FilePath

sys.path.insert(0, str(FilePath(__file__).parent))

from control.places import load_place, place_names, save_place  # noqa: E402
from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError  # noqa: E402

PLACES_DIR = FilePath(__file__).parent / "places"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="what to call this spot")
    parser.add_argument("--list", action="store_true", help="show saved spots")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()

    if args.list or not args.name:
        names = place_names(PLACES_DIR)
        if not names:
            print(f"no places saved yet in {PLACES_DIR}")
            print('Drive somewhere easy, then:  py windows_place.py highway')
            return 1
        for name in names:
            place = load_place(PLACES_DIR, name)
            print(f"  {name:20} ({place.x_m:9.1f}, {place.y_m:9.1f}, {place.z_m:7.1f})")
        return 0

    client = MCPClient(args.endpoint)
    try:
        client.connect()
        status = client.call("get_status")
    except MCPError as error:
        print(f"  {error}", file=sys.stderr)
        return 1
    finally:
        client.close()

    vehicle = (status or {}).get("vehicle", {}) if isinstance(status, dict) else {}
    position = vehicle.get("pos")
    if not position:
        print("No position reported. Is a vehicle spawned?", file=sys.stderr)
        return 1

    place = save_place(
        PLACES_DIR, args.name,
        float(position["x"]), float(position["y"]), float(position.get("z", 0.0)),
    )
    print(f"Saved '{place.name}' at ({place.x_m:.1f}, {place.y_m:.1f}, {place.z_m:.1f})")
    print(f"\nUse it with:")
    print(f'  py windows_drive.py "Aggressive" --mcp --roam --relocate-to {place.name}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
