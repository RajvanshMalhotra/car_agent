#!/usr/bin/env python3
"""Find out what BeamNG's in-engine MCP server can do.

BeamNG 0.39 has a built-in MCP server: Options -> Advanced -> "Enable MCP
server". It needs no BeamNG.tech licence and no third-party mod.

    py windows_mcp_probe.py

It connects, lists every tool with its input schema, and saves the full listing
to `mcp_tools.json`. Send that file back and the gamepad can be replaced with
direct control -- which also gets us traffic and collision state, if the server
exposes them.

Nothing here drives the car. It only reads, unless you pass --try-state, which
calls tools whose names look read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as FilePath

sys.path.insert(0, str(FilePath(__file__).parent))

from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError  # noqa: E402

#: Words that suggest a tool only reads. Anything else is left alone.
READ_ONLY_HINTS = ("state", "status", "info", "get", "list", "read", "query")

#: What this project needs. Reported explicitly so the gaps are obvious.
WANTED = {
    "vehicle pose / position": ("pos", "state", "transform"),
    "vehicle control (throttle/brake/steer)": ("control", "input", "drive"),
    "engine / powertrain": ("engine", "electric", "powertrain", "rpm"),
    "traffic / other vehicles": ("traffic", "vehicles"),
    "collision / damage": ("damage", "collision", "crash"),
    "time stepping / pause": ("step", "pause", "time"),
    "sensors": ("sensor", "lidar", "camera", "radar"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", default="mcp_tools.json")
    parser.add_argument("--try-state", action="store_true",
                        help="also call tools that look read-only")
    args = parser.parse_args()

    print("=" * 70)
    print(f"BeamNG MCP  {args.endpoint}")
    print("=" * 70)

    client = MCPClient(args.endpoint)
    try:
        info = client.connect()
    except MCPError as error:
        print(f"\n  {error}\n")
        print("  In BeamNG: Options > Advanced > General > 'Enable MCP server'.")
        print("  The tooltip shows the address; pass it with --endpoint if it")
        print("  differs from the default.")
        return 1

    server = info.get("serverInfo", {})
    print(f"  connected: {server.get('name', '?')} {server.get('version', '')}")
    print(f"  protocol : {info.get('protocolVersion', '?')}")
    print(f"  capabilities: {sorted(info.get('capabilities', {}))}")

    try:
        tools = client.list_tools()
    except MCPError as error:
        print(f"\n  could not list tools: {error}")
        return 1

    print(f"\n  {len(tools)} tools\n")
    print("=" * 70)
    for tool in sorted(tools, key=lambda t: t["name"]):
        params = sorted((tool.get("inputSchema") or {}).get("properties", {}))
        summary = (tool.get("description") or "").strip().splitlines()
        print(f"  {tool['name']}")
        if summary:
            print(f"      {summary[0][:96]}")
        if params:
            print(f"      args: {', '.join(params)}")

    print("\n" + "=" * 70)
    print("WHAT THIS PROJECT NEEDS")
    print("=" * 70)
    names = [t["name"].lower() for t in tools]
    for need, hints in WANTED.items():
        hits = [n for n in names if any(h in n for h in hints)]
        mark = "yes" if hits else "NO "
        print(f"  [{mark}] {need:42} {', '.join(hits[:4]) if hits else '-'}")

    if args.try_state:
        print("\n" + "=" * 70)
        print("CALLING READ-ONLY-LOOKING TOOLS")
        print("=" * 70)
        for tool in tools:
            name = tool["name"]
            required = (tool.get("inputSchema") or {}).get("required") or []
            if required or not any(h in name.lower() for h in READ_ONLY_HINTS):
                continue
            try:
                result = client.call(name)
                print(f"  {name}: {json.dumps(result)[:220]}")
            except MCPError as error:
                print(f"  {name}: {error}"[:220])

    out = FilePath(args.out)
    out.write_text(json.dumps({"serverInfo": info, "tools": tools}, indent=2))
    print(f"\nFull listing written to {out.resolve()}")
    print("Send me that file and I will wire direct control in.")
    client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
