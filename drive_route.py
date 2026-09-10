#!/usr/bin/env python3
"""Drive the route you picked on the map, in a chosen style, and log it.

    py drive_route.py aggressive
    py drive_route.py economical
    py drive_route.py stops
    py drive_route.py aggressive --roam   no destination, cover the network
    py drive_route.py --show-route        what the game says your route is

Set a destination in BeamNG's map first -- the blue line on the road is the
route. This reads where that line ends, drives there, and records the whole
thing at 100 Hz from inside the vehicle's physics VM.

Three styles, chosen because they are the three that reach a starter battery:

    aggressive   high engine load, hot engine bay -- the corrosion pathway
    economical   the same route, gently -- the baseline to compare against
    stops        frequent stops with the engine idling -- the recharge-deficit
                 pathway, where the alternator at idle RPM may not cover the
                 electrical load

Hard braking is deliberately not a style. It barely touches an SLI battery.

**The AI is commanded from inside the vehicle VM**, not through the MCP
`set_ai` and `drive_to` tools. That is not a preference, it is a measurement:
`drive_to` reported an accepted route to a real navgraph node and the car sat
still, while `ai.setMode('span')` in the vehicle's own Lua covered 32 m in six
seconds from a standstill and released the parking brake on its way.

The car is always handed back when this exits -- normally, on Ctrl+C, or on a
crash. Leaving the AI engaged means the game keeps driving after this has gone.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collect.decode import is_async_notice  # noqa: E402
from collect.mcp_source import MCPTrajectorySource  # noqa: E402
from collect.run import collect_run  # noqa: E402
from collect.scenario import ScenarioSpec  # noqa: E402
from collect.vm import VMError, VehicleVM  # noqa: E402
from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError  # noqa: E402

RUNS = Path(__file__).parent / "runs"

#: 100 Hz. Fast enough for braking and grade transients, and a hundredth of the
#: physics rate, so the hook costs almost nothing.
SAMPLE_INTERVAL_S = 0.01

#: Close enough to count as arrived. The AI stops *at* its target, so a tighter
#: radius means waiting for it to settle.
ARRIVED_M = 25.0


@dataclass(frozen=True)
class Style:
    """How to drive. Everything here reaches the AI directly."""

    name: str
    aggression: float
    #: 'limit' obeys posted limits, 'off' ignores them.
    speed_mode: str
    drive_in_lane: bool
    #: Seconds between stops, as a range. None means never stop.
    stop_every_s: tuple[float, float] | None
    #: How long to sit there, as a range.
    stop_for_s: tuple[float, float]
    #: Amps of accessory load assumed. Not simulated by the game -- recorded so
    #: the log says what was assumed rather than leaving it to be guessed.
    accessory_load_a: float
    why: str


STYLES = {
    "aggressive": Style(
        name="aggressive", aggression=1.0, speed_mode="off", drive_in_lane=True,
        stop_every_s=None, stop_for_s=(0.0, 0.0), accessory_load_a=35.0,
        why="high engine load, hot bay, fastest grid corrosion",
    ),
    "economical": Style(
        name="economical", aggression=0.35, speed_mode="limit", drive_in_lane=True,
        stop_every_s=None, stop_for_s=(0.0, 0.0), accessory_load_a=35.0,
        why="the same route driven gently, as the baseline",
    ),
    "stops": Style(
        name="stops", aggression=0.6, speed_mode="limit", drive_in_lane=True,
        stop_every_s=(45.0, 150.0), stop_for_s=(15.0, 60.0),
        accessory_load_a=55.0,
        why="idling with accessories on, where the alternator may not keep up",
    ),
}


# -- the AI, from inside the vehicle ----------------------------------------


class Ai:
    """BeamNG's own driver, commanded in its own Lua VM.

    Parameters are set before the mode, because setting the mode is what starts
    the car and an aggression applied afterwards arrives late.

    Nothing here injects an input. The AI takes the pedals itself, and releases
    the parking brake on its way -- both observed, both left alone.
    """

    def __init__(self, vm: VehicleVM) -> None:
        self.vm = vm

    def configure(self, style: Style) -> None:
        lane = "on" if style.drive_in_lane else "off"
        self.vm.try_call(f"ai.setAggression({style.aggression}) return true")
        self.vm.try_call(f"ai.driveInLane('{lane}') return true")
        self.vm.try_call(f"ai.setSpeedMode('{style.speed_mode}') return true")
        self.vm.try_call("ai.setAvoidCars('on') return true")

    def roam(self) -> None:
        self.vm.call("ai.setMode('span') return true")

    def drive_to(self, waypoint: str) -> None:
        """Drive to a named navgraph node.

        `setTarget` before `setMode`: manual mode with no target is a car told
        to drive somewhere and not told where.
        """
        self.vm.call(f"ai.setTarget('{waypoint}') return true")
        self.vm.call("ai.setMode('manual') return true")

    def halt(self) -> None:
        """Stop where it is, engine running. That is the pathway being sampled."""
        try:
            self.vm.call("ai.setMode('stop') return true")
        except VMError:
            self.vm.try_call("ai.setMode('disabled') return true")

    def release(self) -> None:
        self.vm.try_call("ai.setMode('disabled') return true")

    def is_driving(self) -> bool | None:
        answer = self.vm.try_call("return {driving = ai.isDriving()}")
        if isinstance(answer, dict) and "driving" in answer:
            return bool(answer["driving"])
        return None

    def state(self) -> dict:
        return self.vm.try_call(
            "return {mode = ai.mode, aggression = ai.extAggression, "
            "lane = ai.driveInLaneFlag}", default={}) or {}


# -- finding the route ------------------------------------------------------

#: `core_groundMarkers` is what draws the blue line. Its field names are
#: undocumented and differ between builds, so this asks several ways and
#: reports what the module actually holds when none of them answers.
DESTINATION_LUA = """
local function xyz(v)
  if type(v) ~= 'table' and type(v) ~= 'cdata' then return nil end
  local ok, out = pcall(function()
    return {x = v.x + 0, y = v.y + 0, z = (v.z or 0) + 0}
  end)
  if ok and out.x and out.y then return out end
  return nil
end

local function try(how, fn)
  local ok, value = pcall(fn)
  if not ok then return nil end
  local pos = xyz(value)
  if pos then return {how = how, pos = pos} end
  return nil
end

local found =
  try('groundMarkers.getTargetPos', function()
        return core_groundMarkers.getTargetPos() end)
  or try('groundMarkers.targetPos', function()
        return core_groundMarkers.targetPos end)
  or try('groundMarkers.endWP', function()
        return core_groundMarkers.endWP end)

if found then return jsonEncode(found) end

local keys = {}
local ok = pcall(function()
  for k, v in pairs(core_groundMarkers) do keys[k] = type(v) end
end)
return jsonEncode({how = 'none', keys = ok and keys or 'core_groundMarkers missing'})
"""


def ask(client, tool: str, arguments: dict | None = None,
        attempts: int = 10, wait_s: float = 0.2):
    """Call a tool and wait out the async notice.

    Several tools answer "requested (async); call again in a moment" and
    deliver on a later call. Taking that notice for the answer is how `get_ai`
    once printed the notice instead of the AI's state.
    """
    for attempt in range(attempts):
        if attempt and wait_s:
            time.sleep(wait_s)
        try:
            raw = client.call(tool, arguments or {})
        except MCPError as error:
            return {"error": str(error)}
        if raw is None or is_async_notice(raw):
            continue
        return raw
    return None


def find_destination(client) -> tuple[dict | None, dict]:
    """Read the endpoint of the blue line, or explain why it could not."""
    raw = ask(client, "run_lua", {"code": DESTINATION_LUA})
    if isinstance(raw, dict) and "how" in raw:
        answer = raw
    else:
        try:
            answer = json.loads(str(raw).strip())
        except (json.JSONDecodeError, ValueError, TypeError):
            return None, {"how": "unparseable", "raw": str(raw)[:300]}
    if answer.get("how") in (None, "none"):
        return None, answer
    return answer["pos"], answer


def waypoint_near(client, destination: dict) -> str | None:
    """The navgraph node the AI can actually be sent to.

    `ai.setTarget` takes a waypoint name, not a coordinate, and the game will
    hold a target that does not exist without complaining. So the name comes
    out of the navgraph rather than out of us.
    """
    graph = ask(client, "get_navgraph", {"near": destination, "radius": 100})
    if not isinstance(graph, dict):
        return None
    closest = graph.get("closestRoad")
    if isinstance(closest, dict):
        return closest.get("from") or closest.get("to")
    nodes = graph.get("nodes")
    if isinstance(nodes, list) and nodes and isinstance(nodes[0], dict):
        return nodes[0].get("name")
    return None


# -- the journey ------------------------------------------------------------


class Journey:
    """Watches the drive: schedules stops, and notices arrival.

    Ticked once a second by the collection loop, which is also when a drain
    happens -- so the position it reasons about is never more than a second old.
    """

    def __init__(self, ai: Ai, source, destination, style, seed=0,
                 waypoint=None):
        self.ai = ai
        self.source = source
        self.destination = destination
        self.style = style
        self.waypoint = waypoint
        self.random = random.Random(seed)
        self.arrived = False
        self.stops = 0
        self.stopped_until = 0.0
        self.next_stop_at = self._schedule(0.0)
        self.last_position: tuple[float, float] | None = None

    def _schedule(self, now: float) -> float | None:
        if self.style.stop_every_s is None:
            return None
        return now + self.random.uniform(*self.style.stop_every_s)

    def remaining_m(self) -> float | None:
        if self.last_position is None or self.destination is None:
            return None
        return math.hypot(self.destination["x"] - self.last_position[0],
                          self.destination["y"] - self.last_position[1])

    def tick(self, rows: int, elapsed: float, total: float) -> None:
        # The drained samples carry the position; asking the game again would
        # be a round trip for something already in hand.
        latest = getattr(self.source, "last_sample", None) or {}
        if latest:
            self.last_position = (latest["x_m"], latest["y_m"])

        left = self.remaining_m()
        if left is not None and left < ARRIVED_M:
            self.arrived = True
            return

        if self.stopped_until:
            if elapsed >= self.stopped_until:
                self.stopped_until = 0.0
                self.send_off()
                self.next_stop_at = self._schedule(elapsed)
        elif self.next_stop_at is not None and elapsed >= self.next_stop_at:
            self.stops += 1
            self.stopped_until = elapsed + self.random.uniform(*self.style.stop_for_s)
            self.ai.halt()

        self._report(rows, elapsed, total, left, latest)

    def send_off(self) -> None:
        """Set the style, then start it. At every leg, restarts included."""
        self.ai.configure(self.style)
        if self.waypoint:
            self.ai.drive_to(self.waypoint)
        else:
            self.ai.roam()

    def _report(self, rows, elapsed, total, left, latest) -> None:
        speed = float(latest.get("speed_mps") or 0.0)
        where = "stopped" if self.stopped_until else (
            f"{left:6.0f} m to go" if left is not None else "roaming")
        print(f"\r  {elapsed:5.0f}s / {total:.0f}s   {rows:7d} rows   "
              f"{speed:5.1f} m/s   {where}   {self.stops} stops",
              end="", flush=True)


class WatchedSource(MCPTrajectorySource):
    """A source that remembers its most recent sample, so the journey sees it."""

    last_sample: dict | None = None

    def drain(self) -> list[dict]:
        samples = super().drain()
        if samples:
            self.last_sample = samples[-1]
        return samples


# -- entry point ------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("style", nargs="?", choices=sorted(STYLES))
    parser.add_argument("--minutes", type=float, default=30.0,
                        help="give up after this long if the route is not done")
    parser.add_argument("--to", help="x,y,z instead of the route on the map")
    parser.add_argument("--roam", action="store_true",
                        help="cover the road network instead of going anywhere")
    parser.add_argument("--seed", type=int, default=0,
                        help="which stops happen where, reproducibly")
    parser.add_argument("--ambient", type=float, default=25.0,
                        help="celsius. Readable from the game but not settable "
                             "in it, so this is an assumption the log records")
    parser.add_argument("--show-route", action="store_true",
                        help="print what the game says your route is, and stop")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args(argv)

    if not args.style and not args.show_route:
        parser.error("pick a style, or pass --show-route")

    client = MCPClient(args.endpoint)
    try:
        client.connect()
        status = client.call("get_status")
    except MCPError as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  In BeamNG: Options > Advanced > 'Enable MCP server'.",
              file=sys.stderr)
        return 1

    status = status if isinstance(status, dict) else {}
    level = str(status.get("level", "unknown"))
    vehicle = status.get("vehicle", {})

    if args.roam:
        destination, how = None, {"how": "roaming"}
    else:
        destination, how = resolve_destination(client, args)
    if args.show_route:
        print(json.dumps(how, indent=2, sort_keys=True))
        return 0
    if destination is None and not args.roam:
        return explain_no_route(how)

    style = STYLES[args.style]
    spec = ScenarioSpec(
        name=f"{style.name} route", aggression=style.aggression,
        minutes=args.minutes, accessory_load_a=style.accessory_load_a,
        ambient_temp_c=args.ambient,
        vehicle_config=f"{vehicle.get('jbeam', '?')}/{vehicle.get('configKey', '')}",
        route="roam" if args.roam else f"route:{how.get('how', 'given')}",
        seed=args.seed,
    )
    csv_path = (RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{style.name}"
                       f"_{spec.scenario_hash}.csv")

    source = WatchedSource(client, interval_s=SAMPLE_INTERVAL_S)
    try:
        source.start()
    except RuntimeError as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  Spawn an undamaged car on a real map and try again.",
              file=sys.stderr)
        return 1

    waypoint = None if destination is None else waypoint_near(client, destination)
    if destination is not None and waypoint is None:
        print("\n  No navgraph node near your destination -- the AI has nowhere "
              "to be sent.", file=sys.stderr)
        print("  Pick a destination on a road, or use --roam.", file=sys.stderr)
        source.stop()
        return 1

    print(f"  style     : {style.name} -- {style.why}")
    if destination is None:
        print("  route     : roaming the road network")
    else:
        print(f"  route     : from {how.get('how', 'given')}, to "
              f"({destination['x']:.0f}, {destination['y']:.0f}) "
              f"via node {waypoint}")
    print(f"  level     : {level}")
    print(f"  vehicle   : {vehicle.get('jbeam', '?')}  {source.mass_kg:.0f} kg")
    if style.stop_every_s:
        print(f"  stops     : every {style.stop_every_s[0]:.0f}-"
              f"{style.stop_every_s[1]:.0f}s, for {style.stop_for_s[0]:.0f}-"
              f"{style.stop_for_s[1]:.0f}s, engine running")
    print(f"  sampling  : {1 / SAMPLE_INTERVAL_S:.0f} Hz inside the vehicle VM")
    print(f"  log       : {csv_path}")
    print(f"  giving up : after {args.minutes:.0f} minutes\n")

    ai = Ai(VehicleVM(client))
    journey = Journey(ai, source, destination, style, seed=args.seed,
                      waypoint=waypoint)

    beamng = {"backend": "mcp", "level": level, "endpoint": client.endpoint,
              "vehicle": vehicle.get("jbeam", "unknown"),
              "mass_kg": source.mass_kg, "destination": destination,
              "waypoint": waypoint, "route_from": how.get("how"),
              "style": style.name, "driver": "ai in the vehicle VM"}

    try:
        journey.send_off()
        confirm(ai)
        summary = collect_run(
            source, spec, csv_path, seconds=args.minutes * 60.0,
            beamng=beamng, on_progress=journey.tick,
            should_stop=lambda: journey.arrived,
        )
    finally:
        ai.release()

    print(f"\n\n  {'arrived' if journey.arrived else 'ran out of time'} "
          f"after {summary['duration_s']:.0f} s, {journey.stops} stops")
    print(f"  {summary['rows']} rows, {summary['dropped']} dropped")
    if summary["capture_error"]:
        print(f"  capture error: {summary['capture_error']}")
    print(f"  {csv_path}")
    return 0


def confirm(ai: Ai) -> None:
    """Say whether the AI took the order, before committing thirty minutes."""
    time.sleep(2.0)
    state = ai.state()
    driving = ai.is_driving()
    print(f"  ai        : mode {state.get('mode', '?')}, "
          f"aggression {state.get('aggression', '?')}, "
          f"{'driving' if driving else 'not driving yet'}\n")


def resolve_destination(client, args) -> tuple[dict | None, dict]:
    if args.to:
        try:
            x, y, *rest = [float(part) for part in args.to.split(",")]
        except ValueError:
            return None, {"how": "bad --to", "given": args.to}
        return {"x": x, "y": y, "z": rest[0] if rest else 0.0}, {"how": "--to"}
    return find_destination(client)


def explain_no_route(how: dict) -> int:
    print("\n  No route found. Set a destination in the map first -- the blue "
          "line on the road is the route.", file=sys.stderr)
    print("  Or drive without one:  py drive_route.py aggressive --roam\n",
          file=sys.stderr)
    print(json.dumps(how, indent=2, sort_keys=True), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
