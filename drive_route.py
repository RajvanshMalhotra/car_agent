#!/usr/bin/env python3
"""Drive the route you picked on the map, in a chosen style, and log it.

    py drive_route.py aggressive
    py drive_route.py economical
    py drive_route.py stops
    py drive_route.py --show-route        what the game says your route is

Set a destination in BeamNG's map first -- the blue line on the road is the
route. This reads that destination, hands it to the game's own AI, and records
the drive at 100 Hz through the vehicle's Lua VM.

Three styles, chosen because they are the three that reach a starter battery:

    aggressive   high engine load, hot engine bay -- the corrosion pathway
    economical   the same route, gently -- the baseline to compare against
    stops        frequent stops with the engine idling -- the recharge-deficit
                 pathway, where the alternator at idle RPM may not cover the
                 electrical load

Hard braking is deliberately not a style. It barely touches an SLI battery.

The car is always handed back when this exits -- normally, on Ctrl+C, or on a
crash. Leaving the AI engaged means the game keeps driving after the script has
gone.
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
    """How to drive. One scalar reaches the AI; the rest is what we do to it."""

    name: str
    aggression: float
    #: BeamNG's speed mode. 'limit' obeys posted limits, 'off' ignores them.
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


# -- finding the route ------------------------------------------------------

#: `core_groundMarkers` is what draws the blue line. Its field names are
#: undocumented and differ between builds, so this asks in several ways and
#: reports which one answered rather than assuming any of them.
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
  try('groundMarkers.targetPos', function()
        return core_groundMarkers.targetPos end)
  or try('groundMarkers.getTargetPos', function()
        return core_groundMarkers.getTargetPos() end)
  or try('groundMarkers.endWP', function()
        return core_groundMarkers.endWP end)
  or try('groundMarkers.getPathTarget', function()
        return core_groundMarkers.getPath()[#core_groundMarkers.getPath()] end)

if found then return jsonEncode(found) end

-- Nothing recognised. Hand back what the module actually contains so the next
-- attempt is informed rather than another guess.
local keys = {}
local ok = pcall(function()
  for k, v in pairs(core_groundMarkers) do keys[k] = type(v) end
end)
return jsonEncode({how = 'none', keys = ok and keys or 'core_groundMarkers missing'})
"""


def find_destination(client) -> tuple[dict | None, dict]:
    """Read the endpoint of the blue line, or explain why it could not."""
    try:
        raw = client.call("run_lua", {"code": DESTINATION_LUA})
    except MCPError as error:
        return None, {"how": "error", "error": str(error)}
    if isinstance(raw, dict):
        answer = raw
    else:
        try:
            answer = json.loads(str(raw).strip())
        except (json.JSONDecodeError, ValueError):
            return None, {"how": "unparseable", "raw": str(raw)[:300]}
    if answer.get("how") in (None, "none"):
        return None, answer
    return answer["pos"], answer


# -- driving ----------------------------------------------------------------


#: What to ask `drive_to` for, richest first. BeamNG's argument names and enum
#: values are undocumented, and **a refused call does not raise** -- it comes
#: back as text and the car simply never sets off, which from outside looks
#: exactly like a broken AI. So drop arguments until one is taken.
DRIVE_TO_ARGUMENTS = (
    ("aggression", "avoidCars", "driveInLane", "routeSpeedMode"),
    ("aggression", "avoidCars", "driveInLane"),
    ("aggression", "avoidCars"),
    ("aggression",),
    (),
)


def refused(result) -> bool:
    """BeamNG reports a bad call in the returned text rather than by raising."""
    text = str(result).lower()
    return any(word in text for word in ("fail", "error", "unknown", "invalid",
                                         "bad argument", "nil value"))


def ask(client, tool: str, arguments: dict | None = None,
        attempts: int = 10, wait_s: float = 0.2):
    """Call a tool and wait out the async notice.

    Several tools answer "requested (async); call again in a moment" and
    deliver on a later call. Taking that notice for the answer is how the first
    probe run came back shifted by two, and how `get_ai` printed the notice
    instead of the AI's state.
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


def vehicle_lua(client, code: str, attempts: int = 10, wait_s: float = 0.2):
    """Run Lua in the vehicle VM and hand back the parsed answer."""
    raw = ask(client, "run_lua_vehicle", {"code": code}, attempts, wait_s)
    if raw is None:
        return None
    for value in (raw.values() if isinstance(raw, dict) else [raw]):
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, ValueError):
                return value
    return raw


#: Ways to let a car go, most likely first. `inject_input` sets an input for a
#: moment and the vehicle's own input system reasserts itself, which is why the
#: parking brake survived being told to release. These go through the vehicle's
#: input system instead, and each one is checked rather than assumed.
RELEASES = (
    ("input.event, FILTER_DIRECT",
     "input.event('parkingbrake', 0, FILTER_DIRECT) "
     "input.event('brake', 0, FILTER_DIRECT) "
     "input.event('throttle', 0, FILTER_DIRECT)"),
    ("input.event, default filter",
     "input.event('parkingbrake', 0) input.event('brake', 0)"),
    ("input.event, FILTER_AI",
     "input.event('parkingbrake', 0, FILTER_AI) input.event('brake', 0, FILTER_AI)"),
    ("electrics, written directly",
     "electrics.values.parkingbrake = 0 "
     "electrics.values.parkingbrake_input = 0 "
     "electrics.values.brake = 0 electrics.values.brake_input = 0"),
)

HOLDING_LUA = ("return jsonEncode({parkingbrake = electrics.values.parkingbrake, "
               "brake = electrics.values.brake, "
               "throttle = electrics.values.throttle, "
               "gear = electrics.values.gearIndex})")


def held_by(client) -> dict:
    """What is currently holding the car still, as the vehicle reports it."""
    state = vehicle_lua(client, HOLDING_LUA)
    return state if isinstance(state, dict) else {}


def free_the_car(client, vehicle_id: int, report=None) -> str | None:
    """Release anything holding the car still, and confirm it let go.

    A spawned vehicle can sit with its parking brake on, and the AI will not
    override it -- which from outside is indistinguishable from a refused
    `drive_to`. Returns the name of whatever worked, or None.
    """
    if not held_by(client).get("parkingbrake"):
        return "already free"

    for name, action in RELEASES:
        vehicle_lua(client, f"""
local ok, err = pcall(function() {action} end)
return jsonEncode({{ok = ok, error = (not ok) and tostring(err) or nil}})
""")
        time.sleep(0.3)
        if not held_by(client).get("parkingbrake"):
            if report:
                report(f"released the parking brake with {name}")
            return name
    if report:
        report("could not release the parking brake by any means")
    return None


def send_off(client, vehicle_id: int, destination: dict, style: Style,
             accepted: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Point the game's AI at the destination, and confirm that it agreed.

    Returns the argument shape the game accepted, so a later call can pass it
    back and skip the probing. Raises if nothing is accepted -- the alternative
    is a silent refusal, which is what "the car doesn't move at all" looks like.

    `drive_to` is issued once per leg and left alone: re-sending it makes the AI
    throw away the route it has planned and start again.
    """
    free_the_car(client, vehicle_id)
    client.call("set_ai", {"id": vehicle_id, "mode": "manual",
                           "aggression": style.aggression, "avoidCars": True})

    available = {
        "aggression": style.aggression,
        "avoidCars": True,
        "driveInLane": style.drive_in_lane,
        "routeSpeedMode": style.speed_mode,
    }
    attempts = []
    for names in ((accepted,) if accepted is not None else DRIVE_TO_ARGUMENTS):
        arguments = {"id": vehicle_id, "pos": destination}
        arguments.update({name: available[name] for name in names})
        result = client.call("drive_to", arguments)
        if not refused(result):
            return names
        attempts.append((names, str(result)[:120]))

    raise RuntimeError(
        "BeamNG would not accept any form of drive_to:\n    "
        + "\n    ".join(f"{list(names) or 'pos only'}: {why}"
                         for names, why in attempts)
    )


def halt(client, vehicle_id: int) -> None:
    """Stop the car where it is, engine still running."""
    client.call("set_ai", {"id": vehicle_id, "mode": "disabled"})
    client.call("inject_input", {"id": vehicle_id, "event": "throttle", "value": 0.0})
    client.call("inject_input", {"id": vehicle_id, "event": "brake", "value": 1.0})


def release(client, vehicle_id: int) -> None:
    client.call("inject_input", {"id": vehicle_id, "event": "brake", "value": 0.0})


def hand_back(client, vehicle_id: int) -> None:
    """Whatever happened, the game gets its car back."""
    for event in ("throttle", "brake", "steering"):
        try:
            client.call("inject_input",
                        {"id": vehicle_id, "event": event, "value": 0.0})
        except MCPError:
            pass
    try:
        client.call("set_ai", {"id": vehicle_id, "mode": "disabled"})
    except MCPError:
        pass


class Journey:
    """Watches the drive: schedules stops, and notices arrival.

    Ticked once a second by the collection loop, which is also when a drain
    happens -- so the position it reasons about is never more than a second old.
    """

    def __init__(self, client, vehicle_id, source, destination, style, seed=0,
                 accepted=None):
        self.accepted = accepted
        self.client = client
        self.vehicle_id = vehicle_id
        self.source = source
        self.destination = destination
        self.style = style
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
        if self.last_position is None:
            return None
        dx = self.destination["x"] - self.last_position[0]
        dy = self.destination["y"] - self.last_position[1]
        return math.hypot(dx, dy)

    def tick(self, rows: int, elapsed: float, total: float) -> None:
        # The samples already drained carry the position; asking the game again
        # would be a round trip for something we have.
        latest = getattr(self.source, "last_sample", None)
        if latest:
            self.last_position = (latest["x_m"], latest["y_m"])

        left = self.remaining_m()
        if left is not None and left < ARRIVED_M:
            self.arrived = True
            return

        if self.stopped_until:
            if elapsed >= self.stopped_until:
                self.stopped_until = 0.0
                release(self.client, self.vehicle_id)
                self.accepted = send_off(self.client, self.vehicle_id,
                                         self.destination, self.style,
                                         accepted=self.accepted)
                self.next_stop_at = self._schedule(elapsed)
        elif self.next_stop_at is not None and elapsed >= self.next_stop_at:
            self.stops += 1
            self.stopped_until = elapsed + self.random.uniform(*self.style.stop_for_s)
            halt(self.client, self.vehicle_id)

        self._report(rows, elapsed, total, left)

    def _report(self, rows, elapsed, total, left) -> None:
        where = "stopped" if self.stopped_until else (
            f"{left:6.0f} m to go" if left is not None else "driving")
        print(f"\r  {elapsed:5.0f}s / {total:.0f}s   {rows:7d} rows   "
              f"{where}   {self.stops} stops", end="", flush=True)


class WatchedSource(MCPTrajectorySource):
    """A source that remembers its most recent sample, so the journey can see it."""

    last_sample: dict | None = None

    def drain(self) -> list[dict]:
        samples = super().drain()
        if samples:
            self.last_sample = samples[-1]
        return samples


# -- entry point ------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("style", nargs="?", choices=sorted(STYLES),
                        help="how to drive")
    parser.add_argument("--minutes", type=float, default=30.0,
                        help="give up after this long if the route is not done")
    parser.add_argument("--to", help="x,y,z instead of the route on the map")
    parser.add_argument("--seed", type=int, default=0,
                        help="which stops happen where, reproducibly")
    parser.add_argument("--ambient", type=float, default=25.0,
                        help="celsius. Readable from the game but not settable "
                             "in it, so this is an assumption the log records")
    parser.add_argument("--show-route", action="store_true",
                        help="print what the game says your route is, and stop")
    parser.add_argument("--diagnose", action="store_true",
                        help="find out why the car will not move, and stop")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args(argv)

    if not args.style and not (args.show_route or args.diagnose):
        parser.error("pick a style, or pass --show-route or --diagnose")
    if args.diagnose and not args.style:
        args.style = "economical"

    client = MCPClient(args.endpoint)
    try:
        client.connect()
        status = client.call("get_status")
        vehicle_id = client.call("get_player_vehicle_id")
    except MCPError as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  In BeamNG: Options > Advanced > 'Enable MCP server'.",
              file=sys.stderr)
        return 1

    status = status if isinstance(status, dict) else {}
    level = str(status.get("level", "unknown"))
    vehicle = status.get("vehicle", {})

    destination, how = resolve_destination(client, args)
    if args.show_route:
        print(json.dumps(how, indent=2, sort_keys=True))
        return 0
    if destination is None:
        return explain_no_route(how)

    style = STYLES[args.style]
    spec = ScenarioSpec(
        name=f"{style.name} route", aggression=style.aggression,
        minutes=args.minutes, accessory_load_a=style.accessory_load_a,
        ambient_temp_c=args.ambient,
        vehicle_config=f"{vehicle.get('jbeam', '?')}/{vehicle.get('configKey', '')}",
        route=f"route:{how.get('how', 'given')}", seed=args.seed,
    )
    csv_path = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{style.name}_{spec.scenario_hash}.csv"

    if args.diagnose:
        return diagnose(client, vehicle_id, destination, STYLES[args.style])

    source = WatchedSource(client, interval_s=SAMPLE_INTERVAL_S)
    try:
        source.start()
    except RuntimeError as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  Spawn an undamaged car on a real map and try again.",
              file=sys.stderr)
        return 1

    print(f"  style     : {style.name} -- {style.why}")
    print(f"  route     : from {how.get('how', 'given')}, to "
          f"({destination['x']:.0f}, {destination['y']:.0f})")
    print(f"  level     : {level}")
    print(f"  vehicle   : {vehicle.get('jbeam', '?')}  {source.mass_kg:.0f} kg")
    if style.stop_every_s:
        print(f"  stops     : every {style.stop_every_s[0]:.0f}-"
              f"{style.stop_every_s[1]:.0f}s, for "
              f"{style.stop_for_s[0]:.0f}-{style.stop_for_s[1]:.0f}s, "
              f"engine running")
    print(f"  sampling  : {1 / SAMPLE_INTERVAL_S:.0f} Hz inside the vehicle VM")
    print(f"  log       : {csv_path}")
    print(f"  giving up : after {args.minutes:.0f} minutes\n")

    journey = Journey(client, vehicle_id, source, destination, style, seed=args.seed)
    beamng = {"backend": "mcp", "level": level, "endpoint": client.endpoint,
              "vehicle": vehicle.get("jbeam", "unknown"),
              "mass_kg": source.mass_kg, "destination": destination,
              "route_from": how.get("how"), "style": style.name}

    try:
        journey.accepted = send_off(client, vehicle_id, destination, style)
        print(f"  accepted  : drive_to{list(journey.accepted) or ' (pos only)'}\n")
        summary = collect_run(
            source, spec, csv_path, seconds=args.minutes * 60.0,
            beamng=beamng, on_progress=journey.tick,
            should_stop=lambda: journey.arrived,
        )
    finally:
        hand_back(client, vehicle_id)

    print(f"\n\n  {'arrived' if journey.arrived else 'ran out of time'} "
          f"after {summary['duration_s']:.0f} s, {journey.stops} stops")
    print(f"  {summary['rows']} rows, {summary['dropped']} dropped")
    if summary["capture_error"]:
        print(f"  capture error: {summary['capture_error']}")
    print(f"  {csv_path}")
    return 0


#: The electrics worth seeing when a car will not move. Each one is a
#: different reason, and they are indistinguishable from the outside.
WHY_STUCK = ("parkingbrake", "ignitionLevel", "engineRunning", "rpm", "gear",
             "throttle", "brake", "clutch", "wheelspeed", "fuel", "damage")


def read_electrics(client, keys) -> dict:
    """Read a few electrics, waiting out the async queue."""
    code = ("return jsonEncode({"
            + ", ".join(f"{k} = electrics.values.{k}" for k in keys)
            + "})")
    state = vehicle_lua(client, code)
    return state if isinstance(state, dict) else {}


def diagnose(client, vehicle_id: int, destination: dict, style: Style) -> int:
    """Work out why the car is not moving, instead of guessing at it.

    Three things stop a car that has been told to drive, and they look
    identical from outside: a refused `drive_to`, a parking brake nobody
    released, and an AI that took the order and cannot route to the target.
    """
    print("\n  BEFORE")
    before = read_electrics(client, WHY_STUCK)
    for key in WHY_STUCK:
        if key in before:
            print(f"    {key:16} {before[key]}")
    if before.get("parkingbrake"):
        print("    ^ the parking brake is on. The AI will not override it.")

    print("\n  LETTING THE CAR GO")
    freed = free_the_car(client, vehicle_id, report=lambda why: print(f"    {why}"))
    print(f"    now: {json.dumps(held_by(client), sort_keys=True)}")
    if freed is None:
        print("    Send this back -- none of the release routes worked.")

    print("\n  SENDING OFF")
    try:
        accepted = send_off(client, vehicle_id, destination, style)
    except RuntimeError as error:
        print(f"    {error}")
        return 1
    print(f"    accepted: drive_to{list(accepted) or ' (pos only)'}")

    print("\n  WHAT THE AI THINKS IT IS DOING")
    state = ask(client, "get_ai", {"id": vehicle_id})
    print(f"    {json.dumps(state, sort_keys=True) if isinstance(state, dict) else state}")

    print("\n  AFTER FIVE SECONDS")
    time.sleep(5.0)
    after = read_electrics(client, WHY_STUCK)
    for key in WHY_STUCK:
        if key in after:
            changed = "" if before.get(key) == after.get(key) else "   <- changed"
            print(f"    {key:16} {after[key]}{changed}")

    moving = float(after.get("wheelspeed") or 0.0) > 0.5
    print(f"\n  {'The car is moving.' if moving else 'The car is still stationary.'}")
    if not moving:
        print("  Nothing above ruled it out. Send this whole output back.")
    hand_back(client, vehicle_id)
    return 0


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
    print("  If you have set one and this still says no, the field name "
          "differs in this build. What the game does have:", file=sys.stderr)
    print(json.dumps(how, indent=2, sort_keys=True), file=sys.stderr)
    print("\n  Send that back, or drive with an explicit target:  "
          "--to x,y,z", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
