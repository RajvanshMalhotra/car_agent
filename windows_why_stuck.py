#!/usr/bin/env python3
"""Find out what is stopping this car, by looking rather than guessing.

    py windows_why_stuck.py

Three fixes have not moved it, so this stops fixing and gathers evidence. It
reads the two things nobody has looked at yet:

**The vehicle's input state.** `inject_input` sets an input through the
vehicle's input system, and every run so far has injected zeros -- in
`free_the_car` before driving, and in `hand_back` on the way out of every
attempt. If those latch, an injected `throttle = 0` outranks whatever the AI
asks for, permanently, and no amount of correct routing will move the car. The
one thing that ever did drive -- `drive.py` on 8 September -- never injected
while the AI had the car.

**The AI module inside the vehicle VM.** `ai` is a global in the vehicle's own
Lua, so the AI can be commanded directly and `set_ai`, `drive_to` and the MCP
wrapper drop out of the question entirely.

Then it runs one experiment at a time, changing one thing, and reports the
metres covered by each. Nothing here is a fix. It is here to say which fix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collect.decode import is_async_notice  # noqa: E402
from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError  # noqa: E402

#: Long enough that a car which is going to move has moved.
SETTLE_S = 6.0
#: A car rocking on its springs covers centimetres. A driving one covers tens
#: of metres in six seconds.
MOVED_M = 3.0


def vehicle_lua(client, code: str, attempts: int = 12, wait_s: float = 0.25):
    """Run Lua in the vehicle VM, waiting out the async notice."""
    for attempt in range(attempts):
        if attempt:
            time.sleep(wait_s)
        try:
            raw = client.call("run_lua_vehicle", {"code": code})
        except MCPError as error:
            return {"mcp_error": str(error)}
        if raw is None or is_async_notice(raw):
            continue
        for value in (raw.values() if isinstance(raw, dict) else [raw]):
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, ValueError):
                    return {"raw": value[:400]}
        return raw
    return {"timeout": True}


WRAP = """
local ok, value = pcall(function() %s end)
if not ok then return jsonEncode({error = tostring(value)}) end
return jsonEncode(value == nil and {ok = true} or value)
"""


def run(client, body: str):
    return vehicle_lua(client, WRAP % body)


# -- what we have never looked at -------------------------------------------

#: `input.state` is where a latched input lives, with the filter that set it.
#: If our injected zeros are sitting here, they are the answer.
INPUT_STATE = """
local out = {}
for _, name in ipairs({'throttle', 'brake', 'steering', 'parkingbrake', 'clutch'}) do
  local entry = input.state and input.state[name]
  if entry then
    out[name] = {val = entry.val, filter = entry.filter,
                 smootherKBD = entry.smootherKBD ~= nil,
                 angle = entry.angle}
  else
    out[name] = 'absent'
  end
end
out['_electrics'] = {throttle = electrics.values.throttle,
                     brake = electrics.values.brake,
                     parkingbrake = electrics.values.parkingbrake,
                     wheelspeed = electrics.values.wheelspeed,
                     rpm = electrics.values.rpm,
                     gearIndex = electrics.values.gearIndex,
                     ignitionLevel = electrics.values.ignitionLevel}
return out
"""

#: Whatever the AI will tell us about itself, from inside the vehicle.
AI_STATE = """
local out = {}
if ai == nil then return {ai = 'absent from the vehicle VM'} end
for k, v in pairs(ai) do
  local t = type(v)
  if t == 'number' or t == 'string' or t == 'boolean' then out[k] = v
  else out['fn_' .. k] = t end
end
if ai.getState then
  local ok, state = pcall(ai.getState)
  if ok and type(state) == 'table' then
    for k, v in pairs(state) do
      local t = type(v)
      if t == 'number' or t == 'string' or t == 'boolean' then
        out['state_' .. k] = v
      end
    end
  end
end
return out
"""

POSITION = """
local p = obj:getPosition()
return {x = p.x, y = p.y, z = p.z, speed = electrics.values.wheelspeed}
"""


def where(client):
    answer = run(client, POSITION)
    if not isinstance(answer, dict) or "x" not in answer:
        return None
    return answer


def moved(client, label: str, before) -> bool:
    """Wait, then report the ground covered. One number, unambiguous."""
    time.sleep(SETTLE_S)
    after = where(client)
    if before is None or after is None:
        print(f"    {label:38} could not read position")
        return False
    distance = ((after["x"] - before["x"]) ** 2
                + (after["y"] - before["y"]) ** 2) ** 0.5
    went = distance > MOVED_M
    print(f"    {label:38} {distance:7.1f} m   {after['speed']:6.2f} m/s   "
          f"{'MOVING' if went else 'stationary'}")
    return went


# -- the experiments, one variable each -------------------------------------


def experiment(client, label: str, body: str) -> bool:
    before = where(client)
    result = run(client, body)
    if isinstance(result, dict) and result.get("error"):
        print(f"    {label:38} refused: {result['error'][:60]}")
        return False
    return moved(client, label, before)


#: Ways to clear an input that has been latched by injection. `input.event`
#: with a value writes; the question is whether anything un-writes.
CLEARERS = (
    ("input.event throttle/brake back to AI filter",
     "input.event('throttle', 0, FILTER_AI) "
     "input.event('brake', 0, FILTER_AI) "
     "input.event('steering', 0, FILTER_AI)"),
    ("input.state entries deleted outright",
     "for _, n in ipairs({'throttle','brake','steering'}) do "
     "if input.state and input.state[n] then input.state[n].val = nil end end"),
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args(argv)

    client = MCPClient(args.endpoint)
    try:
        client.connect()
    except MCPError as error:
        print(f"\n  {error}", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print("  THE INPUT STATE  -- a latched zero here outranks the AI")
    print("=" * 70)
    print(json.dumps(run(client, "return (function() " + INPUT_STATE + " end)()"),
                     indent=2, sort_keys=True))

    print("\n" + "=" * 70)
    print("  THE AI, FROM INSIDE THE VEHICLE")
    print("=" * 70)
    print(json.dumps(run(client, "return (function() " + AI_STATE + " end)()"),
                     indent=2, sort_keys=True))

    print("\n" + "=" * 70)
    print("  WHAT MOVES IT  -- one variable at a time")
    print("=" * 70)
    print(f"    {'attempt':38} {'moved':>7}   {'speed':>6}")

    # 1. The AI commanded directly, with nothing else touched. If this drives,
    #    every MCP wrapper between us and it was the problem.
    span = experiment(client, "ai.setMode('span'), nothing else touched",
                      "ai.setMode('span')")
    run(client, "ai.setMode('disabled')")
    time.sleep(1.0)

    # 2. The same, after clearing whatever the injections may have latched.
    cleared = False
    if not span:
        for label, action in CLEARERS:
            run(client, action)
            time.sleep(0.5)
            cleared = experiment(client, f"span after: {label}",
                                 "ai.setMode('span')")
            run(client, "ai.setMode('disabled')")
            time.sleep(1.0)
            if cleared:
                break

    # 3. A physics reset, which puts the vehicle back to a known state.
    reset = False
    if not (span or cleared):
        try:
            client.call("reset_vehicle")
        except MCPError as error:
            print(f"    reset_vehicle refused: {error}")
        time.sleep(2.0)
        reset = experiment(client, "span after reset_vehicle",
                           "ai.setMode('span')")
        run(client, "ai.setMode('disabled')")

    print("\n" + "=" * 70)
    print("  AFTERWARDS")
    print("=" * 70)
    print(json.dumps(run(client, "return (function() " + INPUT_STATE + " end)()"),
                     indent=2, sort_keys=True))

    print("\n  WHAT THIS SAYS")
    if span:
        print("    The AI drives when commanded from inside the vehicle VM.")
        print("    Everything between us and it -- set_ai, drive_to -- is what")
        print("    was failing. drive_route.py should command the AI directly.")
    elif cleared:
        print("    The car drove once the latched inputs were cleared.")
        print("    Our own injected zeros were holding it. Every inject_input")
        print("    on throttle, brake and steering has to go.")
    elif reset:
        print("    Only a physics reset freed it, so something latched that")
        print("    nothing else clears. Reset before each run, and stop")
        print("    injecting inputs at all.")
    else:
        print("    Nothing moved it, including the AI commanded directly.")
        print("    That rules out this whole layer. Send this output back --")
        print("    the input state above is the part that matters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
