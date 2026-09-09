#!/usr/bin/env python3
"""Find out what BeamNG's vehicle Lua VM actually exposes.

    py windows_lua_probe.py

`get_electrics` returns a curated slice of vehicle state. `run_lua_vehicle`
runs code inside the vehicle's own physics VM, where all of it lives --
accelerometer, road grade, engine torque, per-wheel angular velocity.

**`run_lua_vehicle` is asynchronous.** It queues the code and answers with
"queued in vehicle VM(s)"; the result arrives on a *later* call, keyed by
vehicle id and with nothing saying which call it belongs to. Reading the reply
as if it belonged to the request silently shifts every result by two probes,
which is exactly what happened the first time this ran. So each probe now
carries a tag, and the runner drains until that tag comes back.

The GE VM (`run_lua`) is synchronous and needs none of this.

Field names here are undocumented and version-dependent, so nothing is
assumed: every probe runs under `pcall` and reports whether it resolved.

Writes `lua_probe.json`. Reads only, with one exception: `--set-temperature`
sets ambient, reads it back, and restores the original value.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError  # noqa: E402

#: A JSON encoder defined inline rather than relying on the VM having one.
#: `jsonEncode` exists in some builds and not others, and finding that out the
#: hard way costs a round trip per probe.
SERIALISER = r"""
local function q(s)
  return '"' .. tostring(s):gsub('[%c\\"]', function(c)
    if c == '\\' then return '\\\\' elseif c == '"' then return '\\"' end
    return string.format('\\u%04x', string.byte(c))
  end) .. '"'
end
local function enc(v, depth)
  local t = type(v)
  if t == 'number' then
    if v ~= v or v == math.huge or v == -math.huge then return 'null' end
    return string.format('%.6g', v)
  elseif t == 'boolean' then return tostring(v)
  elseif t == 'string' then return q(v)
  elseif t == 'nil' then return 'null'
  elseif t == 'table' and depth > 0 then
    local parts = {}
    for k, vv in pairs(v) do
      parts[#parts + 1] = q(k) .. ':' .. enc(vv, depth - 1)
    end
    return '{' .. table.concat(parts, ',') .. '}'
  end
  return q('<' .. t .. '>')
end
local function try(fn)
  local ok, value = pcall(fn)
  if not ok then return '{"ok":false,"error":' .. q(tostring(value)) .. '}' end
  return '{"ok":true,"value":' .. enc(value, 3) .. '}'
end
local function tagged(name, fn)
  return '{"tag":' .. q(name) .. ',"result":' .. try(fn) .. '}'
end
"""

#: The last two unknowns, plus enough context to tell a wrong name from a
#: missing feature. Each value is a Lua function body.
PROBES: dict[str, str] = {
    # m -- the one required row with no working accessor. `obj:getTotalMass()`
    # came back nil, so these are the plausible alternatives.
    "mass_node_sum": (
        "local n = obj:getNodeCount() local total = 0 "
        "for i = 0, n - 1 do total = total + obj:getNodeMass(i) end "
        "return {nodeCount = n, totalMass = total}"
    ),
    "mass_physics": "return obj:getPhysicsMass()",
    "mass_beamstate": "return {mass = beamstate.getTotalMass()}",
    "mass_vdata": "return {weight = v.data.totalWeight}",

    # The module list lost to the shift last time.
    "globals": (
        "local names = {} for k, _ in pairs(_G) do names[#names + 1] = k end "
        "table.sort(names) return names"
    ),

    # Whether the vehicle can see ambient at all, or only the GE side can.
    "vehicle_ambient": (
        "return {airflow = electrics.values.airflowspeed, "
        "airspeed = electrics.values.airspeed, "
        "altitude = electrics.values.altitude, "
        "watertemp = electrics.values.watertemp, "
        "oiltemp = electrics.values.oiltemp}"
    ),
}

#: Game-engine side. Synchronous, so no draining needed.
GE_PROBES: dict[str, str] = {
    "mass_from_ge": (
        "return try(function() "
        "return be:getPlayerVehicle(0):getTotalMass() end)"
    ),
    "temperature_now": (
        "return try(function() return {kelvin = core_environment.getTemperatureK(), "
        "celsius = core_environment.getState().temperatureC} end)"
    ),
}

#: Ambient temperature is the dominant driver of grid corrosion, and until this
#: dump nothing suggested the game exposed it at all. Whether it can be *set*
#: decides whether an ambient sweep is simulation or arithmetic -- so it is
#: worth one deliberate write. The original value is put back either way.
SET_TEMPERATURE = r"""
return try(function()
  local before = core_environment.getState().temperatureC
  local state = core_environment.getState()
  state.temperatureC = 42
  core_environment.setState(state)
  local after = core_environment.getState().temperatureC
  local kelvin = core_environment.getTemperatureK()
  local restore = core_environment.getState()
  restore.temperatureC = before
  core_environment.setState(restore)
  return {before = before, after = after, kelvinWhenHot = kelvin,
          restored = core_environment.getState().temperatureC}
end)
"""

#: How long to keep draining before calling a probe lost. The queue turns over
#: in a frame or two; this is generous.
DRAIN_SECONDS = 5.0
DRAIN_INTERVAL_S = 0.15


def _payloads(raw) -> list[dict]:
    """Pull whatever finished results a vehicle-VM reply carries.

    A reply is either the "queued" notice (a string) or a map of vehicle id to
    the JSON our Lua built.
    """
    if not isinstance(raw, dict):
        return []
    found = []
    for value in raw.values():
        if not isinstance(value, str):
            continue
        try:
            found.append(json.loads(value))
        except (json.JSONDecodeError, ValueError):
            continue
    return found


def run_vehicle(client: MCPClient, name: str, body: str) -> dict:
    """Issue a tagged probe and drain the queue until its tag comes back."""
    code = f"{SERIALISER}\nreturn tagged({name!r}, function()\n{body}\nend)"
    noop = f"{SERIALISER}\nreturn tagged('__drain__', function() return 0 end)"

    seen: dict[str, dict] = {}

    def absorb(raw) -> None:
        for payload in _payloads(raw):
            tag = payload.get("tag")
            if tag and tag != "__drain__":
                seen[tag] = payload.get("result", {})

    try:
        absorb(client.call("run_lua_vehicle", {"code": code}))
        deadline = time.monotonic() + DRAIN_SECONDS
        while name not in seen and time.monotonic() < deadline:
            time.sleep(DRAIN_INTERVAL_S)
            absorb(client.call("run_lua_vehicle", {"code": noop}))
    except MCPError as error:
        return {"ok": False, "error": f"MCP: {error}"}

    if name not in seen:
        return {"ok": False, "error": f"no result within {DRAIN_SECONDS:.0f}s"}
    return seen[name]


def run_ge(client: MCPClient, body: str) -> dict:
    try:
        raw = client.call("run_lua", {"code": SERIALISER + "\n" + body})
    except MCPError as error:
        return {"ok": False, "error": f"MCP: {error}"}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw).strip())
    except (json.JSONDecodeError, ValueError):
        return {"ok": False, "error": f"unparseable: {str(raw)[:200]}"}


def summarise(name: str, result: dict) -> str:
    if not result.get("ok"):
        return f"  [ no ] {name:20} {str(result.get('error'))[:76]}"
    value = result.get("value")
    if isinstance(value, dict):
        shown = ", ".join(f"{k}={v}" for k, v in list(value.items())[:4])
        return f"  [ yes] {name:20} {shown[:76]}"
    if isinstance(value, list):
        return f"  [ yes] {name:20} {len(value)} entries"
    return f"  [ yes] {name:20} {value}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", default="lua_probe.json")
    parser.add_argument("--set-temperature", action="store_true",
                        help="set ambient to 42 C, read it back, restore it")
    args = parser.parse_args(argv)

    print("=" * 70)
    print(f"BeamNG vehicle Lua  {args.endpoint}")
    print("=" * 70)

    findings: dict[str, dict] = {}
    try:
        with MCPClient(args.endpoint) as client:
            info = (client.server_info or {}).get("serverInfo", {})
            print(f"  connected: {info.get('name', '?')}\n")

            print("VEHICLE VM  (run_lua_vehicle, drained)")
            for name, body in PROBES.items():
                result = run_vehicle(client, name, body)
                findings[name] = result
                print(summarise(name, result))

            print("\nGAME ENGINE VM  (run_lua)")
            for name, body in GE_PROBES.items():
                result = run_ge(client, body)
                findings[f"ge_{name}"] = result
                print(summarise(f"ge_{name}", result))

            if args.set_temperature:
                print("\nAMBIENT TEMPERATURE  (writes, then restores)")
                result = run_ge(client, SET_TEMPERATURE)
                findings["ge_set_temperature"] = result
                print(summarise("ge_set_temperature", result))
            else:
                print("\n  ambient write skipped. Add --set-temperature to test it.")
    except MCPError as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  In BeamNG: Options > Advanced > 'Enable MCP server'.",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.write_text(json.dumps(findings, indent=2, sort_keys=True))
    print(f"\n  Written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
