#!/usr/bin/env python3
"""Find out what BeamNG's vehicle Lua VM actually exposes.

    py windows_lua_probe.py

`run_lua_vehicle` runs code inside the vehicle's own physics VM, where the
full state lives -- accelerometer, road grade, mass, engine torque, wheel
speeds. `get_electrics` shows only a curated slice of it, and asynchronously
at that.

The names below are how BeamNG's vehicle Lua is *believed* to be shaped. They
are undocumented and change between versions, so nothing here is assumed: each
probe runs under `pcall` and reports whether it resolved. A logger built on a
field name that silently reads `nil` looks exactly like a logger built on a
field that is always zero, which is the failure mode this exists to prevent.

Writes `lua_probe.json`. Send that back.

Nothing here drives, moves or damages the car. Every probe reads.
"""

from __future__ import annotations

import argparse
import json
import sys
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
      local ek = enc(vv, depth - 1)
      if ek ~= nil then parts[#parts + 1] = q(k) .. ':' .. ek end
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
"""

#: Each probe answers one row of the parameter table. Kept separate so one
#: missing module does not take the rest of the dump with it.
PROBES: dict[str, str] = {
    # Everything get_electrics curates, plus everything it does not.
    "electrics_values": "return try(function() return electrics.values end)",

    # a_x. The IMU, not a differentiated speed.
    "sensors": "return try(function() return sensors end)",
    "sensors_gxyz": (
        "return try(function() "
        "return {gx=sensors.gx, gy=sensors.gy, gz=sensors.gz, "
        "gx2=sensors.gx2, gy2=sensors.gy2, gz2=sensors.gz2} end)"
    ),

    # m. Static, but needed for any force or power calculation.
    "mass": "return try(function() return obj:getTotalMass() end)",

    # theta. The z component of the forward vector is sin(pitch), i.e. grade.
    "direction_vector": (
        "return try(function() local d = obj:getDirectionVector() "
        "return {x=d.x, y=d.y, z=d.z} end)"
    ),
    "direction_vector_up": (
        "return try(function() local d = obj:getDirectionVectorUp() "
        "return {x=d.x, y=d.y, z=d.z} end)"
    ),
    "velocity": (
        "return try(function() local v = obj:getVelocity() "
        "return {x=v.x, y=v.y, z=v.z} end)"
    ),

    # T. Engine torque, and whatever else the device carries.
    "powertrain_devices": (
        "return try(function() local names = {} "
        "for name, _ in pairs(powertrain.getDevices()) do names[#names+1] = name end "
        "return names end)"
    ),
    "main_engine": (
        "return try(function() local e = powertrain.getDevice('mainEngine') "
        "local out = {} "
        "for k, v in pairs(e) do local t = type(v) "
        "if t == 'number' or t == 'boolean' or t == 'string' then out[k] = v end end "
        "return out end)"
    ),

    # omega. Per-wheel angular velocity.
    "wheel_count": "return try(function() return wheels.wheelCount end)",
    "wheel_0": (
        "return try(function() local w = wheels.wheels[0] "
        "local out = {} "
        "for k, v in pairs(w) do local t = type(v) "
        "if t == 'number' or t == 'boolean' or t == 'string' then out[k] = v end end "
        "return out end)"
    ),

    # I_bat. Almost certainly absent -- confirm rather than assume.
    "electrics_battery_keys": (
        "return try(function() local hits = {} "
        "for k, v in pairs(electrics.values) do "
        "local low = string.lower(k) "
        "if string.find(low, 'volt') or string.find(low, 'batt') "
        "or string.find(low, 'amp') or string.find(low, 'current') "
        "or string.find(low, 'alternator') or string.find(low, 'charge') "
        "then hits[k] = v end end return hits end)"
    ),

    # What modules exist at all, so a failure above can be told apart from a
    # module that is simply named something else in this build.
    "globals": (
        "return try(function() local names = {} "
        "for k, _ in pairs(_G) do names[#names+1] = k end "
        "table.sort(names) return names end)"
    ),
}

#: Game-engine side, not vehicle side. Ambient temperature is the one axis the
#: tool listing does not cover, and it is the dominant one for corrosion.
GE_PROBES: dict[str, str] = {
    "core_environment": (
        "return try(function() local out = {} "
        "for k, v in pairs(core_environment) do out[k] = type(v) end "
        "return out end)"
    ),
    "environment_state": (
        "return try(function() return core_environment.getState() end)"
    ),
    "temperature_curve": (
        "return try(function() return core_environment.getTemperatureK() end)"
    ),
}


def run(client: MCPClient, tool: str, body: str) -> dict:
    """Run one probe and parse whatever comes back."""
    code = SERIALISER + "\n" + body
    try:
        raw = client.call(tool, {"code": code})
    except MCPError as error:
        return {"ok": False, "error": f"MCP: {error}"}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # run_lua_vehicle returns tostring() of the result, so a VM-level
        # failure arrives as prose rather than JSON.
        return {"ok": False, "error": f"unparseable: {text[:300]}"}


def summarise(name: str, result: dict) -> str:
    if not result.get("ok"):
        return f"  [ no ] {name:24} {str(result.get('error'))[:80]}"
    value = result.get("value")
    if isinstance(value, dict):
        return f"  [ yes] {name:24} {len(value)} keys"
    if isinstance(value, list):
        return f"  [ yes] {name:24} {len(value)} entries"
    return f"  [ yes] {name:24} {value}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--out", default="lua_probe.json")
    args = parser.parse_args(argv)

    print("=" * 70)
    print(f"BeamNG vehicle Lua  {args.endpoint}")
    print("=" * 70)

    findings: dict[str, dict] = {}
    try:
        with MCPClient(args.endpoint) as client:
            info = client.server_info or {}
            print(f"  connected: {info.get('serverInfo', {}).get('name', '?')}\n")

            print("VEHICLE VM  (run_lua_vehicle)")
            for name, body in PROBES.items():
                result = run(client, "run_lua_vehicle", body)
                findings[name] = result
                print(summarise(name, result))

            print("\nGAME ENGINE VM  (run_lua)")
            for name, body in GE_PROBES.items():
                result = run(client, "run_lua", body)
                findings[f"ge_{name}"] = result
                print(summarise(f"ge_{name}", result))
    except MCPError as error:
        print(f"\n  {error}", file=sys.stderr)
        print("  In BeamNG: Options > Advanced > 'Enable MCP server'.",
              file=sys.stderr)
        return 1

    out = Path(args.out)
    out.write_text(json.dumps(findings, indent=2, sort_keys=True))
    print(f"\n  Written to {out.resolve()}")
    print("  Send me that file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
