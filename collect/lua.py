"""The sampler that runs inside the vehicle's own physics VM.

Polling over HTTP caps out at a few hertz, and `run_lua_vehicle` is
asynchronous on top of that, so the shape that fits is not polling at all: Lua
accumulates samples at physics rate into a buffer, and Python drains the buffer
once a second. That is more data at fewer round trips.

Every capture expression comes from `collect.schema`. Nothing is written twice.

The capture runs under `pcall`, and `install_source` returns the first sample it
manages to take. A field name that does not exist in this build therefore fails
at install, with the error, rather than fifteen minutes later with an empty CSV.
"""

from __future__ import annotations

from collect.schema import MEASURED

#: Where the sampler keeps its state. Namespaced: it shares `_G` with the whole
#: vehicle VM.
STATE_GLOBAL = "__car_agent_sampler"

#: Samples held before the buffer starts dropping. At 100 Hz this is a minute
#: of slack, which a 1 Hz drain never needs -- the margin is for a stalled
#: drain, not for normal operation.
BUFFER_LIMIT = 6000


def install_source(interval_s: float) -> str:
    """Lua that installs the sampler, self-tests it, and starts it."""
    captures = ",\n    ".join(c.lua for c in MEASURED)
    return f"""
local S = {{}}
_G.{STATE_GLOBAL} = S
S.t = 0
S.acc = 0
S.seq = 0
S.n = 0
S.dropped = 0
S.err = nil
S.interval = {interval_s!r}
S.buf = {{}}

-- Wheel order is not guaranteed, so index by the name the vehicle gives.
S.wi = {{}}
for i = 0, wheels.wheelCount - 1 do
  S.wi[wheels.wheels[i].name] = i
end

-- 783 nodes on the ETK. Far too expensive per sample, and it changes only when
-- parts do.
local total = 0
for i = 0, obj:getNodeCount() - 1 do
  total = total + obj:getNodeMass(i)
end
S.mass = total

local function capture()
  local p = obj:getPosition()
  local vel = obj:getVelocity()
  local d = obj:getDirectionVector()
  local e = electrics.values
  local eng = powertrain.getDevice('mainEngine')
  local w = wheels.wheels
  return {{
    {captures}
  }}
end
S.capture = capture

function onPhysicsStep(dt)
  local S = _G.{STATE_GLOBAL}
  if not S then return end
  S.t = S.t + dt
  S.acc = S.acc + dt
  if S.acc < S.interval then return end
  S.acc = 0
  if S.n >= {BUFFER_LIMIT} then
    S.dropped = S.dropped + 1
    return
  end
  S.seq = S.seq + 1
  local ok, record = pcall(S.capture)
  if not ok then
    -- Record the first failure and keep the sequence number moving, so the gap
    -- is visible in the log rather than silently closed up.
    S.err = S.err or tostring(record)
    S.dropped = S.dropped + 1
    return
  end
  S.n = S.n + 1
  S.buf[S.n] = record
end

enablePhysicsStepHook(true)

-- Take one sample now, so a bad field name is an install error rather than an
-- empty CSV in fifteen minutes.
local ok, first = pcall(capture)
if not ok then
  S.ok = false
  S.error = tostring(first)
  return jsonEncode({{ok = false, error = S.error}})
end
S.ok = true
S.seq = 1
S.n = 1
S.buf[1] = first
return jsonEncode({{ok = true, mass = S.mass, wheels = S.wi, width = #first}})
"""


def verdict_source() -> str:
    """Lua that reports the install self-test, from the sampler's own state.

    Polling with anything that reports its *own* success answers the question
    with the wrong call -- the mass comes back missing and a failed install
    looks fine. The verdict has to come from where install left it.
    """
    return f"""
local S = _G.{STATE_GLOBAL}
if not S then return jsonEncode({{ok = false, error = 'the install never ran'}}) end
if S.ok == nil then return jsonEncode({{pending = true}}) end
return jsonEncode({{ok = S.ok, error = S.error, mass = S.mass,
                    wheels = S.wi, capture_error = S.err}})
"""


def drain_source() -> str:
    """Lua that hands over everything buffered, and empties the buffer."""
    return f"""
local S = _G.{STATE_GLOBAL}
if not S then return jsonEncode({{error = 'not_installed'}}) end
local taken = S.buf
local dropped = S.dropped
local err = S.err
S.buf = {{}}
S.n = 0
S.dropped = 0
return jsonEncode({{samples = taken, dropped = dropped, err = err,
                    mass = S.mass, t = S.t}})
"""


def uninstall_source() -> str:
    """Lua that stops the sampler. Safe when it was never installed."""
    return f"""
enablePhysicsStepHook(false)
onPhysicsStep = nil
_G.{STATE_GLOBAL} = nil
return 'uninstalled'
"""
