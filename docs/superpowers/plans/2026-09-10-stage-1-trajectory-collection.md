# Stage 1: Trajectory Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 5 Hz HTTP polling loop with a Lua-side 100 Hz sampler drained over MCP, producing a trajectory log whose every column is machine-readably labelled measured or derived.

**Architecture:** A single schema module names every channel once, carrying its unit, its provenance, and (for measured channels) the Lua expression that captures it. The Lua sampler source and the Python decoder are both *generated from that schema*, so a channel cannot exist in one and not the other. A `TrajectorySource` interface has two implementations: the real MCP one and a fake that runs on macOS with no game.

**Tech Stack:** Python 3.11+ stdlib only (no new dependencies), pytest, BeamNG 0.39's first-party MCP server, BeamNG vehicle-VM Lua.

**Spec:** `docs/superpowers/specs/2026-09-09-battery-rul-pipeline-design.md`

## Global Constraints

- **Stdlib only.** The project's sole third-party dependency is `openai`, and this plan removes the code that used it. Add no dependencies.
- **Everything in this plan must be testable on macOS with no BeamNG and no Windows machine.** The real MCP path is tested against a fake client double. This is a hard constraint from the spec: "anything untestable without that laptop is a design smell".
- **No network in tests.** `pytest.ini` sets `filterwarnings = error`; keep tests warning-clean.
- **Every column carries a `measured` or `derived` flag in the sidecar.** Downstream code reads it. No column may be added without one.
- **BeamNG simulates no 12 V system.** No task in this plan may emit a `current_a` or `voltage_v` column. Those belong to stage 3.
- **`run_lua_vehicle` is asynchronous** — it answers `"queued in vehicle VM(s)"` and delivers results on a later call, keyed by vehicle id. Every interaction with it must tolerate that.
- Run the suite with `python3 -m pytest -q` from the repository root.

---

### Task 1: The channel schema

The single source of truth. Lua capture code and the Python decoder are both generated from this, so they cannot drift apart.

**Files:**
- Create: `collect/__init__.py`
- Create: `collect/schema.py`
- Test: `tests/test_collect_schema.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `Channel(name: str, unit: str, provenance: str, lua: str | None, note: str = "")` — frozen dataclass
  - `MEASURED: tuple[Channel, ...]` — every channel read from the game, in capture order
  - `DERIVED: tuple[Channel, ...]` — every channel computed by us
  - `COLUMNS: tuple[str, ...]` — CSV column order, measured then derived
  - `PROVENANCE: dict[str, str]` — column name to `"measured"` or `"derived"`
  - `manifest() -> list[dict]` — the sidecar's channel table

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_schema.py
import pytest

from collect.schema import COLUMNS, DERIVED, MEASURED, PROVENANCE, Channel, manifest


def test_every_measured_channel_carries_a_lua_expression():
    for channel in MEASURED:
        assert channel.lua, f"{channel.name} is measured but has no lua expression"
        assert channel.provenance == "measured"


def test_derived_channels_have_no_lua_expression():
    for channel in DERIVED:
        assert channel.lua is None
        assert channel.provenance == "derived"


def test_no_channel_name_appears_twice():
    names = [c.name for c in MEASURED + DERIVED]
    assert len(names) == len(set(names))


def test_columns_are_measured_then_derived():
    assert COLUMNS == tuple(c.name for c in MEASURED + DERIVED)


def test_provenance_covers_every_column():
    assert set(PROVENANCE) == set(COLUMNS)
    assert set(PROVENANCE.values()) == {"measured", "derived"}


def test_no_battery_channel_exists():
    # BeamNG simulates no 12 V system. Current and voltage are stage 3's job,
    # and a column here would look like a measurement.
    forbidden = ("current_a", "voltage_v", "soc", "battery_temp_c")
    assert not set(COLUMNS) & set(forbidden)


def test_the_channels_the_load_model_needs_are_present():
    # Straight from the spec's parameter table.
    required = (
        "t_s", "speed_mps", "throttle", "brake", "rpm", "mass_kg",
        "engine_torque_nm", "x_m", "y_m", "z_m",
        "ax_mps2", "ay_mps2", "az_mps2", "dir_z",
        "wheel_av_fl", "coolant_c", "ignition_level",
    )
    for name in required:
        assert name in COLUMNS, name


def test_derived_channels_are_the_two_the_spec_names():
    assert tuple(c.name for c in DERIVED) == ("grade_rad", "a_long_mps2")


def test_manifest_describes_every_column():
    entries = manifest()
    assert [e["name"] for e in entries] == list(COLUMNS)
    for entry in entries:
        assert entry["provenance"] in ("measured", "derived")
        assert entry["unit"]


def test_channel_rejects_an_unknown_provenance():
    with pytest.raises(ValueError):
        Channel(name="x", unit="m", provenance="guessed", lua="p.x")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_collect_schema.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/__init__.py
```

```python
# collect/schema.py
"""Every channel, named once.

The Lua sampler and the Python decoder are both generated from this table, so
a channel cannot exist on one side and not the other -- which is the failure
mode that produces a column of zeros nobody notices for a week.

`lua` is an expression evaluated inside the vehicle VM with these locals in
scope, bound once per sample by the preamble in `collect.lua`:

    S     the sampler's own state table (S.t is elapsed sim time, S.mass the
          node-mass sum, S.wi a wheel-name to index map)
    p     obj:getPosition()
    vel   obj:getVelocity()
    d     obj:getDirectionVector()
    e     electrics.values
    eng   powertrain.getDevice('mainEngine')
    w     wheels.wheels

Provenance is not documentation. The sidecar carries it, downstream code reads
it, and nothing may present a derived column as a measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

PROVENANCES = ("measured", "derived")


@dataclass(frozen=True)
class Channel:
    name: str
    unit: str
    provenance: str
    lua: str | None
    note: str = ""

    def __post_init__(self) -> None:
        if self.provenance not in PROVENANCES:
            raise ValueError(
                f"{self.name}: provenance {self.provenance!r} "
                f"is not one of {PROVENANCES}"
            )
        if self.provenance == "measured" and not self.lua:
            raise ValueError(f"{self.name}: measured channels need a lua expression")
        if self.provenance == "derived" and self.lua:
            raise ValueError(f"{self.name}: derived channels must not have one")


def _m(name: str, unit: str, lua: str, note: str = "") -> Channel:
    return Channel(name=name, unit=unit, provenance="measured", lua=lua, note=note)


def _d(name: str, unit: str, note: str = "") -> Channel:
    return Channel(name=name, unit=unit, provenance="derived", lua=None, note=note)


MEASURED: tuple[Channel, ...] = (
    _m("t_s", "s", "S.t", "elapsed physics time, accumulated from physicsDt"),
    _m("seq", "count", "S.seq", "sample counter; gaps mean the buffer overflowed"),
    # pose and motion
    _m("x_m", "m", "p.x"),
    _m("y_m", "m", "p.y"),
    _m("z_m", "m", "p.z"),
    _m("vx_mps", "m/s", "vel.x"),
    _m("vy_mps", "m/s", "vel.y"),
    _m("vz_mps", "m/s", "vel.z"),
    _m("speed_mps", "m/s", "e.wheelspeed"),
    # The IMU. Includes gravity, so a_long is derived, not this.
    _m("ax_mps2", "m/s2", "sensors.gx", "vehicle frame, gravity included"),
    _m("ay_mps2", "m/s2", "sensors.gy", "vehicle frame, gravity included"),
    _m("az_mps2", "m/s2", "sensors.gz", "vehicle frame, gravity included"),
    # Unit forward vector. dir_z is sin(pitch): the road grade, measured.
    _m("dir_x", "1", "d.x"),
    _m("dir_y", "1", "d.y"),
    _m("dir_z", "1", "d.z", "sin(pitch); grade_rad is its arcsine"),
    _m("mass_kg", "kg", "S.mass", "node-mass sum, refreshed on install"),
    # powertrain
    _m("rpm", "rpm", "e.rpm"),
    _m("engine_torque_nm", "N.m", "eng.outputTorque1"),
    _m("engine_av_rads", "rad/s", "eng.outputAV1"),
    _m("engine_load", "1", "e.engineLoad"),
    _m("exhaust_flow", "1", "e.exhaustFlow"),
    _m("gear_index", "1", "e.gearIndex"),
    _m("clutch_ratio", "1", "e.clutchRatio"),
    # driver inputs, as the vehicle actually applied them
    _m("throttle", "1", "e.throttle"),
    _m("brake", "1", "e.brake"),
    _m("steering", "1", "e.steering"),
    # wheels
    _m("wheel_av_fl", "rad/s", "w[S.wi.FL].angularVelocity"),
    _m("wheel_av_fr", "rad/s", "w[S.wi.FR].angularVelocity"),
    _m("wheel_av_rl", "rad/s", "w[S.wi.RL].angularVelocity"),
    _m("wheel_av_rr", "rad/s", "w[S.wi.RR].angularVelocity"),
    _m("brake_temp_fl", "C", "w[S.wi.FL].brakeSurfaceTemperature"),
    _m("brake_temp_fr", "C", "w[S.wi.FR].brakeSurfaceTemperature"),
    _m("brake_temp_rl", "C", "w[S.wi.RL].brakeSurfaceTemperature"),
    _m("brake_temp_rr", "C", "w[S.wi.RR].brakeSurfaceTemperature"),
    _m("downforce_fl", "N", "w[S.wi.FL].downForce"),
    # thermal and trip
    _m("coolant_c", "C", "e.watertemp", "measured; never integrated over"),
    _m("oil_c", "C", "e.oiltemp"),
    _m("fuel_volume_l", "L", "e.fuelVolume"),
    _m("odometer_m", "m", "e.odometer"),
    _m("trip_m", "m", "e.trip"),
    _m("altitude_m", "m", "e.altitude"),
    _m("ignition_level", "1", "e.ignitionLevel", "crank detection"),
    _m("engine_running", "1", "e.engineRunning"),
    _m("damage", "1", "S.damage"),
)

DERIVED: tuple[Channel, ...] = (
    _d("grade_rad", "rad", "asin(dir_z)"),
    _d("a_long_mps2", "m/s2", "ax with the gravity component removed"),
)

COLUMNS: tuple[str, ...] = tuple(c.name for c in MEASURED + DERIVED)

PROVENANCE: dict[str, str] = {c.name: c.provenance for c in MEASURED + DERIVED}


def manifest() -> list[dict]:
    """The channel table the sidecar carries, in column order."""
    return [
        {
            "name": c.name,
            "unit": c.unit,
            "provenance": c.provenance,
            "source": c.lua or "",
            "note": c.note,
        }
        for c in MEASURED + DERIVED
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_collect_schema.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add collect/__init__.py collect/schema.py tests/test_collect_schema.py
git commit -m "Name every channel once, with where it came from"
```

---

### Task 2: Generate the sampler's Lua from the schema

**Files:**
- Create: `collect/lua.py`
- Test: `tests/test_collect_lua.py`

**Interfaces:**
- Consumes: `collect.schema.MEASURED`
- Produces:
  - `install_source(interval_s: float) -> str`
  - `drain_source() -> str`
  - `uninstall_source() -> str`
  - `STATE_GLOBAL: str` — the `_G` key the sampler lives under

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_lua.py
from collect.lua import STATE_GLOBAL, drain_source, install_source, uninstall_source
from collect.schema import MEASURED


def test_install_captures_every_measured_channel_in_order():
    source = install_source(0.01)
    # Each capture expression must appear, and in schema order, so the decoder
    # can read the record positionally.
    positions = [source.index(c.lua) for c in MEASURED]
    assert positions == sorted(positions)


def test_install_sets_the_requested_interval():
    assert "0.01" in install_source(0.01)
    assert "0.05" in install_source(0.05)


def test_install_binds_every_local_the_expressions_use():
    source = install_source(0.01)
    for local in ("local p =", "local vel =", "local d =", "local e =",
                  "local eng =", "local w ="):
        assert local in source


def test_install_builds_the_wheel_index_map():
    source = install_source(0.01)
    assert "wheelCount" in source
    assert ".wi" in source


def test_install_computes_mass_once_not_per_sample():
    source = install_source(0.01)
    # The node loop is expensive; it must sit outside onPhysicsStep.
    before_hook = source.split("onPhysicsStep")[0]
    assert "getNodeMass" in before_hook


def test_install_enables_the_physics_hook():
    assert "enablePhysicsStepHook" in install_source(0.01)


def test_install_caps_the_buffer_and_counts_drops():
    source = install_source(0.01)
    assert "dropped" in source


def test_drain_empties_the_buffer():
    source = drain_source()
    assert STATE_GLOBAL in source
    assert "jsonEncode" in source


def test_drain_reports_when_the_sampler_is_not_installed():
    assert "not_installed" in drain_source()


def test_uninstall_disables_the_hook_and_clears_state():
    source = uninstall_source()
    assert "enablePhysicsStepHook(false)" in source
    assert f"_G.{STATE_GLOBAL} = nil" in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_collect_lua.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect.lua'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/lua.py
"""The sampler that runs inside the vehicle's own physics VM.

Polling over HTTP caps out at a few hertz and `run_lua_vehicle` is
asynchronous on top of that, so the shape that fits is not polling at all:
Lua accumulates samples at physics rate into a ring buffer, and Python drains
the buffer once a second. That is more data at fewer round trips.

Every capture expression comes from `collect.schema`. Nothing is written twice.
"""

from __future__ import annotations

from collect.schema import MEASURED

#: Where the sampler keeps its state. Namespaced because it shares `_G` with
#: the whole vehicle VM.
STATE_GLOBAL = "__car_agent_sampler"

#: Samples held before the buffer starts dropping. At 100 Hz this is 60
#: seconds of slack, which is far more than a 1 Hz drain needs -- the margin is
#: for a stalled drain, not for normal operation.
BUFFER_LIMIT = 6000


def install_source(interval_s: float) -> str:
    """Lua that installs the sampler and starts it accumulating."""
    captures = ",\n      ".join(c.lua for c in MEASURED)
    return f"""
local S = {{}}
_G.{STATE_GLOBAL} = S
S.t = 0
S.acc = 0
S.seq = 0
S.n = 0
S.dropped = 0
S.damage = 0
S.interval = {interval_s!r}
S.buf = {{}}

-- Wheel order is not guaranteed, so index by the name the vehicle gives.
S.wi = {{}}
for i = 0, wheels.wheelCount - 1 do
  S.wi[wheels.wheels[i].name] = i
end

-- 783 nodes on the ETK. Far too expensive to do per sample, and it changes
-- only when parts do.
local total = 0
for i = 0, obj:getNodeCount() - 1 do
  total = total + obj:getNodeMass(i)
end
S.mass = total

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
  local p = obj:getPosition()
  local vel = obj:getVelocity()
  local d = obj:getDirectionVector()
  local e = electrics.values
  local eng = powertrain.getDevice('mainEngine')
  local w = wheels.wheels
  S.n = S.n + 1
  S.buf[S.n] = {{
      {captures}
  }}
end

enablePhysicsStepHook(true)
return 'installed'
"""


def drain_source() -> str:
    """Lua that hands over everything buffered and empties the buffer."""
    return f"""
local S = _G.{STATE_GLOBAL}
if not S then return jsonEncode({{error = 'not_installed'}}) end
local taken = S.buf
local dropped = S.dropped
S.buf = {{}}
S.n = 0
S.dropped = 0
return jsonEncode({{samples = taken, dropped = dropped,
                   mass = S.mass, t = S.t}})
"""


def uninstall_source() -> str:
    """Lua that stops the sampler. Safe to run when it was never installed."""
    return f"""
enablePhysicsStepHook(false)
onPhysicsStep = nil
_G.{STATE_GLOBAL} = nil
return 'uninstalled'
"""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_collect_lua.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add collect/lua.py tests/test_collect_lua.py
git commit -m "Sample inside the physics VM instead of polling it"
```

---

### Task 3: Decode a drain payload

**Files:**
- Create: `collect/decode.py`
- Test: `tests/test_collect_decode.py`

**Interfaces:**
- Consumes: `collect.schema.MEASURED`
- Produces:
  - `Drain(samples: list[dict], dropped: int, mass_kg: float, sim_time_s: float)` — frozen dataclass
  - `decode(payload: object) -> Drain`
  - `DecodeError(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_decode.py
import json

import pytest

from collect.decode import DecodeError, decode
from collect.schema import MEASURED


def a_record(seq=1, t=0.01):
    """One sample in capture order, with recognisable values."""
    values = []
    for index, channel in enumerate(MEASURED):
        if channel.name == "seq":
            values.append(seq)
        elif channel.name == "t_s":
            values.append(t)
        else:
            values.append(float(index))
    return values


def test_decodes_a_json_string_payload():
    payload = json.dumps({"samples": [a_record()], "dropped": 0,
                          "mass": 1510.62, "t": 0.01})
    drain = decode(payload)
    assert len(drain.samples) == 1
    assert drain.mass_kg == 1510.62


def test_decodes_an_already_parsed_payload():
    drain = decode({"samples": [a_record()], "dropped": 0, "mass": 1.0, "t": 0.0})
    assert len(drain.samples) == 1


def test_maps_positions_to_channel_names():
    drain = decode({"samples": [a_record(seq=7, t=0.5)], "dropped": 0,
                    "mass": 1.0, "t": 0.5})
    sample = drain.samples[0]
    assert sample["seq"] == 7
    assert sample["t_s"] == 0.5
    assert set(sample) == {c.name for c in MEASURED}


def test_carries_the_drop_count_through():
    drain = decode({"samples": [], "dropped": 12, "mass": 1.0, "t": 3.0})
    assert drain.dropped == 12


def test_an_empty_drain_is_not_an_error():
    drain = decode({"samples": [], "dropped": 0, "mass": 1.0, "t": 0.0})
    assert drain.samples == []


def test_the_queued_notice_is_an_empty_drain_not_a_failure():
    # run_lua_vehicle answers this while the real result is still coming.
    drain = decode("queued in vehicle VM(s); call again in a moment for results")
    assert drain.samples == []
    assert drain.pending is True


def test_a_result_keyed_by_vehicle_id_is_unwrapped():
    inner = json.dumps({"samples": [a_record()], "dropped": 0,
                        "mass": 1.0, "t": 0.0})
    drain = decode({"73126": inner})
    assert len(drain.samples) == 1


def test_the_sampler_not_being_installed_raises():
    with pytest.raises(DecodeError, match="not installed"):
        decode(json.dumps({"error": "not_installed"}))


def test_a_record_of_the_wrong_width_raises():
    with pytest.raises(DecodeError, match="width"):
        decode({"samples": [[1.0, 2.0]], "dropped": 0, "mass": 1.0, "t": 0.0})


def test_lua_one_based_maps_are_accepted_as_records():
    # jsonEncode turns a Lua array into an object with "1".."N" keys when it
    # cannot prove the table is a sequence.
    record = a_record()
    as_map = {str(i + 1): v for i, v in enumerate(record)}
    drain = decode({"samples": [as_map], "dropped": 0, "mass": 1.0, "t": 0.0})
    assert drain.samples[0]["seq"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_collect_decode.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect.decode'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/decode.py
"""Turn what the vehicle VM hands back into named samples.

Three shapes arrive on this path and only one of them is the data:

- the "queued" notice, because `run_lua_vehicle` is asynchronous;
- a map of vehicle id to the JSON our Lua built;
- that JSON directly, when the client has already parsed it.

Records travel as positional arrays to keep the payload small -- a named map
per sample at 100 Hz is mostly repeated key strings -- so the schema's order is
load-bearing on both sides.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from collect.schema import MEASURED

NAMES = tuple(c.name for c in MEASURED)


class DecodeError(RuntimeError):
    """The payload was not something this can read."""


@dataclass(frozen=True)
class Drain:
    samples: list[dict] = field(default_factory=list)
    dropped: int = 0
    mass_kg: float = 0.0
    sim_time_s: float = 0.0
    pending: bool = False


def _as_object(payload: object) -> dict:
    if isinstance(payload, str):
        text = payload.strip()
        if "queued" in text:
            raise _Pending()
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError) as error:
            raise DecodeError(f"not JSON: {text[:200]}") from error
    if isinstance(payload, dict):
        # A vehicle-keyed reply: {"73126": "<json>"}. Ours always has 'samples'.
        if "samples" not in payload and "error" not in payload:
            for value in payload.values():
                if isinstance(value, str):
                    return _as_object(value)
            raise DecodeError(f"no payload in {list(payload)}")
        return payload
    raise DecodeError(f"cannot read a {type(payload).__name__}")


class _Pending(Exception):
    """The VM has not answered yet. Not a failure."""


def _record_values(record: object) -> list:
    if isinstance(record, list):
        return record
    if isinstance(record, dict):
        # jsonEncode emits "1".."N" when it will not commit to an array.
        try:
            return [record[str(i + 1)] for i in range(len(record))]
        except KeyError as error:
            raise DecodeError(f"record is not a sequence: {error}") from error
    raise DecodeError(f"record is a {type(record).__name__}")


def decode(payload: object) -> Drain:
    try:
        obj = _as_object(payload)
    except _Pending:
        return Drain(pending=True)

    if obj.get("error") == "not_installed":
        raise DecodeError("the sampler is not installed in this vehicle VM")

    samples = []
    for record in obj.get("samples", []):
        values = _record_values(record)
        if len(values) != len(NAMES):
            raise DecodeError(
                f"record width {len(values)}, schema expects {len(NAMES)}"
            )
        samples.append(dict(zip(NAMES, values)))

    return Drain(
        samples=samples,
        dropped=int(obj.get("dropped", 0)),
        mass_kg=float(obj.get("mass", 0.0)),
        sim_time_s=float(obj.get("t", 0.0)),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_collect_decode.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add collect/decode.py tests/test_collect_decode.py
git commit -m "Read the three shapes the vehicle VM answers with"
```

---

### Task 4: The two derived channels

**Files:**
- Create: `collect/derive.py`
- Test: `tests/test_collect_derive.py`

**Interfaces:**
- Consumes: samples as `dict` from `collect.decode`
- Produces: `add_derived(sample: dict) -> dict` — returns the sample with `grade_rad` and `a_long_mps2` added

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_derive.py
import math

from collect.derive import GRAVITY_MPS2, add_derived


def a_sample(**overrides):
    sample = {"dir_x": 1.0, "dir_y": 0.0, "dir_z": 0.0,
              "ax_mps2": 0.0, "ay_mps2": 0.0, "az_mps2": -GRAVITY_MPS2}
    sample.update(overrides)
    return sample


def test_level_ground_is_zero_grade():
    assert add_derived(a_sample())["grade_rad"] == 0.0


def test_grade_is_the_arcsine_of_the_forward_z_component():
    sample = add_derived(a_sample(dir_z=0.5))
    assert math.isclose(sample["grade_rad"], math.asin(0.5))


def test_grade_is_negative_downhill():
    assert add_derived(a_sample(dir_z=-0.1))["grade_rad"] < 0


def test_a_dir_z_outside_the_domain_is_clamped_not_crashed():
    # Numerical drift can put a unit vector marginally over 1.
    sample = add_derived(a_sample(dir_z=1.0000001))
    assert math.isclose(sample["grade_rad"], math.pi / 2)


def test_a_parked_car_on_the_level_has_zero_longitudinal_acceleration():
    # sensors.gx/gy/gz include gravity, so a stationary car reads -9.81 on z.
    assert math.isclose(add_derived(a_sample())["a_long_mps2"], 0.0, abs_tol=1e-6)


def test_a_parked_car_on_a_slope_still_has_zero_longitudinal_acceleration():
    # This is the whole reason the channel is derived. On a 30 degree slope the
    # accelerometer reads a large gx that is entirely gravity.
    grade = math.radians(30)
    sample = a_sample(
        dir_z=math.sin(grade),
        ax_mps2=GRAVITY_MPS2 * math.sin(grade),
        az_mps2=-GRAVITY_MPS2 * math.cos(grade),
    )
    assert math.isclose(add_derived(sample)["a_long_mps2"], 0.0, abs_tol=1e-6)


def test_real_acceleration_on_the_level_passes_through():
    sample = add_derived(a_sample(ax_mps2=2.0))
    assert math.isclose(sample["a_long_mps2"], 2.0)


def test_the_original_channels_are_left_alone():
    sample = add_derived(a_sample(dir_z=0.5, ax_mps2=2.0))
    assert sample["dir_z"] == 0.5
    assert sample["ax_mps2"] == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_collect_derive.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect.derive'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/derive.py
"""The two channels the game does not report and we compute.

Both exist because the raw measurement is not the quantity wanted.

`dir_z` is the z component of a unit forward vector, so its arcsine is the road
grade -- but a grade in radians is what the road-load equation takes, so it is
worth naming.

`sensors.gx` is a real accelerometer and therefore reads gravity as well as
motion. A car parked on a 30 degree slope shows 4.9 m/s^2 of longitudinal
acceleration while going nowhere. Subtracting the gravity component along the
vehicle's forward axis leaves what the vehicle is actually doing.
"""

from __future__ import annotations

import math

GRAVITY_MPS2 = 9.81


def add_derived(sample: dict) -> dict:
    dir_z = max(-1.0, min(1.0, float(sample["dir_z"])))
    sample["grade_rad"] = math.asin(dir_z)
    # The forward axis picks up g * sin(pitch), and dir_z is exactly sin(pitch).
    sample["a_long_mps2"] = float(sample["ax_mps2"]) - GRAVITY_MPS2 * dir_z
    return sample
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_collect_derive.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add collect/derive.py tests/test_collect_derive.py
git commit -m "Take gravity back out of the accelerometer"
```

---

### Task 5: ScenarioSpec replaces BehaviourSpec

**Files:**
- Create: `collect/scenario.py`
- Test: `tests/test_scenario.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `ScenarioSpec(name, aggression, minutes, accessory_load_a, ambient_temp_c, vehicle_config, route, seed)` — frozen dataclass
  - `.scenario_hash: str` — 16 hex characters, content-addressed
  - `.to_dict() -> dict`, `ScenarioSpec.from_dict(d) -> ScenarioSpec`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario.py
import pytest

from collect.scenario import ScenarioSpec


def a_spec(**overrides):
    fields = dict(name="Hot commute", aggression=0.6, minutes=15.0,
                  accessory_load_a=35.0, ambient_temp_c=25.0,
                  vehicle_config="etki/3000ix_A", route="span", seed=0)
    fields.update(overrides)
    return ScenarioSpec(**fields)


def test_a_valid_spec_constructs():
    assert a_spec().name == "Hot commute"


@pytest.mark.parametrize("field,value", [
    ("aggression", 0.0),      # below BeamNG's usable floor
    ("aggression", 2.0),      # above its ceiling
    ("minutes", 0.0),
    ("accessory_load_a", -1.0),
    ("accessory_load_a", 500.0),   # that is a crank, not an accessory
    ("ambient_temp_c", -60.0),
    ("ambient_temp_c", 90.0),
])
def test_out_of_range_values_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        a_spec(**{field: value})


def test_an_empty_name_is_refused():
    with pytest.raises(ValueError, match="name"):
        a_spec(name="  ")


def test_the_hash_is_sixteen_hex_characters():
    digest = a_spec().scenario_hash
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_the_hash_is_stable_across_instances():
    assert a_spec().scenario_hash == a_spec().scenario_hash


def test_the_hash_changes_when_a_field_changes():
    assert a_spec().scenario_hash != a_spec(aggression=0.9).scenario_hash


def test_the_hash_ignores_the_name():
    # The name is a label for humans. Two identically parameterised runs are
    # the same experiment whatever they are called.
    assert a_spec().scenario_hash == a_spec(name="Something else").scenario_hash


def test_a_spec_round_trips_through_a_dict():
    spec = a_spec()
    assert ScenarioSpec.from_dict(spec.to_dict()) == spec


def test_there_is_no_llm_anywhere_in_this_module():
    import collect.scenario as module
    source = open(module.__file__).read().lower()
    for word in ("openai", "deepseek", "prompt", "completion"):
        assert word not in source
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_scenario.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect.scenario'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/scenario.py
"""What to drive, and under what conditions.

This replaces `BehaviourSpec`. The LLM behaviour agent is gone: BeamNG's AI
accepts one scalar, `aggression`, so eight generated parameters were collapsed
to one on the way in, and driving diversity now comes from the world model's
counterfactuals rather than from the recorded drives.

Two fields are not simulated by the game and are declared here so that the log
says what was assumed. `accessory_load_a` is the electrical draw of lights,
blower and ECU, which BeamNG does not model at all. `ambient_temp_c` reads out
of the game but cannot be set in it, so it is an assumption about the world
the drive is meant to represent.

Bounds are checked here rather than trusted from a caller. The check is a code
gate for the same reason it always was: a plausible-looking number outside
physical range produces a run that looks fine and means nothing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

#: BeamNG's AI scale. Below ~0.25 the car crawls; above ~1.2 it drives like it
#: is being chased.
AGGRESSION_RANGE = (0.25, 1.2)
#: Lights, blower, ECU, fuel pump. A crank is hundreds of amps and is not this.
ACCESSORY_LOAD_RANGE_A = (0.0, 120.0)
AMBIENT_RANGE_C = (-40.0, 60.0)


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    aggression: float
    minutes: float
    accessory_load_a: float
    ambient_temp_c: float
    vehicle_config: str
    route: str
    seed: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")
        for field, (low, high) in (
            ("aggression", AGGRESSION_RANGE),
            ("accessory_load_a", ACCESSORY_LOAD_RANGE_A),
            ("ambient_temp_c", AMBIENT_RANGE_C),
        ):
            value = getattr(self, field)
            if not low <= value <= high:
                raise ValueError(f"{field}: {value} outside [{low}, {high}]")
        if self.minutes <= 0:
            raise ValueError(f"minutes: {self.minutes} must be positive")

    @property
    def scenario_hash(self) -> str:
        """Content address. The name is excluded: it labels, it does not define."""
        payload = {k: v for k, v in asdict(self).items() if k != "name"}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScenarioSpec":
        return cls(**data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_scenario.py -q`
Expected: PASS, 16 tests

- [ ] **Step 5: Commit**

```bash
git add collect/scenario.py tests/test_scenario.py
git commit -m "Describe a run without asking a language model"
```

---

### Task 6: The trajectory source interface and its fake

**Files:**
- Create: `collect/source.py`
- Test: `tests/test_collect_fake_source.py`

**Interfaces:**
- Consumes: `collect.decode.Drain`, `collect.derive.add_derived`, `collect.schema.COLUMNS`
- Produces:
  - `TrajectorySource` ABC with `start()`, `drain() -> list[dict]`, `stop()`, `dropped: int`
  - `FakeTrajectorySource(interval_s=0.01, speed_mps=20.0, grade_rad=0.0)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_fake_source.py
import math

from collect.schema import COLUMNS
from collect.source import FakeTrajectorySource


def test_a_drain_before_start_is_empty():
    source = FakeTrajectorySource()
    assert source.drain() == []


def test_every_sample_has_every_column():
    source = FakeTrajectorySource(interval_s=0.01)
    source.start()
    source.advance(0.1)
    for sample in source.drain():
        assert set(sample) == set(COLUMNS)


def test_it_produces_one_sample_per_interval():
    source = FakeTrajectorySource(interval_s=0.01)
    source.start()
    source.advance(1.0)
    assert len(source.drain()) == 100


def test_draining_empties_the_buffer():
    source = FakeTrajectorySource(interval_s=0.01)
    source.start()
    source.advance(0.1)
    assert source.drain()
    assert source.drain() == []


def test_the_sequence_number_never_repeats():
    source = FakeTrajectorySource(interval_s=0.01)
    source.start()
    source.advance(0.1)
    first = [s["seq"] for s in source.drain()]
    source.advance(0.1)
    second = [s["seq"] for s in source.drain()]
    assert first + second == list(range(1, len(first) + len(second) + 1))


def test_time_advances_monotonically():
    source = FakeTrajectorySource(interval_s=0.01)
    source.start()
    source.advance(0.5)
    times = [s["t_s"] for s in source.drain()]
    assert times == sorted(times)


def test_the_car_moves_at_the_speed_it_was_given():
    source = FakeTrajectorySource(interval_s=0.1, speed_mps=20.0)
    source.start()
    source.advance(1.0)
    samples = source.drain()
    travelled = samples[-1]["x_m"] - samples[0]["x_m"]
    assert math.isclose(travelled, 20.0 * 0.9, rel_tol=1e-6)


def test_a_grade_shows_up_as_grade_not_as_acceleration():
    # The derived channel must strip gravity even here, or the fake would let a
    # bug through that the real source would hit.
    source = FakeTrajectorySource(interval_s=0.1, grade_rad=math.radians(10))
    source.start()
    source.advance(0.5)
    sample = source.drain()[0]
    assert math.isclose(sample["grade_rad"], math.radians(10), abs_tol=1e-6)
    assert math.isclose(sample["a_long_mps2"], 0.0, abs_tol=1e-6)


def test_stop_is_safe_to_call_twice():
    source = FakeTrajectorySource()
    source.start()
    source.stop()
    source.stop()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_collect_fake_source.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect.source'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/source.py
"""Where trajectory samples come from.

One interface, two implementations. `MCPTrajectorySource` talks to the game.
`FakeTrajectorySource` does not, which is what makes every layer above this
testable on a machine with no BeamNG and no Windows laptop.

The fake is deliberately crude -- constant speed on a constant grade -- because
its job is to exercise the plumbing, not to model a car. Anything that needs
real dynamics needs the real game.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from collect.derive import GRAVITY_MPS2, add_derived
from collect.schema import MEASURED


class TrajectorySource(ABC):
    """Start sampling, drain what has accumulated, stop."""

    dropped: int = 0

    @abstractmethod
    def start(self) -> None:
        """Begin accumulating samples."""

    @abstractmethod
    def drain(self) -> list[dict]:
        """Hand over everything buffered since the last drain, and empty it."""

    @abstractmethod
    def stop(self) -> None:
        """Stop sampling. Safe to call more than once."""


class FakeTrajectorySource(TrajectorySource):
    def __init__(
        self,
        interval_s: float = 0.01,
        speed_mps: float = 20.0,
        grade_rad: float = 0.0,
    ) -> None:
        self.interval_s = interval_s
        self.speed_mps = speed_mps
        self.grade_rad = grade_rad
        self.running = False
        self.t = 0.0
        self.seq = 0
        self._buffer: list[dict] = []

    def start(self) -> None:
        self.running = True

    def advance(self, seconds: float) -> None:
        """Run the fake forward. Only the fake has this -- the game has time."""
        if not self.running:
            return
        steps = int(round(seconds / self.interval_s))
        for _ in range(steps):
            self.seq += 1
            self.t += self.interval_s
            self._buffer.append(add_derived(self._sample()))

    def _sample(self) -> dict:
        travelled = self.speed_mps * (self.t - self.interval_s)
        sample = {c.name: 0.0 for c in MEASURED}
        sample.update(
            t_s=self.t,
            seq=self.seq,
            x_m=travelled * math.cos(self.grade_rad),
            z_m=travelled * math.sin(self.grade_rad),
            speed_mps=self.speed_mps,
            vx_mps=self.speed_mps,
            dir_x=math.cos(self.grade_rad),
            dir_z=math.sin(self.grade_rad),
            # A real accelerometer reads gravity. Steady speed on a slope is
            # all gravity and no acceleration.
            ax_mps2=GRAVITY_MPS2 * math.sin(self.grade_rad),
            az_mps2=-GRAVITY_MPS2 * math.cos(self.grade_rad),
            mass_kg=1510.62,
            rpm=2500.0,
            coolant_c=90.0,
            engine_running=1.0,
            ignition_level=2.0,
        )
        return sample

    def drain(self) -> list[dict]:
        taken, self._buffer = self._buffer, []
        return taken

    def stop(self) -> None:
        self.running = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_collect_fake_source.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add collect/source.py tests/test_collect_fake_source.py
git commit -m "One way in for samples, and a way to test it without the game"
```

---

### Task 7: The MCP trajectory source

**Files:**
- Create: `collect/mcp_source.py`
- Test: `tests/test_collect_mcp_source.py`

**Interfaces:**
- Consumes: `sim.mcp_client.MCPClient`, `collect.lua`, `collect.decode.decode`, `collect.derive.add_derived`
- Produces: `MCPTrajectorySource(client, interval_s=0.01, drain_attempts=8, wait_s=0.15)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_mcp_source.py
import json

import pytest

from collect.lua import STATE_GLOBAL
from collect.mcp_source import MCPTrajectorySource
from collect.schema import MEASURED


class FakeClient:
    """Stands in for MCPClient, and remembers what it was asked."""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = list(replies or [])

    def call(self, name, arguments=None):
        self.calls.append((name, (arguments or {}).get("code", "")))
        if self.replies:
            return self.replies.pop(0)
        return "queued in vehicle VM(s); call again in a moment for results"


def a_payload(seq=1, t=0.01):
    record = []
    for index, channel in enumerate(MEASURED):
        if channel.name == "seq":
            record.append(seq)
        elif channel.name == "t_s":
            record.append(t)
        elif channel.name == "dir_z":
            record.append(0.0)
        elif channel.name == "ax_mps2":
            record.append(0.0)
        else:
            record.append(float(index))
    return json.dumps({"samples": [record], "dropped": 0, "mass": 1510.62, "t": t})


def test_start_installs_the_sampler():
    client = FakeClient()
    MCPTrajectorySource(client).start()
    tool, code = client.calls[0]
    assert tool == "run_lua_vehicle"
    assert STATE_GLOBAL in code
    assert "enablePhysicsStepHook" in code


def test_start_passes_the_interval_into_the_lua():
    client = FakeClient()
    MCPTrajectorySource(client, interval_s=0.02).start()
    assert "0.02" in client.calls[0][1]


def test_drain_returns_decoded_samples():
    client = FakeClient(replies=[None, a_payload()])
    source = MCPTrajectorySource(client, wait_s=0.0)
    source.start()
    samples = source.drain()
    assert len(samples) == 1
    assert samples[0]["seq"] == 1


def test_drain_adds_the_derived_channels():
    client = FakeClient(replies=[None, a_payload()])
    source = MCPTrajectorySource(client, wait_s=0.0)
    source.start()
    assert "grade_rad" in source.drain()[0]


def test_drain_keeps_retrying_while_the_vm_says_queued():
    client = FakeClient(replies=[None, "queued in vehicle VM(s)",
                                 "queued in vehicle VM(s)", a_payload()])
    source = MCPTrajectorySource(client, wait_s=0.0)
    source.start()
    assert len(source.drain()) == 1


def test_drain_gives_up_after_the_attempt_budget_rather_than_hanging():
    client = FakeClient()   # always queued, never answers
    source = MCPTrajectorySource(client, wait_s=0.0, drain_attempts=3)
    source.start()
    assert source.drain() == []
    # one install, then exactly the budgeted drains
    assert len(client.calls) == 1 + 3


def test_dropped_samples_are_counted_and_accumulate():
    payload = json.dumps({"samples": [], "dropped": 5, "mass": 1.0, "t": 1.0})
    client = FakeClient(replies=[None, payload, payload])
    source = MCPTrajectorySource(client, wait_s=0.0)
    source.start()
    source.drain()
    source.drain()
    assert source.dropped == 10


def test_a_missing_sampler_raises_rather_than_returning_nothing():
    client = FakeClient(replies=[None, json.dumps({"error": "not_installed"})])
    source = MCPTrajectorySource(client, wait_s=0.0)
    source.start()
    with pytest.raises(RuntimeError, match="not installed"):
        source.drain()


def test_stop_uninstalls_the_sampler():
    client = FakeClient()
    source = MCPTrajectorySource(client)
    source.start()
    source.stop()
    assert "enablePhysicsStepHook(false)" in client.calls[-1][1]


def test_stop_twice_only_uninstalls_once():
    client = FakeClient()
    source = MCPTrajectorySource(client)
    source.start()
    source.stop()
    before = len(client.calls)
    source.stop()
    assert len(client.calls) == before
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_collect_mcp_source.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect.mcp_source'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/mcp_source.py
"""The trajectory source that talks to the game.

`run_lua_vehicle` is asynchronous. It answers "queued in vehicle VM(s)" and
delivers the result on a later call, so a drain is not one round trip but a
short retry loop with a budget. The budget matters: without one, a vehicle VM
that has stopped answering hangs the run instead of ending it.
"""

from __future__ import annotations

import time

from collect.decode import DecodeError, decode
from collect.derive import add_derived
from collect.lua import drain_source, install_source, uninstall_source
from collect.source import TrajectorySource


class MCPTrajectorySource(TrajectorySource):
    def __init__(
        self,
        client,
        interval_s: float = 0.01,
        drain_attempts: int = 8,
        wait_s: float = 0.15,
    ) -> None:
        self.client = client
        self.interval_s = interval_s
        self.drain_attempts = drain_attempts
        self.wait_s = wait_s
        self.dropped = 0
        self.mass_kg = 0.0
        self.installed = False

    def start(self) -> None:
        self.client.call("run_lua_vehicle", {"code": install_source(self.interval_s)})
        self.installed = True

    def drain(self) -> list[dict]:
        code = drain_source()
        for attempt in range(self.drain_attempts):
            if attempt and self.wait_s:
                time.sleep(self.wait_s)
            raw = self.client.call("run_lua_vehicle", {"code": code})
            try:
                result = decode(raw)
            except DecodeError as error:
                raise RuntimeError(str(error)) from error
            if result.pending:
                continue
            self.dropped += result.dropped
            self.mass_kg = result.mass_kg or self.mass_kg
            return [add_derived(sample) for sample in result.samples]
        # Budget spent. An empty drain is honest: nothing arrived.
        return []

    def stop(self) -> None:
        if not self.installed:
            return
        self.installed = False
        self.client.call("run_lua_vehicle", {"code": uninstall_source()})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_collect_mcp_source.py -q`
Expected: PASS, 10 tests

- [ ] **Step 5: Commit**

```bash
git add collect/mcp_source.py tests/test_collect_mcp_source.py
git commit -m "Drain the sampler over MCP, with a budget instead of a hang"
```

---

### Task 8: Write the trajectory and its provenance

**Files:**
- Create: `collect/writer.py`
- Test: `tests/test_collect_writer.py`

**Interfaces:**
- Consumes: `collect.schema.COLUMNS`, `collect.schema.manifest`, `collect.scenario.ScenarioSpec`
- Produces: `TrajectoryLog(csv_path, spec, interval_s, beamng: dict | None = None)` with `write(samples: list[dict])`, `summary() -> dict`, `close()`, and context-manager support

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_writer.py
import csv
import json

from collect.scenario import ScenarioSpec
from collect.schema import COLUMNS
from collect.writer import TrajectoryLog


def a_spec():
    return ScenarioSpec(name="Test", aggression=0.6, minutes=1.0,
                        accessory_load_a=35.0, ambient_temp_c=25.0,
                        vehicle_config="etki/3000ix_A", route="span", seed=0)


def some_samples(count=3):
    return [
        {**{c: 0.0 for c in COLUMNS}, "t_s": i * 0.01, "seq": i + 1,
         "speed_mps": 20.0, "coolant_c": 90.0, "engine_running": 1.0}
        for i in range(count)
    ]


def test_the_header_is_the_schema_column_order(tmp_path):
    path = tmp_path / "run.csv"
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        log.write(some_samples())
    assert next(csv.reader(path.open())) == list(COLUMNS)


def test_every_sample_becomes_a_row(tmp_path):
    path = tmp_path / "run.csv"
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        log.write(some_samples(5))
    assert len(list(csv.DictReader(path.open()))) == 5


def test_rows_are_on_disk_before_close(tmp_path):
    # A run is fifteen minutes on someone else's laptop. A crash at minute
    # fourteen must cost the last row, not the run.
    path = tmp_path / "run.csv"
    log = TrajectoryLog(path, a_spec(), 0.01)
    log.write(some_samples(2))
    assert len(list(csv.DictReader(path.open()))) == 2


def test_the_sidecar_carries_the_channel_manifest(tmp_path):
    path = tmp_path / "run.csv"
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        log.write(some_samples())
    sidecar = json.loads((tmp_path / "run.json").read_text())
    names = [c["name"] for c in sidecar["channels"]]
    assert names == list(COLUMNS)


def test_the_sidecar_says_which_columns_are_derived(tmp_path):
    path = tmp_path / "run.csv"
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        log.write(some_samples())
    sidecar = json.loads((tmp_path / "run.json").read_text())
    provenance = {c["name"]: c["provenance"] for c in sidecar["channels"]}
    assert provenance["speed_mps"] == "measured"
    assert provenance["grade_rad"] == "derived"


def test_the_sidecar_carries_the_scenario_and_its_hash(tmp_path):
    path = tmp_path / "run.csv"
    spec = a_spec()
    with TrajectoryLog(path, spec, 0.01) as log:
        log.write(some_samples())
    sidecar = json.loads((tmp_path / "run.json").read_text())
    assert sidecar["scenario_hash"] == spec.scenario_hash
    assert sidecar["scenario"]["accessory_load_a"] == 35.0


def test_the_sidecar_records_what_the_game_was(tmp_path):
    path = tmp_path / "run.csv"
    beamng = {"level": "west_coast_usa", "vehicle": "etki", "build": "0.39"}
    with TrajectoryLog(path, a_spec(), 0.01, beamng=beamng) as log:
        log.write(some_samples())
    sidecar = json.loads((tmp_path / "run.json").read_text())
    assert sidecar["beamng"]["level"] == "west_coast_usa"


def test_the_summary_counts_rows_and_duration(tmp_path):
    path = tmp_path / "run.csv"
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        log.write(some_samples(100))
        summary = log.summary()
    assert summary["rows"] == 100
    assert summary["duration_s"] == 1.0


def test_the_sidecar_is_written_even_when_the_run_raises(tmp_path):
    path = tmp_path / "run.csv"
    try:
        with TrajectoryLog(path, a_spec(), 0.01) as log:
            log.write(some_samples())
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert (tmp_path / "run.json").exists()


def test_a_sample_missing_a_column_is_refused_not_silently_blanked(tmp_path):
    path = tmp_path / "run.csv"
    broken = some_samples(1)
    del broken[0]["speed_mps"]
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        try:
            log.write(broken)
        except KeyError as error:
            assert "speed_mps" in str(error)
        else:
            raise AssertionError("a missing channel must not pass silently")


def test_no_battery_column_is_written(tmp_path):
    path = tmp_path / "run.csv"
    with TrajectoryLog(path, a_spec(), 0.01) as log:
        log.write(some_samples())
    header = next(csv.reader(path.open()))
    assert "current_a" not in header
    assert "voltage_v" not in header
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_collect_writer.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect.writer'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/writer.py
"""A trajectory CSV and the sidecar that makes it interpretable.

Two requirements shape this, and both were learned the hard way.

**A log must be interpretable on its own.** A temperature trace with no ambient
recorded beside it is close to useless, and a column of numbers that does not
say whether it was measured or computed will eventually be presented as a
measurement. The sidecar carries the whole channel manifest, provenance
included, and downstream code reads it.

**A run must survive a crash.** Runs are fifteen minutes of real time on
someone else's laptop. Rows are flushed as they are written and the sidecar is
written even when the run raises, so a failure at minute fourteen costs the
last row rather than the run.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from collect.scenario import ScenarioSpec
from collect.schema import COLUMNS, manifest


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class TrajectoryLog:
    def __init__(
        self,
        csv_path: Path | str,
        spec: ScenarioSpec,
        interval_s: float,
        beamng: dict | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.sidecar_path = self.csv_path.with_suffix(".json")
        self.spec = spec
        self.interval_s = interval_s
        self.beamng = beamng or {}
        self.rows = 0
        self.dropped = 0

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.csv_path.open("w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(COLUMNS)
        self._handle.flush()

    def __enter__(self) -> "TrajectoryLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def write(self, samples: list[dict]) -> None:
        for sample in samples:
            # A missing channel is a schema bug. Writing a blank for it would
            # produce a column that silently means nothing.
            self._writer.writerow([sample[name] for name in COLUMNS])
            self.rows += 1
        self._handle.flush()

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "duration_s": self.rows * self.interval_s,
            "dropped": self.dropped,
            "sample_hz": 1.0 / self.interval_s if self.interval_s else 0.0,
        }

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.close()
        # This class exists so a run survives a crash. It must not be the thing
        # that crashes: the CSV is already on disk, so a sidecar that cannot be
        # written is reported and swallowed.
        try:
            self.sidecar_path.write_text(
                json.dumps(
                    {
                        "csv": self.csv_path.name,
                        "scenario_hash": self.spec.scenario_hash,
                        "scenario": self.spec.to_dict(),
                        "channels": manifest(),
                        "beamng": self.beamng,
                        "git_commit": _git_commit(),
                        "summary": self.summary(),
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        except OSError as error:
            print(f"warning: could not write {self.sidecar_path}: {error}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_collect_writer.py -q`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add collect/writer.py tests/test_collect_writer.py
git commit -m "Log a trajectory that says where each column came from"
```

---

### Task 9: The collection run loop and its entry point

**Files:**
- Create: `collect/run.py`
- Create: `collect_trajectories.py`
- Test: `tests/test_collect_run.py`

**Interfaces:**
- Consumes: everything above
- Produces:
  - `collect_run(source, spec, csv_path, seconds, clock=time.monotonic, sleep=time.sleep, beamng=None) -> dict`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_collect_run.py
import csv
import json

from collect.run import collect_run
from collect.scenario import ScenarioSpec
from collect.source import FakeTrajectorySource


def a_spec(minutes=0.05):
    return ScenarioSpec(name="Test", aggression=0.6, minutes=minutes,
                        accessory_load_a=35.0, ambient_temp_c=25.0,
                        vehicle_config="etki/3000ix_A", route="span", seed=0)


class Clock:
    """A clock the test drives, so no test ever waits on real time."""

    def __init__(self, source, step=1.0):
        self.now = 0.0
        self.source = source
        self.step = step

    def time(self):
        return self.now

    def sleep(self, _seconds):
        self.now += self.step
        self.source.advance(self.step)


def test_it_writes_a_csv_with_rows(tmp_path):
    source = FakeTrajectorySource(interval_s=0.01)
    clock = Clock(source)
    path = tmp_path / "run.csv"
    collect_run(source, a_spec(), path, seconds=3.0,
                clock=clock.time, sleep=clock.sleep)
    assert len(list(csv.DictReader(path.open()))) == 300


def test_it_writes_the_sidecar(tmp_path):
    source = FakeTrajectorySource(interval_s=0.01)
    clock = Clock(source)
    path = tmp_path / "run.csv"
    collect_run(source, a_spec(), path, seconds=2.0,
                clock=clock.time, sleep=clock.sleep)
    assert json.loads((tmp_path / "run.json").read_text())["scenario_hash"]


def test_it_stops_the_source_when_the_time_is_up(tmp_path):
    source = FakeTrajectorySource(interval_s=0.01)
    clock = Clock(source)
    collect_run(source, a_spec(), tmp_path / "run.csv", seconds=2.0,
                clock=clock.time, sleep=clock.sleep)
    assert source.running is False


def test_it_stops_the_source_even_when_the_run_raises(tmp_path):
    class Exploding(FakeTrajectorySource):
        def drain(self):
            raise RuntimeError("the VM went away")

    source = Exploding(interval_s=0.01)
    clock = Clock(source)
    try:
        collect_run(source, a_spec(), tmp_path / "run.csv", seconds=2.0,
                    clock=clock.time, sleep=clock.sleep)
    except RuntimeError:
        pass
    assert source.running is False


def test_a_final_drain_happens_after_the_clock_runs_out(tmp_path):
    # Whatever is in the buffer when time expires is still data.
    source = FakeTrajectorySource(interval_s=0.01)
    clock = Clock(source)
    path = tmp_path / "run.csv"
    result = collect_run(source, a_spec(), path, seconds=1.0,
                         clock=clock.time, sleep=clock.sleep)
    assert result["rows"] == 100


def test_the_summary_is_returned(tmp_path):
    source = FakeTrajectorySource(interval_s=0.01)
    clock = Clock(source)
    result = collect_run(source, a_spec(), tmp_path / "run.csv", seconds=2.0,
                         clock=clock.time, sleep=clock.sleep)
    assert result["rows"] == 200
    assert result["duration_s"] == 2.0


def test_a_ctrl_c_keeps_what_was_collected(tmp_path):
    class Interrupted(FakeTrajectorySource):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.drains = 0

        def drain(self):
            self.drains += 1
            if self.drains > 2:
                raise KeyboardInterrupt
            return super().drain()

    source = Interrupted(interval_s=0.01)
    clock = Clock(source)
    path = tmp_path / "run.csv"
    collect_run(source, a_spec(), path, seconds=60.0,
                clock=clock.time, sleep=clock.sleep)
    assert len(list(csv.DictReader(path.open()))) > 0
    assert (tmp_path / "run.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_collect_run.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'collect.run'`

- [ ] **Step 3: Write minimal implementation**

```python
# collect/run.py
"""Drive the clock, drain the source, write the rows.

The clock and the sleep are injected so the tests never wait on real time. A
fifteen-minute run is not something a test suite can afford to reproduce, and a
loop that can only be tested by waiting is a loop that stops being tested.
"""

from __future__ import annotations

import time
from pathlib import Path

from collect.scenario import ScenarioSpec
from collect.writer import TrajectoryLog

#: How often to drain. The Lua buffer holds a minute of slack, so this is about
#: bounding loss on a crash rather than about keeping up.
DRAIN_INTERVAL_S = 1.0


def collect_run(
    source,
    spec: ScenarioSpec,
    csv_path: Path | str,
    seconds: float,
    clock=time.monotonic,
    sleep=time.sleep,
    beamng: dict | None = None,
) -> dict:
    """Collect for `seconds`, and return the run summary."""
    started = clock()
    with TrajectoryLog(csv_path, spec, source.interval_s, beamng=beamng) as log:
        source.start()
        try:
            while clock() - started < seconds:
                sleep(DRAIN_INTERVAL_S)
                log.write(source.drain())
            # Whatever is still buffered when time expires is still data.
            log.write(source.drain())
        except KeyboardInterrupt:
            # Deliberate: everything already flushed stays, and the sidecar is
            # written on the way out. Stopping early costs the last drain.
            print("\n  stopped early. What was collected is on disk.")
        finally:
            source.stop()
            log.dropped = getattr(source, "dropped", 0)
        return log.summary()
```

```python
#!/usr/bin/env python3
# collect_trajectories.py
"""Record a driving trajectory from BeamNG.

    py collect_trajectories.py "Hot commute" --minutes 15
    py collect_trajectories.py "Hot commute" --fake     no game needed

BeamNG's own AI drives -- it follows roads and avoids traffic, which nothing
here can. This project chooses the route and the style, and keeps every
measurement.

Samples are taken inside the vehicle's physics VM at 100 Hz and drained once a
second, so the HTTP transport is not the sample rate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from collect.run import collect_run  # noqa: E402
from collect.scenario import ScenarioSpec  # noqa: E402
from collect.source import FakeTrajectorySource  # noqa: E402

RUNS = Path(__file__).parent / "runs"
SAMPLE_INTERVAL_S = 0.01


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name")
    parser.add_argument("--minutes", type=float, default=15.0)
    parser.add_argument("--aggression", type=float, default=0.6)
    parser.add_argument("--accessory-load", type=float, default=35.0,
                        help="amps. BeamNG does not model accessories, so this "
                             "is an assumption the sidecar records")
    parser.add_argument("--ambient", type=float, default=25.0,
                        help="celsius. Readable from the game, not settable in "
                             "it, so this is an assumption too")
    parser.add_argument("--vehicle", default="etki/3000ix_A")
    parser.add_argument("--route", default="span")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fake", action="store_true",
                        help="run against the fake source, with no game")
    parser.add_argument("--endpoint", default=None)
    args = parser.parse_args(argv)

    try:
        spec = ScenarioSpec(
            name=args.name, aggression=args.aggression, minutes=args.minutes,
            accessory_load_a=args.accessory_load, ambient_temp_c=args.ambient,
            vehicle_config=args.vehicle, route=args.route, seed=args.seed,
        )
    except ValueError as error:
        print(f"  {error}", file=sys.stderr)
        return 1

    csv_path = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}_{spec.scenario_hash}.csv"

    if args.fake:
        source = FakeTrajectorySource(interval_s=SAMPLE_INTERVAL_S)
        beamng = {"backend": "fake"}
    else:
        from collect.mcp_source import MCPTrajectorySource
        from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient, MCPError

        client = MCPClient(args.endpoint or DEFAULT_ENDPOINT)
        try:
            client.connect()
        except MCPError as error:
            print(f"\n  {error}", file=sys.stderr)
            print("  In BeamNG: Options > Advanced > 'Enable MCP server'.",
                  file=sys.stderr)
            return 1
        source = MCPTrajectorySource(client, interval_s=SAMPLE_INTERVAL_S)
        beamng = {"backend": "mcp", "endpoint": client.endpoint,
                  "level": _level(client), "vehicle": args.vehicle}

    print(f"  scenario : {spec.name}  [{spec.scenario_hash}]")
    print(f"  sampling : {1 / SAMPLE_INTERVAL_S:.0f} Hz, drained every second")
    print(f"  log      : {csv_path}")
    print(f"  running  : {args.minutes:.0f} minutes. Ctrl+C to stop early.\n")

    summary = collect_run(source, spec, csv_path, seconds=args.minutes * 60.0,
                          beamng=beamng)
    print(f"\n  {summary['rows']} rows, {summary['duration_s']:.0f} s, "
          f"{summary['dropped']} dropped")
    print(f"  {csv_path}")
    return 0


def _level(client) -> str:
    from sim.mcp_client import MCPError

    try:
        status = client.call("get_status")
    except MCPError:
        return "unknown"
    return (status or {}).get("level", "unknown") if isinstance(status, dict) else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_collect_run.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Verify the entry point runs end to end with no game**

Run: `python3 collect_trajectories.py "Smoke test" --fake --minutes 0.05`
Expected: exits 0, prints ~3 seconds of rows, writes a CSV and JSON into `runs/`

- [ ] **Step 6: Commit**

```bash
git add collect/run.py collect_trajectories.py tests/test_collect_run.py
git commit -m "Collect a trajectory end to end, game or no game"
```

---

### Task 10: Remove what stage 1 supersedes

Do this **last**. Until task 9 lands, `drive.py` is the only working collector and the code below is what it stands on.

**Files:**
- Delete: `behaviour/`, `agent.py`, `campaign/`, `collect.py`, `learn.py`, `policies/`
- Delete: `control/driver.py`, `control/pid.py`, `control/pure_pursuit.py`, `control/path.py`, `control/policy.py`, `control/policy_driver.py`, `control/learning.py`, `control/tuner.py`, `control/demonstration.py`, `control/calibration.py`, `control/alignment.py`, `control/roam_driver.py`
- Delete: `analysis/`, `does_the_game_matter.py`, `windows_lua_probe.py`, `windows_record.py`, `windows_ai_probe.py`
- Delete: their tests
- Modify: `drive.py` — remove, superseded by `collect_trajectories.py`
- Modify: `README.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: a green suite from task 9
- Produces: a tree in which `python3 -m pytest -q` passes and nothing imports `behaviour` or the controller stack

- [ ] **Step 1: Confirm the suite is green before removing anything**

Run: `python3 -m pytest -q`
Expected: PASS. Record the count — that is the baseline.

- [ ] **Step 2: Delete the superseded packages and their tests**

```bash
git rm -r behaviour campaign policies analysis
git rm agent.py collect.py learn.py drive.py does_the_game_matter.py
git rm windows_lua_probe.py windows_record.py windows_ai_probe.py
git rm control/driver.py control/pid.py control/pure_pursuit.py control/path.py
git rm control/policy.py control/policy_driver.py control/learning.py
git rm control/tuner.py control/demonstration.py control/calibration.py
git rm control/alignment.py control/roam_driver.py
git rm tests/test_behaviour_spec.py tests/test_generator.py tests/test_spec_cache.py
git rm tests/test_deepseek_client.py tests/test_driver.py tests/test_driver_stops.py
git rm tests/test_pid.py tests/test_pure_pursuit.py tests/test_policy_driver.py
git rm tests/test_learning.py tests/test_tuner.py tests/test_demonstration.py
git rm tests/test_calibration.py tests/test_alignment.py tests/test_roam_driver.py
git rm tests/test_collect.py tests/test_calibrated_game_value.py
git rm tests/test_drive_pacing.py tests/test_drive_send_off.py
git rm tests/test_going_nowhere.py tests/test_interventions.py
```

- [ ] **Step 3: Find what still imports the deleted modules**

Run:
```bash
grep -rn "behaviour\|BehaviourSpec\|campaign\|pure_pursuit\|policy_driver" \
  --include=*.py . | grep -v "^./docs"
```
Expected: hits only in `datalog/writer.py` (imports `BehaviourSpec`) and possibly `sim/mcp_backend.py`.

- [ ] **Step 4: Delete the superseded logger, which is the last consumer**

`collect/writer.py` replaces it, and it is the only remaining importer of `BehaviourSpec`.

```bash
git rm -r datalog tests/test_datalog.py
```

- [ ] **Step 5: Run the suite and fix what breaks**

Run: `python3 -m pytest -q`
Expected: PASS. If an import error names a deleted module, delete that consumer too or cut the import — do not reintroduce a deleted module to satisfy one.

- [ ] **Step 6: Rewrite the status section of `CLAUDE.md`**

Replace the "Status" section with:

```markdown
## Status

**Stage 1 built** (`collect/`): a Lua sampler running inside BeamNG's vehicle
physics VM at 100 Hz, drained over the game's first-party MCP server once a
second, writing a trajectory CSV whose every column is labelled measured or
derived in a sidecar.

Not built: the world model, the vehicle load model, the battery physics model,
and the NN/PINN comparison. See
`docs/superpowers/specs/2026-09-09-battery-rul-pipeline-design.md`.

The LLM behaviour agent, the campaign sampler and the hand-written pure-pursuit
controller were removed. BeamNG's AI drives; `ScenarioSpec` says how.

```bash
python3 -m pytest -q
python3 collect_trajectories.py "Smoke test" --fake --minutes 1

# on the Windows machine:
py collect_trajectories.py "Hot commute" --minutes 15
```
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Remove the controller and behaviour stack the MCP server made redundant"
```

---

## Self-review

**Spec coverage.** Section 4 of the spec (stage 1) is covered by tasks 1-9:
the Lua sampler and its async drain (2, 3, 7), the schema with provenance (1, 8),
the derived channels (4), the sidecar's per-run record (8), and the
`ScenarioSpec` that replaces `BehaviourSpec` (5). Section 11's removal list is
task 10. Sections 5-10 are stages 2-7 and are out of scope for this plan by
design — they need their own plans, and each needs stage 1's output to exist
first.

**Deliberately deferred.** Three things the spec asks for that this plan does
not build, because they belong to the stage they serve:

- routing and `aggression` actually reaching the AI. `collect_trajectories.py`
  records the scenario but does not yet call `set_ai`/`drive_to`; `control/ai_driver.py`
  survives task 10 and is wired in at the start of the stage 2 plan.
- moving `battery/` and `sim/engine.py` into `load/`. They are untouched here
  and move in the stage 3 plan.
- ambient temperature control (open question 12.1) and determinism (12.2). Both
  are investigations, not implementation, and neither blocks this plan.

**Placeholder scan.** No TBDs. Every code step carries the code. Every test step
carries the assertions.

**Type consistency.** `Channel`, `MEASURED`, `DERIVED`, `COLUMNS`, `PROVENANCE`
and `manifest()` are defined in task 1 and used under those names in tasks 2, 3,
6 and 8. `Drain` and `decode` from task 3 are used in task 7. `add_derived` from
task 4 is used in tasks 6 and 7. `TrajectorySource.interval_s` is set by both
implementations and read by `collect_run` in task 9. `ScenarioSpec.scenario_hash`
is used in tasks 8 and 9.
