# Battery Load and Cell Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn a driving trajectory into a battery dataset — current, voltage, state of charge, temperature, health and remaining life — with every assumption machine-readably flagged and a leakage detector proven to fire on the known-bad case.

**Architecture:** Two packages either side of one tabular contract. `load/` converts a canonical trajectory into battery current and under-bonnet temperature; `cell/` integrates those into cell state at 1 Hz within a trip, accumulates damage per trip, and extrapolates to end of life over a declared trip schedule. A legacy adapter lifts the recorded OutGauge-shaped CSV into the canonical schema, labelling every channel it had to assume or could not supply at all.

**Tech Stack:** Python 3.11+ stdlib only, pytest. No BeamNG, no Windows machine, no network.

**Spec:** `docs/superpowers/specs/2026-09-11-battery-load-and-cell-design.md`
**Parent spec:** `docs/superpowers/specs/2026-09-09-battery-rul-pipeline-design.md`

## Global Constraints

- **Stdlib only.** No numpy, no pandas, no torch. Nothing in this plan needs them.
- **Everything must be testable on macOS with no BeamNG and no Windows machine.** Hard constraint inherited from the parent plan.
- **No network in tests.** `pytest.ini` sets `filterwarnings = error`; keep tests warning-clean.
- **Positive current is discharge.** Everywhere, without exception. Engine-off parasitic is `+0.03`, cranking is `+350.0`, a charging battery is negative.
- **`collect/schema.py` is not modified.** Stage 1's tests pin its two provenance values. The wider vocabulary lives in `load/legacy.py`.
- **A computation that reads an `absent` column raises.** Never a silent zero.
- **Absolute remaining life is gated.** `cell/life.py` reports `scale_unfitted=True` unless supplied a rate constant fitted against full-life data, which does not exist yet.
- **The trajectory is streamed, never fully materialised.** The recorded file is 181,099 rows across 48 columns; holding it as dicts of floats costs roughly 700 MB.
- Run the suite with `python3 -m pytest -q` from the repository root.

## File Structure

```
load/
  __init__.py       (empty)
  spec.py           BatteryScenario: vehicle, electrical, battery and ambient constants
  legacy.py         Trajectory, the OutGauge adapter, the provenance vocabulary
  mechanical.py     road load and mechanical power
  thermal.py        under-bonnet temperature, including post-shutdown heat soak
  electrical.py     alternator capability, charge acceptance, battery current
  trips.py          trip and soak segmentation, crank events

cell/
  __init__.py       (empty)
  ecm.py            open-circuit voltage, internal resistance, terminal voltage
  aging.py          corrosion, sulfation and shedding states; SOH
  integrate.py      the 1 Hz within-trip loop; emits a trace and a damage vector
  life.py           trip schedule, SOH over time, remaining life, the scale gate
  leakage.py        analytic recoverability report
  dataset.py        two-table assembly and its sidecar

battery_run.py      entry point: trajectory in, dataset out
```

Moved, not rewritten (parent spec section 11): `battery/electrical.py` becomes
`load/electrical.py` in Task 5, and `battery/corrosion.py` becomes a dependency
of `cell/aging.py` in Task 8. The bay model leaves `sim/engine.py` in Task 4.

---

### Task 1: BatteryScenario

Every constant the battery layer needs that the game does not supply. Bounds-checked on construction and content-hashed, following `collect/scenario.py`'s pattern exactly.

`collect.ScenarioSpec` already carries `ambient_temp_c` and `accessory_load_a`. This class does **not** redeclare them as new truth: `from_sidecar` overlays them from a trajectory's sidecar when one exists, so a value is stated in one place.

**Files:**
- Create: `load/__init__.py` (empty)
- Create: `load/spec.py`
- Test: `tests/test_load_spec.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `BatteryScenario` — frozen dataclass, fields below
  - `BatteryScenario.scenario_hash -> str`
  - `BatteryScenario.to_dict() -> dict`, `BatteryScenario.from_dict(data) -> BatteryScenario`
  - `BatteryScenario.from_sidecar(path, name=...) -> BatteryScenario`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_spec.py
import json

import pytest

from load.spec import BatteryScenario


def test_defaults_are_a_mid_size_petrol_car():
    scenario = BatteryScenario(name="baseline")
    assert scenario.mass_kg == pytest.approx(1510.62)
    assert scenario.capacity_ah == pytest.approx(60.0)
    assert scenario.ambient_c == pytest.approx(25.0)


def test_out_of_range_values_are_refused():
    with pytest.raises(ValueError, match="capacity_ah"):
        BatteryScenario(name="x", capacity_ah=0.0)
    with pytest.raises(ValueError, match="ambient_c"):
        BatteryScenario(name="x", ambient_c=200.0)
    with pytest.raises(ValueError, match="hvac"):
        BatteryScenario(name="x", hvac=1.5)


def test_blank_name_is_refused():
    with pytest.raises(ValueError, match="name"):
        BatteryScenario(name="   ")


def test_hash_ignores_the_name_but_not_the_physics():
    a = BatteryScenario(name="one")
    b = BatteryScenario(name="two")
    c = BatteryScenario(name="one", capacity_ah=70.0)
    assert a.scenario_hash == b.scenario_hash
    assert a.scenario_hash != c.scenario_hash


def test_round_trips_through_a_dict():
    scenario = BatteryScenario(name="hot", ambient_c=42.0, lights=True)
    assert BatteryScenario.from_dict(scenario.to_dict()) == scenario


def test_from_sidecar_takes_ambient_and_accessories_from_the_trajectory(tmp_path):
    # The trajectory already recorded what the drive assumed. Restating it here
    # would let the two drift apart silently.
    sidecar = tmp_path / "run.json"
    sidecar.write_text(json.dumps({
        "scenario": {"ambient_temp_c": 38.0, "accessory_load_a": 47.0},
    }))
    scenario = BatteryScenario.from_sidecar(sidecar, name="from-run")
    assert scenario.ambient_c == pytest.approx(38.0)
    assert scenario.accessory_base_a == pytest.approx(47.0)
    assert scenario.name == "from-run"


def test_from_sidecar_falls_back_to_defaults_when_fields_are_missing(tmp_path):
    sidecar = tmp_path / "run.json"
    sidecar.write_text(json.dumps({"scenario": {}}))
    scenario = BatteryScenario.from_sidecar(sidecar, name="bare")
    assert scenario.ambient_c == pytest.approx(25.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_load_spec.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'load'`

- [ ] **Step 3: Write the implementation**

```python
# load/spec.py
"""Every constant the battery layer needs and the game does not supply.

BeamNG simulates no 12 V system at all -- no battery, no alternator, no
accessories. So all of this is declared rather than measured, which is exactly
why it is bounds-checked here and content-hashed: a plausible-looking number
outside physical range produces a run that looks fine and means nothing.

Ambient temperature and accessory load are the two fields that also appear on
`collect.ScenarioSpec`, because the drive itself had to assume them. They are
not redeclared as new truth here; `from_sidecar` reads them back off the
trajectory so the value is stated in one place.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

#: Frontal area times drag coefficient, m^2. A mid-size saloon.
CD_A_RANGE_M2 = (0.1, 2.0)
#: Rolling resistance coefficient. Tarmac and road tyres sit near 0.012.
CRR_RANGE = (0.001, 0.05)
MASS_RANGE_KG = (500.0, 4000.0)
#: A 12 V SLI battery. Anything outside this is a different application.
CAPACITY_RANGE_AH = (20.0, 200.0)
R0_RANGE_OHM = (0.001, 0.05)
ALTERNATOR_RANGE_A = (40.0, 250.0)
ACCESSORY_RANGE_A = (0.0, 120.0)
AMBIENT_RANGE_C = (-40.0, 60.0)
#: Cranking is hundreds of amps for a second or two. The largest current the
#: battery ever sees, and what the sulfation pathway tracks.
CRANK_RANGE_A = (100.0, 800.0)
CRANK_RANGE_S = (0.2, 10.0)
#: Key-off draw: alarm, clock, ECU keep-alive. Tens of milliamps.
PARASITIC_RANGE_A = (0.0, 1.0)
#: Charge acceptance coefficient, per hour. Around 2 gives ~1.2 A at 99% state
#: of charge on a 60 Ah battery and ~12 A at 90%.
C_ACCEPT_RANGE_PER_H = (0.1, 20.0)
C_TH_RANGE_J_PER_K = (1000.0, 100000.0)
H_RANGE_W_PER_K = (0.1, 50.0)
AIR_DENSITY_RANGE = (0.8, 1.5)

_UNIT_FRACTIONS = ("hvac",)


@dataclass(frozen=True)
class BatteryScenario:
    name: str
    # vehicle
    mass_kg: float = 1510.62  # ETK I-Series node-mass sum, measured over MCP
    cd_a_m2: float = 0.75
    crr: float = 0.012
    air_density: float = 1.225
    # electrical
    alternator_rated_a: float = 120.0
    accessory_base_a: float = 22.0  # ECU, ignition, fuel pump, instruments
    lights: bool = False
    hvac: float = 0.0  # blower and condenser fans, 0 to 1
    parasitic_a: float = 0.03
    crank_a: float = 350.0
    crank_s: float = 1.5
    # battery
    capacity_ah: float = 60.0
    r0_ohm: float = 0.006
    c_th_j_per_k: float = 15000.0
    h_w_per_k: float = 2.0
    c_accept_per_h: float = 2.0
    # environment
    ambient_c: float = 25.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")
        for field, (low, high) in (
            ("mass_kg", MASS_RANGE_KG),
            ("cd_a_m2", CD_A_RANGE_M2),
            ("crr", CRR_RANGE),
            ("air_density", AIR_DENSITY_RANGE),
            ("alternator_rated_a", ALTERNATOR_RANGE_A),
            ("accessory_base_a", ACCESSORY_RANGE_A),
            ("parasitic_a", PARASITIC_RANGE_A),
            ("crank_a", CRANK_RANGE_A),
            ("crank_s", CRANK_RANGE_S),
            ("capacity_ah", CAPACITY_RANGE_AH),
            ("r0_ohm", R0_RANGE_OHM),
            ("c_th_j_per_k", C_TH_RANGE_J_PER_K),
            ("h_w_per_k", H_RANGE_W_PER_K),
            ("c_accept_per_h", C_ACCEPT_RANGE_PER_H),
            ("ambient_c", AMBIENT_RANGE_C),
        ):
            value = getattr(self, field)
            if not low <= value <= high:
                raise ValueError(f"{field}: {value} outside [{low}, {high}]")
        for field in _UNIT_FRACTIONS:
            value = getattr(self, field)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field}: {value} outside [0.0, 1.0]")

    @property
    def scenario_hash(self) -> str:
        """Content address. The name labels the scenario, it does not define it."""
        payload = {k: v for k, v in asdict(self).items() if k != "name"}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "BatteryScenario":
        return cls(**data)

    @classmethod
    def from_sidecar(cls, path: Path | str, name: str, **overrides) -> "BatteryScenario":
        """Take ambient and accessory load from a trajectory's own sidecar.

        The drive already had to assume both. Restating them here would let the
        two drift apart without anything noticing.
        """
        recorded = json.loads(Path(path).read_text()).get("scenario", {})
        fields = {"name": name}
        if "ambient_temp_c" in recorded:
            fields["ambient_c"] = float(recorded["ambient_temp_c"])
        if "accessory_load_a" in recorded:
            fields["accessory_base_a"] = float(recorded["accessory_load_a"])
        fields.update(overrides)
        return cls(**fields)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_load_spec.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add load/__init__.py load/spec.py tests/test_load_spec.py
git commit -m "Declare what the game does not simulate about the battery"
```

---

### Task 2: The legacy adapter and its provenance vocabulary

The recorded file is the 34-column OutGauge shape, not `collect/schema.py`'s canonical 48. Lifting it is where assumptions enter the pipeline, so this is the task that has to make them legible.

Three facts about the recorded file, established by profiling it and not to be rediscovered: `brake` holds a 32-byte blob because the recorder captured OutGauge's `display1` field; `oil_pressure`, `clutch` and `game_time` are constant zero; and `mass_kg`, `engine_load`, `engine_torque_nm`, `ignition_level` and the wheel and brake-temperature channels are absent entirely.

**Files:**
- Create: `load/legacy.py`
- Test: `tests/test_load_legacy.py`

**Interfaces:**
- Consumes: `collect.schema.COLUMNS`, `collect.derive.add_derived`.
- Produces:
  - `PROVENANCES = ("measured", "derived", "assumed", "absent")`
  - `AbsentChannel(KeyError)` — raised when an absent column is read
  - `LEGACY_PROVENANCE: dict[str, str]` — canonical column name to provenance
  - `LEGACY_SOURCE: dict[str, str]` — canonical column name to a human-readable origin
  - `Sample(dict)` — refuses `__getitem__` on an absent column
  - `Trajectory` — `.samples() -> Iterator[Sample]`, `.at_hz(hz) -> Iterator[Sample]`, `.provenance: dict[str, str]`, `.source: Path`
  - `read_legacy(path) -> Trajectory`
  - `assumed_inputs(provenance, columns) -> tuple[str, ...]`
  - `manifest() -> list[dict]` — the sidecar's channel table

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_legacy.py
import math

import pytest

from collect.schema import COLUMNS
from load.legacy import (
    LEGACY_PROVENANCE,
    PROVENANCES,
    AbsentChannel,
    Sample,
    assumed_inputs,
    manifest,
    read_legacy,
)

HEADER = (
    "timestamp,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,"
    "up_x,up_y,up_z,roll,pitch,yaw,roll_vel,pitch_vel,yaw_vel,"
    "roll_acc,pitch_acc,yaw_acc,game_time,speed,rpm,turbo,engine_temp,fuel,"
    "oil_pressure,oil_temp,throttle,brake,clutch,gear"
)


def _row(t, speed=10.0, rpm=2000.0, pitch=0.0, engine_temp=90.0, throttle=0.3):
    return (
        f"{t},283.5,-713.4,148.5,-19.2,-0.1,-2.1,0.5,0.7,-0.1,"
        f"-0.11,0.03,0.99,0.03,{pitch},1.56,0.01,-0.02,-0.01,"
        f"0.16,-0.04,-0.09,0,{speed},{rpm},1.17,{engine_temp},0.99,"
        f"0.0,81.5,{throttle},b'\\x00\\x00',0,4"
    )


def _write(tmp_path, rows):
    path = tmp_path / "telemetry.csv"
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_every_canonical_column_has_a_provenance():
    assert set(LEGACY_PROVENANCE) == set(COLUMNS)
    assert set(LEGACY_PROVENANCE.values()) <= set(PROVENANCES)


def test_the_columns_this_file_cannot_supply_are_marked_absent():
    # brake is a blob, clutch and oil_pressure are constant zero, and the rest
    # were never in the OutGauge shape at all.
    for column in (
        "brake", "clutch_ratio", "engine_torque_nm", "ignition_level",
        "wheel_av_fl", "brake_temp_fl", "downforce_fl",
    ):
        assert LEGACY_PROVENANCE[column] == "absent", column


def test_mass_and_engine_load_are_assumed_not_measured():
    assert LEGACY_PROVENANCE["mass_kg"] == "assumed"
    assert LEGACY_PROVENANCE["engine_load"] == "assumed"


def test_speed_and_coolant_are_measured():
    assert LEGACY_PROVENANCE["speed_mps"] == "measured"
    assert LEGACY_PROVENANCE["coolant_c"] == "measured"


def test_reading_an_absent_column_raises_rather_than_returning_zero():
    sample = Sample({"brake": 0.0, "speed_mps": 10.0}, LEGACY_PROVENANCE)
    assert sample["speed_mps"] == pytest.approx(10.0)
    with pytest.raises(AbsentChannel, match="brake"):
        sample["brake"]


def test_samples_carry_every_canonical_column(tmp_path):
    path = _write(tmp_path, [_row(100.0), _row(100.1)])
    trajectory = read_legacy(path)
    first = next(trajectory.samples())
    assert set(first.keys()) == set(COLUMNS)


def test_time_is_rebased_to_zero_at_the_first_row(tmp_path):
    path = _write(tmp_path, [_row(1789128527.0), _row(1789128528.0)])
    times = [s["t_s"] for s in read_legacy(path).samples()]
    assert times[0] == pytest.approx(0.0)
    assert times[1] == pytest.approx(1.0)


def test_grade_comes_from_pitch(tmp_path):
    path = _write(tmp_path, [_row(0.0, pitch=0.1)])
    sample = next(read_legacy(path).samples())
    # dir_z is sin(pitch); grade_rad is its arcsine, so it returns pitch.
    assert sample["grade_rad"] == pytest.approx(0.1, abs=1e-6)


def test_engine_running_is_derived_from_rpm(tmp_path):
    path = _write(tmp_path, [_row(0.0, rpm=2000.0), _row(0.1, rpm=3.0)])
    running = [s["engine_running"] for s in read_legacy(path).samples()]
    assert running == [1.0, 0.0]


def test_at_hz_downsamples_without_loading_everything(tmp_path):
    rows = [_row(i * 0.01) for i in range(500)]  # 5 s at 100 Hz
    path = _write(tmp_path, rows)
    sampled = list(read_legacy(path).at_hz(1.0))
    assert len(sampled) == 5
    assert [round(s["t_s"]) for s in sampled] == [0, 1, 2, 3, 4]


def test_assumed_inputs_names_only_the_assumed_ones():
    assert assumed_inputs(
        LEGACY_PROVENANCE, ("speed_mps", "mass_kg", "coolant_c")
    ) == ("mass_kg",)


def test_manifest_records_why_each_column_is_what_it_is():
    rows = {entry["name"]: entry for entry in manifest()}
    assert rows["brake"]["provenance"] == "absent"
    assert "display1" in rows["brake"]["source"]
    assert rows["mass_kg"]["provenance"] == "assumed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_load_legacy.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'load.legacy'`

- [ ] **Step 3: Write the implementation**

```python
# load/legacy.py
"""Lifting the recorded OutGauge telemetry into the canonical schema.

The recording predates `collect/`. It is the 34-column OutGauge and MotionSim
shape, and the canonical trajectory has 48 columns, so lifting it means filling
some from constants and admitting that others are simply not there. That is
where assumptions enter the pipeline, which is why this module makes them
legible rather than convenient.

Three facts about the recorded file, established by profiling it:

  * `brake` holds a 32-byte blob. The recorder captured OutGauge's `display1`
    field, which sits next to the pedals in the packet layout.
  * `oil_pressure`, `clutch` and `game_time` are constant zero. Never populated.
  * `mass_kg`, `engine_load`, `engine_torque_nm`, `ignition_level`, the four
    wheel speeds, the four brake temperatures and `downforce_fl` were not in
    the OutGauge shape at all.

`collect/schema.py` has two provenance values and stage 1's tests pin that, so
it is untouched. The wider vocabulary lives here:

  measured  straight from the game
  derived   computed from measurements
  assumed   filled from a constant we chose -- an input, not an observation
  absent    not recoverable from this file

Reading an `absent` column raises. A silent zero in a road-load equation is a
car that weighs nothing, and it looks exactly like a working number.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterator

from collect.derive import add_derived
from collect.schema import COLUMNS

PROVENANCES = ("measured", "derived", "assumed", "absent")

#: ETK I-Series node-mass sum, measured once over MCP. A constant here.
ASSUMED_MASS_KG = 1510.62

#: Above this the engine is turning. The recorded file idles around 800 rpm and
#: sits at 3 rpm with the engine off, so the threshold is not delicate.
ENGINE_RUNNING_RPM = 50.0


class AbsentChannel(KeyError):
    """An absent column was read. There is no value to return."""


class Sample(dict):
    """A canonical sample that refuses to serve a column it does not have."""

    __slots__ = ("_provenance",)

    def __init__(self, values: dict, provenance: dict[str, str]) -> None:
        super().__init__(values)
        self._provenance = provenance

    def __getitem__(self, key):
        if self._provenance.get(key) == "absent":
            raise AbsentChannel(
                f"{key} is absent from this trajectory ({LEGACY_SOURCE.get(key, '')}); "
                "it cannot be read, and a zero here would be a silent lie"
            )
        return super().__getitem__(key)


def _measured(legacy_column: str) -> tuple[str, str]:
    return "measured", f"OutGauge/MotionSim column {legacy_column!r}"


LEGACY_SOURCE: dict[str, str] = {
    # pose and motion, straight from MotionSim
    "t_s": "timestamp, rebased to zero at the first row",
    "seq": "row index",
    "x_m": "pos_x", "y_m": "pos_y", "z_m": "pos_z",
    "vx_mps": "vel_x", "vy_mps": "vel_y", "vz_mps": "vel_z",
    "speed_mps": "speed",
    "ax_mps2": "acc_x", "ay_mps2": "acc_y", "az_mps2": "acc_z",
    "dir_x": "cos(pitch) cos(yaw)", "dir_y": "cos(pitch) sin(yaw)",
    "dir_z": "sin(pitch)",
    "altitude_m": "pos_z",
    # powertrain
    "rpm": "rpm",
    "throttle": "throttle",
    "gear_index": "gear",
    "steering": "yaw_vel, scaled",
    # thermal
    "coolant_c": "engine_temp",
    "oil_c": "oil_temp",
    "fuel_volume_l": "fuel fraction times a nominal tank",
    # assumed
    "mass_kg": f"constant {ASSUMED_MASS_KG} kg, the ETK I-Series node-mass sum",
    "engine_load": "throttle, as a stand-in; the real channel was not recorded",
    # absent
    "brake": "the recorder captured OutGauge's display1 blob instead",
    "clutch_ratio": "constant zero in the recording; never populated",
    "engine_torque_nm": "not in the OutGauge shape",
    "engine_av_rads": "not in the OutGauge shape",
    "exhaust_flow": "not in the OutGauge shape",
    "ignition_level": "not in the OutGauge shape",
    "parkingbrake": "not in the OutGauge shape",
    "odometer_m": "not in the OutGauge shape",
    "trip_m": "not in the OutGauge shape",
    "avg_wheel_av": "not in the OutGauge shape",
    "damage": "not in the OutGauge shape",
    "wheel_av_fl": "not in the OutGauge shape",
    "wheel_av_fr": "not in the OutGauge shape",
    "wheel_av_rl": "not in the OutGauge shape",
    "wheel_av_rr": "not in the OutGauge shape",
    "brake_temp_fl": "not in the OutGauge shape",
    "brake_temp_fr": "not in the OutGauge shape",
    "brake_temp_rl": "not in the OutGauge shape",
    "brake_temp_rr": "not in the OutGauge shape",
    "downforce_fl": "not in the OutGauge shape",
    # derived
    "engine_running": f"rpm > {ENGINE_RUNNING_RPM}",
    "grade_rad": "asin(dir_z)",
    "a_long_mps2": "ax_mps2 with the gravity component removed",
}

_ABSENT = frozenset({
    "brake", "clutch_ratio", "engine_torque_nm", "engine_av_rads",
    "exhaust_flow", "ignition_level", "parkingbrake", "odometer_m", "trip_m",
    "avg_wheel_av", "damage",
    "wheel_av_fl", "wheel_av_fr", "wheel_av_rl", "wheel_av_rr",
    "brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr",
    "downforce_fl",
})
_ASSUMED = frozenset({"mass_kg", "engine_load"})
_DERIVED = frozenset({"engine_running", "grade_rad", "a_long_mps2"})


def _build_provenance() -> dict[str, str]:
    table = {}
    for column in COLUMNS:
        if column in _ABSENT:
            table[column] = "absent"
        elif column in _ASSUMED:
            table[column] = "assumed"
        elif column in _DERIVED:
            table[column] = "derived"
        else:
            table[column] = "measured"
    return table


LEGACY_PROVENANCE: dict[str, str] = _build_provenance()

#: The recorded `fuel` channel is a fraction, not litres. A nominal tank turns
#: it into the canonical unit; the size is an assumption, but nothing in this
#: plan consumes fuel volume, so it costs nothing.
NOMINAL_TANK_L = 50.0


def _lift(raw: dict, index: int, t0: float) -> dict:
    pitch = float(raw["pitch"])
    yaw = float(raw["yaw"])
    values = {column: 0.0 for column in COLUMNS}
    values.update({
        "t_s": float(raw["timestamp"]) - t0,
        "seq": float(index),
        "x_m": float(raw["pos_x"]),
        "y_m": float(raw["pos_y"]),
        "z_m": float(raw["pos_z"]),
        "vx_mps": float(raw["vel_x"]),
        "vy_mps": float(raw["vel_y"]),
        "vz_mps": float(raw["vel_z"]),
        "speed_mps": float(raw["speed"]),
        "ax_mps2": float(raw["acc_x"]),
        "ay_mps2": float(raw["acc_y"]),
        "az_mps2": float(raw["acc_z"]),
        "dir_x": math.cos(pitch) * math.cos(yaw),
        "dir_y": math.cos(pitch) * math.sin(yaw),
        "dir_z": math.sin(pitch),
        "mass_kg": ASSUMED_MASS_KG,
        "rpm": float(raw["rpm"]),
        "engine_load": float(raw["throttle"]),
        "gear_index": float(raw["gear"]),
        "throttle": float(raw["throttle"]),
        "steering": float(raw["yaw_vel"]),
        "coolant_c": float(raw["engine_temp"]),
        "oil_c": float(raw["oil_temp"]),
        "fuel_volume_l": float(raw["fuel"]) * NOMINAL_TANK_L,
        "altitude_m": float(raw["pos_z"]),
        "engine_running": 1.0 if float(raw["rpm"]) > ENGINE_RUNNING_RPM else 0.0,
    })
    return add_derived(values)


class Trajectory:
    """A canonical trajectory, streamed.

    The recorded file is 181,099 rows across 48 columns. Materialising it as
    dicts of floats costs several hundred megabytes for no benefit: every
    consumer reads it once, front to back.
    """

    def __init__(self, path: Path, provenance: dict[str, str]) -> None:
        self.source = Path(path)
        self.provenance = provenance

    def samples(self) -> Iterator[Sample]:
        with self.source.open(newline="") as handle:
            t0 = None
            for index, raw in enumerate(csv.DictReader(handle)):
                if t0 is None:
                    t0 = float(raw["timestamp"])
                yield Sample(_lift(raw, index, t0), self.provenance)

    def at_hz(self, hz: float) -> Iterator[Sample]:
        """Every sample whose time crosses the next 1/hz boundary."""
        if hz <= 0.0:
            raise ValueError(f"hz must be positive, got {hz}")
        interval = 1.0 / hz
        next_at = 0.0
        for sample in self.samples():
            if sample["t_s"] + 1e-9 >= next_at:
                yield sample
                next_at += interval


def read_legacy(path: Path | str) -> Trajectory:
    return Trajectory(Path(path), LEGACY_PROVENANCE)


def assumed_inputs(
    provenance: dict[str, str], columns: tuple[str, ...]
) -> tuple[str, ...]:
    """Which of `columns` are assumed, so a result can say it depends on them."""
    return tuple(c for c in columns if provenance.get(c) == "assumed")


def manifest() -> list[dict]:
    """The channel table this adapter's sidecar carries, in column order."""
    return [
        {
            "name": column,
            "provenance": LEGACY_PROVENANCE[column],
            "source": LEGACY_SOURCE.get(column, ""),
        }
        for column in COLUMNS
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_load_legacy.py -q`
Expected: PASS, 12 tests

- [ ] **Step 5: Check it against the real recording**

Copy the recording into the gitignored `runs/` directory and read the first samples:

```bash
cp ~/Downloads/telemetry.csv runs/telemetry.csv
python3 -c "
from load.legacy import read_legacy
t = read_legacy('runs/telemetry.csv')
s = next(t.samples())
print('t_s', s['t_s'], 'speed', s['speed_mps'], 'rpm', s['rpm'], 'coolant', s['coolant_c'])
print('grade_rad', s['grade_rad'], 'a_long', s['a_long_mps2'])
print('at 1 Hz:', sum(1 for _ in t.at_hz(1.0)), 'samples')
"
```

Expected: `t_s 0.0`, speed near 19.29, rpm near 3136, coolant near 82.3, and about 1930 samples at 1 Hz.

- [ ] **Step 6: Commit**

```bash
git add load/legacy.py tests/test_load_legacy.py
git commit -m "Lift the recorded telemetry, and say what had to be assumed"
```

---

### Task 3: Road load and mechanical power

The mechanical path does not reach the battery as current — parent spec section 2 is emphatic that the EV formula is three orders of magnitude wrong here. It reaches it as heat, through Task 4. This task computes the force and power; Task 4 turns them into a bay temperature.

**Files:**
- Create: `load/mechanical.py`
- Test: `tests/test_load_mechanical.py`

**Interfaces:**
- Consumes: `load.spec.BatteryScenario`, a `Sample` from `load.legacy`.
- Produces:
  - `READS: tuple[str, ...]` — the columns this module reads
  - `GRAVITY_MPS2 = 9.81`
  - `road_load_n(sample, scenario) -> float`
  - `mech_power_w(sample, scenario) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_mechanical.py
import math

import pytest

from load.mechanical import READS, road_load_n, mech_power_w
from load.spec import BatteryScenario

SCENARIO = BatteryScenario(name="t", mass_kg=1500.0, cd_a_m2=0.75, crr=0.012,
                           air_density=1.2)


def _sample(speed=0.0, a_long=0.0, grade=0.0, mass=1500.0):
    return {"speed_mps": speed, "a_long_mps2": a_long, "grade_rad": grade,
            "mass_kg": mass}


def test_a_stationary_car_on_the_flat_needs_only_rolling_resistance():
    force = road_load_n(_sample(), SCENARIO)
    assert force == pytest.approx(0.012 * 1500.0 * 9.81, rel=1e-6)


def test_grade_adds_its_own_weight_component():
    flat = road_load_n(_sample(), SCENARIO)
    uphill = road_load_n(_sample(grade=0.05), SCENARIO)
    assert uphill - flat == pytest.approx(
        1500.0 * 9.81 * math.sin(0.05) - 0.012 * 1500.0 * 9.81 * (1 - math.cos(0.05)),
        rel=1e-6,
    )


def test_drag_grows_with_the_square_of_speed():
    at_10 = road_load_n(_sample(speed=10.0), SCENARIO)
    at_20 = road_load_n(_sample(speed=20.0), SCENARIO)
    rolling = 0.012 * 1500.0 * 9.81
    assert (at_20 - rolling) == pytest.approx(4.0 * (at_10 - rolling), rel=1e-6)


def test_acceleration_uses_the_measured_mass_not_the_scenario_default():
    # mass_kg is a trajectory column; a heavier recorded car must show up here.
    light = road_load_n(_sample(a_long=2.0, mass=1000.0), SCENARIO)
    heavy = road_load_n(_sample(a_long=2.0, mass=2000.0), SCENARIO)
    assert heavy > light


def test_power_is_force_times_speed():
    sample = _sample(speed=20.0, a_long=1.0)
    assert mech_power_w(sample, SCENARIO) == pytest.approx(
        road_load_n(sample, SCENARIO) * 20.0, rel=1e-9
    )


def test_a_stationary_car_does_no_mechanical_work():
    assert mech_power_w(_sample(speed=0.0), SCENARIO) == pytest.approx(0.0)


def test_reads_declares_every_column_it_touches():
    assert set(READS) == {"speed_mps", "a_long_mps2", "grade_rad", "mass_kg"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_load_mechanical.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'load.mechanical'`

- [ ] **Step 3: Write the implementation**

```python
# load/mechanical.py
"""Road load, and the mechanical power the engine has to produce for it.

None of this reaches the battery as current. Parent spec section 2: applying
the EV chain `P_elec = P_mech / eta + P_acc` to a combustion car yields battery
currents around 8000 A, three orders of magnitude out, because in a petrol car
the engine supplies the traction and the battery supplies none of it.

Mechanical power matters through a different route entirely -- engine load
heats the engine bay, and bay temperature drives grid corrosion. So this
module's output goes to `load/thermal.py`, never to `load/electrical.py`.

`mass_kg` is read from the trajectory rather than the scenario because the
canonical schema measures it live, as a node-mass sum. On the recorded file it
is an assumed constant, and `load.legacy.assumed_inputs` is how a caller finds
that out.
"""

from __future__ import annotations

import math

from load.spec import BatteryScenario

GRAVITY_MPS2 = 9.81

READS: tuple[str, ...] = ("speed_mps", "a_long_mps2", "grade_rad", "mass_kg")


def road_load_n(sample, scenario: BatteryScenario) -> float:
    """Total tractive force: inertia, gravity, aerodynamic drag, rolling."""
    mass = float(sample["mass_kg"])
    speed = float(sample["speed_mps"])
    grade = float(sample["grade_rad"])
    inertia = mass * float(sample["a_long_mps2"])
    gravity = mass * GRAVITY_MPS2 * math.sin(grade)
    drag = 0.5 * scenario.air_density * scenario.cd_a_m2 * speed * speed
    rolling = scenario.crr * mass * GRAVITY_MPS2 * math.cos(grade)
    return inertia + gravity + drag + rolling


def mech_power_w(sample, scenario: BatteryScenario) -> float:
    """Force times speed. Zero when stationary, however hard the engine works."""
    return road_load_n(sample, scenario) * float(sample["speed_mps"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_load_mechanical.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Commit**

```bash
git add load/mechanical.py tests/test_load_mechanical.py
git commit -m "Compute road load, and route it to heat rather than to current"
```

---
### Task 4: Detect and exclude the OutGauge dropout

Profiling the recorded file turned up a trap that would otherwise be consumed as data. From t = 1779.18 s to the end at t = 1929.94 s — 14,283 contiguous rows, 150.8 s — the **entire OutGauge packet is exactly zero**: `engine_temp`, `oil_temp`, `fuel`, `rpm` and `turbo` all drop to `0.0` in a single sample, while MotionSim keeps streaming a frozen position.

It is not a measurement. Fuel goes from 0.9256 to 0.0 between consecutive samples 10 ms apart, and coolant from 101.77 C to 0.0. OutGauge stopped sending; the car was parked and something ended the session.

Left in, coolant reads 0 C for the last two and a half minutes of the soak, the bay estimate collapses towards ambient, and the corrosion integral loses the hottest part of the soak — in a process that is exponential in temperature. This is the single most damaging thing that could silently enter this pipeline.

Excluded rather than interpolated: we do not know what the engine bay did, and inventing a decay curve there would be fabricating the measurement the whole model is meant to consume.

This is not legacy-only. A Lua drain can drop too, so the filter is its own module rather than a special case inside the adapter.

**Files:**
- Create: `load/dropout.py`
- Modify: `load/legacy.py` — `Trajectory.samples` filters through the detector
- Test: `tests/test_load_dropout.py`

**Interfaces:**
- Consumes: `load.legacy.Sample`.
- Produces:
  - `OUTGAUGE_CHANNELS: tuple[str, ...]`
  - `Dropout` — frozen dataclass `(start_s: float, end_s: float, rows: int)`
  - `DropoutFilter` — callable over an iterator of samples; `.spans -> tuple[Dropout, ...]`, `.rows_excluded -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_dropout.py
import pytest

from load.dropout import OUTGAUGE_CHANNELS, Dropout, DropoutFilter


def _sample(t, coolant=90.0, oil=95.0, fuel=40.0, speed=10.0):
    return {"t_s": t, "coolant_c": coolant, "oil_c": oil,
            "fuel_volume_l": fuel, "speed_mps": speed}


def test_live_samples_pass_through_untouched():
    samples = [_sample(0.0), _sample(1.0), _sample(2.0)]
    kept = list(DropoutFilter()(iter(samples)))
    assert [s["t_s"] for s in kept] == [0.0, 1.0, 2.0]


def test_a_block_where_every_outgauge_channel_is_zero_is_excluded():
    samples = [
        _sample(0.0), _sample(1.0),
        _sample(2.0, coolant=0.0, oil=0.0, fuel=0.0),
        _sample(3.0, coolant=0.0, oil=0.0, fuel=0.0),
    ]
    kept = list(DropoutFilter()(iter(samples)))
    assert [s["t_s"] for s in kept] == [0.0, 1.0]


def test_the_excluded_span_is_reported_for_the_sidecar():
    samples = [
        _sample(0.0),
        _sample(1.0, coolant=0.0, oil=0.0, fuel=0.0),
        _sample(2.0, coolant=0.0, oil=0.0, fuel=0.0),
    ]
    detector = DropoutFilter()
    list(detector(iter(samples)))
    assert detector.spans == (Dropout(start_s=1.0, end_s=2.0, rows=2),)
    assert detector.rows_excluded == 2


def test_one_zero_channel_alone_is_not_a_dropout():
    # An empty tank is a real measurement. Only the whole packet going dark is
    # a dropout.
    samples = [_sample(0.0), _sample(1.0, fuel=0.0)]
    kept = list(DropoutFilter()(iter(samples)))
    assert len(kept) == 2


def test_leading_zeros_before_any_live_sample_are_excluded_too():
    # Coolant is a real temperature even with the engine cold, so a zero here
    # still means no packet arrived.
    samples = [
        _sample(0.0, coolant=0.0, oil=0.0, fuel=0.0),
        _sample(1.0),
    ]
    kept = list(DropoutFilter()(iter(samples)))
    assert [s["t_s"] for s in kept] == [1.0]


def test_two_separate_dropouts_are_reported_separately():
    samples = [
        _sample(0.0),
        _sample(1.0, coolant=0.0, oil=0.0, fuel=0.0),
        _sample(2.0),
        _sample(3.0, coolant=0.0, oil=0.0, fuel=0.0),
    ]
    detector = DropoutFilter()
    list(detector(iter(samples)))
    assert len(detector.spans) == 2
    assert detector.spans[0].start_s == pytest.approx(1.0)
    assert detector.spans[1].start_s == pytest.approx(3.0)


def test_the_watched_channels_are_the_outgauge_ones():
    assert set(OUTGAUGE_CHANNELS) == {"coolant_c", "oil_c", "fuel_volume_l"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_load_dropout.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'load.dropout'`

- [ ] **Step 3: Write the implementation**

```python
# load/dropout.py
"""Telemetry that stopped arriving, as distinct from telemetry that says zero.

The recorded file ends with 14,283 contiguous rows -- 150.8 s -- in which every
OutGauge channel is exactly zero at once, while MotionSim keeps streaming a
frozen position. Fuel goes from 0.9256 to 0.0 between two samples 10 ms apart
and coolant from 101.77 C to 0.0. No engine does that. OutGauge stopped
sending.

Consumed as data it would read as the engine bay collapsing to ambient during
the hottest part of the soak, in a corrosion process that is exponential in
temperature. That is the most damaging thing that could silently enter this
pipeline, so it is detected rather than trusted.

Excluded, not interpolated. We do not know what the bay did during those
seconds, and a plausible decay curve drawn across the gap would be a
fabrication of exactly the measurement the model exists to consume.

The test is that *every* watched channel is zero simultaneously. One zero on
its own is a measurement -- an empty tank is a real state -- and only the whole
packet going dark indicates no packet at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

#: The channels an OutGauge packet carries that cannot all be truly zero at
#: once. Coolant is a real temperature even on a stone-cold engine.
OUTGAUGE_CHANNELS: tuple[str, ...] = ("coolant_c", "oil_c", "fuel_volume_l")

#: Floating-point slack. These arrive as exact zeros, not small numbers.
ZERO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Dropout:
    start_s: float
    end_s: float
    rows: int


class DropoutFilter:
    """Drops samples where the watched channels are all simultaneously zero.

    Stateful so a caller can read `spans` afterwards and record them in the
    sidecar. An excluded span that nothing reports is indistinguishable from
    data that was never collected.
    """

    def __init__(self, channels: tuple[str, ...] = OUTGAUGE_CHANNELS) -> None:
        self.channels = channels
        self._spans: list[Dropout] = []

    @property
    def spans(self) -> tuple[Dropout, ...]:
        return tuple(self._spans)

    @property
    def rows_excluded(self) -> int:
        return sum(span.rows for span in self._spans)

    def _is_dropout(self, sample) -> bool:
        return all(
            abs(float(sample[channel])) < ZERO_TOLERANCE
            for channel in self.channels
        )

    def __call__(self, samples: Iterable) -> Iterator:
        open_span: list | None = None
        for sample in samples:
            if self._is_dropout(sample):
                t = float(sample["t_s"])
                if open_span is None:
                    open_span = [t, t, 0]
                open_span[1] = t
                open_span[2] += 1
                continue
            if open_span is not None:
                self._spans.append(
                    Dropout(start_s=open_span[0], end_s=open_span[1], rows=open_span[2])
                )
                open_span = None
            yield sample
        if open_span is not None:
            self._spans.append(
                Dropout(start_s=open_span[0], end_s=open_span[1], rows=open_span[2])
            )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_load_dropout.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Wire the filter into the adapter**

In `load/legacy.py`, add the import and change `Trajectory` so every consumer
gets filtered samples and can read the spans afterwards:

```python
from load.dropout import DropoutFilter
```

```python
class Trajectory:
    def __init__(self, path: Path, provenance: dict[str, str]) -> None:
        self.source = Path(path)
        self.provenance = provenance
        self.dropouts = DropoutFilter()

    def _raw_samples(self) -> Iterator[Sample]:
        with self.source.open(newline="") as handle:
            t0 = None
            for index, raw in enumerate(csv.DictReader(handle)):
                if t0 is None:
                    t0 = float(raw["timestamp"])
                yield Sample(_lift(raw, index, t0), self.provenance)

    def samples(self) -> Iterator[Sample]:
        # A fresh detector per pass, or a second read would double-count spans.
        self.dropouts = DropoutFilter()
        yield from self.dropouts(self._raw_samples())
```

- [ ] **Step 6: Add the adapter test for it**

```python
# tests/test_load_legacy.py -- append

def test_a_dead_outgauge_packet_is_excluded_from_the_trajectory(tmp_path):
    live = _row(0.0, engine_temp=100.0)
    dead = (
        "1.0,283.5,-713.4,148.5,0,0,0,0,0,0,"
        "-0.11,0.03,0.99,0.03,0.0,1.56,0,0,0,"
        "0,0,0,0,0.0,0.0,0.0,0.0,0.0,"
        "0.0,0.0,0.0,b'\\x00',0,0"
    )
    path = _write(tmp_path, [live, dead])
    trajectory = read_legacy(path)
    kept = list(trajectory.samples())
    assert [s["t_s"] for s in kept] == [0.0]
    assert trajectory.dropouts.rows_excluded == 1
```

- [ ] **Step 7: Run the whole load suite**

Run: `python3 -m pytest tests/test_load_legacy.py tests/test_load_dropout.py -q`
Expected: PASS, 20 tests

- [ ] **Step 8: Check it against the real recording**

```bash
python3 -c "
from load.legacy import read_legacy
t = read_legacy('runs/telemetry.csv')
kept = sum(1 for _ in t.samples())
print('kept', kept, 'excluded', t.dropouts.rows_excluded)
for span in t.dropouts.spans:
    print(f'  dropout {span.start_s:.2f}s to {span.end_s:.2f}s, {span.rows} rows')
"
```

Expected: 166,816 kept, 14,283 excluded, one span from about 1779.18 s to 1929.94 s.

- [ ] **Step 9: Commit**

```bash
git add load/dropout.py load/legacy.py tests/test_load_dropout.py tests/test_load_legacy.py
git commit -m "Refuse to read a dead telemetry packet as a cold engine"
```

---

### Task 5: Under-bonnet temperature, including the heat soak

The dominant ageing pathway runs entirely through this number, and nothing measures it. BeamNG reports coolant and oil; neither is the air around the battery.

This task also corrects `sim/engine.py`, which sets bay coupling to `0.0` the instant the engine stops. At shutdown airflow stops while the block is still near 100 C, so bay temperature *rises* for several minutes before decaying. That peak is the hottest the battery ever gets and corrosion is Arrhenius in it. The recorded file is exactly this case: 587 s of usable soak straight after 1,192 s of driving with coolant at 130 C.

**Files:**
- Create: `load/thermal.py`
- Test: `tests/test_load_thermal.py`

**Interfaces:**
- Consumes: `load.spec.BatteryScenario`.
- Produces:
  - `READS: tuple[str, ...]`
  - `BAY_COUPLING_STATIC`, `BAY_COUPLING_SOAK`, `BAY_LOAD_GAIN`, `BAY_AIRFLOW_GAIN`, `BAY_TAU_S`
  - `bay_target_c(coolant_c, speed_mps, engine_load, engine_running, ambient_c) -> float`
  - `BayTemperature` — `.__init__(ambient_c, initial_c=None)`, `.step(sample, dt_s) -> float`, `.temperature_c`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_thermal.py
import pytest

from load.thermal import READS, BayTemperature, bay_target_c


def _sample(coolant=100.0, speed=0.0, load=0.3, running=1.0):
    return {"coolant_c": coolant, "speed_mps": speed,
            "engine_load": load, "engine_running": running}


def test_the_bay_sits_between_ambient_and_coolant():
    target = bay_target_c(coolant_c=100.0, speed_mps=20.0, engine_load=0.3,
                          engine_running=1.0, ambient_c=25.0)
    assert 25.0 < target < 100.0


def test_airflow_cools_the_bay():
    stopped = bay_target_c(100.0, 0.0, 0.3, 1.0, 25.0)
    moving = bay_target_c(100.0, 25.0, 0.3, 1.0, 25.0)
    assert moving < stopped


def test_load_heats_the_bay():
    light = bay_target_c(100.0, 20.0, 0.0, 1.0, 25.0)
    heavy = bay_target_c(100.0, 20.0, 1.0, 1.0, 25.0)
    assert heavy > light


def test_the_bay_never_reads_below_ambient():
    target = bay_target_c(coolant_c=10.0, speed_mps=30.0, engine_load=0.0,
                          engine_running=1.0, ambient_c=25.0)
    assert target >= 25.0


def test_shutting_a_hot_engine_off_makes_the_bay_hotter_not_cooler():
    # The heat soak. Airflow stops, the block is still at 130 C, and the bay
    # climbs. It is the hottest the battery ever gets, and the old model threw
    # it away by setting coupling to zero at shutdown.
    bay = BayTemperature(ambient_c=25.0)
    for _ in range(600):
        bay.step(_sample(coolant=130.0, speed=20.0, load=0.4, running=1.0), dt_s=1.0)
    driving = bay.temperature_c

    bay.step(_sample(coolant=130.0, speed=0.0, load=0.0, running=0.0), dt_s=1.0)
    for _ in range(120):
        bay.step(_sample(coolant=130.0, speed=0.0, load=0.0, running=0.0), dt_s=1.0)
    soaking = bay.temperature_c

    assert soaking > driving + 10.0


def test_the_bay_follows_the_coolant_down_once_the_engine_is_cold():
    bay = BayTemperature(ambient_c=25.0, initial_c=95.0)
    for _ in range(600):
        bay.step(_sample(coolant=25.0, speed=0.0, load=0.0, running=0.0), dt_s=1.0)
    assert bay.temperature_c == pytest.approx(25.0, abs=1.0)


def test_the_bay_lags_rather_than_jumping():
    bay = BayTemperature(ambient_c=25.0, initial_c=25.0)
    bay.step(_sample(coolant=130.0, speed=0.0, load=1.0, running=1.0), dt_s=1.0)
    # One second into a 60 s time constant moves it a little, not all the way.
    assert 25.0 < bay.temperature_c < 35.0


def test_a_zero_or_negative_step_is_refused():
    bay = BayTemperature(ambient_c=25.0)
    with pytest.raises(ValueError, match="dt_s"):
        bay.step(_sample(), dt_s=0.0)


def test_reads_declares_every_column_it_touches():
    assert set(READS) == {"coolant_c", "speed_mps", "engine_load", "engine_running"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_load_thermal.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'load.thermal'`

- [ ] **Step 3: Write the implementation**

```python
# load/thermal.py
"""Under-bonnet temperature: the number the dominant ageing pathway runs on.

Nothing measures it. BeamNG models coolant and oil temperature and OutGauge
reports coolant, but neither is the temperature of the air around the battery,
and BeamNG does not model an engine bay at all. So it is estimated here, and
the same estimator runs unchanged on real vehicle telemetry.

Measured coolant is used verbatim. The model never integrates over a
measurement -- it only ever lags behind one.

The structure is defensible; the coefficients are not calibrated against a real
engine bay. Ratios between conditions are the usable output and absolute
temperatures are indicative, which is a caveat that must travel with every
number this module produces.

**Heat soak.** `sim/engine.py` set the coupling to zero the moment the engine
stopped, which says the bay falls to ambient immediately. It does the opposite.
Airflow stops while the block is still at 130 C, so the bay climbs for several
minutes before decaying with the coolant -- and that peak is both the hottest
the battery ever gets and, corrosion being Arrhenius, disproportionately
damaging. `BAY_COUPLING_SOAK` is higher than `BAY_COUPLING_STATIC` for exactly
this reason.
"""

from __future__ import annotations

#: Fraction of the coolant-to-ambient rise the bay sees with the engine running
#: and no airflow.
BAY_COUPLING_STATIC = 0.62

#: And with the engine stopped. Higher, not lower: no fan, no airflow, and a
#: block still at operating temperature radiating into still air.
BAY_COUPLING_SOAK = 0.95

#: Extra bay heating at full load.
BAY_LOAD_GAIN = 0.25

#: How quickly road speed carries bay heat away, per m/s.
BAY_AIRFLOW_GAIN = 0.10

#: Bay thermal lag, seconds. Air responds faster than coolant.
BAY_TAU_S = 60.0

#: Cool-down time constant of a stopped engine, seconds. Coolant sheds its heat
#: in about a quarter of an hour; the battery, with a time constant of hours,
#: does not. The two must not be conflated -- using battery temperature as a
#: stand-in for coolant during a soak keeps the bay hot for hours that never
#: happened, in a term that is exponential in temperature.
COOLDOWN_TAU_S = 900.0

READS: tuple[str, ...] = ("coolant_c", "speed_mps", "engine_load", "engine_running")


def bay_target_c(
    coolant_c: float,
    speed_mps: float,
    engine_load: float,
    engine_running: float,
    ambient_c: float,
) -> float:
    """The temperature the bay is heading towards right now.

    Never below ambient: the bay cannot be colder than the air being drawn
    through it, whatever the coolant reads.
    """
    if engine_running >= 0.5:
        coupling = (
            BAY_COUPLING_STATIC
            * (1.0 + BAY_LOAD_GAIN * min(1.0, max(0.0, engine_load)))
            / (1.0 + BAY_AIRFLOW_GAIN * max(0.0, speed_mps))
        )
    else:
        coupling = BAY_COUPLING_SOAK
    rise = coupling * (coolant_c - ambient_c)
    return ambient_c + max(0.0, rise)


class BayTemperature:
    """First-order lag towards `bay_target_c`."""

    def __init__(self, ambient_c: float, initial_c: float | None = None) -> None:
        self.ambient_c = ambient_c
        self.temperature_c = ambient_c if initial_c is None else initial_c

    def step(self, sample, dt_s: float) -> float:
        if dt_s <= 0.0:
            raise ValueError(f"dt_s must be positive, got {dt_s}")
        target = bay_target_c(
            coolant_c=float(sample["coolant_c"]),
            speed_mps=float(sample["speed_mps"]),
            engine_load=float(sample["engine_load"]),
            engine_running=float(sample["engine_running"]),
            ambient_c=self.ambient_c,
        )
        # Clamped so a step longer than the time constant cannot overshoot.
        alpha = min(1.0, dt_s / BAY_TAU_S)
        self.temperature_c += (target - self.temperature_c) * alpha
        return self.temperature_c
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_load_thermal.py -q`
Expected: PASS, 9 tests

- [ ] **Step 5: Commit**

```bash
git add load/thermal.py tests/test_load_thermal.py
git commit -m "Estimate bay temperature, and stop discarding the heat soak"
```

---

### Task 6: Alternator, charge acceptance, and battery current

Where driving becomes the two channels the real dataset has. `battery/electrical.py` moves here and gains the charge-acceptance taper, without which the alternator refills the battery instantly after every crank and two of the five ageing pathways stop existing in the model.

**Positive is discharge**, throughout. The parent spec contradicted itself on this; section 5.3 of the current spec settles it.

**Files:**
- Create: `load/electrical.py`
- Delete: `battery/electrical.py`
- Modify: `datalog/writer.py` — update the import if it references `battery.electrical`
- Test: `tests/test_load_electrical.py`
- Delete: `tests/test_electrical.py`

**Interfaces:**
- Consumes: `load.spec.BatteryScenario`.
- Produces:
  - `PULLEY_RATIO`, `CUT_IN_SHAFT_RPM`, `FULL_OUTPUT_SHAFT_RPM`, `REGULATED_VOLTAGE_V`, `LIGHTS_LOAD_A`, `HVAC_LOAD_A`
  - `alternator_capability_a(rpm, rated_a) -> float`
  - `accessory_load_a(scenario) -> float`
  - `acceptance_temperature_factor(temp_c) -> float`
  - `charge_acceptance_a(soc, temp_c, scenario) -> float`
  - `battery_current_a(rpm, soc, temp_c, scenario, engine_running, cranking=False) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_electrical.py
import pytest

from load.electrical import (
    CUT_IN_SHAFT_RPM,
    accessory_load_a,
    acceptance_temperature_factor,
    alternator_capability_a,
    battery_current_a,
    charge_acceptance_a,
)
from load.spec import BatteryScenario

IDLE = BatteryScenario(name="idle", alternator_rated_a=120.0, accessory_base_a=22.0)
LOADED = BatteryScenario(name="loaded", alternator_rated_a=120.0,
                         accessory_base_a=22.0, lights=True, hvac=1.0)


def test_the_alternator_produces_nothing_below_cut_in():
    rpm = (CUT_IN_SHAFT_RPM / 2.6) - 100.0
    assert alternator_capability_a(rpm, 120.0) == pytest.approx(0.0)


def test_the_alternator_saturates_at_its_rating():
    assert alternator_capability_a(6000.0, 120.0) == pytest.approx(120.0)


def test_the_alternator_gives_roughly_a_quarter_of_its_rating_at_idle():
    # 800 rpm through a 2.6 pulley is a 2080 rpm shaft, a quarter of the way
    # from cut-in to full output.
    assert alternator_capability_a(800.0, 120.0) == pytest.approx(27.8, abs=1.0)


def test_accessory_load_adds_lights_and_blower():
    assert accessory_load_a(IDLE) == pytest.approx(22.0)
    assert accessory_load_a(LOADED) == pytest.approx(22.0 + 12.0 + 28.0)


def test_a_full_battery_accepts_no_charge():
    assert charge_acceptance_a(soc=1.0, temp_c=25.0, scenario=IDLE) == pytest.approx(0.0)


def test_acceptance_grows_as_the_battery_empties():
    nearly_full = charge_acceptance_a(0.99, 25.0, IDLE)
    depleted = charge_acceptance_a(0.90, 25.0, IDLE)
    assert nearly_full == pytest.approx(1.2, abs=0.1)
    assert depleted == pytest.approx(12.0, abs=0.5)


def test_a_cold_battery_accepts_charge_poorly():
    assert acceptance_temperature_factor(25.0) == pytest.approx(1.0)
    assert acceptance_temperature_factor(-10.0) < 0.5
    assert charge_acceptance_a(0.9, -10.0, IDLE) < charge_acceptance_a(0.9, 25.0, IDLE)


def test_cranking_is_a_large_discharge_whatever_else_is_true():
    current = battery_current_a(rpm=0.0, soc=1.0, temp_c=25.0, scenario=IDLE,
                                engine_running=0.0, cranking=True)
    assert current == pytest.approx(IDLE.crank_a)


def test_a_parked_car_draws_its_parasitic_load():
    current = battery_current_a(rpm=0.0, soc=1.0, temp_c=25.0, scenario=IDLE,
                                engine_running=0.0)
    assert current == pytest.approx(IDLE.parasitic_a)


def test_idling_with_everything_on_runs_a_deficit():
    # 62 A of load against about 28 A of alternator. This is the recharge
    # deficit pathway, and it only exists because the alternator is rpm-limited.
    current = battery_current_a(rpm=800.0, soc=1.0, temp_c=25.0, scenario=LOADED,
                                engine_running=1.0)
    assert current > 30.0


def test_cruising_charges_the_battery():
    current = battery_current_a(rpm=2500.0, soc=0.95, temp_c=25.0, scenario=IDLE,
                                engine_running=1.0)
    assert current < 0.0


def test_a_full_battery_at_cruise_draws_only_what_the_accessories_need():
    # With no acceptance headroom the alternator supplies the load and no more,
    # so the battery current is essentially zero rather than a phantom charge.
    current = battery_current_a(rpm=2500.0, soc=1.0, temp_c=25.0, scenario=IDLE,
                                engine_running=1.0)
    assert current == pytest.approx(0.0, abs=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_load_electrical.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'load.electrical'`

- [ ] **Step 3: Write the implementation**

```python
# load/electrical.py
"""Turning driving into battery current.

This is where the simulated and the measured data finally share a
representation: the real dataset's only usable columns are current and voltage,
and the simulator produces neither. BeamNG has no 12 V system at all -- the
~200 keys of `electrics.values` contain nothing matching volt, batt, amp,
current or alternator -- so all of this is modelled, always.

**Positive is discharge.** Everywhere, without exception. The parent spec wrote
engine-off parasitic as negative in one section and assumed the opposite sign
in the next; this is the settled convention and `dSoC/dt = -I/Q` follows it.

**Charge acceptance is what makes two of the ageing pathways exist.** A lead
battery near full charge will not take current at any voltage the regulator can
offer, because the limit is kinetic rather than ohmic. Without the `(1 - SoC)`
taper the alternator refills the battery instantly after every crank, and the
recharge-deficit and sulfation pathways vanish from the model entirely.

Coefficients are typical for a mid-size petrol car, not measurements of one.
Comparisons between runs are usable; absolute amps are indicative.
"""

from __future__ import annotations

from load.spec import BatteryScenario

#: Alternator shaft speed relative to the crank, through the pulley.
PULLEY_RATIO = 2.6

#: Below this shaft speed the alternator produces nothing.
CUT_IN_SHAFT_RPM = 1200.0

#: And at this one it reaches its rating.
FULL_OUTPUT_SHAFT_RPM = 5000.0

#: What the regulator holds the bus at while charging.
REGULATED_VOLTAGE_V = 14.2

#: Headlights, side lights and tail lights together.
LIGHTS_LOAD_A = 12.0

#: Blower and condenser fans at full. The compressor is belt-driven and costs
#: fuel rather than amps; its fans do not.
HVAC_LOAD_A = 28.0

#: Charge acceptance falls off below this and is unimpaired above it.
ACCEPTANCE_WARM_C = 25.0
ACCEPTANCE_COLD_C = -10.0
ACCEPTANCE_COLD_FACTOR = 0.2


def alternator_capability_a(rpm: float, rated_a: float) -> float:
    """The most the alternator could supply at this engine speed.

    At idle a typical alternator gives roughly a quarter to a half of its
    rating, which is why a stationary car with the blower and lights on runs a
    deficit however large the alternator is.
    """
    shaft_rpm = max(0.0, rpm) * PULLEY_RATIO
    if shaft_rpm <= CUT_IN_SHAFT_RPM:
        return 0.0
    fraction = (shaft_rpm - CUT_IN_SHAFT_RPM) / (
        FULL_OUTPUT_SHAFT_RPM - CUT_IN_SHAFT_RPM
    )
    return rated_a * min(1.0, fraction)


def accessory_load_a(scenario: BatteryScenario) -> float:
    """What the car's electrics are asking for. A scenario input, not a measurement."""
    return (
        scenario.accessory_base_a
        + HVAC_LOAD_A * min(1.0, max(0.0, scenario.hvac))
        + (LIGHTS_LOAD_A if scenario.lights else 0.0)
    )


def acceptance_temperature_factor(temp_c: float) -> float:
    """Cold batteries take charge badly. Linear between the two anchors."""
    if temp_c >= ACCEPTANCE_WARM_C:
        return 1.0
    if temp_c <= ACCEPTANCE_COLD_C:
        return ACCEPTANCE_COLD_FACTOR
    span = ACCEPTANCE_WARM_C - ACCEPTANCE_COLD_C
    return ACCEPTANCE_COLD_FACTOR + (1.0 - ACCEPTANCE_COLD_FACTOR) * (
        (temp_c - ACCEPTANCE_COLD_C) / span
    )


def charge_acceptance_a(
    soc: float, temp_c: float, scenario: BatteryScenario
) -> float:
    """The most current the battery will take, whatever is on offer.

    Goes to zero at full charge. This taper is the whole reason a short trip
    fails to recover the charge a crank took.
    """
    headroom = max(0.0, 1.0 - min(1.0, soc))
    return (
        scenario.c_accept_per_h
        * scenario.capacity_ah
        * headroom
        * acceptance_temperature_factor(temp_c)
    )


def battery_current_a(
    rpm: float,
    soc: float,
    temp_c: float,
    scenario: BatteryScenario,
    engine_running: float,
    cranking: bool = False,
) -> float:
    """Net current at the battery. Positive is discharge.

        engine off      +parasitic
        cranking        +crank_a
        engine running  accessories - what the alternator actually delivers
    """
    if cranking:
        return scenario.crank_a
    if engine_running < 0.5:
        return scenario.parasitic_a
    load = accessory_load_a(scenario)
    wanted = load + charge_acceptance_a(soc, temp_c, scenario)
    delivered = min(alternator_capability_a(rpm, scenario.alternator_rated_a), wanted)
    return load - delivered
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_load_electrical.py -q`
Expected: PASS, 12 tests

- [ ] **Step 5: Remove the superseded module**

`battery/electrical.py` is superseded in full. Check what still imports it, fix
those imports, then delete it and its test.

```bash
grep -rn "battery.electrical\|battery import electrical" --include="*.py" .
git rm battery/electrical.py tests/test_electrical.py
python3 -m pytest -q
```

Expected: the suite passes. If `datalog/writer.py` imported it, point that
import at `load.electrical` and adjust for the sign convention.

- [ ] **Step 6: Commit**

```bash
git add -A load/electrical.py tests/test_load_electrical.py battery tests datalog
git commit -m "Make the alternator obey what the battery will actually accept"
```

---

### Task 7: Trips, soaks and cranks

The sulfation pathway is about repeated cranking without a full recharge, so trip boundaries are the unit the ageing model works in. On the recorded file this finds one trip and one soak and — correctly — **zero cranks**, because the engine is already running at t = 0. A crank we did not observe must not be invented.

**Files:**
- Create: `load/trips.py`
- Test: `tests/test_load_trips.py`

**Interfaces:**
- Consumes: samples from `load.legacy.Trajectory`.
- Produces:
  - `MIN_SOAK_S = 5.0`
  - `Segment` — frozen dataclass `(kind: str, start_s: float, end_s: float, cranked: bool)` where `kind` is `"trip"` or `"soak"`
  - `segment(samples, min_soak_s=MIN_SOAK_S) -> list[Segment]`
  - `crank_times(segments) -> tuple[float, ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_load_trips.py
import pytest

from load.trips import Segment, crank_times, segment


def _run(spans):
    """spans: list of (running, seconds). One sample per second."""
    t = 0.0
    for running, seconds in spans:
        for _ in range(seconds):
            yield {"t_s": t, "engine_running": 1.0 if running else 0.0}
            t += 1.0


def test_a_drive_then_a_park_is_one_trip_and_one_soak():
    segments = segment(_run([(True, 60), (False, 60)]))
    assert [s.kind for s in segments] == ["trip", "soak"]
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[0].end_s == pytest.approx(59.0)


def test_the_first_trip_is_not_a_crank_when_the_engine_is_already_running():
    # The recording starts mid-drive. Counting that as a crank would invent a
    # 350 A event that never happened.
    segments = segment(_run([(True, 60), (False, 60)]))
    assert segments[0].cranked is False
    assert crank_times(segments) == ()


def test_a_trip_that_follows_a_soak_is_a_crank():
    segments = segment(_run([(True, 30), (False, 30), (True, 30)]))
    assert [s.kind for s in segments] == ["trip", "soak", "trip"]
    assert segments[2].cranked is True
    assert crank_times(segments) == (60.0,)


def test_a_brief_stall_does_not_split_a_trip():
    # A momentary rpm dip is not a trip boundary, and treating it as one would
    # manufacture cranks out of noise.
    segments = segment(_run([(True, 30), (False, 2), (True, 30)]))
    assert [s.kind for s in segments] == ["trip"]
    assert crank_times(segments) == ()


def test_a_recording_that_starts_parked_gives_a_soak_first():
    segments = segment(_run([(False, 30), (True, 30)]))
    assert [s.kind for s in segments] == ["soak", "trip"]
    assert segments[1].cranked is True


def test_an_empty_trajectory_has_no_segments():
    assert segment(iter(())) == []


def test_segments_cover_the_whole_trajectory_without_gaps():
    segments = segment(_run([(True, 30), (False, 30), (True, 30)]))
    for earlier, later in zip(segments, segments[1:]):
        assert later.start_s > earlier.end_s
    assert segments[0].start_s == pytest.approx(0.0)
    assert segments[-1].end_s == pytest.approx(89.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_load_trips.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'load.trips'`

- [ ] **Step 3: Write the implementation**

```python
# load/trips.py
"""Trip boundaries, which is the unit the ageing model actually works in.

One forty-minute drive and eight five-minute drives cover the same distance and
are entirely different for an SLI battery: each restart is a crank of several
hundred amps, and a short trip may not run long enough to put back what the
crank took. That is the sulfation pathway, and it is invisible unless the
trajectory is cut into trips.

Two rules that exist to stop the model inventing events:

**A recording that starts mid-drive contributes no crank.** The recorded file
begins with the engine already turning at 3136 rpm. Counting that as a crank
would add a 350 A event that never happened.

**A momentary dip below the running threshold is not a trip boundary.** Engine
speed passes through zero during a stall, a gear change on a rough model, or a
single dropped sample. `MIN_SOAK_S` is what separates a park from a stumble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

#: Shorter than this and the engine did not really stop.
MIN_SOAK_S = 5.0


@dataclass(frozen=True)
class Segment:
    kind: str  # "trip" or "soak"
    start_s: float
    end_s: float
    cranked: bool


def _runs(samples: Iterable) -> list[list]:
    """Collapse samples into [running, start_s, end_s] runs."""
    runs: list[list] = []
    for sample in samples:
        running = float(sample["engine_running"]) >= 0.5
        t = float(sample["t_s"])
        if runs and runs[-1][0] == running:
            runs[-1][2] = t
        else:
            runs.append([running, t, t])
    return runs


def segment(samples: Iterable, min_soak_s: float = MIN_SOAK_S) -> list[Segment]:
    """Cut a trajectory into trips and soaks, marking which trips were cranked."""
    runs = _runs(samples)
    if not runs:
        return []

    # Absorb too-short stops back into the trip around them.
    merged: list[list] = []
    for run in runs:
        running, start, end = run
        too_short = (not running) and (end - start) < min_soak_s
        if too_short and merged and merged[-1][0]:
            merged[-1][2] = end
            continue
        if merged and merged[-1][0] == running:
            merged[-1][2] = end
            continue
        merged.append(list(run))

    segments: list[Segment] = []
    for index, (running, start, end) in enumerate(merged):
        # A trip is cranked only if we watched the engine stop beforehand.
        cranked = bool(running and index > 0 and not merged[index - 1][0])
        segments.append(
            Segment(
                kind="trip" if running else "soak",
                start_s=start,
                end_s=end,
                cranked=cranked,
            )
        )
    return segments


def crank_times(segments: Iterable[Segment]) -> tuple[float, ...]:
    """When the starter turned, as far as we actually observed."""
    return tuple(s.start_s for s in segments if s.cranked)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_load_trips.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Check it against the real recording**

```bash
python3 -c "
from load.legacy import read_legacy
from load.trips import crank_times, segment
t = read_legacy('runs/telemetry.csv')
segments = segment(t.samples())
for s in segments:
    print(f'{s.kind:5s} {s.start_s:8.1f} to {s.end_s:8.1f} ({s.end_s-s.start_s:7.1f} s) cranked={s.cranked}')
print('cranks observed:', crank_times(segments))
"
```

Expected: one trip of about 1,192 s, one soak of about 587 s, and no cranks —
the dropout having already removed the final 150.8 s.

- [ ] **Step 6: Commit**

```bash
git add load/trips.py tests/test_load_trips.py
git commit -m "Cut the drive into trips without inventing a crank we never saw"
```

---
### Task 8: The equivalent circuit

Open-circuit voltage, internal resistance and terminal voltage. The resistance feedback matters more than it looks: as health falls the resistance rises, so cranking sags harder and dissipates more heat, which ages it faster still. That loop is what bends the fade curve instead of leaving it a straight line.

**Files:**
- Create: `cell/__init__.py` (empty)
- Create: `cell/ecm.py`
- Test: `tests/test_cell_ecm.py`

**Interfaces:**
- Consumes: `load.electrical.REGULATED_VOLTAGE_V`.
- Produces:
  - `R_SOH_GAIN`, `R_TEMP_COEFF`, `R_SOC_GAIN`, `CHARGE_SAG_OHM`
  - `open_circuit_v(soc) -> float`
  - `r_int_ohm(r0_ohm, soh, temp_c, soc) -> float`
  - `terminal_v(soc, current_a, r_int_ohm, alternator_capability_a) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cell_ecm.py
import pytest

from cell.ecm import r_int_ohm, terminal_v


def test_a_full_battery_rests_at_about_12_7_volts():
    assert open_circuit_v(1.0) == pytest.approx(12.7)


def test_open_circuit_voltage_falls_with_state_of_charge():
    assert open_circuit_v(0.5) < open_circuit_v(1.0)


def test_state_of_charge_outside_the_range_is_clamped_not_extrapolated():
    assert open_circuit_v(1.4) == pytest.approx(open_circuit_v(1.0))
    assert open_circuit_v(-0.2) == pytest.approx(open_circuit_v(0.0))


def test_a_healthy_warm_full_battery_is_at_its_nominal_resistance():
    assert r_int_ohm(0.006, soh=1.0, temp_c=25.0, soc=1.0) == pytest.approx(0.006)


def test_resistance_rises_as_the_battery_ages():
    assert r_int_ohm(0.006, 0.8, 25.0, 1.0) > r_int_ohm(0.006, 1.0, 25.0, 1.0)


def test_resistance_rises_in_the_cold():
    # Why a marginal battery survives all summer and fails on the first frost.
    assert r_int_ohm(0.006, 1.0, -18.0, 1.0) > 1.5 * r_int_ohm(0.006, 1.0, 25.0, 1.0)


def test_resistance_rises_as_the_battery_discharges():
    assert r_int_ohm(0.006, 1.0, 25.0, 0.5) > r_int_ohm(0.006, 1.0, 25.0, 1.0)


def test_a_discharging_battery_sags_below_its_open_circuit_voltage():
    v = terminal_v(soc=1.0, current_a=350.0, r_int_ohm=0.006,
                   alternator_capability_a=0.0)
    assert v == pytest.approx(12.7 - 350.0 * 0.006)


def test_a_worn_battery_sags_further_on_the_same_crank():
    healthy = terminal_v(1.0, 350.0, r_int_ohm(0.006, 1.0, 25.0, 1.0), 0.0)
    worn = terminal_v(1.0, 350.0, r_int_ohm(0.006, 0.8, 25.0, 1.0), 0.0)
    assert worn < healthy


def test_the_regulator_holds_the_bus_up_while_the_alternator_is_working():
    # Charging, the reading says more about the alternator than the battery.
    v = terminal_v(soc=0.9, current_a=-20.0, r_int_ohm=0.006,
                   alternator_capability_a=120.0)
    assert 13.8 < v < 14.3


def test_a_charging_current_with_no_alternator_is_not_a_regulated_bus():
    v = terminal_v(soc=0.9, current_a=-20.0, r_int_ohm=0.006,
                   alternator_capability_a=0.0)
    assert v < 13.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cell_ecm.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cell'`

- [ ] **Step 3: Write the implementation**

```python
# cell/ecm.py
"""The equivalent circuit: what a voltmeter at the terminals would read.

Voltage is one of only two channels the real measured dataset has, so this is
half of the shared representation between simulation and measurement.

While the alternator carries the load the regulator holds the bus up, and the
reading says more about the alternator than about the battery. It is when the
battery is carrying the load -- cranking above all -- that terminal voltage
reveals its condition, which is exactly why a failing battery is discovered on
a cold morning rather than during a drive.

The resistance feedback is the important part. As health falls, resistance
rises; cranking then sags harder and dissipates more heat, which ages the
battery faster still. Without that loop the fade curve is a straight line and
says nothing.
"""

from __future__ import annotations

from load.electrical import REGULATED_VOLTAGE_V

#: Resting voltage spans 11.9 V empty to 12.7 V full, roughly linear over the
#: usable range for a flooded 12 V battery.
OCV_EMPTY_V = 11.9
OCV_SPAN_V = 0.8

#: Resistance multiplier at end of life. A worn battery is a high-resistance
#: battery; this is what makes the crank sag diagnostic.
R_SOH_GAIN = 2.0

#: Resistance roughly doubles between 25 C and -18 C.
R_TEMP_COEFF = 0.016

#: And rises as the battery discharges and the electrolyte weakens.
R_SOC_GAIN = 0.5

#: Small bus sag under heavy charging current.
CHARGE_SAG_OHM = 0.002


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def open_circuit_v(soc: float) -> float:
    """Resting voltage. Clamped rather than extrapolated outside [0, 1]."""
    return OCV_EMPTY_V + OCV_SPAN_V * _clamp(soc)


def r_int_ohm(r0_ohm: float, soh: float, temp_c: float, soc: float) -> float:
    """Internal resistance: nominal, times age, times cold, times depletion.

    Each factor is 1.0 at the reference condition -- healthy, 25 C, full -- so
    `r0_ohm` stays the meaning it has on a datasheet.
    """
    f_soh = 1.0 + R_SOH_GAIN * (1.0 - _clamp(soh))
    f_temp = 1.0 + R_TEMP_COEFF * max(0.0, 25.0 - temp_c)
    f_soc = 1.0 + R_SOC_GAIN * (1.0 - _clamp(soc))
    return r0_ohm * f_soh * f_temp * f_soc


def terminal_v(
    soc: float,
    current_a: float,
    r_int_ohm: float,
    alternator_capability_a: float,
) -> float:
    """Terminal voltage. Positive `current_a` is discharge."""
    charging = current_a < 0.0 and alternator_capability_a > 0.0
    if charging:
        return REGULATED_VOLTAGE_V - CHARGE_SAG_OHM * abs(current_a)
    return open_circuit_v(soc) - current_a * r_int_ohm
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cell_ecm.py -q`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add cell/__init__.py cell/ecm.py tests/test_cell_ecm.py
git commit -m "Let a worn battery sag the way a worn battery does"
```

---

### Task 9: Corrosion, sulfation, shedding, and the health they add up to

Three ageing states and the rate constants that turn damage into health. `battery/corrosion.py` supplies the Arrhenius temperature dependence unchanged — it is already anchored to the "+10 C halves life" rule and lands at ~62 kJ/mol, inside the 50-70 kJ/mol band reported for lead-acid grid corrosion.

The rate constants here are **not fitted**. No battery in this project has reached end of life. They are set so that ordinary usage lands inside the literature band of 3-5 years, which is a sanity anchor and not a calibration — `AgingRates.fitted` carries that fact into every result downstream.

Sulfation is a genuine state rather than a summary statistic: crystals grow more slowly as they get larger, and dissolve in proportion to how much is there. Both terms depend on the current crystal size, so no accumulated total can express them.

**Files:**
- Create: `cell/aging.py`
- Test: `tests/test_cell_aging.py`

**Interfaces:**
- Consumes: `battery.corrosion.corrosion_rate`.
- Produces:
  - `EOL_SOH = 0.8`, `LOW_SOC`, `FULL_SOC`
  - `Damage` — frozen dataclass, fields listed in the implementation
  - `AgingRates` — frozen dataclass with `fitted: bool = False`
  - `AgingState` — mutable dataclass `(corrosion_hours, crystal, shedding)`
  - `Damage.zero() -> Damage`, `Damage.__add__`
  - `accumulate(state, damage, rates) -> AgingState`
  - `soh(state, rates) -> float`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cell_aging.py
import dataclasses

import pytest

from cell.aging import EOL_SOH, AgingRates, AgingState, Damage, accumulate, soh

RATES = AgingRates()


def _damage(**kwargs):
    return dataclasses.replace(Damage.zero(), **kwargs)


def test_a_new_battery_is_in_perfect_health():
    assert soh(AgingState(), RATES) == pytest.approx(1.0)


def test_corrosion_hours_reduce_health():
    state = accumulate(AgingState(), _damage(corrosion_equivalent_h=5000.0), RATES)
    assert soh(state, RATES) < 1.0


def test_corrosion_damage_is_sublinear_in_time():
    # Schiffer: the corrosion layer grows as roughly t^0.6, so doubling the
    # exposure does less than double the damage.
    once = accumulate(AgingState(), _damage(corrosion_equivalent_h=10000.0), RATES)
    twice = accumulate(AgingState(), _damage(corrosion_equivalent_h=20000.0), RATES)
    loss_once = 1.0 - soh(once, RATES)
    loss_twice = 1.0 - soh(twice, RATES)
    assert loss_twice < 2.0 * loss_once
    assert loss_twice > loss_once


def test_sulfation_growth_slows_as_crystals_get_larger():
    # The rate depends on the current crystal size, which is why this has to be
    # a state and cannot be a running total.
    fresh = accumulate(AgingState(), _damage(low_soc_hours=100.0), RATES)
    sulfated = accumulate(
        AgingState(crystal=0.8), _damage(low_soc_hours=100.0), RATES
    )
    assert (fresh.crystal - 0.0) > (sulfated.crystal - 0.8)


def test_a_full_charge_dissolves_sulfation():
    state = accumulate(
        AgingState(crystal=0.5), _damage(full_charge_hours=200.0), RATES
    )
    assert state.crystal < 0.5


def test_sulfation_stays_inside_its_bounds():
    state = accumulate(AgingState(crystal=0.99), _damage(low_soc_hours=1e6), RATES)
    assert 0.0 <= state.crystal <= 1.0
    state = accumulate(AgingState(crystal=0.01), _damage(full_charge_hours=1e6), RATES)
    assert 0.0 <= state.crystal <= 1.0


def test_vibration_sheds_plate_material():
    state = accumulate(AgingState(), _damage(vibration_dose=1e5), RATES)
    assert state.shedding > 0.0
    assert soh(state, RATES) < 1.0


def test_health_never_goes_negative():
    state = AgingState(corrosion_hours=1e9, crystal=1.0, shedding=1.0)
    assert soh(state, RATES) >= 0.0


def test_damage_adds_componentwise():
    a = _damage(duration_s=10.0, ah_throughput=1.0, low_soc_hours=2.0)
    b = _damage(duration_s=5.0, ah_throughput=0.5, low_soc_hours=1.0)
    total = a + b
    assert total.duration_s == pytest.approx(15.0)
    assert total.ah_throughput == pytest.approx(1.5)
    assert total.low_soc_hours == pytest.approx(3.0)


def test_the_default_rate_constants_are_flagged_unfitted():
    # No battery in this project has reached end of life. Anything built on
    # these constants has to say so.
    assert AgingRates().fitted is False
    assert AgingRates(fitted=True).fitted is True


#: A day of ordinary use, in Arrhenius-weighted hours at 25 C. Derived in the
#: AgingRates docstring: 1.5 h driving with a 60 C bay, 1 h of heat soak near
#: 75 C, and 21.5 h parked at 25 C.
ORDINARY_DAY_H = 78.7


def _years_to_eol(daily_h, rates=RATES, limit_years=40):
    state = AgingState()
    day = 0
    while soh(state, rates) > EOL_SOH and day < limit_years * 365:
        state = accumulate(state, _damage(corrosion_equivalent_h=daily_h), rates)
        day += 1
    return day / 365


def test_the_default_constants_land_inside_the_literature_band():
    # A sanity anchor, not a calibration: SLI batteries last 3-5 years. A model
    # saying 12 is broken.
    years = _years_to_eol(ORDINARY_DAY_H)
    assert 3.0 <= years <= 5.0, f"end of life at {years:.2f} years"


def test_a_hot_climate_is_markedly_worse():
    # 42 C ambient is about 3.1x the weighted exposure of 25 C.
    years = _years_to_eol(ORDINARY_DAY_H * 3.09)
    assert years < _years_to_eol(ORDINARY_DAY_H)
    assert years < 2.0


def test_a_gently_used_garaged_car_is_not_claimed_to_last_forever():
    # The other end of the sanity band. Nothing here may predict 12 years.
    assert _years_to_eol(28.1) < 12.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cell_aging.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cell.aging'`

- [ ] **Step 3: Write the implementation**

```python
# cell/aging.py
"""Ageing: corrosion, sulfation, shedding, and the health they add up to.

Follows Schiffer et al. (2007), the weighted-Ah-throughput model for lead-acid,
rather than the Wang power law, which is lithium and does not apply to this
chemistry at all.

**The rate constants are not fitted.** No battery in this project has reached
end of life and the open full-life lead-acid dataset has not been obtained, so
there is nothing to fit against. They are set so ordinary usage lands inside
the literature band -- SLI batteries last 3-5 years, 2-3 in hot climates -- and
that is a sanity anchor, not a calibration. `AgingRates.fitted` is False by
default and `cell/life.py` propagates it into every absolute figure.

Ratios survive this uncertainty. "Profile A ages the battery twice as fast as
profile B" divides the unfitted constant out; "this battery has 847 days left"
does not.

**Sulfation is a state, not a total.** Crystals grow more slowly as they get
larger and dissolve in proportion to how much is present, so both terms depend
on the current crystal size. An accumulated sum of low-charge hours cannot
express that, which is why this module carries state across trips.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: End of life. The usual definition: 80% of nominal capacity.
EOL_SOH = 0.8

#: Below this state of charge, sulfation accumulates.
LOW_SOC = 0.9

#: And above this, a full charge dissolves it again.
FULL_SOC = 0.98

#: Capacity loss at end of life, i.e. 1.0 - EOL_SOH.
EOL_LOSS = 1.0 - EOL_SOH


@dataclass(frozen=True)
class Damage:
    """What one trip, or one soak, did to the battery."""

    duration_s: float
    corrosion_equivalent_h: float  # Arrhenius-weighted hours at 25 C
    ah_throughput: float
    low_soc_hours: float
    full_charge_hours: float
    vibration_dose: float

    @classmethod
    def zero(cls) -> "Damage":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __add__(self, other: "Damage") -> "Damage":
        return Damage(
            duration_s=self.duration_s + other.duration_s,
            corrosion_equivalent_h=(
                self.corrosion_equivalent_h + other.corrosion_equivalent_h
            ),
            ah_throughput=self.ah_throughput + other.ah_throughput,
            low_soc_hours=self.low_soc_hours + other.low_soc_hours,
            full_charge_hours=self.full_charge_hours + other.full_charge_hours,
            vibration_dose=self.vibration_dose + other.vibration_dose,
        )

    def scaled(self, factor: float) -> "Damage":
        return Damage(
            duration_s=self.duration_s * factor,
            corrosion_equivalent_h=self.corrosion_equivalent_h * factor,
            ah_throughput=self.ah_throughput * factor,
            low_soc_hours=self.low_soc_hours * factor,
            full_charge_hours=self.full_charge_hours * factor,
            vibration_dose=self.vibration_dose * factor,
        )


@dataclass(frozen=True)
class AgingRates:
    """The unfitted constants. `fitted` is the honesty gate, not a hint.

    `corrosion_eol_h` is the Arrhenius-weighted exposure at which corrosion
    **alone** reaches end of life, and it means exactly that: a corrosion
    fraction of 1.0 puts health at `EOL_SOH` with no help from the other two
    pathways. The sulfation and shedding weights then say how much those
    contribute relative to a full corrosion life.

    The value was set by working backwards from ordinary use. A day of it is
    about 79 Arrhenius-weighted hours -- 1.5 h driving with the bay near 60 C
    is 20.8 of them, an hour of post-shutdown heat soak near 75 C is another
    36.4, and the 21.5 h parked at 25 C are 21.5 more. Note that the heat soak
    contributes more than the driving does. At 79 a day, 115,000 puts end of
    life at 4.0 years, mid-band.

    **A recorded discrepancy, not tuned away.** Because life is inversely
    proportional to weighted exposure, this model makes a 42 C ambient age the
    battery about 3.1 times faster than a 25 C one. That agrees with this
    project's own earlier measurement of ~2.9x, and it is stronger than the
    literature's population-level bands imply (3-5 years temperate against 2-3
    hot, roughly 1.7x). Those bands compare whole populations with many
    confounders and the ratio is not a like-for-like check, so the model is
    left alone and the disagreement is stated.
    """

    corrosion_eol_h: float = 115000.0
    corrosion_exponent: float = 0.6  # Schiffer's sublinear layer growth
    sulfation_weight: float = 0.30  # relative to a full corrosion life
    shedding_weight: float = 0.15
    sulfation_per_low_soc_h: float = 2.0e-4
    sulfation_recovery_per_full_h: float = 5.0e-5
    shedding_per_dose: float = 1.0e-7
    fitted: bool = False


@dataclass
class AgingState:
    corrosion_hours: float = 0.0
    crystal: float = 0.0
    shedding: float = 0.0

    def copy(self) -> "AgingState":
        return replace(self)


def accumulate(state: AgingState, damage: Damage, rates: AgingRates) -> AgingState:
    """Advance the ageing states by one trip's or one soak's damage."""
    corrosion_hours = state.corrosion_hours + damage.corrosion_equivalent_h

    # Growth slows as crystals enlarge; dissolution is proportional to what is
    # there. Both depend on the current size, which is why this is a state.
    growth = (
        rates.sulfation_per_low_soc_h * damage.low_soc_hours * (1.0 - state.crystal)
    )
    recovery = (
        rates.sulfation_recovery_per_full_h
        * damage.full_charge_hours
        * state.crystal
    )
    crystal = max(0.0, min(1.0, state.crystal + growth - recovery))

    shedding = min(
        1.0, state.shedding + rates.shedding_per_dose * damage.vibration_dose
    )
    return AgingState(
        corrosion_hours=corrosion_hours, crystal=crystal, shedding=shedding
    )


def soh(state: AgingState, rates: AgingRates) -> float:
    """State of health: 1.0 new, `EOL_SOH` at end of life."""
    if state.corrosion_hours > 0.0:
        exposure = state.corrosion_hours / rates.corrosion_eol_h
        corrosion_fraction = exposure ** rates.corrosion_exponent
    else:
        corrosion_fraction = 0.0
    loss = EOL_LOSS * (
        corrosion_fraction
        + rates.sulfation_weight * state.crystal
        + rates.shedding_weight * state.shedding
    )
    return max(0.0, 1.0 - loss)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cell_aging.py -q`
Expected: PASS, 11 tests. If `test_the_default_constants_land_inside_the_literature_band` fails, adjust `corrosion_eol_h` until end of life falls between 3 and 6 years — and change nothing else, because that test is the only anchor these constants have.

- [ ] **Step 5: Commit**

```bash
git add cell/aging.py tests/test_cell_aging.py
git commit -m "Age the battery three ways, and admit the rates are unfitted"
```

---

### Task 10: The within-trip integration

Where it all comes together at 1 Hz. Health is held fixed for the duration of a trip — over 1,930 s it moves by about 1e-7 — and the trip emits a damage vector for `cell/life.py` to work in.

**Files:**
- Create: `cell/integrate.py`
- Test: `tests/test_cell_integrate.py`

**Interfaces:**
- Consumes: `load.spec.BatteryScenario`, `load.thermal.BayTemperature`, `load.electrical`, `cell.ecm`, `cell.aging`.
- Produces:
  - `Step` — frozen dataclass `(t_s, i_bat_a, v_bat_v, t_bat_c, t_bay_c, soc, r_int_ohm)`
  - `CellState` — mutable dataclass `(soc=1.0, temp_c=25.0, aging=AgingState())`
  - `TripResult` — frozen dataclass `(steps: tuple[Step, ...], damage: Damage, state: CellState)`
  - `run_trip(samples, scenario, state, rates, cranked=False, dt_s=1.0) -> TripResult`
  - `run_soak(seconds, scenario, state, rates, dt_s=60.0) -> TripResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cell_integrate.py
import pytest

from cell.aging import AgingRates, soh
from cell.integrate import CellState, run_soak, run_trip
from load.spec import BatteryScenario

RATES = AgingRates()
CRUISE = BatteryScenario(name="cruise", ambient_c=25.0)
IDLE_HOT = BatteryScenario(name="idle-hot", ambient_c=42.0, hvac=1.0, lights=True)


def _drive(seconds, speed=20.0, rpm=2500.0, coolant=90.0, running=1.0):
    for t in range(seconds):
        yield {
            "t_s": float(t), "speed_mps": speed, "rpm": rpm,
            "coolant_c": coolant, "engine_load": 0.3, "engine_running": running,
            "ax_mps2": 0.0, "ay_mps2": 0.0, "az_mps2": 9.81,
        }


def test_cruising_recharges_the_battery():
    state = CellState(soc=0.95)
    result = run_trip(_drive(600), CRUISE, state, RATES)
    assert result.state.soc > 0.95


def test_state_of_charge_is_bounded():
    state = CellState(soc=0.999)
    result = run_trip(_drive(3600), CRUISE, state, RATES)
    assert 0.0 <= result.state.soc <= 1.0


def test_idling_hot_with_everything_on_discharges_the_battery():
    # The recharge-deficit pathway. It exists only because the alternator is
    # rpm-limited and acceptance is SoC-limited.
    state = CellState(soc=1.0)
    result = run_trip(_drive(900, speed=0.0, rpm=800.0), IDLE_HOT, state, RATES)
    assert result.state.soc < 1.0
    assert any(step.i_bat_a > 0.0 for step in result.steps)


def test_a_crank_is_the_largest_current_in_the_trip():
    state = CellState(soc=1.0)
    result = run_trip(_drive(60), CRUISE, state, RATES, cranked=True)
    assert max(step.i_bat_a for step in result.steps) == pytest.approx(
        CRUISE.crank_a
    )
    assert result.steps[0].i_bat_a == pytest.approx(CRUISE.crank_a)


def test_a_crank_sags_the_terminal_voltage():
    state = CellState(soc=1.0)
    result = run_trip(_drive(60), CRUISE, state, RATES, cranked=True)
    assert result.steps[0].v_bat_v < 11.5


def test_a_hot_ambient_ages_the_battery_faster_than_a_cool_one():
    # The headline claim the whole project rests on, at its smallest scale.
    cool = run_trip(_drive(1800), BatteryScenario(name="c", ambient_c=25.0),
                    CellState(), RATES)
    hot = run_trip(_drive(1800), BatteryScenario(name="h", ambient_c=42.0),
                   CellState(), RATES)
    assert hot.damage.corrosion_equivalent_h > cool.damage.corrosion_equivalent_h


def test_the_battery_never_overshoots_the_bay_it_sits_in():
    # At the currents an SLI battery sees, self-heating is watts and
    # h (T - T_bay) is the whole thermal model, so the battery only ever
    # chases the bay.
    result = run_trip(_drive(3600), CRUISE, CellState(temp_c=25.0), RATES)
    for step in result.steps:
        assert step.t_bat_c <= step.t_bay_c + 0.5


def test_the_battery_lags_the_bay_by_hours():
    # C_th / h is 15000 / 2 = 7500 s, so half an hour into a drive the battery
    # is still well behind the bay. This is why a short trip heats the bay
    # without much heating the battery, and why soak time matters so much.
    result = run_trip(_drive(1800), CRUISE, CellState(temp_c=25.0), RATES)
    last = result.steps[-1]
    assert last.t_bat_c < last.t_bay_c - 5.0


def test_health_does_not_move_within_a_single_trip():
    # Health is held fixed for the trip, so resistance moves only with
    # temperature and state of charge -- not with age.
    state = CellState()
    result = run_trip(_drive(1800), CRUISE, state, RATES)
    assert result.steps[0].r_int_ohm == pytest.approx(
        result.steps[-1].r_int_ohm, rel=0.05
    )


def test_the_trip_still_records_the_damage_it_did():
    state = CellState()
    before = soh(state.aging, RATES)
    result = run_trip(_drive(1800), CRUISE, state, RATES)
    assert result.damage.corrosion_equivalent_h > 0.0
    assert soh(result.state.aging, RATES) < before


def test_a_soak_draws_only_the_parasitic_load():
    state = CellState(soc=1.0, temp_c=80.0)
    result = run_soak(8 * 3600, CRUISE, state, RATES)
    assert all(
        step.i_bat_a == pytest.approx(CRUISE.parasitic_a) for step in result.steps
    )
    assert result.state.soc < 1.0


def test_a_soak_still_corrodes_because_the_bay_is_still_hot():
    state = CellState(soc=1.0, temp_c=90.0)
    result = run_soak(3600, CRUISE, state, RATES)
    assert result.damage.corrosion_equivalent_h > 0.0


def test_every_step_is_reported_once_per_interval():
    result = run_trip(_drive(600), CRUISE, CellState(), RATES, dt_s=1.0)
    assert len(result.steps) == 600
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cell_integrate.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cell.integrate'`

- [ ] **Step 3: Write the implementation**

```python
# cell/integrate.py
"""The 1 Hz within-trip loop, and the damage vector it hands to `cell/life.py`.

Health is held fixed for the length of a trip. Over the recorded 1,930 s it
moves by about one part in ten million, so integrating it here would spend
every cycle recomputing a constant -- and reaching end of life that way means
10^8 steps, which is not happening in stdlib Python. `cell/life.py` advances it
instead, in units of trips.

The battery's own temperature barely leaves the bay temperature: at 20 A
through 6 mOhm self-heating is 2.4 W, and even a 350 A crank is 735 W for 1.5 s
into a ~15 kJ/K mass, which is 0.07 K. The `I^2 R` term is kept because it
costs nothing and becomes visible as resistance rises with age, but
`h (T - T_bay)` is effectively the whole thermal model. Get the bay wrong and
nothing else here matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from battery.corrosion import corrosion_rate
from cell.aging import FULL_SOC, LOW_SOC, AgingRates, AgingState, Damage, accumulate, soh
from cell.ecm import open_circuit_v, r_int_ohm, terminal_v
from load.electrical import alternator_capability_a, battery_current_a
from load.spec import BatteryScenario
from load.thermal import COOLDOWN_TAU_S, BayTemperature

GRAVITY_MPS2 = 9.81

#: Coolant temperature a soak starts from when the caller does not say.
OPERATING_TEMP_C = 90.0


@dataclass(frozen=True)
class Step:
    t_s: float
    i_bat_a: float
    v_bat_v: float
    t_bat_c: float
    t_bay_c: float
    soc: float
    r_int_ohm: float


@dataclass
class CellState:
    soc: float = 1.0
    temp_c: float = 25.0
    aging: AgingState = field(default_factory=AgingState)


@dataclass(frozen=True)
class TripResult:
    steps: tuple[Step, ...]
    damage: Damage
    state: CellState
    cranked: bool = False


def _advance(
    state: CellState,
    scenario: BatteryScenario,
    t_s: float,
    rpm: float,
    engine_running: float,
    bay_c: float,
    cranking: bool,
    vibration: float,
    dt_s: float,
    health: float,
) -> tuple[Step, Damage]:
    capacity_ah = scenario.capacity_ah * health
    resistance = r_int_ohm(scenario.r0_ohm, health, state.temp_c, state.soc)
    current = battery_current_a(
        rpm=rpm, soc=state.soc, temp_c=state.temp_c, scenario=scenario,
        engine_running=engine_running, cranking=cranking,
    )
    capability = (
        alternator_capability_a(rpm, scenario.alternator_rated_a)
        if engine_running >= 0.5 else 0.0
    )
    voltage = terminal_v(state.soc, current, resistance, capability)

    # Positive current is discharge, so state of charge falls with it.
    state.soc = max(0.0, min(1.0, state.soc - current * dt_s / 3600.0 / capacity_ah))

    # Self-heating in, bay coupling out. The second term dominates completely.
    heating = current * current * resistance
    cooling = scenario.h_w_per_k * (state.temp_c - bay_c)
    state.temp_c += (heating - cooling) * dt_s / scenario.c_th_j_per_k

    hours = dt_s / 3600.0
    damage = Damage(
        duration_s=dt_s,
        corrosion_equivalent_h=corrosion_rate(state.temp_c) * hours,
        ah_throughput=abs(current) * hours,
        low_soc_hours=hours if state.soc < LOW_SOC else 0.0,
        full_charge_hours=hours if state.soc > FULL_SOC else 0.0,
        vibration_dose=vibration * dt_s,
    )
    step = Step(
        t_s=t_s, i_bat_a=current, v_bat_v=voltage, t_bat_c=state.temp_c,
        t_bay_c=bay_c, soc=state.soc, r_int_ohm=resistance,
    )
    return step, damage


def run_trip(
    samples: Iterable,
    scenario: BatteryScenario,
    state: CellState,
    rates: AgingRates,
    cranked: bool = False,
    dt_s: float = 1.0,
) -> TripResult:
    """Integrate one trip. `samples` should already be downsampled to `dt_s`."""
    bay = BayTemperature(ambient_c=scenario.ambient_c, initial_c=state.temp_c)
    health = soh(state.aging, rates)
    steps: list[Step] = []
    total = Damage.zero()
    elapsed = 0.0

    for sample in samples:
        bay_c = bay.step(sample, dt_s)
        # Vibration proxy: how far total acceleration departs from gravity.
        magnitude = (
            float(sample["ax_mps2"]) ** 2
            + float(sample["ay_mps2"]) ** 2
            + float(sample["az_mps2"]) ** 2
        ) ** 0.5
        step, damage = _advance(
            state=state, scenario=scenario,
            t_s=float(sample["t_s"]), rpm=float(sample["rpm"]),
            engine_running=float(sample["engine_running"]), bay_c=bay_c,
            cranking=cranked and elapsed < scenario.crank_s,
            vibration=abs(magnitude - GRAVITY_MPS2), dt_s=dt_s, health=health,
        )
        steps.append(step)
        total = total + damage
        elapsed += dt_s

    state.aging = accumulate(state.aging, total, rates)
    return TripResult(
        steps=tuple(steps), damage=total, state=state, cranked=cranked
    )


def run_soak(
    seconds: float,
    scenario: BatteryScenario,
    state: CellState,
    rates: AgingRates,
    initial_coolant_c: float | None = None,
    dt_s: float = 60.0,
) -> TripResult:
    """Integrate a park: engine off, parasitic draw, and a bay that stays hot.

    The soak is not idle time for the battery. Immediately after shutdown the
    bay is hotter than it was while driving, and corrosion is exponential in
    temperature, so a good part of a day's damage happens in a car park.
    """
    health = soh(state.aging, rates)
    bay = BayTemperature(ambient_c=scenario.ambient_c, initial_c=state.temp_c)
    steps: list[Step] = []
    total = Damage.zero()
    elapsed = 0.0

    # Coolant decays on its own time constant, which is nothing like the
    # battery's. Nothing measures it during a soak, so it is modelled here and
    # labelled as modelled.
    coolant = (
        OPERATING_TEMP_C if initial_coolant_c is None else initial_coolant_c
    )

    while elapsed < seconds:
        step_s = min(dt_s, seconds - elapsed)
        coolant += (scenario.ambient_c - coolant) * min(1.0, step_s / COOLDOWN_TAU_S)
        bay_c = bay.step(
            {"coolant_c": coolant, "speed_mps": 0.0, "engine_load": 0.0,
             "engine_running": 0.0},
            step_s,
        )
        step, damage = _advance(
            state=state, scenario=scenario, t_s=elapsed, rpm=0.0,
            engine_running=0.0, bay_c=bay_c, cranking=False, vibration=0.0,
            dt_s=step_s, health=health,
        )
        steps.append(step)
        total = total + damage
        elapsed += step_s

    state.aging = accumulate(state.aging, total, rates)
    return TripResult(steps=tuple(steps), damage=total, state=state)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cell_integrate.py -q`
Expected: PASS, 11 tests

- [ ] **Step 5: Commit**

```bash
git add cell/integrate.py tests/test_cell_integrate.py
git commit -m "Integrate one trip, and hand the damage on in trip-sized units"
```

---
### Task 11: From one trip to end of life

Advancing health over a declared trip schedule, and the gate that stops an unfitted rate constant being reported as a number of days.

**Files:**
- Create: `cell/life.py`
- Test: `tests/test_cell_life.py`

**Interfaces:**
- Consumes: `cell.aging`.
- Produces:
  - `TripSchedule` — frozen dataclass `(trips_per_day=2.0, soak_s=28800.0, horizon_days=3650.0)`
  - `UnfittedScale(RuntimeError)`
  - `LifeEstimate` — frozen dataclass with `.headline_days()` and `.ratio_to(other)`
  - `project(day_damage, rates, schedule, resim_threshold=0.01) -> LifeEstimate`
  - `ensemble(day_damage, rates, schedule, draws=32, seed=0, spread=0.35) -> LifeEstimate`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cell_life.py
import dataclasses

import pytest

from cell.aging import AgingRates, Damage
from cell.life import TripSchedule, UnfittedScale, ensemble, project

RATES = AgingRates()
SCHEDULE = TripSchedule()


def _daily(corrosion_h, low_soc_h=0.0):
    """A day's damage that does not depend on health."""
    def day_damage(health):
        return dataclasses.replace(
            Damage.zero(),
            duration_s=86400.0,
            corrosion_equivalent_h=corrosion_h,
            low_soc_hours=low_soc_h,
            full_charge_hours=max(0.0, 20.0 - low_soc_h),
        )
    return day_damage


#: A day of ordinary use in Arrhenius-weighted hours; see cell/aging.py.
ORDINARY_DAY_H = 78.7


def test_ordinary_use_reaches_end_of_life_inside_the_literature_band():
    # SLI batteries last 3-5 years. A model saying 12 is broken.
    estimate = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    assert estimate.eol_days is not None
    assert 3 * 365 <= estimate.eol_days <= 5 * 365


def test_a_hotter_bay_shortens_life():
    cool = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    hot = project(_daily(ORDINARY_DAY_H * 3.09), RATES, SCHEDULE)
    assert hot.eol_days < cool.eol_days


def test_short_trips_shorten_life_through_sulfation():
    healthy = project(_daily(ORDINARY_DAY_H, low_soc_h=0.0), RATES, SCHEDULE)
    sulfating = project(_daily(ORDINARY_DAY_H, low_soc_h=18.0), RATES, SCHEDULE)
    assert sulfating.eol_days < healthy.eol_days


def test_a_battery_that_survives_the_horizon_reports_no_end_of_life():
    estimate = project(_daily(0.001), RATES, dataclasses.replace(
        SCHEDULE, horizon_days=365.0))
    assert estimate.eol_days is None


def test_the_health_curve_starts_at_one_and_falls():
    estimate = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    days, healths = zip(*estimate.soh_curve)
    assert healths[0] == pytest.approx(1.0, abs=0.01)
    assert healths[-1] < healths[0]
    assert list(days) == sorted(days)


def test_an_unfitted_estimate_refuses_to_give_a_headline_figure():
    # No battery in this project has reached end of life, so there is nothing
    # to fit the absolute scale against.
    estimate = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    assert estimate.scale_unfitted is True
    with pytest.raises(UnfittedScale, match="fitted"):
        estimate.headline_days()


def test_a_fitted_estimate_gives_one():
    fitted = dataclasses.replace(RATES, fitted=True)
    estimate = project(_daily(ORDINARY_DAY_H), fitted, SCHEDULE)
    assert estimate.scale_unfitted is False
    assert estimate.headline_days() > 0.0


def test_ratios_are_allowed_even_when_the_scale_is_unfitted():
    # The unfitted constant cancels, which is why the relative claims carry
    # much less validation burden than the absolute one.
    cool = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    hot = project(_daily(ORDINARY_DAY_H * 3.09), RATES, SCHEDULE)
    assert cool.ratio_to(hot) > 2.5


def test_the_ensemble_brackets_the_point_estimate():
    point = project(_daily(ORDINARY_DAY_H), RATES, SCHEDULE)
    spread = ensemble(_daily(ORDINARY_DAY_H), RATES, SCHEDULE, draws=16, seed=7)
    low, high = spread.interval_days
    assert low < point.eol_days < high


def test_the_ensemble_interval_is_parameter_uncertainty_not_a_world_model():
    spread = ensemble(_daily(ORDINARY_DAY_H), RATES, SCHEDULE, draws=16, seed=7)
    assert "parameter" in spread.uncertainty_source
    assert "latent" not in spread.uncertainty_source


def test_the_ensemble_is_reproducible_from_its_seed():
    a = ensemble(_daily(ORDINARY_DAY_H), RATES, SCHEDULE, draws=16, seed=3)
    b = ensemble(_daily(ORDINARY_DAY_H), RATES, SCHEDULE, draws=16, seed=3)
    assert a.interval_days == b.interval_days


def test_damage_is_recomputed_when_health_has_drifted():
    calls = []

    def day_damage(health):
        calls.append(health)
        return dataclasses.replace(
            Damage.zero(), duration_s=86400.0, corrosion_equivalent_h=78.7
        )

    project(day_damage, RATES, SCHEDULE, resim_threshold=0.02)
    # Resistance rises as health falls, which changes current and temperature,
    # so the per-trip damage cannot be computed once and reused forever.
    assert len(calls) > 1
    assert calls == sorted(calls, reverse=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cell_life.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cell.life'`

- [ ] **Step 3: Write the implementation**

```python
# cell/life.py
"""Advancing health over a trip schedule, and the gate on absolute figures.

A battery life is about 10^8 seconds. Integrating the cell at 1 Hz to reach it
is not possible here, and would be pointless anyway: over one trip health moves
by about one part in ten million. So `cell/integrate.py` produces damage in
trip-sized units and this module advances the ageing states in days.

The per-day damage is recomputed whenever health has drifted, because it is not
constant: resistance rises as the battery ages, which changes both the current
it draws and the heat it makes, which changes the damage. Recomputing once per
drift threshold costs a few hundred evaluations over a full life instead of
10^8.

**The scale gate.** Physics gives the fade curve its shape; the rate constant
that turns shape into days has to come from cells that actually reached end of
life, and there are none. `headline_days` therefore raises unless the rates are
flagged fitted. Ratios are always available -- the unfitted constant divides
out -- and they are usually the more useful claim anyway.

The ensemble interval represents **parameter uncertainty**: how much the answer
moves when the unfitted constant is varied over a plausible range. It is not
the stochastic-latent distribution the world model will eventually supply, and
`uncertainty_source` says so in the sidecar so the two are never conflated.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Callable

from cell.aging import EOL_SOH, AgingRates, AgingState, Damage, accumulate, soh

#: How often a point is recorded on the health curve.
CURVE_INTERVAL_DAYS = 30

DayDamage = Callable[[float], Damage]


class UnfittedScale(RuntimeError):
    """An absolute figure was asked for from an unfitted rate constant."""


@dataclass(frozen=True)
class TripSchedule:
    trips_per_day: float = 2.0
    soak_s: float = 28800.0
    horizon_days: float = 3650.0

    def __post_init__(self) -> None:
        if self.trips_per_day <= 0.0:
            raise ValueError(f"trips_per_day must be positive, got {self.trips_per_day}")
        if self.horizon_days <= 0.0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")


@dataclass(frozen=True)
class LifeEstimate:
    eol_days: float | None
    soh_curve: tuple[tuple[float, float], ...]
    scale_unfitted: bool
    interval_days: tuple[float, float] | None = None
    uncertainty_source: str = ""

    def headline_days(self) -> float:
        """The conservative figure a fleet operator would act on.

        Refuses to answer while the rate constant is unfitted, because the
        number would look like a measurement and is not one.
        """
        if self.scale_unfitted:
            raise UnfittedScale(
                "the ageing rate constant is not fitted against full-life data, "
                "so an absolute remaining life cannot be reported; use ratio_to "
                "for relative claims, which do not depend on it"
            )
        if self.eol_days is None:
            raise UnfittedScale("end of life was not reached within the horizon")
        if self.interval_days is not None:
            return self.interval_days[0]  # report the low quantile, not the mean
        return self.eol_days

    def ratio_to(self, other: "LifeEstimate") -> float:
        """How many times longer this battery lasts than `other`.

        Valid with an unfitted scale: the constant cancels.
        """
        if self.eol_days is None or other.eol_days is None:
            raise ValueError("both estimates must reach end of life to be compared")
        return self.eol_days / other.eol_days


def project(
    day_damage: DayDamage,
    rates: AgingRates,
    schedule: TripSchedule,
    resim_threshold: float = 0.01,
) -> LifeEstimate:
    """Advance health day by day until end of life or the horizon."""
    state = AgingState()
    health = soh(state, rates)
    damage = day_damage(health)
    curve: list[tuple[float, float]] = [(0.0, health)]
    eol_days: float | None = None

    day = 0
    while day < schedule.horizon_days:
        state = accumulate(state, damage, rates)
        day += 1
        current = soh(state, rates)
        if day % CURVE_INTERVAL_DAYS == 0:
            curve.append((float(day), current))
        if current <= EOL_SOH:
            eol_days = float(day)
            curve.append((float(day), current))
            break
        if health - current >= resim_threshold:
            health = current
            damage = day_damage(health)

    return LifeEstimate(
        eol_days=eol_days,
        soh_curve=tuple(curve),
        scale_unfitted=not rates.fitted,
    )


def ensemble(
    day_damage: DayDamage,
    rates: AgingRates,
    schedule: TripSchedule,
    draws: int = 32,
    seed: int = 0,
    spread: float = 0.35,
) -> LifeEstimate:
    """Vary the unfitted constant over a plausible range and report quantiles.

    This is the interval the output contract requires. It is parameter
    uncertainty, and it must never be presented as the world model's stochastic
    latent -- a different quantity answering a different question.
    """
    rng = random.Random(seed)
    lives: list[float] = []
    point = project(day_damage, rates, schedule)

    for _ in range(draws):
        factor = 1.0 + rng.uniform(-spread, spread)
        drawn = dataclasses.replace(
            rates, corrosion_eol_h=rates.corrosion_eol_h * factor
        )
        result = project(day_damage, drawn, schedule)
        if result.eol_days is not None:
            lives.append(result.eol_days)

    interval = None
    if len(lives) >= 2:
        lives.sort()
        low = lives[max(0, int(0.05 * (len(lives) - 1)))]
        high = lives[min(len(lives) - 1, int(0.95 * (len(lives) - 1) + 0.999))]
        interval = (low, high)

    return LifeEstimate(
        eol_days=point.eol_days,
        soh_curve=point.soh_curve,
        scale_unfitted=not rates.fitted,
        interval_days=interval,
        uncertainty_source=(
            "parameter uncertainty over the unfitted corrosion rate constant; "
            "not a world-model stochastic latent"
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cell_life.py -q`
Expected: PASS, 12 tests

- [ ] **Step 5: Commit**

```bash
git add cell/life.py tests/test_cell_life.py
git commit -m "Project a life, and refuse to name a date we cannot support"
```

---

### Task 12: The leakage detector, tested on the case that caught us

The parent spec requires the generator to report any column pair whose relationship it can recover analytically. The load-bearing test is on the **detector**, not on our data: it is fed the real dataset's known identity and must catch it. A detector nobody has seen fire is not evidence of anything.

The real dataset's identity, verified exact on every sampled row, is:

    ah_consumed   = current * dt_hours
    total_ah_used = cumsum(ah_consumed)
    True_SoC      = 100 (1 - total_ah_used / 16.069411)

That is why its reported ~99.7% accuracy is label leakage: the model recovers a division.

**Files:**
- Create: `cell/leakage.py`
- Test: `tests/test_cell_leakage.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `EXACT_TOLERANCE = 1e-6`
  - `Finding` — frozen dataclass `(target, features, relation, max_residual)`
  - `linear_fit(rows, target, features) -> tuple[tuple[float, ...], float] | None`
  - `find_linear(rows, target, features, tol) -> list[Finding]`
  - `find_cumulative(rows, target, feature, dt_column, tol) -> Finding | None`
  - `report(rows, targets, features, dt_column=None, tol=EXACT_TOLERANCE) -> tuple[Finding, ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cell_leakage.py
import pytest

from cell.leakage import find_cumulative, find_linear, linear_fit, report

CAPACITY_AH = 16.069411


def _bulb_dataset(n=200):
    """The real dataset's shape: a constant load and a closed-form target.

    Voltage carries a little noise, as the measured channel does. Without it
    voltage is itself an exact function of the target and the detector fires on
    it too -- passing the test for a reason the real data does not share.
    """
    import random
    rng = random.Random(11)
    rows = []
    total = 0.0
    for _ in range(n):
        current, dt_hours = 3.0, 1.0 / 3600.0
        ah = current * dt_hours
        total += ah
        rows.append({
            "current": current,
            "voltage": 12.6 - 0.01 * total + rng.gauss(0.0, 0.002),
            "dt_hours": dt_hours,
            "ah_consumed": ah,
            "total_ah_used": total,
            "True_SoC": 100.0 * (1.0 - total / CAPACITY_AH),
        })
    return rows


def test_it_catches_the_identity_that_caught_us():
    # True_SoC = 100 - (100 / 16.069411) * total_ah_used, exact on every row.
    # This test is the whole point of the module.
    findings = report(
        _bulb_dataset(),
        targets=("True_SoC",),
        features=("current", "voltage", "dt_hours", "total_ah_used"),
    )
    assert any(
        f.target == "True_SoC" and "total_ah_used" in f.features for f in findings
    ), "the detector failed to find the known leak"
    # And it refuses the singular fits: current and dt_hours are constant in
    # this dataset, so nothing may be claimed about them.
    assert all(f.features != ("current",) for f in findings)
    assert all(f.features != ("dt_hours",) for f in findings)


def test_it_catches_the_cumulative_sum_identity():
    finding = find_cumulative(
        _bulb_dataset(), target="total_ah_used", feature="current",
        dt_column="dt_hours", tol=1e-6,
    )
    assert finding is not None
    assert finding.relation.startswith("cumsum")


def test_it_catches_the_product_identity():
    findings = find_linear(
        _bulb_dataset(), target="ah_consumed", features=("current", "dt_hours"),
        tol=1e-6,
    )
    assert findings


def test_a_genuinely_noisy_relationship_is_not_reported():
    import random
    rng = random.Random(0)
    rows = [
        {"x": rng.uniform(0, 10), "y": rng.uniform(0, 10), "z": rng.uniform(0, 10)}
        for _ in range(200)
    ]
    assert report(rows, targets=("z",), features=("x", "y")) == ()


def test_a_constant_column_does_not_produce_a_spurious_finding():
    # dt_hours is constant in the real dataset. A degenerate fit must be
    # refused, not reported as a discovery.
    rows = [{"k": 1.0, "y": float(i)} for i in range(50)]
    assert linear_fit(rows, target="y", features=("k",)) is None


def test_a_finding_records_how_exact_it_is():
    findings = report(
        _bulb_dataset(), targets=("True_SoC",), features=("total_ah_used",)
    )
    assert findings[0].max_residual < 1e-6


def test_too_few_rows_to_fit_is_not_a_finding():
    rows = [{"x": 1.0, "y": 2.0}]
    assert report(rows, targets=("y",), features=("x",)) == ()


def test_an_empty_dataset_reports_nothing():
    assert report([], targets=("y",), features=("x",)) == ()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cell_leakage.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cell.leakage'`

- [ ] **Step 3: Write the implementation**

```python
# cell/leakage.py
"""Finding targets that are closed-form functions of the features beside them.

This module exists because of a specific failure. The real battery dataset's
target satisfies, exactly on every row:

    ah_consumed   = current * dt_hours
    total_ah_used = cumsum(ah_consumed)
    True_SoC      = 100 (1 - total_ah_used / 16.069411)

The capacity constant is stable to six decimals. A model reported at ~99.7%
accuracy on that target had learned a division, and no amount of extra data
would have changed the number or demonstrated anything. The synthetic dataset
must not repeat it, and asserting that it does not requires a detector that has
been seen to work -- which is why the load-bearing test here feeds it the
identity above and requires it to fire.

Three relationships are checked: exact linear in one feature, exact linear in
two, and the cumulative-sum identity that produced `total_ah_used`. Degenerate
fits are refused rather than reported: a constant column can be fitted to
anything, and calling that a discovery would bury the real findings in noise.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

#: What counts as exact. These identities hold to floating-point precision, not
#: approximately, so the threshold can be tight.
EXACT_TOLERANCE = 1e-6

#: A fit needs more rows than coefficients, with room to spare.
MIN_ROWS = 8


@dataclass(frozen=True)
class Finding:
    target: str
    features: tuple[str, ...]
    relation: str
    max_residual: float


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting. None if singular."""
    n = len(matrix)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[pivot][col]) < 1e-12:
            return None
        a[col], a[pivot] = a[pivot], a[col]
        for row in range(n):
            if row == col:
                continue
            factor = a[row][col] / a[col][col]
            for k in range(col, n + 1):
                a[row][k] -= factor * a[col][k]
    return [a[i][n] / a[i][i] for i in range(n)]


def linear_fit(
    rows: Sequence[dict], target: str, features: tuple[str, ...]
) -> tuple[tuple[float, ...], float] | None:
    """Least squares `target = sum(c_i f_i) + c_0`, with its worst residual.

    None when there are too few rows or the normal equations are singular --
    which is what a constant feature produces.
    """
    if len(rows) < MIN_ROWS:
        return None
    design = [[float(row[f]) for f in features] + [1.0] for row in rows]
    targets = [float(row[target]) for row in rows]
    width = len(features) + 1

    normal = [[0.0] * width for _ in range(width)]
    rhs = [0.0] * width
    for vector, value in zip(design, targets):
        for i in range(width):
            rhs[i] += vector[i] * value
            for j in range(width):
                normal[i][j] += vector[i] * vector[j]

    coefficients = _solve(normal, rhs)
    if coefficients is None:
        return None

    worst = 0.0
    for vector, value in zip(design, targets):
        predicted = sum(c * v for c, v in zip(coefficients, vector))
        worst = max(worst, abs(predicted - value))
    return tuple(coefficients), worst


def _scale(rows: Sequence[dict], target: str) -> float:
    values = [abs(float(row[target])) for row in rows]
    return max(1.0, max(values)) if values else 1.0


def find_linear(
    rows: Sequence[dict],
    target: str,
    features: tuple[str, ...],
    tol: float = EXACT_TOLERANCE,
) -> list[Finding]:
    """Exact linear relationships in one or two of `features`."""
    findings: list[Finding] = []
    scale = _scale(rows, target)
    candidates = [(f,) for f in features] + list(combinations(features, 2))
    for subset in candidates:
        fit = linear_fit(rows, target, subset)
        if fit is None:
            continue
        coefficients, worst = fit
        if worst / scale > tol:
            continue
        terms = " + ".join(
            f"{c:.10g}*{f}" for c, f in zip(coefficients, subset)
        )
        findings.append(
            Finding(
                target=target,
                features=subset,
                relation=f"{target} = {terms} + {coefficients[-1]:.10g}",
                max_residual=worst,
            )
        )
    return findings


def find_cumulative(
    rows: Sequence[dict],
    target: str,
    feature: str,
    dt_column: str,
    tol: float = EXACT_TOLERANCE,
) -> Finding | None:
    """Is `target` an affine function of `cumsum(feature * dt)`?

    This is the shape that produced `total_ah_used`, and a plain linear fit on
    the instantaneous columns will not see it.
    """
    if len(rows) < MIN_ROWS:
        return None
    running = 0.0
    augmented = []
    for row in rows:
        running += float(row[feature]) * float(row[dt_column])
        augmented.append({"_cumulative": running, target: float(row[target])})

    fit = linear_fit(augmented, target, ("_cumulative",))
    if fit is None:
        return None
    coefficients, worst = fit
    if worst / _scale(rows, target) > tol:
        return None
    return Finding(
        target=target,
        features=(feature, dt_column),
        relation=(
            f"{target} = {coefficients[0]:.10g}*cumsum({feature}*{dt_column})"
            f" + {coefficients[1]:.10g}"
        ),
        max_residual=worst,
    )


def report(
    rows: Sequence[dict],
    targets: tuple[str, ...],
    features: tuple[str, ...],
    dt_column: str | None = None,
    tol: float = EXACT_TOLERANCE,
) -> tuple[Finding, ...]:
    """Every analytically recoverable relationship between targets and features."""
    if not rows:
        return ()
    findings: list[Finding] = []
    for target in targets:
        usable = tuple(f for f in features if f != target)
        findings.extend(find_linear(rows, target, usable, tol))
        if dt_column is not None:
            for feature in usable:
                if feature == dt_column:
                    continue
                found = find_cumulative(rows, target, feature, dt_column, tol)
                if found is not None:
                    findings.append(found)
    return tuple(findings)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cell_leakage.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Commit**

```bash
git add cell/leakage.py tests/test_cell_leakage.py
git commit -m "Detect the leak that made the last model look 99.7% accurate"
```

---
### Task 13: The dataset, in two tables, with a declared feature/target split

Two tables rather than one, because broadcasting a per-trip remaining-life figure across roughly two thousand near-identical 1 Hz rows lets a network recover it from the row index — the same leakage in new clothes.

There is a second, subtler trap this task has to close, and it is in *our* data rather than the old dataset. Within a trip, `soc` is by construction `soc_0 - cumsum(i_bat_a * dt) / Q`. If `i_bat_a` is offered as a feature alongside `soc` as a target, the detector fires on our own output and it is right to. The resolution is not to hide it: **the driving channels are the features and the battery channels are the targets**, declared in the sidecar and machine-readable. That is the actual research question anyway — predict battery state from driving dynamics, not from battery current.

**Files:**
- Create: `cell/dataset.py`
- Test: `tests/test_cell_dataset.py`

**Interfaces:**
- Consumes: `cell.integrate`, `cell.aging`, `cell.leakage`, `cell.life`, `load.spec`, `load.legacy`.
- Produces:
  - `FEATURES: tuple[str, ...]` — driving channels only
  - `TARGETS: tuple[str, ...]` — battery channels only
  - `WITHIN_TRIP_COLUMNS`, `TRIP_DAMAGE_COLUMNS`
  - `write_dataset(out_dir, scenario, trips, life, provenance, dropouts, rates) -> dict` — returns the sidecar it wrote

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cell_dataset.py
import csv
import json

import pytest

import dataclasses

from cell.aging import AgingRates, Damage
from cell.dataset import FEATURES, TARGETS, write_dataset
from cell.integrate import CellState, run_trip
from cell.life import TripSchedule, project
from load.spec import BatteryScenario

RATES = AgingRates()
SCENARIO = BatteryScenario(name="fixture", ambient_c=30.0)


def _drive(seconds, t0=0.0):
    for t in range(seconds):
        yield {
            "t_s": t0 + t, "speed_mps": 18.0, "rpm": 2400.0, "coolant_c": 92.0,
            "engine_load": 0.3, "engine_running": 1.0, "throttle": 0.3,
            "grade_rad": 0.0, "a_long_mps2": 0.0, "mass_kg": 1510.62,
            "ax_mps2": 0.1, "ay_mps2": 0.0, "az_mps2": 9.81,
        }


def _fixture(tmp_path):
    trips = []
    state = CellState()
    for index in range(3):
        samples = list(_drive(120, t0=index * 1000.0))
        result = run_trip(iter(samples), SCENARIO, state, RATES, cranked=index > 0)
        trips.append((samples, result))
    life = project(
        lambda health: dataclasses.replace(
            Damage.zero(), duration_s=86400.0, corrosion_equivalent_h=78.7
        ),
        RATES, TripSchedule(),
    )
    return write_dataset(
        out_dir=tmp_path, scenario=SCENARIO, trips=trips, life=life,
        provenance={"speed_mps": "measured", "mass_kg": "assumed"},
        dropouts=(), rates=RATES,
    )


def test_it_writes_two_tables_and_a_sidecar(tmp_path):
    _fixture(tmp_path)
    assert (tmp_path / "within_trip.csv").exists()
    assert (tmp_path / "trip_damage.csv").exists()
    assert (tmp_path / "dataset.json").exists()


def test_the_within_trip_table_has_one_row_per_step(tmp_path):
    _fixture(tmp_path)
    with (tmp_path / "within_trip.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3 * 120


def test_the_trip_damage_table_has_one_row_per_trip(tmp_path):
    _fixture(tmp_path)
    with (tmp_path / "trip_damage.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3


def test_remaining_life_appears_only_on_the_trip_table(tmp_path):
    # Broadcasting it across 1 Hz rows is recoverable from the row index.
    _fixture(tmp_path)
    with (tmp_path / "within_trip.csv").open() as handle:
        within = csv.DictReader(handle).fieldnames
    with (tmp_path / "trip_damage.csv").open() as handle:
        per_trip = csv.DictReader(handle).fieldnames
    assert "rul_days" not in within
    assert "rul_days" in per_trip


def test_battery_current_is_a_target_not_a_feature(tmp_path):
    # soc is cumsum(i_bat_a * dt) / Q by construction, so offering both would
    # make the target analytically recoverable.
    assert "i_bat_a" in TARGETS
    assert "i_bat_a" not in FEATURES
    assert "soc" in TARGETS


def test_the_features_are_driving_channels(tmp_path):
    assert set(FEATURES) <= {
        "t_s", "speed_mps", "rpm", "throttle", "coolant_c", "engine_load",
        "grade_rad", "a_long_mps2", "engine_running", "t_bay_c",
    }


def test_the_sidecar_carries_the_leakage_report(tmp_path):
    sidecar = _fixture(tmp_path)
    assert "leakage" in sidecar
    assert sidecar["leakage"]["findings"] == []
    assert sidecar["leakage"]["features"] == list(FEATURES)


def test_the_sidecar_says_the_scale_is_unfitted(tmp_path):
    sidecar = _fixture(tmp_path)
    assert sidecar["life"]["scale_unfitted"] is True


def test_the_sidecar_carries_provenance_and_dropouts(tmp_path):
    sidecar = _fixture(tmp_path)
    assert sidecar["channels"]["mass_kg"] == "assumed"
    assert sidecar["dropouts"] == []


def test_the_sidecar_is_addressed_by_the_scenario_hash(tmp_path):
    sidecar = _fixture(tmp_path)
    assert sidecar["scenario_hash"] == SCENARIO.scenario_hash
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_cell_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'cell.dataset'`

- [ ] **Step 3: Write the implementation**

```python
# cell/dataset.py
"""Assembling the dataset, in two tables, with the split declared.

**Two tables, not one.** Remaining life is a per-trip quantity. Broadcast onto
two thousand near-identical 1 Hz rows it becomes recoverable from the row
index, which is the label leakage of the original dataset wearing a different
hat. The 1 Hz table trains state-of-charge and voltage models; the per-trip
table trains the remaining-life model.

**The split is declared, not implied.** Within a trip,
`soc = soc_0 - cumsum(i_bat_a * dt) / Q` by construction, so offering current
as a feature beside state of charge as a target hands a network the identity.
It is closed by naming the driving channels as features and the battery
channels as targets, in the sidecar, where downstream code reads it. That is
also the actual research question: predict battery state from driving, not from
battery current.

The leakage report then runs over the declared split rather than over every
column pair, which is what makes it mean something.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from cell.aging import AgingRates, soh
from cell.leakage import report
from cell.life import LifeEstimate
from load.spec import BatteryScenario

#: Driving channels. What the world model rolls out and the networks see.
FEATURES: tuple[str, ...] = (
    "speed_mps", "rpm", "throttle", "coolant_c", "engine_load",
    "grade_rad", "a_long_mps2", "engine_running", "t_bay_c",
)

#: Battery channels. What the networks predict.
TARGETS: tuple[str, ...] = ("i_bat_a", "v_bat_v", "t_bat_c", "soc", "r_int_ohm")

WITHIN_TRIP_COLUMNS: tuple[str, ...] = ("trip", "t_s") + FEATURES + TARGETS

TRIP_DAMAGE_COLUMNS: tuple[str, ...] = (
    "trip", "cranked", "duration_s", "corrosion_equivalent_h", "ah_throughput",
    "low_soc_hours", "full_charge_hours", "vibration_dose",
    "soc_start", "soc_end", "soh", "rul_days",
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _within_trip_rows(trips) -> list[dict]:
    rows: list[dict] = []
    for index, (samples, result) in enumerate(trips):
        for sample, step in zip(samples, result.steps):
            row = {"trip": index, "t_s": step.t_s}
            for column in FEATURES:
                row[column] = (
                    step.t_bay_c if column == "t_bay_c" else float(sample[column])
                )
            row.update({
                "i_bat_a": step.i_bat_a, "v_bat_v": step.v_bat_v,
                "t_bat_c": step.t_bat_c, "soc": step.soc,
                "r_int_ohm": step.r_int_ohm,
            })
            rows.append(row)
    return rows


def _trip_damage_rows(trips, rates: AgingRates, life: LifeEstimate) -> list[dict]:
    rows: list[dict] = []
    for index, (_samples, result) in enumerate(trips):
        damage = result.damage
        rows.append({
            "trip": index,
            "cranked": int(result.cranked),
            "duration_s": damage.duration_s,
            "corrosion_equivalent_h": damage.corrosion_equivalent_h,
            "ah_throughput": damage.ah_throughput,
            "low_soc_hours": damage.low_soc_hours,
            "full_charge_hours": damage.full_charge_hours,
            "vibration_dose": damage.vibration_dose,
            "soc_start": result.steps[0].soc if result.steps else "",
            "soc_end": result.steps[-1].soc if result.steps else "",
            "soh": soh(result.state.aging, rates),
            "rul_days": "" if life.eol_days is None else life.eol_days,
        })
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] for column in columns])


def write_dataset(
    out_dir: Path | str,
    scenario: BatteryScenario,
    trips,
    life: LifeEstimate,
    provenance: dict[str, str],
    dropouts,
    rates: AgingRates,
) -> dict:
    """Write both tables and the sidecar. Returns the sidecar."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    within = _within_trip_rows(trips)
    per_trip = _trip_damage_rows(trips, rates, life)
    _write_csv(out / "within_trip.csv", WITHIN_TRIP_COLUMNS, within)
    _write_csv(out / "trip_damage.csv", TRIP_DAMAGE_COLUMNS, per_trip)

    findings = report(within, targets=TARGETS, features=FEATURES)

    sidecar = {
        "scenario": scenario.to_dict(),
        "scenario_hash": scenario.scenario_hash,
        "git_commit": _git_commit(),
        "tables": {
            "within_trip": {"rows": len(within), "hz": 1.0,
                            "columns": list(WITHIN_TRIP_COLUMNS)},
            "trip_damage": {"rows": len(per_trip),
                            "columns": list(TRIP_DAMAGE_COLUMNS)},
        },
        "channels": dict(provenance),
        "dropouts": [asdict(span) for span in dropouts],
        "leakage": {
            "features": list(FEATURES),
            "targets": list(TARGETS),
            "findings": [asdict(f) for f in findings],
            "note": (
                "soc is cumsum(i_bat_a * dt) / Q by construction, so battery "
                "channels are targets and never features. Training on a target "
                "as a feature reproduces the original dataset's label leakage. "
                "Separately: t_bat_c chases t_bay_c with no other forcing of "
                "consequence, so predicting it from the t_bay_c feature is "
                "close to trivial and must not be reported as a result. The "
                "exact-relationship detector will not flag it, because the "
                "thermal lag makes it approximate rather than analytic."
            ),
        },
        "life": {
            "eol_days": life.eol_days,
            "interval_days": list(life.interval_days) if life.interval_days else None,
            "scale_unfitted": life.scale_unfitted,
            "uncertainty_source": life.uncertainty_source,
            "note": (
                "The ageing rate constant is not fitted against full-life data. "
                "Ratios between scenarios are usable; absolute days are not."
            ),
        },
        "ageing_rates": asdict(rates),
    }
    (out / "dataset.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return sidecar
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_cell_dataset.py -q`
Expected: PASS, 10 tests. If `test_the_sidecar_carries_the_leakage_report` fails with findings present, do **not** loosen the tolerance — read the finding and fix the split it exposed.

- [ ] **Step 5: Commit**

```bash
git add cell/dataset.py tests/test_cell_dataset.py
git commit -m "Split driving from battery so the target is not in the features"
```

---

### Task 14: The entry point, and the end-to-end run

One command from a recorded trajectory to a dataset, plus the end-to-end test over the real 181k-row file.

**Files:**
- Create: `battery_run.py`
- Test: `tests/test_battery_run.py`
- Modify: `README.md` — add the command
- Modify: `CLAUDE.md` — update the Status section

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `build(trajectory_path, scenario, schedule, rates, repeats, out_dir) -> dict`
  - `main(argv=None) -> int`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_battery_run.py
import json

import pytest

from battery_run import build, main
from cell.aging import AgingRates
from cell.life import TripSchedule
from load.spec import BatteryScenario

HEADER = (
    "timestamp,pos_x,pos_y,pos_z,vel_x,vel_y,vel_z,acc_x,acc_y,acc_z,"
    "up_x,up_y,up_z,roll,pitch,yaw,roll_vel,pitch_vel,yaw_vel,"
    "roll_acc,pitch_acc,yaw_acc,game_time,speed,rpm,turbo,engine_temp,fuel,"
    "oil_pressure,oil_temp,throttle,brake,clutch,gear"
)


def _row(t, speed=18.0, rpm=2400.0, engine_temp=92.0):
    return (
        f"{t},283.5,-713.4,148.5,-19.2,-0.1,-2.1,0.5,0.7,9.8,"
        f"-0.11,0.03,0.99,0.03,0.0,1.56,0.01,-0.02,-0.01,"
        f"0.16,-0.04,-0.09,0,{speed},{rpm},1.17,{engine_temp},0.99,"
        f"0.0,81.5,0.3,b'\\x00',0,4"
    )


@pytest.fixture
def trajectory(tmp_path):
    path = tmp_path / "telemetry.csv"
    rows = [_row(i * 0.1) for i in range(1200)]  # 120 s at 10 Hz
    rows += [_row(120.0 + i * 0.1, speed=0.0, rpm=0.0) for i in range(600)]
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_it_produces_a_dataset(trajectory, tmp_path):
    out = tmp_path / "dataset"
    sidecar = build(
        trajectory, BatteryScenario(name="t"), TripSchedule(), AgingRates(),
        repeats=2, out_dir=out,
    )
    assert (out / "within_trip.csv").exists()
    assert sidecar["tables"]["within_trip"]["rows"] > 0


def test_repeating_the_trip_produces_cranks(trajectory, tmp_path):
    # One recorded trip has no crank in it. Repeating it under a schedule is
    # what gives the sulfation pathway anything to work with.
    sidecar = build(
        trajectory, BatteryScenario(name="t"), TripSchedule(), AgingRates(),
        repeats=3, out_dir=tmp_path / "d",
    )
    assert sidecar["tables"]["trip_damage"]["rows"] == 3


def test_the_dataset_has_no_leakage_findings(trajectory, tmp_path):
    sidecar = build(
        trajectory, BatteryScenario(name="t"), TripSchedule(), AgingRates(),
        repeats=2, out_dir=tmp_path / "d",
    )
    assert sidecar["leakage"]["findings"] == []


def test_physical_bounds_hold(trajectory, tmp_path):
    import csv
    out = tmp_path / "d"
    build(trajectory, BatteryScenario(name="t", ambient_c=25.0), TripSchedule(),
          AgingRates(), repeats=2, out_dir=out)
    with (out / "within_trip.csv").open() as handle:
        for row in csv.DictReader(handle):
            assert 0.0 <= float(row["soc"]) <= 1.0
            assert 9.0 < float(row["v_bat_v"]) < 15.0
            assert float(row["t_bay_c"]) >= 25.0 - 1e-6


def test_a_hot_ambient_ages_it_faster_than_a_cool_one(trajectory, tmp_path):
    cool = build(trajectory, BatteryScenario(name="c", ambient_c=25.0),
                 TripSchedule(), AgingRates(), 2, tmp_path / "cool")
    hot = build(trajectory, BatteryScenario(name="h", ambient_c=42.0),
                TripSchedule(), AgingRates(), 2, tmp_path / "hot")
    assert hot["life"]["eol_days"] < cool["life"]["eol_days"]


def test_the_cli_runs(trajectory, tmp_path, capsys):
    code = main([str(trajectory), "--name", "cli", "--out", str(tmp_path / "d"),
                 "--repeats", "2"])
    assert code == 0
    assert (tmp_path / "d" / "dataset.json").exists()


def test_the_cli_reports_that_the_scale_is_unfitted(trajectory, tmp_path, capsys):
    main([str(trajectory), "--name", "cli", "--out", str(tmp_path / "d")])
    assert "unfitted" in capsys.readouterr().out.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_battery_run.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'battery_run'`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Recorded driving in, battery dataset out.

    python3 battery_run.py runs/telemetry.csv --name baseline --out runs/dataset

One recorded trip contains one crank at most, and the recorded file contains
none at all -- it begins with the engine already running. Repeating the trip
under a declared schedule is what gives the sulfation and parasitic pathways
any data, and every repeat's soak duration is a scenario input recorded in the
sidecar, never a measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cell.aging import AgingRates, Damage
from cell.dataset import write_dataset
from cell.integrate import CellState, run_soak, run_trip
from cell.life import TripSchedule, ensemble
from load.legacy import LEGACY_PROVENANCE, read_legacy
from load.spec import BatteryScenario
from load.trips import segment


def build(
    trajectory_path: Path | str,
    scenario: BatteryScenario,
    schedule: TripSchedule,
    rates: AgingRates,
    repeats: int,
    out_dir: Path | str,
) -> dict:
    """Integrate the trajectory `repeats` times and write the dataset."""
    trajectory = read_legacy(trajectory_path)

    # Downsampled to 1 Hz once and held: the cell integrates at 1 Hz, and the
    # recorded file is 94 Hz, so this is about 1,780 samples rather than 167k.
    samples = list(trajectory.at_hz(1.0))
    dropouts = trajectory.dropouts.spans
    segments = segment(iter(samples))
    driving = [
        s for s in samples
        if any(
            seg.kind == "trip" and seg.start_s <= s["t_s"] <= seg.end_s
            for seg in segments
        )
    ]

    state = CellState(temp_c=scenario.ambient_c)
    trips = []
    for index in range(repeats):
        result = run_trip(
            iter(driving), scenario, state, rates, cranked=index > 0
        )
        trips.append((driving, result))
        if index < repeats - 1:
            run_soak(
                schedule.soak_s, scenario, state, rates,
                initial_coolant_c=float(driving[-1]["coolant_c"]),
            )

    # A day's damage at a given health: the trip damage scaled by trips per
    # day, plus the soaks between them. Recomputed as resistance rises.
    def day_damage(health: float) -> Damage:
        probe = CellState(
            soc=state.soc, temp_c=scenario.ambient_c, aging=state.aging.copy()
        )
        trip = run_trip(iter(driving), scenario, probe, rates, cranked=True).damage
        soak = run_soak(
            schedule.soak_s, scenario, probe, rates,
            initial_coolant_c=float(driving[-1]["coolant_c"]),
        ).damage
        return trip.scaled(schedule.trips_per_day) + soak.scaled(
            schedule.trips_per_day
        )

    life = ensemble(day_damage, rates, schedule, draws=16, seed=scenario.seed)

    return write_dataset(
        out_dir=out_dir, scenario=scenario, trips=trips, life=life,
        provenance=LEGACY_PROVENANCE, dropouts=dropouts, rates=rates,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", help="a recorded OutGauge-shaped CSV")
    parser.add_argument("--name", default="run")
    parser.add_argument("--out", default="runs/dataset")
    parser.add_argument("--ambient", type=float, default=25.0)
    parser.add_argument("--hvac", type=float, default=0.0)
    parser.add_argument("--lights", action="store_true")
    parser.add_argument("--repeats", type=int, default=4,
                        help="how many times the recorded trip is driven")
    parser.add_argument("--trips-per-day", type=float, default=2.0)
    parser.add_argument("--soak-hours", type=float, default=8.0)
    args = parser.parse_args(argv)

    scenario = BatteryScenario(
        name=args.name, ambient_c=args.ambient, hvac=args.hvac,
        lights=args.lights,
    )
    schedule = TripSchedule(
        trips_per_day=args.trips_per_day, soak_s=args.soak_hours * 3600.0
    )
    sidecar = build(
        args.trajectory, scenario, schedule, AgingRates(), args.repeats, args.out
    )

    life = sidecar["life"]
    print(f"wrote {args.out}/within_trip.csv "
          f"({sidecar['tables']['within_trip']['rows']} rows at 1 Hz)")
    print(f"wrote {args.out}/trip_damage.csv "
          f"({sidecar['tables']['trip_damage']['rows']} trips)")
    if sidecar["dropouts"]:
        for span in sidecar["dropouts"]:
            print(f"  excluded a telemetry dropout: {span['start_s']:.1f}s to "
                  f"{span['end_s']:.1f}s, {span['rows']} rows")
    if life["eol_days"] is not None:
        interval = life["interval_days"]
        print(f"  end of life at about {life['eol_days']:.0f} days"
              + (f" (90% interval {interval[0]:.0f} to {interval[1]:.0f})"
                 if interval else ""))
    print("  scale is UNFITTED: no battery in this project reached end of life, "
          "so absolute days are indicative. Ratios between scenarios are the "
          "usable output.")
    findings = sidecar["leakage"]["findings"]
    print(f"  leakage findings: {len(findings)}")
    for finding in findings:
        print(f"    {finding['relation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_battery_run.py -q`
Expected: PASS, 7 tests

- [ ] **Step 5: Run the whole suite**

Run: `python3 -m pytest -q`
Expected: PASS. The stage 1 suite must be untouched — `collect/schema.py` was not modified.

- [ ] **Step 6: Run it against the real recording**

```bash
python3 battery_run.py runs/telemetry.csv --name baseline --ambient 25 --out runs/baseline
python3 battery_run.py runs/telemetry.csv --name hot --ambient 42 --hvac 1.0 --lights --out runs/hot
```

Sanity-check the output before believing any of it:

- the dropout from about 1779 s to 1930 s is reported as excluded
- state of charge stays inside [0, 1] and voltage inside [9, 15]
- bay temperature peaks **after** the engine stops, not during the drive
- the hot run's end of life is shorter than the baseline's
- leakage findings: 0

Then compare them, which is the claim that does not depend on the unfitted constant:

```bash
python3 -c "
import json
cool = json.load(open('runs/baseline/dataset.json'))['life']['eol_days']
hot = json.load(open('runs/hot/dataset.json'))['life']['eol_days']
print(f'hot ages it {cool/hot:.2f}x faster than baseline')
"
```

- [ ] **Step 7: Update the docs**

In `README.md`, add the command under the existing usage block. In `CLAUDE.md`,
update the Status section: `load/` and `cell/` are built, stages 3 to 5 are
implemented, and the remaining unbuilt pieces are the world model (stage 2) and
the networks (stages 6 and 7).

- [ ] **Step 8: Commit**

```bash
git add battery_run.py tests/test_battery_run.py README.md CLAUDE.md
git commit -m "Turn a recorded drive into a battery dataset in one command"
```

---

## Self-review notes

**Spec coverage.** Section 4 (adapter honesty) is Task 2; the dropout filter in
Task 4 is new and not in the spec — the spec is amended to record it. Section
5.1 is Task 3, 5.2 is Task 5, 5.3 is Task 6, 5.4 is Task 7. Section 6.1 is
Tasks 8 and 10, 6.2 is Task 9, 6.3 is Tasks 10 and 11, 6.4 is Task 11. Section
7 is Task 1 plus `TripSchedule` in Task 11 — the spec said one `ScenarioSpec`
carrying trips, and the plan splits it, because `collect.ScenarioSpec` already
owns ambient and accessory load and restating them would let the two drift.
Section 8 is Tasks 12 and 13. Section 9 is spread across every task.

**Deferred deliberately.** Vibration uses the IMU only; the per-wheel
`downForce` channel the spec mentions is `absent` in this recording, so it
enters when canonical data does. Start-stop remains off, as parent open
question 5 leaves it.
