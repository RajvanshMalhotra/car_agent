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
from load.dropout import DropoutFilter

#: The wider provenance vocabulary. `collect/schema.py` only needs the first
#: two -- stage 1's tests pin that -- so this module carries the rest.
PROVENANCES = ("measured", "derived", "assumed", "absent")

#: ETK I-Series node-mass sum, measured once over MCP. A constant here.
ASSUMED_MASS_KG = 1510.62

#: Above this the engine is turning. The recorded file idles around 800 rpm and
#: sits at 3 rpm with the engine off, so the threshold is not delicate.
ENGINE_RUNNING_RPM = 50.0

#: The recorded `fuel` channel is a fraction, not litres. A nominal tank turns
#: it into the canonical unit; the size is an assumption, but nothing in this
#: plan consumes fuel volume, so it costs nothing.
NOMINAL_TANK_L = 50.0


class AbsentChannel(KeyError):
    """An absent column was read. There is no value to return."""


class Sample(dict):
    """A canonical sample that refuses to serve a column it does not have.

    An absent column is never stored as a key at all -- `_lift()` only seeds
    the columns it can populate -- so `dict(sample)`, `.items()` and
    `.values()` simply have no entry for it. `__getitem__` and `get()` are
    overridden only to turn the resulting plain `KeyError` / `None` into an
    informative `AbsentChannel` before the caller can mistake either for "the
    column is present, its value happens to be falsy".
    """

    __slots__ = ("_provenance",)

    def __init__(self, values: dict, provenance: dict[str, str]) -> None:
        super().__init__(values)
        self._provenance = provenance

    def _absent_error(self, key: str) -> AbsentChannel:
        return AbsentChannel(
            f"{key} is absent from this trajectory "
            f"({LEGACY_SOURCE.get(key, '')}); it cannot be read, and a "
            "zero here would be a silent lie"
        )

    def __getitem__(self, key):
        if self._provenance.get(key) == "absent":
            raise self._absent_error(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        if self._provenance.get(key) == "absent":
            raise self._absent_error(key)
        return super().get(key, default)


LEGACY_SOURCE: dict[str, str] = {
    # pose and motion, straight from MotionSim
    "t_s": "timestamp, rebased to zero at the first row",
    "seq": "row index",
    "x_m": "pos_x", "y_m": "pos_y", "z_m": "pos_z",
    "vx_mps": "vel_x", "vy_mps": "vel_y", "vz_mps": "vel_z",
    "speed_mps": "speed",
    "ax_mps2": "acc_x", "ay_mps2": "acc_y", "az_mps2": "acc_z",
    "altitude_m": "pos_z",
    # powertrain
    "rpm": "rpm",
    "throttle": "throttle",
    "gear_index": "gear",
    # thermal
    "coolant_c": "engine_temp",
    "oil_c": "oil_temp",
    # assumed
    "mass_kg": f"constant {ASSUMED_MASS_KG} kg, the ETK I-Series node-mass sum",
    "engine_load": "throttle, as a stand-in; the real channel was not recorded",
    "fuel_volume_l": (
        "fuel fraction times a nominal tank size of "
        f"{NOMINAL_TANK_L} L, a constant we chose -- the recording carries "
        "no litre reading"
    ),
    # absent
    "brake": "the recorder captured OutGauge's display1 blob instead",
    "clutch_ratio": "constant zero in the recording; never populated",
    "steering": (
        "the recording carries no steering channel; yaw_vel is yaw RATE, a "
        "different physical quantity, and is not a substitute for steering "
        "angle without speed and a vehicle model"
    ),
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
    "dir_x": "cos(pitch) cos(yaw); the recording has no direction-vector column",
    "dir_y": "cos(pitch) sin(yaw); the recording has no direction-vector column",
    "dir_z": "sin(pitch); the recording has no direction-vector column",
    "engine_running": f"rpm > {ENGINE_RUNNING_RPM}",
    "grade_rad": "asin(dir_z)",
    "a_long_mps2": "ax_mps2 with the gravity component removed",
}

_ABSENT = frozenset({
    "brake", "clutch_ratio", "engine_torque_nm", "engine_av_rads",
    "exhaust_flow", "ignition_level", "parkingbrake", "odometer_m", "trip_m",
    "avg_wheel_av", "damage", "steering",
    "wheel_av_fl", "wheel_av_fr", "wheel_av_rl", "wheel_av_rr",
    "brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr",
    "downforce_fl",
})
_ASSUMED = frozenset({"mass_kg", "engine_load", "fuel_volume_l"})
_DERIVED = frozenset({
    "dir_x", "dir_y", "dir_z", "engine_running", "grade_rad", "a_long_mps2",
})


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


def _lift(raw: dict, index: int, t0: float) -> dict:
    pitch = float(raw["pitch"])
    yaw = float(raw["yaw"])
    # Seed only the columns this adapter actually populates. An absent column
    # must never become a key here -- if it did, `dict(sample)`, `.values()`
    # and `.get()` would all hand out a silent 0.0 for it, which is exactly
    # the failure this module exists to prevent.
    values = {column: 0.0 for column in COLUMNS if column not in _ABSENT}
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
