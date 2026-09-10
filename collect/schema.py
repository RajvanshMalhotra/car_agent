"""Every channel, named once.

The Lua sampler and the Python decoder are both generated from this table, so a
channel cannot exist on one side and not the other -- which is the failure mode
that produces a column of zeros nobody notices for a week.

`lua` is an expression evaluated inside the vehicle VM with these locals bound
once per sample by the preamble in `collect.lua`:

    S     the sampler's state (S.t elapsed sim time, S.mass the node-mass sum,
          S.wi a wheel-name to index map)
    p     obj:getPosition()
    vel   obj:getVelocity()
    d     obj:getDirectionVector()
    e     electrics.values
    eng   powertrain.getDevice('mainEngine')
    w     wheels.wheels

Provenance is not documentation. The sidecar carries it, downstream code reads
it, and nothing may present a derived column as a measurement.

There is deliberately no battery channel here. BeamNG simulates no 12 V system:
the ~200 keys of `electrics.values` contain nothing matching volt, batt, amp,
current or alternator. Battery current and voltage are modelled downstream, and
a column here would read as a measurement.
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
                f"{self.name}: provenance {self.provenance!r} is not one of {PROVENANCES}"
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
    _m("t_s", "s", "S.t", "elapsed physics time, accumulated from dt"),
    _m("seq", "count", "S.seq", "sample counter; a gap means the buffer overflowed"),
    # pose and motion
    _m("x_m", "m", "p.x"),
    _m("y_m", "m", "p.y"),
    _m("z_m", "m", "p.z"),
    _m("vx_mps", "m/s", "vel.x"),
    _m("vy_mps", "m/s", "vel.y"),
    _m("vz_mps", "m/s", "vel.z"),
    _m("speed_mps", "m/s", "e.wheelspeed"),
    # The IMU. Reads gravity as well as motion, so a_long is derived, not this.
    _m("ax_mps2", "m/s2", "sensors.gx", "vehicle frame, gravity included"),
    _m("ay_mps2", "m/s2", "sensors.gy", "vehicle frame, gravity included"),
    _m("az_mps2", "m/s2", "sensors.gz", "vehicle frame, gravity included"),
    # Unit forward vector. dir_z is sin(pitch): the road grade, measured.
    _m("dir_x", "1", "d.x"),
    _m("dir_y", "1", "d.y"),
    _m("dir_z", "1", "d.z", "sin(pitch); grade_rad is its arcsine"),
    _m("mass_kg", "kg", "S.mass", "node-mass sum, computed once at install"),
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
    _m("ignition_level", "1", "e.ignitionLevel", "crank and trip detection"),
    _m("engine_running", "1", "e.engineRunning"),
    _m("damage", "1", "(beamstate and beamstate.damage) or 0"),
)

DERIVED: tuple[Channel, ...] = (
    _d("grade_rad", "rad", "asin(dir_z)"),
    _d("a_long_mps2", "m/s2", "ax_mps2 with the gravity component removed"),
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
