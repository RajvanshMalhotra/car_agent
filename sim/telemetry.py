"""BeamNG UDP telemetry decoding: OutGauge and OutSim.

Both are LFS-derived formats that BeamNG re-emits. The split matters:

  OutGauge  -- speed, RPM, coolant temperature, pedals, gear. **No position.**
  MotionSim -- position, velocity, orientation. **No engine data.** This is what
               BeamNG actually emits; it is a BeamNG format, not the LFS one,
               and it is tagged with the magic "BNG1".
  OutSim    -- the LFS format. Kept for a build that emits it instead.

BeamNG sends OutGauge and MotionSim to whatever port each is configured with,
and in practice both end up on the same one. `identify()` exists so a single
socket can be demultiplexed by content rather than by port.

Path following needs pose, so OutSim is not optional. Neither carries
under-bonnet temperature; that is estimated in `sim/engine.py` from coolant
temperature and airflow.

The layouts below are the documented LFS ones. BeamNG's exact emission on the
current build is unverified -- run `windows_probe.py` on the machine with the
game to confirm before trusting a decode.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

# time, car[4], flags, gear, plid, speed, rpm, turbo, engTemp, fuel,
# oilPressure, oilTemp, dashLights, showLights, throttle, brake, clutch,
# display1[16], display2[16]
OUTGAUGE_LAYOUT = "<I4sH2B7f2I3f16s16s"
OUTGAUGE_SIZE = struct.calcsize(OUTGAUGE_LAYOUT)

# time, angVel[3], heading, pitch, roll, accel[3], vel[3], pos[3] (1 m = 65536)
OUTSIM_LAYOUT = "<I3f3f3f3f3i"
OUTSIM_SIZE = struct.calcsize(OUTSIM_LAYOUT)

#: OutSim positions are fixed-point integers.
OUTSIM_POSITION_SCALE = 65536.0

# BeamNG MotionSim: magic "BNG1" then 21 floats --
# pos[3], vel[3], acc[3], up[3], rollPos/pitchPos/yawPos,
# rollVel/pitchVel/yawVel, rollAcc/pitchAcc/yawAcc.
MOTIONSIM_MAGIC = b"BNG1"
MOTIONSIM_LAYOUT = "<4s21f"
MOTIONSIM_SIZE = struct.calcsize(MOTIONSIM_LAYOUT)

#: Below this speed the velocity vector is too noisy to give a heading.
HEADING_FROM_VELOCITY_MPS = 0.5


class TelemetryError(ValueError):
    """A UDP datagram did not decode as the expected telemetry format."""


def _unpack(layout: str, size: int, data: bytes, name: str):
    # Both formats may carry an optional trailing int ID.
    if len(data) not in (size, size + 4):
        raise TelemetryError(
            f"{name}: expected {size} or {size + 4} bytes, got {len(data)} bytes"
        )
    return struct.unpack(layout, data[:size])


@dataclass(frozen=True)
class OutGaugePacket:
    time_ms: int
    gear: int
    speed_mps: float
    rpm: float
    coolant_temp_c: float
    oil_temp_c: float
    fuel_fraction: float
    throttle: float
    brake: float
    clutch: float

    @classmethod
    def parse(cls, data: bytes) -> "OutGaugePacket":
        f = _unpack(OUTGAUGE_LAYOUT, OUTGAUGE_SIZE, data, "OutGauge")
        return cls(
            time_ms=f[0],
            gear=f[3] - 1,  # wire: reverse 0, neutral 1, first 2
            speed_mps=f[5],
            rpm=f[6],
            coolant_temp_c=f[8],
            oil_temp_c=f[11],
            fuel_fraction=f[9],
            throttle=f[14],
            brake=f[15],
            clutch=f[16],
        )


@dataclass(frozen=True)
class OutSimPacket:
    time_ms: int
    x_m: float
    y_m: float
    z_m: float
    heading_rad: float
    speed_mps: float

    @classmethod
    def parse(cls, data: bytes) -> "OutSimPacket":
        f = _unpack(OUTSIM_LAYOUT, OUTSIM_SIZE, data, "OutSim")
        return cls(
            time_ms=f[0],
            heading_rad=f[4],
            speed_mps=math.sqrt(f[10] ** 2 + f[11] ** 2 + f[12] ** 2),
            x_m=f[13] / OUTSIM_POSITION_SCALE,
            y_m=f[14] / OUTSIM_POSITION_SCALE,
            z_m=f[15] / OUTSIM_POSITION_SCALE,
        )


@dataclass(frozen=True)
class MotionSimPacket:
    """BeamNG's own motion protocol. The only source of vehicle pose."""

    x_m: float
    y_m: float
    z_m: float
    speed_mps: float
    heading_rad: float
    yaw_rad: float

    @classmethod
    def parse(cls, data: bytes) -> "MotionSimPacket":
        if len(data) != MOTIONSIM_SIZE:
            raise TelemetryError(
                f"MotionSim: expected {MOTIONSIM_SIZE} bytes, got {len(data)} bytes"
            )
        fields = struct.unpack(MOTIONSIM_LAYOUT, data)
        if fields[0] != MOTIONSIM_MAGIC:
            raise TelemetryError(
                f"MotionSim: expected magic {MOTIONSIM_MAGIC!r}, got {fields[0]!r}"
            )
        x, y, z = fields[1:4]
        vx, vy, vz = fields[4:7]
        yaw = fields[15]  # yawPos

        # Heading is taken from the velocity vector whenever the car is moving:
        # velocity is guaranteed to be in the same frame as the positions, while
        # the sign and zero convention of yawPos is undocumented -- and getting
        # that wrong inverts the steering and spirals the car off the road.
        speed = math.sqrt(vx * vx + vy * vy + vz * vz)
        planar = math.hypot(vx, vy)
        heading = math.atan2(vy, vx) if planar > HEADING_FROM_VELOCITY_MPS else yaw
        return cls(
            x_m=x, y_m=y, z_m=z, speed_mps=speed, heading_rad=heading, yaw_rad=yaw
        )


def identify(data: bytes) -> str | None:
    """Which telemetry format a datagram is, or None.

    Both streams share a port in practice, so routing is by content.
    """
    if len(data) == MOTIONSIM_SIZE and data[:4] == MOTIONSIM_MAGIC:
        return "motionsim"
    if len(data) in (OUTGAUGE_SIZE, OUTGAUGE_SIZE + 4):
        return "outgauge"
    if len(data) in (OUTSIM_SIZE, OUTSIM_SIZE + 4):
        return "outsim"
    return None
