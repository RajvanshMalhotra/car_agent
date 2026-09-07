"""BeamNG UDP telemetry decoding: OutGauge and OutSim.

Both are LFS-derived formats that BeamNG re-emits. The split matters:

  OutGauge -- speed, RPM, coolant temperature, pedals, gear. **No position.**
  OutSim   -- position, heading, velocity. **No engine data.**

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
