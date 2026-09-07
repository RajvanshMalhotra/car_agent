"""Measuring a vehicle instead of assuming one.

The controller needs to know how hard the car accelerates, how hard it brakes,
and how sharply it turns. Those were hardcoded to a passenger car -- 2.7 m
wheelbase, 0.5 rad lock, 3 m/s^2, 8 m/s^2 -- which makes every pedal and
steering command mis-scaled on anything else. A 6x6 truck is very much anything
else, and a mis-scaled steering command is exactly what wandering looks like.

So they are measured, on the vehicle, in about twenty seconds, and cached per
vehicle so it happens once.

Only the ratio `max_steer_rad / wheelbase_m` is identifiable from yaw rate, so
the lock is held fixed and the wheelbase carries the measurement. That is the
only combination pure pursuit uses.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path as FilePath

from sim.backend import ControlInput, SimBackend

#: What a road vehicle can plausibly do. A measurement outside these is a
#: timing artefact or a stuck vehicle, not a fast car -- and driving on it
#: mis-scales every command, which is exactly the wandering this module exists
#: to prevent. Physics guardrails, same as everywhere else in this project.
PLAUSIBLE: dict[str, tuple[float, float]] = {
    "max_accel_mps2": (0.3, 15.0),
    "max_decel_mps2": (0.5, 20.0),
    "wheelbase_m": (1.0, 25.0),
}


class CalibrationError(ValueError):
    """A measured vehicle limit was not physically plausible."""


#: Held fixed; the wheelbase absorbs the measured steering response.
NOMINAL_STEER_LOCK_RAD = 0.5

#: Steering input used for the turn test. Large enough to give a clean yaw
#: rate, small enough that the small-angle error stays a few percent.
PROBE_STEER = 0.5


@dataclass(frozen=True)
class VehicleLimits:
    vehicle: str
    max_accel_mps2: float
    max_decel_mps2: float
    wheelbase_m: float
    max_steer_rad: float = NOMINAL_STEER_LOCK_RAD

    def validate(self) -> "VehicleLimits":
        """Raise unless every measurement is physically plausible."""
        violations = [
            f"{name}: {getattr(self, name):.3g} outside [{low}, {high}]"
            for name, (low, high) in PLAUSIBLE.items()
            if not low <= getattr(self, name) <= high
        ]
        if violations:
            raise CalibrationError(
                f"implausible calibration for {self.vehicle!r}: "
                + "; ".join(violations)
            )
        return self


#: Guard against a backend whose clock never advances.
MAX_STEPS_PER_PHASE = 100_000


def _drive(backend: SimBackend, control: ControlInput, dt: float, seconds: float):
    """Hold a control for `seconds` of *simulated* time.

    Paced by the vehicle's own clock, not by an iteration count. Over MCP each
    iteration is an HTTP round trip and covers far less than `dt`, so counting
    iterations would end the phase a fraction of the way through.
    """
    samples = []
    start = backend.read_state().sim_time_s
    for _ in range(MAX_STEPS_PER_PHASE):
        backend.apply_control(control)
        state = backend.read_state()
        samples.append((state.sim_time_s, state.speed_mps, state.heading_rad))
        if state.sim_time_s - start >= seconds:
            break
    return samples


def _peak_rate(samples, window: int = 5) -> float:
    """Largest sustained rate of speed change across the samples."""
    best = 0.0
    for a, b in zip(samples, samples[window:]):
        span = b[0] - a[0]
        if span > 0:
            best = max(best, abs(b[1] - a[1]) / span)
    return best


def calibrate(backend: SimBackend, dt: float, vehicle: str = "unknown") -> VehicleLimits:
    """Measure the vehicle's limits by driving it. Releases controls when done."""
    # 1. How hard does it accelerate?
    accel_samples = _drive(backend, ControlInput(1.0, 0.0, 0.0), dt, seconds=6.0)
    max_accel = _peak_rate(accel_samples)
    if max(s[1] for s in accel_samples) < 1.0:
        backend.apply_control(ControlInput(0.0, 0.0, 0.0))
        raise RuntimeError(
            "the vehicle did not move under full throttle. Check it is in gear, "
            "on the ground, and that controls are reaching it."
        )

    # 2. How hard does it stop? Measured from whatever speed it reached.
    brake_samples = _drive(backend, ControlInput(0.0, 1.0, 0.0), dt, seconds=4.0)
    max_decel = _peak_rate(brake_samples)

    # 3. How sharply does it turn? Yaw rate at a known speed and steering input.
    _drive(backend, ControlInput(1.0, 0.0, 0.0), dt, seconds=5.0)
    turn_samples = _drive(backend, ControlInput(0.3, 0.0, PROBE_STEER), dt, seconds=4.0)
    backend.apply_control(ControlInput(0.0, 0.0, 0.0))

    # Skip the first samples: the steering rack takes a moment to respond.
    settled = turn_samples[len(turn_samples) // 3:]
    span = settled[-1][0] - settled[0][0]
    swept = abs(settled[-1][2] - settled[0][2])
    mean_speed = sum(s[1] for s in settled) / len(settled)
    yaw_rate = swept / span if span > 0 else 0.0

    if yaw_rate <= 1e-6 or mean_speed <= 0.1:
        wheelbase = 2.7
    else:
        # yaw_rate = speed / wheelbase * tan(steer * lock)
        wheelbase = mean_speed * math.tan(PROBE_STEER * NOMINAL_STEER_LOCK_RAD) / yaw_rate

    return VehicleLimits(
        vehicle=vehicle,
        max_accel_mps2=max(0.2, max_accel),
        max_decel_mps2=max(0.5, max_decel),
        wheelbase_m=min(30.0, max(0.5, wheelbase)),
    )


def save_limits(directory: FilePath | str, limits: VehicleLimits) -> None:
    directory = FilePath(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{limits.vehicle}.json").write_text(
        json.dumps(
            {**asdict(limits), "measured_at": datetime.now(timezone.utc).isoformat()},
            indent=2,
        )
    )


def load_limits(directory: FilePath | str, vehicle: str) -> VehicleLimits | None:
    path = FilePath(directory) / f"{vehicle}.json"
    if not path.exists():
        return None
    stored = json.loads(path.read_text())
    stored.pop("measured_at", None)
    return VehicleLimits(**stored)
