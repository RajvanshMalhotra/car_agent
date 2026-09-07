"""A learned driving policy.

The hand-built path is: measure the vehicle, then tune a controller for it.
That means a wheelbase, an acceleration limit, a braking limit, a calibration
step that can fail, and an alignment roll before every run. All of it is
vehicle-specific, and all of it is a thing that can be got wrong.

This is the other route. The policy sees six numbers -- how far off the target
speed it is, how far off the line, which way the line goes, how fast it is
going, how sharply the road bends ahead, and what it did last -- and produces
the three controls. Nothing about the vehicle appears anywhere. It learns the
response of whatever it is driving, because it is trained across randomised
vehicles.

Deliberately small: six features, two output rows, fourteen numbers. That is
enough to express what a PID and pure pursuit express together, it trains in
seconds by policy search, and you can read the result.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path as FilePath
from typing import Any, Sequence

from control.path import Path
from sim.backend import ControlInput

#: speed error, cross-track, heading error, speed, curvature ahead, last steer,
#: and a bias term.
FEATURE_COUNT = 7

#: Observations are scaled so every feature is order 1 and then clipped. A
#: policy that sees an unbounded input produces an unbounded output, and one
#: bad frame should not throw the car off the road.
FEATURE_CLIP = 5.0

#: How far ahead the bend is measured.
CURVATURE_LOOKAHEAD_M = 15.0

#: Which features each output is allowed to use.
#:
#: The pedals do not get to see cross-track or heading error. Left free, the
#: search learns to brake whenever the car is pointing off-line -- reasonable
#: while moving, fatal at a standstill, where it cannot correct its heading
#: without moving and cannot move because it is braking. It then sits there for
#: the rest of the run. This is a structural prior added after watching exactly
#: that happen: which way you are pointing is the steering's business, and
#: reaches the pedals only through the bend ahead.
FEATURE_NAMES = (
    "speed_error", "cross_track", "heading_error", "speed", "curvature",
    "previous_steering", "bias",
)
STEERING_MASK = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
PEDAL_MASK = (1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)


def _wrap(angle: float) -> float:
    """Signed angle in (-pi, pi]: 179 degrees off is a large error, not a small one."""
    return math.atan2(math.sin(angle), math.cos(angle))


def observe(
    path: Path,
    x: float,
    y: float,
    heading: float,
    speed: float,
    target_speed: float,
    previous_steering: float,
    progress_m: float | None = None,
) -> list[float]:
    """What the policy sees. Every value scaled to order 1 and clipped."""
    arc_length = path.closest_arc_length((x, y), near_arc_length=progress_m)
    near_x, near_y = path.point_at(arc_length)

    # Signed cross-track: positive when the line is to the vehicle's left.
    dx, dy = near_x - x, near_y - y
    cross_track = -dx * math.sin(heading) + dy * math.cos(heading)

    ahead_x, ahead_y = path.point_at(arc_length + CURVATURE_LOOKAHEAD_M)
    heading_error = _wrap(math.atan2(ahead_y - y, ahead_x - x) - heading)
    curvature = path.curvature_at(arc_length + CURVATURE_LOOKAHEAD_M / 2)

    raw = [
        (target_speed - speed) / 10.0,
        cross_track / 5.0,
        heading_error,
        speed / 20.0,
        curvature * 20.0,
        previous_steering,
        1.0,
    ]
    return [min(FEATURE_CLIP, max(-FEATURE_CLIP, value)) for value in raw]


@dataclass(frozen=True)
class DrivingPolicy:
    """Two linear rows over the observation: one for steering, one for pedals."""

    steering_weights: tuple[float, ...] = field(default=())
    pedal_weights: tuple[float, ...] = field(default=())

    def __post_init__(self) -> None:
        object.__setattr__(self, "steering_weights", tuple(self.steering_weights))
        object.__setattr__(self, "pedal_weights", tuple(self.pedal_weights))
        for name in ("steering_weights", "pedal_weights"):
            if len(getattr(self, name)) != FEATURE_COUNT:
                raise ValueError(
                    f"{name}: expected {FEATURE_COUNT} weights, "
                    f"got {len(getattr(self, name))}"
                )

    @classmethod
    def zero(cls) -> "DrivingPolicy":
        return cls((0.0,) * FEATURE_COUNT, (0.0,) * FEATURE_COUNT)

    @classmethod
    def random(cls, seed: int = 0, scale: float = 0.5) -> "DrivingPolicy":
        rng = random.Random(seed)
        return cls(
            tuple(rng.gauss(0.0, scale) for _ in range(FEATURE_COUNT)),
            tuple(rng.gauss(0.0, scale) for _ in range(FEATURE_COUNT)),
        )

    @property
    def weights(self) -> list[float]:
        return list(self.steering_weights) + list(self.pedal_weights)

    @classmethod
    def from_weights(cls, weights: Sequence[float]) -> "DrivingPolicy":
        return cls(
            tuple(weights[:FEATURE_COUNT]), tuple(weights[FEATURE_COUNT:])
        )

    def act(self, features: Sequence[float]) -> ControlInput:
        steering = math.tanh(
            sum(w * f * m for w, f, m in zip(self.steering_weights, features, STEERING_MASK))
        )
        # One pedal axis: positive is throttle, negative is brake. The car
        # physically cannot do both, so the policy is not allowed to ask.
        pedal = math.tanh(
            sum(w * f * m for w, f, m in zip(self.pedal_weights, features, PEDAL_MASK))
        )
        return ControlInput(
            throttle=max(0.0, pedal),
            brake=max(0.0, -pedal),
            steering=min(1.0, max(-1.0, steering)),
        )

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "steering_weights": list(self.steering_weights),
            "pedal_weights": list(self.pedal_weights),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DrivingPolicy":
        return cls(data["steering_weights"], data["pedal_weights"])

    def save(self, file_path: FilePath | str, metadata: dict[str, Any] | None = None) -> None:
        file_path = FilePath(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps({**self.to_dict(), "metadata": metadata or {}}, indent=2)
        )

    @classmethod
    def load(cls, file_path: FilePath | str) -> "DrivingPolicy":
        return cls.from_dict(json.loads(FilePath(file_path).read_text()))
