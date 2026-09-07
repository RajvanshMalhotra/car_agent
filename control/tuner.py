"""Learning controller gains for a vehicle.

The gains were hand-tuned for a passenger car -- and wrong gains on a 6.5 m
truck are exactly what wandering looks like. `control.calibration` measures what
the vehicle can do; this learns how to drive it.

The method is the cross-entropy method: sample gains from a Gaussian, score
each by driving a whole run, keep the best, refit the Gaussian, repeat. It is
policy search over five continuous parameters rather than a network over raw
observations, and that choice is deliberate:

  * it trains against the **fake** backend configured with the vehicle's
    *measured* limits, so a few hundred episodes take seconds instead of days
    of BeamNG time;
  * five parameters need hundreds of episodes, not millions;
  * the result is five numbers you can read, not a policy you have to trust.

What it cannot do is perceive anything. It learns to track a route well. It does
not learn to avoid obstacles, because nothing here sees them.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any

from behaviour.spec import BehaviourSpec
from control.calibration import VehicleLimits
from control.path import Path

#: (low, high) for each learnable parameter.
BOUNDS = {
    "kp": (0.05, 4.0),
    "ki": (0.0, 1.5),
    "kd": (0.0, 0.5),
    "lookahead_gain_s": (0.1, 2.0),
    "min_lookahead_m": (2.0, 25.0),
}

#: How badly to punish losing the route: it should dominate any tracking gain.
LOST_PENALTY = 1000.0


@dataclass(frozen=True)
class ControllerGains:
    kp: float
    ki: float
    kd: float
    lookahead_gain_s: float
    min_lookahead_m: float

    @classmethod
    def default(cls) -> "ControllerGains":
        """The hand-tuned passenger-car values."""
        return cls(kp=0.8, ki=0.15, kd=0.02, lookahead_gain_s=0.6, min_lookahead_m=5.0)

    def is_valid(self) -> bool:
        return all(
            low <= getattr(self, name) <= high for name, (low, high) in BOUNDS.items()
        )

    def clipped(self) -> "ControllerGains":
        return ControllerGains(
            **{
                name: min(high, max(low, getattr(self, name)))
                for name, (low, high) in BOUNDS.items()
            }
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "ControllerGains":
        return cls(**data)


def _winding_course(length_m: float = 1500.0) -> Path:
    """A course with straights and bends, so gains cannot overfit a straight."""
    return Path(
        [(i * 2.0, 22.0 * math.sin(i * 2.0 / 70.0)) for i in range(int(length_m / 2))]
    )


def evaluate(
    gains: ControllerGains,
    spec: BehaviourSpec,
    limits: VehicleLimits,
    seconds: float = 60.0,
    speed_limit_mps: float = 20.0,
    dt: float = 0.02,
    seed: int = 0,
    path: Path | None = None,
) -> float:
    """Score one set of gains. Higher is better; scores are penalties, so < 0.

    Penalises distance from the route and speed error, and heavily penalises
    losing the route entirely.
    """
    from control.driver import Driver
    from sim.fake import FakeBackend

    course = path or _winding_course()
    backend = FakeBackend(
        dt=dt,
        wheelbase_m=limits.wheelbase_m,
        max_steer_rad=limits.max_steer_rad,
        max_accel_mps2=limits.max_accel_mps2,
        max_decel_mps2=limits.max_decel_mps2,
        seed=seed,
    )
    backend.reset()
    driver = Driver(
        spec,
        course,
        dt=dt,
        speed_limit_mps=speed_limit_mps,
        seed=seed,
        wheelbase_m=limits.wheelbase_m,
        max_steer_rad=limits.max_steer_rad,
        max_accel_mps2=limits.max_accel_mps2,
        max_decel_mps2=limits.max_decel_mps2,
        gains=gains,
    )

    target = speed_limit_mps * spec.target_speed_factor
    deviation_sum = 0.0
    speed_error_sum = 0.0
    samples = 0
    for _ in range(int(seconds / dt)):
        backend.apply_control(driver.step(backend.read_state()))
        state = backend.read_state()
        samples += 1
        deviation_sum += driver.deviation_m(state)
        speed_error_sum += abs(state.speed_mps - target)
        if driver.is_lost(state):
            return -LOST_PENALTY
        if driver.is_finished(state):
            break

    if samples == 0:
        return -LOST_PENALTY
    # Tracking dominates; speed adherence is the tie-breaker.
    return -(deviation_sum / samples) - 0.1 * (speed_error_sum / samples)


def tune(
    spec: BehaviourSpec,
    limits: VehicleLimits,
    iterations: int = 8,
    population: int = 24,
    elite_fraction: float = 0.25,
    seconds: float = 60.0,
    seed: int = 0,
    **evaluate_kwargs: Any,
) -> ControllerGains:
    """Cross-entropy search for gains that suit this vehicle."""
    rng = random.Random(seed)
    names = list(BOUNDS)
    mean = {name: getattr(ControllerGains.default(), name) for name in names}
    # Start wide enough to escape the passenger-car defaults.
    spread = {name: (high - low) / 4.0 for name, (low, high) in BOUNDS.items()}
    elite_count = max(2, int(population * elite_fraction))
    best = ControllerGains.default()
    best_score = -math.inf

    for _ in range(iterations):
        scored = []
        for _ in range(population):
            candidate = ControllerGains(
                **{name: rng.gauss(mean[name], spread[name]) for name in names}
            ).clipped()
            score = evaluate(
                candidate, spec, limits, seconds=seconds, seed=seed, **evaluate_kwargs
            )
            scored.append((score, candidate))
            if score > best_score:
                best_score, best = score, candidate

        scored.sort(key=lambda item: item[0], reverse=True)
        elite = [candidate for _, candidate in scored[:elite_count]]
        for name in names:
            values = [getattr(candidate, name) for candidate in elite]
            mean[name] = sum(values) / len(values)
            variance = sum((v - mean[name]) ** 2 for v in values) / len(values)
            # A floor on the spread keeps the search from collapsing early.
            spread[name] = max(math.sqrt(variance), (BOUNDS[name][1] - BOUNDS[name][0]) / 50)

    return best
