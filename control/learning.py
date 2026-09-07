"""Learning to drive, across vehicles.

Every episode randomises the vehicle -- wheelbase, acceleration, braking,
steering rack speed -- and the target speed. A policy that scores well has to
work on all of them, which is what removes the per-vehicle calibration step: it
is not tuned for a car and then rejected on a truck, it is trained on both.

Optimised by the cross-entropy method: sample policies, score each on several
episodes, keep the best, refit, repeat. Gradient-free, so there is no framework
to install, and fourteen weights converge in a few hundred episodes -- seconds
against the fake backend rather than days against the game.

What this does not do is perceive. It learns to follow a line at a speed. It
does not learn to avoid anything, because nothing here can see.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from control.path import Path
from control.policy import FEATURE_COUNT, DrivingPolicy, observe
from sim.fake import FakeBackend

#: The span of vehicles a single policy has to cope with -- roughly a hot
#: hatchback through to a loaded 6x6.
VEHICLE_RANGES: dict[str, tuple[float, float]] = {
    "wheelbase_m": (2.2, 9.0),
    "max_accel_mps2": (0.8, 5.0),
    "max_decel_mps2": (2.5, 9.0),
    "max_steer_rad": (0.35, 0.7),
    "max_steer_rate_rad_per_s": (0.5, 2.5),
}

#: Both crawling and open road, because a policy tuned at one speed is wrong at
#: the other.
SPEED_RANGE_MPS = (3.0, 25.0)

#: Part of every episode asks for a full stop, and it sits in the *middle*.
#: Two reasons. Without any stop the policy is never asked for zero and never
#: learns to brake to a halt, and stops are the whole point of the data being
#: collected. And with the stop at the end it never has to pull away again --
#: which produced a policy that braked whenever it was pointing off-line, got
#: stuck at a standstill it could not steer out of, and sat there for the rest
#: of the run.
STOP_WINDOW = (0.40, 0.65)

#: Motion when asked to stop is penalised harder than a speed error while
#: cruising. Idling with a hot engine is the mechanism the whole dataset is
#: about, and a car creeping at walking pace never registers as stopped at all.
STOP_ERROR_WEIGHT = 5.0

#: Off the line by more than this and the episode is over.
OFF_ROAD_M = 10.0

#: Weight on getting down the course. This is what gives the search something
#: to climb: with only a flat penalty for leaving the road, a policy that
#: survives five metres and one that survives four hundred score the same, and
#: there is no gradient at all from a standing start.
#:
#: Progress is measured against what holding the *target* speed would have
#: covered, and capped at 1. Measured against the clock instead, the fastest
#: policy always wins and the search learns to pin the throttle and ignore the
#: target speed entirely -- which is exactly what it did.
PROGRESS_WEIGHT = 10.0

#: Driving at the speed asked for is an objective in its own right, not a
#: tie-breaker: the whole point is accurate driving at high *and* low speeds.
SPEED_ERROR_WEIGHT = 1.0

DT = 0.05


@dataclass(frozen=True)
class Episode:
    score: float
    mean_cross_track_m: float
    #: Physical speed error while driving, in m/s. Excludes the stop window --
    #: mixing the two produced a "speed error" that was really a reward term,
    #: and made a policy that drives fine look as though it capped out at
    #: walking pace.
    mean_speed_error_mps: float
    #: Mean speed actually held while asked to drive.
    mean_driving_speed_mps: float
    distance_m: float
    left_the_road: bool
    target_speed_mps: float


def random_vehicle(seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    return {
        name: rng.uniform(low, high) for name, (low, high) in VEHICLE_RANGES.items()
    }


def _course(seed: int) -> Path:
    """A different winding course each episode, so nothing overfits one shape."""
    rng = random.Random(seed * 7919)
    amplitude = rng.uniform(0.0, 30.0)
    wavelength = rng.uniform(60.0, 260.0)
    phase = rng.uniform(0.0, math.tau)
    # Shifted so the course passes through the origin: the vehicle starts
    # there, and a course that begins tens of metres away scores every policy
    # as off-road before it has done anything.
    offset = amplitude * math.sin(phase)
    return Path(
        [
            (i * 3.0, amplitude * math.sin(i * 3.0 / wavelength + phase) - offset)
            for i in range(700)
        ]
    )


def drive_episode(
    policy: DrivingPolicy,
    seed: int,
    seconds: float = 30.0,
    target_speed_mps: float | None = None,
) -> Episode:
    """Drive one randomised vehicle on one randomised course."""
    rng = random.Random(seed)
    vehicle = random_vehicle(seed)
    target = (
        target_speed_mps
        if target_speed_mps is not None
        else rng.uniform(*SPEED_RANGE_MPS)
    )
    course = _course(seed)

    backend = FakeBackend(dt=DT, seed=seed, **vehicle)
    backend.reset()
    # Start slightly off the line and askew: a policy that only works from a
    # perfect start is no use after a recovery.
    backend.state.y_m = rng.uniform(-2.0, 2.0)
    backend.state.heading_rad = rng.uniform(-0.3, 0.3)

    cross_track_sum = 0.0
    penalty_sum = 0.0
    driving_error_sum = 0.0
    driving_speed_sum = 0.0
    driving_steps = 0
    steps = 0
    progress = 0.0
    previous_steering = 0.0
    left_the_road = False

    total_steps = int(seconds / DT)
    stop_from = int(total_steps * STOP_WINDOW[0])
    stop_until = int(total_steps * STOP_WINDOW[1])
    progress_when_asked_to_stop = None
    progress_when_released = None
    for step in range(total_steps):
        # The last stretch of each episode asks for a standstill.
        stopping = stop_from <= step < stop_until
        wanted = 0.0 if stopping else target
        if step == stop_from:
            # Progress is frozen through the stop. Otherwise creeping forward
            # still earns progress reward and the policy learns to idle along
            # at walking pace instead of stopping.
            progress_when_asked_to_stop = progress
        elif step == stop_until:
            progress_when_released = progress
        state = backend.read_state()
        features = observe(
            course, state.x_m, state.y_m, state.heading_rad, state.speed_mps,
            wanted, previous_steering, progress_m=progress,
        )
        control = policy.act(features)
        backend.apply_control(control)
        previous_steering = control.steering

        state = backend.read_state()
        progress = course.closest_arc_length(
            (state.x_m, state.y_m), near_arc_length=progress
        )
        near_x, near_y = course.point_at(progress)
        offset = math.hypot(near_x - state.x_m, near_y - state.y_m)

        cross_track_sum += offset
        error = abs(state.speed_mps - wanted)
        penalty_sum += (STOP_ERROR_WEIGHT if stopping else 1.0) * error
        if not stopping:
            driving_error_sum += error
            driving_speed_sum += state.speed_mps
            driving_steps += 1
        steps += 1

        if offset > OFF_ROAD_M:
            left_the_road = True
            break
        if progress >= course.length_m - 20.0:
            break

    mean_cross_track = cross_track_sum / max(1, steps)
    mean_penalty = penalty_sum / max(1, steps)
    mean_speed_error = driving_error_sum / max(1, driving_steps)
    mean_driving_speed = driving_speed_sum / max(1, driving_steps)
    # Leaving the road needs no separate penalty: it ends the episode, so the
    # progress term collapses on its own.
    driving_seconds = seconds * (1.0 - (STOP_WINDOW[1] - STOP_WINDOW[0]))
    expected_distance = max(1.0, min(target * driving_seconds, course.length_m))
    # Distance covered before the stop, plus distance covered after pulling
    # away again. The stop itself earns nothing, so creeping through it is
    # never worth anything -- but getting going afterwards is.
    if progress_when_asked_to_stop is None:
        counted = progress
    elif progress_when_released is None:
        counted = progress_when_asked_to_stop
    else:
        counted = progress_when_asked_to_stop + (progress - progress_when_released)
    progress_fraction = min(1.0, counted / expected_distance)
    score = (
        PROGRESS_WEIGHT * progress_fraction
        - mean_cross_track
        - SPEED_ERROR_WEIGHT * mean_penalty
    )
    return Episode(
        score=score,
        mean_cross_track_m=mean_cross_track,
        mean_speed_error_mps=mean_speed_error,
        mean_driving_speed_mps=mean_driving_speed,
        distance_m=progress,
        left_the_road=left_the_road,
        target_speed_mps=target,
    )


def _fitness(policy: DrivingPolicy, seeds: list[int], seconds: float) -> float:
    """Mean score over several vehicles. One easy vehicle proves nothing."""
    return sum(drive_episode(policy, seed=s, seconds=seconds).score for s in seeds) / len(seeds)


#: Seeds never used in training, for choosing between restarts honestly.
SELECTION_SEEDS = tuple(range(80_000, 80_012))


def learn_with_restarts(
    restarts: int = 4,
    seed: int = 0,
    progress: Any = None,
    **kwargs: Any,
) -> tuple[DrivingPolicy, list[dict[str, float]]]:
    """Train several times and keep the policy that generalises best.

    Cross-entropy search is not reliable from a single start here: measured
    across seeds at a fixed budget, one run left the road on 0 of 16 unseen
    vehicles and another on 16 of 16, and more iterations did not rescue the bad
    one. It is stuck in a local optimum, not undertrained.

    Restarts are the fix, and they are affordable because a run takes about a
    minute. The winner is chosen on episodes never used in training, so the
    choice is not itself overfitting.
    """
    best: DrivingPolicy | None = None
    best_score = -math.inf
    best_history: list[dict[str, float]] = []
    attempts: list[float] = []

    for attempt in range(restarts):
        policy, history = learn_to_drive(seed=seed + attempt * 977, progress=progress, **kwargs)
        held_out = [
            drive_episode(policy, seed=s, seconds=kwargs.get("seconds", 30.0))
            for s in SELECTION_SEEDS
        ]
        score = sum(e.score for e in held_out) / len(held_out)
        off_road = sum(e.left_the_road for e in held_out)
        attempts.append(score)
        if progress is not None:
            progress({
                "iteration": -1,
                "restart": attempt,
                "held_out_score": score,
                "off_road": off_road,
                "best": score,
                "mean": score,
            })
        if score > best_score:
            best_score, best, best_history = score, policy, history

    assert best is not None
    return best, best_history


def learn_to_drive(
    iterations: int = 20,
    population: int = 40,
    episodes_per_candidate: int = 6,
    seconds: float = 30.0,
    elite_fraction: float = 0.25,
    seed: int = 0,
    initial_scale: float = 1.0,
    progress: Any = None,
) -> tuple[DrivingPolicy, list[dict[str, float]]]:
    """Cross-entropy search for weights that drive anything, at any speed."""
    rng = random.Random(seed)
    size = FEATURE_COUNT * 2
    mean = [0.0] * size
    spread = [initial_scale] * size
    elite_count = max(2, int(population * elite_fraction))

    best = DrivingPolicy.zero()
    best_fitness = -math.inf
    history: list[dict[str, float]] = []

    # A fixed training set, the same for every iteration. Drawing fresh
    # vehicles each iteration makes the objective move under the search: scores
    # from different iterations are not comparable, the best-so-far is chosen
    # by luck of the draw, and the run thrashes instead of converging.
    seeds = [rng.randrange(1_000_000) for _ in range(episodes_per_candidate)]

    for iteration in range(iterations):
        scored = []
        for _ in range(population):
            weights = [rng.gauss(mean[i], spread[i]) for i in range(size)]
            candidate = DrivingPolicy.from_weights(weights)
            fitness = _fitness(candidate, seeds, seconds)
            scored.append((fitness, weights, candidate))

        scored.sort(key=lambda item: item[0], reverse=True)
        elite = scored[:elite_count]
        for i in range(size):
            values = [weights[i] for _, weights, _ in elite]
            mean[i] = sum(values) / len(values)
            variance = sum((v - mean[i]) ** 2 for v in values) / len(values)
            # A floor keeps the search from collapsing before it has explored.
            spread[i] = max(math.sqrt(variance), 0.02)

        # Keep whichever of the distribution's centre and the best sample
        # actually scores best on the training set.
        centre = DrivingPolicy.from_weights(mean)
        centre_fitness = _fitness(centre, seeds, seconds)
        for fitness, candidate in ((centre_fitness, centre), (elite[0][0], elite[0][2])):
            if fitness > best_fitness:
                best_fitness, best = fitness, candidate

        entry = {
            "iteration": iteration,
            "best": elite[0][0],
            "mean": sum(f for f, _, _ in scored) / len(scored),
            "centre": centre_fitness,
        }
        history.append(entry)
        if progress is not None:
            progress(entry)

    return best, history
