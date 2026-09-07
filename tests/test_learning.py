"""Learning to drive.

Trained across randomised vehicles and target speeds, so one policy handles a
hatchback and a 6x6 truck at 4 m/s and at 25 m/s without anything being measured
or hand-tuned per vehicle. That is the whole point: no calibration step, nothing
to reject, nothing hardcoded about the car.

Training runs against the fake backend, which is why it takes seconds. What it
buys is a policy; what it cannot buy is perception, because nothing here sees.
"""

import pytest

from control.learning import (
    VEHICLE_RANGES,
    Episode,
    drive_episode,
    learn_to_drive,
    learn_with_restarts,
    random_vehicle,
)
from control.policy import DrivingPolicy


# -- randomised vehicles --------------------------------------------------


def test_a_random_vehicle_is_within_the_stated_ranges():
    vehicle = random_vehicle(seed=1)
    for name, (low, high) in VEHICLE_RANGES.items():
        assert low <= vehicle[name] <= high, name


def test_different_seeds_give_different_vehicles():
    assert random_vehicle(seed=1) != random_vehicle(seed=2)


def test_the_same_seed_gives_the_same_vehicle():
    assert random_vehicle(seed=7) == random_vehicle(seed=7)


# -- one episode ----------------------------------------------------------


def test_an_episode_reports_how_well_it_drove():
    result = drive_episode(DrivingPolicy.zero(), seed=1, seconds=20.0)
    assert isinstance(result, Episode)
    assert result.score <= 0.0


def test_a_policy_that_does_nothing_covers_no_ground():
    # Not distance_m: that is progress *along the course*, and an episode starts
    # at a random lateral offset, so a stationary car still projects onto a
    # small arc length. What is actually zero is the speed.
    episode = drive_episode(DrivingPolicy.zero(), seed=1, seconds=20.0)
    assert episode.mean_driving_speed_mps == 0.0
    assert episode.distance_m < 1.0


def test_an_episode_is_deterministic():
    first = drive_episode(DrivingPolicy.random(seed=2), seed=5, seconds=15.0)
    second = drive_episode(DrivingPolicy.random(seed=2), seed=5, seconds=15.0)
    assert first.score == second.score


def test_leaving_the_road_ends_the_episode_early():
    # Full lock one way plus full throttle: it spirals off whatever it sees.
    reckless = DrivingPolicy((0, 0, 0, 0, 0, 0, 5.0), (0, 0, 0, 0, 0, 0, 3.0))
    result = drive_episode(reckless, seed=1, seconds=60.0)
    assert result.left_the_road
    assert result.distance_m < 100.0


def test_an_episode_reports_the_speed_it_was_asked_for():
    result = drive_episode(DrivingPolicy.zero(), seed=3, seconds=10.0)
    assert 1.0 <= result.target_speed_mps <= 30.0


# -- learning -------------------------------------------------------------


def _known_good_policy():
    """Hand-written weights that track a line: the bar learning must clear."""
    steering = [0.0, 0.9, 1.4, 0.0, 0.0, 0.0, 0.0]
    pedal = [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    return DrivingPolicy(tuple(steering), tuple(pedal))


@pytest.mark.slow
def test_learning_beats_doing_nothing(trained):
    # Measured over 40 episodes: about -1.4 learned against -10.8 for a policy
    # that never moves, and 168 m driven against none.
    assert _mean_score(trained) > _mean_score(DrivingPolicy.zero()) + 5.0


pytestmark_slow = pytest.mark.slow


@pytest.fixture(scope="module")
def trained():
    """One real training run, shared by the tests that judge its quality.

    Training is the slow part of this suite, and every quality claim below is
    about the same policy -- which is also a fairer test than seed-hunting a
    fresh one for each assertion.
    """
    # Restarts, not a single run: measured across seeds at a fixed budget one
    # run left the road on 0 of 16 unseen vehicles and another on 16 of 16, and
    # more iterations did not rescue the bad one.
    policy, _ = learn_with_restarts(restarts=3, iterations=20, population=32,
                                    seconds=25.0, episodes_per_candidate=6, seed=2)
    return policy


@pytest.mark.slow
def test_learning_produces_a_policy_that_actually_follows_the_line(trained):
    results = [drive_episode(trained, seed=s, seconds=25.0) for s in range(20, 26)]
    assert sum(r.left_the_road for r in results) == 0
    assert max(r.mean_cross_track_m for r in results) < 2.0


def test_learning_reports_its_progress():
    _, history = learn_to_drive(iterations=3, population=10, seconds=15.0,
                                episodes_per_candidate=1, seed=1)
    assert len(history) == 3
    assert all("best" in entry and "mean" in entry for entry in history)


@pytest.mark.slow
def test_learning_is_reproducible():
    first, _ = learn_to_drive(iterations=3, population=10, seconds=15.0,
                              episodes_per_candidate=1, seed=4)
    second, _ = learn_to_drive(iterations=3, population=10, seconds=15.0,
                               episodes_per_candidate=1, seed=4)
    assert first == second


@pytest.mark.slow
def test_a_learned_policy_handles_a_vehicle_it_never_saw(trained):
    # Trained on randomised vehicles, judged on seeds never used in training.
    held_out = [drive_episode(trained, seed=s, seconds=25.0) for s in range(500, 516)]
    assert sum(r.left_the_road for r in held_out) == 0
    assert max(r.mean_cross_track_m for r in held_out) < 3.0


@pytest.mark.slow
def test_a_learned_policy_handles_both_crawling_and_open_road(trained):
    # One policy, both ends of the speed range, on unseen vehicles.
    for target in (4.0, 12.0, 22.0):
        runs = [
            drive_episode(trained, seed=s, seconds=30.0, target_speed_mps=target)
            for s in range(600, 606)
        ]
        assert sum(r.left_the_road for r in runs) == 0, target
        mean_error = sum(r.mean_speed_error_mps for r in runs) / len(runs)
        assert mean_error < 8.0, (target, mean_error)


@pytest.mark.slow
def test_learning_removes_the_need_to_configure_the_vehicle(trained):
    # The point of the exercise: the extremes of the vehicle range, which no
    # single hand-tuned parameter set covers, driven by one policy.
    from control.learning import VEHICLE_RANGES

    assert len(VEHICLE_RANGES) == 5
    results = [drive_episode(trained, seed=s, seconds=25.0) for s in range(700, 716)]
    assert sum(r.left_the_road for r in results) <= 1


#: Four episodes is not enough to separate two policies here: the scores
#: overlap and the comparison flips on noise. Forty settles it.
COMPARISON_SEEDS = range(300, 340)


def _mean_score(policy):
    return sum(
        drive_episode(policy, seed=s, seconds=25.0).score
        for s in COMPARISON_SEEDS
    ) / len(COMPARISON_SEEDS)


# -- the training environment itself --------------------------------------

from control.learning import _course  # noqa: E402


@pytest.mark.parametrize("seed", range(500, 512))
def test_the_course_starts_where_the_vehicle_starts(seed):
    # A random phase once put the course start up to 30 m away, so the policy
    # was scored as off-road before it had done anything. Every policy scored
    # identically because none of them was the problem.
    assert abs(_course(seed).points[0][1]) < 0.01


def test_a_sensible_policy_stays_on_the_road():
    # If a hand-reasoned policy cannot pass this, the environment is broken,
    # not the policy.
    sensible = DrivingPolicy((0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0),
                             (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    results = [drive_episode(sensible, seed=s, seconds=25.0) for s in range(500, 512)]
    assert sum(r.left_the_road for r in results) <= 1


@pytest.mark.slow
def test_restarts_do_better_than_the_worst_single_run():
    # The point of restarts: a single run can land in a local optimum that no
    # amount of extra training escapes, and picking between runs on unseen
    # vehicles avoids shipping that one.
    from control.learning import SELECTION_SEEDS

    def off_road(policy):
        return sum(
            drive_episode(policy, seed=s, seconds=25.0).left_the_road
            for s in SELECTION_SEEDS
        )

    picked, _ = learn_with_restarts(restarts=3, iterations=15, population=24,
                                    seconds=25.0, episodes_per_candidate=4, seed=11)
    single, _ = learn_to_drive(iterations=15, population=24, seconds=25.0,
                               episodes_per_candidate=4, seed=11)
    assert off_road(picked) <= off_road(single)


@pytest.mark.slow
def test_the_policy_holds_every_target_speed_once_it_is_up_to_speed(trained):
    """Speed tracking, measured where speed tracking can be measured.

    A 30 s episode with a stop in the middle spends most of its time
    accelerating from rest -- twice -- so mean speed over one is dominated by
    the ramps and reads far below target even when tracking is fine. Judged on a
    straight with time to settle, the policy holds its target at both ends of
    the range.
    """
    import statistics

    from control.learning import random_vehicle
    from control.path import Path
    from control.policy import observe
    from sim.fake import FakeBackend

    course = Path([(i * 3.0, 0.0) for i in range(1500)])
    for target in (4.0, 12.0, 22.0):
        settled = []
        for seed in range(600, 604):
            backend = FakeBackend(dt=0.05, seed=seed, **random_vehicle(seed))
            backend.reset()
            progress, previous, speeds = 0.0, 0.0, []
            for step in range(int(120 / 0.05)):
                state = backend.read_state()
                control = trained.act(
                    observe(course, state.x_m, state.y_m, state.heading_rad,
                            state.speed_mps, target, previous, progress_m=progress)
                )
                backend.apply_control(control)
                previous = control.steering
                state = backend.read_state()
                progress = course.closest_arc_length(
                    (state.x_m, state.y_m), near_arc_length=progress
                )
                if step > int(90 / 0.05):
                    speeds.append(state.speed_mps)
                if progress > course.length_m - 50:
                    break
            if speeds:
                settled.append(statistics.mean(speeds))
        held = statistics.mean(settled)
        assert abs(held - target) / target < 0.15, (target, held)
