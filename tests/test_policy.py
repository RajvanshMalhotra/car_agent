"""A learned driving policy.

Instead of measuring a vehicle and hand-tuning a controller for it, the policy
learns the mapping from what it can see -- speed error, how far off the line it
is, which way the line goes -- to the three controls.

Nothing about the vehicle is hardcoded. No wheelbase, no acceleration limit, no
calibration step that can be rejected. The policy is trained across randomised
vehicles, so one policy drives a hatchback and a 6x6 truck.
"""

import math

import pytest

from control.policy import FEATURE_COUNT, FEATURE_NAMES, DrivingPolicy, observe
from control.path import Path


def straight(length=1000.0):
    return Path([(float(i), 0.0) for i in range(int(length))])


# -- what the policy sees -------------------------------------------------


def test_the_observation_has_the_expected_width():
    assert len(observe(straight(), x=0.0, y=0.0, heading=0.0, speed=10.0,
                       target_speed=10.0, previous_steering=0.0)) == FEATURE_COUNT


def test_being_slow_shows_as_positive_speed_error():
    slow = observe(straight(), 0.0, 0.0, 0.0, speed=5.0, target_speed=15.0,
                   previous_steering=0.0)
    fast = observe(straight(), 0.0, 0.0, 0.0, speed=20.0, target_speed=15.0,
                   previous_steering=0.0)
    assert slow[0] > 0.0 > fast[0]


def test_being_left_of_the_line_shows_as_a_signed_offset():
    left = observe(straight(), 0.0, 3.0, 0.0, 10.0, 10.0, 0.0)
    right = observe(straight(), 0.0, -3.0, 0.0, 10.0, 10.0, 0.0)
    assert left[1] * right[1] < 0.0


def test_pointing_away_from_the_line_shows_as_heading_error():
    aligned = observe(straight(), 0.0, 0.0, 0.0, 10.0, 10.0, 0.0)
    askew = observe(straight(), 0.0, 0.0, 0.6, 10.0, 10.0, 0.0)
    assert abs(askew[2]) > abs(aligned[2])


def test_heading_error_wraps_the_short_way_round():
    # Pointing 179 degrees off is a large error either way, not a small one.
    askew = observe(straight(), 0.0, 0.0, math.pi - 0.05, 10.0, 10.0, 0.0)
    assert abs(askew[2]) > 1.0


def test_a_bend_ahead_shows_as_curvature():
    circle = Path([(50 * math.sin(t / 100), 50 * (1 - math.cos(t / 100)))
                   for t in range(600)])
    on_a_bend = observe(circle, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0)
    on_a_straight = observe(straight(), 0.0, 0.0, 0.0, 10.0, 10.0, 0.0)
    assert abs(on_a_bend[4]) > abs(on_a_straight[4])


def test_observations_are_bounded_however_wild_the_situation():
    # A policy that sees an unbounded input produces an unbounded output.
    wild = observe(straight(), 0.0, 5000.0, 12.0, 200.0, 0.0, 1.0)
    assert all(-5.0 <= value <= 5.0 for value in wild), wild


# -- the policy itself ----------------------------------------------------


def test_a_policy_produces_a_valid_control():
    control = DrivingPolicy.zero().act(
        observe(straight(), 0.0, 0.0, 0.0, 10.0, 10.0, 0.0)
    )
    assert 0.0 <= control.throttle <= 1.0
    assert 0.0 <= control.brake <= 1.0
    assert -1.0 <= control.steering <= 1.0


def test_throttle_and_brake_are_never_both_applied():
    policy = DrivingPolicy.random(seed=3)
    for speed in (0.0, 5.0, 15.0, 30.0):
        control = policy.act(observe(straight(), 0.0, 0.0, 0.0, speed, 12.0, 0.0))
        assert control.throttle == 0.0 or control.brake == 0.0


def test_a_zero_policy_does_nothing():
    control = DrivingPolicy.zero().act([0.0] * FEATURE_COUNT)
    assert control.throttle == 0.0 and control.brake == 0.0
    assert control.steering == 0.0


def test_the_policy_is_deterministic():
    policy = DrivingPolicy.random(seed=5)
    features = observe(straight(), 0.0, 1.0, 0.1, 8.0, 12.0, 0.0)
    assert policy.act(features) == policy.act(features)


def test_the_same_seed_makes_the_same_policy():
    assert DrivingPolicy.random(seed=9) == DrivingPolicy.random(seed=9)


def test_different_seeds_make_different_policies():
    assert DrivingPolicy.random(seed=1) != DrivingPolicy.random(seed=2)


def test_a_policy_round_trips_through_a_dict():
    policy = DrivingPolicy.random(seed=4)
    assert DrivingPolicy.from_dict(policy.to_dict()) == policy


def test_a_policy_survives_being_saved_and_loaded(tmp_path):
    policy = DrivingPolicy.random(seed=6)
    policy.save(tmp_path / "p.json")
    assert DrivingPolicy.load(tmp_path / "p.json") == policy


def test_the_saved_policy_records_what_it_was_trained_on(tmp_path):
    import json

    policy = DrivingPolicy.random(seed=6)
    policy.save(tmp_path / "p.json", metadata={"episodes": 400, "score": -0.42})
    stored = json.loads((tmp_path / "p.json").read_text())
    assert stored["metadata"]["episodes"] == 400


# -- which features reach which output ------------------------------------

from control.policy import PEDAL_MASK, STEERING_MASK  # noqa: E402


def test_the_pedals_cannot_see_heading_error():
    # Free to use it, the search learns to brake whenever the car points
    # off-line -- which deadlocks at a standstill, because it cannot steer
    # straight without moving and cannot move because it is braking.
    assert PEDAL_MASK[FEATURE_NAMES.index("heading_error")] == 0.0
    assert PEDAL_MASK[FEATURE_NAMES.index("cross_track")] == 0.0


def test_the_pedals_can_still_see_the_bend_ahead():
    assert PEDAL_MASK[FEATURE_NAMES.index("curvature")] == 1.0
    assert PEDAL_MASK[FEATURE_NAMES.index("speed_error")] == 1.0


def test_steering_does_not_chase_the_speed_error():
    assert STEERING_MASK[FEATURE_NAMES.index("speed_error")] == 0.0


def test_a_masked_feature_cannot_change_the_pedals():
    from control.policy import DrivingPolicy

    policy = DrivingPolicy((1.0,) * FEATURE_COUNT, (1.0,) * FEATURE_COUNT)
    base = [0.0] * FEATURE_COUNT
    askew = list(base)
    askew[FEATURE_NAMES.index("heading_error")] = 2.0
    assert policy.act(base).throttle == policy.act(askew).throttle
