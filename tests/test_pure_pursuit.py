"""Geometric steering: aim at a point one lookahead ahead on the path."""

import math

import pytest

from control.pure_pursuit import PurePursuit


def test_a_target_dead_ahead_needs_no_steering():
    controller = PurePursuit(wheelbase_m=2.7, max_steer_rad=0.5)
    steering = controller.steering(pose=(0.0, 0.0, 0.0), target=(10.0, 0.0))
    assert steering == pytest.approx(0.0)


def test_a_target_to_the_left_steers_left():
    controller = PurePursuit(wheelbase_m=2.7, max_steer_rad=0.5)
    assert controller.steering(pose=(0.0, 0.0, 0.0), target=(10.0, 5.0)) > 0.0


def test_a_target_to_the_right_steers_right():
    controller = PurePursuit(wheelbase_m=2.7, max_steer_rad=0.5)
    assert controller.steering(pose=(0.0, 0.0, 0.0), target=(10.0, -5.0)) < 0.0


def test_steering_accounts_for_vehicle_heading():
    # Target is due north; the vehicle already points north, so drive straight.
    controller = PurePursuit(wheelbase_m=2.7, max_steer_rad=0.5)
    steering = controller.steering(pose=(0.0, 0.0, math.pi / 2), target=(0.0, 10.0))
    assert steering == pytest.approx(0.0)


def test_steer_angle_follows_the_pure_pursuit_law():
    controller = PurePursuit(wheelbase_m=2.7, max_steer_rad=0.5)
    target = (8.0, 6.0)  # 10 m away, 36.87 degrees off axis
    lookahead = math.hypot(*target)
    alpha = math.atan2(target[1], target[0])
    expected = math.atan2(2 * 2.7 * math.sin(alpha), lookahead)
    steering = controller.steering(pose=(0.0, 0.0, 0.0), target=target)
    assert steering * 0.5 == pytest.approx(expected)


def test_steering_saturates_at_the_lock_limit():
    controller = PurePursuit(wheelbase_m=2.7, max_steer_rad=0.5)
    steering = controller.steering(pose=(0.0, 0.0, 0.0), target=(0.5, 4.0))
    assert steering == pytest.approx(1.0)


def test_lookahead_grows_with_speed():
    controller = PurePursuit(wheelbase_m=2.7, max_steer_rad=0.5)
    assert controller.lookahead_m(speed_mps=25.0) > controller.lookahead_m(speed_mps=5.0)


def test_lookahead_has_a_floor_so_a_stopped_vehicle_still_aims_somewhere():
    controller = PurePursuit(wheelbase_m=2.7, max_steer_rad=0.5, min_lookahead_m=4.0)
    assert controller.lookahead_m(speed_mps=0.0) == pytest.approx(4.0)
