"""Longitudinal speed control. Output is a signed effort: + throttle, - brake."""

import pytest

from control.pid import PID


def test_no_effort_when_already_at_the_target():
    pid = PID(kp=0.5, ki=0.1, kd=0.0)
    assert pid.update(error=0.0, dt=0.1) == pytest.approx(0.0)


def test_positive_error_asks_for_positive_effort():
    pid = PID(kp=0.5, ki=0.0, kd=0.0)
    assert pid.update(error=1.0, dt=0.1) == pytest.approx(0.5)


def test_negative_error_asks_for_negative_effort():
    pid = PID(kp=0.5, ki=0.0, kd=0.0)
    assert pid.update(error=-4.0, dt=0.1) < 0.0


def test_integral_term_accumulates_a_standing_error():
    pid = PID(kp=0.0, ki=1.0, kd=0.0)
    first = pid.update(error=1.0, dt=0.1)
    second = pid.update(error=1.0, dt=0.1)
    assert second > first


def test_integral_is_clamped_against_windup():
    pid = PID(kp=0.0, ki=1.0, kd=0.0, integral_limit=0.5)
    for _ in range(1000):
        pid.update(error=10.0, dt=0.1)
    assert pid.update(error=10.0, dt=0.1) == pytest.approx(0.5)


def test_derivative_term_opposes_a_growing_error():
    pid = PID(kp=0.0, ki=0.0, kd=0.05)
    pid.update(error=0.0, dt=0.1)
    assert pid.update(error=1.0, dt=0.1) == pytest.approx(0.5)


def test_output_is_clamped_to_the_effort_range():
    pid = PID(kp=100.0, ki=0.0, kd=0.0)
    assert pid.update(error=50.0, dt=0.1) == pytest.approx(1.0)
    assert pid.update(error=-50.0, dt=0.1) == pytest.approx(-1.0)


def test_reset_clears_accumulated_state():
    pid = PID(kp=0.0, ki=1.0, kd=0.0)
    for _ in range(10):
        pid.update(error=1.0, dt=0.1)
    pid.reset()
    assert pid.update(error=1.0, dt=0.1) == pytest.approx(0.1)


def test_zero_dt_does_not_divide_by_zero():
    pid = PID(kp=1.0, ki=1.0, kd=1.0)
    assert pid.update(error=1.0, dt=0.0) == pytest.approx(1.0)
