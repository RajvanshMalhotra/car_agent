"""Measuring the vehicle instead of assuming one.

The controller needs four numbers: how hard the car accelerates, how hard it
brakes, and how sharply it turns. Hardcoding a passenger car makes every
command mis-scaled on anything else -- and a 6x6 truck is very much anything
else.

These are measured on the vehicle itself, in about twenty seconds, and cached
per vehicle so it happens once.
"""

import json

import pytest

from control.calibration import VehicleLimits, calibrate, load_limits, save_limits
from sim.fake import FakeBackend


def test_acceleration_is_recovered():
    backend = FakeBackend(dt=0.02, max_accel_mps2=1.8)
    backend.reset()
    assert calibrate(backend, dt=0.02).max_accel_mps2 == pytest.approx(1.8, rel=0.15)


def test_a_stronger_engine_measures_stronger():
    def accel_of(truth):
        backend = FakeBackend(dt=0.02, max_accel_mps2=truth)
        backend.reset()
        return calibrate(backend, dt=0.02).max_accel_mps2

    assert accel_of(4.5) > accel_of(1.2) + 1.0


def test_braking_is_recovered():
    backend = FakeBackend(dt=0.02, max_decel_mps2=5.0)
    backend.reset()
    assert calibrate(backend, dt=0.02).max_decel_mps2 == pytest.approx(5.0, rel=0.2)


def test_the_steering_response_is_recovered_as_a_wheelbase():
    backend = FakeBackend(dt=0.02, wheelbase_m=6.0, max_steer_rad=0.5)
    backend.reset()
    limits = calibrate(backend, dt=0.02)
    # Only the ratio max_steer/wheelbase is identifiable from yaw rate, so the
    # lock is fixed and the wheelbase carries the measurement.
    ratio = limits.max_steer_rad / limits.wheelbase_m
    assert ratio == pytest.approx(0.5 / 6.0, rel=0.2)


def test_a_long_vehicle_measures_as_less_responsive():
    def ratio_of(wheelbase):
        backend = FakeBackend(dt=0.02, wheelbase_m=wheelbase)
        backend.reset()
        limits = calibrate(backend, dt=0.02)
        return limits.max_steer_rad / limits.wheelbase_m

    assert ratio_of(8.0) < ratio_of(2.5)


def test_the_controls_are_released_when_it_finishes():
    backend = FakeBackend(dt=0.02)
    backend.reset()
    calibrate(backend, dt=0.02)
    assert backend.read_state().throttle == pytest.approx(0.0)


def test_a_vehicle_that_will_not_move_is_reported():
    class Stuck(FakeBackend):
        def apply_control(self, control):
            pass

    backend = Stuck(dt=0.02)
    backend.reset()
    with pytest.raises(RuntimeError, match="did not move"):
        calibrate(backend, dt=0.02)


def test_the_measured_limits_are_physically_plausible():
    backend = FakeBackend(dt=0.02)
    backend.reset()
    limits = calibrate(backend, dt=0.02)
    assert 0.2 < limits.max_accel_mps2 < 15.0
    assert 0.5 < limits.max_decel_mps2 < 20.0
    assert 0.5 < limits.wheelbase_m < 30.0


# -- caching --------------------------------------------------------------


def test_limits_survive_a_round_trip(tmp_path):
    limits = VehicleLimits(
        vehicle="midtruck", max_accel_mps2=1.4, max_decel_mps2=4.2,
        wheelbase_m=5.5, max_steer_rad=0.5,
    )
    save_limits(tmp_path, limits)
    assert load_limits(tmp_path, "midtruck") == limits


def test_an_unmeasured_vehicle_has_no_cached_limits(tmp_path):
    assert load_limits(tmp_path, "never-seen") is None


def test_each_vehicle_gets_its_own_entry(tmp_path):
    save_limits(tmp_path, VehicleLimits("midtruck", 1.4, 4.2, 5.5, 0.5))
    save_limits(tmp_path, VehicleLimits("etk800", 3.1, 8.0, 2.7, 0.5))
    assert load_limits(tmp_path, "midtruck").max_accel_mps2 == pytest.approx(1.4)
    assert load_limits(tmp_path, "etk800").max_accel_mps2 == pytest.approx(3.1)


def test_the_cache_records_when_it_was_measured(tmp_path):
    save_limits(tmp_path, VehicleLimits("midtruck", 1.4, 4.2, 5.5, 0.5))
    stored = json.loads((tmp_path / "midtruck.json").read_text())
    assert "measured_at" in stored


# -- refusing implausible measurements ------------------------------------

from control.calibration import CalibrationError, PLAUSIBLE  # noqa: E402


def test_plausible_limits_are_accepted():
    VehicleLimits("car", 3.0, 8.0, 2.7).validate()


def test_an_impossible_acceleration_is_refused():
    # Measured 129 m/s^2 on a truck once, from a timing artefact. Driving with
    # that mis-scales every pedal command, which is exactly the wandering this
    # module exists to prevent.
    with pytest.raises(CalibrationError, match="max_accel_mps2"):
        VehicleLimits("truck", 129.8, 4.0, 6.5).validate()


def test_an_impossible_braking_rate_is_refused():
    with pytest.raises(CalibrationError, match="max_decel_mps2"):
        VehicleLimits("truck", 1.2, 479.0, 6.5).validate()


def test_a_wheelbase_at_the_clamp_is_refused():
    with pytest.raises(CalibrationError, match="wheelbase_m"):
        VehicleLimits("truck", 1.2, 4.0, 0.5).validate()


def test_the_error_reports_every_bad_value_and_the_range():
    with pytest.raises(CalibrationError) as exc:
        VehicleLimits("truck", 129.8, 479.0, 0.5).validate()
    message = str(exc.value)
    for field in PLAUSIBLE:
        assert field in message


def test_a_measurement_on_a_real_vehicle_is_plausible():
    backend = FakeBackend(dt=0.02, max_accel_mps2=1.2, max_decel_mps2=4.0,
                          wheelbase_m=6.5)
    backend.reset()
    calibrate(backend, dt=0.02).validate()


# -- real-time backends ---------------------------------------------------


class SlowBackend(FakeBackend):
    """A backend that advances less than dt per call, like one over HTTP.

    The MCP backend's clock is wall-clock and its physics runs in real time,
    while the control loop iterates as fast as the round trips allow. So each
    iteration covers far less time than `dt` assumes -- a loop that counts
    iterations finishes a fraction of the way through and measures nonsense.
    """

    def __init__(self, fraction=5, **kwargs):
        super().__init__(**kwargs)
        self.fraction = fraction
        self._pending = 0

    def apply_control(self, control):
        self._pending += 1
        if self._pending % self.fraction == 0:
            super().apply_control(control)


def test_calibration_paces_itself_by_the_clock_not_by_iterations():
    backend = SlowBackend(dt=0.02, max_accel_mps2=1.2, max_decel_mps2=4.0,
                          wheelbase_m=6.5)
    backend.reset()
    limits = calibrate(backend, dt=0.02, vehicle="slow")
    limits.validate()
    assert limits.max_accel_mps2 == pytest.approx(1.2, rel=0.25)


def test_alignment_also_paces_itself_by_the_clock():
    from control.alignment import measure_heading

    backend = SlowBackend(dt=0.02)
    backend.reset()
    x, _, heading = measure_heading(backend, dt=0.02)
    assert x >= 4.0
    assert heading == pytest.approx(0.0, abs=0.05)
