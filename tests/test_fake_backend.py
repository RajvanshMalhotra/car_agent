"""The fake backend is core infrastructure: it lets everything downstream of the
sim boundary be tested on macOS with no Windows machine and no game."""

import math

import pytest

from sim.backend import ControlInput, SimBackend, VehicleState
from sim.fake import FakeBackend

STRAIGHT = ControlInput(throttle=0.0, brake=0.0, steering=0.0)
FULL_THROTTLE = ControlInput(throttle=1.0, brake=0.0, steering=0.0)


def drive(backend, control, seconds):
    for _ in range(round(seconds / backend.dt)):
        backend.apply_control(control)
    return backend.read_state()


def test_fake_backend_implements_the_sim_boundary():
    assert issubclass(FakeBackend, SimBackend)


def test_reset_puts_the_vehicle_at_rest_at_the_origin():
    state = FakeBackend().reset()
    assert (state.x_m, state.y_m, state.heading_rad) == (0.0, 0.0, 0.0)
    assert state.speed_mps == 0.0
    assert state.sim_time_s == 0.0


def test_read_state_returns_a_vehicle_state():
    backend = FakeBackend()
    backend.reset()
    assert isinstance(backend.read_state(), VehicleState)


def test_an_idle_vehicle_does_not_move():
    backend = FakeBackend()
    backend.reset()
    state = drive(backend, STRAIGHT, seconds=5.0)
    assert state.speed_mps == 0.0
    assert state.x_m == 0.0


def test_throttle_accelerates_the_vehicle_forward():
    backend = FakeBackend()
    backend.reset()
    state = drive(backend, FULL_THROTTLE, seconds=3.0)
    assert state.speed_mps > 0.0
    assert state.x_m > 0.0


def test_acceleration_is_bounded_by_the_vehicle_limit():
    backend = FakeBackend(max_accel_mps2=3.0)
    backend.reset()
    state = drive(backend, FULL_THROTTLE, seconds=2.0)
    assert state.speed_mps <= 3.0 * 2.0 + 1e-9


def test_braking_slows_the_vehicle():
    backend = FakeBackend()
    backend.reset()
    fast = drive(backend, FULL_THROTTLE, seconds=5.0).speed_mps
    slowed = drive(backend, ControlInput(0.0, 1.0, 0.0), seconds=1.0).speed_mps
    assert slowed < fast


def test_braking_never_reverses_the_vehicle():
    backend = FakeBackend()
    backend.reset()
    drive(backend, FULL_THROTTLE, seconds=2.0)
    state = drive(backend, ControlInput(0.0, 1.0, 0.0), seconds=30.0)
    assert state.speed_mps == 0.0


def test_coasting_decelerates_through_drag():
    backend = FakeBackend()
    backend.reset()
    fast = drive(backend, FULL_THROTTLE, seconds=10.0).speed_mps
    coasted = drive(backend, STRAIGHT, seconds=5.0).speed_mps
    assert coasted < fast


def test_positive_steering_turns_the_vehicle_left():
    backend = FakeBackend()
    backend.reset()
    drive(backend, FULL_THROTTLE, seconds=3.0)
    state = drive(backend, ControlInput(0.5, 0.0, 1.0), seconds=2.0)
    assert state.heading_rad > 0.0
    assert state.y_m > 0.0


def test_a_stationary_vehicle_does_not_turn_when_steered():
    # Kinematic bicycle: yaw rate is proportional to speed.
    backend = FakeBackend()
    backend.reset()
    state = drive(backend, ControlInput(0.0, 0.0, 1.0), seconds=3.0)
    assert state.heading_rad == 0.0


def test_yaw_rate_follows_the_bicycle_model():
    backend = FakeBackend(wheelbase_m=2.7, max_steer_rad=0.5)
    backend.reset()
    backend.state.speed_mps = 10.0
    backend.apply_control(ControlInput(0.0, 0.0, 1.0))
    expected = 10.0 / 2.7 * math.tan(0.5) * backend.dt
    assert backend.read_state().heading_rad == pytest.approx(expected, rel=1e-6)


def test_sim_time_advances_by_one_timestep_per_control():
    backend = FakeBackend(dt=0.05)
    backend.reset()
    backend.apply_control(STRAIGHT)
    backend.apply_control(STRAIGHT)
    assert backend.read_state().sim_time_s == pytest.approx(0.1)


def test_reset_clears_a_driven_vehicle():
    backend = FakeBackend()
    backend.reset()
    drive(backend, FULL_THROTTLE, seconds=5.0)
    state = backend.reset()
    assert state.speed_mps == 0.0
    assert state.x_m == 0.0
    assert state.sim_time_s == 0.0


def test_control_rejects_out_of_range_throttle():
    with pytest.raises(ValueError, match="throttle"):
        ControlInput(throttle=1.5, brake=0.0, steering=0.0)


def test_control_rejects_out_of_range_steering():
    with pytest.raises(ValueError, match="steering"):
        ControlInput(throttle=0.0, brake=0.0, steering=-2.0)


def test_the_same_seed_reproduces_the_same_trace():
    def trace(seed):
        backend = FakeBackend(seed=seed)
        backend.reset()
        return [drive(backend, FULL_THROTTLE, 0.5).speed_mps for _ in range(10)]

    assert trace(7) == trace(7)


def test_close_is_idempotent():
    backend = FakeBackend()
    backend.reset()
    backend.close()
    backend.close()


def test_read_state_returns_a_snapshot_not_a_live_reference():
    # A caller that holds a state to compare against the next one must not see
    # it mutate underneath them.
    backend = FakeBackend()
    backend.reset()
    before = backend.read_state()
    drive(backend, FULL_THROTTLE, seconds=1.0)
    assert before.speed_mps == 0.0
    assert before.sim_time_s == 0.0


def test_reset_cranks_the_engine():
    state = FakeBackend().reset()
    assert state.engine_on
    assert state.crank_count == 1


def test_the_state_carries_the_engine_channels():
    backend = FakeBackend(ambient_temp_c=18.0)
    state = backend.reset()
    assert state.coolant_temp_c == 18.0
    assert state.underbonnet_temp_c == 18.0
    assert state.rpm == 0.0


def test_an_idling_engine_turns_at_idle_speed():
    backend = FakeBackend()
    backend.reset()
    state = drive(backend, STRAIGHT, seconds=1.0)
    assert state.rpm == pytest.approx(800.0)


def test_driving_warms_the_coolant():
    backend = FakeBackend(ambient_temp_c=10.0)
    backend.reset()
    state = drive(backend, FULL_THROTTLE, seconds=600.0)
    assert state.coolant_temp_c > 80.0


def test_the_engine_bay_runs_above_ambient_once_warm():
    backend = FakeBackend(ambient_temp_c=30.0)
    backend.reset()
    state = drive(backend, ControlInput(0.3, 0.0, 0.0), seconds=900.0)
    assert state.underbonnet_temp_c > 40.0


def test_the_ambient_temperature_reaches_the_engine_bay():
    def bay_after(ambient_c):
        backend = FakeBackend(ambient_temp_c=ambient_c)
        backend.reset()
        return drive(backend, ControlInput(0.3, 0.0, 0.0), seconds=900.0).underbonnet_temp_c

    assert bay_after(45.0) > bay_after(5.0) + 20.0
