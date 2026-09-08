"""Driving with a learned policy.

Same interface as the hand-built Driver, so the run loop, the logging and the
data collection are unchanged. What changes is that nothing about the vehicle
is configured: no wheelbase, no acceleration limit, no calibration.
"""

import math

import pytest

from control.policy import DrivingPolicy
from control.policy_driver import PolicyDriver
from control.path import Path
from sim.fake import FakeBackend
from tests.helpers import a_behaviour_spec

DT = 0.05

# Weights of the shape learning arrives at: steer on cross-track and heading,
# pedal on speed error.
TRAINED = DrivingPolicy(
    (0.0232, 1.0753, 1.2097, 0.3299, 0.9848, -0.5012, -0.1761),
    (1.6654, 0.39, -0.8881, -0.1963, -1.3603, -1.098, 0.3344),
)


def straight(length=1500.0):
    return Path([(float(i), 0.0) for i in range(int(length))])


def run(driver, backend, seconds):
    trace = []
    for _ in range(int(seconds / DT)):
        backend.apply_control(driver.step(backend.read_state()))
        state = backend.read_state()
        trace.append((state.sim_time_s, state.speed_mps, state.x_m, state.y_m))
        if driver.is_finished(state):
            break
    return trace


def a_driver(path=None, **overrides):
    backend = FakeBackend(dt=DT, wheelbase_m=6.5, max_accel_mps2=1.2,
                          max_decel_mps2=4.0)
    backend.reset()
    driver = PolicyDriver(
        TRAINED, a_behaviour_spec(**overrides), path or straight(),
        dt=DT, speed_limit_mps=15.0,
    )
    return driver, backend


def test_it_produces_a_control_from_a_state():
    driver, backend = a_driver()
    control = driver.step(backend.read_state())
    assert 0.0 <= control.throttle <= 1.0
    assert -1.0 <= control.steering <= 1.0


def test_it_needs_nothing_configured_about_the_vehicle():
    # The whole point: no wheelbase, no accel limit, no calibration.
    import inspect

    parameters = set(inspect.signature(PolicyDriver.__init__).parameters)
    assert not parameters & {
        "wheelbase_m", "max_steer_rad", "max_accel_mps2", "max_decel_mps2",
    }


def test_it_drives_the_route():
    driver, backend = a_driver()
    trace = run(driver, backend, seconds=60.0)
    assert trace[-1][2] > 300.0


def test_it_tracks_the_line():
    driver, backend = a_driver()
    trace = run(driver, backend, seconds=60.0)
    assert max(abs(row[3]) for row in trace[100:]) < 3.0


def test_it_honours_the_behaviour_target_speed():
    slow, backend = a_driver(target_speed_factor=0.5)
    slow_trace = run(slow, backend, seconds=60.0)
    fast, backend = a_driver(target_speed_factor=1.3)
    fast_trace = run(fast, backend, seconds=60.0)
    mean = lambda t: sum(r[1] for r in t[200:]) / max(1, len(t[200:]))
    assert mean(fast_trace) > mean(slow_trace) + 2.0


def test_it_reports_how_far_off_the_route_it_is():
    driver, backend = a_driver()
    driver.step(backend.read_state())
    backend.state.y_m = 4.0
    assert driver.deviation_m(backend.read_state()) == pytest.approx(4.0, abs=0.2)


def test_it_knows_when_it_has_lost_the_route():
    driver, backend = a_driver()
    while driver.travelled_m < driver.lost_grace_m + 20.0:
        backend.apply_control(driver.step(backend.read_state()))
    backend.state.y_m = 40.0
    assert driver.is_lost(backend.read_state())


def test_it_knows_when_the_route_is_done():
    driver, backend = a_driver(path=straight(120.0))
    run(driver, backend, seconds=120.0)
    assert driver.is_finished(backend.read_state())


def test_it_stops_at_a_waypoint():
    from behaviour.route_spec import Stop

    backend = FakeBackend(dt=DT)
    backend.reset()
    driver = PolicyDriver(
        TRAINED, a_behaviour_spec(), straight(), dt=DT, speed_limit_mps=15.0,
        stops=[Stop(arc_length_m=200.0, duration_s=10.0)],
    )
    trace = run(driver, backend, seconds=120.0)
    at_stop = [row for row in trace if 185.0 < row[2] < 235.0]
    assert at_stop and min(row[1] for row in at_stop) < 1.0


def test_the_route_can_be_replaced_after_a_recovery():
    # After the game recovers the car to a road it is somewhere else entirely,
    # so the route is re-laid from wherever it ended up.
    driver, backend = a_driver()
    run(driver, backend, seconds=20.0)
    driver.restart(Path([(500.0, 500.0), (600.0, 500.0), (700.0, 500.0)]))
    backend.state.x_m, backend.state.y_m = 500.0, 500.0
    assert not driver.is_lost(backend.read_state())
    assert driver.deviation_m(backend.read_state()) == pytest.approx(0.0, abs=0.5)


# -- driving a demonstration ----------------------------------------------


def demo_driver(speed=10.0, factor=1.0):
    from control.demonstration import Demonstration

    demo = Demonstration.from_samples(
        [(i * 2.0, 0.0, speed) for i in range(400)], name="lap"
    )
    backend = FakeBackend(dt=DT, wheelbase_m=6.5, max_accel_mps2=1.2)
    backend.reset()
    driver = PolicyDriver(
        TRAINED, a_behaviour_spec(target_speed_factor=factor), demo.to_path(),
        dt=DT, speed_limit_mps=99.0, stops=demo.stops(), demonstration=demo,
    )
    return driver, backend, demo


def test_the_demonstrated_speed_becomes_the_target():
    driver, backend, _ = demo_driver(speed=9.0)
    driver.progress_m = 100.0
    assert driver.target_speed_mps(backend.read_state()) == pytest.approx(9.0, rel=0.1)


def test_an_aggressive_behaviour_drives_the_demonstration_faster():
    calm, backend, _ = demo_driver(speed=9.0, factor=0.7)
    brisk, _, _ = demo_driver(speed=9.0, factor=1.3)
    calm.progress_m = brisk.progress_m = 100.0
    state = backend.read_state()
    assert brisk.target_speed_mps(state) > calm.target_speed_mps(state) + 3.0


def test_the_speed_limit_no_longer_sets_the_pace():
    # The human's own drive does. That is the point of watching them.
    driver, backend, _ = demo_driver(speed=7.0)
    driver.progress_m = 100.0
    assert driver.target_speed_mps(backend.read_state()) < 20.0


def test_it_actually_drives_the_demonstrated_route():
    driver, backend, demo = demo_driver(speed=8.0)
    for _ in range(int(120 / DT)):
        backend.apply_control(driver.step(backend.read_state()))
        if driver.is_finished(backend.read_state()):
            break
    assert driver.progress_m > 300.0
    assert abs(backend.read_state().y_m) < 2.0


# -- room to turn onto the line -------------------------------------------


def test_it_is_not_judged_lost_before_it_has_had_room_to_converge():
    # Same reasoning as the hand-built driver: a car put on the start of a route
    # facing the wrong way needs road to turn onto the line. Judging it in the
    # first few metres aborts runs that would have recovered.
    driver, backend = a_driver()
    backend.state.y_m = 6.0
    driver.step(backend.read_state())
    assert not driver.is_lost(backend.read_state())


def test_it_holds_its_line_against_a_steady_sideways_drag():
    # 10 m/s of sideways drag settles at about 4 m of offset: the policy
    # counteracts it rather than being pushed off the road. Worth pinning,
    # because it is easy to assume the opposite and write a test that fails.
    driver, backend = a_driver()
    while driver.travelled_m < driver.lost_grace_m + 20.0:
        backend.apply_control(driver.step(backend.read_state()))
    for _ in range(int(30 / DT)):
        backend.apply_control(driver.step(backend.read_state()))
        backend.state.y_m += 0.5
    assert driver.deviation_m(backend.read_state()) < 6.0
    assert not driver.is_lost(backend.read_state())


def test_the_grace_distance_is_configurable():
    from control.policy_driver import PolicyDriver

    backend = FakeBackend(dt=DT)
    backend.reset()
    driver = PolicyDriver(
        TRAINED, a_behaviour_spec(), straight(), dt=DT, speed_limit_mps=15.0,
        lost_grace_m=0.0,
    )
    driver.step(backend.read_state())
    backend.state.y_m = 40.0
    assert driver.is_lost(backend.read_state())


def test_restarting_gives_it_room_again():
    # After a recovery the car is somewhere new and has to turn onto the route
    # afresh, so the grace has to come back with it.
    driver, backend = a_driver()
    for _ in range(int(30 / DT)):
        backend.apply_control(driver.step(backend.read_state()))
    driver.restart(straight())
    backend.state.y_m = 6.0
    assert not driver.is_lost(backend.read_state())


def test_rejoining_mid_route_finds_where_it_actually_is():
    # After a recovery the car rejoins partway along, not at the start.
    # Assuming progress is zero makes the windowed lookup search the wrong
    # stretch of road, and the driver behaves as though the route is elsewhere.
    driver, backend = a_driver(path=straight(1500.0))
    backend.state.x_m, backend.state.y_m = 900.0, 0.0
    driver.restart(straight(1500.0))
    driver.step(backend.read_state())
    assert driver.progress_m == pytest.approx(900.0, abs=5.0)


def test_it_keeps_driving_after_rejoining_mid_route():
    driver, backend = a_driver(path=straight(1500.0))
    backend.state.x_m, backend.state.y_m = 900.0, 0.0
    driver.restart(straight(1500.0))
    for _ in range(int(30 / DT)):
        backend.apply_control(driver.step(backend.read_state()))
    assert backend.read_state().speed_mps > 2.0
    assert backend.read_state().x_m > 930.0
