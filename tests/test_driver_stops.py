"""Stopping at waypoints.

Stops are the point of a manoeuvre route: they produce the idle time, the
braking and the re-acceleration that an SLI battery actually ages under.
Until now `idle_fraction` was recorded in the spec and never honoured.
"""

import pytest

from behaviour.route_spec import Stop
from control.driver import Driver
from control.path import Path
from sim.fake import FakeBackend
from tests.helpers import a_behaviour_spec

DT = 0.02


def straight(length=1200.0):
    return Path([(float(i), 0.0) for i in range(int(length) + 1)])


def run(driver, backend, seconds):
    trace = []
    for _ in range(int(seconds / DT)):
        backend.apply_control(driver.step(backend.read_state()))
        state = backend.read_state()
        trace.append((state.sim_time_s, state.speed_mps, state.x_m))
        if driver.is_finished(state):
            break
    return trace


def driver_with(stops, **overrides):
    backend = FakeBackend(dt=DT)
    backend.reset()
    driver = Driver(
        a_behaviour_spec(**overrides), straight(), dt=DT,
        speed_limit_mps=20.0, stops=stops,
    )
    return driver, backend


def test_the_vehicle_stops_at_a_stop_point():
    driver, backend = driver_with([Stop(arc_length_m=200.0, duration_s=10.0)])
    trace = run(driver, backend, seconds=60.0)
    at_stop = [row for row in trace if 195.0 < row[2] < 215.0]
    assert min(row[1] for row in at_stop) < 0.5


def test_the_vehicle_waits_at_the_stop():
    driver, backend = driver_with([Stop(arc_length_m=200.0, duration_s=15.0)])
    trace = run(driver, backend, seconds=90.0)
    stationary = [row[0] for row in trace if row[1] < 0.5 and row[2] > 100.0]
    assert max(stationary) - min(stationary) == pytest.approx(15.0, abs=3.0)


def test_the_vehicle_moves_on_after_waiting():
    driver, backend = driver_with([Stop(arc_length_m=200.0, duration_s=10.0)])
    trace = run(driver, backend, seconds=120.0)
    assert trace[-1][2] > 400.0


def test_a_longer_stop_means_more_idle_time():
    def idle_seconds(duration):
        driver, backend = driver_with([Stop(arc_length_m=200.0, duration_s=duration)])
        trace = run(driver, backend, seconds=150.0)
        return sum(DT for row in trace if row[1] < 0.5 and row[2] > 100.0)

    assert idle_seconds(40.0) > idle_seconds(10.0) + 20.0


def test_every_stop_on_the_route_is_honoured():
    stops = [
        Stop(arc_length_m=150.0, duration_s=5.0),
        Stop(arc_length_m=400.0, duration_s=5.0),
        Stop(arc_length_m=650.0, duration_s=5.0),
    ]
    driver, backend = driver_with(stops)
    trace = run(driver, backend, seconds=200.0)
    for stop in stops:
        nearby = [r for r in trace if abs(r[2] - stop.arc_length_m) < 20.0]
        assert nearby and min(r[1] for r in nearby) < 0.5, stop


def test_a_route_without_stops_is_unaffected():
    driver, backend = driver_with([])
    trace = run(driver, backend, seconds=40.0)
    assert min(row[1] for row in trace[500:]) > 1.0


def test_the_vehicle_still_finishes_the_route_after_stopping():
    driver, backend = driver_with(
        [Stop(arc_length_m=200.0, duration_s=5.0)], decel_limit_mps2=4.0
    )
    trace = run(driver, backend, seconds=300.0)
    assert trace[-1][1] < 0.5
    assert trace[-1][2] > 1150.0
