"""The behaviour agent's executor: a BehaviourSpec driven by a classical
controller. No LLM is in this loop -- the spec arrives already validated."""

import math
import statistics

import pytest

from behaviour.spec import BehaviourSpec
from control.driver import Driver
from control.path import Path
from sim.fake import FakeBackend

DT = 0.02


def spec(**overrides):
    fields = dict(
        name="test",
        target_speed_factor=1.0,
        accel_limit_mps2=2.0,
        decel_limit_mps2=3.0,
        jerk_limit_mps3=4.0,
        following_distance_s=1.8,
        corner_speed_factor=0.9,
        reaction_lag_s=0.0,
        erraticness=0.0,
        trip_duration_s=1200.0,
        idle_fraction=0.0,
        hvac_setting=0.5,
        ambient_temp_c=22.0,
        cold_start=True,
        start_stop_enabled=False,
    )
    fields.update(overrides)
    return BehaviourSpec(**fields)


def straight_path(length=4000.0):
    return Path([(float(i), 0.0) for i in range(int(length) + 1)])


def arc_path(radius=40.0, sweep_rad=4.0, points=1600):
    # Sweep must stay under 2*pi: a path that retraces the same circle is
    # self-overlapping, and arc-length lookup on it is genuinely ambiguous.
    return Path(
        [
            (
                radius * math.sin(sweep_rad * i / points),
                radius * (1 - math.cos(sweep_rad * i / points)),
            )
            for i in range(points + 1)
        ]
    )


def run(behaviour, path, seconds=60.0, speed_limit_mps=20.0, start=None, seed=0):
    backend = FakeBackend(dt=DT)
    backend.reset()
    if start is not None:
        backend.state.x_m, backend.state.y_m = start
    driver = Driver(
        behaviour, path, dt=DT, speed_limit_mps=speed_limit_mps, seed=seed
    )
    trace = []
    for _ in range(int(seconds / DT)):
        backend.apply_control(driver.step(backend.read_state()))
        state = backend.read_state()
        trace.append((state.sim_time_s, state.speed_mps, state.x_m, state.y_m))
        if driver.is_finished(state):
            break
    return trace


def speeds(trace):
    return [row[1] for row in trace]


def settled_speed(trace, tail=0.25):
    tail_speeds = speeds(trace)[-int(len(trace) * tail) :]
    return statistics.mean(tail_speeds)


def cruise_speed(trace):
    """Mean speed over the middle of the run, clear of the setting-off ramp and
    the end-of-route stop."""
    values = speeds(trace)
    return statistics.mean(values[int(len(values) * 0.3) : int(len(values) * 0.7)])


def accelerations(trace):
    return [
        (b[1] - a[1]) / (b[0] - a[0])
        for a, b in zip(trace, trace[1:])
        if b[0] > a[0]
    ]


def test_the_vehicle_sets_off_from_rest():
    assert speeds(run(spec(), straight_path(), seconds=5.0))[-1] > 1.0


def test_speed_converges_on_the_behaviour_target():
    trace = run(spec(target_speed_factor=0.8), straight_path(), seconds=60.0)
    assert settled_speed(trace) == pytest.approx(16.0, rel=0.08)


def test_a_faster_behaviour_settles_at_a_higher_speed():
    calm = settled_speed(run(spec(target_speed_factor=0.7), straight_path()))
    brisk = settled_speed(run(spec(target_speed_factor=1.1), straight_path()))
    assert brisk > calm + 5.0


def test_acceleration_stays_within_the_behaviour_limit():
    trace = run(spec(accel_limit_mps2=1.5), straight_path(), seconds=40.0)
    assert max(accelerations(trace)) <= 1.5 + 0.05


def test_a_gentle_behaviour_takes_longer_to_reach_speed():
    def time_to_15(behaviour):
        for t, v, *_ in run(behaviour, straight_path(), seconds=60.0):
            if v >= 15.0:
                return t
        return math.inf

    assert time_to_15(spec(accel_limit_mps2=1.0)) > time_to_15(
        spec(accel_limit_mps2=3.0)
    )


def test_jerk_stays_within_the_behaviour_limit():
    trace = run(spec(jerk_limit_mps3=2.0), straight_path(), seconds=40.0)
    accels = accelerations(trace)
    jerks = [abs(b - a) / DT for a, b in zip(accels, accels[1:])]
    assert max(jerks) <= 2.0 + 0.2


def test_a_straight_path_is_tracked_with_negligible_lateral_error():
    trace = run(spec(), straight_path(), seconds=40.0)
    assert max(abs(row[3]) for row in trace) < 0.1


def test_a_lateral_offset_is_steered_out():
    trace = run(spec(), straight_path(), seconds=40.0, start=(0.0, 2.0))
    assert abs(trace[-1][3]) < 0.3


def tracking_errors(trace, path, after_s=0.0):
    """Distance from the path, measured with monotonic progress so a lookup
    never jumps to a different part of a curved route.

    Progress accumulates over the whole trace; `after_s` only filters what is
    collected, so the hint is never seeded from the wrong place.
    """
    errors = []
    progress = 0.0
    for t, _, x, y in trace:
        progress = path.closest_arc_length((x, y), near_arc_length=progress)
        if progress >= path.length_m - 5.0:
            break  # past the route end there is no path left to track
        if t >= after_s:
            nearest_x, nearest_y = path.point_at(progress)
            errors.append(math.hypot(nearest_x - x, nearest_y - y))
    return errors


def test_a_curve_is_tracked_within_a_lane_width():
    path = arc_path()
    assert max(tracking_errors(run(spec(), path, seconds=60.0), path)) < 1.5


def test_corners_are_taken_more_slowly_than_straights():
    straight = cruise_speed(run(spec(), straight_path(), seconds=60.0))
    curved = cruise_speed(run(spec(), arc_path(radius=25.0), seconds=60.0))
    assert curved < straight


def test_a_timid_corner_behaviour_slows_more_than_a_confident_one():
    timid = cruise_speed(run(spec(corner_speed_factor=0.5), arc_path(), seconds=60.0))
    confident = cruise_speed(
        run(spec(corner_speed_factor=1.1), arc_path(), seconds=60.0)
    )
    assert timid < confident


def test_the_vehicle_stops_at_the_end_of_the_route():
    trace = run(spec(), straight_path(length=120.0), seconds=120.0)
    assert speeds(trace)[-1] < 0.5


def test_the_vehicle_stops_short_of_the_route_end_rather_than_overrunning_it():
    trace = run(spec(), straight_path(length=120.0), seconds=120.0)
    assert 117.0 < trace[-1][2] <= 120.5


def test_the_stopping_point_does_not_depend_on_the_deceleration_limit():
    def stop_x(decel):
        return run(
            spec(decel_limit_mps2=decel), straight_path(length=120.0), seconds=120.0
        )[-1][2]

    assert stop_x(2.0) == pytest.approx(stop_x(6.0), abs=1.0)


def test_reaction_lag_makes_the_driver_overshoot_the_target_speed():
    # Dead time in the loop shows up as overshoot, not as sluggishness: during
    # the ramp the driver is acceleration-limited either way.
    prompt = max(speeds(run(spec(reaction_lag_s=0.0), straight_path())))
    sluggish = max(speeds(run(spec(reaction_lag_s=1.5), straight_path())))
    assert sluggish > prompt + 0.5


def test_erraticness_makes_the_speed_wander():
    steady = speeds(run(spec(erraticness=0.0), straight_path(), seconds=60.0))[-1500:]
    erratic = speeds(run(spec(erraticness=0.8), straight_path(), seconds=60.0))[-1500:]
    assert statistics.pstdev(erratic) > statistics.pstdev(steady) + 0.5


def test_erraticness_stays_bounded_and_never_reverses_the_vehicle():
    trace = run(spec(erraticness=1.0), straight_path(), seconds=60.0)
    assert min(speeds(trace)) >= 0.0
    assert max(speeds(trace)) < 30.0


def test_the_same_seed_reproduces_the_same_drive():
    behaviour = spec(erraticness=0.6)
    assert speeds(run(behaviour, straight_path(), seed=42)) == speeds(
        run(behaviour, straight_path(), seed=42)
    )


def test_different_seeds_produce_different_drives():
    behaviour = spec(erraticness=0.6)
    assert speeds(run(behaviour, straight_path(), seed=1)) != speeds(
        run(behaviour, straight_path(), seed=2)
    )


def winding_path():
    return Path([(i * 0.5, 25 * math.sin(i * 0.5 / 60)) for i in range(2400)])


def max_tracking_error(behaviour, path, speed_limit_mps=22.0):
    trace = run(behaviour, path, seconds=900.0, speed_limit_mps=speed_limit_mps)
    return max(tracking_errors(trace, path, after_s=10.0))


@pytest.mark.parametrize("lag", [0.0, 0.3, 0.5, 1.0, 1.5, 2.0])
def test_lane_keeping_is_stable_across_the_whole_reaction_lag_range(lag):
    # Reaction time governs response to events, not the continuous compensatory
    # tracking that keeps a car in its lane. Feeding the full lag into the
    # steering loop makes it oscillate and then diverge.
    assert max_tracking_error(spec(reaction_lag_s=lag), winding_path()) < 1.0


def test_a_slow_reacting_driver_still_tracks_a_bend():
    assert max_tracking_error(spec(reaction_lag_s=2.0), arc_path()) < 1.5
