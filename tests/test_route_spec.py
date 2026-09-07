"""LLM-designed manoeuvre routes.

On an empty grid there is no road to follow, so the route is whatever we invent.
That makes it a good fit for the LLM: it designs a sequence of manoeuvres --
straights, turns, stops -- and the geometry is built deterministically from
that. Same discipline as BehaviourSpec: generated once, bounds-checked in code,
cached.

Stops matter more than they look. They are what produce idle time, brake and
acceleration cycling and trip structure -- the channels the battery layer
actually needs, and the ones a constant-speed straight line never exercises.
"""

import math

import pytest

from behaviour.route_spec import RouteSpec, RouteValidationError


def a_route(segments=None, **overrides):
    fields = dict(
        name="test route",
        segments=segments
        or [
            {"type": "straight", "length_m": 100.0},
            {"type": "turn", "radius_m": 30.0, "angle_deg": 90.0},
            {"type": "straight", "length_m": 50.0},
        ],
    )
    fields.update(overrides)
    return RouteSpec(**fields)


# -- validation -----------------------------------------------------------


def test_a_valid_route_keeps_its_segments():
    assert len(a_route().segments) == 3


def test_a_route_needs_at_least_two_segments():
    with pytest.raises(RouteValidationError, match="at least two"):
        a_route(segments=[{"type": "straight", "length_m": 100.0}])


def test_an_unknown_segment_type_is_rejected():
    with pytest.raises(RouteValidationError, match="teleport"):
        a_route(segments=[{"type": "teleport"}, {"type": "straight", "length_m": 50.0}])


def test_an_impossibly_tight_turn_is_rejected():
    with pytest.raises(RouteValidationError, match="radius_m"):
        a_route(segments=[
            {"type": "turn", "radius_m": 0.5, "angle_deg": 90.0},
            {"type": "straight", "length_m": 50.0},
        ])


def test_an_absurdly_long_straight_is_rejected():
    with pytest.raises(RouteValidationError, match="length_m"):
        a_route(segments=[
            {"type": "straight", "length_m": 90000.0},
            {"type": "straight", "length_m": 50.0},
        ])


def test_a_stop_longer_than_the_bound_is_rejected():
    with pytest.raises(RouteValidationError, match="duration_s"):
        a_route(segments=[
            {"type": "stop", "duration_s": 9999.0},
            {"type": "straight", "length_m": 50.0},
        ])


def test_a_missing_field_is_reported_with_the_segment_index():
    with pytest.raises(RouteValidationError, match="segment 0"):
        a_route(segments=[{"type": "straight"}, {"type": "straight", "length_m": 50.0}])


def test_every_bad_segment_is_reported_not_just_the_first():
    with pytest.raises(RouteValidationError) as exc:
        a_route(segments=[
            {"type": "straight", "length_m": 90000.0},
            {"type": "turn", "radius_m": 0.1, "angle_deg": 90.0},
        ])
    assert "segment 0" in str(exc.value) and "segment 1" in str(exc.value)


def test_a_route_that_is_all_stops_is_rejected():
    with pytest.raises(RouteValidationError, match="never moves"):
        a_route(segments=[
            {"type": "stop", "duration_s": 10.0},
            {"type": "stop", "duration_s": 10.0},
        ])


# -- geometry -------------------------------------------------------------


def test_a_straight_route_has_the_length_asked_for():
    route = a_route(segments=[
        {"type": "straight", "length_m": 100.0},
        {"type": "straight", "length_m": 50.0},
    ])
    assert route.to_path().length_m == pytest.approx(150.0, rel=0.02)


def test_a_turn_adds_its_arc_length():
    quarter = 2 * math.pi * 30.0 / 4
    route = a_route(segments=[
        {"type": "straight", "length_m": 100.0},
        {"type": "turn", "radius_m": 30.0, "angle_deg": 90.0},
    ])
    assert route.to_path().length_m == pytest.approx(100.0 + quarter, rel=0.03)


def test_a_left_turn_goes_left():
    route = a_route(segments=[
        {"type": "straight", "length_m": 50.0},
        {"type": "turn", "radius_m": 20.0, "angle_deg": 90.0},
    ])
    assert route.to_path().points[-1][1] > 0.0


def test_a_right_turn_goes_right():
    route = a_route(segments=[
        {"type": "straight", "length_m": 50.0},
        {"type": "turn", "radius_m": 20.0, "angle_deg": -90.0},
    ])
    assert route.to_path().points[-1][1] < 0.0


def test_the_route_starts_at_the_given_origin_and_heading():
    route = a_route()
    path = route.to_path(origin=(100.0, 25.0), heading_rad=math.pi / 2)
    assert path.points[0] == pytest.approx((100.0, 25.0))
    x, y = path.points[3]
    assert y > 25.0 and x == pytest.approx(100.0, abs=0.5)


def test_a_stop_adds_no_distance():
    with_stop = a_route(segments=[
        {"type": "straight", "length_m": 100.0},
        {"type": "stop", "duration_s": 30.0},
        {"type": "straight", "length_m": 100.0},
    ]).to_path().length_m
    without = a_route(segments=[
        {"type": "straight", "length_m": 100.0},
        {"type": "straight", "length_m": 100.0},
    ]).to_path().length_m
    assert with_stop == pytest.approx(without, rel=0.02)


# -- stops ----------------------------------------------------------------


def test_stops_are_reported_with_where_and_how_long():
    route = a_route(segments=[
        {"type": "straight", "length_m": 100.0},
        {"type": "stop", "duration_s": 30.0},
        {"type": "straight", "length_m": 100.0},
    ])
    stops = route.stops()
    assert len(stops) == 1
    assert stops[0].arc_length_m == pytest.approx(100.0, rel=0.02)
    assert stops[0].duration_s == 30.0


def test_a_route_with_no_stops_reports_none():
    assert a_route().stops() == []


def test_several_stops_come_back_in_order():
    route = a_route(segments=[
        {"type": "straight", "length_m": 100.0},
        {"type": "stop", "duration_s": 10.0},
        {"type": "straight", "length_m": 200.0},
        {"type": "stop", "duration_s": 20.0},
        {"type": "straight", "length_m": 50.0},
    ])
    positions = [s.arc_length_m for s in route.stops()]
    assert positions == sorted(positions)
    assert positions[1] == pytest.approx(300.0, rel=0.02)


def test_the_total_idle_time_is_the_sum_of_the_stops():
    route = a_route(segments=[
        {"type": "straight", "length_m": 100.0},
        {"type": "stop", "duration_s": 10.0},
        {"type": "straight", "length_m": 100.0},
        {"type": "stop", "duration_s": 25.0},
        {"type": "straight", "length_m": 100.0},
    ])
    assert route.total_stop_time_s() == pytest.approx(35.0)


# -- reproducibility ------------------------------------------------------


def test_identical_routes_hash_identically():
    assert a_route().route_hash == a_route().route_hash


def test_different_routes_hash_differently():
    other = a_route(segments=[
        {"type": "straight", "length_m": 101.0},
        {"type": "straight", "length_m": 50.0},
    ])
    assert a_route().route_hash != other.route_hash


def test_the_route_round_trips_through_a_dict():
    route = a_route()
    assert RouteSpec.from_dict(route.to_dict()) == route
