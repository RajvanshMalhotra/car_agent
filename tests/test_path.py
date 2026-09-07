"""A route the controller follows, and the lookahead lookup pure pursuit needs."""

import pytest

from control.path import Path


def straight_path(length=100.0, spacing=1.0):
    n = int(length / spacing) + 1
    return Path([(i * spacing, 0.0) for i in range(n)])


def test_rejects_a_path_with_fewer_than_two_points():
    with pytest.raises(ValueError, match="at least two"):
        Path([(0.0, 0.0)])


def test_length_of_a_straight_path_is_its_span():
    assert straight_path(length=10.0).length_m == pytest.approx(10.0)


def test_closest_point_to_a_position_on_the_path_is_that_position():
    assert straight_path().closest_arc_length((5.0, 0.0)) == pytest.approx(5.0)


def test_closest_point_ignores_lateral_offset():
    assert straight_path().closest_arc_length((5.0, 3.0)) == pytest.approx(5.0)


def test_target_point_sits_one_lookahead_ahead_along_the_path():
    x, y = straight_path().target_point((5.0, 0.0), lookahead_m=8.0)
    assert (x, y) == pytest.approx((13.0, 0.0))


def test_target_point_clamps_to_the_final_point_near_the_end():
    x, y = straight_path(length=10.0).target_point((9.0, 0.0), lookahead_m=8.0)
    assert (x, y) == pytest.approx((10.0, 0.0))


def test_a_vehicle_at_the_last_point_has_finished():
    assert straight_path(length=10.0).is_finished((10.0, 0.0))


def test_a_vehicle_mid_route_has_not_finished():
    assert not straight_path(length=10.0).is_finished((4.0, 0.0))


def test_curvature_of_a_straight_path_is_zero():
    assert straight_path().curvature_at(arc_length_m=5.0) == pytest.approx(0.0, abs=1e-9)


def test_curvature_of_a_circular_path_is_one_over_its_radius():
    import math

    radius = 20.0
    points = [
        (radius * math.sin(t / 200), radius * (1 - math.cos(t / 200)))
        for t in range(400)
    ]
    path = Path(points)
    assert path.curvature_at(arc_length_m=10.0) == pytest.approx(1 / radius, rel=0.05)
