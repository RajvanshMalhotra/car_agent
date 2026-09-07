"""Recorded routes.

The controller tracks a path to ~2 cm. It will still drive into a tree if the
path goes through one -- the synthetic sine-wave route has no relationship to
the road. The fix is to record a human driving the road and follow that.
"""

import json
import math

import pytest

from control.path import Path
from control.route import decimate, load_route, save_route


def a_straight_drive(n=200, spacing=0.2):
    return [(i * spacing, 0.0) for i in range(n)]


def test_a_saved_route_loads_back_as_a_path(tmp_path):
    save_route(tmp_path / "r.json", a_straight_drive(), name="test lap")
    assert isinstance(load_route(tmp_path / "r.json"), Path)


def test_the_route_keeps_its_shape_through_a_round_trip(tmp_path):
    save_route(tmp_path / "r.json", a_straight_drive(n=100, spacing=1.0), name="x")
    assert load_route(tmp_path / "r.json").length_m == pytest.approx(99.0, rel=0.02)


def test_the_route_records_where_and_when_it_came_from(tmp_path):
    save_route(tmp_path / "r.json", a_straight_drive(), name="west coast run")
    stored = json.loads((tmp_path / "r.json").read_text())
    assert stored["name"] == "west coast run"
    assert "recorded_at" in stored
    assert stored["point_count"] == len(stored["points"])


def test_recording_fewer_than_two_points_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        save_route(tmp_path / "r.json", [(0.0, 0.0)], name="x")


# -- decimation -----------------------------------------------------------


def test_decimation_thins_a_densely_sampled_drive():
    # 50 Hz at 10 m/s is a point every 20 cm; that is far denser than needed
    # and makes every path lookup slower for no benefit.
    assert len(decimate(a_straight_drive(n=1000, spacing=0.2), min_spacing_m=2.0)) < 120


def test_decimation_keeps_points_at_least_the_spacing_apart():
    # The final gap is exempt: the end of the route is preserved exactly, which
    # matters more than a uniform spacing.
    thinned = decimate(a_straight_drive(n=1000, spacing=0.2), min_spacing_m=2.0)
    gaps = [math.dist(a, b) for a, b in zip(thinned, thinned[1:])]
    assert min(gaps[:-1]) >= 2.0 - 1e-9


def test_decimation_preserves_the_start_and_the_end():
    points = a_straight_drive(n=100, spacing=0.5)
    thinned = decimate(points, min_spacing_m=2.0)
    assert thinned[0] == points[0]
    assert thinned[-1] == points[-1]


def test_decimation_collapses_time_spent_stationary():
    # Sitting at a junction produces hundreds of identical points.
    points = [(0.0, 0.0)] * 500 + [(float(i), 0.0) for i in range(1, 50)]
    assert len(decimate(points, min_spacing_m=2.0)) < 30


def test_decimation_does_not_shorten_the_route():
    points = [(i * 0.2, 5 * math.sin(i * 0.2 / 20)) for i in range(2000)]
    original = Path(points).length_m
    assert Path(decimate(points, min_spacing_m=2.0)).length_m == pytest.approx(
        original, rel=0.02
    )


def test_a_route_that_never_moves_is_refused():
    with pytest.raises(ValueError, match="did not move"):
        decimate([(0.0, 0.0)] * 100, min_spacing_m=2.0)
