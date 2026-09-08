"""Driving to a place you picked on the map.

Point at a destination, the game's AI works out the route and drives there. The
same journey can then be driven in different styles, which is the comparison the
whole project is built to make.
"""

import pytest

from control.journey import ARRIVED_M, arrived, remaining_m
from control.places import Place


def a_place(x=1000.0, y=500.0):
    return Place(name="depot", x_m=x, y_m=y, z_m=100.0)


def test_the_distance_left_is_reported():
    assert remaining_m(0.0, 0.0, a_place(300.0, 400.0)) == pytest.approx(500.0)


def test_a_car_at_the_destination_has_arrived():
    assert arrived(1000.0, 500.0, a_place())


def test_a_car_across_the_map_has_not():
    assert not arrived(0.0, 0.0, a_place())


def test_close_enough_counts_as_arrived():
    # drive_to snaps to the nearest road node, so the car stops near the point
    # rather than on it.
    place = a_place()
    assert arrived(place.x_m + ARRIVED_M * 0.8, place.y_m, place)


def test_just_too_far_does_not_count():
    place = a_place()
    assert not arrived(place.x_m + ARRIVED_M * 2.0, place.y_m, place)


def test_the_tolerance_allows_for_the_nearest_road_node():
    # A destination dropped on a map is rarely on a road; the AI goes to the
    # closest node it can reach, which can be a street away.
    assert 10.0 <= ARRIVED_M <= 60.0
