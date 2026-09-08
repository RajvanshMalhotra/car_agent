"""Named spots on the map.

Somewhere to send the car when it keeps crashing where it is -- a stretch of
highway, a quiet road -- chosen by driving there once and saving the spot.
"""

import pytest

from control.places import load_place, place_names, save_place


def test_a_saved_place_comes_back(tmp_path):
    save_place(tmp_path, "highway", 1250.5, -830.25, 96.0)
    place = load_place(tmp_path, "highway")
    assert (place.x_m, place.y_m, place.z_m) == pytest.approx((1250.5, -830.25, 96.0))


def test_an_unknown_place_is_not_found(tmp_path):
    assert load_place(tmp_path, "nowhere") is None


def test_places_are_listed(tmp_path):
    save_place(tmp_path, "highway", 1.0, 2.0, 3.0)
    save_place(tmp_path, "quiet-lane", 4.0, 5.0, 6.0)
    assert sorted(place_names(tmp_path)) == ["highway", "quiet-lane"]


def test_saving_the_same_name_replaces_it(tmp_path):
    save_place(tmp_path, "highway", 1.0, 2.0, 3.0)
    save_place(tmp_path, "highway", 9.0, 9.0, 9.0)
    assert load_place(tmp_path, "highway").x_m == pytest.approx(9.0)
    assert place_names(tmp_path) == ["highway"]


def test_a_place_records_when_it_was_saved(tmp_path):
    save_place(tmp_path, "highway", 1.0, 2.0, 3.0)
    assert load_place(tmp_path, "highway").saved_at
