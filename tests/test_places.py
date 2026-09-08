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


# -- which map a place belongs to -----------------------------------------


def test_a_place_remembers_which_map_it_was_saved_on(tmp_path):
    save_place(tmp_path, "depot", 900.0, 300.0, 100.0, level="west_coast_usa")
    assert load_place(tmp_path, "depot").level == "west_coast_usa"


def test_a_place_saved_without_a_map_still_loads(tmp_path):
    # Older places.json files, and any build whose status does not name the
    # level, have nothing to record.
    save_place(tmp_path, "depot", 900.0, 300.0, 100.0)
    assert load_place(tmp_path, "depot").level == ""


def test_a_place_from_another_map_is_wrong_for_this_one(tmp_path):
    place = save_place(tmp_path, "depot", 900.0, 300.0, 100.0, level="italy")
    assert place.belongs_to("west_coast_usa") is False


def test_a_place_from_this_map_is_right_for_it(tmp_path):
    place = save_place(tmp_path, "depot", 900.0, 300.0, 100.0, level="italy")
    assert place.belongs_to("italy") is True


def test_an_unknown_map_is_never_called_wrong(tmp_path):
    # Not knowing which map you are on is not evidence of a mismatch, and
    # refusing to drive on a guess would be worse than driving.
    place = save_place(tmp_path, "depot", 900.0, 300.0, 100.0, level="italy")
    assert place.belongs_to("") is True
    assert save_place(tmp_path, "b", 0.0, 0.0, 0.0).belongs_to("italy") is True


def test_the_level_is_read_from_whichever_key_the_game_uses():
    from control.places import level_from_status

    for key in ("level", "levelName", "mapName", "map"):
        assert level_from_status({key: "italy"}) == "italy"


def test_a_status_naming_no_level_yields_nothing():
    from control.places import level_from_status

    assert level_from_status({"vehicle": {"pos": {"x": 1.0}}}) == ""
    assert level_from_status(None) == ""


def test_a_level_path_is_reduced_to_its_name():
    from control.places import level_from_status

    assert level_from_status({"level": "/levels/west_coast_usa/info.json"}) == \
        "west_coast_usa"


def test_driving_to_a_place_from_another_map_is_refused():
    # drive_to snaps to the nearest navgraph node, so a place from another map
    # is not refused by the game -- it is accepted and routed somewhere
    # arbitrary. That is worse than an error, so it is caught here.
    import drive
    from control.places import Place

    place = Place(name="depot", x_m=900.0, y_m=300.0, z_m=100.0, level="italy")
    wrong, why = drive.wrong_map(place, "west_coast_usa")
    assert wrong
    assert "italy" in why and "west_coast_usa" in why


def test_driving_to_a_place_on_this_map_is_allowed():
    import drive
    from control.places import Place

    place = Place(name="depot", x_m=900.0, y_m=300.0, z_m=100.0, level="italy")
    assert drive.wrong_map(place, "italy") == (False, "")


def test_an_unknown_map_does_not_block_the_run():
    import drive
    from control.places import Place

    place = Place(name="depot", x_m=900.0, y_m=300.0, z_m=100.0)
    assert drive.wrong_map(place, "italy") == (False, "")
