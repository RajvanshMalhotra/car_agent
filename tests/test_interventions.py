"""Counting what actually happened to the car, kind by kind.

Going off the road is not damage. Reporting a recovery as a repair overstates
how battered the run was, and damage is not a cosmetic detail here -- it is the
thing that decides whether a run's thermals are trustworthy.
"""

from control.interventions import Interventions


def test_a_fresh_run_has_nothing_to_report():
    tally = Interventions()
    assert tally.total == 0
    assert tally.summary() == ""


def test_going_off_the_road_is_not_recorded_as_damage():
    tally = Interventions()
    tally.record("recover")
    assert tally.repairs == 0
    assert tally.recoveries == 1
    assert "damage" not in tally.summary()
    assert "1 recovery" in tally.summary()


def test_damage_is_recorded_as_damage():
    tally = Interventions()
    tally.record("repair")
    assert tally.repairs == 1
    assert "1 repair" in tally.summary()


def test_relocations_are_their_own_thing():
    tally = Interventions()
    tally.record("relocate")
    assert tally.relocations == 1
    assert tally.repairs == 0
    assert "1 relocation" in tally.summary()


def test_every_kind_appears_when_every_kind_happened():
    tally = Interventions()
    for action in ("repair", "repair", "recover", "relocate"):
        tally.record(action)
    summary = tally.summary()
    assert "2 repairs" in summary
    assert "1 recovery" in summary
    assert "1 relocation" in summary
    assert tally.total == 4


def test_only_repairs_make_a_run_thermally_suspect():
    # A recovered car is the same car. A repaired one has had its damage --
    # and so its drag, its engine load and its thermals -- reset mid-run.
    off_road = Interventions()
    off_road.record("recover")
    assert not off_road.thermally_disturbed

    pranged = Interventions()
    pranged.record("repair")
    assert pranged.thermally_disturbed
