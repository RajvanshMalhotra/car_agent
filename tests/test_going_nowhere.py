"""Noticing early that the car is not going anywhere.

Every way this fails looks identical from outside: the car sits still. Waiting
out a fifteen-minute run to discover it never set off is the worst way to find
out, so the run says so within the first half-minute.
"""

from control.journey import going_nowhere, GRACE_S


def test_a_car_closing_on_its_destination_is_fine():
    assert not going_nowhere(started_m=500.0, now_m=430.0, elapsed_s=30.0)


def test_a_car_that_has_not_moved_at_all_is_reported():
    assert going_nowhere(started_m=500.0, now_m=500.0, elapsed_s=GRACE_S + 1)


def test_nothing_is_said_before_the_car_has_had_a_chance():
    # Cranking, selecting a gear and pulling away all take a moment.
    assert not going_nowhere(started_m=500.0, now_m=500.0, elapsed_s=2.0)


def test_driving_away_from_the_destination_is_still_progress():
    # The AI follows roads, so it routinely goes the "wrong" way first. Only a
    # car that is not moving at all is a problem.
    assert not going_nowhere(started_m=500.0, now_m=560.0, elapsed_s=GRACE_S + 1)


def test_it_is_only_said_once():
    assert going_nowhere(500.0, 500.0, GRACE_S + 1)
    assert not going_nowhere(500.0, 500.0, GRACE_S + 1, already_said=True)
