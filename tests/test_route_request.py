"""Asking BeamNG's AI to drive somewhere, and knowing whether it agreed.

BeamNG reports a refused `drive_to` in the text it returns rather than by
raising, so a call with an argument the build does not know looks exactly like
a call that worked -- until the car does not move. Every route request has to
drop arguments until one is accepted, and say so when none is.
"""

import pytest

from control.ai_driver import DRIVE_TO_ARGUMENTS, request_route


class FakeAI:
    """Accepts drive_to only when given exactly `takes`, like the real one."""

    def __init__(self, takes=("aggression", "avoidCars", "driveInLane",
                              "routeSpeed", "routeSpeedMode")):
        self.takes = set(takes)
        self.calls = []

    def drive_to(self, x, y, z=None, **settings):
        self.calls.append(settings)
        unknown = set(settings) - self.takes
        if unknown:
            return f"error: unknown argument {sorted(unknown)[0]}"
        return {"ok": True}


def test_a_full_house_is_asked_for_first():
    ai = FakeAI()
    accepted = request_route(ai, 10.0, 20.0, 5.0, aggression=0.8)
    assert accepted == DRIVE_TO_ARGUMENTS[0]
    assert len(ai.calls) == 1


def test_arguments_are_dropped_until_the_game_accepts():
    ai = FakeAI(takes=("aggression", "avoidCars"))
    accepted = request_route(ai, 10.0, 20.0, 5.0, aggression=0.8)
    assert accepted == ("aggression", "avoidCars")
    assert ai.calls[-1] == {"aggression": 0.8, "avoidCars": True}


def test_aggression_is_the_last_thing_given_up():
    # Without it every behaviour drives identically, which is the whole point.
    ai = FakeAI(takes=("aggression",))
    accepted = request_route(ai, 10.0, 20.0, 5.0, aggression=0.8)
    assert accepted == ("aggression",)


def test_a_known_good_shape_is_reused_without_probing_again():
    ai = FakeAI(takes=("aggression",))
    request_route(ai, 10.0, 20.0, 5.0, aggression=0.8, accepted=("aggression",))
    assert len(ai.calls) == 1


def test_a_game_that_refuses_everything_is_reported_not_ignored():
    # The silent version of this is a car that simply never sets off.
    ai = FakeAI(takes=())

    def refuse_all(x, y, z=None, **settings):
        return "error: drive_to failed"

    ai.drive_to = refuse_all
    with pytest.raises(RuntimeError, match="drive_to"):
        request_route(ai, 10.0, 20.0, 5.0, aggression=0.8)


def test_the_destination_is_passed_through_untouched():
    seen = {}

    class Recorder:
        def drive_to(self, x, y, z=None, **settings):
            seen.update(x=x, y=y, z=z)
            return {"ok": True}

    request_route(Recorder(), 123.5, -67.25, 9.75, aggression=0.5)
    assert seen == {"x": 123.5, "y": -67.25, "z": 9.75}
