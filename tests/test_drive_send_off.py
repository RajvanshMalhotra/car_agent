"""Handing the car to the AI, and knowing it took it.

Every silent failure here looks the same from the outside -- a car that sits
there -- so the script has to check what the game said rather than assume.
"""

import types

import pytest

import drive
from control.places import Place


class Game:
    """A game that only accepts drive_to arguments it knows about."""

    def __init__(self, takes=("aggression", "avoidCars", "driveInLane",
                              "routeSpeed", "routeSpeedMode")):
        self.takes = set(takes)
        self.ai_calls = []
        self.route_calls = []

    def set_ai(self, **settings):
        self.ai_calls.append(settings)
        return {"ok": True}

    def drive_to(self, x, y, z=None, **settings):
        self.route_calls.append(dict(settings, x=x, y=y, z=z))
        unknown = set(settings) - self.takes
        if unknown:
            return f"error: unknown argument {sorted(unknown)[0]}"
        return {"ok": True}


def some_args(**overrides):
    return types.SimpleNamespace(mode="span", **overrides)


DEPOT = Place(name="depot", x_m=900.0, y_m=300.0, z_m=100.0)


def test_a_refused_route_is_retried_until_the_game_takes_it():
    game = Game(takes=("aggression", "avoidCars"))
    drive.send_off(game, some_args(), 0.8, DEPOT)
    assert game.route_calls[-1]["aggression"] == 0.8
    assert "driveInLane" not in game.route_calls[-1]


def test_the_accepted_shape_is_reused_rather_than_probed_again():
    game = Game(takes=("aggression",))
    accepted = drive.send_off(game, some_args(), 0.8, DEPOT)
    before = len(game.route_calls)
    drive.send_off(game, some_args(), 0.8, DEPOT, accepted=accepted)
    assert len(game.route_calls) == before + 1


def test_a_game_that_refuses_every_route_is_not_left_silent():
    game = Game(takes=())
    game.drive_to = lambda x, y, z=None, **s: "error: drive_to failed"
    with pytest.raises(RuntimeError, match="drive_to"):
        drive.send_off(game, some_args(), 0.8, DEPOT)


def test_the_destination_actually_reaches_the_game():
    game = Game()
    drive.send_off(game, some_args(), 0.8, DEPOT)
    sent = game.route_calls[-1]
    assert (sent["x"], sent["y"], sent["z"]) == (900.0, 300.0, 100.0)


def test_roaming_needs_no_route_at_all():
    game = Game()
    drive.send_off(game, some_args(), 0.8, None)
    assert game.route_calls == []
    assert game.ai_calls[-1]["mode"] == "span"
