"""Letting BeamNG's own AI drive.

The custom controller existed because BeamNG.drive had no API. The MCP server
ended that: `drive_to` and `set_ai` are the game's own driver, and they do the
two things we cannot -- stay on the road and avoid other vehicles -- because
they can see, and nothing we build can.

So the split changes. BeamNG drives. We choose the route, choose the style, and
keep every measurement: the thermal model, the logging, the corrosion figure.
"""

import math

import pytest

from behaviour.spec import BehaviourSpec
from control.ai_driver import AIDriver, aggression_for
from control.path import Path
from tests.helpers import a_behaviour_spec


class FakeAIBackend:
    """Records the AI commands sent to it."""

    def __init__(self):
        self.commands = []
        self.x_m = 0.0
        self.y_m = 0.0
        self.z_m = 0.6
        self.speed_mps = 0.0

    def drive_to(self, x, y, z=None, **kwargs):
        self.commands.append(("drive_to", {"x": x, "y": y, **kwargs}))

    def set_ai(self, **kwargs):
        self.commands.append(("set_ai", kwargs))

    def read_state(self):
        from sim.backend import VehicleState

        return VehicleState(x_m=self.x_m, y_m=self.y_m, z_m=self.z_m,
                            speed_mps=self.speed_mps)


def straight(length=1000.0):
    return Path([(float(i), 0.0) for i in range(int(length))])


def a_driver(spec=None, path=None, **kwargs):
    backend = FakeAIBackend()
    driver = AIDriver(
        backend, spec or a_behaviour_spec(), path or straight(),
        dt=0.05, speed_limit_mps=20.0, **kwargs,
    )
    return driver, backend


def commands(backend, name):
    return [args for called, args in backend.commands if called == name]


# -- style becomes aggression ---------------------------------------------


def test_a_brisk_behaviour_maps_to_high_aggression():
    brisk = a_behaviour_spec(target_speed_factor=1.3, accel_limit_mps2=3.8)
    assert aggression_for(brisk) > 0.8


def test_a_gentle_behaviour_maps_to_low_aggression():
    gentle = a_behaviour_spec(target_speed_factor=0.6, accel_limit_mps2=1.0)
    assert aggression_for(gentle) < 0.5


def test_aggression_stays_within_what_the_game_accepts():
    for factor in (0.5, 0.9, 1.3):
        for accel in (0.5, 2.0, 4.0):
            value = aggression_for(
                a_behaviour_spec(target_speed_factor=factor, accel_limit_mps2=accel)
            )
            assert 0.1 <= value <= 1.5, (factor, accel)


# -- driving ---------------------------------------------------------------


def test_starting_configures_the_ai_before_moving():
    driver, backend = a_driver()
    driver.start()
    assert commands(backend, "set_ai")


def test_it_keeps_the_car_in_lane_and_away_from_traffic():
    driver, backend = a_driver()
    driver.start()
    sent = commands(backend, "set_ai")[0]
    assert sent["avoidCars"] is True


def test_it_is_sent_to_a_point_on_the_route():
    driver, backend = a_driver()
    driver.start()
    driver.step(backend.read_state())
    target = commands(backend, "drive_to")[0]
    assert target["y"] == pytest.approx(0.0, abs=0.5)
    assert target["x"] > 0.0


def test_it_is_sent_ahead_of_where_the_car_is():
    driver, backend = a_driver()
    driver.start()
    backend.x_m = 200.0
    driver.step(backend.read_state())
    assert commands(backend, "drive_to")[-1]["x"] > 200.0


def test_the_next_waypoint_is_only_sent_once_the_last_is_reached():
    # Re-issuing drive_to every tick makes the AI restart its route planning.
    driver, backend = a_driver()
    driver.start()
    driver.step(backend.read_state())
    before = len(commands(backend, "drive_to"))
    for _ in range(20):
        driver.step(backend.read_state())
    assert len(commands(backend, "drive_to")) == before


def test_reaching_a_waypoint_sends_the_next_one():
    driver, backend = a_driver()
    driver.start()
    driver.step(backend.read_state())
    first = commands(backend, "drive_to")[-1]
    backend.x_m = first["x"]
    driver.step(backend.read_state())
    assert commands(backend, "drive_to")[-1]["x"] > first["x"]


def test_the_route_end_is_the_last_waypoint():
    driver, backend = a_driver(path=straight(100.0))
    driver.start()
    backend.x_m = 95.0
    driver.step(backend.read_state())
    assert commands(backend, "drive_to")[-1]["x"] <= 99.0


def test_it_reports_when_the_route_is_done():
    driver, backend = a_driver(path=straight(100.0))
    driver.start()
    backend.x_m, backend.speed_mps = 99.0, 0.0
    driver.step(backend.read_state())
    assert driver.is_finished(backend.read_state())


# -- what the run loop still needs ----------------------------------------


def test_it_sends_no_controls_of_its_own():
    # BeamNG is driving. Sending pedal commands as well would fight it.
    driver, backend = a_driver()
    driver.start()
    assert driver.step(backend.read_state()) is None


def test_it_still_reports_distance_from_the_route():
    driver, backend = a_driver()
    driver.start()
    backend.y_m = 4.0
    assert driver.deviation_m(backend.read_state()) == pytest.approx(4.0, abs=0.5)


def test_it_is_never_considered_lost():
    # The game's AI follows roads, not our line. Wandering from our route while
    # driving a real road is the AI doing its job, not a failure.
    driver, backend = a_driver()
    driver.start()
    backend.y_m = 500.0
    assert not driver.is_lost(backend.read_state())


def test_stopping_hands_control_back():
    driver, backend = a_driver()
    driver.start()
    driver.stop()
    assert commands(backend, "set_ai")[-1]["mode"] == "disabled"


# -- coping with a game that rejects some arguments ------------------------


class PickyBackend(FakeAIBackend):
    """Accepts drive_to only without the arguments named in `rejects`."""

    def __init__(self, rejects=("routeSpeedMode", "driveInLane")):
        super().__init__()
        self.rejects = rejects
        self.accepted = []

    def drive_to(self, x, y, z=None, **kwargs):
        self.commands.append(("drive_to", {"x": x, "y": y, **kwargs}))
        offending = [name for name in self.rejects if name in kwargs]
        if offending:
            return f"drive_to failed: unknown argument {offending[0]!r}"
        self.accepted.append(kwargs)
        return "ok"


def test_it_drops_arguments_the_game_will_not_take():
    # BeamNG's exact argument names and enum values are undocumented, and a
    # rejected call means the car simply does not move.
    backend = PickyBackend()
    driver = AIDriver(backend, a_behaviour_spec(), straight(), dt=0.05,
                      speed_limit_mps=20.0)
    driver.start()
    driver.step(backend.read_state())
    assert backend.accepted, "never found a call the game would accept"


def test_it_keeps_as_much_control_as_the_game_allows():
    backend = PickyBackend(rejects=("routeSpeedMode",))
    driver = AIDriver(backend, a_behaviour_spec(), straight(), dt=0.05,
                      speed_limit_mps=20.0)
    driver.start()
    driver.step(backend.read_state())
    # driveInLane was not the problem, so it should survive.
    assert "driveInLane" in backend.accepted[0]


def test_it_remembers_what_worked_instead_of_retrying_every_time():
    backend = PickyBackend()
    driver = AIDriver(backend, a_behaviour_spec(), straight(1500.0), dt=0.05,
                      speed_limit_mps=20.0)
    driver.start()
    driver.step(backend.read_state())
    attempts_for_first = len(commands(backend, "drive_to"))
    backend.x_m = 200.0
    driver.step(backend.read_state())
    attempts_for_second = len(commands(backend, "drive_to")) - attempts_for_first
    assert attempts_for_second == 1


def test_aggression_is_the_last_thing_given_up():
    # Without it every behaviour drives identically, which defeats the point.
    backend = PickyBackend(rejects=("routeSpeedMode", "driveInLane", "avoidCars"))
    driver = AIDriver(backend, a_behaviour_spec(), straight(), dt=0.05,
                      speed_limit_mps=20.0)
    driver.start()
    driver.step(backend.read_state())
    assert "aggression" in backend.accepted[0]


def test_a_game_that_refuses_everything_is_reported():
    backend = PickyBackend(rejects=("aggression",))

    class Hopeless(PickyBackend):
        def drive_to(self, x, y, z=None, **kwargs):
            return "drive_to failed: no"

    driver = AIDriver(Hopeless(), a_behaviour_spec(), straight(), dt=0.05,
                      speed_limit_mps=20.0)
    driver.start()
    with pytest.raises(RuntimeError, match="would not accept"):
        driver.step(driver.backend.read_state())
