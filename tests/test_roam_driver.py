"""Letting the AI simply drive around.

The probe showed `set_ai(mode='span')` drives the road network unaided at
24 m/s, indefinitely, with no target of any kind. For this project's actual
purpose that is enough: the battery layer wants realistic engine load and trip
structure, and does not care *which* roads produced them.

So this drops the route, the waypoints, the deviation check and the recovery --
all of which existed to make a car follow a line it had no need to follow.
"""

import pytest

from control.roam_driver import RoamDriver
from tests.helpers import a_behaviour_spec


class FakeAIBackend:
    def __init__(self):
        self.commands = []
        self.speed_mps = 0.0

    def set_ai(self, **kwargs):
        self.commands.append(kwargs)
        return "set ai ok"

    def read_state(self):
        from sim.backend import VehicleState

        return VehicleState(speed_mps=self.speed_mps)


def a_driver(spec=None, **kwargs):
    backend = FakeAIBackend()
    return RoamDriver(backend, spec or a_behaviour_spec(), **kwargs), backend


def test_starting_sets_the_ai_roaming():
    driver, backend = a_driver()
    driver.start()
    assert backend.commands[0]["mode"] == "span"


def test_the_behaviour_sets_how_hard_it_drives():
    brisk, brisk_backend = a_driver(a_behaviour_spec(target_speed_factor=1.3,
                                                    accel_limit_mps2=3.8))
    gentle, gentle_backend = a_driver(a_behaviour_spec(target_speed_factor=0.6,
                                                       accel_limit_mps2=1.0))
    brisk.start()
    gentle.start()
    assert brisk_backend.commands[0]["aggression"] > gentle_backend.commands[0]["aggression"]


def test_it_avoids_other_traffic():
    driver, backend = a_driver()
    driver.start()
    assert backend.commands[0]["avoidCars"] is True


def test_the_roaming_mode_can_be_chosen():
    driver, backend = a_driver(mode="random")
    driver.start()
    assert backend.commands[0]["mode"] == "random"


def test_it_sends_no_controls_of_its_own():
    driver, backend = a_driver()
    driver.start()
    assert driver.step(backend.read_state()) is None


def test_it_does_not_reissue_the_command_every_tick():
    driver, backend = a_driver()
    driver.start()
    for _ in range(50):
        driver.step(backend.read_state())
    assert len(backend.commands) == 1


def test_it_never_runs_out_of_route():
    # There is no route. The run ends when its time is up, not before.
    driver, backend = a_driver()
    driver.start()
    driver.step(backend.read_state())
    assert not driver.is_finished(backend.read_state())


def test_it_is_never_lost():
    driver, backend = a_driver()
    driver.start()
    assert not driver.is_lost(backend.read_state())


def test_it_reports_no_deviation_because_there_is_nothing_to_deviate_from():
    driver, backend = a_driver()
    assert driver.deviation_m(backend.read_state()) == 0.0


def test_stopping_hands_the_car_back():
    driver, backend = a_driver()
    driver.start()
    driver.stop()
    assert backend.commands[-1]["mode"] == "disabled"


def test_it_starts_again_after_a_recovery():
    driver, backend = a_driver()
    driver.start()
    driver.restart(None)
    assert backend.commands[-1]["mode"] == "span"
