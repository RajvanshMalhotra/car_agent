"""Letting the AI simply drive around.

`set_ai(mode='span')` drives BeamNG's road network unaided -- measured at 24 m/s
on a real map, indefinitely, with no target of any kind. For this project's
actual purpose that is enough. The battery layer wants engine load, bay
temperature and trip structure; it does not care *which* roads produced them.

So this drops the route, the waypoints, the deviation check and the recovery.
All of those existed to make a car follow a line, and following a particular
line was never the goal -- it was scaffolding from the days when the only way to
move the car was a virtual gamepad.

The behaviour still decides how hard it drives, through the same `aggression`
mapping the route-following driver uses. What is given up is control over
*where* it goes, which for measuring how driving ages a battery costs nothing.
"""

from __future__ import annotations

from typing import Any

from behaviour.spec import BehaviourSpec
from control.ai_driver import aggression_for
from sim.backend import VehicleState

#: 'span' covers the network systematically; 'random' wanders. Either produces
#: real driving on real roads.
DEFAULT_MODE = "span"


class RoamDriver:
    def __init__(
        self,
        backend: Any,
        spec: BehaviourSpec,
        mode: str = DEFAULT_MODE,
        avoid_cars: bool = True,
        **_ignored: Any,
    ) -> None:
        self.backend = backend
        self.spec = spec
        self.mode = mode
        self.avoid_cars = avoid_cars
        self.aggression = aggression_for(spec)

    def start(self) -> None:
        self.backend.set_ai(
            mode=self.mode,
            aggression=self.aggression,
            avoidCars=self.avoid_cars,
        )

    def stop(self) -> None:
        """Hand the car back, or the AI keeps driving after the run ends."""
        self.backend.set_ai(mode="disabled")

    def restart(self, path: Any = None) -> None:
        self.start()

    def step(self, state: VehicleState) -> None:
        """Nothing to do. The AI is already driving, and re-sending the command
        every tick would only interrupt whatever route it has planned."""
        return None

    # -- the questions the run loop asks ----------------------------------

    def deviation_m(self, state: VehicleState) -> float:
        return 0.0

    def is_lost(self, state: VehicleState) -> bool:
        return False

    def is_finished(self, state: VehicleState) -> bool:
        """Never. There is no route to reach the end of; the run ends on time."""
        return False
