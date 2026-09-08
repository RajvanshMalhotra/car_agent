"""Driving to a place you picked on the map.

Point at a destination and the game's AI works out its own route there, using
the road network. The same journey driven in different styles is the comparison
this project exists to make: one route, one map, one car, only the driving
differs.

`drive_to` snaps to the nearest navgraph node, so a destination dropped on a map
is rarely reached exactly -- the AI goes to the closest road it can, which may be
a street away. Arrival is judged with that in mind.
"""

from __future__ import annotations

import math
from typing import Any

#: How close counts as arrived. Generous, because the destination is snapped to
#: the nearest road node and a point picked off a map is seldom on tarmac.
ARRIVED_M = 30.0


def remaining_m(x: float, y: float, destination: Any) -> float:
    return math.dist((x, y), (destination.x_m, destination.y_m))


def arrived(x: float, y: float, destination: Any, tolerance_m: float = ARRIVED_M) -> bool:
    return remaining_m(x, y, destination) <= tolerance_m


#: How long to give the car before concluding it is not going anywhere.
#: Cranking, selecting a gear and pulling away all take a moment, and the AI
#: sometimes pauses to plan before it moves.
GRACE_S = 20.0

#: How far it has to have got in that time. Anything less is a car that never
#: set off, not a car driving slowly.
MOVED_M = 5.0


def going_nowhere(
    started_m: float,
    now_m: float,
    elapsed_s: float,
    already_said: bool = False,
    grace_s: float = GRACE_S,
) -> bool:
    """Whether the car has failed to set off at all.

    Every silent failure -- a refused `drive_to`, a destination off the
    navgraph, the AI never engaging -- presents the same way: a stationary car.
    Waiting out a fifteen-minute run to discover it is the worst way to find
    out.

    Distance *to* the destination is deliberately not the test. The AI follows
    roads, so it routinely sets off away from the target and the gap grows
    before it shrinks. Only a car that has not moved at all has failed.
    """
    if already_said or elapsed_s < grace_s:
        return False
    return abs(now_m - started_m) < MOVED_M
