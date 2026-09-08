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
