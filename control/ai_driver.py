"""Letting BeamNG's own AI drive.

The custom controller existed for one reason: BeamNG.drive had no API, so the
car had to be driven through a virtual gamepad by something we wrote. The MCP
server ended that. `drive_to` and `set_ai` are the game's own driver, and they
do the two things nothing here can -- stay on the road and avoid other traffic --
because the game can see and we cannot.

So the split changes. BeamNG drives. This project chooses the route, chooses the
style, and keeps every measurement: the engine and thermal model, the logging,
the corrosion figure. Those were always the point; the driving was scaffolding.

What is given up: `aggression` is a single scalar where `BehaviourSpec` has
eight parameters, so how the car drives is coarser. What is gained is a car that
stays on the road.

**This needs a real map.** `drive_to` snaps to the nearest navgraph node, and a
level with no roads -- `smallgrid`, for instance -- has no navgraph at all.
"""

from __future__ import annotations

import math
from typing import Any

from behaviour.spec import BehaviourSpec
from control.path import Path
from sim.backend import VehicleState

#: BeamNG's aggression scale. Below ~0.3 the car crawls; above ~1.2 it drives
#: like it is being chased.
MIN_AGGRESSION = 0.25
MAX_AGGRESSION = 1.2

#: How far ahead the next waypoint is placed. Too close and the AI brakes for
#: each one; too far and it stops following the recorded route at all.
WAYPOINT_SPACING_M = 60.0

#: A waypoint counts as reached inside this. The AI stops *at* its target, so
#: the next one has to be issued before the car settles there.
WAYPOINT_REACHED_M = 25.0

#: How far along the route to search for the car's position. The AI follows
#: roads rather than our line and is only consulted a few times a second, so it
#: can be a long way from where it last was -- a narrow window would find the
#: wrong stretch of road and the route would appear to stall.
SEARCH_WINDOW_M = 400.0

#: What to ask `drive_to` for, richest first. BeamNG's exact argument names and
#: enum values are undocumented, and a rejected call does not raise -- the car
#: simply does not move, which is indistinguishable from the AI being broken. So
#: the driver drops arguments until the game accepts one, and remembers which.
#:
#: Aggression is given up last: without it every behaviour drives identically,
#: which defeats the point of having behaviours at all.
DRIVE_TO_ARGUMENTS = (
    ("aggression", "avoidCars", "driveInLane", "routeSpeed", "routeSpeedMode"),
    ("aggression", "avoidCars", "driveInLane", "routeSpeed"),
    ("aggression", "avoidCars", "driveInLane"),
    ("aggression", "avoidCars"),
    ("aggression",),
    (),
)


def _refused(result: Any) -> bool:
    """BeamNG reports a bad call in the returned text rather than by raising."""
    text = str(result).lower()
    return "fail" in text or "error" in text or "unknown" in text


def request_route(
    backend: Any,
    x: float,
    y: float,
    z: float | None = None,
    accepted: tuple[str, ...] | None = None,
    **settings: Any,
) -> tuple[str, ...]:
    """Ask the game's AI to drive to a point, and confirm that it agreed.

    Returns the argument shape the game accepted, so the caller can pass it
    back and skip the probing next time. Raises if nothing is accepted --
    the alternative is a silent refusal, which looks from outside exactly
    like a car that will not move.
    """
    available = {
        "aggression": settings.get("aggression", 0.6),
        "avoidCars": settings.get("avoidCars", True),
        "driveInLane": settings.get("driveInLane", True),
        "routeSpeed": settings.get("routeSpeed", 0.0),
        "routeSpeedMode": settings.get("routeSpeedMode", "limit"),
    }
    candidates = (accepted,) if accepted is not None else DRIVE_TO_ARGUMENTS
    for names in candidates:
        result = backend.drive_to(
            x=x, y=y, z=z, **{name: available[name] for name in names}
        )
        if not _refused(result):
            return names
    raise RuntimeError(
        "BeamNG would not accept any form of drive_to. Run "
        "windows_ai_probe.py to see what it says."
    )


def aggression_for(spec: BehaviourSpec) -> float:
    """Collapse a behaviour into the one number the game's AI takes.

    Speed matters more than acceleration in how the AI actually drives, hence
    the weighting. This is lossy by nature -- eight parameters into one -- and
    it is the price of letting the game drive.
    """
    speed_part = (spec.target_speed_factor - 0.5) / 0.8
    accel_part = (spec.accel_limit_mps2 - 0.5) / 3.5
    blended = 0.6 * speed_part + 0.4 * accel_part
    return min(MAX_AGGRESSION, max(MIN_AGGRESSION, 0.3 + 0.75 * blended))


class AIDriver:
    """Steers the game's AI along a route, rather than driving the car itself."""

    def __init__(
        self,
        backend: Any,
        spec: BehaviourSpec,
        path: Path,
        dt: float,
        speed_limit_mps: float,
        demonstration: Any = None,
        drive_in_lane: bool = True,
        avoid_cars: bool = True,
        waypoint_spacing_m: float = WAYPOINT_SPACING_M,
    ) -> None:
        self.backend = backend
        self.spec = spec
        self.path = path
        self.dt = dt
        self.speed_limit_mps = speed_limit_mps
        self.demonstration = demonstration
        self.drive_in_lane = drive_in_lane
        self.avoid_cars = avoid_cars
        self.waypoint_spacing_m = waypoint_spacing_m

        self.aggression = aggression_for(spec)
        self.progress_m: float | None = 0.0
        self.waypoint: tuple[float, float] | None = None
        self.waypoint_arc_m = 0.0
        #: Which argument set the game accepted, once we have found one.
        self.accepted_arguments: tuple[str, ...] | None = None

    # -- driving ----------------------------------------------------------

    def start(self) -> None:
        self.backend.set_ai(
            mode="manual",
            aggression=self.aggression,
            avoidCars=self.avoid_cars,
        )

    def stop(self) -> None:
        """Hand the car back. Leaving the AI engaged would keep it driving."""
        self.backend.set_ai(mode="disabled")

    def target_speed_mps(self, arc_length_m: float) -> float:
        if self.demonstration is not None:
            return self.demonstration.target_speed_at(arc_length_m, self.spec)
        return self.speed_limit_mps * self.spec.target_speed_factor

    def _send_waypoint(self) -> None:
        arc = min(self.path.length_m, (self.progress_m or 0.0) + self.waypoint_spacing_m)
        x, y = self.path.point_at(arc)
        self.waypoint = (x, y)
        self.waypoint_arc_m = arc

        available = {
            "aggression": self.aggression,
            "avoidCars": self.avoid_cars,
            "driveInLane": self.drive_in_lane,
            "routeSpeed": self.target_speed_mps(arc),
            "routeSpeedMode": "limit",
        }
        candidates = (
            (self.accepted_arguments,)
            if self.accepted_arguments is not None
            else DRIVE_TO_ARGUMENTS
        )
        for names in candidates:
            result = self.backend.drive_to(
                x=x, y=y, **{name: available[name] for name in names}
            )
            if not _refused(result):
                if self.accepted_arguments is None:
                    dropped = set(DRIVE_TO_ARGUMENTS[0]) - set(names)
                    if dropped:
                        print(f"  note: the game would not take "
                              f"{', '.join(sorted(dropped))} on drive_to; "
                              f"driving without them")
                    self.accepted_arguments = names
                return
        raise RuntimeError(
            "BeamNG would not accept any form of drive_to. Run "
            "windows_ai_probe.py to see what it says."
        )

    def step(self, state: VehicleState) -> None:
        """Advance the route. Returns nothing: BeamNG is working the controls.

        Sending pedal commands as well would fight the AI, and re-issuing
        `drive_to` every tick makes it restart its route planning, so the next
        waypoint only goes out once the last is reached.
        """
        self.progress_m = self.path.closest_arc_length(
            (state.x_m, state.y_m),
            near_arc_length=self.progress_m,
            window_m=SEARCH_WINDOW_M,
        )
        # Judged on progress along the route, not distance to the waypoint: a
        # car that overshoots its target is *past* it, and waiting to be near it
        # again would leave it without a target at all.
        reached = self.progress_m >= self.waypoint_arc_m - WAYPOINT_REACHED_M
        more_route = self.waypoint_arc_m < self.path.length_m - 1.0
        if self.waypoint is None or (reached and more_route):
            self._send_waypoint()
        return None

    # -- the questions the run loop asks ----------------------------------

    def deviation_m(self, state: VehicleState) -> float:
        arc = self.path.closest_arc_length(
            (state.x_m, state.y_m),
            near_arc_length=self.progress_m,
            window_m=SEARCH_WINDOW_M,
        )
        near_x, near_y = self.path.point_at(arc)
        return math.hypot(near_x - state.x_m, near_y - state.y_m)

    def is_lost(self, state: VehicleState) -> bool:
        """Never. The AI follows roads; our route is only a suggestion of where.

        Wandering from the recorded line while driving a real road is the AI
        doing its job. Judging it against our path would abandon good runs.
        """
        return False

    def is_finished(self, state: VehicleState) -> bool:
        return (
            self.progress_m is not None
            and self.progress_m >= self.path.length_m - WAYPOINT_REACHED_M
            and state.speed_mps < 1.0
        )

    def restart(self, path: Path) -> None:
        self.path = path
        self.progress_m = None
        self.waypoint = None
        self.start()
