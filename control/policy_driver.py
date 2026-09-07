"""Driving a route with a learned policy.

Presents the same interface as the hand-built `Driver`, so the run loop, the
logging and the data collection are unchanged. The difference is what it needs
to be told: nothing about the vehicle. No wheelbase, no acceleration limit, no
steering lock, and so no calibration step that can be rejected.

The behaviour spec still sets what to do -- target speed, how much to slow for
bends, where to stop, how erratically to drive. The policy only supplies how to
work the controls to achieve it.
"""

from __future__ import annotations

import math
import random
from typing import Any, Sequence

from behaviour.spec import BehaviourSpec
from control.path import Path
from control.policy import DrivingPolicy, observe
from sim.backend import ControlInput, VehicleState

#: Lateral acceleration a driver of nominal confidence accepts in a bend;
#: `corner_speed_factor` scales it.
NOMINAL_LATERAL_ACCEL_MPS2 = 3.0

#: How far off the route counts as lost.
DEFAULT_MAX_DEVIATION_M = 10.0

#: A waypoint counts as reached once the car is within this of it and stopped.
STOP_REACHED_M = 8.0

WANDER_TIME_CONSTANT_S = 4.0
WANDER_SCALE = 0.15

#: A target below this means "be stopped", and the brake is held rather than the
#: decision left to the policy. A linear policy settles at a slow creep instead
#: of a standstill, and a creeping car never registers as stopped -- which would
#: silently empty the idle time out of every dataset, and idling with a hot
#: engine is the mechanism the data exists to capture. Stated plainly: coming to
#: rest is not learned, it is enforced. The approach is still the policy's, and
#: still shaped by the behaviour's deceleration limit.
STANDSTILL_TARGET_MPS = 0.05


class PolicyDriver:
    def __init__(
        self,
        policy: DrivingPolicy,
        spec: BehaviourSpec,
        path: Path,
        dt: float,
        speed_limit_mps: float,
        seed: int = 0,
        stops: Sequence[Any] | None = None,
        max_deviation_m: float = DEFAULT_MAX_DEVIATION_M,
        demonstration: Any | None = None,
    ) -> None:
        self.policy = policy
        self.spec = spec
        self.path = path
        self.dt = dt
        self.speed_limit_mps = speed_limit_mps
        self.max_deviation_m = max_deviation_m
        # When a human drive is supplied it sets the pace instead of a flat
        # speed limit: where they slowed, the agent slows.
        self.demonstration = demonstration
        self.rng = random.Random(seed)

        self.stops = list(stops or [])
        self.stop_index = 0
        self.waiting_until_s: float | None = None
        self.progress_m = 0.0
        self.previous_steering = 0.0
        self.wander = 0.0

    # -- target speed: the behaviour's business, not the policy's ---------

    def _corner_speed_mps(self, arc_length_m: float) -> float:
        horizon_m = max(10.0, self.speed_limit_mps * self.spec.following_distance_s)
        lateral_accel = NOMINAL_LATERAL_ACCEL_MPS2 * self.spec.corner_speed_factor
        limit = math.inf
        for i in range(9):
            curvature = self.path.curvature_at(arc_length_m + horizon_m * i / 8)
            if curvature > 1e-6:
                limit = min(limit, math.sqrt(lateral_accel / curvature))
        return limit

    def _wander_factor(self) -> float:
        decay = math.exp(-self.dt / WANDER_TIME_CONSTANT_S)
        self.wander = decay * self.wander + math.sqrt(1 - decay * decay) * self.rng.gauss(0.0, 1.0)
        self.wander = min(2.0, max(-2.0, self.wander))
        return 1.0 + WANDER_SCALE * self.spec.erraticness * self.wander

    def _next_stop(self):
        if self.stop_index < len(self.stops):
            return self.stops[self.stop_index]
        return None

    def _update_stop(self, state: VehicleState) -> None:
        stop = self._next_stop()
        if stop is None:
            return
        reached = (
            self.progress_m >= stop.arc_length_m - STOP_REACHED_M
            and state.speed_mps < 0.5
        )
        if self.waiting_until_s is None:
            if reached:
                self.waiting_until_s = state.sim_time_s + stop.duration_s
        elif state.sim_time_s >= self.waiting_until_s:
            self.waiting_until_s = None
            self.stop_index += 1

    def target_speed_mps(self, state: VehicleState) -> float:
        if self.waiting_until_s is not None:
            return 0.0
        if self.demonstration is not None:
            target = self.demonstration.target_speed_at(self.progress_m, self.spec)
        else:
            target = self.speed_limit_mps * self.spec.target_speed_factor
        target = min(target, self._corner_speed_mps(self.progress_m))
        target *= self._wander_factor()

        stop_at = self.path.length_m
        stop = self._next_stop()
        if stop is not None:
            stop_at = min(stop_at, stop.arc_length_m)
        remaining = max(0.0, stop_at - self.progress_m - 2.0)
        # The policy handles the pedals, but it cannot know where the route
        # ends, so the target is bled off approaching a stop.
        target = min(target, math.sqrt(2 * 0.5 * self.spec.decel_limit_mps2 * remaining))
        return max(0.0, target)

    # -- the loop ---------------------------------------------------------

    def step(self, state: VehicleState) -> ControlInput:
        self.progress_m = self.path.closest_arc_length(
            (state.x_m, state.y_m), near_arc_length=self.progress_m
        )
        self._update_stop(state)
        target = self.target_speed_mps(state)
        control = self.policy.act(
            observe(
                self.path,
                state.x_m,
                state.y_m,
                state.heading_rad,
                state.speed_mps,
                target,
                self.previous_steering,
                progress_m=self.progress_m,
            )
        )
        if target <= STANDSTILL_TARGET_MPS:
            control = ControlInput(0.0, 1.0, control.steering)
        self.previous_steering = control.steering
        return control

    def restart(self, path: Path) -> None:
        """Follow a new route from wherever the vehicle now is.

        Used after the game recovers a crashed car to the nearest road: it is
        somewhere else entirely, so the route is re-laid rather than the old one
        being chased across the map.
        """
        self.path = path
        self.progress_m = 0.0
        self.previous_steering = 0.0
        self.waiting_until_s = None

    # -- the same questions the hand-built driver answers ------------------

    def deviation_m(self, state: VehicleState) -> float:
        arc_length = self.path.closest_arc_length(
            (state.x_m, state.y_m), near_arc_length=self.progress_m
        )
        near_x, near_y = self.path.point_at(arc_length)
        return math.hypot(near_x - state.x_m, near_y - state.y_m)

    def is_lost(self, state: VehicleState) -> bool:
        return self.deviation_m(state) > self.max_deviation_m

    def is_finished(self, state: VehicleState) -> bool:
        if self._next_stop() is not None:
            return False
        return (
            self.progress_m >= self.path.length_m - 3.0 and state.speed_mps < 0.5
        )
