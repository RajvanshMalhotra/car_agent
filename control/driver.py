"""Executes a BehaviourSpec: pure-pursuit steering + PID speed control.

The LLM is nowhere near this loop. A spec arrives already bounds-checked, and
this module turns it into throttle/brake/steering at every timestep.
"""

from __future__ import annotations

import math
import random

from behaviour.spec import BehaviourSpec
from control.path import Path
from control.pid import PID
from control.pure_pursuit import PurePursuit
from sim.backend import ControlInput, VehicleState

#: Lateral acceleration a driver of nominal confidence will accept in a bend.
#: `corner_speed_factor` scales it, so timid behaviours brake earlier.
NOMINAL_LATERAL_ACCEL_MPS2 = 3.0

#: Wander is an Ornstein-Uhlenbeck process; this is its correlation time. Slow
#: enough to read as an inattentive driver rather than as sensor noise.
WANDER_TIME_CONSTANT_S = 4.0

#: Lane keeping is a continuous compensatory tracking task, limited by
#: neuromuscular delay rather than by reaction time. `reaction_lag_s` -- which
#: governs response to discrete events -- must not be fed into the steering
#: loop: past about 0.4 s it oscillates, and by 1.5 s it diverges entirely.
NEUROMUSCULAR_LAG_S = 0.15

#: Speed-target deviation at `erraticness` 1.0, as a fraction, one sigma.
WANDER_SCALE = 0.15

#: How far off the route the car may stray before the run is abandoned. Nothing
#: in this project perceives obstacles, so noticing the car is no longer near
#: the road -- and stopping -- is the only honest safety response available.
DEFAULT_MAX_DEVIATION_M = 8.0

#: Pure pursuit needs road to turn onto the line. Judging deviation before the
#: car has travelled this far aborts runs that would have converged fine --
#: which is exactly what happens when a run starts with a heading error.
DEFAULT_LOST_GRACE_M = 40.0

#: Aim to stop this far before the route end. Without the margin the stopping
#: profile is exactly critical, and any loop lag turns into an overshoot.
STOP_MARGIN_M = 2.0

#: Fraction of the behaviour's deceleration limit the stopping profile plans
#: on. The remainder is authority held back for the controller to correct with:
#: planning on the full limit overshoots the stop point by metres.
STOP_DECEL_RESERVE = 0.5

#: A waypoint counts as reached once the car is within this of it and stopped.
STOP_REACHED_M = 8.0


class Driver:
    def __init__(
        self,
        spec: BehaviourSpec,
        path: Path,
        dt: float,
        speed_limit_mps: float,
        seed: int = 0,
        wheelbase_m: float = 2.7,
        max_steer_rad: float = 0.5,
        max_accel_mps2: float = 3.0,
        max_decel_mps2: float = 8.0,
        max_deviation_m: float = DEFAULT_MAX_DEVIATION_M,
        lost_grace_m: float = DEFAULT_LOST_GRACE_M,
        stops: "Sequence[Any] | None" = None,
        gains: "Any | None" = None,
    ) -> None:
        self.spec = spec
        self.path = path
        self.dt = dt
        self.speed_limit_mps = speed_limit_mps
        self.max_accel_mps2 = max_accel_mps2
        self.max_decel_mps2 = max_decel_mps2
        self.max_deviation_m = max_deviation_m
        self.lost_grace_m = lost_grace_m
        self.travelled_m = 0.0
        # Waypoint stops: what produces idle time, braking and re-acceleration.
        # Without them `idle_fraction` is recorded in the spec and never happens.
        self.stops = list(stops or [])
        self.stop_index = 0
        self.waiting_until_s: float | None = None
        self._previous_position: tuple[float, float] | None = None

        # Gains default to the hand-tuned passenger-car values; `control.tuner`
        # learns better ones for a vehicle whose limits differ.
        if gains is None:
            from control.tuner import ControllerGains

            gains = ControllerGains.default()
        self.gains = gains
        self.steering_controller = PurePursuit(
            wheelbase_m,
            max_steer_rad,
            lookahead_gain_s=gains.lookahead_gain_s,
            min_lookahead_m=gains.min_lookahead_m,
        )
        self.speed_controller = PID(
            kp=gains.kp, ki=gains.ki, kd=gains.kd, integral_limit=4.0
        )
        self.rng = random.Random(seed)

        self.lag_steps = int(round(spec.reaction_lag_s / dt))
        self.steering_lag_steps = int(
            round(min(NEUROMUSCULAR_LAG_S, spec.reaction_lag_s) / dt)
        )
        self.observed: list[VehicleState] = []
        self.commanded_accel_mps2 = 0.0
        self.wander = 0.0
        self.progress_m = 0.0
        self.decision_progress_m = 0.0

    # -- perception -------------------------------------------------------

    def _perceive(self, state: VehicleState) -> tuple[VehicleState, VehicleState]:
        """The two views a driver acts on.

        Returns `(decision, steering)`: what the driver has had time to think
        about (`reaction_lag_s` in the past) and what their hands are tracking
        (a neuromuscular delay in the past).
        """
        self.observed.append(state.snapshot())
        if len(self.observed) > self.lag_steps + 1:
            self.observed.pop(0)
        steering_index = max(0, len(self.observed) - 1 - self.steering_lag_steps)
        return self.observed[0], self.observed[steering_index]

    # -- speed target -----------------------------------------------------

    def _corner_speed_mps(self, arc_length_m: float) -> float:
        """Speed the bend ahead allows, looked at over the braking distance."""
        horizon_m = max(10.0, self.speed_limit_mps * self.spec.following_distance_s)
        lateral_accel = NOMINAL_LATERAL_ACCEL_MPS2 * self.spec.corner_speed_factor
        samples = 8
        limit = math.inf
        for i in range(samples + 1):
            curvature = self.path.curvature_at(arc_length_m + horizon_m * i / samples)
            if curvature > 1e-6:
                limit = min(limit, math.sqrt(lateral_accel / curvature))
        return limit

    def _wander_factor(self) -> float:
        """Bounded drift on the speed target, driven by `erraticness`.

        Discretised so the process has unit variance regardless of `dt`;
        clipped at two sigma so a long run cannot produce an absurd target.
        """
        decay = math.exp(-self.dt / WANDER_TIME_CONSTANT_S)
        self.wander = decay * self.wander + math.sqrt(1 - decay * decay) * self.rng.gauss(0.0, 1.0)
        self.wander = min(2.0, max(-2.0, self.wander))
        return 1.0 + WANDER_SCALE * self.spec.erraticness * self.wander

    def _next_stop(self):
        """The stop the vehicle is still working towards, if any."""
        if self.stop_index < len(self.stops):
            return self.stops[self.stop_index]
        return None

    def _update_stop(self, arc_length_m: float, state: VehicleState) -> None:
        stop = self._next_stop()
        if stop is None:
            return
        reached = (
            arc_length_m >= stop.arc_length_m - STOP_REACHED_M
            and state.speed_mps < 0.5
        )
        if self.waiting_until_s is None:
            if reached:
                self.waiting_until_s = state.sim_time_s + stop.duration_s
        elif state.sim_time_s >= self.waiting_until_s:
            self.waiting_until_s = None
            self.stop_index += 1

    def _target_speed_mps(self, arc_length_m: float) -> float:
        arc_length = arc_length_m
        target = self.speed_limit_mps * self.spec.target_speed_factor
        target = min(target, self._corner_speed_mps(arc_length))
        target *= self._wander_factor()

        # Bleed off speed so the vehicle arrives at the route end stopped,
        # using this behaviour's own comfortable deceleration.
        # Whichever comes first: the next waypoint stop, or the route end.
        stop_at = self.path.length_m
        stop = self._next_stop()
        if stop is not None:
            stop_at = min(stop_at, stop.arc_length_m)
        remaining = max(0.0, stop_at - arc_length - STOP_MARGIN_M)
        planned_decel = STOP_DECEL_RESERVE * self.spec.decel_limit_mps2
        target = min(target, math.sqrt(2 * planned_decel * remaining))
        return max(0.0, target)

    # -- control ----------------------------------------------------------

    def step(self, state: VehicleState) -> ControlInput:
        position = (state.x_m, state.y_m)
        if self._previous_position is not None:
            self.travelled_m += math.dist(position, self._previous_position)
        self._previous_position = position

        decision, steering = self._perceive(state)
        # Progress is tracked so path lookups stay local rather than scanning
        # the whole route on every timestep.
        self.progress_m = self.path.closest_arc_length(
            (steering.x_m, steering.y_m), near_arc_length=self.progress_m
        )
        self.decision_progress_m = self.path.closest_arc_length(
            (decision.x_m, decision.y_m), near_arc_length=self.decision_progress_m
        )
        self._update_stop(self.progress_m, state)
        if self.waiting_until_s is not None:
            target_speed = 0.0
        else:
            target_speed = self._target_speed_mps(self.decision_progress_m)

        effort = self.speed_controller.update(
            error=target_speed - decision.speed_mps, dt=self.dt
        )
        desired_accel = effort * (
            self.spec.accel_limit_mps2 if effort >= 0 else self.spec.decel_limit_mps2
        )

        # Jerk limit shapes how fast the pedal demand itself may change.
        max_change = self.spec.jerk_limit_mps3 * self.dt
        delta = desired_accel - self.commanded_accel_mps2
        self.commanded_accel_mps2 += min(max_change, max(-max_change, delta))
        accel = min(
            self.spec.accel_limit_mps2,
            max(-self.spec.decel_limit_mps2, self.commanded_accel_mps2),
        )

        if accel >= 0.0:
            throttle, brake = min(1.0, accel / self.max_accel_mps2), 0.0
        else:
            throttle, brake = 0.0, min(1.0, -accel / self.max_decel_mps2)

        target_point = self.path.point_at(
            self.progress_m + self.steering_controller.lookahead_m(steering.speed_mps)
        )
        steering_command = self.steering_controller.steering(
            (steering.x_m, steering.y_m, steering.heading_rad), target_point
        )
        return ControlInput(
            throttle=throttle, brake=brake, steering=steering_command
        )

    def deviation_m(self, state: VehicleState) -> float:
        """How far the vehicle is from the route it is meant to be on."""
        arc_length = self.path.closest_arc_length(
            (state.x_m, state.y_m), near_arc_length=self.progress_m
        )
        near_x, near_y = self.path.point_at(arc_length)
        return math.hypot(near_x - state.x_m, near_y - state.y_m)

    def is_lost(self, state: VehicleState) -> bool:
        """True once the vehicle is too far off the route to be recovered.

        This is not obstacle avoidance -- there is no perception here. It only
        catches the case where the car has clearly left the road.
        """
        if self.travelled_m < self.lost_grace_m:
            return False
        return self.deviation_m(state) > self.max_deviation_m

    def is_finished(self, state: VehicleState) -> bool:
        arc_length = self.path.closest_arc_length(
            (state.x_m, state.y_m), near_arc_length=self.progress_m
        )
        # The route counts as done once the vehicle is stopped inside the
        # planned stopping margin -- it deliberately never reaches the end.
        if self._next_stop() is not None:
            return False  # still stops to make
        return (
            arc_length >= self.path.length_m - (STOP_MARGIN_M + 1.0)
            and state.speed_mps < 0.5
        )
