"""Geometric path tracking: steer toward a point one lookahead ahead."""

from __future__ import annotations

import math

Point = tuple[float, float]
Pose = tuple[float, float, float]


class PurePursuit:
    def __init__(
        self,
        wheelbase_m: float,
        max_steer_rad: float,
        lookahead_gain_s: float = 0.6,
        min_lookahead_m: float = 5.0,
        max_lookahead_m: float = 30.0,
    ) -> None:
        self.wheelbase_m = wheelbase_m
        self.max_steer_rad = max_steer_rad
        self.lookahead_gain_s = lookahead_gain_s
        self.min_lookahead_m = min_lookahead_m
        self.max_lookahead_m = max_lookahead_m

    def lookahead_m(self, speed_mps: float) -> float:
        return min(
            self.max_lookahead_m,
            max(self.min_lookahead_m, self.lookahead_gain_s * speed_mps),
        )

    def steering(self, pose: Pose, target: Point) -> float:
        """Normalised steering command in [-1, 1], positive left."""
        x, y, heading = pose
        dx, dy = target[0] - x, target[1] - y
        # Into the vehicle frame.
        forward = dx * math.cos(heading) + dy * math.sin(heading)
        left = -dx * math.sin(heading) + dy * math.cos(heading)
        lookahead = math.hypot(forward, left)
        if lookahead == 0.0:
            return 0.0
        alpha = math.atan2(left, forward)
        steer_rad = math.atan2(2.0 * self.wheelbase_m * math.sin(alpha), lookahead)
        return min(1.0, max(-1.0, steer_rad / self.max_steer_rad))
