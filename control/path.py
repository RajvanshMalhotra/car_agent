"""A route as a polyline, with the arc-length lookups pure pursuit needs."""

from __future__ import annotations

import bisect
import math
from typing import Sequence

Point = tuple[float, float]


class Path:
    def __init__(self, points: Sequence[Point]) -> None:
        if len(points) < 2:
            raise ValueError("a path needs at least two points")
        self.points: list[Point] = [(float(x), float(y)) for x, y in points]
        self.arc_lengths: list[float] = [0.0]
        for (x0, y0), (x1, y1) in zip(self.points, self.points[1:]):
            self.arc_lengths.append(self.arc_lengths[-1] + math.hypot(x1 - x0, y1 - y0))

    @property
    def length_m(self) -> float:
        return self.arc_lengths[-1]

    def _segment_range(self, near_arc_length: float | None, window_m: float):
        """Segment indices worth searching, given a progress hint."""
        if near_arc_length is None:
            return 0, len(self.points) - 1
        low = bisect.bisect_left(self.arc_lengths, near_arc_length - window_m) - 1
        high = bisect.bisect_right(self.arc_lengths, near_arc_length + window_m)
        return max(0, low), min(len(self.points) - 1, high)

    def closest_arc_length(
        self,
        position: Point,
        near_arc_length: float | None = None,
        window_m: float = 60.0,
    ) -> float:
        """Arc length of the point on the path nearest `position`.

        `near_arc_length` restricts the search to a window around known
        progress, which turns an O(n) scan into a constant-cost one. Callers
        that track their own progress should pass it.
        """
        px, py = position
        start, stop = self._segment_range(near_arc_length, window_m)
        best_distance = math.inf
        best_s = self.arc_lengths[start]
        for index in range(start, stop):
            (x0, y0), (x1, y1) = self.points[index], self.points[index + 1]
            dx, dy = x1 - x0, y1 - y0
            segment_length_sq = dx * dx + dy * dy
            if segment_length_sq == 0.0:
                continue
            t = ((px - x0) * dx + (py - y0) * dy) / segment_length_sq
            t = min(1.0, max(0.0, t))
            distance = math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
            if distance < best_distance:
                best_distance = distance
                best_s = self.arc_lengths[index] + t * math.sqrt(segment_length_sq)
        return best_s

    def point_at(self, arc_length_m: float) -> Point:
        """Interpolate the path at an arc length, clamped to its ends."""
        s = min(self.length_m, max(0.0, arc_length_m))
        index = min(
            bisect.bisect_left(self.arc_lengths, s), len(self.points) - 1
        )
        if index == 0:
            return self.points[0]
        s0, s1 = self.arc_lengths[index - 1], self.arc_lengths[index]
        span = s1 - s0
        t = 0.0 if span == 0.0 else (s - s0) / span
        (x0, y0), (x1, y1) = self.points[index - 1], self.points[index]
        return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

    def target_point(
        self,
        position: Point,
        lookahead_m: float,
        near_arc_length: float | None = None,
    ) -> Point:
        return self.point_at(
            self.closest_arc_length(position, near_arc_length) + lookahead_m
        )

    def is_finished(self, position: Point, tolerance_m: float = 0.5) -> bool:
        return self.closest_arc_length(position) >= self.length_m - tolerance_m

    def curvature_at(self, arc_length_m: float, window_m: float = 1.0) -> float:
        """Menger curvature from three samples straddling `arc_length_m`."""
        ax, ay = self.point_at(arc_length_m - window_m)
        bx, by = self.point_at(arc_length_m)
        cx, cy = self.point_at(arc_length_m + window_m)
        a = math.hypot(bx - ax, by - ay)
        b = math.hypot(cx - bx, cy - by)
        c = math.hypot(cx - ax, cy - ay)
        if a == 0.0 or b == 0.0 or c == 0.0:
            return 0.0
        # Twice the triangle area via the cross product.
        area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
        return 2.0 * area2 / (a * b * c)
