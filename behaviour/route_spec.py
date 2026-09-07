"""LLM-designed manoeuvre routes.

On an empty grid there is no road to follow, so the route is whatever we invent
-- which makes it a good fit for the LLM. It designs a sequence of manoeuvres
and the geometry is built from that deterministically. Same discipline as
`BehaviourSpec`: generated once, offline, bounds-checked in code, cached.

Stops matter more than they look. They are what produce idle time, brake and
acceleration cycling and trip structure -- the channels an SLI battery actually
ages under, and precisely the ones a constant-speed straight line never
exercises.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from control.path import Path

#: Geometry is sampled at this spacing. Fine enough for curvature to be
#: meaningful, coarse enough to keep path lookups cheap.
SAMPLE_SPACING_M = 2.0

BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "straight": {"length_m": (5.0, 2000.0)},
    "turn": {"radius_m": (5.0, 500.0), "angle_deg": (-180.0, 180.0)},
    "stop": {"duration_s": (1.0, 600.0)},
}

MIN_TURN_ANGLE_DEG = 5.0


class RouteValidationError(ValueError):
    """A route segment was malformed or outside its physical bounds."""


@dataclass(frozen=True)
class Stop:
    arc_length_m: float
    duration_s: float


@dataclass(frozen=True)
class RouteSpec:
    name: str
    segments: Sequence[dict[str, Any]] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple(dict(s) for s in self.segments))
        violations: list[str] = []

        if len(self.segments) < 2:
            raise RouteValidationError("a route needs at least two segments")

        for index, segment in enumerate(self.segments):
            kind = segment.get("type")
            if kind not in BOUNDS:
                violations.append(f"segment {index}: unknown type {kind!r}")
                continue
            for name, (low, high) in BOUNDS[kind].items():
                if name not in segment:
                    violations.append(f"segment {index}: missing {name}")
                    continue
                value = segment[name]
                if not low <= value <= high:
                    violations.append(
                        f"segment {index}: {name} {value} outside [{low}, {high}]"
                    )
            if kind == "turn" and abs(segment.get("angle_deg", 0.0)) < MIN_TURN_ANGLE_DEG:
                violations.append(
                    f"segment {index}: angle_deg {segment.get('angle_deg')} is "
                    f"below the {MIN_TURN_ANGLE_DEG} degree minimum"
                )

        if violations:
            raise RouteValidationError("; ".join(violations))

        if not any(s["type"] in ("straight", "turn") for s in self.segments):
            raise RouteValidationError("the route never moves: it is all stops")

    # -- geometry ---------------------------------------------------------

    def to_path(
        self, origin: tuple[float, float] = (0.0, 0.0), heading_rad: float = 0.0
    ) -> Path:
        points = [origin]
        x, y = origin
        heading = heading_rad

        for segment in self.segments:
            if segment["type"] == "straight":
                length = segment["length_m"]
                steps = max(1, int(length / SAMPLE_SPACING_M))
                for _ in range(steps):
                    step = length / steps
                    x += step * math.cos(heading)
                    y += step * math.sin(heading)
                    points.append((x, y))
            elif segment["type"] == "turn":
                radius = segment["radius_m"]
                sweep = math.radians(segment["angle_deg"])
                arc = abs(sweep) * radius
                steps = max(2, int(arc / SAMPLE_SPACING_M))
                for _ in range(steps):
                    step_angle = sweep / steps
                    # Advance along the chord, then rotate: for small steps this
                    # is the arc to within a fraction of the sample spacing.
                    step = arc / steps
                    heading += step_angle / 2
                    x += step * math.cos(heading)
                    y += step * math.sin(heading)
                    heading += step_angle / 2
                    points.append((x, y))
            # A stop adds no distance; it is a speed instruction, not geometry.

        return Path(points)

    def stops(self) -> list[Stop]:
        """Where the vehicle should stop, and for how long."""
        found: list[Stop] = []
        distance = 0.0
        for segment in self.segments:
            if segment["type"] == "straight":
                distance += segment["length_m"]
            elif segment["type"] == "turn":
                distance += abs(math.radians(segment["angle_deg"])) * segment["radius_m"]
            else:
                found.append(Stop(arc_length_m=distance, duration_s=segment["duration_s"]))
        return found

    def total_stop_time_s(self) -> float:
        return sum(stop.duration_s for stop in self.stops())

    # -- reproducibility --------------------------------------------------

    @property
    def route_hash(self) -> str:
        encoded = json.dumps(
            [dict(sorted(s.items())) for s in self.segments],
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "segments": [dict(s) for s in self.segments]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RouteSpec":
        missing = {"name", "segments"} - set(data)
        if missing:
            raise RouteValidationError(f"missing fields: {sorted(missing)}")
        return cls(name=data["name"], segments=data["segments"])
