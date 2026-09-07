"""Recorded routes.

The controller tracks a path to about two centimetres. That is not the same as
driving safely: it will follow the path into a tree if the path goes through
one. The synthetic sine-wave route has no relationship to the road, so the only
honest way to keep the car on tarmac is to record a human driving the road and
follow that line.

A recorded route is the road centreline as far as this project is concerned.
Nothing here perceives obstacles -- see the README on why that needs sensors we
do not have.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Sequence

from control.path import Path

Point = tuple[float, float]

#: Recording at 50 Hz and 10 m/s gives a point every 20 cm. Two metres is dense
#: enough for pure pursuit and keeps path lookups cheap.
DEFAULT_SPACING_M = 2.0


def decimate(points: Sequence[Point], min_spacing_m: float = DEFAULT_SPACING_M) -> list[Point]:
    """Thin a densely sampled drive, keeping the start and the end.

    Also collapses time spent stationary, which otherwise contributes hundreds
    of identical points at every junction.
    """
    if not points:
        raise ValueError("no points to decimate")

    kept: list[Point] = [tuple(points[0])]
    for point in points[1:]:
        if math.dist(kept[-1], point) >= min_spacing_m:
            kept.append(tuple(point))

    if len(kept) < 2:
        raise ValueError(
            f"the vehicle did not move more than {min_spacing_m} m during the recording"
        )

    last = tuple(points[-1])
    if last != kept[-1] and math.dist(kept[-1], last) > 1e-9:
        kept.append(last)
    return kept


def save_route(
    file_path: FilePath | str,
    points: Sequence[Point],
    name: str,
    min_spacing_m: float = DEFAULT_SPACING_M,
    metadata: dict[str, Any] | None = None,
) -> list[Point]:
    """Write a recorded drive out as a route. Returns the thinned points."""
    if len(points) < 2:
        raise ValueError("a route needs at least two points")
    thinned = decimate(points, min_spacing_m)
    file_path = FilePath(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(
            {
                "name": name,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "min_spacing_m": min_spacing_m,
                "raw_point_count": len(points),
                "point_count": len(thinned),
                "length_m": Path(thinned).length_m,
                "points": [[round(x, 3), round(y, 3)] for x, y in thinned],
                **(metadata or {}),
            },
            indent=2,
        )
    )
    return thinned


def load_route(file_path: FilePath | str) -> Path:
    stored = json.loads(FilePath(file_path).read_text())
    return Path([(x, y) for x, y in stored["points"]])
