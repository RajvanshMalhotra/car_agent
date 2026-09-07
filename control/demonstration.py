"""A route learned by watching someone drive it.

Drive A to B once. The recording keeps where you went *and how fast you went
there*, and that speed profile becomes the baseline a behaviour modulates: the
same trip, driven aggressively or economically.

This removes the part of the project most prone to being wrong. There is no
reward function to design -- and nearly every failure while building the learned
policy came from designing one badly: rewarding progress against the clock
taught it to pin the throttle, and a stop window at the end of each episode
taught it to brake whenever it pointed off-line and then sit there. A
demonstration has no reward to get wrong. It also has no sim-to-real gap,
because it was recorded on the actual vehicle in the actual game.

Where the human stopped is kept as a stop. Standing at a junction is data, not
noise: idling with a hot engine is the mechanism this project exists to measure.
"""

from __future__ import annotations

import bisect
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path as FilePath
from typing import Any, Sequence

from behaviour.route_spec import Stop
from behaviour.spec import BehaviourSpec
from control.path import Path

#: Below this the vehicle counts as stationary.
STOPPED_SPEED_MPS = 0.5

#: A pause shorter than this is hesitation, not a stop worth reproducing.
MIN_STOP_S = 3.0

#: Samples closer together than this add nothing but lookup cost.
MIN_SPACING_M = 2.0


@dataclass(frozen=True)
class Demonstration:
    name: str
    points: tuple[tuple[float, float], ...]
    speeds: tuple[float, ...]
    arc_lengths: tuple[float, ...]
    stop_list: tuple[Stop, ...] = ()

    # -- building ---------------------------------------------------------

    @classmethod
    def from_samples(
        cls,
        samples: Sequence[tuple[float, float, float]],
        name: str,
        sample_hz: float = 10.0,
        min_spacing_m: float = MIN_SPACING_M,
    ) -> "Demonstration":
        """Build from `(x, y, speed)` samples taken at a steady rate."""
        if len(samples) < 2:
            raise ValueError("a demonstration needs at least two samples")

        points: list[tuple[float, float]] = [(samples[0][0], samples[0][1])]
        speeds: list[float] = [samples[0][2]]
        arc: list[float] = [0.0]

        # Stationary stretches are collapsed to a single point and recorded as
        # a stop, so standing still does not become hundreds of duplicate
        # samples -- and is not silently discarded either.
        stops: list[Stop] = []
        still_from: int | None = None

        for index, (x, y, speed) in enumerate(samples[1:], start=1):
            if speed < STOPPED_SPEED_MPS:
                if still_from is None:
                    still_from = index
                continue
            if still_from is not None:
                seconds = (index - still_from) / sample_hz
                if seconds >= MIN_STOP_S:
                    stops.append(Stop(arc_length_m=arc[-1], duration_s=seconds))
                still_from = None

            step = math.dist(points[-1], (x, y))
            if step >= min_spacing_m:
                points.append((x, y))
                speeds.append(speed)
                arc.append(arc[-1] + step)

        if still_from is not None:
            seconds = (len(samples) - still_from) / sample_hz
            if seconds >= MIN_STOP_S:
                stops.append(Stop(arc_length_m=arc[-1], duration_s=seconds))

        if len(points) < 2:
            raise ValueError(
                "the vehicle did not move far enough during the recording"
            )
        return cls(
            name=name,
            points=tuple(points),
            speeds=tuple(speeds),
            arc_lengths=tuple(arc),
            stop_list=tuple(stops),
        )

    # -- reading it back --------------------------------------------------

    def to_path(self) -> Path:
        return Path(list(self.points))

    def stops(self) -> list[Stop]:
        return list(self.stop_list)

    @property
    def length_m(self) -> float:
        return self.arc_lengths[-1]

    @property
    def mean_speed_mps(self) -> float:
        return sum(self.speeds) / len(self.speeds)

    def speed_at(self, arc_length_m: float) -> float:
        """The speed the human held here, interpolated between samples."""
        if arc_length_m <= self.arc_lengths[0]:
            return self.speeds[0]
        if arc_length_m >= self.arc_lengths[-1]:
            return self.speeds[-1]
        index = bisect.bisect_left(self.arc_lengths, arc_length_m)
        before, after = index - 1, index
        span = self.arc_lengths[after] - self.arc_lengths[before]
        if span <= 0:
            return self.speeds[before]
        t = (arc_length_m - self.arc_lengths[before]) / span
        return self.speeds[before] + t * (self.speeds[after] - self.speeds[before])

    # -- driving it in a style --------------------------------------------

    def target_speed_at(self, arc_length_m: float, spec: BehaviourSpec) -> float:
        """What this behaviour should be doing here.

        The human's own speed profile scaled by the behaviour's
        `target_speed_factor`, so the *shape* of the drive survives: where the
        human slowed for a bend or a junction, the agent still slows -- it just
        does the whole trip more or less briskly.

        A stop stays a stop at any style. Reproducing "aggressive" by driving
        through the junction the human waited at would be reproducing something
        they did not do.
        """
        for stop in self.stop_list:
            if abs(arc_length_m - stop.arc_length_m) < MIN_SPACING_M * 2:
                return 0.0
        return max(0.0, self.speed_at(arc_length_m) * spec.target_speed_factor)


def save_demonstration(
    file_path: FilePath | str,
    demonstration: Demonstration,
    metadata: dict[str, Any] | None = None,
) -> None:
    file_path = FilePath(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(
            {
                "name": demonstration.name,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "length_m": round(demonstration.length_m, 2),
                "mean_speed_mps": round(demonstration.mean_speed_mps, 3),
                "point_count": len(demonstration.points),
                "points": [[round(x, 3), round(y, 3)] for x, y in demonstration.points],
                "speeds": [round(v, 3) for v in demonstration.speeds],
                "arc_lengths": [round(s, 3) for s in demonstration.arc_lengths],
                "stops": [
                    {"arc_length_m": round(s.arc_length_m, 2),
                     "duration_s": round(s.duration_s, 2)}
                    for s in demonstration.stop_list
                ],
                "metadata": metadata or {},
            },
            indent=2,
        )
    )


def load_demonstration(file_path: FilePath | str) -> Demonstration:
    stored = json.loads(FilePath(file_path).read_text())
    return Demonstration(
        name=stored["name"],
        points=tuple((x, y) for x, y in stored["points"]),
        speeds=tuple(stored["speeds"]),
        arc_lengths=tuple(stored["arc_lengths"]),
        stop_list=tuple(
            Stop(arc_length_m=s["arc_length_m"], duration_s=s["duration_s"])
            for s in stored.get("stops", [])
        ),
    )
