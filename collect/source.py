"""Where trajectory samples come from.

One interface, two implementations. `MCPTrajectorySource` talks to the game.
`FakeTrajectorySource` does not, which is what makes every layer above this
testable on a machine with no BeamNG and no Windows laptop.

The fake is deliberately crude -- constant speed on a constant grade -- because
its job is to exercise the plumbing, not to model a car. Anything needing real
dynamics needs the real game.
"""

from __future__ import annotations

import math
import time
from abc import ABC, abstractmethod

from collect.derive import GRAVITY_MPS2, add_derived
from collect.schema import MEASURED


class TrajectorySource(ABC):
    """Start sampling, drain what has accumulated, stop."""

    dropped: int = 0
    interval_s: float = 0.01

    @abstractmethod
    def start(self) -> None:
        """Begin accumulating samples."""

    @abstractmethod
    def drain(self) -> list[dict]:
        """Hand over everything buffered since the last drain, and empty it."""

    @abstractmethod
    def stop(self) -> None:
        """Stop sampling. Safe to call more than once."""


class FakeTrajectorySource(TrajectorySource):
    def __init__(
        self,
        interval_s: float = 0.01,
        speed_mps: float = 20.0,
        grade_rad: float = 0.0,
        auto: bool = True,
        clock=time.monotonic,
    ) -> None:
        self.interval_s = interval_s
        #: With `auto`, the fake catches up to the wall clock on every drain, so
        #: the CLI produces data without the game. Tests turn it off and step it
        #: by hand, because a test that waits on real time is a test that gets
        #: deleted.
        self.auto = auto
        self.clock = clock
        self._last_wall: float | None = None
        self.speed_mps = speed_mps
        self.grade_rad = grade_rad
        self.running = False
        self.dropped = 0
        self.t = 0.0
        self.seq = 0
        self._buffer: list[dict] = []

    def start(self) -> None:
        self.running = True
        self._last_wall = self.clock()

    def advance(self, seconds: float) -> None:
        """Run the fake forward. Only the fake has this -- the game has time."""
        if not self.running:
            return
        for _ in range(int(round(seconds / self.interval_s))):
            self.seq += 1
            self.t += self.interval_s
            self._buffer.append(add_derived(self._sample()))

    def _sample(self) -> dict:
        travelled = self.speed_mps * (self.t - self.interval_s)
        sample = {c.name: 0.0 for c in MEASURED}
        sample.update(
            t_s=self.t,
            seq=self.seq,
            x_m=travelled * math.cos(self.grade_rad),
            z_m=travelled * math.sin(self.grade_rad),
            speed_mps=self.speed_mps,
            vx_mps=self.speed_mps,
            dir_x=math.cos(self.grade_rad),
            dir_z=math.sin(self.grade_rad),
            # A real accelerometer reads gravity. Steady speed on a slope is
            # all gravity and no acceleration.
            ax_mps2=GRAVITY_MPS2 * math.sin(self.grade_rad),
            az_mps2=-GRAVITY_MPS2 * math.cos(self.grade_rad),
            mass_kg=1510.62,
            rpm=2500.0,
            coolant_c=90.0,
            engine_running=1.0,
            ignition_level=2.0,
        )
        return sample

    def drain(self) -> list[dict]:
        if self.auto and self.running:
            now = self.clock()
            self.advance(now - (self._last_wall if self._last_wall is not None else now))
            self._last_wall = now
        taken, self._buffer = self._buffer, []
        return taken

    def stop(self) -> None:
        self.running = False
