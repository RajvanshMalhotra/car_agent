"""Telemetry that stopped arriving, as distinct from telemetry that says zero.

The recorded file ends with 14,283 contiguous rows -- 150.8 s -- in which every
OutGauge channel is exactly zero at once, while MotionSim keeps streaming a
frozen position. Fuel goes from 0.9256 to 0.0 between two samples 10 ms apart
and coolant from 101.77 C to 0.0. No engine does that. OutGauge stopped
sending.

Consumed as data it would read as the engine bay collapsing to ambient during
the hottest part of the soak, in a corrosion process that is exponential in
temperature. That is the most damaging thing that could silently enter this
pipeline, so it is detected rather than trusted.

Excluded, not interpolated. We do not know what the bay did during those
seconds, and a plausible decay curve drawn across the gap would be a
fabrication of exactly the measurement the model exists to consume.

The test is that *every* watched channel is zero simultaneously. One zero on
its own is a measurement -- an empty tank is a real state -- and only the whole
packet going dark indicates no packet at all.

Not legacy-only: a Lua drain can drop too, so this filter is its own module
rather than a special case buried in `load/legacy.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

#: The channels an OutGauge packet carries that cannot all be truly zero at
#: once. Coolant is a real temperature even on a stone-cold engine.
OUTGAUGE_CHANNELS: tuple[str, ...] = ("coolant_c", "oil_c", "fuel_volume_l")

#: Floating-point slack. These arrive as exact zeros, not small numbers.
ZERO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Dropout:
    start_s: float
    end_s: float
    rows: int


class DropoutFilter:
    """Drops samples where the watched channels are all simultaneously zero.

    Stateful so a caller can read `spans` afterwards and record them in the
    sidecar. An excluded span that nothing reports is indistinguishable from
    data that was never collected.
    """

    def __init__(self, channels: tuple[str, ...] = OUTGAUGE_CHANNELS) -> None:
        self.channels = channels
        self._spans: list[Dropout] = []

    @property
    def spans(self) -> tuple[Dropout, ...]:
        return tuple(self._spans)

    @property
    def rows_excluded(self) -> int:
        return sum(span.rows for span in self._spans)

    def _is_dropout(self, sample) -> bool:
        return all(
            abs(float(sample[channel])) < ZERO_TOLERANCE
            for channel in self.channels
        )

    def __call__(self, samples: Iterable) -> Iterator:
        open_span: list | None = None
        for sample in samples:
            if self._is_dropout(sample):
                t = float(sample["t_s"])
                if open_span is None:
                    open_span = [t, t, 0]
                open_span[1] = t
                open_span[2] += 1
                continue
            if open_span is not None:
                self._spans.append(
                    Dropout(start_s=open_span[0], end_s=open_span[1], rows=open_span[2])
                )
                open_span = None
            yield sample
        if open_span is not None:
            self._spans.append(
                Dropout(start_s=open_span[0], end_s=open_span[1], rows=open_span[2])
            )
