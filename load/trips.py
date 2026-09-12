"""Trip boundaries, which is the unit the ageing model actually works in.

One forty-minute drive and eight five-minute drives cover the same distance and
are entirely different for an SLI battery: each restart is a crank of several
hundred amps, and a short trip may not run long enough to put back what the
crank took. That is the sulfation pathway, and it is invisible unless the
trajectory is cut into trips.

Two rules that exist to stop the model inventing events:

**A recording that starts mid-drive contributes no crank.** The recorded file
begins with the engine already turning at 3136 rpm. Counting that as a crank
would add a 350 A event that never happened.

**A momentary dip below the running threshold is not a trip boundary.** Engine
speed passes through zero during a stall, a gear change on a rough model, or a
single dropped sample. `MIN_SOAK_S` is what separates a park from a stumble.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

#: Shorter than this and the engine did not really stop.
MIN_SOAK_S = 5.0


@dataclass(frozen=True)
class Segment:
    kind: str  # "trip" or "soak"
    start_s: float
    end_s: float
    cranked: bool


def _runs(samples: Iterable[dict[str, Any]]) -> list[list[Any]]:
    """Collapse samples into [running, start_s, end_s] runs."""
    runs: list[list[Any]] = []
    for sample in samples:
        running = float(sample["engine_running"]) >= 0.5
        t = float(sample["t_s"])
        if runs and runs[-1][0] == running:
            runs[-1][2] = t
        else:
            runs.append([running, t, t])
    return runs


def segment(samples: Iterable[dict[str, Any]], min_soak_s: float = MIN_SOAK_S) -> list[Segment]:
    """Cut a trajectory into trips and soaks, marking which trips were cranked."""
    runs = _runs(samples)
    if not runs:
        return []

    # Pass 1: Absorb too-short stops backward into preceding trips.
    merged: list[list[Any]] = []
    for run in runs:
        running, start, end = run
        too_short = (not running) and (end - start) < min_soak_s
        if too_short and merged and merged[-1][0]:
            # Absorb backward into preceding trip
            merged[-1][2] = end
            continue
        if merged and merged[-1][0] == running:
            # Merge consecutive same-state runs
            merged[-1][2] = end
            continue
        merged.append(list(run))

    # Pass 2: Absorb too-short stops forward into following trips.
    # This handles leading dips that have no preceding trip to absorb into.
    final: list[list[Any]] = []
    i = 0
    while i < len(merged):
        run = merged[i]
        running, start, end = run
        too_short = (not running) and (end - start) < min_soak_s

        if too_short and i + 1 < len(merged):
            # Absorb forward: extend the next run's start back to include this dip
            next_run = merged[i + 1]
            next_run[1] = start
            i += 1
            continue

        final.append(run)
        i += 1

    segments: list[Segment] = []
    for index, (running, start, end) in enumerate(final):
        # A trip is cranked only if we watched the engine stop beforehand.
        cranked = bool(running and index > 0 and not final[index - 1][0])
        segments.append(
            Segment(
                kind="trip" if running else "soak",
                start_s=start,
                end_s=end,
                cranked=cranked,
            )
        )
    return segments


def crank_times(segments: Iterable[Segment]) -> tuple[float, ...]:
    """When the starter turned, as far as we actually observed."""
    return tuple(s.start_s for s in segments if s.cranked)
