"""Drive the clock, drain the source, write the rows.

The clock and the sleep are injected so tests never wait on real time. A
fifteen-minute run is not something a suite can afford to reproduce, and a loop
that can only be tested by waiting is a loop that stops being tested.
"""

from __future__ import annotations

import time
from pathlib import Path

from collect.scenario import ScenarioSpec
from collect.writer import TrajectoryLog

#: How often to drain. The Lua buffer holds a minute of slack, so this bounds
#: loss on a crash rather than keeping up with the sampler.
DRAIN_INTERVAL_S = 1.0


def collect_run(
    source,
    spec: ScenarioSpec,
    csv_path: Path | str,
    seconds: float,
    clock=time.monotonic,
    sleep=time.sleep,
    beamng: dict | None = None,
    on_progress=None,
) -> dict:
    """Collect for `seconds`, and return the run summary."""
    started = clock()
    with TrajectoryLog(csv_path, spec, source.interval_s, beamng=beamng) as log:
        source.start()
        try:
            while clock() - started < seconds:
                sleep(DRAIN_INTERVAL_S)
                log.write(source.drain())
                if on_progress:
                    on_progress(log.rows, clock() - started, seconds)
            # Whatever is still buffered when time expires is still data.
            log.write(source.drain())
        except KeyboardInterrupt:
            # Deliberate: everything flushed stays, and the sidecar is written
            # on the way out. Stopping early costs the last drain.
            print("\n  stopped early. What was collected is on disk.")
        finally:
            source.stop()
            log.dropped = getattr(source, "dropped", 0)
            log.capture_error = getattr(source, "capture_error", None)
        return log.summary()
