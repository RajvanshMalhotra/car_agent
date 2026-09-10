"""A trajectory CSV and the sidecar that makes it interpretable.

Two requirements shape this, both learned the hard way.

**A log must be interpretable on its own.** A temperature trace with no ambient
recorded beside it is close to useless, and a column of numbers that does not
say whether it was measured or computed will eventually be presented as a
measurement. The sidecar carries the whole channel manifest, provenance
included, and downstream code reads it.

**A run must survive a crash.** Runs are fifteen minutes of real time on
someone else's laptop. Rows are flushed as they are written and the sidecar is
written even when the run raises, so a failure at minute fourteen costs the
last drain rather than the run.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

from collect.scenario import ScenarioSpec
from collect.schema import COLUMNS, manifest


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class TrajectoryLog:
    def __init__(
        self,
        csv_path: Path | str,
        spec: ScenarioSpec,
        interval_s: float,
        beamng: dict | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.sidecar_path = self.csv_path.with_suffix(".json")
        self.spec = spec
        self.interval_s = interval_s
        self.beamng = beamng or {}
        self.rows = 0
        self.dropped = 0
        self.capture_error: str | None = None

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.csv_path.open("w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(COLUMNS)
        self._handle.flush()

    def __enter__(self) -> "TrajectoryLog":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def write(self, samples: list[dict]) -> None:
        for sample in samples:
            # A missing channel is a schema bug. Writing a blank would produce a
            # column that silently means nothing.
            self._writer.writerow([sample[name] for name in COLUMNS])
            self.rows += 1
        self._handle.flush()

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "duration_s": round(self.rows * self.interval_s, 3),
            "dropped": self.dropped,
            "sample_hz": round(1.0 / self.interval_s, 1) if self.interval_s else 0.0,
            "capture_error": self.capture_error,
        }

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.close()
        # This class exists so a run survives a crash. It must not be the thing
        # that crashes: the CSV is already on disk, so a sidecar that cannot be
        # written is reported and swallowed.
        try:
            self.sidecar_path.write_text(
                json.dumps(
                    {
                        "csv": self.csv_path.name,
                        "scenario_hash": self.spec.scenario_hash,
                        "scenario": self.spec.to_dict(),
                        "channels": manifest(),
                        "beamng": self.beamng,
                        "git_commit": _git_commit(),
                        "summary": self.summary(),
                    },
                    indent=2, sort_keys=True,
                )
            )
        except OSError as error:
            print(f"warning: could not write {self.sidecar_path}: {error}")
