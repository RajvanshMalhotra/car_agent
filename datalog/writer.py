"""Run logging: a CSV of channels plus a JSON sidecar of provenance.

Two requirements shape this.

**A log must be interpretable on its own.** A temperature trace with no ambient
temperature recorded beside it is close to useless, and a row of numbers with no
`spec_hash` cannot be traced back to the behaviour that produced it. So the
sidecar carries the whole `BehaviourSpec`, the scenario, and the seed --
`(spec_hash, scenario, seed)` is the reproducibility key for the whole project.

**A run must survive a crash.** Runs are ~10 minutes of real time on someone
else's laptop. Rows are flushed as they are written and the sidecar is written
even when the run raises, so a failure at minute 40 costs the last row, not the
whole run.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from battery.corrosion import REFERENCE_TEMP_C, equivalent_hours
from behaviour.spec import BehaviourSpec
from sim.backend import ControlInput, VehicleState

#: Below this the vehicle counts as stationary.
IDLE_SPEED_MPS = 0.5

COLUMNS = [
    "t_s",
    "trip",
    # pose
    "x_m",
    "y_m",
    "heading_rad",
    "speed_mps",
    # engine and thermal -- what the battery layer consumes
    "rpm",
    "coolant_c",
    "underbonnet_c",
    "oil_c",
    "gear",
    "fuel",
    "engine_on",
    "crank_count",
    "idling",
    "damage",
    # actual pedals, then what the controller asked for
    "throttle",
    "brake",
    "throttle_cmd",
    "brake_cmd",
    "steering_cmd",
]


class RunLog:
    def __init__(
        self,
        csv_path: Path | str,
        spec: BehaviourSpec,
        scenario: str,
        seed: int,
        log_hz: float,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.sidecar_path = self.csv_path.with_suffix(".json")
        self.spec = spec
        self.scenario = scenario
        self.seed = seed
        self.log_hz = log_hz

        self.trip = 0
        self.rows = 0
        self.idle_rows = 0
        self.bay_temps: list[float] = []
        self.max_damage = 0.0

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.csv_path.open("w", newline="")
        self._writer = csv.writer(self._handle)
        self._writer.writerow(COLUMNS)
        self._handle.flush()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc_info) -> None:
        # Runs are long and expensive. Write the sidecar whatever happened.
        self.close()

    def end_trip(self) -> None:
        """Mark a key-off. Trip segmentation is what sulfation is defined over."""
        self.trip += 1

    def record(self, state: VehicleState, control: ControlInput | None) -> None:
        """Record one row.

        `control` is None when the game's own AI is driving: there is no
        commanded value from us to record, and inventing zeros would read as
        "we asked for no throttle" rather than "we did not ask".
        """
        idling = state.engine_on and state.speed_mps < IDLE_SPEED_MPS
        self._writer.writerow(
            [
                round(state.sim_time_s, 3),
                self.trip,
                round(state.x_m, 3),
                round(state.y_m, 3),
                round(state.heading_rad, 4),
                round(state.speed_mps, 3),
                round(state.rpm, 1),
                round(state.coolant_temp_c, 2),
                round(state.underbonnet_temp_c, 2),
                round(state.oil_temp_c, 2),
                state.gear,
                round(state.fuel_fraction, 4),
                int(state.engine_on),
                state.crank_count,
                int(idling),
                round(state.damage, 2),
                round(state.throttle, 4),
                round(state.brake, 4),
                "" if control is None else round(control.throttle, 4),
                "" if control is None else round(control.brake, 4),
                "" if control is None else round(control.steering, 4),
            ]
        )
        self._handle.flush()

        self.rows += 1
        self.idle_rows += int(idling)
        self.bay_temps.append(state.underbonnet_temp_c)
        self.max_damage = max(self.max_damage, state.damage)

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "duration_s": self.rows / self.log_hz if self.log_hz else 0.0,
            "trips": self.trip + 1,
            "idle_fraction": self.idle_rows / self.rows if self.rows else 0.0,
            "mean_underbonnet_c": (
                sum(self.bay_temps) / len(self.bay_temps) if self.bay_temps else None
            ),
            "max_underbonnet_c": max(self.bay_temps) if self.bay_temps else None,
            "equivalent_hours": equivalent_hours(
                self.bay_temps, dt_s=1.0 / self.log_hz if self.log_hz else 1.0
            ),
            "equivalent_hours_reference_c": REFERENCE_TEMP_C,
            "max_damage": self.max_damage,
        }

    def close(self) -> None:
        if self._handle.closed:
            return
        self._handle.close()
        # This class exists so a run survives a crash. It must not be the thing
        # that crashes: the CSV is already on disk, so a sidecar that cannot be
        # written is reported and swallowed rather than taking the run with it.
        try:
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_sidecar()
        except OSError as error:
            print(f"warning: could not write {self.sidecar_path}: {error}")

    def _write_sidecar(self) -> None:
        self.sidecar_path.write_text(
            json.dumps(
                {
                    "csv": self.csv_path.name,
                    "spec_hash": self.spec.spec_hash,
                    "scenario": self.scenario,
                    "seed": self.seed,
                    "log_hz": self.log_hz,
                    # Pulled out of the spec because a reader needs them to
                    # interpret the temperature trace at all.
                    "ambient_temp_c": self.spec.ambient_temp_c,
                    "hvac_setting": self.spec.hvac_setting,
                    "cold_start": self.spec.cold_start,
                    "start_stop_enabled": self.spec.start_stop_enabled,
                    "spec": self.spec.to_dict(),
                    "summary": self.summary(),
                },
                indent=2,
                sort_keys=True,
            )
        )
