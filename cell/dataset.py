"""Assembling the dataset, in two tables, with the split declared.

**Two tables, not one.** Remaining life is a per-trip quantity. Broadcast onto
two thousand near-identical 1 Hz rows it becomes recoverable from the row
index, which is the label leakage of the original dataset wearing a different
hat. The 1 Hz table trains state-of-charge and voltage models; the per-trip
table trains the remaining-life model.

**The split is declared, not implied.** Within a trip,
`soc = soc_0 - cumsum(i_bat_a * dt) / Q` by construction, so offering current
as a feature beside state of charge as a target hands a network the identity.
It is closed by naming the driving channels as features and the battery
channels as targets, in the sidecar, where downstream code reads it. That is
also the actual research question: predict battery state from driving, not from
battery current.

The leakage report then runs over the declared split rather than over every
column pair, which is what makes it mean something.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from cell.aging import AgingRates, soh
from cell.leakage import report
from cell.life import LifeEstimate
from load.spec import BatteryScenario

#: Driving channels. What the world model rolls out and the networks see.
FEATURES: tuple[str, ...] = (
    "speed_mps", "rpm", "throttle", "coolant_c", "engine_load",
    "grade_rad", "a_long_mps2", "engine_running", "t_bay_c",
)

#: Battery channels. What the networks predict. Never a feature -- soc is a
#: cumulative sum of i_bat_a by construction, so offering both would hand a
#: network the identity that made the original dataset's label leak.
TARGETS: tuple[str, ...] = ("i_bat_a", "v_bat_v", "t_bat_c", "soc", "r_int_ohm")

#: The 1 Hz table: one row per within-trip step, trains SoC/voltage models.
WITHIN_TRIP_COLUMNS: tuple[str, ...] = ("trip", "t_s") + FEATURES + TARGETS

#: The per-trip table: one row per trip, trains the remaining-life model.
#: `rul_days` lives here and only here -- see the module docstring.
TRIP_DAMAGE_COLUMNS: tuple[str, ...] = (
    "trip", "cranked", "duration_s", "corrosion_equivalent_h", "ah_throughput",
    "low_soc_hours", "full_charge_hours", "vibration_dose",
    "soc_start", "soc_end", "soh", "rul_days",
)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _within_trip_rows(trips) -> list[dict]:
    rows: list[dict] = []
    for index, (samples, result) in enumerate(trips):
        for sample, step in zip(samples, result.steps):
            row = {"trip": index, "t_s": step.t_s}
            for column in FEATURES:
                row[column] = (
                    step.t_bay_c if column == "t_bay_c" else float(sample[column])
                )
            row.update({
                "i_bat_a": step.i_bat_a, "v_bat_v": step.v_bat_v,
                "t_bat_c": step.t_bat_c, "soc": step.soc,
                "r_int_ohm": step.r_int_ohm,
            })
            rows.append(row)
    return rows


def _trip_damage_rows(trips, rates: AgingRates, life: LifeEstimate) -> list[dict]:
    rows: list[dict] = []
    for index, (_samples, result) in enumerate(trips):
        damage = result.damage
        rows.append({
            "trip": index,
            "cranked": int(result.cranked),
            "duration_s": damage.duration_s,
            "corrosion_equivalent_h": damage.corrosion_equivalent_h,
            "ah_throughput": damage.ah_throughput,
            "low_soc_hours": damage.low_soc_hours,
            "full_charge_hours": damage.full_charge_hours,
            "vibration_dose": damage.vibration_dose,
            "soc_start": result.steps[0].soc if result.steps else "",
            "soc_end": result.steps[-1].soc if result.steps else "",
            "soh": soh(result.state.aging, rates),
            "rul_days": "" if life.eol_days is None else life.eol_days,
        })
    return rows


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[column] for column in columns])


def write_dataset(
    out_dir: Path | str,
    scenario: BatteryScenario,
    trips,
    life: LifeEstimate,
    provenance: dict[str, str],
    dropouts,
    rates: AgingRates,
) -> dict:
    """Write both tables and the sidecar. Returns the sidecar.

    `trips` is a sequence of `(samples, TripResult)` pairs: the raw driving
    samples fed to `cell.integrate.run_trip`, paired with what it returned, so
    the within-trip table can carry both the driving channels and the battery
    channels it produced from them.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    within = _within_trip_rows(trips)
    per_trip = _trip_damage_rows(trips, rates, life)
    _write_csv(out / "within_trip.csv", WITHIN_TRIP_COLUMNS, within)
    _write_csv(out / "trip_damage.csv", TRIP_DAMAGE_COLUMNS, per_trip)

    # No dt_column: battery channels are targets and never features, so there
    # is no (feature, dt) pair whose cumulative sum could equal a target. The
    # declared split is the defence here; the cumulative check has nothing to
    # examine.
    findings = report(within, targets=TARGETS, features=FEATURES)

    sidecar = {
        "scenario": scenario.to_dict(),
        "scenario_hash": scenario.scenario_hash,
        "git_commit": _git_commit(),
        "tables": {
            "within_trip": {
                "rows": len(within), "hz": 1.0,
                "columns": list(WITHIN_TRIP_COLUMNS),
            },
            "trip_damage": {
                "rows": len(per_trip), "columns": list(TRIP_DAMAGE_COLUMNS),
            },
        },
        "channels": dict(provenance),
        "dropouts": [asdict(span) for span in dropouts],
        "leakage": {
            "features": list(FEATURES),
            "targets": list(TARGETS),
            "findings": [asdict(finding) for finding in findings],
            "note": (
                "soc is cumsum(i_bat_a * dt) / Q by construction, so battery "
                "channels are targets and never features. Training on a target "
                "as a feature reproduces the original dataset's label leakage. "
                "Separately: t_bat_c chases t_bay_c with no other forcing of "
                "consequence, so predicting it from the t_bay_c feature is "
                "close to trivial and must not be reported as a result. The "
                "exact-relationship detector will not flag it, because the "
                "thermal lag makes it approximate rather than analytic."
            ),
        },
        "life": {
            "eol_days": life.eol_days,
            "interval_days": (
                list(life.interval_days) if life.interval_days else None
            ),
            "scale_unfitted": life.scale_unfitted,
            "uncertainty_source": life.uncertainty_source,
            "note": (
                "The ageing rate constant is not fitted against full-life data. "
                "Ratios between scenarios are usable; absolute days are not. "
                "It was also derived from engine-bay exposure but is evaluated "
                "at battery temperature, which lags and runs cooler, so on the "
                "recorded scenario measured exposure is about 0.32x what the "
                "constant assumed and absolute days read roughly 3x too long "
                "-- see AgingRates in cell/aging.py."
            ),
        },
        "ageing_rates": asdict(rates),
    }
    (out / "dataset.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    return sidecar
