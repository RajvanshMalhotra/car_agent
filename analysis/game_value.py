"""Does the game's measurement change the answer?

The honest question about this whole project. The simulator is expensive to run
and hard to keep working, and most of what it produces is fed into models with
invented coefficients. If the corrosion figure comes out the same whether the
coolant temperature is measured by BeamNG or predicted by our own engine model,
then the game is not earning its place on that channel, and driving patterns
could be sampled statistically for a fraction of the trouble.

One game run answers it. The same speed and throttle trace is replayed through
the engine model, and the two corrosion figures are compared. Route, idle,
ambient and behaviour are held identical because it is literally the same run --
the only thing that differs is where the coolant number came from.

This does not test everything the game supplies. The driving *pattern* itself --
how much idling actually happens on a real road network with traffic -- is not
something this can isolate, because there is no counterfactual pattern to
compare against.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from battery.corrosion import equivalent_hours
from sim.engine import (
    BAY_AIRFLOW_GAIN,
    BAY_COUPLING_STATIC,
    BAY_LOAD_GAIN,
    BAY_TAU_S,
    EngineModel,
)

#: Below this the two answers agree closely enough that the measurement is not
#: buying anything the model could not have guessed.
MATTERS_ABOVE = 0.15


@dataclass(frozen=True)
class Sample:
    t_s: float
    speed_mps: float
    rpm: float
    coolant_c: float
    throttle: float


@dataclass(frozen=True)
class Side:
    label: str
    equivalent_hours: float
    mean_coolant_c: float
    mean_bay_c: float
    max_bay_c: float


@dataclass(frozen=True)
class Comparison:
    measured: Side
    modelled: Side
    seconds: float

    @property
    def ratio(self) -> float:
        if self.modelled.equivalent_hours <= 0.0:
            return float("inf")
        return self.measured.equivalent_hours / self.modelled.equivalent_hours

    @property
    def game_matters(self) -> bool:
        """Whether the measurement changed the answer enough to be worth having."""
        return abs(self.ratio - 1.0) > MATTERS_ABOVE

    @property
    def coolant_gap_c(self) -> float:
        return self.measured.mean_coolant_c - self.modelled.mean_coolant_c


def read_run(csv_path: Path | str) -> list[Sample]:
    with open(csv_path) as handle:
        return [
            Sample(
                t_s=float(row["t_s"]),
                speed_mps=float(row["speed_mps"]),
                rpm=float(row.get("rpm") or 0.0),
                coolant_c=float(row["coolant_c"]),
                throttle=float(row.get("throttle") or 0.0),
            )
            for row in csv.DictReader(handle)
        ]


def _bay_trace(
    samples: Sequence[Sample], coolants: Sequence[float], ambient_c: float
) -> list[float]:
    """Bay temperature from a coolant trace, using the same model either way."""
    bay = ambient_c
    trace = []
    previous_t = samples[0].t_s
    for sample, coolant in zip(samples, coolants):
        dt = max(1e-3, sample.t_s - previous_t)
        previous_t = sample.t_s
        coupling = (
            BAY_COUPLING_STATIC
            * (1.0 + BAY_LOAD_GAIN * min(1.0, sample.throttle))
            / (1.0 + BAY_AIRFLOW_GAIN * max(0.0, sample.speed_mps))
        )
        target = ambient_c + coupling * (coolant - ambient_c)
        bay += (target - bay) * (dt / BAY_TAU_S)
        trace.append(bay)
    return trace


def _side(label, samples, coolants, ambient_c, dt_s) -> Side:
    bay = _bay_trace(samples, coolants, ambient_c)
    return Side(
        label=label,
        equivalent_hours=equivalent_hours(bay, dt_s=dt_s),
        mean_coolant_c=sum(coolants) / len(coolants),
        mean_bay_c=sum(bay) / len(bay),
        max_bay_c=max(bay),
    )


def compare_thermal_sources(
    samples: Sequence[Sample], ambient_c: float, cold_start: bool = True
) -> Comparison:
    """Corrosion from the measured coolant, against corrosion from a modelled one."""
    if not samples:
        raise ValueError("no rows in the run")

    dt_s = (
        (samples[-1].t_s - samples[0].t_s) / max(1, len(samples) - 1)
        if len(samples) > 1
        else 1.0
    )

    # The model never sees the measurement: it is driven only by the speed and
    # throttle the car actually used, which is the whole point of the test.
    engine = EngineModel(ambient_temp_c=ambient_c, cold_start=cold_start)
    engine.start()
    modelled_coolants = []
    for sample in samples:
        engine.step(speed_mps=sample.speed_mps, throttle=sample.throttle, dt=dt_s)
        modelled_coolants.append(engine.state.coolant_temp_c)

    return Comparison(
        measured=_side("measured by the game",
                       samples, [s.coolant_c for s in samples], ambient_c, dt_s),
        modelled=_side("predicted by our engine model",
                       samples, modelled_coolants, ambient_c, dt_s),
        seconds=samples[-1].t_s - samples[0].t_s,
    )
