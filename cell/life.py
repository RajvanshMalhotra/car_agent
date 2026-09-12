"""Advancing health over a trip schedule, and the gate on absolute figures.

A battery life is about 10^8 seconds. Integrating the cell at 1 Hz to reach it
is not possible here, and would be pointless anyway: over one trip health moves
by about one part in ten million. So `cell/integrate.py` produces damage in
trip-sized units and this module advances the ageing states in days.

The per-day damage is recomputed whenever health has drifted, because it is not
constant: resistance rises as the battery ages, which changes both the current
it draws and the heat it makes, which changes the damage. Recomputing once per
drift threshold costs a few hundred evaluations over a full life instead of
10^8.

**The scale gate.** Physics gives the fade curve its shape; the rate constant
that turns shape into days has to come from cells that actually reached end of
life, and there are none. `headline_days` therefore raises unless the rates are
flagged fitted. Ratios are always available -- the unfitted constant divides
out -- and they are usually the more useful claim anyway.

The ensemble interval represents **parameter uncertainty**: how much the answer
moves when the unfitted constant is varied over a plausible range. It is not
the stochastic-latent distribution the world model will eventually supply, and
`uncertainty_source` says so in the sidecar so the two are never conflated.
"""

from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass
from typing import Callable

from cell.aging import EOL_SOH, AgingRates, AgingState, Damage, accumulate, soh

#: How often a point is recorded on the health curve.
CURVE_INTERVAL_DAYS = 30

DayDamage = Callable[[float], Damage]


class UnfittedScale(RuntimeError):
    """An absolute figure was asked for from an unfitted rate constant."""


@dataclass(frozen=True)
class TripSchedule:
    trips_per_day: float = 2.0
    soak_s: float = 28800.0
    # 20 years: a simulation ceiling, not a physical claim. It has to clear
    # scenarios far lighter than "ordinary use" -- e.g. a single short recorded
    # trip repeated under this same default schedule, which under-fills a day
    # (a couple of minutes of driving plus one soak, not the ~24 h the ageing
    # constant was anchored against) and so ages many times slower than
    # ORDINARY_DAY_H. 10 years was too tight for that case: a real but light
    # scenario would silently read as "never reaches end of life" instead of
    # producing the (long, honestly-caveated) number it should.
    horizon_days: float = 7300.0

    def __post_init__(self) -> None:
        if self.trips_per_day <= 0.0:
            raise ValueError(f"trips_per_day must be positive, got {self.trips_per_day}")
        if self.horizon_days <= 0.0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")


@dataclass(frozen=True)
class LifeEstimate:
    eol_days: float | None
    soh_curve: tuple[tuple[float, float], ...]
    scale_unfitted: bool
    interval_days: tuple[float, float] | None = None
    uncertainty_source: str = ""

    def headline_days(self) -> float:
        """The conservative figure a fleet operator would act on.

        Refuses to answer while the rate constant is unfitted, because the
        number would look like a measurement and is not one.
        """
        if self.scale_unfitted:
            raise UnfittedScale(
                "the ageing rate constant is not fitted against full-life data, "
                "so an absolute remaining life cannot be reported; use ratio_to "
                "for relative claims, which do not depend on it"
            )
        if self.eol_days is None:
            raise UnfittedScale("end of life was not reached within the horizon")
        if self.interval_days is not None:
            return self.interval_days[0]  # report the low quantile, not the mean
        return self.eol_days

    def ratio_to(self, other: "LifeEstimate") -> float:
        """How many times longer this battery lasts than `other`.

        Valid with an unfitted scale: the constant cancels.
        """
        if self.eol_days is None or other.eol_days is None:
            raise ValueError("both estimates must reach end of life to be compared")
        return self.eol_days / other.eol_days


def project(
    day_damage: DayDamage,
    rates: AgingRates,
    schedule: TripSchedule,
    resim_threshold: float = 0.01,
) -> LifeEstimate:
    """Advance health day by day until end of life or the horizon."""
    state = AgingState()
    health = soh(state, rates)
    damage = day_damage(health)
    curve: list[tuple[float, float]] = [(0.0, health)]
    eol_days: float | None = None

    day = 0
    while day < schedule.horizon_days:
        state = accumulate(state, damage, rates)
        day += 1
        current = soh(state, rates)
        if day % CURVE_INTERVAL_DAYS == 0:
            curve.append((float(day), current))
        if current <= EOL_SOH:
            eol_days = float(day)
            curve.append((float(day), current))
            break
        if health - current >= resim_threshold:
            health = current
            damage = day_damage(health)

    return LifeEstimate(
        eol_days=eol_days,
        soh_curve=tuple(curve),
        scale_unfitted=not rates.fitted,
    )


def ensemble(
    day_damage: DayDamage,
    rates: AgingRates,
    schedule: TripSchedule,
    draws: int = 32,
    seed: int = 0,
    spread: float = 0.35,
) -> LifeEstimate:
    """Vary the unfitted constant over a plausible range and report quantiles.

    This is the interval the output contract requires. It is parameter
    uncertainty, and it must never be presented as the world model's stochastic
    latent -- a different quantity answering a different question.
    """
    rng = random.Random(seed)
    lives: list[float] = []
    point = project(day_damage, rates, schedule)

    for _ in range(draws):
        factor = 1.0 + rng.uniform(-spread, spread)
        drawn = dataclasses.replace(
            rates, corrosion_eol_h=rates.corrosion_eol_h * factor
        )
        result = project(day_damage, drawn, schedule)
        if result.eol_days is not None:
            lives.append(result.eol_days)

    interval = None
    if len(lives) >= 2:
        lives.sort()
        low = lives[max(0, int(0.05 * (len(lives) - 1)))]
        high = lives[min(len(lives) - 1, int(0.95 * (len(lives) - 1) + 0.999))]
        interval = (low, high)

    return LifeEstimate(
        eol_days=point.eol_days,
        soh_curve=point.soh_curve,
        scale_unfitted=not rates.fitted,
        interval_days=interval,
        uncertainty_source=(
            "parameter uncertainty over the unfitted corrosion rate constant; "
            "not the stochastic distribution a world model would supply"
        ),
    )
