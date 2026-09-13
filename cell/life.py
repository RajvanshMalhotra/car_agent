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

**State of charge is carried too, but not extrapolated like damage is.**
`day_damage` reports the state of charge one day's driving-and-soak pattern
ends at, alongside the damage, and each subsequent probe starts from that
value rather than a fixed initial one -- see `project`'s docstring for why the
value is held rather than compounded as a per-day rate between resims. Without
carrying it forward at all, a probe rebuilt from a fixed starting SoC every
call cannot drift over a multi-year projection, and the sulfation pathway --
`low_soc_hours`, gated on SoC actually falling below `LOW_SOC` -- is
structurally unreachable no matter how long the horizon runs.

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


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class DayStart:
    """What a `day_damage` probe needs before it can run one day's pattern."""

    health: float
    soc: float


#: `day_damage` takes the state a day starts from and returns that day's
#: damage plus the state of charge it ended at -- not a delta, an absolute
#: value, so the caller need not know how the probe got there.
DayDamage = Callable[[DayStart], tuple[Damage, float]]


class UnfittedScale(RuntimeError):
    """An absolute figure was asked for from an unfitted rate constant."""


@dataclass(frozen=True)
class TripSchedule:
    trips_per_day: float = 2.0
    soak_s: float = 28800.0
    horizon_days: float = 3650.0

    def __post_init__(self) -> None:
        if self.trips_per_day <= 0.0:
            raise ValueError(f"trips_per_day must be positive, got {self.trips_per_day}")
        if self.horizon_days <= 0.0:
            raise ValueError(f"horizon_days must be positive, got {self.horizon_days}")


#: One sampled point of `LifeEstimate.state_curve`: everything a downstream
#: counterfactual experiment needs to know about the hidden state at a given
#: day, not just the scalar health `soh_curve` carries.
StatePoint = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class LifeEstimate:
    eol_days: float | None
    soh_curve: tuple[tuple[float, float], ...]
    scale_unfitted: bool
    interval_days: tuple[float, float] | None = None
    uncertainty_source: str = ""
    #: (day, soc, crystal, corrosion_hours, shedding, health) at the same
    #: cadence as `soh_curve`. `crystal` is the hidden, path-dependent state
    #: the counterfactual-identifiability experiment turns on -- see
    #: `cell/aging.py` -- so it is carried here rather than only at the end.
    state_curve: tuple[StatePoint, ...] = ()
    #: The `Damage` actually applied to `accumulate()`, averaged day-by-day
    #: over the whole projection (to end of life, or the horizon). NOT the
    #: single day_damage() evaluation at day 0 -- health degrades over a
    #: multi-year projection, and `capacity_ah = capacity_ah * health` in
    #: `cell/integrate.py` means a pathway invisible at health=1.0 (low
    #: corrosion, say, but sulfation-triggering low_soc_hours only once the
    #: battery has aged enough that its shrunken capacity no longer absorbs a
    #: crank) can still show up later in the life and be entirely missed by a
    #: day-0 snapshot. Averaging what was actually integrated reports what
    #: really happened over the life, not what the first day looked like.
    avg_damage_per_day: Damage = Damage.zero()

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
    soc0: float = 1.0,
) -> LifeEstimate:
    """Advance health, and state of charge, day by day to end of life or the horizon.

    State of charge is resimulated on the same schedule as damage -- at the
    start and whenever health has drifted past `resim_threshold` -- rather
    than every day, for the same reason: `day_damage` runs a full trip+soak
    integration and calling it 10^8 times is not on the table.

    Between resim points, SoC is held at what the last probe reported, NOT
    re-derived daily from a held per-day delta the way `damage` is. `damage`
    is a rate -- roughly the same amount accrues each day, so repeating it is
    a fair approximation. A day's net SoC change is not a rate: charge
    acceptance grows as the battery empties (`load/electrical.py`'s `(1 -
    SoC)` taper), so the true day-over-day map is strongly self-correcting.
    Treating one probed day's delta as if it applied every day until the next
    resim compounds a nonlinear step into a linear one and was verified to
    oscillate the projected SoC between 0 and 1 every resim interval, rather
    than settle near the equilibrium a repeated day actually reaches. Holding
    the last reported value instead still carries SoC forward -- consecutive
    probes see whatever the last one actually returned, never a value reset
    to the start -- so a genuinely charge-negative `day_damage` still drifts
    it down over the projection; it just does not extrapolate a single day's
    map across days it was never evaluated for.
    """
    state = AgingState()
    health = soh(state, rates)
    soc = _clamp01(soc0)
    damage, soc = day_damage(DayStart(health=health, soc=soc))
    soc = _clamp01(soc)
    curve: list[tuple[float, float]] = [(0.0, health)]
    state_curve: list[StatePoint] = [
        (0.0, soc, state.crystal, state.corrosion_hours, state.shedding, health)
    ]
    eol_days: float | None = None
    total_damage_days = Damage.zero()

    day = 0
    while day < schedule.horizon_days:
        state = accumulate(state, damage, rates)
        # `damage` is what was actually integrated into `state` on this
        # calendar day, whether freshly probed or held from the last resim --
        # summing it here (as opposed to the day-0 probe alone) is what makes
        # `avg_damage_per_day` reflect the whole life, not just its start.
        total_damage_days = total_damage_days + damage
        day += 1
        current = soh(state, rates)
        if day % CURVE_INTERVAL_DAYS == 0:
            curve.append((float(day), current))
            state_curve.append((
                float(day), soc, state.crystal, state.corrosion_hours,
                state.shedding, current,
            ))
        if current <= EOL_SOH:
            eol_days = float(day)
            curve.append((float(day), current))
            state_curve.append((
                float(day), soc, state.crystal, state.corrosion_hours,
                state.shedding, current,
            ))
            break
        if health - current >= resim_threshold:
            health = current
            damage, soc = day_damage(DayStart(health=health, soc=soc))
            soc = _clamp01(soc)

    avg_damage_per_day = (
        total_damage_days.scaled(1.0 / day) if day > 0 else damage
    )

    return LifeEstimate(
        eol_days=eol_days,
        soh_curve=tuple(curve),
        scale_unfitted=not rates.fitted,
        state_curve=tuple(state_curve),
        avg_damage_per_day=avg_damage_per_day,
    )


def ensemble(
    day_damage: DayDamage,
    rates: AgingRates,
    schedule: TripSchedule,
    draws: int = 32,
    seed: int = 0,
    spread: float = 0.35,
    soc0: float = 1.0,
) -> LifeEstimate:
    """Vary the unfitted constant over a plausible range and report quantiles.

    This is the interval the output contract requires. It is parameter
    uncertainty, and it must never be presented as the world model's stochastic
    latent -- a different quantity answering a different question.
    """
    rng = random.Random(seed)
    lives: list[float] = []
    point = project(day_damage, rates, schedule, soc0=soc0)

    for _ in range(draws):
        factor = 1.0 + rng.uniform(-spread, spread)
        drawn = dataclasses.replace(
            rates, corrosion_eol_h=rates.corrosion_eol_h * factor
        )
        result = project(day_damage, drawn, schedule, soc0=soc0)
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
        state_curve=point.state_curve,
        avg_damage_per_day=point.avg_damage_per_day,
    )
