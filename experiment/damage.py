"""Battery damage from a day's observables, in torch so it can be optimised through.

`cell/aging.py` computes damage from a 1 Hz trace. The world model predicts
DAILY summaries, so this rebuilds the same two pathways from those summaries:

    corrosion   hours * corrosion_rate(t_bat)      Arrhenius, 25 C = 1.0
    sulfation   crystal grows with hours spent below LOW_SOC, and the growth
                slows as the crystal gets bigger (it is a state, not a total)

Measured against the recorded hidden state before this module was written:
corrosion rebuilt this way lands within 3-4% at 25 C and 42 C, and the implied
low-charge hours come out at 0.97-1.00x the true value, because once these
batteries drop below the threshold they stay there all day.

**Hard and soft versions.** `daily_damage` uses the real thresholds and is what
gets checked against `cell/aging.py`. `soft_daily_damage` replaces the
threshold with a sigmoid so a gradient can flow back to the actions -- the
planner needs that, and the two agree wherever SoC is not sitting exactly on
the threshold.

Nothing here converts damage into days of life. That needs a rate constant
fitted to batteries that actually died, which this project does not have, and
`cell/life.py` refuses to report one. Damage ratios do not need it: the
constant cancels.
"""

from __future__ import annotations

import torch

from battery.corrosion import ACTIVATION_ENERGY_J_PER_MOL, GAS_CONSTANT, KELVIN, REFERENCE_TEMP_C
from cell.aging import CORROSION_LOSS_EXPONENT, EOL_LOSS, FULL_SOC, LOW_SOC, AgingRates

#: Width of the sigmoid that replaces the SoC threshold in the soft version,
#: in SoC units. Small enough that a day clearly below the threshold counts
#: fully, wide enough that a gradient survives near it.
SOC_SOFTNESS = 0.02

#: How much of the day's corrosion rate comes from the PEAK battery
#: temperature rather than the mean. Heat damage is exponential in
#: temperature, so it does not average: a day that swings does more damage
#: than a flat day at the same mean, and the mean alone understates it.
#:
#: Fitted by least squares on half the round 12 batteries and checked on the
#: other half, which the fit never saw:
#:
#:     mean only        median -7.2%, worst 10% 21.6%   (biased low)
#:     blended w=0.118  median +0.3%, worst 10%  9.3%
#:
#: Left at the fitted value rather than rounded, and never refitted against a
#: planner's own output -- that would be fitting the objective to itself.
PEAK_TEMPERATURE_WEIGHT = 0.1182


def corrosion_rate(t_bat_c: torch.Tensor) -> torch.Tensor:
    """`battery/corrosion.py`'s rate, elementwise: 1.0 at 25 C, doubling per +10 C."""
    reference_k = REFERENCE_TEMP_C + KELVIN
    temperature_k = t_bat_c + KELVIN
    exponent = (ACTIVATION_ENERGY_J_PER_MOL / GAS_CONSTANT) * (
        1.0 / reference_k - 1.0 / temperature_k
    )
    return torch.exp(exponent)


def blended_rate(t_bat_mean: torch.Tensor, t_bat_max: torch.Tensor) -> torch.Tensor:
    """The day's effective corrosion rate, from its mean AND peak temperature."""
    w = PEAK_TEMPERATURE_WEIGHT
    return (1.0 - w) * corrosion_rate(t_bat_mean) + w * corrosion_rate(t_bat_max)


def daily_damage(
    t_bat_mean: torch.Tensor, soc: torch.Tensor, hours: torch.Tensor,
    t_bat_max: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One day's damage terms, with the real thresholds.

    `t_bat_max` is optional only so older callers keep working; without it the
    corrosion term falls back to the mean-only form, which is biased ~7% low.
    """
    rate = corrosion_rate(t_bat_mean) if t_bat_max is None else blended_rate(t_bat_mean, t_bat_max)
    return {
        "corrosion_equivalent_h": hours * rate,
        "low_soc_hours": hours * (soc < LOW_SOC).to(hours.dtype),
        "full_charge_hours": hours * (soc > FULL_SOC).to(hours.dtype),
    }


def soft_daily_damage(
    t_bat_mean: torch.Tensor, soc: torch.Tensor, hours: torch.Tensor,
    t_bat_max: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Same, with the thresholds smoothed so gradients reach the actions."""
    rate = corrosion_rate(t_bat_mean) if t_bat_max is None else blended_rate(t_bat_mean, t_bat_max)
    below = torch.sigmoid((LOW_SOC - soc) / SOC_SOFTNESS)
    above = torch.sigmoid((soc - FULL_SOC) / SOC_SOFTNESS)
    return {
        "corrosion_equivalent_h": hours * rate,
        "low_soc_hours": hours * below,
        "full_charge_hours": hours * above,
    }


def accumulate_crystal(
    crystal0: torch.Tensor, low_soc_hours: torch.Tensor, full_charge_hours: torch.Tensor,
    rates: AgingRates | None = None,
) -> torch.Tensor:
    """Walk `cell/aging.py:accumulate`'s sulfation state over a run of days.

    `low_soc_hours` and `full_charge_hours` are `[B, K]`. Growth slows as the
    crystal enlarges and dissolution is proportional to what is there, so this
    has to step day by day rather than sum.
    """
    rates = rates or AgingRates()
    crystal = crystal0
    for day in range(low_soc_hours.shape[1]):
        growth = rates.sulfation_per_low_soc_h * low_soc_hours[:, day] * (1.0 - crystal)
        recovery = rates.sulfation_recovery_per_full_h * full_charge_hours[:, day] * crystal
        crystal = torch.clamp(crystal + growth - recovery, 0.0, 1.0)
    return crystal


def health_loss(
    corrosion_hours: torch.Tensor, crystal: torch.Tensor,
    rates: AgingRates | None = None,
) -> torch.Tensor:
    """`cell/aging.py:soh`'s loss term -- how far health has fallen from 1.0.

    Shedding is left out: it is driven by vibration, which no dial here
    changes, so it would be a constant added to every candidate schedule.
    """
    rates = rates or AgingRates()
    exposure = torch.clamp(corrosion_hours, min=0.0) / rates.corrosion_eol_h
    layer_fraction = exposure ** rates.corrosion_exponent
    corrosion_fraction = layer_fraction ** CORROSION_LOSS_EXPONENT
    return EOL_LOSS * (corrosion_fraction + rates.sulfation_weight * crystal)


def hours_from_actions(
    is_layup: torch.Tensor, driving_minutes: torch.Tensor, soak_hours: torch.Tensor,
) -> torch.Tensor:
    """How many hours a day actually integrates, from the action row.

    A parked day is a full 24 h; a driving day is its trips plus their soaks.
    Getting this from the actions rather than assuming 24 h is what makes the
    rebuilt corrosion match the recorded one -- parked days accrue MORE
    corrosion-equivalent hours than driving days precisely because they are
    longer.
    """
    driving_day = driving_minutes / 60.0 + soak_hours
    return torch.where(is_layup >= 0.5, torch.full_like(driving_day, 24.0), driving_day)
