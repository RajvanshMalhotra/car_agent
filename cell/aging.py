"""Ageing: corrosion, sulfation, shedding, and the health they add up to.

Follows Schiffer et al. (2007), the weighted-Ah-throughput model for lead-acid,
rather than the Wang power law, which is lithium and does not apply to this
chemistry at all.

**The rate constants are not fitted.** No battery in this project has reached
end of life and the open full-life lead-acid dataset has not been obtained, so
there is nothing to fit against. They are set so ordinary usage lands inside
the literature band -- SLI batteries last 3-5 years, 2-3 in hot climates -- and
that is a sanity anchor, not a calibration. `AgingRates.fitted` is False by
default and a later module refuses to report an absolute remaining life while
that flag is false.

Ratios survive this uncertainty. "Profile A ages the battery twice as fast as
profile B" divides the unfitted constant out; "this battery has 847 days left"
does not.

**Sulfation is a state, not a total.** Crystals grow more slowly as they get
larger and dissolve in proportion to how much is present, so both terms depend
on the current crystal size. An accumulated sum of low-charge hours cannot
express that, which is why this module carries state across trips.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: End of life. The usual definition: 80% of nominal capacity.
EOL_SOH = 0.8

#: Below this state of charge, sulfation accumulates.
LOW_SOC = 0.9

#: And above this, a full charge dissolves it again.
FULL_SOC = 0.98

#: Capacity loss at end of life, i.e. 1.0 - EOL_SOH.
EOL_LOSS = 1.0 - EOL_SOH


@dataclass(frozen=True)
class Damage:
    """What one trip, or one soak, did to the battery."""

    duration_s: float = 0.0
    corrosion_equivalent_h: float = 0.0  # Arrhenius-weighted hours at 25 C
    ah_throughput: float = 0.0
    low_soc_hours: float = 0.0
    full_charge_hours: float = 0.0
    vibration_dose: float = 0.0

    @classmethod
    def zero(cls) -> "Damage":
        return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def __add__(self, other: "Damage") -> "Damage":
        return Damage(
            duration_s=self.duration_s + other.duration_s,
            corrosion_equivalent_h=(
                self.corrosion_equivalent_h + other.corrosion_equivalent_h
            ),
            ah_throughput=self.ah_throughput + other.ah_throughput,
            low_soc_hours=self.low_soc_hours + other.low_soc_hours,
            full_charge_hours=self.full_charge_hours + other.full_charge_hours,
            vibration_dose=self.vibration_dose + other.vibration_dose,
        )

    def scaled(self, factor: float) -> "Damage":
        return Damage(
            duration_s=self.duration_s * factor,
            corrosion_equivalent_h=self.corrosion_equivalent_h * factor,
            ah_throughput=self.ah_throughput * factor,
            low_soc_hours=self.low_soc_hours * factor,
            full_charge_hours=self.full_charge_hours * factor,
            vibration_dose=self.vibration_dose * factor,
        )


@dataclass(frozen=True)
class AgingRates:
    """The unfitted constants. `fitted` is the honesty gate, not a hint.

    `corrosion_eol_h` is the Arrhenius-weighted exposure at which corrosion
    **alone** reaches end of life, and it means exactly that: a corrosion
    fraction of 1.0 puts health at `EOL_SOH` with no help from the other two
    pathways. The sulfation and shedding weights then say how much those
    contribute relative to a full corrosion life -- they are not shares of a
    total that must sum to 1.0.

    The value was set by working backwards from ordinary use. A day of it is
    about 79 Arrhenius-weighted hours -- 1.5 h driving with the bay near 60 C
    is 20.8 of them, an hour of post-shutdown heat soak near 75 C is another
    36.4, and the 21.5 h parked at 25 C are 21.5 more. Note that the heat soak
    contributes more than the driving does. At 79 a day, 105,000 puts end of
    life at 3.6 years, mid-band.

    **A recorded discrepancy, not tuned away.** Because life is inversely
    proportional to weighted exposure, this model makes a 42 C ambient age the
    battery about 3.1 times faster than a 25 C one. That agrees with this
    project's own earlier measurement of ~2.9x, and it is stronger than the
    literature's population-level bands imply (3-5 years temperate against 2-3
    hot, roughly 1.7x). Those bands compare whole populations with many
    confounders and the ratio is not a like-for-like check, so the model is
    left alone and the disagreement is stated.
    """

    corrosion_eol_h: float = 105000.0
    corrosion_exponent: float = 0.6  # Schiffer's sublinear layer growth
    sulfation_weight: float = 0.30  # relative to a full corrosion life
    shedding_weight: float = 0.15
    sulfation_per_low_soc_h: float = 2.0e-4
    sulfation_recovery_per_full_h: float = 5.0e-5
    shedding_per_dose: float = 1.0e-7
    fitted: bool = False


@dataclass
class AgingState:
    corrosion_hours: float = 0.0
    crystal: float = 0.0
    shedding: float = 0.0

    def copy(self) -> "AgingState":
        return replace(self)


def accumulate(state: AgingState, damage: Damage, rates: AgingRates) -> AgingState:
    """Advance the ageing states by one trip's or one soak's damage."""
    corrosion_hours = state.corrosion_hours + damage.corrosion_equivalent_h

    # Growth slows as crystals enlarge; dissolution is proportional to what is
    # there. Both depend on the current size, which is why this is a state.
    growth = (
        rates.sulfation_per_low_soc_h * damage.low_soc_hours * (1.0 - state.crystal)
    )
    recovery = (
        rates.sulfation_recovery_per_full_h
        * damage.full_charge_hours
        * state.crystal
    )
    crystal = max(0.0, min(1.0, state.crystal + growth - recovery))

    shedding = min(
        1.0, state.shedding + rates.shedding_per_dose * damage.vibration_dose
    )
    return AgingState(
        corrosion_hours=corrosion_hours, crystal=crystal, shedding=shedding
    )


def soh(state: AgingState, rates: AgingRates) -> float:
    """State of health: 1.0 new, `EOL_SOH` at end of life."""
    if state.corrosion_hours > 0.0:
        exposure = state.corrosion_hours / rates.corrosion_eol_h
        corrosion_fraction = exposure ** rates.corrosion_exponent
    else:
        corrosion_fraction = 0.0
    loss = EOL_LOSS * (
        corrosion_fraction
        + rates.sulfation_weight * state.crystal
        + rates.shedding_weight * state.shedding
    )
    return max(0.0, 1.0 - loss)
