"""Grid corrosion, the dominant ageing path for an SLI battery in service.

An SLI battery is held near 100% SoC by the alternator and never deep-cycles,
so "hard acceleration drains the battery" is not the mechanism. What kills it is
heat: positive grid corrosion under the bonnet, following Arrhenius.

Calibrated against one anchor only -- the standard rule that +10 degC roughly
halves service life. That fixes the *shape* of the temperature dependence and
nothing else. Output is therefore relative: hours-equivalent at a reference
temperature. Converting that to a life in years needs a rate constant fitted to
cells that actually reached end of life, which we do not have.
"""

from __future__ import annotations

import math
from typing import Sequence

GAS_CONSTANT = 8.314462618  # J/(mol K)
KELVIN = 273.15

#: Temperature the rate is normalised to. 25 degC is the usual datasheet basis.
REFERENCE_TEMP_C = 25.0

#: Chosen so the rate doubles per +10 degC around typical bay temperatures,
#: which is the +10 degC halves life rule. Lands at ~62 kJ/mol, inside the
#: 50-70 kJ/mol band reported for lead-acid grid corrosion -- a useful check
#: that the rule of thumb and the literature agree.
_ANCHOR_LOW_K = 50.0 + KELVIN
_ANCHOR_HIGH_K = 60.0 + KELVIN
ACTIVATION_ENERGY_J_PER_MOL = (
    GAS_CONSTANT * math.log(2.0) / (1.0 / _ANCHOR_LOW_K - 1.0 / _ANCHOR_HIGH_K)
)


def corrosion_rate(temperature_c: float) -> float:
    """Corrosion rate relative to `REFERENCE_TEMP_C`, which is 1.0."""
    reference_k = REFERENCE_TEMP_C + KELVIN
    temperature_k = temperature_c + KELVIN
    exponent = (ACTIVATION_ENERGY_J_PER_MOL / GAS_CONSTANT) * (
        1.0 / reference_k - 1.0 / temperature_k
    )
    return math.exp(exponent)


def equivalent_hours(
    underbonnet_temps_c: Sequence[float], dt_s: float
) -> float:
    """Damage from a temperature trace, as hours at `REFERENCE_TEMP_C`.

    Two traces with the same equivalent hours have done the same amount of
    corrosion, whatever their temperature histories looked like.
    """
    return sum(corrosion_rate(t) for t in underbonnet_temps_c) * dt_s / 3600.0
