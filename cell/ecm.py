"""The equivalent circuit: what a voltmeter at the terminals would read.

Voltage is one of only two channels the real measured dataset has, so this is
half of the shared representation between simulation and measurement.

While the alternator carries the load the regulator holds the bus up, and the
reading says more about the alternator than about the battery. It is when the
battery is carrying the load -- cranking above all -- that terminal voltage
reveals its condition, which is exactly why a failing battery is discovered on
a cold morning rather than during a drive.

The resistance feedback is the important part. As health falls, resistance
rises; cranking then sags harder and dissipates more heat, which ages the
battery faster still. Without that loop the fade curve is a straight line and
says nothing.
"""

from __future__ import annotations

from load.electrical import REGULATED_VOLTAGE_V

#: Resting voltage spans 11.9 V empty to 12.7 V full, roughly linear over the
#: usable range for a flooded 12 V battery. This is still an assumed textbook
#: relationship, not a fitted one.
#:
#: A fit was attempted against 955 at-rest (|I| < 0.5 A) measurements from a
#: 1200 Ah OPzS cell (Rocha-Henriquez et al., Zenodo record 17252822, CC BY
#: 4.0) and rejected: that dataset's own `%SOC` column never exceeds 67%
#: across any of its Cell 1 files, and its OCV at ">60% SoC" (2.1263 V/cell)
#: already equals the textbook value for a FULLY charged flooded lead-acid
#: cell (2.12 V/cell). Their 65% reads as 100% on the textbook scale, so their
#: `%SOC` is evidently normalised to something else -- most likely Ah drawn
#: within a specific test, or a C20 reference -- not the absolute state of
#: charge this function takes. Fitting a line over their 25-65% and applying
#: it across our 0-100% extrapolates beyond the data at both ends and was
#: producing a curve 0.3-0.6 V too high. Validating this curve needs a dataset
#: whose state of charge is on the same absolute basis as ours.
OCV_EMPTY_V = 11.9
OCV_SPAN_V = 0.8

#: Resistance multiplier at end of life. A worn battery is a high-resistance
#: battery; this is what makes the crank sag diagnostic.
R_SOH_GAIN = 2.0

#: Resistance roughly doubles between 25 C and -18 C.
R_TEMP_COEFF = 0.016

#: And rises as the battery discharges and the electrolyte weakens.
R_SOC_GAIN = 0.5

#: Small bus sag under heavy charging current.
CHARGE_SAG_OHM = 0.002


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def open_circuit_v(soc: float) -> float:
    """Resting voltage. Clamped rather than extrapolated outside [0, 1]."""
    return OCV_EMPTY_V + OCV_SPAN_V * _clamp(soc)


def r_int_ohm(r0_ohm: float, soh: float, temp_c: float, soc: float) -> float:
    """Internal resistance: nominal, times age, times cold, times depletion.

    Each factor is 1.0 at the reference condition -- healthy, 25 C, full -- so
    `r0_ohm` stays the meaning it has on a datasheet.
    """
    f_soh = 1.0 + R_SOH_GAIN * (1.0 - _clamp(soh))
    f_temp = 1.0 + R_TEMP_COEFF * max(0.0, 25.0 - temp_c)
    f_soc = 1.0 + R_SOC_GAIN * (1.0 - _clamp(soc))
    return r0_ohm * f_soh * f_temp * f_soc


def terminal_v(
    soc: float,
    current_a: float,
    r_int_ohm: float,
    alternator_capability_a: float,
) -> float:
    """Terminal voltage. Positive `current_a` is discharge."""
    charging = current_a < 0.0 and alternator_capability_a > 0.0
    if charging:
        return REGULATED_VOLTAGE_V - CHARGE_SAG_OHM * abs(current_a)
    return open_circuit_v(soc) - current_a * r_int_ohm
