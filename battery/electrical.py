"""Turning driving into battery current and voltage.

This is the interface the whole downstream argument rests on. The real dataset's
usable columns are current and voltage; the simulator produces neither. Until
driving is converted into current, the simulated and measured data do not share
a representation at all -- and the masked real-data latents, the one thing
breaking the circularity in the research argument, have nothing to fuse with.

It also supplies two ageing pathways that were producing no data. The
recharge deficit *is* net discharge at idle, when the alternator is turning too
slowly to cover the load and the battery makes up the difference. Sulfation
follows from cranking and the state-of-charge history after it.

An SLI battery is held near full charge by the alternator whenever the engine
runs, so the interesting quantity is not depth of discharge but *when the
balance goes the wrong way*: idling, hot, with the blower and lights on.

The coefficients here are typical values for a mid-size petrol car, not
measurements of any particular one. Treat comparisons between runs as usable and
absolute amps as indicative -- the same caveat as the thermal model.
"""

from __future__ import annotations

import math

#: Cranking draws hundreds of amps for a second or two. This is the single
#: largest current an SLI battery ever sees, and it is what sulfation tracks.
CRANK_CURRENT_A = 350.0

#: Always-on load: engine management, ignition, fuel pump, instruments.
BASE_LOAD_A = 22.0

#: Headlights, side lights, tail lights together.
LIGHTS_LOAD_A = 12.0

#: Blower and condenser fans at full. The compressor itself is belt-driven and
#: costs fuel rather than amps, but its fans do not.
HVAC_LOAD_A = 28.0

#: Alternator speed relative to the crank, through the pulley ratio.
PULLEY_RATIO = 2.6

#: Alternator shaft speed at which it reaches full output.
FULL_OUTPUT_SHAFT_RPM = 5000.0

#: Below this it produces nothing at all.
CUT_IN_SHAFT_RPM = 1200.0

#: What the regulator holds the bus at while charging.
REGULATED_VOLTAGE_V = 14.2

#: Open-circuit voltage of a healthy, fully charged 12 V lead-acid battery.
RESTING_VOLTAGE_V = 12.7

#: Internal resistance of a healthy battery, ohms. Rises as it ages, which is
#: what makes a worn battery sag when cranked.
NOMINAL_RESISTANCE_OHM = 0.006

#: Resistance roughly doubles as the battery goes from 25 C to -18 C, which is
#: why cold mornings are what finally kill a marginal battery.
RESISTANCE_TEMPERATURE_COEFFICIENT = 0.016


def alternator_output_a(rpm: float, rated_a: float) -> float:
    """What the alternator can supply at this engine speed.

    Output rises with shaft speed and saturates at the rating. At idle a typical
    alternator gives roughly a third to a half of its rating, which is why a
    stationary car with the blower and lights on can run a deficit.
    """
    shaft_rpm = rpm * PULLEY_RATIO
    if shaft_rpm <= CUT_IN_SHAFT_RPM:
        return 0.0
    fraction = (shaft_rpm - CUT_IN_SHAFT_RPM) / (
        FULL_OUTPUT_SHAFT_RPM - CUT_IN_SHAFT_RPM
    )
    return rated_a * min(1.0, fraction)


class ElectricalModel:
    def __init__(
        self,
        alternator_rated_a: float = 120.0,
        base_load_a: float = BASE_LOAD_A,
        internal_resistance_ohm: float = NOMINAL_RESISTANCE_OHM,
        temperature_c: float = 25.0,
        state_of_charge: float = 1.0,
    ) -> None:
        self.alternator_rated_a = alternator_rated_a
        self.base_load_a = base_load_a
        self.internal_resistance_ohm = internal_resistance_ohm
        self.temperature_c = temperature_c
        self.state_of_charge = state_of_charge

    # -- load --------------------------------------------------------------

    def accessory_load_a(self, hvac: float, lights: bool) -> float:
        return (
            self.base_load_a
            + HVAC_LOAD_A * max(0.0, min(1.0, hvac))
            + (LIGHTS_LOAD_A if lights else 0.0)
        )

    def battery_current_a(
        self,
        rpm: float,
        hvac: float = 0.0,
        lights: bool = False,
        cranking: bool = False,
    ) -> float:
        """Net current at the battery. Positive is discharge.

        Whatever the accessories want and the alternator cannot supply comes
        out of the battery; the surplus goes back in.
        """
        if cranking:
            return CRANK_CURRENT_A
        load = self.accessory_load_a(hvac, lights)
        return load - alternator_output_a(rpm, self.alternator_rated_a)

    # -- voltage -----------------------------------------------------------

    @property
    def resistance_ohm(self) -> float:
        """Internal resistance at this temperature.

        Cold batteries have markedly higher resistance, which is why a marginal
        battery survives all summer and fails on the first frost.
        """
        cold = max(0.0, 25.0 - self.temperature_c)
        return self.internal_resistance_ohm * (
            1.0 + RESISTANCE_TEMPERATURE_COEFFICIENT * cold
        )

    def open_circuit_voltage_v(self) -> float:
        """Roughly linear in state of charge over the usable range."""
        return 11.9 + 0.8 * max(0.0, min(1.0, self.state_of_charge))

    def terminal_voltage_v(self, current_a: float, rpm: float) -> float:
        """What a voltmeter at the terminals would read.

        While the alternator is supplying the load the regulator holds the bus
        up, so the reading says more about the alternator than the battery. It
        is when the battery is carrying the load -- cranking above all -- that
        the voltage reveals its condition.
        """
        charging = current_a < 0.0 and alternator_output_a(
            rpm, self.alternator_rated_a
        ) > 0.0
        if charging:
            # A small sag under heavy charging current.
            return REGULATED_VOLTAGE_V - 0.002 * abs(current_a)
        return self.open_circuit_voltage_v() - current_a * self.resistance_ohm

    # -- the two channels the real data has --------------------------------

    def from_driving(
        self,
        rpm: float,
        engine_on: bool,
        hvac: float = 0.0,
        lights: bool = False,
        cranking: bool = False,
    ) -> tuple[float, float]:
        """`(current_a, voltage_v)` for one moment of driving.

        These are the two channels the measured dataset has, so this is what
        puts simulated and real data in the same space.
        """
        effective_rpm = rpm if engine_on else 0.0
        current = self.battery_current_a(effective_rpm, hvac, lights, cranking)
        return current, self.terminal_voltage_v(current, effective_rpm)
