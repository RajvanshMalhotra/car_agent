"""Turning driving into battery current.

This is where the simulated and the measured data finally share a
representation: the real dataset's only usable columns are current and voltage,
and the simulator produces neither. BeamNG has no 12 V system at all -- the
~200 keys of `electrics.values` contain nothing matching volt, batt, amp,
current or alternator -- so all of this is modelled, always.

**Positive is discharge.** Everywhere, without exception. The parent spec wrote
engine-off parasitic as negative in one section and assumed the opposite sign
in the next; this is the settled convention and `dSoC/dt = -I/Q` follows it.

**Charge acceptance is what makes two of the ageing pathways exist.** A lead
battery near full charge will not take current at any voltage the regulator can
offer, because the limit is kinetic rather than ohmic. Without the `(1 - SoC)`
taper the alternator refills the battery instantly after every crank, and the
recharge-deficit and sulfation pathways vanish from the model entirely.

Coefficients are typical for a mid-size petrol car, not measurements of one.
Comparisons between runs are usable; absolute amps are indicative.

This module deliberately holds no state -- `battery/electrical.py`, which this
supersedes for the offline battery layer, wraps the same coefficients in an
`ElectricalModel` class for the live sim loop (`sim/fake.py`,
`sim/mcp_backend.py`) and is left in place for that reason. The state this
module would otherwise carry (state of charge, temperature) lives in a later
module's `CellState` instead.
"""

from __future__ import annotations

from load.spec import BatteryScenario

#: Alternator shaft speed relative to the crank, through the pulley.
PULLEY_RATIO = 2.6

#: Below this shaft speed the alternator produces nothing.
CUT_IN_SHAFT_RPM = 1200.0

#: And at this one it reaches its rating.
FULL_OUTPUT_SHAFT_RPM = 5000.0

#: What the regulator holds the bus at while charging.
REGULATED_VOLTAGE_V = 14.2

#: Headlights, side lights and tail lights together.
LIGHTS_LOAD_A = 12.0

#: Blower and condenser fans at full. The compressor is belt-driven and costs
#: fuel rather than amps; its fans do not.
HVAC_LOAD_A = 28.0

#: Charge acceptance falls off below this and is unimpaired above it.
ACCEPTANCE_WARM_C = 25.0
ACCEPTANCE_COLD_C = -10.0
ACCEPTANCE_COLD_FACTOR = 0.2


def alternator_capability_a(rpm: float, rated_a: float) -> float:
    """The most the alternator could supply at this engine speed.

    At idle a typical alternator gives roughly a quarter to a half of its
    rating, which is why a stationary car with the blower and lights on runs a
    deficit however large the alternator is.
    """
    shaft_rpm = max(0.0, rpm) * PULLEY_RATIO
    if shaft_rpm <= CUT_IN_SHAFT_RPM:
        return 0.0
    fraction = (shaft_rpm - CUT_IN_SHAFT_RPM) / (
        FULL_OUTPUT_SHAFT_RPM - CUT_IN_SHAFT_RPM
    )
    return rated_a * min(1.0, fraction)


def accessory_load_a(scenario: BatteryScenario) -> float:
    """What the car's electrics are asking for. A scenario input, not a measurement."""
    return (
        scenario.accessory_base_a
        + HVAC_LOAD_A * min(1.0, max(0.0, scenario.hvac))
        + (LIGHTS_LOAD_A if scenario.lights else 0.0)
    )


def acceptance_temperature_factor(temp_c: float) -> float:
    """Cold batteries take charge badly. Linear between the two anchors."""
    if temp_c >= ACCEPTANCE_WARM_C:
        return 1.0
    if temp_c <= ACCEPTANCE_COLD_C:
        return ACCEPTANCE_COLD_FACTOR
    span = ACCEPTANCE_WARM_C - ACCEPTANCE_COLD_C
    return ACCEPTANCE_COLD_FACTOR + (1.0 - ACCEPTANCE_COLD_FACTOR) * (
        (temp_c - ACCEPTANCE_COLD_C) / span
    )


def charge_acceptance_a(
    soc: float, temp_c: float, scenario: BatteryScenario
) -> float:
    """The most current the battery will take, whatever is on offer.

    Goes to zero at full charge. This taper is the whole reason a short trip
    fails to recover the charge a crank took.
    """
    headroom = max(0.0, 1.0 - min(1.0, soc))
    return (
        scenario.c_accept_per_h
        * scenario.capacity_ah
        * headroom
        * acceptance_temperature_factor(temp_c)
    )


def battery_current_a(
    rpm: float,
    soc: float,
    temp_c: float,
    scenario: BatteryScenario,
    engine_running: float,
    cranking: bool = False,
) -> float:
    """Net current at the battery. Positive is discharge.

        engine off      +parasitic
        cranking        +crank_a
        engine running  accessories - what the alternator actually delivers
    """
    if cranking:
        return scenario.crank_a
    if engine_running < 0.5:
        return scenario.parasitic_a
    load = accessory_load_a(scenario)
    wanted = load + charge_acceptance_a(soc, temp_c, scenario)
    delivered = min(alternator_capability_a(rpm, scenario.alternator_rated_a), wanted)
    return load - delivered
