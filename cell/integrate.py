"""The 1 Hz within-trip loop, and the damage vector it hands to `cell/life.py`.

Health is held fixed for the length of a trip. Over the recorded 1,930 s it
moves by about one part in ten million, so integrating it here would spend
every cycle recomputing a constant -- and reaching end of life that way means
10^8 steps, which is not happening in stdlib Python. `cell/life.py` advances it
instead, in units of trips.

The battery's own temperature barely leaves the bay temperature: at 20 A
through 6 mOhm self-heating is 2.4 W, and even a 350 A crank is 735 W for 1.5 s
into a ~15 kJ/K mass, which is 0.07 K. The `I^2 R` term is kept because it
costs nothing and becomes visible as resistance rises with age, but
`h (T - T_bay)` is effectively the whole thermal model. Get the bay wrong and
nothing else here matters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from battery.corrosion import corrosion_rate
from cell.aging import FULL_SOC, LOW_SOC, AgingRates, AgingState, Damage, accumulate, soh
from cell.ecm import r_int_ohm, terminal_v
from load.electrical import alternator_capability_a, battery_current_a
from load.spec import BatteryScenario
from load.thermal import COOLDOWN_TAU_S, BayTemperature

#: Magnitude of standing gravity, for the vibration-dose proxy below.
GRAVITY_MPS2 = 9.81

#: Coolant temperature a soak starts from when the caller does not say.
OPERATING_TEMP_C = 90.0


@dataclass(frozen=True)
class Step:
    """One second (or one soak tick) of recorded cell state."""

    t_s: float
    i_bat_a: float
    v_bat_v: float
    t_bat_c: float
    t_bay_c: float
    soc: float
    r_int_ohm: float


@dataclass
class CellState:
    """What carries across trips. Health lives in `aging`, not as a bare number.

    `bay_temp_c` is the bay's own last value, carried across the trip/soak
    boundary so the next `BayTemperature` does not get seeded from the
    battery's temperature (which lags the bay by hours -- see the module
    docstring). `None` means "no prior bay reading", i.e. a fresh `CellState`,
    in which case callers seed from ambient or `temp_c` as before.
    """

    soc: float = 1.0
    temp_c: float = 25.0
    aging: AgingState = field(default_factory=AgingState)
    bay_temp_c: float | None = None

    def copy(self) -> "CellState":
        """A snapshot independent of this object -- see `TripResult.state`."""
        return CellState(
            soc=self.soc, temp_c=self.temp_c, aging=self.aging.copy(),
            bay_temp_c=self.bay_temp_c,
        )


@dataclass(frozen=True)
class TripResult:
    steps: tuple[Step, ...]
    damage: Damage
    state: CellState
    cranked: bool = False


def _advance(
    state: CellState,
    scenario: BatteryScenario,
    t_s: float,
    rpm: float,
    engine_running: float,
    bay_c: float,
    cranking: bool,
    vibration: float,
    dt_s: float,
    health: float,
) -> tuple[Step, Damage]:
    """Advance state of charge and temperature by one `dt_s`, and report both."""
    capacity_ah = scenario.capacity_ah * health
    resistance = r_int_ohm(scenario.r0_ohm, health, state.temp_c, state.soc)
    current = battery_current_a(
        rpm=rpm, soc=state.soc, temp_c=state.temp_c, scenario=scenario,
        engine_running=engine_running, cranking=cranking,
    )
    capability = (
        alternator_capability_a(rpm, scenario.alternator_rated_a)
        if engine_running >= 0.5 else 0.0
    )
    voltage = terminal_v(state.soc, current, resistance, capability)

    # Positive current is discharge, so state of charge falls with it.
    state.soc = max(0.0, min(1.0, state.soc - current * dt_s / 3600.0 / capacity_ah))

    # Self-heating in, bay coupling out. The second term dominates completely
    # -- see the module docstring for the wattage that makes that true.
    heating = current * current * resistance
    cooling = scenario.h_w_per_k * (state.temp_c - bay_c)
    state.temp_c += (heating - cooling) * dt_s / scenario.c_th_j_per_k

    hours = dt_s / 3600.0
    damage = Damage(
        duration_s=dt_s,
        corrosion_equivalent_h=corrosion_rate(state.temp_c) * hours,
        ah_throughput=abs(current) * hours,
        low_soc_hours=hours if state.soc < LOW_SOC else 0.0,
        full_charge_hours=hours if state.soc > FULL_SOC else 0.0,
        vibration_dose=vibration * dt_s,
    )
    step = Step(
        t_s=t_s, i_bat_a=current, v_bat_v=voltage, t_bat_c=state.temp_c,
        t_bay_c=bay_c, soc=state.soc, r_int_ohm=resistance,
    )
    return step, damage


def run_trip(
    samples: Iterable,
    scenario: BatteryScenario,
    state: CellState,
    rates: AgingRates,
    cranked: bool = False,
    dt_s: float = 1.0,
    health: float | None = None,
) -> TripResult:
    """Integrate one trip. `samples` should already be downsampled to `dt_s`.

    Health is read once, at the start, and held fixed for every step -- see
    the module docstring for why. `cell/life.py` is what moves it, between
    trips. By default health is derived from `state.aging` via `soh()`; pass
    `health` explicitly to override that (`cell/life.py`'s resim-on-drift loop
    needs this so a probed trip actually reflects the health it was asked
    about, rather than silently re-deriving it from the caller's own aging
    state, which is what made the two-rate feedback inert).
    """
    bay = BayTemperature(
        ambient_c=scenario.ambient_c,
        initial_c=state.bay_temp_c if state.bay_temp_c is not None else state.temp_c,
    )
    health = soh(state.aging, rates) if health is None else health
    steps: list[Step] = []
    total = Damage.zero()
    elapsed = 0.0

    for sample in samples:
        bay_c = bay.step(sample, dt_s)
        # Vibration proxy: how far total acceleration departs from gravity.
        magnitude = (
            float(sample["ax_mps2"]) ** 2
            + float(sample["ay_mps2"]) ** 2
            + float(sample["az_mps2"]) ** 2
        ) ** 0.5
        step, damage = _advance(
            state=state, scenario=scenario,
            t_s=float(sample["t_s"]), rpm=float(sample["rpm"]),
            engine_running=float(sample["engine_running"]), bay_c=bay_c,
            cranking=cranked and elapsed < scenario.crank_s,
            vibration=abs(magnitude - GRAVITY_MPS2), dt_s=dt_s, health=health,
        )
        steps.append(step)
        total = total + damage
        elapsed += dt_s

    state.aging = accumulate(state.aging, total, rates)
    state.bay_temp_c = bay.temperature_c
    return TripResult(
        steps=tuple(steps), damage=total, state=state.copy(), cranked=cranked
    )


def run_soak(
    seconds: float,
    scenario: BatteryScenario,
    state: CellState,
    rates: AgingRates,
    initial_coolant_c: float | None = None,
    dt_s: float = 10.0,
    health: float | None = None,
) -> TripResult:
    """Integrate a park: engine off, parasitic draw, and a bay that stays hot.

    The soak is not idle time for the battery. Immediately after shutdown the
    bay is hotter than it was while driving, and corrosion is exponential in
    temperature, so a good part of a day's damage happens in a car park.

    `dt_s` defaults well below `BAY_TAU_S` (60 s) so the bay's lag is actually
    integrated rather than snapping straight to target every tick -- at
    `dt_s == BAY_TAU_S`, `alpha = min(1.0, dt_s / BAY_TAU_S)` is 1.0 and the
    multi-minute heat-soak climb (see `load/thermal.py`) never happens. Pass
    `health` to override the internally-derived value, as `run_trip` does --
    see its docstring.
    """
    health = soh(state.aging, rates) if health is None else health
    bay = BayTemperature(
        ambient_c=scenario.ambient_c,
        initial_c=state.bay_temp_c if state.bay_temp_c is not None else state.temp_c,
    )
    steps: list[Step] = []
    total = Damage.zero()
    elapsed = 0.0

    # Coolant decays on its own time constant, which is nothing like the
    # battery's. Nothing measures it during a soak, so it is modelled here and
    # labelled as modelled -- never stood in for by the battery's own
    # temperature, which would keep the bay hot for hours that never happened.
    coolant = (
        OPERATING_TEMP_C if initial_coolant_c is None else initial_coolant_c
    )

    while elapsed < seconds:
        step_s = min(dt_s, seconds - elapsed)
        coolant += (scenario.ambient_c - coolant) * min(1.0, step_s / COOLDOWN_TAU_S)
        bay_c = bay.step(
            {"coolant_c": coolant, "speed_mps": 0.0, "engine_load": 0.0,
             "engine_running": 0.0},
            step_s,
        )
        step, damage = _advance(
            state=state, scenario=scenario, t_s=elapsed, rpm=0.0,
            engine_running=0.0, bay_c=bay_c, cranking=False, vibration=0.0,
            dt_s=step_s, health=health,
        )
        steps.append(step)
        total = total + damage
        elapsed += step_s

    state.aging = accumulate(state.aging, total, rates)
    state.bay_temp_c = bay.temperature_c
    return TripResult(steps=tuple(steps), damage=total, state=state.copy())
