"""Engine and under-bonnet thermal model.

The battery layer downstream needs engine bay temperature, RPM, idle time and
crank events -- not traction. Of those, **under-bonnet temperature is not
measurable from the simulator**: BeamNG models coolant and oil temperature, and
OutGauge reports coolant temperature, but neither reports the temperature of the
air around the battery. So it is estimated here from coolant temperature,
airflow and ambient, and the same estimator runs unchanged on real telemetry.

The structure is defensible; the coefficients are not calibrated against a real
engine bay. Treat absolute temperatures as indicative and comparisons between
behaviours as the usable output.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

#: Thermostat setpoint. Coolant is held here once warm.
OPERATING_TEMP_C = 90.0

#: Warm-up time constant at moderate load, seconds. Heavy load warms faster.
#: Sized so a car reaches operating temperature in roughly 5-10 minutes.
WARMUP_TAU_S = 100.0

#: Cool-down time constant of a stopped engine, seconds.
COOLDOWN_TAU_S = 900.0

#: Fraction of the coolant-to-ambient rise the bay sees with no airflow.
BAY_COUPLING_STATIC = 0.62

#: How quickly airflow carries bay heat away, per m/s of road speed.
BAY_AIRFLOW_GAIN = 0.10

#: Extra bay heating under load, at full throttle.
BAY_LOAD_GAIN = 0.25

#: Bay thermal lag, seconds. Air in the bay responds faster than the coolant.
BAY_TAU_S = 60.0

#: RPM per m/s for each gear, and the RPM the gearbox shifts up at.
GEAR_RPM_PER_MPS = (220.0, 130.0, 92.0, 72.0, 58.0)
UPSHIFT_RPM = 2600.0


@dataclass
class EngineState:
    rpm: float = 0.0
    coolant_temp_c: float = 20.0
    underbonnet_temp_c: float = 20.0
    engine_on: bool = False
    crank_count: int = 0

    def snapshot(self) -> "EngineState":
        return dataclasses.replace(self)


class EngineModel:
    def __init__(
        self,
        ambient_temp_c: float,
        cold_start: bool = True,
        idle_rpm: float = 800.0,
        warm_start_temp_c: float = 80.0,
    ) -> None:
        self.ambient_temp_c = ambient_temp_c
        self.cold_start = cold_start
        self.idle_rpm = idle_rpm
        self.warm_start_temp_c = warm_start_temp_c
        self.state = EngineState(
            coolant_temp_c=ambient_temp_c,
            underbonnet_temp_c=ambient_temp_c,
        )

    def start(self) -> None:
        """Crank the engine. Each crank is a 200-600 A draw on the battery."""
        if not self.state.engine_on:
            self.state.crank_count += 1
            self.state.engine_on = True
            if not self.cold_start and self.state.crank_count == 1:
                self.state.coolant_temp_c = self.warm_start_temp_c

    def stop(self) -> None:
        self.state.engine_on = False
        self.state.rpm = 0.0

    def read_state(self) -> EngineState:
        return self.state.snapshot()

    def _rpm(self, speed_mps: float) -> float:
        if not self.state.engine_on:
            return 0.0
        # Highest gear that still keeps the engine above the upshift point.
        for rpm_per_mps in GEAR_RPM_PER_MPS[::-1]:
            rpm = speed_mps * rpm_per_mps
            if rpm >= UPSHIFT_RPM:
                return max(self.idle_rpm, rpm)
        for rpm_per_mps in GEAR_RPM_PER_MPS:
            rpm = speed_mps * rpm_per_mps
            if rpm <= UPSHIFT_RPM:
                return max(self.idle_rpm, rpm)
        return max(self.idle_rpm, speed_mps * GEAR_RPM_PER_MPS[-1])

    def step(
        self,
        speed_mps: float,
        throttle: float,
        dt: float,
        coolant_temp_c: float | None = None,
    ) -> EngineState:
        """Advance one step.

        `coolant_temp_c` is a real measurement -- BeamNG reports it over
        OutGauge. When given it is used verbatim and the warm-up model is
        bypassed, so the model never drifts away from measured data. The bay
        estimate is always modelled: nothing measures it.
        """
        state = self.state
        state.rpm = self._rpm(speed_mps)

        if coolant_temp_c is not None:
            state.coolant_temp_c = coolant_temp_c
        elif state.engine_on:
            # Load shortens the warm-up: more fuel burnt, more heat rejected.
            load = 0.25 + 0.75 * min(1.0, throttle)
            tau = WARMUP_TAU_S / load
            state.coolant_temp_c += (OPERATING_TEMP_C - state.coolant_temp_c) * (
                dt / tau
            )
        else:
            state.coolant_temp_c += (self.ambient_temp_c - state.coolant_temp_c) * (
                dt / COOLDOWN_TAU_S
            )

        # Bay temperature chases a coupling-weighted fraction of the coolant
        # rise. Airflow strips heat away; load adds it. With the car stopped
        # there is no airflow at all, which is why idling runs the bay hot.
        coupling = (
            BAY_COUPLING_STATIC
            * (1.0 + BAY_LOAD_GAIN * min(1.0, throttle))
            / (1.0 + BAY_AIRFLOW_GAIN * max(0.0, speed_mps))
        )
        if not state.engine_on:
            coupling = 0.0
        bay_target = self.ambient_temp_c + coupling * (
            state.coolant_temp_c - self.ambient_temp_c
        )
        state.underbonnet_temp_c += (bay_target - state.underbonnet_temp_c) * (
            dt / BAY_TAU_S
        )
        return state
