"""Kinematic-bicycle stand-in for BeamNG.

Not a soft-body simulation and not trying to be. Its job is to make the
controller, campaign and logging layers testable without the Windows machine.
"""

from __future__ import annotations

import math
import random

from sim.backend import ControlInput, SimBackend, VehicleState
from sim.engine import EngineModel


class FakeBackend(SimBackend):
    def __init__(
        self,
        dt: float = 0.02,
        wheelbase_m: float = 2.7,
        max_steer_rad: float = 0.5,
        max_accel_mps2: float = 3.0,
        max_decel_mps2: float = 8.0,
        rolling_drag: float = 0.02,
        aero_drag: float = 0.0004,
        seed: int = 0,
        ambient_temp_c: float = 20.0,
        cold_start: bool = True,
    ) -> None:
        self.dt = dt
        self.wheelbase_m = wheelbase_m
        self.max_steer_rad = max_steer_rad
        self.max_accel_mps2 = max_accel_mps2
        self.max_decel_mps2 = max_decel_mps2
        self.rolling_drag = rolling_drag
        self.aero_drag = aero_drag
        self.rng = random.Random(seed)
        self.ambient_temp_c = ambient_temp_c
        self.cold_start = cold_start
        self.engine = EngineModel(ambient_temp_c, cold_start=cold_start)
        self.state = self._fresh_state()

    def _fresh_state(self) -> VehicleState:
        return VehicleState(
            coolant_temp_c=self.ambient_temp_c,
            underbonnet_temp_c=self.ambient_temp_c,
        )

    def apply_control(self, control: ControlInput) -> None:
        state = self.state
        speed = state.speed_mps

        # Pose integrates on the speed at the start of the step, so yaw rate is
        # exactly v/L * tan(delta) for the reported speed.
        steer_rad = control.steering * self.max_steer_rad
        state.x_m += speed * math.cos(state.heading_rad) * self.dt
        state.y_m += speed * math.sin(state.heading_rad) * self.dt
        state.heading_rad += speed / self.wheelbase_m * math.tan(steer_rad) * self.dt

        drag = self.rolling_drag * speed + self.aero_drag * speed * speed
        accel = (
            control.throttle * self.max_accel_mps2
            - control.brake * self.max_decel_mps2
            - drag
        )
        state.speed_mps = max(0.0, speed + accel * self.dt)
        state.sim_time_s += self.dt

        engine = self.engine.step(
            speed_mps=state.speed_mps, throttle=control.throttle, dt=self.dt
        )
        state.rpm = engine.rpm
        state.throttle = control.throttle
        state.brake = control.brake
        state.coolant_temp_c = engine.coolant_temp_c
        state.underbonnet_temp_c = engine.underbonnet_temp_c
        state.engine_on = engine.engine_on
        state.crank_count = engine.crank_count

    def read_state(self) -> VehicleState:
        return self.state.snapshot()

    def reset(self) -> VehicleState:
        self.engine = EngineModel(self.ambient_temp_c, cold_start=self.cold_start)
        self.engine.start()
        self.state = self._fresh_state()
        self.state.engine_on = True
        self.state.crank_count = self.engine.state.crank_count
        if not self.cold_start:
            self.state.coolant_temp_c = self.engine.state.coolant_temp_c
        return self.state.snapshot()

    def close(self) -> None:
        return None
