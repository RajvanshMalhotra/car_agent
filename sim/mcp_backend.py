"""BeamNG backend over the game's own MCP server.

BeamNG 0.39 exposes 86 first-party tools on `http://127.0.0.1:29292/mcp`. This
backend uses a handful of them and replaces the entire virtual-gamepad and UDP
stack: no ViGEmBus, no packet decoding, no re-plug trick, and no guessing at
coordinate conventions.

What it uses:

  inject_input        throttle, brake and steering as analog values
  get_status          position, speed and damage in one round trip
  get_electrics       rpm, gear, throttle, brake, coolant and oil temperature
  get_vehicles        other vehicles -- the first perception this project has had
  recover_vehicle     put the car back on the road between runs

Two quirks of the real server are handled here. `get_electrics` answers
asynchronously: the first call returns a sentence saying to ask again, so the
last good reading is held. And every tool call is an HTTP round trip, so
unchanged control values are not resent.

Under-bonnet temperature is still estimated -- none of the 86 tools reports the
temperature of the air around the battery.
"""

from __future__ import annotations

import math
import time
from typing import Any

from sim.backend import ControlInput, SimBackend, VehicleState
from sim.engine import EngineModel

#: Damage above this counts as a crash. BeamNG's damageSum is unitless and
#: grows with severity; a light kerb strike registers in the tens.
DEFAULT_CRASH_DAMAGE = 100.0

#: Below this speed, displacement is too small to give a heading.
HEADING_FROM_MOVEMENT_MPS = 1.0

#: A reply that is a string rather than a mapping is the server saying "asked,
#: come back in a moment".
def _is_async_notice(result: Any) -> bool:
    return not isinstance(result, dict)


class MCPBackend(SimBackend):
    def __init__(
        self,
        client: Any = None,
        endpoint: str | None = None,
        ambient_temp_c: float = 20.0,
        cold_start: bool = True,
        crash_damage: float = DEFAULT_CRASH_DAMAGE,
    ) -> None:
        if client is None:
            from sim.mcp_client import DEFAULT_ENDPOINT, MCPClient

            client = MCPClient(endpoint or DEFAULT_ENDPOINT)
            client.connect()
        self.client = client
        self.ambient_temp_c = ambient_temp_c
        self.cold_start = cold_start
        self.crash_damage = crash_damage

        self.engine = EngineModel(ambient_temp_c, cold_start=cold_start)
        self.engine.start()
        self.state = VehicleState(
            coolant_temp_c=ambient_temp_c,
            underbonnet_temp_c=ambient_temp_c,
            engine_on=True,
            crank_count=1,
        )

        self._vehicle_id: int | None = None
        self._origin: tuple[float, float] | None = None
        self._last_position: tuple[float, float] | None = None
        self._electrics: dict[str, Any] = {}
        self._sent: dict[str, float] = {}
        self._baseline_damage = 0.0
        self._damage = 0.0
        self._start_wall = time.monotonic()
        self._last_read = self._start_wall

    # -- plumbing ---------------------------------------------------------

    @property
    def vehicle_id(self) -> int:
        """Looked up once. It does not change while a vehicle is spawned."""
        if self._vehicle_id is None:
            self._vehicle_id = self.client.call("get_player_vehicle_id")
        return self._vehicle_id

    def _inject(self, event: str, value: float, force: bool = False) -> None:
        # Every call is an HTTP round trip; resending an unchanged value is
        # pure latency in a loop that runs many times a second.
        if not force and abs(self._sent.get(event, math.nan) - value) < 1e-4:
            return
        self.client.call(
            "inject_input", {"id": self.vehicle_id, "event": event, "value": value}
        )
        self._sent[event] = value

    # -- SimBackend -------------------------------------------------------

    def apply_control(self, control: ControlInput) -> None:
        self._inject("throttle", control.throttle)
        self._inject("brake", control.brake)
        self._inject("steering", control.steering)

    def read_state(self) -> VehicleState:
        now = time.monotonic()
        dt = max(1e-3, now - self._last_read)
        self._last_read = now
        state = self.state

        status = self.client.call("get_status")
        vehicle = (status or {}).get("vehicle", {}) if isinstance(status, dict) else {}
        position = vehicle.get("pos")
        if position:
            x, y = float(position["x"]), float(position["y"])
            if self._origin is None:
                self._origin = (x, y)
            if self._last_position is not None:
                dx, dy = x - self._last_position[0], y - self._last_position[1]
                # Heading from displacement: guaranteed to be in the same frame
                # as the positions, unlike a quaternion whose forward-axis
                # convention is undocumented.
                if math.hypot(dx, dy) > HEADING_FROM_MOVEMENT_MPS * dt:
                    state.heading_rad = math.atan2(dy, dx)
            self._last_position = (x, y)
            state.x_m = x - self._origin[0]
            state.y_m = y - self._origin[1]
        if "speed" in vehicle:
            state.speed_mps = float(vehicle["speed"])
        if "damage" in vehicle:
            self._damage = float(vehicle["damage"])
            state.damage = self._damage

        electrics = self.client.call("get_electrics")
        if not _is_async_notice(electrics):
            self._electrics = electrics
        if self._electrics:
            state.rpm = float(self._electrics.get("rpm", state.rpm))
            state.gear = int(self._electrics.get("gear", state.gear) or 0)
            state.fuel_fraction = float(
                self._electrics.get("fuel", state.fuel_fraction)
            )
            state.throttle = float(self._electrics.get("throttle", state.throttle))
            state.brake = float(self._electrics.get("brake", state.brake))
            state.oil_temp_c = float(
                self._electrics.get("oiltemp", state.oil_temp_c)
            )

        measured_coolant = (
            float(self._electrics["watertemp"])
            if "watertemp" in self._electrics
            else None
        )
        engine = self.engine.step(
            speed_mps=state.speed_mps,
            throttle=state.throttle,
            dt=dt,
            coolant_temp_c=measured_coolant,
        )
        state.coolant_temp_c = engine.coolant_temp_c
        state.underbonnet_temp_c = engine.underbonnet_temp_c
        state.engine_on = True
        state.crank_count = self.engine.state.crank_count
        state.sim_time_s = now - self._start_wall
        return state.snapshot()

    def reset(self) -> VehicleState:
        for event in ("throttle", "brake", "steering"):
            self._inject(event, 0.0, force=True)
        self.client.call("recover_vehicle", {"id": self.vehicle_id})

        self.engine = EngineModel(self.ambient_temp_c, cold_start=self.cold_start)
        self.engine.start()
        self._origin = None
        self._last_position = None
        self._start_wall = time.monotonic()
        self._last_read = self._start_wall
        self.state = VehicleState(
            coolant_temp_c=self.ambient_temp_c,
            underbonnet_temp_c=self.ambient_temp_c,
            engine_on=True,
            crank_count=1,
        )
        snapshot = self.read_state()
        # Damage already on the car is not this run's fault.
        self._baseline_damage = self._damage
        return snapshot

    def close(self) -> None:
        try:
            for event in ("throttle", "brake", "steering"):
                self._inject(event, 0.0, force=True)
        finally:
            close = getattr(self.client, "close", None)
            if close is not None:
                close()

    # -- perception, finally ----------------------------------------------

    @property
    def damage_since_start(self) -> float:
        return self._damage - self._baseline_damage

    def has_crashed(self) -> bool:
        """True once the vehicle has taken damage since the run began.

        This is the first real collision signal in the project: the UDP path
        had no way to know the car had hit anything.
        """
        return (self._damage - self._baseline_damage) > self.crash_damage

    def other_vehicles(self) -> list[dict[str, Any]]:
        """Every spawned vehicle, this one included. The basis for traffic."""
        result = self.client.call("get_vehicles")
        return result if isinstance(result, list) else []
