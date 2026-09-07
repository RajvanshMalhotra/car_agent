"""BeamNG backend: virtual gamepad out, UDP telemetry in.

This is the Windows-only half. It implements the same `SimBackend` interface as
the fake backend, so the controller, behaviour and logging layers are unchanged.

Control goes through a virtual Xbox 360 pad (vgamepad + the ViGEmBus driver)
because BeamNG.drive has no scripting API -- `beamngpy` needs BeamNG.tech.
Telemetry comes back over UDP: OutSim for pose, OutGauge for the engine.

Under-bonnet temperature is not in either stream, so it is estimated by the same
`EngineModel` the fake backend uses, driven by the *real* coolant temperature
and road speed rather than by a modelled engine.
"""

from __future__ import annotations

import socket
import time

from sim.backend import ControlInput, SimBackend, VehicleState
from sim.engine import EngineModel
from sim.telemetry import OutGaugePacket, OutSimPacket, TelemetryError

OUTGAUGE_PORT = 4444
OUTSIM_PORT = 4445


class GamepadUDPBackend(SimBackend):
    def __init__(
        self,
        ambient_temp_c: float = 20.0,
        cold_start: bool = True,
        outgauge_port: int = OUTGAUGE_PORT,
        outsim_port: int = OUTSIM_PORT,
        gamepad=None,
        bind_host: str = "0.0.0.0",
    ) -> None:
        self.ambient_temp_c = ambient_temp_c
        self.engine = EngineModel(ambient_temp_c, cold_start=cold_start)
        self.engine.start()
        self.state = VehicleState(
            coolant_temp_c=ambient_temp_c,
            underbonnet_temp_c=ambient_temp_c,
            engine_on=True,
            crank_count=1,
        )
        self._last_read = time.monotonic()
        self._start_wall = self._last_read
        self._origin: tuple[float, float] | None = None

        self.gamepad = gamepad if gamepad is not None else self._open_gamepad()
        self.outgauge = self._bind(bind_host, outgauge_port)
        self.outsim = self._bind(bind_host, outsim_port)

    @staticmethod
    def _open_gamepad():
        try:
            import vgamepad
        except ImportError as error:  # pragma: no cover - Windows only
            raise RuntimeError(
                "vgamepad is not installed. Run: py -m pip install vgamepad\n"
                "It also needs the ViGEmBus driver: "
                "https://github.com/nefarius/ViGEmBus/releases"
            ) from error
        return vgamepad.VX360Gamepad()

    @staticmethod
    def _bind(host: str, port: int) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        sock.setblocking(False)
        return sock

    # -- SimBackend -------------------------------------------------------

    def apply_control(self, control: ControlInput) -> None:
        """Write the control to the virtual pad.

        Unlike the fake backend this does not advance time -- the game is
        running in real time and `read_state` reports whatever has arrived.
        """
        self.gamepad.left_joystick_float(
            x_value_float=-control.steering,  # pad +x is right; our +steer is left
            y_value_float=0.0,
        )
        self.gamepad.right_trigger_float(value_float=control.throttle)
        self.gamepad.left_trigger_float(value_float=control.brake)
        self.gamepad.update()

    def read_state(self) -> VehicleState:
        self._drain()
        return self.state.snapshot()

    def reset(self) -> VehicleState:
        """Release the controls and re-zero the pose origin.

        The vehicle itself cannot be teleported without BeamNG.tech -- respawn
        it in-game (Ctrl+R) before starting a run.
        """
        self.apply_control(ControlInput(0.0, 0.0, 0.0))
        self.engine = EngineModel(self.ambient_temp_c, cold_start=True)
        self.engine.start()
        self._origin = None
        self._start_wall = time.monotonic()
        self._last_read = self._start_wall
        self.state = VehicleState(
            coolant_temp_c=self.ambient_temp_c,
            underbonnet_temp_c=self.ambient_temp_c,
            engine_on=True,
            crank_count=1,
        )
        self._drain()
        return self.state.snapshot()

    def close(self) -> None:
        for sock in (getattr(self, "outgauge", None), getattr(self, "outsim", None)):
            if sock is not None:
                sock.close()

    # -- telemetry --------------------------------------------------------

    def _latest(self, sock: socket.socket) -> bytes | None:
        """Newest datagram on the socket, discarding any backlog.

        UDP buffers, and a stale pose is worse than none: the controller must
        act on the most recent frame, not work through a queue.
        """
        newest = None
        while True:
            try:
                newest = sock.recv(4096)
            except BlockingIOError:
                return newest

    def _drain(self) -> None:
        now = time.monotonic()
        dt = max(1e-3, now - self._last_read)
        self._last_read = now
        state = self.state

        pose = self._latest(self.outsim)
        if pose is not None:
            try:
                packet = OutSimPacket.parse(pose)
            except TelemetryError:
                packet = None
            if packet is not None:
                if self._origin is None:
                    self._origin = (packet.x_m, packet.y_m)
                state.x_m = packet.x_m - self._origin[0]
                state.y_m = packet.y_m - self._origin[1]
                state.heading_rad = packet.heading_rad
                state.speed_mps = packet.speed_mps

        engine = self._latest(self.outgauge)
        throttle = 0.0
        measured_coolant: float | None = None
        if engine is not None:
            try:
                packet = OutGaugePacket.parse(engine)
            except TelemetryError:
                packet = None
            if packet is not None:
                state.rpm = packet.rpm
                state.speed_mps = packet.speed_mps
                state.throttle = packet.throttle
                state.brake = packet.brake
                state.oil_temp_c = packet.oil_temp_c
                state.gear = packet.gear
                state.fuel_fraction = packet.fuel_fraction
                throttle = packet.throttle
                # Real coolant temperature is used verbatim; only the bay
                # estimate is model-driven, because nothing reports it.
                measured_coolant = packet.coolant_temp_c

        bay = self.engine.step(
            speed_mps=state.speed_mps,
            throttle=throttle,
            dt=dt,
            coolant_temp_c=measured_coolant,
        )
        state.coolant_temp_c = bay.coolant_temp_c
        state.underbonnet_temp_c = bay.underbonnet_temp_c
        state.engine_on = True
        state.crank_count = self.engine.state.crank_count
        state.sim_time_s = now - self._start_wall
