"""BeamNG backend: virtual gamepad out, UDP telemetry in.

This is the Windows-only half. It implements the same `SimBackend` interface as
the fake backend, so the controller, behaviour and logging layers are unchanged.

Control goes through a virtual Xbox 360 pad (vgamepad + the ViGEmBus driver)
because BeamNG.drive has no scripting API -- `beamngpy` needs BeamNG.tech.
Telemetry comes back over UDP: MotionSim for pose, OutGauge for the engine.
BeamNG sends both to whichever port each is configured with, and in practice
that is the same port -- so every configured port is bound and each datagram is
routed by its content, not by which socket it arrived on.

Under-bonnet temperature is not in either stream, so it is estimated by the same
`EngineModel` the fake backend uses, driven by the *real* coolant temperature
and road speed rather than by a modelled engine.
"""

from __future__ import annotations

import socket
import time

from sim.backend import ControlInput, SimBackend, VehicleState
from sim.engine import EngineModel
from sim.telemetry import (
    MotionSimPacket,
    OutGaugePacket,
    OutSimPacket,
    TelemetryError,
    identify,
)

#: Bind both by default: BeamNG may be configured either way, and binding a
#: port nothing sends to costs nothing.
DEFAULT_PORTS = (4444, 4445)


class GamepadUDPBackend(SimBackend):
    def __init__(
        self,
        ambient_temp_c: float = 20.0,
        cold_start: bool = True,
        ports: "tuple[int, ...] | list[int]" = DEFAULT_PORTS,
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

        self.unknown_packets = 0
        self.packet_counts = {"outgauge": 0, "motionsim": 0, "outsim": 0}
        self.gamepad = gamepad if gamepad is not None else self._open_gamepad()
        self.sockets = [self._bind(bind_host, port) for port in dict.fromkeys(ports)]

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
        for sock in getattr(self, "sockets", []):
            sock.close()

    # -- telemetry --------------------------------------------------------

    def _collect(self) -> dict[str, bytes]:
        """Newest datagram of each kind, across every bound port.

        UDP buffers, and a stale pose is worse than none: the controller must
        act on the most recent frame, not work through a queue.
        """
        newest: dict[str, bytes] = {}
        for sock in self.sockets:
            while True:
                try:
                    data = sock.recv(4096)
                except BlockingIOError:
                    break
                kind = identify(data)
                if kind is None:
                    self.unknown_packets += 1
                    continue
                self.packet_counts[kind] += 1
                newest[kind] = data
        return newest

    def _drain(self) -> None:
        now = time.monotonic()
        dt = max(1e-3, now - self._last_read)
        self._last_read = now
        state = self.state
        newest = self._collect()

        pose = None
        if "motionsim" in newest:
            pose = self._safe(MotionSimPacket, newest["motionsim"])
        elif "outsim" in newest:
            pose = self._safe(OutSimPacket, newest["outsim"])
        if pose is not None:
            if self._origin is None:
                self._origin = (pose.x_m, pose.y_m)
            state.x_m = pose.x_m - self._origin[0]
            state.y_m = pose.y_m - self._origin[1]
            state.heading_rad = pose.heading_rad
            state.speed_mps = pose.speed_mps

        throttle = 0.0
        measured_coolant: float | None = None
        if "outgauge" in newest:
            packet = self._safe(OutGaugePacket, newest["outgauge"])
            if packet is not None:
                state.rpm = packet.rpm
                # OutGauge speed is authoritative; the pose stream's velocity
                # vector is only used for heading.
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

    @staticmethod
    def _safe(parser, data: bytes):
        """Parse, or return None. One bad datagram must not end a 10-minute run."""
        try:
            return parser.parse(data)
        except TelemetryError:
            return None
