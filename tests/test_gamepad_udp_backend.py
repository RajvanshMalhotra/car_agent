"""The BeamNG backend, exercised over real localhost UDP with a stand-in pad.

Everything here runs on any OS: only the vgamepad import is Windows-only, and
that is injected.
"""

import math
import socket
import struct
import time

import pytest

from sim.backend import ControlInput, SimBackend
from sim.gamepad_udp import GamepadUDPBackend
from sim.telemetry import MOTIONSIM_LAYOUT, MOTIONSIM_MAGIC, OUTGAUGE_LAYOUT, OUTSIM_LAYOUT

OUTGAUGE_PORT = 47444
OUTSIM_PORT = 47445


class FakePad:
    def __init__(self):
        self.steering = None
        self.throttle = None
        self.brake = None
        self.updates = 0

    def left_joystick_float(self, x_value_float, y_value_float):
        self.steering = x_value_float

    def right_trigger_float(self, value_float):
        self.throttle = value_float

    def left_trigger_float(self, value_float):
        self.brake = value_float

    def update(self):
        self.updates += 1


def send_motionsim(x=0.0, y=0.0, vx=0.0, vy=0.0, yaw=0.0, port=OUTGAUGE_PORT):
    send(
        struct.pack(
            MOTIONSIM_LAYOUT, MOTIONSIM_MAGIC, x, y, 120.0, vx, vy, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, yaw,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ),
        port,
    )


@pytest.fixture
def backend():
    pad = FakePad()
    instance = GamepadUDPBackend(
        ambient_temp_c=30.0,
        ports=[OUTGAUGE_PORT, OUTSIM_PORT],
        gamepad=pad,
        bind_host="127.0.0.1",
    )
    instance.pad = pad
    yield instance
    instance.close()


def send(data, port):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.sendto(data, ("127.0.0.1", port))


def send_outsim(x=0.0, y=0.0, heading=0.0, speed=0.0, port=OUTSIM_PORT):
    send(
        struct.pack(
            OUTSIM_LAYOUT, 0, 0, 0, 0, heading, 0, 0, 0, 0, 0,
            speed, 0.0, 0.0, int(x * 65536), int(y * 65536), 0,
        ),
        port,
    )


def send_outgauge(speed=0.0, rpm=800.0, coolant=90.0, throttle=0.0, port=OUTGAUGE_PORT):
    send(
        struct.pack(
            OUTGAUGE_LAYOUT, 0, b"etk8", 0, 2, 0, speed, rpm, 0.0, coolant,
            0.7, 3.5, 95.0, 0, 0, throttle, 0.0, 0.0, b"", b"",
        ),
        port,
    )


def settle(backend, seconds=0.15):
    time.sleep(seconds)
    return backend.read_state()


def test_it_implements_the_sim_boundary():
    assert issubclass(GamepadUDPBackend, SimBackend)


def test_throttle_and_brake_reach_the_triggers(backend):
    backend.apply_control(ControlInput(throttle=0.7, brake=0.2, steering=0.0))
    assert backend.pad.throttle == pytest.approx(0.7)
    assert backend.pad.brake == pytest.approx(0.2)


def test_steering_is_inverted_for_the_pad(backend):
    # Our convention is positive-left (ISO 8855); a gamepad stick is positive-right.
    backend.apply_control(ControlInput(throttle=0.0, brake=0.0, steering=1.0))
    assert backend.pad.steering == pytest.approx(-1.0)


def test_every_control_is_pushed_to_the_device(backend):
    backend.apply_control(ControlInput(0.0, 0.0, 0.0))
    backend.apply_control(ControlInput(0.0, 0.0, 0.0))
    assert backend.pad.updates == 2


def test_pose_arrives_from_outsim(backend):
    backend.reset()
    send_outsim(x=100.0, y=50.0, heading=0.75, speed=12.0)
    state = settle(backend)
    assert state.heading_rad == pytest.approx(0.75)
    assert state.speed_mps == pytest.approx(12.0, abs=0.01)


def test_position_is_reported_relative_to_where_the_run_started(backend):
    send_outsim(x=1000.0, y=-500.0)
    settle(backend)
    send_outsim(x=1030.0, y=-500.0)
    state = settle(backend)
    assert state.x_m == pytest.approx(30.0, abs=0.01)
    assert state.y_m == pytest.approx(0.0, abs=0.01)


def test_only_the_newest_pose_is_used(backend):
    # UDP buffers. A stale pose is worse than none -- the controller must act on
    # the latest frame, not work through a backlog.
    for x in (10.0, 20.0, 30.0, 40.0):
        send_outsim(x=x)
    # The backlog is discarded, so the origin comes from x=40, not x=10.
    assert settle(backend).x_m == pytest.approx(0.0, abs=0.01)
    send_outsim(x=90.0)
    assert settle(backend).x_m == pytest.approx(50.0, abs=0.01)


def test_engine_channels_arrive_from_outgauge(backend):
    send_outgauge(speed=18.0, rpm=2900.0, coolant=87.5)
    state = settle(backend)
    assert state.rpm == pytest.approx(2900.0)
    assert state.coolant_temp_c == pytest.approx(87.5)


def test_the_real_coolant_temperature_overrides_the_model(backend):
    send_outgauge(coolant=64.0)
    settle(backend)
    assert backend.engine.state.coolant_temp_c == pytest.approx(64.0)


def test_the_bay_temperature_is_estimated_because_no_stream_carries_it(backend):
    for _ in range(20):
        send_outgauge(speed=0.0, coolant=95.0, throttle=0.1)
        settle(backend, 0.05)
    state = backend.read_state()
    assert 30.0 < state.underbonnet_temp_c < 95.0


def test_a_malformed_datagram_does_not_bring_the_run_down(backend):
    send(b"garbage", OUTSIM_PORT)
    assert settle(backend) is not None


def test_read_state_returns_a_snapshot(backend):
    send_outsim(x=0.0)
    settle(backend)
    held = backend.read_state()
    send_outsim(x=200.0)
    settle(backend)
    assert held.x_m == pytest.approx(0.0, abs=0.01)


def test_close_is_idempotent(backend):
    backend.close()
    backend.close()


# --- both streams on one port (what BeamNG actually does) ------------------


def test_motionsim_and_outgauge_are_demultiplexed_on_a_shared_port(backend):
    # BeamNG sends both to whatever port each is configured with, and in
    # practice that is the same port. Routing must be by content, not by port.
    backend.reset()
    send_motionsim(x=200.0, y=50.0, vx=10.0, vy=0.0, port=OUTGAUGE_PORT)
    send_outgauge(speed=10.0, rpm=3100.0, coolant=85.3, port=OUTGAUGE_PORT)
    state = settle(backend)
    assert state.rpm == pytest.approx(3100.0)
    assert state.coolant_temp_c == pytest.approx(85.3)
    assert state.x_m == pytest.approx(0.0, abs=0.01)  # first pose set the origin


def test_pose_arrives_from_motionsim(backend):
    backend.reset()
    send_motionsim(x=100.0, y=20.0, vx=5.0, vy=0.0)
    settle(backend)
    send_motionsim(x=130.0, y=20.0, vx=5.0, vy=0.0)
    state = settle(backend)
    assert state.x_m == pytest.approx(30.0, abs=0.01)
    assert state.y_m == pytest.approx(0.0, abs=0.01)


def test_heading_from_motionsim_follows_the_velocity_vector(backend):
    backend.reset()
    send_motionsim(vx=0.0, vy=8.0, yaw=99.0)
    assert settle(backend).heading_rad == pytest.approx(math.pi / 2, abs=1e-3)


def test_a_single_port_is_enough(backend_factory=None):
    pad = FakePad()
    backend = GamepadUDPBackend(
        ambient_temp_c=20.0, ports=[OUTGAUGE_PORT], gamepad=pad, bind_host="127.0.0.1"
    )
    try:
        backend.reset()
        send_motionsim(x=10.0, vx=4.0, port=OUTGAUGE_PORT)
        send_outgauge(rpm=2000.0, port=OUTGAUGE_PORT)
        state = settle(backend)
        assert state.rpm == pytest.approx(2000.0)
    finally:
        backend.close()


def test_an_unrecognised_datagram_is_counted_not_crashed_on(backend):
    send(b"x" * 51, OUTGAUGE_PORT)
    settle(backend)
    assert backend.unknown_packets == 1
