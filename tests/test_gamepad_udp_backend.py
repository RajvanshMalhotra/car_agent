"""The BeamNG backend, exercised over real localhost UDP with a stand-in pad.

Everything here runs on any OS: only the vgamepad import is Windows-only, and
that is injected.
"""

import socket
import struct
import time

import pytest

from sim.backend import ControlInput, SimBackend
from sim.gamepad_udp import GamepadUDPBackend
from sim.telemetry import OUTGAUGE_LAYOUT, OUTSIM_LAYOUT

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


@pytest.fixture
def backend():
    pad = FakePad()
    instance = GamepadUDPBackend(
        ambient_temp_c=30.0,
        outgauge_port=OUTGAUGE_PORT,
        outsim_port=OUTSIM_PORT,
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
