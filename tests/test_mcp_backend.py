"""Driving BeamNG through its own MCP server.

This replaces the whole virtual-gamepad and UDP stack: control via
`inject_input`, state via `get_status` and `get_electrics`. No ViGEmBus, no
packet decoding, no ambiguity about which way the heading points.

Tested against a stub that reproduces the real server's quirks -- notably that
`get_electrics` answers asynchronously the first time it is asked.
"""

import math

import pytest

from sim.backend import ControlInput, SimBackend, VehicleState
from sim.mcp_backend import MCPBackend

ASYNC_NOTICE = "electrics requested (async); call again in a moment"


class StubServer:
    """Stands in for BeamNG's MCP server, quirks included."""

    def __init__(self):
        self.calls = []
        self.inputs = {}
        self.pos = {"x": 100.0, "y": 200.0, "z": 0.6}
        self.speed = 0.0
        self.damage = 0.0
        self.electrics_ready = False
        self.electrics = {
            "rpm": 2400.0, "gear": 3, "fuel": 0.6, "throttle": 0.4,
            "brake": 0.0, "watertemp": 88.0, "oiltemp": 95.0, "wheelspeed": 12.0,
        }
        self.vehicles = [{"id": 1, "jbeam": "etk800"}]

    def call(self, name, arguments=None):
        arguments = arguments or {}
        self.calls.append((name, arguments))
        if name == "get_player_vehicle_id":
            return 87942
        if name == "get_status":
            return {
                "sim": {"physicsRunning": True, "timeScale": 1},
                "vehicle": {"id": 87942, "pos": dict(self.pos),
                            "speed": self.speed, "damage": self.damage},
            }
        if name == "get_electrics":
            if not self.electrics_ready:
                self.electrics_ready = True
                return ASYNC_NOTICE
            return dict(self.electrics)
        if name == "inject_input":
            self.inputs[arguments["event"]] = arguments["value"]
            return "ok"
        if name == "get_vehicles":
            return list(self.vehicles)
        if name == "get_vehicle_damage":
            return {"id": 87942, "damageSum": self.damage}
        if name in ("reset_vehicle", "recover_vehicle", "pause_physics",
                    "resume_physics"):
            return "ok"
        raise AssertionError(f"unexpected tool {name}")

    def close(self):
        pass


@pytest.fixture
def stub():
    return StubServer()


@pytest.fixture
def backend(stub):
    instance = MCPBackend(client=stub, ambient_temp_c=30.0)
    yield instance
    instance.close()


def tool_calls(stub, name):
    return [args for called, args in stub.calls if called == name]


# -- the boundary ---------------------------------------------------------


def test_it_implements_the_sim_boundary():
    assert issubclass(MCPBackend, SimBackend)


def test_read_state_returns_a_vehicle_state(backend):
    assert isinstance(backend.read_state(), VehicleState)


def test_read_state_returns_a_snapshot(backend, stub):
    held = backend.read_state()
    stub.speed = 25.0
    backend.read_state()
    assert held.speed_mps != 25.0


# -- control --------------------------------------------------------------


def test_throttle_brake_and_steering_are_injected(backend, stub):
    backend.apply_control(ControlInput(throttle=0.6, brake=0.1, steering=0.4))
    assert stub.inputs["throttle"] == pytest.approx(0.6)
    assert stub.inputs["brake"] == pytest.approx(0.1)
    assert stub.inputs["steering"] == pytest.approx(0.4)


def test_control_is_addressed_to_the_player_vehicle(backend, stub):
    backend.apply_control(ControlInput(0.5, 0.0, 0.0))
    assert all(args["id"] == 87942 for args in tool_calls(stub, "inject_input"))


def test_the_vehicle_id_is_looked_up_once_not_every_step(backend, stub):
    for _ in range(5):
        backend.apply_control(ControlInput(0.5, 0.0, 0.0))
    assert len(tool_calls(stub, "get_player_vehicle_id")) == 1


def test_unchanged_controls_are_not_resent(backend, stub):
    # One HTTP round trip per input per step is the cost model here; resending
    # an unchanged value is pure latency.
    backend.apply_control(ControlInput(0.5, 0.0, 0.0))
    before = len(tool_calls(stub, "inject_input"))
    backend.apply_control(ControlInput(0.5, 0.0, 0.0))
    assert len(tool_calls(stub, "inject_input")) == before


def test_a_changed_control_is_resent(backend, stub):
    backend.apply_control(ControlInput(0.5, 0.0, 0.0))
    before = len(tool_calls(stub, "inject_input"))
    backend.apply_control(ControlInput(0.9, 0.0, 0.0))
    assert len(tool_calls(stub, "inject_input")) > before


# -- state ----------------------------------------------------------------


def test_position_is_relative_to_where_the_run_started(backend, stub):
    backend.reset()
    stub.pos = {"x": 130.0, "y": 200.0, "z": 0.6}
    state = backend.read_state()
    assert state.x_m == pytest.approx(30.0)
    assert state.y_m == pytest.approx(0.0)


def test_speed_comes_from_the_status_snapshot(backend, stub):
    stub.speed = 18.5
    assert backend.read_state().speed_mps == pytest.approx(18.5)


def test_heading_is_derived_from_movement(backend, stub):
    # Same reasoning as the UDP backend: displacement is guaranteed to be in
    # the frame the positions are in, unlike a quaternion whose forward-axis
    # convention is undocumented.
    backend.reset()
    stub.pos = {"x": 100.0, "y": 210.0, "z": 0.6}
    stub.speed = 10.0
    assert backend.read_state().heading_rad == pytest.approx(math.pi / 2, abs=1e-3)


def test_heading_is_held_while_stationary(backend, stub):
    backend.reset()
    stub.pos, stub.speed = {"x": 110.0, "y": 200.0, "z": 0.6}, 10.0
    backend.read_state()
    stub.speed = 0.0
    assert backend.read_state().heading_rad == pytest.approx(0.0, abs=1e-3)


def test_the_async_electrics_reply_is_not_mistaken_for_data(backend, stub):
    # The first get_electrics returns a sentence, not a dict.
    assert backend.read_state().rpm == 0.0
    assert backend.read_state().rpm == pytest.approx(2400.0)


def test_engine_channels_come_from_electrics(backend, stub):
    backend.read_state()
    state = backend.read_state()
    assert state.rpm == pytest.approx(2400.0)
    assert state.gear == 3
    assert state.coolant_temp_c == pytest.approx(88.0)
    assert state.oil_temp_c == pytest.approx(95.0)
    assert state.throttle == pytest.approx(0.4)


def test_the_last_good_electrics_survive_an_async_reply(backend, stub):
    backend.read_state()
    backend.read_state()
    stub.electrics_ready = False  # server answers async again
    assert backend.read_state().rpm == pytest.approx(2400.0)


def test_the_engine_bay_temperature_is_still_estimated(backend, stub):
    # Nothing in 86 tools reports under-bonnet air temperature.
    backend.read_state()
    state = backend.read_state()
    assert state.underbonnet_temp_c > 0.0


# -- collisions, at last --------------------------------------------------


def test_an_undamaged_vehicle_has_not_crashed(backend, stub):
    backend.reset()
    assert not backend.has_crashed()


def test_damage_is_reported_as_a_crash(backend, stub):
    backend.reset()
    stub.damage = 500.0
    backend.read_state()
    assert backend.has_crashed()


def test_the_crash_threshold_is_configurable(stub):
    backend = MCPBackend(client=stub, ambient_temp_c=20.0, crash_damage=10.0)
    backend.reset()
    stub.damage = 50.0
    backend.read_state()
    assert backend.has_crashed()


def test_damage_present_at_the_start_is_not_counted_as_a_new_crash(stub):
    stub.damage = 300.0
    backend = MCPBackend(client=stub, ambient_temp_c=20.0)
    backend.reset()
    backend.read_state()
    assert not backend.has_crashed()


def test_other_vehicles_can_be_listed(backend, stub):
    stub.vehicles = [{"id": 1}, {"id": 2}, {"id": 3}]
    assert len(backend.other_vehicles()) == 3


# -- reset ----------------------------------------------------------------


def test_reset_repairs_the_vehicle(backend, stub):
    backend.reset()
    assert tool_calls(stub, "recover_vehicle") or tool_calls(stub, "reset_vehicle")


def test_reset_releases_the_controls(backend, stub):
    backend.apply_control(ControlInput(1.0, 0.0, 0.5))
    backend.reset()
    assert stub.inputs["throttle"] == pytest.approx(0.0)
    assert stub.inputs["brake"] == pytest.approx(0.0)


def test_reset_rebases_the_origin(backend, stub):
    stub.pos = {"x": 500.0, "y": 900.0, "z": 1.0}
    state = backend.reset()
    assert (state.x_m, state.y_m) == pytest.approx((0.0, 0.0))
