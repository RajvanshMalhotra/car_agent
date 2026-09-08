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
        if name in ("set_ai", "drive_to"):
            return "ok"
        if name == "get_navgraph":
            return {"nodeCount": 4213}
        if name == "set_position":
            self.pos = dict(arguments["pos"])
            self.damage = 0.0
            return "ok"
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


# -- the real server's types are not the ones we assumed ------------------


def test_a_neutral_gear_reported_as_a_letter_is_understood(backend, stub):
    # BeamNG reports gear as a display string: 'N', 'R', 'P', 'D' or a number.
    stub.electrics["gear"] = "N"
    backend.read_state()
    assert backend.read_state().gear == 0


def test_reverse_is_understood(backend, stub):
    stub.electrics["gear"] = "R"
    backend.read_state()
    assert backend.read_state().gear == -1


def test_park_is_understood(backend, stub):
    stub.electrics["gear"] = "P"
    backend.read_state()
    assert backend.read_state().gear == 0


def test_a_numbered_gear_as_a_string_is_understood(backend, stub):
    stub.electrics["gear"] = "4"
    backend.read_state()
    assert backend.read_state().gear == 4


def test_a_drive_gear_prefers_the_numeric_index_when_present(backend, stub):
    stub.electrics["gear"] = "D"
    stub.electrics["gearIndex"] = 3
    backend.read_state()
    assert backend.read_state().gear == 3


def test_an_unrecognisable_gear_does_not_bring_the_run_down(backend, stub):
    stub.electrics["gear"] = "M1"
    backend.read_state()
    assert isinstance(backend.read_state().gear, int)


def test_a_missing_numeric_field_does_not_bring_the_run_down(backend, stub):
    stub.electrics["rpm"] = None
    backend.read_state()
    assert backend.read_state().rpm == 0.0


def test_a_numeric_field_arriving_as_a_string_is_still_used(backend, stub):
    stub.electrics["rpm"] = "2750.5"
    backend.read_state()
    assert backend.read_state().rpm == pytest.approx(2750.5)


def test_junk_in_a_numeric_field_is_ignored_rather_than_fatal(backend, stub):
    stub.electrics["watertemp"] = "n/a"
    backend.read_state()
    assert backend.read_state().coolant_temp_c > 0.0


# -- world coordinates, for following a recorded drive --------------------


def test_position_is_rebased_to_the_start_by_default(backend, stub):
    stub.pos = {"x": 9531.4, "y": 383.4, "z": 0.6}
    backend.reset()
    assert backend.read_state().x_m == pytest.approx(0.0, abs=0.01)


def test_world_coordinates_can_be_reported_instead(stub):
    # A recorded drive is a fixed place in the world. Rebasing to wherever this
    # run happened to start puts the vehicle kilometres from its own route.
    stub.pos = {"x": 9531.4, "y": 383.4, "z": 0.6}
    backend = MCPBackend(client=stub, ambient_temp_c=20.0, rebase_origin=False)
    try:
        backend.reset()
        state = backend.read_state()
        assert state.x_m == pytest.approx(9531.4, abs=0.01)
        assert state.y_m == pytest.approx(383.4, abs=0.01)
    finally:
        backend.close()


def test_world_coordinates_still_move_with_the_vehicle(stub):
    backend = MCPBackend(client=stub, ambient_temp_c=20.0, rebase_origin=False)
    try:
        backend.reset()
        stub.pos = {"x": 9561.4, "y": 383.4, "z": 0.6}
        assert backend.read_state().x_m == pytest.approx(9561.4, abs=0.01)
    finally:
        backend.close()


def test_a_recovery_does_not_rebase_world_coordinates(stub):
    backend = MCPBackend(client=stub, ambient_temp_c=20.0, rebase_origin=False)
    try:
        backend.reset()
        stub.pos = {"x": 9600.0, "y": 400.0, "z": 0.6}
        backend.recover()
        assert backend.read_state().x_m == pytest.approx(9600.0, abs=0.01)
    finally:
        backend.close()


# -- putting the car back at the start ------------------------------------


def test_the_vehicle_can_be_moved_to_a_point(backend, stub):
    backend.teleport_to(9600.0, 420.0)
    call = [args for name, args in stub.calls if name == "set_position"][0]
    assert call["pos"]["x"] == pytest.approx(9600.0)
    assert call["pos"]["y"] == pytest.approx(420.0)


def test_the_teleport_is_addressed_to_the_player_vehicle(backend, stub):
    backend.teleport_to(1.0, 2.0)
    call = [args for name, args in stub.calls if name == "set_position"][0]
    assert call["id"] == 87942


def test_the_controls_are_released_before_teleporting(backend, stub):
    backend.apply_control(ControlInput(1.0, 0.0, 0.5))
    backend.teleport_to(1.0, 2.0)
    assert stub.inputs["throttle"] == pytest.approx(0.0)


def test_teleporting_does_not_count_as_a_crash(backend, stub):
    # set_position repairs the vehicle, and a repair is not damage this run did.
    backend.reset()
    stub.damage = 900.0
    backend.teleport_to(1.0, 2.0)
    assert not backend.has_crashed()


def test_the_reported_height_is_kept_if_not_given(backend, stub):
    backend.teleport_to(1.0, 2.0)
    call = [args for name, args in stub.calls if name == "set_position"][0]
    assert "z" in call["pos"]


# -- handing the wheel to the game's own AI --------------------------------


def test_the_ai_can_be_configured(backend, stub):
    backend.set_ai(mode="manual", aggression=0.8, avoidCars=True)
    call = [args for name, args in stub.calls if name == "set_ai"][0]
    assert call["aggression"] == pytest.approx(0.8)
    assert call["mode"] == "manual"
    assert call["id"] == 87942


def test_the_ai_can_be_sent_to_a_point(backend, stub):
    backend.drive_to(x=100.0, y=250.0, aggression=0.9, driveInLane=True)
    call = [args for name, args in stub.calls if name == "drive_to"][0]
    assert call["pos"]["x"] == pytest.approx(100.0)
    assert call["pos"]["y"] == pytest.approx(250.0)
    assert call["driveInLane"] is True


def test_the_drive_target_carries_a_height(backend, stub):
    # drive_to snaps to the nearest navgraph node, but the point still needs
    # all three coordinates.
    backend.drive_to(x=1.0, y=2.0)
    assert "z" in [args for name, args in stub.calls if name == "drive_to"][0]["pos"]


def test_the_road_network_can_be_checked(backend, stub):
    # A level with no roads has no navgraph, and the AI cannot drive at all.
    assert backend.has_road_network() in (True, False)


def test_the_vehicle_can_be_repaired_where_it_stands(backend, stub):
    backend.repair()
    assert any(name == "reset_vehicle" for name, _ in stub.calls)


def test_repairing_does_not_count_as_a_crash(backend, stub):
    backend.reset()
    stub.damage = 900.0
    backend.read_state()
    backend.repair()
    assert not backend.has_crashed()


def test_repairing_releases_the_controls_first(backend, stub):
    backend.apply_control(ControlInput(1.0, 0.0, 0.5))
    backend.repair()
    assert stub.inputs["throttle"] == pytest.approx(0.0)
