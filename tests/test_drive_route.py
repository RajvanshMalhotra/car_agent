"""Tests for the route driver.

Everything here runs against a fake MCP client. The game is on someone else's
Windows laptop, and a driver that can only be tested there is a driver that
stops being tested -- which is how three wrong fixes shipped in a row.
"""

import json

import pytest

import drive_route
from collect.vm import VehicleVM
from drive_route import Ai, Journey, STYLES, find_destination, resolve_destination
from sim.mcp_client import MCPError


class FakeClient:
    """An MCP client double that answers vehicle-VM calls with the right tag.

    The real VM answers late and unlabelled, and `VehicleVM` exists to pair a
    reply with its request. A double that answered untagged would make every
    vehicle read time out, which is a different bug from the one under test.
    """

    def __init__(self, answers=None):
        self.calls = []
        self.answers = answers or {}

    def call(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        answer = self.answers.get(name, {})
        if callable(answer):
            answer = answer(arguments)
        if name == "run_lua_vehicle":
            return self._tagged(arguments or {}, answer)
        return answer

    @staticmethod
    def _tagged(arguments, value):
        code = arguments.get("code", "")
        if "tag = '" not in code:
            return value
        tag = code.split("tag = '")[1].split("'")[0]
        return {"73126": json.dumps({"tag": tag, "ok": True, "value": value})}

    def tools(self, name):
        return [args for tool, args in self.calls if tool == name]

    def lua(self):
        """The Lua asked for, one string per vehicle-VM call."""
        return [args["code"] for tool, args in self.calls
                if tool == "run_lua_vehicle"]

    def commanded(self):
        """Just the Lua we wrote, with the transport's drains filtered out."""
        return [code for code in self.lua() if "return 0" not in code]


def an_ai(client=None):
    return Ai(VehicleVM(client or FakeClient(), wait_s=0.0))


# -- styles -----------------------------------------------------------------


def test_the_three_styles_exist():
    assert set(STYLES) == {"aggressive", "economical", "stops"}


def test_aggressive_drives_harder_than_economical():
    assert STYLES["aggressive"].aggression > STYLES["economical"].aggression


def test_only_the_stops_style_stops():
    assert STYLES["stops"].stop_every_s is not None
    assert STYLES["aggressive"].stop_every_s is None
    assert STYLES["economical"].stop_every_s is None


def test_stopping_assumes_more_accessory_load_than_driving():
    # Idling with the blower and lights on is the recharge-deficit pathway.
    assert STYLES["stops"].accessory_load_a > STYLES["economical"].accessory_load_a


def test_economical_obeys_speed_limits_and_aggressive_does_not():
    assert STYLES["economical"].speed_mode == "limit"
    assert STYLES["aggressive"].speed_mode == "off"


def test_every_aggression_is_inside_the_range_a_scenario_accepts():
    from collect.scenario import AGGRESSION_RANGE

    low, high = AGGRESSION_RANGE
    for style in STYLES.values():
        assert low <= style.aggression <= high, style.name


# -- the AI, commanded inside the vehicle VM --------------------------------


def test_the_ai_is_commanded_in_the_vehicle_vm_not_through_set_ai():
    # Measured, not preferred: drive_to reported an accepted route to a real
    # navgraph node and the car sat still; ai.setMode('span') in the vehicle's
    # own Lua covered 32 m in six seconds.
    client = FakeClient()
    an_ai(client).roam()
    assert client.tools("set_ai") == []
    assert client.tools("drive_to") == []
    assert any("ai.setMode('span')" in code for code in client.commanded())


def test_nothing_injects_an_input():
    # The AI takes the pedals itself and releases the parking brake on its way.
    client = FakeClient()
    ai = an_ai(client)
    ai.configure(STYLES["aggressive"])
    ai.roam()
    ai.halt()
    ai.release()
    assert client.tools("inject_input") == []


def test_configure_sets_the_style_the_ai_will_drive_with():
    client = FakeClient()
    an_ai(client).configure(STYLES["aggressive"])
    commanded = " ".join(client.commanded())
    assert "ai.setAggression(1.0)" in commanded
    assert "ai.setSpeedMode('off')" in commanded
    assert "ai.driveInLane('on')" in commanded


def test_driving_to_a_waypoint_sets_the_target_before_the_mode():
    # Manual mode with no target is a car told to drive somewhere and not told
    # where.
    client = FakeClient()
    an_ai(client).drive_to("DR835_8")
    commanded = client.commanded()
    target = next(i for i, c in enumerate(commanded) if "setTarget" in c)
    mode = next(i for i, c in enumerate(commanded) if "setMode('manual')" in c)
    assert target < mode
    assert "ai.setTarget('DR835_8')" in commanded[target]


def test_halting_leaves_the_engine_running():
    # Idling is the pathway being sampled. Switching off would sample nothing.
    client = FakeClient()
    an_ai(client).halt()
    commanded = " ".join(client.commanded())
    assert "ai.setMode('stop')" in commanded
    assert "ignition" not in commanded


def test_halting_falls_back_when_stop_mode_is_refused():
    class NoStopMode(FakeClient):
        def call(self, name, arguments=None):
            code = (arguments or {}).get("code", "")
            if "setMode('stop')" in code:
                tag = code.split("tag = '")[1].split("'")[0]
                self.calls.append((name, arguments or {}))
                return {"73126": json.dumps(
                    {"tag": tag, "ok": False, "error": "unknown mode"})}
            return super().call(name, arguments)

    client = NoStopMode()
    an_ai(client).halt()
    assert any("setMode('disabled')" in code for code in client.commanded())


def test_releasing_hands_the_car_back():
    client = FakeClient()
    an_ai(client).release()
    assert any("ai.setMode('disabled')" in code for code in client.commanded())


def test_the_ai_can_be_asked_whether_it_is_actually_driving():
    client = FakeClient({"run_lua_vehicle": {"driving": True}})
    assert an_ai(client).is_driving() is True


def test_a_vm_that_will_not_answer_leaves_driving_unknown_rather_than_false():
    class Silent(FakeClient):
        def call(self, name, arguments=None):
            self.calls.append((name, arguments or {}))
            return "queued in vehicle VM(s); call again in a moment"

    ai = Ai(VehicleVM(Silent(), attempts=2, wait_s=0.0))
    assert ai.is_driving() is None


# -- finding the route ------------------------------------------------------


def route_reply(how="groundMarkers.getTargetPos", x=100.0, y=200.0, z=5.0):
    return json.dumps({"how": how, "pos": {"x": x, "y": y, "z": z}})


def test_it_reads_the_destination_the_map_route_ends_at():
    client = FakeClient({"run_lua": route_reply()})
    destination, how = find_destination(client)
    assert destination == {"x": 100.0, "y": 200.0, "z": 5.0}
    assert how["how"] == "groundMarkers.getTargetPos"


def test_an_already_parsed_reply_is_accepted():
    client = FakeClient({"run_lua": json.loads(route_reply())})
    assert find_destination(client)[0]["x"] == 100.0


def test_no_route_reports_what_the_module_does_have():
    # So the next attempt is informed rather than another guess.
    client = FakeClient({"run_lua": json.dumps(
        {"how": "none", "keys": {"setPath": "function"}})})
    destination, how = find_destination(client)
    assert destination is None
    assert how["keys"] == {"setPath": "function"}


def test_an_unreadable_reply_is_reported_not_crashed():
    client = FakeClient({"run_lua": "core_groundMarkers is nil"})
    destination, how = find_destination(client)
    assert destination is None and how["how"] == "unparseable"


def test_an_explicit_target_skips_the_map_entirely():
    class Args:
        to = "10,20,30"

    destination, how = resolve_destination(FakeClient(), Args())
    assert destination == {"x": 10.0, "y": 20.0, "z": 30.0}
    assert how["how"] == "--to"


def test_an_explicit_target_may_omit_z():
    class Args:
        to = "10,20"

    assert resolve_destination(FakeClient(), Args())[0]["z"] == 0.0


def test_a_malformed_explicit_target_is_refused():
    class Args:
        to = "over there"

    destination, how = resolve_destination(FakeClient(), Args())
    assert destination is None and how["how"] == "bad --to"


# -- the waypoint the AI can actually be sent to ----------------------------


def test_the_waypoint_comes_from_the_navgraph_not_from_us():
    # setTarget takes a name, and the game holds a nonexistent one without
    # complaining.
    client = FakeClient({"get_navgraph": {
        "closestRoad": {"from": "DR835_8", "to": "DR835_81", "dist": 0.77}}})
    assert drive_route.waypoint_near(client, {"x": 1.0, "y": 2.0}) == "DR835_8"


def test_a_destination_with_no_navgraph_near_it_has_no_waypoint():
    client = FakeClient({"get_navgraph": {"nodeCount": 0}})
    assert drive_route.waypoint_near(client, {"x": 1.0, "y": 2.0}) is None


def test_the_navgraph_is_searched_around_the_destination():
    client = FakeClient({"get_navgraph": {"closestRoad": {"from": "A"}}})
    drive_route.waypoint_near(client, {"x": 5.0, "y": 6.0})
    assert client.tools("get_navgraph")[0]["near"] == {"x": 5.0, "y": 6.0}


# -- the journey ------------------------------------------------------------


class Source:
    interval_s = 0.01

    def __init__(self, x=0.0, y=0.0):
        self.last_sample = {"x_m": x, "y_m": y, "speed_mps": 0.0}

    def at(self, x, y):
        self.last_sample = {"x_m": x, "y_m": y, "speed_mps": 20.0}


#: `None` is a meaningful destination -- it means roaming -- so the default
#: needs a value that is not None.
SOMEWHERE = {"x": 1000.0, "y": 0.0}


def a_journey(style="economical", destination=SOMEWHERE, source=None,
              client=None, waypoint="DR835_8"):
    return Journey(an_ai(client), source or Source(), destination,
                   STYLES[style], seed=0, waypoint=waypoint)


def test_it_is_not_arrived_at_the_start():
    journey = a_journey()
    journey.tick(0, 0.0, 60.0)
    assert journey.arrived is False


def test_it_arrives_when_the_car_reaches_the_destination():
    source = Source()
    journey = a_journey(source=source)
    source.at(1000.0, 0.0)
    journey.tick(0, 10.0, 600.0)
    assert journey.arrived is True


def test_arrival_has_a_radius_because_the_ai_stops_at_its_target():
    source = Source()
    journey = a_journey(source=source)
    source.at(1000.0 - drive_route.ARRIVED_M / 2, 0.0)
    journey.tick(0, 10.0, 600.0)
    assert journey.arrived is True


def test_a_car_still_far_away_has_not_arrived():
    source = Source()
    journey = a_journey(source=source)
    source.at(500.0, 0.0)
    journey.tick(0, 10.0, 600.0)
    assert journey.arrived is False


def test_roaming_never_arrives_because_there_is_nowhere_to_arrive():
    journey = a_journey(destination=None, waypoint=None)
    journey.source.at(1000.0, 0.0)
    journey.tick(0, 10.0, 600.0)
    assert journey.arrived is False


def test_sending_off_with_a_waypoint_drives_to_it():
    client = FakeClient()
    a_journey(client=client).send_off()
    assert any("ai.setTarget('DR835_8')" in code for code in client.commanded())


def test_sending_off_without_a_waypoint_roams():
    client = FakeClient()
    a_journey(client=client, waypoint=None, destination=None).send_off()
    assert any("ai.setMode('span')" in code for code in client.commanded())


def test_the_style_is_applied_before_the_car_is_started():
    client = FakeClient()
    a_journey(style="aggressive", client=client).send_off()
    commanded = client.commanded()
    aggression = next(i for i, c in enumerate(commanded) if "setAggression" in c)
    started = next(i for i, c in enumerate(commanded) if "setMode('manual')" in c)
    assert aggression < started


def test_a_style_without_stops_never_stops():
    journey = a_journey(style="aggressive")
    for second in range(600):
        journey.tick(0, float(second), 600.0)
    assert journey.stops == 0


def test_the_stops_style_does_stop():
    journey = a_journey(style="stops")
    for second in range(600):
        journey.tick(0, float(second), 600.0)
    assert journey.stops > 1


def test_a_stop_halts_the_ai():
    client = FakeClient()
    journey = a_journey(style="stops", client=client)
    for second in range(300):
        journey.tick(0, float(second), 600.0)
        if journey.stops:
            break
    assert any("setMode('stop')" in code for code in client.commanded())


def test_a_stop_ends_and_the_car_is_sent_off_again():
    client = FakeClient()
    journey = a_journey(style="stops", client=client)
    for second in range(600):
        journey.tick(0, float(second), 600.0)
    restarts = sum("setMode('manual')" in code for code in client.commanded())
    assert restarts >= journey.stops - 1


def test_the_same_seed_gives_the_same_stops():
    def when():
        journey = a_journey(style="stops")
        moments = []
        for second in range(600):
            before = journey.stops
            journey.tick(0, float(second), 600.0)
            if journey.stops != before:
                moments.append(second)
        return moments

    assert when() == when()


def test_a_different_seed_gives_different_stops():
    def when(seed):
        journey = Journey(an_ai(), Source(), {"x": 1000.0, "y": 0.0},
                          STYLES["stops"], seed=seed, waypoint="DR835_8")
        moments = []
        for second in range(600):
            before = journey.stops
            journey.tick(0, float(second), 600.0)
            if journey.stops != before:
                moments.append(second)
        return moments

    assert when(0) != when(99)


def test_distance_is_unknown_until_a_sample_has_arrived():
    class Empty:
        last_sample = None

    assert a_journey(source=Empty()).remaining_m() is None


@pytest.mark.parametrize("style", sorted(STYLES))
def test_no_style_reaches_arrival_without_moving(style):
    journey = a_journey(style=style)
    for second in range(120):
        journey.tick(0, float(second), 600.0)
    assert journey.arrived is False


# -- waiting out the async notice -------------------------------------------


def test_ask_waits_the_notice_out():
    replies = ["ai state requested (async); call again in a moment",
               {"mode": "manual", "aggression": 1.0}]
    client = FakeClient({"get_ai": lambda _a: replies.pop(0)})
    assert drive_route.ask(client, "get_ai", wait_s=0.0)["mode"] == "manual"


def test_ask_gives_up_rather_than_waiting_forever():
    client = FakeClient({"get_ai": "requested (async); call again in a moment"})
    assert drive_route.ask(client, "get_ai", attempts=3, wait_s=0.0) is None
    assert len(client.tools("get_ai")) == 3


def test_an_mcp_failure_is_reported_rather_than_retried_to_no_end():
    class Broken(FakeClient):
        def call(self, name, arguments=None):
            self.calls.append((name, arguments or {}))
            raise MCPError("the server went away")

    assert "error" in drive_route.ask(Broken(), "get_ai", wait_s=0.0)
