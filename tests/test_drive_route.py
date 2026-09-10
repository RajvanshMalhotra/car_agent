"""Tests for the route driver.

Everything here runs against a fake MCP client. The game is on someone else's
Windows laptop, and a driver that can only be tested there is a driver that
stops being tested.
"""

import json

import pytest

import drive_route
from sim.mcp_client import MCPError
from drive_route import STYLES, Journey, Style, find_destination, resolve_destination


class FakeClient:
    def __init__(self, answers=None):
        self.calls = []
        self.answers = answers or {}

    def call(self, name, arguments=None):
        self.calls.append((name, arguments or {}))
        answer = self.answers.get(name, {})
        return answer(arguments) if callable(answer) else answer

    def tools(self, name):
        return [args for tool, args in self.calls if tool == name]


def route_reply(how="groundMarkers.targetPos", x=100.0, y=200.0, z=5.0):
    return json.dumps({"how": how, "pos": {"x": x, "y": y, "z": z}})


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


# -- finding the route ------------------------------------------------------


def test_it_reads_the_destination_the_map_route_ends_at():
    client = FakeClient({"run_lua": route_reply()})
    destination, how = find_destination(client)
    assert destination == {"x": 100.0, "y": 200.0, "z": 5.0}
    assert how["how"] == "groundMarkers.targetPos"


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


# -- sending the car off ----------------------------------------------------


def test_send_off_hands_the_destination_to_the_games_ai():
    client = FakeClient()
    drive_route.send_off(client, 7, {"x": 1.0, "y": 2.0, "z": 3.0},
                         STYLES["aggressive"])
    drive_to = client.tools("drive_to")[0]
    assert drive_to["pos"] == {"x": 1.0, "y": 2.0, "z": 3.0}
    assert drive_to["aggression"] == STYLES["aggressive"].aggression
    assert drive_to["id"] == 7


def holding(parkingbrake=1, brake=0.3):
    """A vehicle VM reply describing what is holding the car still."""
    return {"7": json.dumps({"parkingbrake": parkingbrake, "brake": brake,
                             "throttle": 0, "gear": 0})}


def test_a_free_car_is_left_alone():
    client = FakeClient({"run_lua_vehicle": holding(parkingbrake=0)})
    assert drive_route.free_the_car(client, 7) == "already free"
    assert len(client.tools("run_lua_vehicle")) == 1


def test_the_parking_brake_goes_through_the_vehicles_own_input_system():
    # `inject_input` sets an input for a moment and the vehicle reasserts
    # itself, which is why the brake survived being told to release.
    replies = [holding(1), {"7": json.dumps({"ok": True})}, holding(0)]
    client = FakeClient({"run_lua_vehicle": lambda _a: replies.pop(0)})
    assert drive_route.free_the_car(client, 7) == "input.event, FILTER_DIRECT"
    code = client.tools("run_lua_vehicle")[1]["code"]
    assert "input.event('parkingbrake', 0, FILTER_DIRECT)" in code


def test_it_moves_on_to_the_next_release_when_one_does_not_take():
    # still held after the first, free after the second
    replies = [holding(1), {"7": json.dumps({"ok": True})}, holding(1),
               {"7": json.dumps({"ok": True})}, holding(0)]
    client = FakeClient({"run_lua_vehicle": lambda _a: replies.pop(0)})
    assert drive_route.free_the_car(client, 7) == "input.event, default filter"


def test_it_reports_when_nothing_lets_the_car_go():
    client = FakeClient({"run_lua_vehicle": holding(1)})
    said = []
    assert drive_route.free_the_car(client, 7, report=said.append) is None
    assert "could not release" in said[-1]


def test_every_release_route_is_tried_before_giving_up():
    client = FakeClient({"run_lua_vehicle": holding(1)})
    drive_route.free_the_car(client, 7)
    attempted = [args["code"] for args in client.tools("run_lua_vehicle")]
    for _name, action in drive_route.RELEASES:
        assert any(action in code for code in attempted)


def test_an_async_notice_is_not_mistaken_for_an_answer():
    # Two wordings exist and checking for only one reads a notice as a reply.
    from collect.decode import is_async_notice

    assert is_async_notice("queued in vehicle VM(s); call again in a moment")
    assert is_async_notice("ai state requested (async); call again in a moment")
    assert not is_async_notice('{"parkingbrake": 0}')


def test_a_picture_is_taken_and_its_path_kept():
    client = FakeClient({"screenshot": "C:/BeamNG/screenshots/shot.jpg"})
    shots = []
    drive_route.look(client, "after span", shots)
    assert shots == [("after span", "C:/BeamNG/screenshots/shot.jpg")]


def test_a_screenshot_that_is_not_ready_yet_is_waited_for():
    replies = ["screenshot requested (async); call again in a moment",
               "C:/BeamNG/screenshots/shot.jpg"]
    client = FakeClient({"screenshot": lambda _a: replies.pop(0)})
    shots = []
    drive_route.look(client, "after span", shots)
    assert shots[0][1].endswith("shot.jpg")


def test_a_failed_screenshot_does_not_take_the_diagnosis_with_it():
    class Broken(FakeClient):
        def call(self, name, arguments=None):
            if name == "screenshot":
                raise MCPError("no screenshot on this build")
            return super().call(name, arguments)

    shots = []
    drive_route.look(Broken(), "after span", shots)
    assert shots == []


def test_the_destination_is_drawn_so_a_picture_shows_the_target():
    client = FakeClient()
    drive_route.show_the_target(client, {"x": 1.0, "y": 2.0, "z": 3.0})
    drawn = client.tools("debug_draw")[0]
    assert drawn["pos"] == {"x": 1.0, "y": 2.0, "z": 3.0}


def test_where_it_is_reports_whether_the_ground_is_drivable():
    client = FakeClient({
        "get_status": {"vehicle": {"pos": {"x": 1.0}, "damage": 0}},
        "get_ground_at_point": {"drivability": 0.0, "surfaceHeight": 100.0},
    })
    where = drive_route.where_is_it(client)
    assert where["drivability_under_car"] == 0.0
    assert where["damage"] == 0


def test_ask_waits_the_notice_out():
    replies = ["ai state requested (async); call again in a moment",
               {"mode": "manual", "aggression": 1.0}]
    client = FakeClient({"get_ai": lambda _a: replies.pop(0)})
    assert drive_route.ask(client, "get_ai", wait_s=0.0)["mode"] == "manual"


def test_ask_gives_up_rather_than_waiting_forever():
    client = FakeClient({"get_ai": "requested (async); call again in a moment"})
    assert drive_route.ask(client, "get_ai", attempts=3, wait_s=0.0) is None
    assert len(client.tools("get_ai")) == 3


def test_send_off_asks_for_everything_first():
    client = FakeClient()
    drive_route.send_off(client, 7, {"x": 0.0, "y": 0.0, "z": 0.0},
                         STYLES["aggressive"])
    assert set(client.tools("drive_to")[0]) == {
        "id", "pos", "aggression", "avoidCars", "driveInLane", "routeSpeedMode"}


def test_send_off_drops_arguments_until_the_game_takes_one():
    # A refused drive_to does not raise. It comes back as text and the car
    # never sets off.
    def fussy(arguments):
        if "routeSpeedMode" in arguments:
            return "drive_to failed: unknown argument routeSpeedMode"
        if "driveInLane" in arguments:
            return "error: bad argument driveInLane"
        return {}

    client = FakeClient({"drive_to": fussy})
    accepted = drive_route.send_off(client, 7, {"x": 0.0, "y": 0.0, "z": 0.0},
                                    STYLES["aggressive"])
    assert accepted == ("aggression", "avoidCars")
    assert len(client.tools("drive_to")) == 3


def test_send_off_raises_when_nothing_is_accepted_and_says_what_was_tried():
    client = FakeClient({"drive_to": "drive_to failed: no navgraph"})
    with pytest.raises(RuntimeError, match="would not accept"):
        drive_route.send_off(client, 7, {"x": 0.0, "y": 0.0, "z": 0.0},
                             STYLES["economical"])


def test_send_off_reuses_a_shape_that_already_worked():
    client = FakeClient()
    drive_route.send_off(client, 7, {"x": 0.0, "y": 0.0, "z": 0.0},
                         STYLES["economical"], accepted=("aggression",))
    assert set(client.tools("drive_to")[0]) == {"id", "pos", "aggression"}


@pytest.mark.parametrize("reply", [
    "drive_to failed", "error: nil value", "unknown argument",
    "invalid pos", "bad argument #2",
])
def test_a_refusal_is_recognised_however_it_is_worded(reply):
    assert drive_route.refused(reply) is True


@pytest.mark.parametrize("reply", [{}, "ok", "route set", 1, None])
def test_an_acceptance_is_not_mistaken_for_a_refusal(reply):
    assert drive_route.refused(reply) is False


def test_send_off_puts_the_ai_in_manual_so_it_follows_the_route():
    client = FakeClient()
    drive_route.send_off(client, 7, {"x": 0.0, "y": 0.0, "z": 0.0},
                         STYLES["economical"])
    assert client.tools("set_ai")[0]["mode"] == "manual"


def test_handing_back_disables_the_ai_and_releases_the_pedals():
    client = FakeClient()
    drive_route.hand_back(client, 7)
    assert client.tools("set_ai")[-1]["mode"] == "disabled"
    events = {args["event"] for args in client.tools("inject_input")}
    assert events == {"throttle", "brake", "steering"}


# -- the journey ------------------------------------------------------------


class Source:
    def __init__(self, x=0.0, y=0.0):
        self.last_sample = {"x_m": x, "y_m": y}

    def at(self, x, y):
        self.last_sample = {"x_m": x, "y_m": y}


def a_journey(style="economical", destination=None, source=None, client=None):
    return Journey(
        client or FakeClient(), 7, source or Source(),
        destination or {"x": 1000.0, "y": 0.0, "z": 0.0},
        STYLES[style], seed=0,
    )


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


def test_a_style_without_stops_never_stops():
    client = FakeClient()
    journey = a_journey(style="aggressive", client=client)
    for second in range(600):
        journey.tick(0, float(second), 600.0)
    assert journey.stops == 0


def test_the_stops_style_does_stop():
    client = FakeClient()
    journey = a_journey(style="stops", client=client)
    for second in range(600):
        journey.tick(0, float(second), 600.0)
    assert journey.stops > 1


def test_a_stop_brakes_and_disengages_the_ai():
    client = FakeClient()
    journey = a_journey(style="stops", client=client)
    for second in range(300):
        journey.tick(0, float(second), 600.0)
        if journey.stops:
            break
    assert any(args.get("event") == "brake" and args.get("value") == 1.0
               for args in client.tools("inject_input"))
    assert client.tools("set_ai")[-1]["mode"] == "disabled"


def test_a_stop_ends_and_the_car_is_sent_off_again():
    client = FakeClient()
    journey = a_journey(style="stops", client=client)
    for second in range(600):
        journey.tick(0, float(second), 600.0)
    # Every stop but possibly the last is followed by a fresh drive_to.
    assert len(client.tools("drive_to")) >= journey.stops - 1


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
        journey = Journey(FakeClient(), 7, Source(),
                          {"x": 1000.0, "y": 0.0, "z": 0.0},
                          STYLES["stops"], seed=seed)
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
