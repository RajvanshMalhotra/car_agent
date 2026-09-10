"""Tests for the route driver.

Everything here runs against a fake MCP client. The game is on someone else's
Windows laptop, and a driver that can only be tested there is a driver that
stops being tested.
"""

import json

import pytest

import drive_route
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
