"""LLM route generation: same offline, validated, cached discipline."""

import pytest

from behaviour.generator import BehaviourGenerationError, RouteGenerator
from behaviour.route_spec import RouteSpec

GOOD = {
    "name": "delivery round",
    "segments": [
        {"type": "straight", "length_m": 120.0},
        {"type": "stop", "duration_s": 45.0},
        {"type": "turn", "radius_m": 12.0, "angle_deg": 90.0},
        {"type": "straight", "length_m": 80.0},
        {"type": "stop", "duration_s": 30.0},
        {"type": "turn", "radius_m": 15.0, "angle_deg": -120.0},
        {"type": "straight", "length_m": 200.0},
    ],
}

BAD = {
    "name": "impossible",
    "segments": [
        {"type": "turn", "radius_m": 0.2, "angle_deg": 90.0},
        {"type": "straight", "length_m": 99000.0},
    ],
}


class RecordingClient:
    model = "deepseek-v4-pro"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete_json(self, system, user, schema):
        self.requests.append({"system": system, "user": user})
        return self.responses.pop(0)


class ExplodingClient:
    model = "deepseek-v4-pro"

    def complete_json(self, *args, **kwargs):
        raise AssertionError("the LLM must not be called")


def test_a_description_becomes_a_validated_route(tmp_path):
    route = RouteGenerator(RecordingClient(GOOD), cache_dir=tmp_path).generate(
        "a delivery round with frequent stops"
    )
    assert isinstance(route, RouteSpec)
    assert len(route.stops()) == 2


def test_the_generated_route_builds_a_path(tmp_path):
    route = RouteGenerator(RecordingClient(GOOD), cache_dir=tmp_path).generate("x")
    assert route.to_path().length_m > 300.0


def test_the_description_reaches_the_model(tmp_path):
    client = RecordingClient(GOOD)
    RouteGenerator(client, cache_dir=tmp_path).generate("a tight delivery round")
    assert "a tight delivery round" in client.requests[0]["user"]


def test_the_prompt_states_the_segment_bounds(tmp_path):
    client = RecordingClient(GOOD)
    RouteGenerator(client, cache_dir=tmp_path).generate("x")
    assert "2000" in client.requests[0]["system"]
    assert "stop" in client.requests[0]["system"]


def test_a_repeated_description_is_served_from_cache(tmp_path):
    client = RecordingClient(GOOD)
    generator = RouteGenerator(client, cache_dir=tmp_path)
    generator.generate("a round")
    generator.generate("a round")
    assert len(client.requests) == 1


def test_the_cache_works_without_a_client(tmp_path):
    RouteGenerator(RecordingClient(GOOD), cache_dir=tmp_path).generate("x")
    offline = RouteGenerator(ExplodingClient(), cache_dir=tmp_path)
    assert offline.generate("x").name == "delivery round"


def test_an_invalid_route_is_sent_back_for_correction(tmp_path):
    client = RecordingClient(BAD, GOOD)
    route = RouteGenerator(client, cache_dir=tmp_path).generate("x")
    assert route.name == "delivery round"
    assert len(client.requests) == 2


def test_the_correction_names_the_offending_segment(tmp_path):
    client = RecordingClient(BAD, GOOD)
    RouteGenerator(client, cache_dir=tmp_path).generate("x")
    assert "segment 0" in client.requests[1]["user"]


def test_a_model_that_never_complies_raises(tmp_path):
    client = RecordingClient(BAD, BAD)
    with pytest.raises(BehaviourGenerationError, match="2 attempts"):
        RouteGenerator(client, cache_dir=tmp_path, max_attempts=2).generate("x")


def test_an_invalid_route_is_never_cached(tmp_path):
    client = RecordingClient(BAD, BAD)
    with pytest.raises(BehaviourGenerationError):
        RouteGenerator(client, cache_dir=tmp_path, max_attempts=2).generate("x")
    assert list(tmp_path.glob("*.json")) == []


def test_routes_and_behaviours_do_not_collide_in_the_cache(tmp_path):
    from behaviour.generator import BehaviourGenerator

    routes = RouteGenerator(RecordingClient(GOOD), cache_dir=tmp_path)
    behaviours = BehaviourGenerator(RecordingClient(GOOD), cache_dir=tmp_path)
    assert routes.cache_key("same words") != behaviours.cache_key("same words")
