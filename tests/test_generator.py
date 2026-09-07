"""The behaviour agent: plain English in, a bounds-checked BehaviourSpec out.

Generation happens once per behaviour, offline, and is cached. No API call
happens while a vehicle is moving.
"""

import pytest

from behaviour.generator import (
    BehaviourGenerationError,
    BehaviourGenerator,
    PROMPT_VERSION,
)
from behaviour.spec import BehaviourSpec

VALID_FIELDS = dict(
    name="delhi delivery driver",
    target_speed_factor=1.1,
    accel_limit_mps2=3.0,
    decel_limit_mps2=5.0,
    jerk_limit_mps3=10.0,
    following_distance_s=0.8,
    corner_speed_factor=1.05,
    reaction_lag_s=0.2,
    erraticness=0.7,
    trip_duration_s=420.0,
    idle_fraction=0.45,
    hvac_setting=1.0,
    ambient_temp_c=42.0,
    cold_start=True,
    start_stop_enabled=False,
)


class RecordingClient:
    """A stand-in for the LLM at the network seam. Returns queued responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete_json(self, system, user, schema):
        self.requests.append({"system": system, "user": user, "schema": schema})
        if not self.responses:
            raise AssertionError("client called more often than expected")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ExplodingClient:
    def complete_json(self, system, user, schema):
        raise AssertionError("the LLM must not be called")


def out_of_bounds(**overrides):
    fields = dict(VALID_FIELDS, decel_limit_mps2=8.83)  # 0.9g, past the tyre limit
    fields.update(overrides)
    return fields


def test_a_description_becomes_a_validated_spec(tmp_path):
    generator = BehaviourGenerator(RecordingClient(VALID_FIELDS), cache_dir=tmp_path)
    spec = generator.generate("a delhi delivery driver in summer")
    assert isinstance(spec, BehaviourSpec)
    assert spec.idle_fraction == 0.45


def test_the_description_reaches_the_model(tmp_path):
    client = RecordingClient(VALID_FIELDS)
    BehaviourGenerator(client, cache_dir=tmp_path).generate("a delhi delivery driver")
    assert "a delhi delivery driver" in client.requests[0]["user"]


def test_the_allowed_ranges_are_stated_in_the_prompt(tmp_path):
    client = RecordingClient(VALID_FIELDS)
    BehaviourGenerator(client, cache_dir=tmp_path).generate("anything")
    prompt = client.requests[0]["system"]
    assert "decel_limit_mps2" in prompt
    assert "8.0" in prompt  # its upper bound


def test_a_repeated_description_is_served_from_cache(tmp_path):
    client = RecordingClient(VALID_FIELDS)
    generator = BehaviourGenerator(client, cache_dir=tmp_path)
    first = generator.generate("a delhi delivery driver")
    second = generator.generate("a delhi delivery driver")
    assert first == second
    assert len(client.requests) == 1


def test_the_cache_works_across_process_lifetimes(tmp_path):
    BehaviourGenerator(RecordingClient(VALID_FIELDS), cache_dir=tmp_path).generate("x")
    # A generator with no working client at all must still answer from disk.
    offline = BehaviourGenerator(ExplodingClient(), cache_dir=tmp_path)
    assert offline.generate("x").name == "delhi delivery driver"


def test_a_different_description_is_generated_afresh(tmp_path):
    client = RecordingClient(VALID_FIELDS, dict(VALID_FIELDS, name="commuter"))
    generator = BehaviourGenerator(client, cache_dir=tmp_path)
    generator.generate("a delhi delivery driver")
    generator.generate("a calm commuter")
    assert len(client.requests) == 2


def test_changing_the_prompt_invalidates_the_cache(tmp_path):
    client = RecordingClient(VALID_FIELDS, VALID_FIELDS)
    BehaviourGenerator(client, cache_dir=tmp_path).generate("x")
    BehaviourGenerator(
        client, cache_dir=tmp_path, prompt_version=PROMPT_VERSION + 1
    ).generate("x")
    assert len(client.requests) == 2


def test_changing_the_model_invalidates_the_cache(tmp_path):
    client = RecordingClient(VALID_FIELDS, VALID_FIELDS)
    BehaviourGenerator(client, cache_dir=tmp_path, model="model-a").generate("x")
    BehaviourGenerator(client, cache_dir=tmp_path, model="model-b").generate("x")
    assert len(client.requests) == 2


def test_an_out_of_bounds_answer_is_sent_back_for_correction(tmp_path):
    client = RecordingClient(out_of_bounds(), VALID_FIELDS)
    spec = BehaviourGenerator(client, cache_dir=tmp_path).generate("aggressive")
    assert spec.decel_limit_mps2 == 5.0
    assert len(client.requests) == 2


def test_the_correction_names_the_offending_field(tmp_path):
    client = RecordingClient(out_of_bounds(), VALID_FIELDS)
    BehaviourGenerator(client, cache_dir=tmp_path).generate("aggressive")
    assert "decel_limit_mps2" in client.requests[1]["user"]
    assert "8.83" in client.requests[1]["user"]


def test_a_model_that_never_complies_raises(tmp_path):
    client = RecordingClient(out_of_bounds(), out_of_bounds(), out_of_bounds())
    with pytest.raises(BehaviourGenerationError, match="3 attempts"):
        BehaviourGenerator(client, cache_dir=tmp_path, max_attempts=3).generate("x")


def test_an_invalid_spec_is_never_cached(tmp_path):
    client = RecordingClient(out_of_bounds(), out_of_bounds())
    with pytest.raises(BehaviourGenerationError):
        BehaviourGenerator(client, cache_dir=tmp_path, max_attempts=2).generate("x")
    assert list(tmp_path.glob("*.json")) == []


def test_a_missing_field_is_sent_back_for_correction(tmp_path):
    incomplete = {k: v for k, v in VALID_FIELDS.items() if k != "hvac_setting"}
    client = RecordingClient(incomplete, VALID_FIELDS)
    spec = BehaviourGenerator(client, cache_dir=tmp_path).generate("x")
    assert spec.hvac_setting == 1.0


def test_an_unexpected_extra_field_is_sent_back_for_correction(tmp_path):
    client = RecordingClient(dict(VALID_FIELDS, top_speed_mph=95), VALID_FIELDS)
    spec = BehaviourGenerator(client, cache_dir=tmp_path).generate("x")
    assert isinstance(spec, BehaviourSpec)


def test_the_cached_spec_reconstructs_identically(tmp_path):
    client = RecordingClient(VALID_FIELDS)
    generator = BehaviourGenerator(client, cache_dir=tmp_path)
    assert generator.generate("x").spec_hash == generator.generate("x").spec_hash


def test_several_descriptions_generate_in_one_call(tmp_path):
    client = RecordingClient(VALID_FIELDS, dict(VALID_FIELDS, name="commuter"))
    specs = BehaviourGenerator(client, cache_dir=tmp_path).generate_many(
        ["a courier", "a commuter"]
    )
    assert [s.name for s in specs] == ["delhi delivery driver", "commuter"]


def test_the_model_recorded_is_the_one_the_client_will_actually_use(tmp_path):
    # The model is part of the cache key and the provenance record, so it must
    # not be a default that disagrees with the client doing the work.
    client = RecordingClient(VALID_FIELDS)
    client.model = "deepseek-v4-pro"
    generator = BehaviourGenerator(client, cache_dir=tmp_path)
    assert generator.model == "deepseek-v4-pro"


def test_an_explicit_model_overrides_the_client(tmp_path):
    client = RecordingClient(VALID_FIELDS)
    client.model = "deepseek-v4-pro"
    generator = BehaviourGenerator(client, cache_dir=tmp_path, model="pinned")
    assert generator.model == "pinned"


def test_a_client_without_a_model_attribute_still_works(tmp_path):
    generator = BehaviourGenerator(RecordingClient(VALID_FIELDS), cache_dir=tmp_path)
    assert generator.generate("x").name == "delhi delivery driver"
