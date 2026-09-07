"""Runs must be reproducible from (spec_hash, scenario, seed), so generated
specs are cached to disk and never regenerated for the same request."""

import json

import pytest

from behaviour.cache import SpecCache


def test_a_miss_returns_nothing(tmp_path):
    assert SpecCache(tmp_path).get("some-key") is None


def test_what_was_stored_comes_back(tmp_path):
    cache = SpecCache(tmp_path)
    cache.put("k", {"name": "commuter"}, provenance={"description": "a commuter"})
    assert cache.get("k") == {"name": "commuter"}


def test_a_stored_entry_survives_a_new_cache_object(tmp_path):
    SpecCache(tmp_path).put("k", {"name": "commuter"}, provenance={})
    assert SpecCache(tmp_path).get("k") == {"name": "commuter"}


def test_keys_do_not_collide(tmp_path):
    cache = SpecCache(tmp_path)
    cache.put("a", {"name": "one"}, provenance={})
    cache.put("b", {"name": "two"}, provenance={})
    assert cache.get("a") == {"name": "one"}


def test_provenance_is_recorded_alongside_the_spec(tmp_path):
    cache = SpecCache(tmp_path)
    cache.put("k", {"name": "commuter"}, provenance={"description": "a calm commuter"})
    stored = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert stored["provenance"]["description"] == "a calm commuter"
    assert "generated_at" in stored["provenance"]


def test_the_cache_directory_is_created_on_demand(tmp_path):
    cache = SpecCache(tmp_path / "nested" / "specs")
    cache.put("k", {"name": "commuter"}, provenance={})
    assert cache.get("k") == {"name": "commuter"}


def test_a_corrupt_entry_is_reported_not_silently_ignored(tmp_path):
    cache = SpecCache(tmp_path)
    cache.put("k", {"name": "commuter"}, provenance={})
    next(tmp_path.glob("*.json")).write_text("{ not json")
    with pytest.raises(ValueError, match="corrupt"):
        cache.get("k")
