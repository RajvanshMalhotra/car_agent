import pytest

from collect.scenario import ScenarioSpec


def a_spec(**overrides):
    fields = dict(name="Hot commute", aggression=0.6, minutes=15.0,
                  accessory_load_a=35.0, ambient_temp_c=25.0,
                  vehicle_config="etki/3000ix_A", route="span", seed=0)
    fields.update(overrides)
    return ScenarioSpec(**fields)


def test_a_valid_spec_constructs():
    assert a_spec().name == "Hot commute"


@pytest.mark.parametrize("field,value", [
    ("aggression", 0.0),            # below BeamNG's usable floor
    ("aggression", 2.0),            # above its ceiling
    ("minutes", 0.0),
    ("accessory_load_a", -1.0),
    ("accessory_load_a", 500.0),    # that is a crank, not an accessory
    ("ambient_temp_c", -60.0),
    ("ambient_temp_c", 90.0),
])
def test_out_of_range_values_are_refused(field, value):
    with pytest.raises(ValueError, match=field):
        a_spec(**{field: value})


def test_an_empty_name_is_refused():
    with pytest.raises(ValueError, match="name"):
        a_spec(name="  ")


def test_the_hash_is_sixteen_hex_characters():
    digest = a_spec().scenario_hash
    assert len(digest) == 16
    assert all(c in "0123456789abcdef" for c in digest)


def test_the_hash_is_stable_across_instances():
    assert a_spec().scenario_hash == a_spec().scenario_hash


def test_the_hash_changes_when_a_parameter_changes():
    assert a_spec().scenario_hash != a_spec(aggression=0.9).scenario_hash


def test_the_hash_ignores_the_name():
    # The name labels a run for humans. Two identically parameterised runs are
    # the same experiment whatever they are called.
    assert a_spec().scenario_hash == a_spec(name="Something else").scenario_hash


def test_a_spec_round_trips_through_a_dict():
    spec = a_spec()
    assert ScenarioSpec.from_dict(spec.to_dict()) == spec


def test_no_language_model_is_involved():
    import collect.scenario as module

    with open(module.__file__) as handle:
        source = handle.read().lower()
    for word in ("openai", "deepseek", "completion("):
        assert word not in source
