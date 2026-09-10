import pytest

from collect.schema import COLUMNS, DERIVED, MEASURED, PROVENANCE, Channel, manifest


def test_every_measured_channel_carries_a_lua_expression():
    for channel in MEASURED:
        assert channel.lua, f"{channel.name} is measured but has no lua expression"


def test_derived_channels_have_no_lua_expression():
    assert all(c.lua is None for c in DERIVED)


def test_no_channel_name_appears_twice():
    names = [c.name for c in MEASURED + DERIVED]
    assert len(names) == len(set(names))


def test_columns_are_measured_then_derived():
    assert COLUMNS == tuple(c.name for c in MEASURED + DERIVED)


def test_provenance_covers_every_column():
    assert set(PROVENANCE) == set(COLUMNS)
    assert set(PROVENANCE.values()) == {"measured", "derived"}


def test_no_battery_channel_exists():
    # BeamNG simulates no 12 V system. A column here would read as a
    # measurement of something the game never reported.
    assert not set(COLUMNS) & {"current_a", "voltage_v", "soc", "battery_temp_c"}


def test_the_channels_the_load_model_needs_are_present():
    for name in ("t_s", "speed_mps", "throttle", "brake", "rpm", "mass_kg",
                 "engine_torque_nm", "x_m", "y_m", "z_m", "ax_mps2", "dir_z",
                 "wheel_av_fl", "coolant_c", "ignition_level"):
        assert name in COLUMNS, name


def test_manifest_describes_every_column():
    entries = manifest()
    assert [e["name"] for e in entries] == list(COLUMNS)
    assert all(e["unit"] for e in entries)


def test_channel_rejects_an_unknown_provenance():
    with pytest.raises(ValueError, match="provenance"):
        Channel(name="x", unit="m", provenance="guessed", lua="p.x")


def test_a_measured_channel_without_a_source_is_refused():
    with pytest.raises(ValueError, match="lua expression"):
        Channel(name="x", unit="m", provenance="measured", lua=None)
