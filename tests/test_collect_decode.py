import json

import pytest

from collect.decode import DecodeError, decode, decode_install
from collect.schema import MEASURED


def a_record(seq=1, t=0.01):
    values = []
    for index, channel in enumerate(MEASURED):
        if channel.name == "seq":
            values.append(seq)
        elif channel.name == "t_s":
            values.append(t)
        else:
            values.append(float(index))
    return values


def a_payload(**overrides):
    payload = {"samples": [a_record()], "dropped": 0, "mass": 1510.62, "t": 0.01}
    payload.update(overrides)
    return payload


def test_decodes_a_json_string():
    drain = decode(json.dumps(a_payload()))
    assert len(drain.samples) == 1 and drain.mass_kg == 1510.62


def test_decodes_an_already_parsed_payload():
    assert len(decode(a_payload()).samples) == 1


def test_maps_positions_to_channel_names():
    sample = decode(a_payload(samples=[a_record(seq=7, t=0.5)])).samples[0]
    assert sample["seq"] == 7 and sample["t_s"] == 0.5
    assert set(sample) == {c.name for c in MEASURED}


def test_carries_the_drop_count_through():
    assert decode(a_payload(samples=[], dropped=12)).dropped == 12


def test_an_empty_drain_is_not_an_error():
    assert decode(a_payload(samples=[])).samples == []


def test_the_queued_notice_is_pending_not_a_failure():
    # run_lua_vehicle answers this while the real result is still coming.
    drain = decode("queued in vehicle VM(s); call again in a moment for results")
    assert drain.samples == [] and drain.pending is True


def test_a_result_keyed_by_vehicle_id_is_unwrapped():
    assert len(decode({"73126": json.dumps(a_payload())}).samples) == 1


def test_the_sampler_not_being_installed_raises():
    with pytest.raises(DecodeError, match="not installed"):
        decode(json.dumps({"error": "not_installed"}))


def test_a_record_of_the_wrong_width_raises():
    with pytest.raises(DecodeError, match="width"):
        decode(a_payload(samples=[[1.0, 2.0]]))


def test_a_lua_one_based_map_is_accepted_as_a_record():
    # jsonEncode emits "1".."N" when it will not commit to an array.
    as_map = {str(i + 1): v for i, v in enumerate(a_record())}
    assert decode(a_payload(samples=[as_map])).samples[0]["seq"] == 1


def test_a_capture_error_is_carried_out_rather_than_swallowed():
    assert decode(a_payload(err="attempt to index a nil value")).error


def test_install_reports_success_with_the_mass():
    result = decode_install(json.dumps({"ok": True, "mass": 1510.62, "width": 44}))
    assert result["mass"] == 1510.62


def test_install_raises_when_the_sampler_could_not_read_the_vehicle():
    with pytest.raises(DecodeError, match="could not read"):
        decode_install(json.dumps({"ok": False, "error": "nil value"}))


def test_install_pending_is_an_empty_result_not_a_failure():
    assert decode_install("queued in vehicle VM(s)") == {}
