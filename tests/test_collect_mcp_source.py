import json

import pytest

from collect.lua import STATE_GLOBAL
from collect.mcp_source import MCPTrajectorySource
from collect.schema import MEASURED

INSTALLED = json.dumps({"ok": True, "mass": 1510.62, "width": len(MEASURED)})


class FakeClient:
    """Stands in for MCPClient, and remembers what it was asked."""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = list(replies or [])

    def call(self, name, arguments=None):
        self.calls.append((name, (arguments or {}).get("code", "")))
        if self.replies:
            return self.replies.pop(0)
        return "queued in vehicle VM(s); call again in a moment for results"


def a_payload(seq=1, t=0.01, **overrides):
    record = []
    for index, channel in enumerate(MEASURED):
        if channel.name == "seq":
            record.append(seq)
        elif channel.name == "t_s":
            record.append(t)
        elif channel.name in ("dir_z", "ax_mps2"):
            record.append(0.0)
        else:
            record.append(float(index))
    payload = {"samples": [record], "dropped": 0, "mass": 1510.62, "t": t}
    payload.update(overrides)
    return json.dumps(payload)


def a_source(client, **kwargs):
    kwargs.setdefault("wait_s", 0.0)
    return MCPTrajectorySource(client, **kwargs)


def test_start_installs_the_sampler():
    client = FakeClient(replies=[None, INSTALLED])
    a_source(client).start()
    tool, code = client.calls[0]
    assert tool == "run_lua_vehicle"
    assert STATE_GLOBAL in code and "enablePhysicsStepHook" in code


def test_start_passes_the_interval_into_the_lua():
    client = FakeClient(replies=[None, INSTALLED])
    a_source(client, interval_s=0.02).start()
    assert "0.02" in client.calls[0][1]


def test_start_learns_the_mass_from_the_install_self_test():
    client = FakeClient(replies=[None, INSTALLED])
    source = a_source(client)
    source.start()
    assert source.mass_kg == 1510.62


def test_start_raises_when_the_sampler_cannot_read_the_vehicle():
    # A field name that does not exist in this build must fail here, not as an
    # empty CSV fifteen minutes later.
    failed = json.dumps({"ok": False, "error": "attempt to index a nil value"})
    client = FakeClient(replies=[None, failed])
    with pytest.raises(RuntimeError, match="could not read"):
        a_source(client).start()


def test_drain_returns_decoded_samples():
    client = FakeClient(replies=[None, INSTALLED, a_payload()])
    source = a_source(client)
    source.start()
    samples = source.drain()
    assert len(samples) == 1 and samples[0]["seq"] == 1


def test_drain_adds_the_derived_channels():
    client = FakeClient(replies=[None, INSTALLED, a_payload()])
    source = a_source(client)
    source.start()
    assert "grade_rad" in source.drain()[0]


def test_drain_keeps_retrying_while_the_vm_says_queued():
    client = FakeClient(replies=[None, INSTALLED, "queued in vehicle VM(s)",
                                 "queued in vehicle VM(s)", a_payload()])
    source = a_source(client)
    source.start()
    assert len(source.drain()) == 1


def test_drain_gives_up_on_its_budget_rather_than_hanging():
    client = FakeClient(replies=[None, INSTALLED])   # then always queued
    source = a_source(client, attempts=3)
    source.start()
    before = len(client.calls)
    assert source.drain() == []
    assert len(client.calls) - before == 3


def test_dropped_samples_accumulate_across_drains():
    payload = a_payload(samples=[], dropped=5)
    client = FakeClient(replies=[None, INSTALLED, payload, payload])
    source = a_source(client)
    source.start()
    source.drain()
    source.drain()
    assert source.dropped == 10


def test_a_capture_error_is_kept_and_not_overwritten():
    first = a_payload(err="first failure")
    second = a_payload(err="a later one")
    client = FakeClient(replies=[None, INSTALLED, first, second])
    source = a_source(client)
    source.start()
    source.drain()
    source.drain()
    assert source.capture_error == "first failure"


def test_a_missing_sampler_raises_rather_than_returning_nothing():
    gone = json.dumps({"error": "not_installed"})
    client = FakeClient(replies=[None, INSTALLED, gone])
    source = a_source(client)
    source.start()
    with pytest.raises(RuntimeError, match="not installed"):
        source.drain()


def test_stop_uninstalls_the_sampler():
    client = FakeClient(replies=[None, INSTALLED])
    source = a_source(client)
    source.start()
    source.stop()
    assert "enablePhysicsStepHook(false)" in client.calls[-1][1]


def test_stop_twice_only_uninstalls_once():
    client = FakeClient(replies=[None, INSTALLED])
    source = a_source(client)
    source.start()
    source.stop()
    before = len(client.calls)
    source.stop()
    assert len(client.calls) == before
