"""The tagged vehicle-VM transport.

Every test here encodes a wrong conclusion this project actually reached by
reading an untagged reply as the answer to the wrong question.
"""

import json

import pytest

from collect.vm import VehicleVM, VMError


def tagged(tag, value=None, ok=True, error=None):
    payload = {"tag": tag, "ok": ok}
    if ok:
        payload["value"] = value if value is not None else {}
    else:
        payload["error"] = error
    return {"73126": json.dumps(payload)}


class FakeVM:
    """A vehicle VM that answers each request one call late, like the real one."""

    def __init__(self, results=None, lag=1):
        self.calls = []
        self.pending = []
        self.results = results or {}
        self.lag = lag

    def call(self, name, arguments=None):
        code = (arguments or {}).get("code", "")
        tag = code.split("tag = '")[1].split("'")[0]
        self.calls.append((tag, code))
        self.pending.append(tag)
        if len(self.pending) <= self.lag:
            return "queued in vehicle VM(s); call again in a moment for results"
        due = self.pending.pop(0)
        return tagged(due, self.results.get(due, {"answered": due}))


def test_it_gets_its_own_answer_back():
    vm = VehicleVM(FakeVM(), wait_s=0.0)
    assert vm.call("return 1")["answered"].startswith("ca")


def test_it_does_not_take_an_earlier_requests_answer():
    # `where()` read `{ok = true}` from a preceding ai.setMode and reported
    # "could not read position" about a car doing 9 m/s.
    client = FakeVM(lag=2)
    vm = VehicleVM(client, wait_s=0.0)
    first = vm.call("return 'a'")
    second = vm.call("return 'b'")
    assert first["answered"] != second["answered"]


def test_the_answer_matches_the_request_that_asked_for_it():
    client = FakeVM(lag=3)
    vm = VehicleVM(client, wait_s=0.0)
    answers = [vm.call(f"return {i}")["answered"] for i in range(4)]
    asked = [tag for tag, _code in client.calls if not tag.startswith("drain")]
    assert answers == asked


def test_a_position_read_is_not_satisfied_by_someone_elses_ok():
    client = FakeVM(lag=2)
    client.results = {}
    vm = VehicleVM(client, wait_s=0.0)
    vm.call("ai.setMode('span')")
    position = vm.call("local p = obj:getPosition() return {x = p.x}")
    assert "answered" in position   # its own reply, not the setMode one


def test_a_failing_chunk_raises_with_what_lua_said():
    class Failing(FakeVM):
        def call(self, name, arguments=None):
            code = (arguments or {}).get("code", "")
            tag = code.split("tag = '")[1].split("'")[0]
            return tagged(tag, ok=False, error="attempt to index a nil value")

    with pytest.raises(VMError, match="nil value"):
        VehicleVM(Failing(), wait_s=0.0).call("return obj:nope()")


def test_a_vm_that_never_answers_gives_up_rather_than_hanging():
    class Silent:
        def call(self, name, arguments=None):
            return "queued in vehicle VM(s); call again in a moment"

    with pytest.raises(VMError, match="did not answer"):
        VehicleVM(Silent(), attempts=3, wait_s=0.0).call("return 1")


def test_giving_up_costs_a_bounded_number_of_calls():
    class Silent:
        def __init__(self):
            self.calls = 0

        def call(self, name, arguments=None):
            self.calls += 1
            return "queued in vehicle VM(s)"

    client = Silent()
    with pytest.raises(VMError):
        VehicleVM(client, attempts=5, wait_s=0.0).call("return 1")
    assert client.calls == 6   # the request, then five drains


def test_try_call_hands_back_a_default_instead_of_raising():
    class Silent:
        def call(self, name, arguments=None):
            return "queued in vehicle VM(s)"

    vm = VehicleVM(Silent(), attempts=2, wait_s=0.0)
    assert vm.try_call("return 1", default={"unknown": True}) == {"unknown": True}


def test_the_chunk_is_wrapped_so_a_lua_error_does_not_kill_the_session():
    client = FakeVM()
    VehicleVM(client, wait_s=0.0).call("return 1")
    assert "pcall" in client.calls[0][1]


def test_untagged_replies_are_ignored_rather_than_misread():
    class Chatty(FakeVM):
        def call(self, name, arguments=None):
            code = (arguments or {}).get("code", "")
            tag = code.split("tag = '")[1].split("'")[0]
            self.pending.append(tag)
            if len(self.pending) == 1:
                # Something else's reply, with no tag at all.
                return {"73126": json.dumps({"ok": True, "mass": 1510.62})}
            return tagged(self.pending.pop(0), {"answered": self.pending and 1})

    assert VehicleVM(Chatty(), wait_s=0.0).call("return 1") is not None
