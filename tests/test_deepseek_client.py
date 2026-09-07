"""DeepSeek at the network seam. OpenAI-compatible, but JSON mode has no schema
enforcement -- it only guarantees parseable JSON -- so the code-side
BehaviourSpec gate is what actually protects the physics."""

import json
import types

import pytest

from behaviour.deepseek import DeepSeekClient, DeepSeekError

SCHEMA = {
    "type": "object",
    "properties": {"idle_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0}},
    "required": ["idle_fraction"],
    "additionalProperties": False,
}


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self.content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def fake_client(content):
    completions = FakeCompletions(content)
    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))
    return client, completions


def test_the_parsed_object_is_returned():
    client, _ = fake_client('{"idle_fraction": 0.45}')
    result = DeepSeekClient(api_key="k", client=client).complete_json(
        "system", "user", SCHEMA
    )
    assert result == {"idle_fraction": 0.45}


def test_the_prompts_are_sent_through():
    client, calls = fake_client('{"idle_fraction": 0.1}')
    DeepSeekClient(api_key="k", client=client).complete_json(
        "be a driver", "describe a courier", SCHEMA
    )
    messages = calls.calls[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "be a driver" in messages[0]["content"]
    assert "describe a courier" in messages[1]["content"]


def test_json_mode_is_requested():
    client, calls = fake_client('{"idle_fraction": 0.1}')
    DeepSeekClient(api_key="k", client=client).complete_json("s", "u", SCHEMA)
    assert calls.calls[0]["response_format"] == {"type": "json_object"}


def test_the_word_json_appears_in_the_prompt():
    # DeepSeek rejects or misbehaves in JSON mode unless the prompt says "json".
    client, calls = fake_client('{"idle_fraction": 0.1}')
    DeepSeekClient(api_key="k", client=client).complete_json("s", "u", SCHEMA)
    prompt = " ".join(m["content"] for m in calls.calls[0]["messages"])
    assert "json" in prompt.lower()


def test_the_schema_is_shown_to_the_model():
    # JSON mode does not enforce a schema, so the schema has to go in the prompt.
    client, calls = fake_client('{"idle_fraction": 0.1}')
    DeepSeekClient(api_key="k", client=client).complete_json("s", "u", SCHEMA)
    prompt = " ".join(m["content"] for m in calls.calls[0]["messages"])
    assert "idle_fraction" in prompt


def test_the_configured_model_is_used():
    client, calls = fake_client('{"idle_fraction": 0.1}')
    DeepSeekClient(api_key="k", model="deepseek-v4-flash", client=client).complete_json(
        "s", "u", SCHEMA
    )
    assert calls.calls[0]["model"] == "deepseek-v4-flash"


def test_unparseable_output_raises_a_clear_error():
    client, _ = fake_client("here you go: {oops")
    with pytest.raises(DeepSeekError, match="not valid JSON"):
        DeepSeekClient(api_key="k", client=client).complete_json("s", "u", SCHEMA)


def test_a_json_array_is_rejected():
    client, _ = fake_client('[{"idle_fraction": 0.1}]')
    with pytest.raises(DeepSeekError, match="object"):
        DeepSeekClient(api_key="k", client=client).complete_json("s", "u", SCHEMA)


def test_an_empty_response_raises():
    client, _ = fake_client(None)
    with pytest.raises(DeepSeekError, match="empty"):
        DeepSeekClient(api_key="k", client=client).complete_json("s", "u", SCHEMA)


def test_a_missing_api_key_is_reported_before_any_request(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(DeepSeekError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient()


def test_the_key_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "from-env")
    client, _ = fake_client('{"idle_fraction": 0.1}')
    assert DeepSeekClient(client=client).api_key == "from-env"


def test_the_key_is_not_exposed_in_the_repr():
    assert "secret" not in repr(DeepSeekClient(api_key="secret-key-value"))
