"""DeepSeek client for behaviour generation.

DeepSeek is OpenAI-compatible, so this goes through the `openai` SDK pointed at
DeepSeek's base URL. Its JSON mode guarantees *parseable* JSON and nothing
more -- no schema enforcement -- so the schema is stated in the prompt and the
real gate stays where it belongs, in `BehaviourSpec.__post_init__`.
"""

from __future__ import annotations

import json
import os
from typing import Any

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
API_KEY_ENV = "DEEPSEEK_API_KEY"


class DeepSeekError(RuntimeError):
    """The provider was unreachable, or answered with something unusable."""


class DeepSeekClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        client: Any = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
    ) -> None:
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        if not self.api_key:
            raise DeepSeekError(
                f"no API key: pass api_key= or set {API_KEY_ENV} in the environment"
            )
        self.model = model
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = client

    def __repr__(self) -> str:
        # Never let the key reach a log line or a traceback.
        return f"DeepSeekClient(model={self.model!r}, base_url={self.base_url!r})"

    @property
    def client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        instruction = (
            "Reply with a single json object and nothing else -- no prose, no "
            "markdown fence. It must match this json schema exactly:\n\n"
            f"{json.dumps(schema, indent=2, sort_keys=True)}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": f"{system}\n\n{instruction}"},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise DeepSeekError("the model returned an empty response")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as error:
            raise DeepSeekError(
                f"the response was not valid JSON ({error}): {content[:200]!r}"
            ) from error
        if not isinstance(parsed, dict):
            raise DeepSeekError(
                f"expected a JSON object, got {type(parsed).__name__}: {content[:200]!r}"
            )
        return parsed
