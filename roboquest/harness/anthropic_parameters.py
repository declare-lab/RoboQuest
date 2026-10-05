"""Explicit native thinking settings applied before generation capture/retries."""

from copy import deepcopy
import json
import re

import httpx


class AnthropicParametersTransport(httpx.BaseTransport):
    """Override only the configured top-level Messages thinking parameters.

    Wrap ``RetryingGenerationTransport`` with this transport, so each counted
    attempt and its wire capture contain the settings actually transmitted.
    Native message history, including thinking signatures, passes unchanged.
    This wrapper owns no network connection, authentication, or retry loop.
    """

    def __init__(self, inner, *, model, thinking, output_config):
        if not isinstance(inner, httpx.BaseTransport):
            raise ValueError("Anthropic parameter transport requires an inner transport")
        if not isinstance(model, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@-]{0,199}", model) is None:
            raise ValueError("Invalid expected Anthropic model")
        if not isinstance(thinking, dict):
            raise ValueError("Thinking settings must be an explicit object")
        kind = thinking.get("type")
        if kind == "enabled":
            if (set(thinking) != {"type", "budget_tokens"}
                    or type(thinking["budget_tokens"]) is not int
                    or thinking["budget_tokens"] < 1024):
                raise ValueError("Enabled thinking requires an integer budget of at least 1024 tokens")
        elif kind not in ("adaptive", "disabled") or set(thinking) != {"type"}:
            raise ValueError("Unsupported thinking settings")
        if output_config is not None and (
                not isinstance(output_config, dict) or set(output_config) != {"effort"}
                or output_config["effort"] not in ("low", "medium", "high", "xhigh", "max")):
            raise ValueError("Output configuration must contain only a supported named effort")
        self.inner = inner
        self._model = model
        self._thinking = deepcopy(thinking)
        self._output_config = deepcopy(output_config)

    def handle_request(self, request):
        if request.method != "POST" or request.url.path not in ("/messages", "/v1/messages"):
            raise ValueError("Anthropic parameters require a Messages POST request")
        try:
            payload = json.loads(request.read())
        except (ValueError, UnicodeError):
            raise ValueError("Anthropic Messages body must be a JSON object") from None
        if not isinstance(payload, dict):
            raise ValueError("Anthropic Messages body must be a JSON object")
        if payload.get("model") != self._model:
            raise ValueError("Messages model does not match the configured parameter model")
        if self._thinking["type"] == "enabled" and (
                type(payload.get("max_tokens")) is not int
                or payload["max_tokens"] <= self._thinking["budget_tokens"]):
            raise ValueError("Manual thinking budget must be less than max_tokens")
        payload["thinking"] = deepcopy(self._thinking)
        if self._output_config is None:
            payload.pop("output_config", None)
        else:
            payload["output_config"] = deepcopy(self._output_config)
        content = json.dumps(payload, allow_nan=False).encode()
        headers = httpx.Headers(request.headers)
        # The body is materialized bytes now; regenerate its framing headers.
        headers.pop("content-length", None)
        headers.pop("transfer-encoding", None)
        outgoing = httpx.Request(request.method, request.url, headers=headers,
                                 content=content, extensions=dict(request.extensions))
        return self.inner.handle_request(outgoing)

    def close(self):
        self.inner.close()
