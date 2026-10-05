"""Explicit OpenRouter provider routing applied before generation capture/retries."""

from copy import deepcopy
import json
import re

import httpx


class OpenRouterRoutingTransport(httpx.BaseTransport):
    """Add the configured OpenRouter ``provider`` routing block to each Responses request.

    Wrap ``RetryingGenerationTransport`` with this transport, so each counted
    attempt and its wire capture contain the routing actually transmitted. The
    history, tools and native reasoning replay pass unchanged. This wrapper owns
    no network connection, authentication, or retry loop.
    """

    def __init__(self, inner, *, model, routing):
        if not isinstance(inner, httpx.BaseTransport):
            raise ValueError("OpenRouter routing transport requires an inner transport")
        if not isinstance(model, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@/:-]{0,199}", model) is None:
            raise ValueError("Invalid expected OpenRouter model")
        if (not isinstance(routing, dict) or "only" not in routing
                or not set(routing) <= {"only", "allow_fallbacks"}):
            raise ValueError("Routing must name the allowed providers and at most allow_fallbacks")
        only = routing["only"]
        if (not isinstance(only, list) or not only or any(
                not isinstance(tag, str) or re.fullmatch(r"[a-z0-9][a-z0-9._/-]{0,63}", tag) is None
                for tag in only)):
            raise ValueError("Allowed providers must be a non-empty list of provider tags")
        if "allow_fallbacks" in routing and type(routing["allow_fallbacks"]) is not bool:
            raise ValueError("allow_fallbacks must be a boolean")
        self.inner = inner
        self._model = model
        self._routing = deepcopy(routing)

    def handle_request(self, request):
        if request.method != "POST" or not request.url.path.endswith("/responses"):
            raise ValueError("OpenRouter routing requires a Responses POST request")
        try:
            payload = json.loads(request.read())
        except (ValueError, UnicodeError):
            raise ValueError("Responses body must be a JSON object") from None
        if not isinstance(payload, dict):
            raise ValueError("Responses body must be a JSON object")
        if payload.get("model") != self._model:
            raise ValueError("Responses model does not match the configured routing model")
        payload["provider"] = deepcopy(self._routing)
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
