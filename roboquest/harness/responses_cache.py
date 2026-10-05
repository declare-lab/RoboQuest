"""Explicit prompt-cache breakpoints for stateless OpenAI Responses requests."""

from copy import deepcopy
import json
import re

import httpx

BREAKPOINT = {"mode": "explicit"}


class ResponsesCacheTransport(httpx.BaseTransport):
    """Mark the newest image-free observations so each request reads the prefix the previous one wrote.

    GPT-5.6 and later cache at breakpoints: a read requires the rendered prefix
    to match up to a breakpoint carried by the current request, and implicit
    mode only writes one at the end of the prompt, which the next turn never
    matches. The policy strips images from all but the newest observations in
    place, so an image-free observation never changes again; marking the last
    text block of the newest ``breakpoints`` such observations lets request n+1
    read everything request n wrote (writes cost more than plain input, reads
    a tenth). Nothing is marked before an image-free observation exists. Wrap
    ``RetryingGenerationTransport`` (or the routing transport) with this
    transport, so captures show the fields actually transmitted. This wrapper
    owns no network connection, authentication, or retry loop.
    """

    def __init__(self, inner, *, model, breakpoints=2, ttl="30m"):
        if not isinstance(inner, httpx.BaseTransport):
            raise ValueError("Responses cache transport requires an inner transport")
        if not isinstance(model, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@/:-]{0,199}", model) is None:
            raise ValueError("Invalid expected Responses model")
        if type(breakpoints) is not int or not 1 <= breakpoints <= 4:
            raise ValueError("Between one and four cache breakpoints per request")
        if ttl not in ("30m",):
            raise ValueError("Unsupported prompt cache ttl")
        self.inner = inner
        self._model = model
        self._breakpoints = breakpoints
        self._ttl = ttl

    def handle_request(self, request):
        if request.method != "POST" or not request.url.path.endswith("/responses"):
            raise ValueError("Cache breakpoints require a Responses POST request")
        try:
            payload = json.loads(request.read())
        except (ValueError, UnicodeError):
            raise ValueError("Responses body must be a JSON object") from None
        if not isinstance(payload, dict):
            raise ValueError("Responses body must be a JSON object")
        if payload.get("model") != self._model:
            raise ValueError("Responses model does not match the configured cache model")
        if isinstance(payload.get("input"), list) and mark_breakpoints(payload["input"], self._breakpoints):
            payload["prompt_cache_options"] = {"mode": "explicit", "ttl": self._ttl}
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


def image_free_observations(items):
    """User messages with part lists and no image part: the observations whose images were stripped."""
    return [item for item in items
            if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), list)
            and item["content"] and not any(isinstance(part, dict) and part.get("type") == "input_image"
                                           for part in item["content"])]


def mark_breakpoints(items, count):
    """Put a breakpoint on the last text block of the newest ``count`` image-free observations; return how many."""
    marked = 0
    for item in image_free_observations(items)[-count:]:
        texts = [part for part in item["content"] if isinstance(part, dict) and part.get("type") == "input_text"]
        if texts:
            texts[-1]["prompt_cache_breakpoint"] = deepcopy(BREAKPOINT)
            marked += 1
    return marked
