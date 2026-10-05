"""Private Vertex routing beneath the public Anthropic Messages wire capture.

Authentication is supplied by the caller. This module reads no credential files
and performs no token refresh or retries; the outer generation transport owns
the attempt budget and captures the unmodified Messages request.
"""

import json
import re
from collections.abc import Callable

import httpx


_PROJECT = re.compile(r"(?:[a-z][a-z0-9-]{4,28}[a-z0-9]|[0-9]{1,30})\Z")
_REGION = re.compile(r"(?:global|[a-z]+(?:-[a-z]+)+[0-9])\Z")
_MODEL = re.compile(r"claude-[a-z0-9]+(?:[-.][a-z0-9]+)*(?:@[0-9]{8})?\Z")
_ERROR_TYPES = frozenset({
    "invalid_request_error", "authentication_error", "permission_error",
    "not_found_error", "request_too_large", "rate_limit_error", "api_error",
    "overloaded_error", "billing_error", "insufficient_quota",
    "billing_hard_limit_reached", "billing_not_active", "credit_balance_too_low",
    "invalid_api_key", "request_too_large_error",
    "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED", "NOT_FOUND",
    "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED", "INTERNAL",
    "UNAVAILABLE", "DATA_LOSS", "UNAUTHENTICATED",
})


def _validated(value, pattern, name):
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        # Never include caller-supplied routing or authentication data in errors.
        raise ValueError(f"Invalid Vertex {name}")
    return value


def _safe_error(response, payload):
    """Keep actionable status/type, omit provider messages and identity details."""
    raw = payload.get("error") if isinstance(payload, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    error_type = next((raw[key] for key in ("type", "status")
                       if isinstance(raw.get(key), str) and raw[key] in _ERROR_TYPES),
                      "vertex_http_error")
    message = f"Vertex request failed (HTTP {response.status_code}; {error_type})."
    # The public policy can recover a rejected thinking-history continuation.
    # Preserve this known classification, never its provider message verbatim.
    provider_message = raw.get("message")
    if isinstance(provider_message, str) and all(
            word in provider_message.lower() for word in ("invalid", "thinking", "signature")):
        message += " Invalid thinking signature."
    result = {"type": error_type, "message": message}
    if isinstance(raw.get("status"), str) and raw["status"] in _ERROR_TYPES:
        result["status"] = raw["status"]
    if type(raw.get("code")) is int and 0 <= raw["code"] <= 599:
        result["code"] = raw["code"]
    elif isinstance(raw.get("code"), str) and raw["code"] in _ERROR_TYPES:
        result["code"] = raw["code"]
    return {"type": "error", "error": result}


class VertexAnthropicTransport(httpx.BaseTransport):
    """Translate one nonstreaming Messages attempt to Vertex ``rawPredict``.

    Place this *inside* ``RetryingGenerationTransport``. Response and exception
    request references point to the original Messages request, so its capture
    and any public error do not expose the Vertex project URL or OAuth token.
    """

    def __init__(self, project_id: str, model: str, *, region: str = "global",
                 token_provider: Callable[[], str], transport: httpx.BaseTransport | None = None):
        self._project_id = _validated(project_id, _PROJECT, "project ID")
        self._model = _validated(model, _MODEL, "model ID")
        self._region = _validated(region, _REGION, "region")
        if not callable(token_provider):
            raise ValueError("Vertex token_provider must be callable")
        self._token_provider = token_provider
        host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
        self._url = (f"https://{host}/v1/projects/{project_id}/locations/{region}"
                     f"/publishers/anthropic/models/{model}:rawPredict")
        self._transport = transport if transport is not None else httpx.HTTPTransport(
            trust_env=False, retries=0)

    def handle_request(self, request):
        if (request.method != "POST" or request.url.path != "/v1/messages"
                or request.url.query or request.url.fragment or request.url.username
                or request.url.password):
            raise ValueError("Vertex transport accepts only POST /v1/messages without URL credentials or query")
        try:
            payload = json.loads(request.read())
        except (ValueError, UnicodeError):
            raise ValueError("Vertex Messages request must contain a JSON object") from None
        if not isinstance(payload, dict):
            raise ValueError("Vertex Messages request must contain a JSON object")
        if payload.get("model") != self._model:
            raise ValueError("Messages model does not match the configured Vertex model")
        if payload.get("stream", False) is not False:
            raise ValueError("Vertex transport supports only nonstreaming Messages requests")
        payload.pop("model")
        payload["anthropic_version"] = "vertex-2023-10-16"

        try:
            token = self._token_provider()
        except Exception as exc:
            raise httpx.ConnectError(
                f"Vertex token acquisition failed ({type(exc).__name__})", request=request) from None
        if (not isinstance(token, str) or not token
                or any(not 33 <= ord(character) <= 126 for character in token)):
            raise httpx.ConnectError("Vertex token provider returned an invalid bearer token", request=request)
        # Construct a fresh header allowlist; no API key, cookie, proxy auth or
        # obsolete Anthropic version header can reach the Google endpoint.
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                   "Accept": "application/json"}
        if "anthropic-beta" in request.headers:
            headers["anthropic-beta"] = request.headers["anthropic-beta"]
        outgoing = httpx.Request("POST", self._url, headers=headers,
                                 content=json.dumps(payload).encode(),
                                 extensions=dict(request.extensions))
        try:
            response = self._transport.handle_request(outgoing)
            content = response.read()
            response.close()
        except httpx.TransportError as exc:
            raise type(exc)(f"Vertex request failed ({type(exc).__name__})", request=request) from None
        except Exception as exc:
            raise httpx.TransportError(
                f"Vertex dispatch failed ({type(exc).__name__})", request=request) from None

        # Keep retry timing while dropping provider routing/account headers and
        # content-length/content-encoding belonging to the original body.
        reply_headers = {name: response.headers[name]
                         for name in ("content-type", "retry-after", "retry-after-ms")
                         if name in response.headers}
        try:
            parsed = json.loads(content)
        except (ValueError, UnicodeError):
            parsed = None
        if response.status_code != 200 or (isinstance(parsed, dict) and "error" in parsed):
            return httpx.Response(response.status_code, headers=reply_headers,
                                  json=_safe_error(response, parsed), request=request)
        return httpx.Response(response.status_code, headers=reply_headers,
                              content=content, request=request)

    def close(self):
        self._transport.close()


_GOOGLE_MODEL = re.compile(r"gemini-[a-z0-9]+(?:[-.][a-z0-9]+)*\Z")
_API_VERSIONS = ("v1", "v1beta1")


def _bearer_token(token_provider, request):
    """The caller's OAuth token, or a connect error that names neither the token nor the provider's message."""
    try:
        token = token_provider()
    except Exception as exc:
        raise httpx.ConnectError(
            f"Vertex token acquisition failed ({type(exc).__name__})", request=request) from None
    if (not isinstance(token, str) or not token
            or any(not 33 <= ord(character) <= 126 for character in token)):
        raise httpx.ConnectError("Vertex token provider returned an invalid bearer token", request=request)
    return token


def vertex_contents(contents):
    """The native history in the turn shape Vertex accepts, or ``None`` when it needs no change.

    The native Gemini API takes a user turn that carries the function responses and the next observation
    (text and images) in one content; Vertex rejects that content (HTTP 400, "Requests ending with a model
    turn are not supported") and accepts the responses in their own content of role ``function`` followed
    by the observation as a user content, which is the shape used here. A user content holding only
    function responses, and every other content, is left as it is.
    """
    changed, reshaped = False, []
    for content in contents if isinstance(contents, list) else []:
        parts = content.get("parts") if isinstance(content, dict) else None
        if content.get("role") == "user" and isinstance(parts, list):
            responses = [part for part in parts if isinstance(part, dict) and "functionResponse" in part]
            others = [part for part in parts if not (isinstance(part, dict) and "functionResponse" in part)]
            if responses and others:
                reshaped.append(dict(content, role="function", parts=responses))
                reshaped.append(dict(content, role="user", parts=others))
                changed = True
                continue
        reshaped.append(content)
    return reshaped if changed else None


class VertexGeminiTransport(httpx.BaseTransport):
    """Translate one nonstreaming native Gemini ``generateContent`` attempt to the Vertex publisher endpoint.

    The drop-in for ``GeminiAPITransport`` when a Google-published model runs through Vertex: it accepts
    the same sanitised request (``POST /v1/models/<model>:generateContent`` on the client's
    ``gemini.invalid`` base URL), signs it with the caller's OAuth token instead of an API key and forwards
    the body to ``publishers/google/models/<model>:generateContent`` of the project. The only change to
    the body is the turn shape of :func:`vertex_contents`; the capture upstream keeps the native form.
    Responses keep the Gemini wire format (identical on both endpoints, ``usageMetadata`` included) and
    errors are sanitised the same way as on the native route. Place it inside ``RetryingGenerationTransport``.
    """

    def __init__(self, project_id: str, model: str, *, region: str = "global",
                 token_provider: Callable[[], str], transport: httpx.BaseTransport | None = None,
                 api_version: str = "v1beta1"):
        self._project_id = _validated(project_id, _PROJECT, "project ID")
        self._model = _validated(model, _GOOGLE_MODEL, "model ID")
        self._region = _validated(region, _REGION, "region")
        if api_version not in _API_VERSIONS:
            raise ValueError("Invalid Vertex API version")
        if not callable(token_provider):
            raise ValueError("Vertex token_provider must be callable")
        self._token_provider = token_provider
        self._path = f"/v1/models/{model}:generateContent"
        host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
        self._url = (f"https://{host}/{api_version}/projects/{project_id}/locations/{region}"
                     f"/publishers/google/models/{model}:generateContent")
        self._transport = transport if transport is not None else httpx.HTTPTransport(
            trust_env=False, retries=0)

    def handle_request(self, request):
        if (request.method != "POST" or request.url.path != self._path
                or request.url.query or request.url.fragment or request.url.username or request.url.password):
            raise ValueError("Vertex Gemini transport accepts only the configured native nonstreaming endpoint")
        try:
            content = request.read()
            payload = json.loads(content)
        except (ValueError, UnicodeError):
            raise ValueError("Gemini request must contain a JSON object") from None
        if not isinstance(payload, dict) or "model" in payload or "stream" in payload:
            raise ValueError("Gemini request must be a native nonstreaming JSON object")
        contents = vertex_contents(payload.get("contents"))
        if contents is not None:
            content = json.dumps(dict(payload, contents=contents)).encode()
        token = _bearer_token(self._token_provider, request)
        # A fresh header allowlist: no API key, cookie or proxy header reaches the Google endpoint.
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json",
                   "Accept": "application/json"}
        outgoing = httpx.Request("POST", self._url, headers=headers, content=content,
                                 extensions=dict(request.extensions))
        try:
            response = self._transport.handle_request(outgoing)
            body = response.read()
            response.close()
        except httpx.TransportError as exc:
            raise type(exc)(f"Vertex request failed ({type(exc).__name__})", request=request) from None
        except Exception as exc:
            raise httpx.TransportError(
                f"Vertex dispatch failed ({type(exc).__name__})", request=request) from None
        reply_headers = {name: response.headers[name]
                         for name in ("content-type", "retry-after", "retry-after-ms")
                         if name in response.headers}
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeError):
            parsed = None
        if response.status_code != 200 or (isinstance(parsed, dict) and "error" in parsed):
            from roboquest.harness.gemini_native import _safe_error as gemini_safe_error      # same wire, same sanitiser
            return httpx.Response(response.status_code, headers=reply_headers,
                                  json=gemini_safe_error(response.status_code, parsed), request=request)
        return httpx.Response(response.status_code, headers=reply_headers, content=body, request=request)

    def close(self):
        self._transport.close()
