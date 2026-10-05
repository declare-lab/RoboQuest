"""Native Gemini API boundary for the observation-only robot policy.

The outer RetryingGenerationTransport owns retries, counting and wire capture.
The private transport adds the API key only after that boundary. Native model
content is kept verbatim in assistant transcript metadata so thought signatures
on text, tool calls, or empty parts survive every continuation.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import base64
import json
import math
import re
import time

import httpx

from inspect_robots_agent._llm import AssistantMessage, ToolCall


_MODEL = re.compile(r"gemini-[a-z0-9]+(?:[-.][a-z0-9]+)*\Z")
_PNG_PREFIX = "data:image/png;base64,"
_FINISH_REASONS = frozenset({
    "MAX_TOKENS", "SAFETY", "RECITATION", "LANGUAGE", "OTHER", "BLOCKLIST",
    "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL", "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "IMAGE_OTHER", "NO_IMAGE",
    "UNEXPECTED_TOOL_CALL", "TOO_MANY_TOOL_CALLS", "MISSING_THOUGHT_SIGNATURE",
})
_BLOCK_REASONS = frozenset({"SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY"})
_ERROR_STATUSES = frozenset({
    "CANCELLED", "UNKNOWN", "INVALID_ARGUMENT", "DEADLINE_EXCEEDED", "NOT_FOUND",
    "ALREADY_EXISTS", "PERMISSION_DENIED", "RESOURCE_EXHAUSTED",
    "FAILED_PRECONDITION", "ABORTED", "OUT_OF_RANGE", "UNIMPLEMENTED", "INTERNAL",
    "UNAVAILABLE", "DATA_LOSS", "UNAUTHENTICATED",
})


def _model_id(value):
    if not isinstance(value, str) or _MODEL.fullmatch(value) is None:
        raise ValueError("Invalid Gemini model ID")
    return value


def _safe_error(status, payload):
    error = payload.get("error", {}) if isinstance(payload, dict) else {}
    error = error if isinstance(error, dict) else {}
    kind = error.get("status")
    kind = kind if isinstance(kind, str) and kind in _ERROR_STATUSES else "gemini_http_error"
    safe = {"type": kind, "message": f"Gemini request failed (HTTP {status}; {kind})."}
    if kind in _ERROR_STATUSES:
        safe["status"] = kind
    code = error.get("code")
    if type(code) is int and 0 <= code <= 599:
        safe["code"] = code
    # Preserve only this specific recovery classification, never a provider
    # message (which may contain account, key, URL or prompt details).
    message = str(error.get("message", "")).lower().replace("_", " ")
    if (status == 400 and "thought" in message and "signature" in message
            and any(word in message for word in ("invalid", "missing", "required", "not valid"))):
        safe["code"] = "invalid_thinking_signature"
        safe["message"] += " Invalid thinking signature."
    return {"error": safe}


class GeminiAPITransport(httpx.BaseTransport):
    """Route exactly one native request with private header-only authentication.

    Put this inside RetryingGenerationTransport. Public request references and
    all returned error bodies omit the API key and provider account details.
    No credential files, service accounts, token refreshes or retries occur here.
    """

    def __init__(self, model, *, api_key, transport=None):
        self._model = _model_id(model)
        if (not isinstance(api_key, str) or not api_key
                or any(not 33 <= ord(character) <= 126 for character in api_key)):
            raise ValueError("Gemini API key must be a nonempty ASCII token")
        self._api_key = api_key
        self._path = f"/v1/models/{self._model}:generateContent"
        self._url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent"
        self._transport = transport if transport is not None else httpx.HTTPTransport(
            trust_env=False, retries=0)

    def handle_request(self, request):
        if (request.method != "POST" or request.url.path != self._path
                or request.url.query or request.url.fragment or request.url.username or request.url.password):
            raise ValueError("Gemini transport accepts only the configured native nonstreaming endpoint")
        try:
            content = request.read()
            payload = json.loads(content)
        except (ValueError, UnicodeError):
            raise ValueError("Gemini request must contain a JSON object") from None
        if not isinstance(payload, dict) or "model" in payload or "stream" in payload:
            raise ValueError("Gemini request must be a native nonstreaming JSON object")
        outgoing = httpx.Request("POST", self._url, content=content,
            headers={"x-goog-api-key": self._api_key, "Content-Type": "application/json",
                     "Accept": "application/json"}, extensions=dict(request.extensions))
        try:
            response = self._transport.handle_request(outgoing)
            body = response.read()
            response.close()
        except httpx.TransportError as exc:
            raise type(exc)(f"Gemini request failed ({type(exc).__name__})", request=request) from None
        except Exception as exc:
            raise httpx.TransportError(
                f"Gemini dispatch failed ({type(exc).__name__})", request=request) from None
        headers = {name: response.headers[name]
                   for name in ("content-type", "retry-after", "retry-after-ms") if name in response.headers}
        try:
            parsed = json.loads(body)
        except (ValueError, UnicodeError):
            parsed = None
        if response.status_code != 200 or isinstance(parsed, dict) and "error" in parsed:
            return httpx.Response(response.status_code, headers=headers,
                                 json=_safe_error(response.status_code, parsed), request=request)
        return httpx.Response(response.status_code, headers=headers, content=body, request=request)

    def close(self):
        self._transport.close()


def _native_parts(content):
    if (not isinstance(content, dict) or content.get("role") != "model"
            or not isinstance(content.get("parts"), list) or not content["parts"]):
        raise ValueError("Invalid Gemini model content")
    for part in content["parts"]:
        if not isinstance(part, dict):
            raise ValueError("Invalid Gemini model part")
        if "thought" in part and type(part["thought"]) is not bool:
            raise ValueError("Invalid Gemini thought marker")
        if "thoughtSignature" in part and not isinstance(part["thoughtSignature"], str):
            raise ValueError("Invalid Gemini thought signature")
        kinds = set(part) - {"thought", "thoughtSignature"}
        if kinds == {"text"} and isinstance(part["text"], str):
            continue
        if not kinds and part.get("thoughtSignature"):
            continue  # Empty signature-bearing parts must also be replayed.
        call = part.get("functionCall")
        if (kinds == {"functionCall"} and isinstance(call, dict)
                and isinstance(call.get("name"), str) and call["name"]
                and isinstance(call.get("args", {}), dict)
                and ("id" not in call or isinstance(call["id"], str) and call["id"])):
            json.dumps(call.get("args", {}), allow_nan=False)
            continue
        raise ValueError("Unsupported or malformed Gemini model part")
    return content["parts"]


def terminal_reason(payload):
    """Validate a native envelope; only STOP may produce executable actions.

    Unknown native finish reasons fail closed as provider terminal output.
    Malformed envelopes raise rather than being mistaken for successful stops.
    """
    if not isinstance(payload, dict) or "error" in payload:
        raise ValueError("Invalid Gemini response envelope")
    for key in ("responseId", "modelVersion"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise ValueError("Invalid Gemini response metadata")
    if "usageMetadata" in payload and not isinstance(payload["usageMetadata"], dict):
        raise ValueError("Invalid Gemini usage metadata")
    feedback = payload.get("promptFeedback", {})
    if not isinstance(feedback, dict):
        raise ValueError("Invalid Gemini prompt feedback")
    block = feedback.get("blockReason")
    if block is not None:
        if not isinstance(block, str) or not block:
            raise ValueError("Invalid Gemini prompt block reason")
        if block != "BLOCK_REASON_UNSPECIFIED":
            return "prompt_" + (block.lower() if block in _BLOCK_REASONS else "unknown_block_reason")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("Gemini must return exactly one response candidate")
    candidate = candidates[0]
    if "index" in candidate and (type(candidate["index"]) is not int or candidate["index"] != 0):
        raise ValueError("Invalid Gemini candidate index")
    reason = candidate.get("finishReason")
    if not isinstance(reason, str) or not reason or reason == "FINISH_REASON_UNSPECIFIED":
        raise ValueError("Gemini response has no completed finish reason")
    if reason != "STOP":
        return reason.lower() if reason in _FINISH_REASONS else "unknown_finish_reason"
    _native_parts(candidate.get("content"))
    return None


@dataclass(frozen=True)
class GeminiAssistantMessage(AssistantMessage):
    native_content: dict = field(default_factory=dict, repr=False)

    def raw(self):
        message = super().raw()
        message["_gemini_content"] = deepcopy(self.native_content)
        return message


def _user_parts(content):
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list) or not content:
        raise ValueError("Gemini user content must be text or nonempty multipart content")
    translated = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Invalid Gemini input content part")
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            translated.append({"text": part["text"]})
        elif part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else None
            if not isinstance(url, str) or not url.startswith(_PNG_PREFIX):
                raise ValueError("Gemini robot images must be inline PNG data URLs")
            data = url[len(_PNG_PREFIX):]
            try:
                decoded = base64.b64decode(data, validate=True)
            except (ValueError, UnicodeError):
                raise ValueError("Gemini robot image has invalid base64") from None
            if not decoded.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Gemini robot image is not PNG")
            translated.append({"inlineData": {"mimeType": "image/png", "data": data}})
        else:
            raise ValueError("Unsupported Gemini input content part")
    return translated


def _folded_turn(native):
    """A model turn rendered as one plain text part: its visible text and each call as ``name(args)``.

    Gemini validates the thought signature of every function-call part in the history, so a turn
    outside the signature window cannot keep its call parts; as text it needs no signature and the
    model still sees what it did. The matching tool results are rendered as text by the caller.
    """
    lines = []
    for part in native["parts"]:
        if part.get("thought", False):
            continue
        if "text" in part and part["text"]:
            lines.append(str(part["text"]))
        if "functionCall" in part:
            call = part["functionCall"]
            lines.append(f"[tool call] {call['name']}({json.dumps(call.get('args', {}), allow_nan=False)})")
    return {"role": "model", "parts": [{"text": "\n".join(lines) or "[no output]"}]}


def _translate_messages(messages, thought_signature_window=None):
    """Chat history to (system parts, contents). With a window, only the newest that many model turns
    are replayed natively with their thought signatures; older turns and their tool results become
    plain text, since Gemini rejects historical function calls without a valid signature."""
    system, contents, pending, seen_ids = [], [], {}, set()
    model_turns = sum(1 for message in messages if isinstance(message, dict) and message.get("role") == "assistant")
    model_index = 0

    def user(parts):
        if contents and contents[-1]["role"] == "user":
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": "user", "parts": parts})

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Invalid Gemini transcript message")
        role = message.get("role")
        if role == "system":
            if contents or not isinstance(message.get("content"), str):
                raise ValueError("Gemini system messages must precede the conversation")
            system.append({"text": message["content"]})
        elif role == "user":
            if pending:
                raise ValueError("Gemini tool feedback is missing")
            user(_user_parts(message.get("content")))
        elif role == "assistant":
            if pending:
                raise ValueError("Gemini tool feedback is missing")
            native = deepcopy(message.get("_gemini_content"))
            parts = _native_parts(native)  # Never synthesize missing signed content.
            native_calls = [part["functionCall"] for part in parts if "functionCall" in part]
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list) or len(calls) != len(native_calls):
                raise ValueError("Gemini canonical and native tool history differ")
            model_index += 1
            folded = thought_signature_window is not None and model_index <= model_turns - thought_signature_window
            for call, original in zip(calls, native_calls):
                call_id = call.get("id") if isinstance(call, dict) else None
                function = call.get("function", {}) if isinstance(call, dict) else {}
                if (not isinstance(call_id, str) or not call_id or call_id in seen_ids
                        or not isinstance(function, dict) or function.get("name") != original["name"]):
                    raise ValueError("Invalid Gemini canonical tool identity")
                try:
                    arguments = json.loads(function["arguments"])
                except (ValueError, KeyError, TypeError):
                    raise ValueError("Invalid Gemini canonical tool arguments") from None
                if arguments != original.get("args", {}):
                    raise ValueError("Gemini canonical and native tool arguments differ")
                seen_ids.add(call_id)
                pending[call_id] = {"original": original, "folded": folded}
            contents.append(_folded_turn(native) if folded else native)
        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in pending or call_id != next(iter(pending)):
                raise ValueError("Gemini tool feedback must match the native call order")
            entry = pending.pop(call_id)
            original = entry["original"]
            try:
                result = json.loads(message["content"])
            except (ValueError, KeyError, TypeError):
                raise ValueError("Gemini tool feedback must contain a JSON object") from None
            if not isinstance(result, dict):
                raise ValueError("Gemini tool feedback must contain a JSON object")
            if entry["folded"]:
                user([{"text": f"[tool result] {original['name']}: {json.dumps(result, allow_nan=False)}"}])
                continue
            reply = {"name": original["name"], "response": result}
            if "id" in original:
                reply["id"] = original["id"]
            user([{"functionResponse": reply}])
        else:
            raise ValueError("Unsupported Gemini transcript role")
    if pending or not contents or contents[-1]["role"] != "user":
        raise ValueError("Gemini request requires complete feedback and a final user observation")
    return system, contents


def _translate_tools(tools):
    declarations = []
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if (not isinstance(tool, dict) or tool.get("type") != "function"
                or not isinstance(function, dict) or not isinstance(function.get("name"), str)
                or not function["name"] or not isinstance(function.get("parameters"), dict)):
            raise ValueError("Invalid Gemini function schema")
        declaration = {"name": function["name"], "parametersJsonSchema": deepcopy(function["parameters"])}
        if "description" in function:
            if not isinstance(function["description"], str):
                raise ValueError("Invalid Gemini function description")
            declaration["description"] = function["description"]
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else []


class GeminiClient:
    """One native generation per call, returning Inspect-compatible messages."""

    def __init__(self, provider, *, max_output_tokens=16000, timeout_s=600.,
                 transport=None, max_retries=1, backoff_s=0., capture=None, thought_signature_window=None):
        self._model = _model_id(provider.model)
        if thought_signature_window is not None and (type(thought_signature_window) is not int
                                                     or thought_signature_window < 1):
            raise ValueError("Gemini thought signature window must be a positive integer or None")
        self._signature_window = thought_signature_window
        # A prompt block of reason OTHER is transient (the same request replayed cleanly), so it is
        # retried a bounded number of times through the counted outer transport before it is terminal.
        self._prompt_block_retries = 2
        if provider.wire != "gemini" or provider.api_key:
            raise ValueError("Use Gemini wire with authentication isolated in GeminiAPITransport")
        url = httpx.URL(provider.base_url)
        if (url.scheme != "https" or url.host != "gemini.invalid" or url.path.rstrip("/") != "/v1"
                or url.query or url.fragment or url.username or url.password):
            raise ValueError("Gemini client requires the sanitized https://gemini.invalid/v1 base URL")
        if transport is None or max_retries != 1 or backoff_s != 0 or capture is not None:
            raise ValueError("Gemini counting, capture and retry belong to the supplied outer transport")
        if type(max_output_tokens) is not int or max_output_tokens < 1:
            raise ValueError("Gemini maximum output tokens must be a positive integer")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Gemini timeout must be positive")
        self._max_output_tokens = max_output_tokens
        self._response_count = 0
        self._http = httpx.Client(base_url=provider.base_url, transport=transport, trust_env=False,
                                 timeout=httpx.Timeout(timeout_s, connect=min(10., timeout_s)))

    def complete(self, messages, tools, temperature=None, reasoning_effort="medium"):
        effort = "medium" if reasoning_effort is None else reasoning_effort
        if not isinstance(effort, str) or effort not in {"low", "medium", "high"}:
            raise ValueError("Gemini reasoning effort must be low, medium or high")
        system, contents = _translate_messages(messages, self._signature_window)
        config = {"candidateCount": 1, "maxOutputTokens": self._max_output_tokens,
                  "thinkingConfig": {"thinkingLevel": effort.upper()}}
        if temperature is not None:
            if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                    or not math.isfinite(temperature) or not 0 <= temperature <= 2):
                raise ValueError("Gemini temperature must be finite and between zero and two")
            config["temperature"] = temperature
        body = {"contents": contents, "generationConfig": config}
        if system:
            body["systemInstruction"] = {"parts": system}
        if tools:
            body["tools"] = _translate_tools(tools)
        # Validate locally before dispatch; no NaN or silently coerced feedback.
        encoded = json.dumps(body, allow_nan=False).encode("utf-8")
        for attempt in range(self._prompt_block_retries + 1):
            response = self._http.post(f"/models/{self._model}:generateContent", content=encoded,
                                       headers={"Content-Type": "application/json"})
            if response.status_code != 200:
                raise RuntimeError(f"Gemini request failed (HTTP {response.status_code})")
            payload = response.json()
            reason = terminal_reason(payload)
            if reason == "prompt_other" and attempt < self._prompt_block_retries:
                time.sleep(2.)
                continue
            break
        if reason is not None:
            raise RuntimeError("Gemini terminal provider output: " + reason)
        native = payload["candidates"][0]["content"]
        text, calls = [], []
        for part in native["parts"]:
            if "text" in part and not part.get("thought", False):
                text.append(part["text"])
            if "functionCall" in part:
                call = part["functionCall"]
                calls.append(ToolCall(id=f"gemini_call_{self._response_count}_{len(calls)}",
                    name=call["name"], arguments=json.dumps(call.get("args", {}), allow_nan=False)))
        self._response_count += 1
        return GeminiAssistantMessage(content="".join(text) or None, tool_calls=tuple(calls),
                                      usage=deepcopy(payload.get("usageMetadata")), native_content=deepcopy(native))

    def clear_continuation(self):
        """No external cache: replacing the transcript removes native history."""

    def close(self):
        self._http.close()
