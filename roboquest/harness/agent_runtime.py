"""Evaluator-side API setup and deterministic calibration responses."""

import json
from dataclasses import asdict
from datetime import timezone
from email.utils import parsedate_to_datetime
import hashlib
import math
import os
from pathlib import Path
import re
import shlex
import stat
import time
from urllib.parse import urlsplit
import httpx
import numpy as np
from scipy.spatial.transform import Rotation
from roboquest.harness.cartesian import LABELS, decode_rotation, encode_rotation


def provider_environment(key_env, key_file=None, env_file=None):
    """Read only the explicitly named credential into memory; never serialize it."""
    value = os.environ.get(key_env, "")
    if key_file is not None and env_file is not None:
        raise ValueError("Supply either key_file or env_file")
    if key_file is not None or env_file is not None:
        path = Path(key_file if key_file is not None else env_file).expanduser()
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise ValueError("API key file must be private (chmod 600)")
        if key_file is not None:
            value = path.read_text().strip()
        else:
            value = ""
            for line in path.read_text().splitlines():
                match = re.match(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$", line)
                if match and match[1] == key_env:
                    # Parse the selected value as data. Never source/execute a .env file.
                    try:
                        parts = shlex.split(match[2], comments=True)
                    except ValueError:
                        raise ValueError("Selected API key has invalid quoting") from None
                    if len(parts) != 1:
                        raise ValueError("Selected API key must contain one value")
                    value = parts[0]
                    break  # Preserve the first assignment if a name is duplicated.
    if not value:
        raise ValueError(f"No API key available via {key_env} or the supplied private key file")
    if any(character.isspace() for character in value):
        raise ValueError("API key must be a single nonempty value")
    return {key_env: value}


def validate_endpoint(base_url):
    parsed = urlsplit(base_url)
    if (parsed.scheme not in ("http", "https") or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment):
        raise ValueError("Endpoint must be an HTTP(S) base URL without credentials or query parameters")


def check_model(base_url, model, environment, key_env):
    validate_endpoint(base_url)
    with httpx.Client(trust_env=False, timeout=20.) as client:
        response = client.get(base_url.rstrip("/") + "/models",
                              headers={"Authorization": "Bearer " + environment[key_env]})
        if response.status_code != 200:
            raise RuntimeError(f"Models check failed with HTTP {response.status_code}")
        available = {row["id"] for row in response.json().get("data", [])}
        if model not in available:
            raise ValueError(f"Requested model {model!r} is absent from the relay models response")
    return {"model": model, "model_list_verified": True}


class GenerationInterrupted(RuntimeError):
    """Infrastructure interruption, never a model decision or physical stop."""

    def __init__(self, reason, *, status=None, error_type=None):
        self.reason = reason
        self.status = status
        self.error_type = error_type
        # Never include provider bodies, exception text, or credentials here.
        super().__init__(f"API interruption: {reason}" + (f" (HTTP {status})" if status else ""))

    def as_dict(self):
        return {"reason": self.reason, "http_status": self.status, "error_type": self.error_type}


class GenerationBudgetExhausted(GenerationInterrupted):
    """No additional generation request may be sent in this trial."""

    def __init__(self):
        super().__init__("generation_attempt_budget_exhausted")


class RequestBudgetTransport(httpx.BaseTransport):
    """Count HTTP generation attempts, with an optional independent limit."""

    def __init__(self, transport, budget):
        if budget is not None and (type(budget) is not int or budget < 1):
            raise ValueError("generation attempt budget must be a positive integer or None")
        self.transport = transport
        self.budget = budget
        self.attempts = 0

    def handle_request(self, request):
        if self.budget is not None and self.attempts >= self.budget:
            raise GenerationBudgetExhausted()
        self.attempts += 1
        return self.transport.handle_request(request)

    def close(self):
        self.transport.close()


def retry_after_seconds(headers, now=None):
    """Parse a server delay without treating invalid/negative values as advice."""
    value = headers.get("retry-after")
    if value is not None:
        try:
            delay = float(value)
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                delay = max(0., date.timestamp() - (time.time() if now is None else now))
            except (TypeError, ValueError, OverflowError):
                delay = math.nan
        if math.isfinite(delay) and delay >= 0:
            return delay
    try:
        delay = float(headers["retry-after-ms"]) / 1000
    except (KeyError, ValueError):
        delay = None
    if delay is not None:
        return delay if math.isfinite(delay) and delay >= 0 else None
    # OpenRouter's per-minute limits carry no Retry-After, only the window's end as an epoch in milliseconds.
    try:
        reset = float(headers["x-ratelimit-reset"]) / 1000
    except (KeyError, ValueError):
        return None
    if not math.isfinite(reset):
        return None
    return max(0., reset - (time.time() if now is None else now))


def _permanent_provider_error(response):
    """Known quota/billing errors need operator action even when HTTP is 429."""
    try:
        payload = response.json()
    except ValueError:
        return False
    error = payload.get("error") if isinstance(payload, dict) else None
    if not isinstance(error, dict):
        return False
    permanent = {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
                 "billing_error", "credit_balance_too_low", "invalid_api_key",
                 "authentication_error", "permission_error"}
    return any(isinstance(error.get(key), str) and error[key] in permanent for key in ("code", "type"))


TRANSIENT_FAILURE_CODES = {"rate_limit_exceeded", "rate_limit_error", "server_error", "upstream_error",
                           "overloaded_error", "overloaded", "service_unavailable", "timeout", "internal_error"}


def _transient_failure_in_body(response):
    """OpenRouter can answer HTTP 200 with a Responses body of ``status: failed`` and an ``error`` block when the
    upstream is rate-limited ("temporarily rate-limited upstream. Please retry shortly"). Such a body is a
    transient failure to retry like a 429, not a response to parse. Returns the error code, or None."""
    if response is None or response.status_code != 200:
        return None
    try:
        payload = response.json()
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("status") != "failed":
        return None
    error = payload.get("error")
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, str) and code in TRANSIENT_FAILURE_CODES else None


class RetryingGenerationTransport(RequestBudgetTransport):
    """Retry only the frozen HTTP request, before any tool/action is returned.

    The pinned provider client must have retries and capture disabled. This layer
    captures every physical HTTP attempt using the policy's existing WireCapture,
    while the provider continues parsing and preserving raw reasoning items.
    No observation, conversation, tool or simulator operation is replayed here.
    """

    def __init__(self, transport, budget=40, *, max_request_attempts=6, backoff_s=0.,
                 max_retry_delay_s=120., max_retry_wait_s=300., before_attempt=None,
                 after_attempt=None, sleep=None, honor_retry_after=False):
        super().__init__(transport, budget)
        if (isinstance(max_request_attempts, bool) or not isinstance(max_request_attempts, int)
                or max_request_attempts < 1):
            raise ValueError("max_request_attempts must be a positive integer")
        for name, value in (("backoff_s", backoff_s), ("max_retry_delay_s", max_retry_delay_s),
                            ("max_retry_wait_s", max_retry_wait_s)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.max_request_attempts = max_request_attempts
        self.backoff_s = backoff_s
        self.max_retry_delay_s = max_retry_delay_s
        self.max_retry_wait_s = max_retry_wait_s
        self.honor_retry_after = honor_retry_after
        self.before_attempt = before_attempt
        self.after_attempt = after_attempt
        self.sleep = time.sleep if sleep is None else sleep
        self.capture = None
        self.logical_requests = 0
        self.accepted_responses = 0
        self.events = []
        self.interruption = None
        # The policy may inspect a completed response rejected by the pinned
        # parser (e.g. a native Messages output-limit stop). Never serialize
        # this Response object: WireCapture already owns the raw evidence.
        self.last_response = None

    def interrupt(self, reason, *, status=None, error_type=None):
        error = GenerationInterrupted(reason, status=status, error_type=error_type)
        self.interruption = error.as_dict()
        raise error

    def _ledger_hook(self, callback, *args):
        if callback is not None:
            try:
                callback(*args)
            except Exception as exc:
                self.interrupt("batch_attempt_ledger_failed", error_type=type(exc).__name__)

    def handle_request(self, request):
        self.last_response = None
        # Materialize once: the same exact JSON, headers and URL reach each retry.
        content = request.read()
        payload = json.loads(content)
        self.logical_requests += 1
        call = self.logical_requests - 1
        digest = hashlib.sha256(content).hexdigest()
        waited = 0.
        for attempt in range(self.max_request_attempts):
            if self.budget is not None and self.attempts >= self.budget:
                self.interrupt("generation_attempt_budget_exhausted")
            self._ledger_hook(self.before_attempt)
            outgoing = httpx.Request(request.method, request.url, headers=request.headers,
                                     content=content, extensions=dict(request.extensions))
            started = time.time()
            clock_start = time.monotonic()
            response = None
            error_type = None
            try:
                response = super().handle_request(outgoing)
                response.read()  # Fully consume/close before retry; no streaming replay.
            except httpx.TransportError as exc:
                error_type = type(exc).__name__
                if response is not None:
                    response.close()
                response = None
            duration = time.monotonic() - clock_start
            status = response.status_code if response is not None else None
            server_delay = retry_after_seconds(response.headers) if response is not None else None
            event = {"call": call, "attempt": attempt, "http_attempt": self.attempts,
                     "status": status, "error_type": error_type, "duration_s": duration,
                     "request_sha256": digest, "retry_after_s": server_delay,
                     "retry_delay_s": None}
            self.events.append(event)
            if self.capture is not None:
                endpoint = next((suffix for suffix in ("/chat/completions", "/responses", "/messages")
                                 if request.url.path.endswith(suffix)), request.url.path)
                self.capture.record(attempt=attempt, endpoint=endpoint, request=payload,
                                    status=status, response_text=response.text if response is not None else None,
                                    error=error_type, t_start=started, duration_s=duration)
            failed_code = _transient_failure_in_body(response)
            if status == 200 and failed_code is None:
                self._ledger_hook(self.after_attempt, status, None)
                self.last_response = response
                return response
            if failed_code is not None:         # HTTP 200 carrying a failed status: retry it like a 429
                error_type = f"failed:{failed_code}"
                event["error_type"] = error_type
            transient = (response is None or status in (408, 409, 429) or status >= 500 or failed_code is not None)
            if response is not None and _permanent_provider_error(response):
                transient = False
            if not transient:
                self._ledger_hook(self.after_attempt, status, None)
                self.interrupt("permanent_provider_error", status=status, error_type=error_type)
            # Company-API default: retry promptly, with no client pacing/jitter or
            # server-delay wait. Optional explicit delay mode honors Retry-After
            # as a minimum and defers rather than retrying before a long hint.
            exponential = min(self.max_retry_delay_s, self.backoff_s * 2 ** min(attempt, 30))
            delay = max(exponential, (server_delay or 0.) if self.honor_retry_after else 0.)
            self._ledger_hook(self.after_attempt, status, delay)
            if self.budget is not None and self.attempts >= self.budget:
                self.interrupt("generation_attempt_budget_exhausted", status=status, error_type=error_type)
            if attempt + 1 >= self.max_request_attempts:
                self.interrupt("request_retry_attempts_exhausted", status=status, error_type=error_type)
            if self.honor_retry_after and server_delay is not None and server_delay > self.max_retry_delay_s:
                self.interrupt("retry_after_exceeds_delay_limit", status=status, error_type=error_type)
            if waited + delay > self.max_retry_wait_s:
                self.interrupt("request_retry_wait_exhausted", status=status, error_type=error_type)
            event["retry_delay_s"] = delay
            if delay > 0:
                self.sleep(delay)
            waited += delay

    def summary(self):
        return {"generation_attempts": self.attempts, "logical_requests": self.logical_requests,
                "accepted_responses": self.accepted_responses,
                "http_200_responses": sum(row["status"] == 200 for row in self.events),
                "api_interruption": self.interruption, "attempts": self.events}


class CalibrationResponses:
    """Script reads ONLY the outgoing proprioception text, through MockTransport.

    Exercises the actual Responses client and plugin without contacting a model.
    It has no embodiment reference, simulator lookup, or task-solving routine.
    """

    def __init__(self):
        self.phases = [("translation", axis, sign * 0.03)
                       for axis in range(3) for sign in (1, -1)]
        self.phases += [("rotation", axis, sign * 0.15)
                        for axis in range(3) for sign in (1, -1)]
        self.phases += [("gripper", 0, value) for value in (0., 0.4, 1.)]
        self.calls = 0
        self.pending = None
        self.measurements = []

    def __call__(self, request):
        payload = json.loads(request.content)
        texts = [part["text"] for item in payload["input"]
                 if item.get("role") == "user" and isinstance(item.get("content"), list)
                 for part in item["content"] if part.get("type") == "input_text"
                 and "state[eef_target_state]:" in part.get("text", "")]
        line = texts[-1].split("state[eef_target_state]:", 1)[1].split("\n", 1)[0]
        values = {name: float(value) for name, value in re.findall(r"(\w+)=([-+\deE.]+)", line)}
        measured = np.array([values[label] for label in LABELS])
        if self.pending is not None:
            phase, before, target = self.pending
            error = [float(np.linalg.norm(target[:3] - measured[:3])),
                     float(Rotation.from_matrix(decode_rotation(target[3:9]) @
                                                decode_rotation(measured[3:9]).T).magnitude()),
                     float(abs(target[-1] - measured[-1]))]
            self.measurements.append({"phase": phase, "before": before.tolist(),
                                      "target": target.tolist(), "achieved": measured.tolist(),
                                      "error_m_rad_opening": error})
        if self.calls < len(self.phases):
            kind, axis, amount = self.phases[self.calls]
            target = measured.copy()
            if kind == "translation":
                target[axis] += amount
                targets = {LABELS[axis]: float(target[axis])}
            elif kind == "rotation":
                rotated = Rotation.from_rotvec(np.eye(3)[axis] * amount).as_matrix()
                target[3:9] = encode_rotation(rotated @ decode_rotation(measured[3:9]))
                targets = dict(zip(LABELS[3:9], target[3:9].tolist()))
            else:
                target[-1] = amount
                targets = {"gripper": amount}
            phase = f"{kind}_{axis}_{amount:+.2f}"
            self.pending = phase, measured, target
            name = "move_to"
            arguments = {"targets": targets, "note": f"Scripted control calibration: {phase}."}
        else:
            self.pending = None
            name = "done"
            arguments = {"summary": "Control calibration ended; this is not a task solution.",
                         "hindsight": "none"}
        self.calls += 1
        return httpx.Response(200, json={"status": "completed", "model": "offline-calibration",
            "output": [{"type": "function_call", "call_id": f"calibration_{self.calls}",
                        "name": name, "arguments": json.dumps(arguments)}]})

    def passed(self):
        return (len(self.measurements) == len(self.phases)
                and all(np.all(np.asarray(row["error_m_rad_opening"]) < [0.006, 0.035, 0.07])
                        for row in self.measurements))


def wire_usage(log_dir):
    rows = []
    for path in Path(log_dir).glob("wire/*/*/calls.jsonl"):
        for line in path.read_text().splitlines():
            captured = json.loads(line)
            response = captured.get("response")
            if isinstance(response, dict):
                rows.append({"call": captured["call"], "attempt": captured["attempt"],
                             "http_status": captured["status"], "model": response.get("model"),
                             "usage": response.get("usage")})
    # Preserve provider-native details; unknown usage/cost is not zero.
    return {"responses": rows, "usage_available": any(row["usage"] is not None for row in rows),
            "cost_usd": None}


def evaluation_outcome(log):
    """eval(fail_on_error=True) can RETURN an error log instead of raising."""
    completed = log.status == "success" and log.results.errored_trials == 0
    return {"completed": completed, "eval_status": log.status,
            "inspect_results": asdict(log.results),
            "final_placement": log.results.metrics.get("final_placement"),
            "score_is_official": "final_placement" in log.results.metrics,
            "trial_error_types": [sample.error.split(":", 1)[0]
                                  for sample in log.samples if sample.error]}
