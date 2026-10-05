"""Small simulation adaptations around the pinned, actual Inspect agent plugin.

Provider transport, history, tool validation, interpolation and logging remain
upstream. Private hooks used here are pinned and covered by integration tests.
"""

import copy
from dataclasses import replace
import numpy as np
from inspect_robots import Action, ActionChunk
from inspect_robots.approver import ChainApprover, ClampApprover, DeltaLimitApprover
from inspect_robots_agent import LLMAgentPolicy
from inspect_robots_agent.policy import _MAX_CONSECUTIVE_FAILURES
from inspect_robots_agent._tools import ToolResult
from roboquest.harness.agent_runtime import GenerationInterrupted, RetryingGenerationTransport
from roboquest.harness.cartesian import SETTLE_STEPS, validate_waypoints


class _NativeResponseStop(Exception):
    """A complete native model response, rather than an API failure."""

    def __init__(self, payload):
        self.payload = payload


def _native_terminal_payload(response):
    """Validate the Messages envelope before accepting a terminal model turn.

    The pinned parser rejects these stop reasons before it examines content.
    Do not infer a model refusal from its RuntimeError text: an error envelope,
    missing tool fields, or invalid JSON must remain infrastructure evidence.
    Unknown block/stop types require an explicit compatibility review.
    """
    if response is None or response.status_code != 200:
        return None
    try:
        payload = response.json()
    except (ValueError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("stop_reason"), str) or payload["stop_reason"] not in {
            "refusal", "max_tokens", "model_context_window_exceeded", "pause_turn"}:
        return None
    if (payload.get("type") != "message" or payload.get("role") != "assistant"
            or not all(isinstance(payload.get(key), str) and payload[key] for key in ("id", "model"))
            or not isinstance(payload.get("content"), list)):
        return None
    for block in payload["content"]:
        if not isinstance(block, dict):
            return None
        kind = block.get("type")
        if not isinstance(kind, str):
            return None
        string_fields = {"text": ("text",), "thinking": ("thinking", "signature"),
                         "redacted_thinking": ("data",), "tool_use": ("id", "name")}.get(kind)
        if string_fields is None or not all(isinstance(block.get(key), str) for key in string_fields):
            return None
        if kind == "tool_use" and (not block["id"] or not block["name"]
                                   or not isinstance(block.get("input"), dict)):
            return None
    usage = payload.get("usage")
    if not isinstance(usage, dict) or not all(
            isinstance(usage.get(key), int) and not isinstance(usage[key], bool) and usage[key] >= 0
            for key in ("input_tokens", "output_tokens")):
        return None
    if payload.get("stop_details") is not None and not isinstance(payload["stop_details"], dict):
        return None
    return payload


def _model_output_exhaustion(exc):
    """Recognize only the pinned policy's own consecutive-output failure exits.

    Upstream uses untyped RuntimeError for these two exits. Checking the origin
    and its failure counter avoids swallowing equally worded provider/tool
    exceptions. These private hooks are tied to the pinned plugin and tested
    through its actual act loop; changing that loop requires review.
    """
    traceback = exc.__traceback__
    while traceback is not None and traceback.tb_next is not None:
        traceback = traceback.tb_next
    if traceback is None or traceback.tb_frame.f_code is not LLMAgentPolicy.act.__code__:
        return None
    state = traceback.tb_frame.f_locals
    failures = state.get("failures", 0)
    if not isinstance(failures, int) or failures < _MAX_CONSECUTIVE_FAILURES:
        return None
    message = state.get("message")
    if message is not None and not message.tool_calls:
        kind = "missing_tool_calls"
    elif state.get("last_error") is not None:
        kind = "invalid_tool_calls"
    else:
        return None
    return {"failure_kind": kind, "consecutive_failures": failures}


class CartesianToolset:
    def __init__(self, upstream):
        self.upstream = upstream

    def __getattr__(self, name):
        return getattr(self.upstream, name)

    def schemas(self):
        schemas = copy.deepcopy(self.upstream.schemas())
        move = schemas[0]["function"]
        move["description"] = (
            "Move to absolute grip-site targets. x/y/z are world metres; the six "
            "r1/r2 components are dimensionless rot6d columns relative to trial "
            "start, as defined in the embodiment notes. Gripper is normalized "
            "opening (0 closed, 1 open). Omitted dimensions hold measured values. "
            "Supply all six rotation components together when rotating. The "
            "framework linearly interpolates targets, projects rot6d bases, and "
            f"holds the final target for {SETTLE_STEPS} additional 20 Hz ticks. "
            "All ticks count toward the physical horizon. A move lasts at most "
            "10 seconds including settling. Arrival is not guaranteed; check "
            "measured state and tracking_error. " + self.upstream._bounds_text
        )
        return schemas

    def execute(self, call, observation):
        result = self.upstream.execute(call, observation)
        if result.chunk is None or result.target is None:
            return result
        actions = list(result.chunk.actions)
        if len(actions) + SETTLE_STEPS > 200:
            return ToolResult(error="motion plus settling exceeds 10 seconds; split the move")
        actions[-1] = replace(actions[-1], meta={})
        actions[0] = replace(actions[0], meta={"chunk_start": True})
        actions.extend(Action(result.target.copy()) for _ in range(SETTLE_STEPS))
        actions[-1] = replace(actions[-1], meta={"chunk_final": True})
        return replace(result, chunk=ActionChunk(actions, control_hz=20.),
                       note=f"executing move_to over {len(actions)} steps "
                            f"({len(actions) / 20:.2f}s), including {SETTLE_STEPS} settling steps")


class SimulationAgentPolicy(LLMAgentPolicy):
    def __init__(self, retry_backoff_s=0., **kwargs):
        if not np.isfinite(retry_backoff_s) or not 0 <= retry_backoff_s <= 30:
            raise ValueError("retry_backoff_s must be between 0 and 30 seconds")
        super().__init__(pre_check=validate_waypoints, **kwargs)
        self.api_interruption = None
        self.model_stop = None
        self.decision_budget_exhausted = False
        self.api_transport = kwargs.get("transport")
        if isinstance(self.api_transport, RetryingGenerationTransport):
            # One retry owner. Keep the real client instance (Anthropic history
            # eviction uses isinstance) and its raw response caches untouched.
            self._client._max_retries = 1
            self._client._capture = None
            self.api_transport.capture = self._capture
            self._provider_complete = self._client.complete
            self._client.complete = self._complete
        else:
            # Compatibility for direct offline/plugin callers without the wrapper.
            self._client._backoff_s = retry_backoff_s

    def _complete(self, *args, **kwargs):
        attempts_before = self.api_transport.attempts
        try:
            message = self._provider_complete(*args, **kwargs)
        except GenerationInterrupted:
            raise
        except (ValueError, KeyError, TypeError, RuntimeError, AttributeError, IndexError) as exc:
            # An HTTP 200 with an invalid body is still a charged attempt, but
            # not an accepted decision. Do not replay a malformed success.
            if (self.api_transport.attempts > attempts_before and self.api_transport.events
                    and self.api_transport.events[-1]["status"] == 200):
                terminal = (_native_terminal_payload(self.api_transport.last_response)
                            if self._wire == "messages" else None)
                if terminal is not None:
                    self.api_transport.accepted_responses += 1
                    raise _NativeResponseStop(terminal) from None
                self.api_transport.interrupt("invalid_provider_response", error_type=type(exc).__name__)
            raise
        self.api_transport.accepted_responses += 1
        return message

    @property
    def successful_model_decisions(self):
        # Upstream increments only after complete() returns, including accepted
        # responses whose tool call is absent or later rejected by validation.
        return self._calls_used

    def bind(self, embodiment_info):
        super().bind(embodiment_info)
        self._toolset = CartesianToolset(self._toolset)

    def reset(self, scene):
        super().reset(scene)
        self.api_interruption = None
        self.model_stop = None
        self.decision_budget_exhausted = False
        text = self._messages[0]["content"]
        expected = "You are controlling a real robot embodiment"
        if not text.startswith(expected):
            raise RuntimeError("Pinned Inspect system template changed; review simulation adaptation")
        self._messages[0]["content"] = text.replace(
            expected, "You are controlling a simulated robot embodiment", 1)

    def close(self):
        self._client.close()

    def act(self, observation):
        if self.model_stop is not None:
            return self._model_stop_chunk(observation)
        try:
            return super().act(observation)
        except _NativeResponseStop as exc:
            # complete() did not return, so upstream has not counted/logged this
            # accepted turn. Record it exactly once and never execute its tools:
            # max_tokens can contain a partially produced tool_use block.
            payload = exc.payload
            self._calls_used += 1
            for key, value in payload["usage"].items():
                if isinstance(value, int) and not isinstance(value, bool):
                    self._usage_totals[key] = self._usage_totals.get(key, 0) + value
            self._messages.append({"role": "assistant", "content": copy.deepcopy(payload["content"]),
                                   "provider_stop_reason": payload["stop_reason"]})
            self.model_stop = {"source": "provider", "wire": self._wire,
                               "reason": payload["stop_reason"], "response_id": payload["id"],
                               "termination_reason": "provider_" + payload["stop_reason"]}
            return self._model_stop_chunk(observation)
        except GenerationInterrupted as exc:
            self.api_interruption = exc.as_dict()
            raise
        except RuntimeError as exc:
            exhausted = _model_output_exhaustion(exc)
            if exhausted is None:
                raise
            # These turns were already counted and appended by upstream. End
            # normally at the actual state, without fabricating a give_up call.
            self.model_stop = {"source": "policy", "wire": self._wire,
                               "reason": "model_output_exhausted",
                               "termination_reason": "model_output_exhausted", **exhausted}
            return self._model_stop_chunk(observation)

    def _model_stop_chunk(self, observation):
        self._echo(f"[agent] -- {self.model_stop['termination_reason']}; ending at current state")
        return ActionChunk([Action(np.asarray(observation.state["eef_target_state"]).copy(),
            meta={"request_stop": True, "stop_reason": self.model_stop["termination_reason"],
                  "model_stop": dict(self.model_stop)})], control_hz=20.)

    def _forced_give_up(self, toolset, observation, why):
        self.decision_budget_exhausted = why == "LLM call budget exhausted"
        return super()._forced_give_up(toolset, observation, why)

    def on_trial_end(self, record, log_dir, run_id):
        super().on_trial_end(record, log_dir, run_id)
        record.metadata["successful_model_decisions"] = self.successful_model_decisions
        record.metadata["decision_budget_exhausted"] = self.decision_budget_exhausted
        if self.model_stop is not None:
            record.metadata["model_stop"] = self.model_stop
        if isinstance(self.api_transport, RetryingGenerationTransport):
            record.metadata["api_requests"] = {key: value for key, value in self.api_transport.summary().items()
                                                if key != "attempts"}
        if self.api_interruption is not None:
            record.metadata["api_interruption"] = self.api_interruption


class CartesianApprover:
    """Rebase each new chunk's limiter on achieved pose, not the last command.

    The generic absolute limiter otherwise clamps a new measured-pose trajectory
    toward an old unreached target after contact or partial chunk execution.
    """

    def __init__(self, embodiment):
        self.body = embodiment
        self.chain = ChainApprover(ClampApprover(embodiment.info.action_space),
                                   DeltaLimitApprover(embodiment.info.action_space))

    def review(self, action, store):
        if action.meta.get("request_stop"):
            return action
        if action.meta.get("chunk_start"):
            # Set an initial reference too: upstream's first action is otherwise unbounded.
            measured = self.body.observe().state["eef_target_state"]
            DeltaLimitApprover(self.body.info.action_space).review(Action(measured), store)
            DeltaLimitApprover.rewind_reference(store, measured)
        return self.chain.review(action, store)
