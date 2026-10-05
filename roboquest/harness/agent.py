"""Observation-only frontier policy using pinned Inspect provider clients.

This module receives public packets, never an environment or task specification.
It preserves native provider continuation and delegates frozen-request retries to
the existing company-API transport. It does not own simulator control or scoring.
"""
from copy import deepcopy
import base64
from io import BytesIO
import json
import math
import time

import httpx
import numpy as np
from PIL import Image

from roboquest.harness.agent_runtime import GenerationInterrupted, RetryingGenerationTransport, validate_endpoint
from roboquest.harness.contract import CAMERAS, PUBLIC_GOAL, policy_observation
from roboquest.harness.profiles import trusted_goal, trusted_control_version


# Every model the runner runs is registered here from the command line (scripts/run_episode.py: --model
# <provider>/<model id>, --effort, --max-output-tokens). api_model is the id sent to the API.
MODEL_CONFIGS = {}


def api_model(model):
    """The model id sent to the provider for a registered name (an alias may map several settings to one id)."""
    return MODEL_CONFIGS[model].get('api_model', model)
SYSTEM_PROMPT = (
    'You control a simulated mobile manipulation robot. Use only the provided '
    'camera images, robot proprioception and action/budget feedback. Determine '
    'what the goal requires from the scene. Choose exactly one arm, base, wait '
    'or stop tool per response. There are no semantic object-location or grasp '
    'tools. Images are upright RGB from three fixed cameras; new views require '
    'physical robot or object motion. The arm tool controls the right gripper '
    'site in world metres, with optional world orientation as quaternion xyzw. '
    'World z is up. Gripper -1 opens, +1 closes and 0 keeps its current setpoint. '
    'The base tool takes normalized body-frame forward, left and yaw velocity '
    'commands; positive yaw turns left. The arm stays relative to the base during '
    'base motion. The torso is held. A tick is 0.05 seconds. Commands consume their '
    'entire requested tick count, at most 200 ticks each; there is no automatic '
    'success stop or extra settling. Check observed outcomes: commanded targets '
    'are not guaranteed to be reached. Stop ends at the current physical state '
    'without advancing time. Invalid or multiple tool calls perform no physics '
    'and still consume this model response. Only the terminal state is evaluated.'
)


OBSERVATION_TEXT_VERSION = 'v2-goal-in-system-prompt-rounded-mm'


def goal_block(scene='baseline_counter'):
    """The public goal and control contract, stated once in the system message."""
    return ' Goal: ' + trusted_goal(scene) + ' Control contract: ' + trusted_control_version(scene) + '.'


def policy_system_prompt(max_decisions, scene='baseline_counter', include_goal=False):
    """Use the same public control instructions for API and in-session policies.

    ``include_goal`` appends the goal block; the API policy then omits the goal
    from every observation instead of repeating it each turn.
    """
    from roboquest.harness.profiles import completion_protocol
    prompt = SYSTEM_PROMPT
    if completion_protocol(scene) == 'physical_submit_v1':
        prompt = prompt[:prompt.index('Commands consume their')] + (
            'Commands consume up to 200 ticks each, ending early on the first physical '
            'SUBMIT button press. Check observed outcomes: commanded targets are not '
            'guaranteed to be reached. Physically press SUBMIT to commit the result; '
            'the first press freezes the score and ends the episode immediately. '
            'Wrong submission, timeout without submission, or calling stop without '
            'submission fails. The stop tool abandons the episode without advancing '
            'time. Invalid or multiple tool calls perform no physics and still consume '
            'this model response. No intermediate success is awarded.')
    if max_decisions is None:
        prompt += (
            ' There is no response-count cutoff. Follow the task completion instructions. '
            'A null remaining response or HTTP-attempt budget '
            'means there is no count limit.'
        )
    if include_goal:
        prompt += goal_block(scene)
    return prompt


def mark_cache_anchors(messages, count=2):
    """Flag the newest ``count`` image-free observations as prompt-cache anchors.

    Once an observation loses its images it never changes again, so the prefix
    ending there stays valid. The pinned Anthropic client turns the flag into a
    ``cache_control`` breakpoint (it also marks the system prompt and the final
    turn, so at most four breakpoints in total); each turn then reads the prefix
    the previous turn wrote instead of re-writing the whole history. The
    Responses wire has its own breakpoint transport and the Gemini client reads
    only role and content, so both ignore the flag.
    """
    observations = [m for m in messages if m.get('role') == 'user' and isinstance(m.get('content'), list)]
    image_free = [m for m in observations if not any(p.get('type') == 'image_url' for p in m['content'])]
    for message in observations:
        message.pop('cache_anchor', None)
    for message in image_free[-count:]:
        message['cache_anchor'] = True
    return len(image_free[-count:])


def _rounded(value, digits):
    """Round every float in a nested list/dict structure; other values pass through."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_rounded(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: _rounded(item, digits) for key, item in value.items()}
    return value


def _remaining_budget(limit, consumed):
    return None if limit is None else limit-consumed


def tool_schemas(scene='baseline_counter'):
    from roboquest.harness.profiles import SCENE_PROFILES
    # RoboQuestAdapter publishes no countdown (spec 1.2: the fixed budget is never stated), so
    # its scenes are not pointed at a field their packets do not carry.
    counts_down = SCENE_PROFILES.get(scene, {}).get('adapter_class') != 'RoboQuestAdapter'
    ticks = {'type': 'integer', 'minimum': 1, 'maximum': 200,
             'description': 'Physical 20 Hz ticks, bounded by remaining_ticks.' if counts_down
                            else 'Physical 20 Hz ticks.'}
    gripper = {'type': 'number', 'minimum': -1, 'maximum': 1,
               'description': '-1 open, +1 close, 0 retain setpoint.'}
    vector = lambda n: {'type': 'array', 'items': {'type': 'number'},
                        'minItems': n, 'maxItems': n}
    definitions = [
        ('arm', 'Move toward an absolute world gripper-site pose. Defaults: 120 ticks, hold orientation and gripper. Command envelope x/y ±6 m, z 0.3 to 2.2 m; reachability is not guaranteed.',
         {'position_world_m': vector(3), 'quaternion_xyzw': vector(4),
          'gripper': gripper, 'ticks': ticks}, ['position_world_m']),
        ('base', 'Drive the mobile base in its body frame. Each normalized forward/left/yaw input is in [-0.5,0.5]; zero brakes. Defaults to 20 ticks.',
         {'velocity_body': vector(3), 'ticks': ticks}, ['velocity_body']),
        ('wait', 'Hold arm/base targets, optionally update gripper, and allow physical settling. Defaults to 20 ticks.',
         {'gripper': gripper, 'ticks': ticks}, []),
        ('stop', 'Declare the task finished and end immediately with no extra physical ticks.', {}, []),
    ]
    from roboquest.harness.profiles import completion_protocol
    if completion_protocol(scene) == 'physical_submit_v1':
        definitions[-1] = ('stop', 'Abandon the episode without physical submission; this fails the task.', {}, [])
    return [{'type': 'function', 'function': {
        'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties,
                       'required': required, 'additionalProperties': False}}}
        for name, description, properties, required in definitions]


def public_packet(observation, *, scene='baseline_counter', image_size=256):
    """Reproject the allowlist before any model-visible serialization."""
    if type(image_size) is not int or image_size not in (256, 512):
        raise ValueError('Policy image size must be 256 or 512')
    clean = policy_observation(observation['rgb'], observation['proprio'])
    clean['instruction'] = trusted_goal(scene)
    clean['contract_version'] = trusted_control_version(scene)
    for camera in CAMERAS:
        rgb = np.asarray(clean['rgb'][camera])
        if rgb.shape != (image_size, image_size, 3) or rgb.dtype != np.uint8:
            raise ValueError(f'Policy requires three native {image_size}px uint8 RGB images')
        clean['rgb'][camera] = rgb
    if any(len(clean['proprio'][key]) != 13 for key in ('joint_positions', 'joint_velocities')):
        raise ValueError('Policy requires exactly 13 robot and gripper coordinates')
    # remaining_ticks is optional: a scene whose fixed budget is never disclosed (spec 1.2)
    # publishes no countdown, and the field is then simply unknown to the policy. It is left
    # out of the packet rather than filled in, so no substitute value can stand for the budget.
    for key, optional in (('step', False), ('remaining_ticks', True)):
        value = observation.get(key)
        if value is None and optional:
            continue
        if type(value) is not int or value < 0:
            raise ValueError('Invalid physical budget')
        clean[key] = value
    elapsed = observation['sim_time_s']
    if isinstance(elapsed, bool) or not isinstance(elapsed, (float, int)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid physical time')
    clean['sim_time_s'] = elapsed
    if type(observation['episode_ended']) is not bool:
        raise ValueError('Invalid episode status')
    clean['episode_ended'] = observation['episode_ended']
    return clean


def observation_message(packet, remaining_decisions, remaining_attempts, *, scene='baseline_counter', image_size=256,
                        goal_in_system=True):
    """One observation as a user message: the public packet text (goal omitted when the system message
    carries it, proprioception at millimetre/milliradian precision) followed by the labelled camera images."""
    clean = public_packet(packet, scene=scene, image_size=image_size)
    text = {key: value for key, value in clean.items() if key != 'rgb'}
    if goal_in_system:
        text.pop('instruction', None)
        text.pop('contract_version', None)
    text['proprio'] = _rounded(text['proprio'], 3)
    text['sim_time_s'] = _rounded(text.get('sim_time_s'), 2)
    text['remaining_model_responses'] = remaining_decisions
    text['remaining_http_attempts'] = remaining_attempts
    parts = [{'type': 'text', 'text': json.dumps(text, allow_nan=False)}]
    for camera in CAMERAS:
        data = BytesIO()
        Image.fromarray(clean['rgb'][camera]).save(data, format='PNG')
        parts += [{'type': 'text', 'text': 'Camera: ' + camera},
                  {'type': 'image_url', 'image_url': {
                      'url': 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode('ascii')}}]
    return {'role': 'user', 'content': parts}


class _ContinuationObserver(httpx.BaseTransport):
    """Recognize narrowly defined continuation errors without persisting bodies."""
    def __init__(self, inner, wire):
        self.inner, self.wire = inner, wire
        self.failure = None

    def handle_request(self, request):
        self.failure = None
        response = self.inner.handle_request(request)
        response.read()
        if response.status_code == 400:
            try:
                error = response.json().get('error', {})
                code = error.get('code', error.get('type')) if isinstance(error, dict) else None
                recognized = {'invalid_encrypted_content', 'invalid_previous_response_id',
                              'previous_response_not_found', 'invalid_thinking_signature'}
                if code in recognized:
                    self.failure = code
                elif self.wire == 'messages' and isinstance(error, dict):
                    message = str(error.get('message', '')).lower()
                    if all(word in message for word in ('invalid', 'signature', 'thinking')):
                        self.failure = 'invalid_thinking_signature'
            except (ValueError, AttributeError, TypeError):
                pass
        return response

    def close(self):
        self.inner.close()


class TableSettingPolicy:
    """One successful provider response is one decision, including invalid tools."""
    def __init__(self, model, base_url, api_key='', *, transport=None, max_decisions=12,
                 max_attempts=24, max_request_attempts=6, max_restarts=2,
                 timeout_s=120., capture=None, scene='baseline_counter', image_size=256,
                 verify_model_identity=False, retry_backoff_s=0., honor_retry_after=False):
        if model not in MODEL_CONFIGS:
            raise ValueError('Use an explicitly registered model name')
        validate_endpoint(base_url)
        trusted_goal(scene)
        self.scene = scene
        if type(image_size) is not int or image_size not in (256, 512):
            raise ValueError('Policy image size must be 256 or 512')
        self.image_size = image_size
        if any(n is not None and (type(n) is not int or n < 1)
               for n in (max_decisions, max_attempts)):
            raise ValueError('Decision and total attempt limits must be positive integers or None')
        if type(max_request_attempts) is not int or max_request_attempts < 1:
            raise ValueError('Request attempt limit must be a positive integer')
        if type(max_restarts) is not int or max_restarts < 0:
            raise ValueError('Restart limit must be a nonnegative integer')
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Timeout must be positive')
        from inspect_robots_agent._llm import ChatClient, Provider
        from inspect_robots_agent._responses import ResponsesClient
        from inspect_robots_agent._anthropic import AnthropicClient
        self.model, self.config = model, dict(MODEL_CONFIGS[model])
        self.verify_model_identity = verify_model_identity
        self.returned_model_ids = []
        self.max_decisions, self.max_restarts = max_decisions, max_restarts
        self.raw_transport = _ContinuationObserver(
            transport if transport is not None else httpx.HTTPTransport(trust_env=False), self.config['wire'])
        # Default: retry a transient error at once (the company-API convention). A provider with per-minute
        # limits (OpenRouter) gets an exponential wait and honours its reset time; the wait only stretches
        # wall time, never the episode's decisions.
        self.transport = RetryingGenerationTransport(
            self.raw_transport, budget=max_attempts, max_request_attempts=max_request_attempts,
            backoff_s=retry_backoff_s, honor_retry_after=honor_retry_after)
        self.transport.capture = capture
        self.api_model = api_model(model)
        provider = Provider(base_url=base_url, api_key=api_key, model=self.api_model, wire=self.config['wire'])
        kwargs = {'transport': self.transport, 'max_retries': 1, 'backoff_s': 0.,
                  'timeout_s': timeout_s, 'capture': None}
        if self.config['wire'] == 'messages' and 'thinking' in self.config:
            from roboquest.harness.anthropic_parameters import AnthropicParametersTransport
            kwargs['transport'] = AnthropicParametersTransport(
                self.transport, model=self.api_model, thinking=self.config['thinking'],
                output_config=self.config['output_config'])
        if self.config['wire'] == 'responses' and 'provider_routing' in self.config:
            from roboquest.harness.openrouter_parameters import OpenRouterRoutingTransport
            kwargs['transport'] = OpenRouterRoutingTransport(
                self.transport, model=self.api_model, routing=self.config['provider_routing'])
        if self.config['wire'] == 'responses' and self.config.get('cache_breakpoints'):
            from roboquest.harness.responses_cache import ResponsesCacheTransport
            kwargs['transport'] = ResponsesCacheTransport(
                kwargs['transport'], model=self.api_model, breakpoints=self.config['cache_breakpoints'])
        if self.config['wire'] == 'chat':
            from roboquest.harness.chat_parameters import ChatParametersTransport
            kwargs['transport'] = ChatParametersTransport(kwargs['transport'],
                                                          max_output_tokens=self.config['max_output_tokens'])
        if self.config['wire'] == 'gemini':
            from roboquest.harness.gemini_native import GeminiClient
            self.client = GeminiClient(provider, max_output_tokens=self.config['max_output_tokens'],
                                       thought_signature_window=self.config.get('thought_signature_window'), **kwargs)
        else:
            self.client = (ResponsesClient(provider, **kwargs) if self.config['wire'] == 'responses'
                           else ChatClient(provider, **kwargs) if self.config['wire'] == 'chat'
                           else AnthropicClient(provider, max_output_tokens=self.config['max_output_tokens'], **kwargs))
        self.system_prompt = policy_system_prompt(max_decisions, scene, include_goal=True)
        self.messages = [{'role': 'system', 'content': self.system_prompt}]
        self.decisions = 0
        self.restarts, self.turns, self.usage = [], [], []
        self.archived_messages = []
        self.pending_call_ids = []

    def _restart(self, observation):
        if len(self.restarts) >= self.max_restarts or self.decisions == 0:
            return False
        self.restarts.append({'after_decision': self.decisions,
                              'http_attempt': self.transport.attempts,
                              'physical_step': observation['step'],
                              'reason': self.raw_transport.failure})
        self.archived_messages.append(deepcopy(self.messages))
        self.messages = [{'role': 'system', 'content': self.system_prompt},
                         observation_message(observation, _remaining_budget(self.max_decisions, self.decisions),
                                             _remaining_budget(self.transport.budget, self.transport.attempts),
                                             scene=self.scene, image_size=self.image_size)]
        for attribute in ('_raw_items_by_call_id', '_raw_blocks_by_tool_use_id'):
            if hasattr(self.client, attribute):
                getattr(self.client, attribute).clear()
        self.transport.interruption = None
        self.pending_call_ids = []
        return True

    def _terminal_native_response(self):
        response = self.transport.last_response
        if response is None or response.status_code != 200:
            return None
        if self.config['wire'] == 'messages':
            from roboquest.harness.inspect_agent import _native_terminal_payload
            payload = _native_terminal_payload(response)
            return payload['stop_reason'] if payload is not None else None
        if self.config['wire'] == 'gemini':
            from roboquest.harness.gemini_native import terminal_reason
            try:
                return terminal_reason(response.json())
            except (ValueError, TypeError, KeyError):
                return None
        try:
            payload = response.json()
        except (ValueError, UnicodeError):
            return None
        if not isinstance(payload, dict):
            return None
        if self.config['wire'] == 'chat':
            choices = payload.get('choices')
            first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
            return {'length': 'max_tokens', 'content_filter': 'refusal'}.get(first.get('finish_reason'))
        output = payload.get('output')
        if not isinstance(output, list):
            return None
        if (payload.get('status') == 'incomplete' and isinstance(payload.get('output'), list)
                and isinstance(payload.get('incomplete_details'), dict)):
            return 'incomplete'
        if payload.get('status') == 'completed' and any(
                isinstance(item, dict) and item.get('type') == 'message'
                and isinstance(item.get('content'), list) and any(
                    isinstance(part, dict) and part.get('type') == 'refusal'
                    and isinstance(part.get('refusal'), str)
                    for part in item['content']) for item in output):
            return 'refusal'
        return None

    def decide(self, observation):
        if self.max_decisions is not None and self.decisions >= self.max_decisions:
            raise RuntimeError('Model response budget exhausted')
        if self.pending_call_ids:
            raise RuntimeError('Tool feedback must be recorded before another decision')
        self.messages.append(observation_message(
            observation, _remaining_budget(self.max_decisions, self.decisions),
            _remaining_budget(self.transport.budget, self.transport.attempts),
            scene=self.scene, image_size=self.image_size))
        # Keep the most recent two physical observations' images; all allowed
        # text, previous model messages and tool feedback remain in history.
        image_turns = [m for m in self.messages if m.get('role') == 'user'
                       and isinstance(m.get('content'), list)
                       and any(p.get('type') == 'image_url' for p in m['content'])]
        for message in image_turns[:-2]:
            message['content'] = [part for part in message['content'] if part.get('type') != 'image_url']
        mark_cache_anchors(self.messages)
        started = time.monotonic()
        first_attempt = self.transport.attempts
        while True:
            try:
                message = self.client.complete(self.messages, tool_schemas(self.scene),
                                               reasoning_effort=self.config['effort'])
                payload = self.transport.last_response.json()
                model_field = 'modelVersion' if self.config['wire'] == 'gemini' else 'model'
                returned_model = payload.get(model_field) if isinstance(payload, dict) else None
                if self.verify_model_identity and returned_model not in (
                        self.model, self.api_model, *self.config.get('accepted_returned_models', [])) and not (
                        # OpenAI-format servers may answer with a dated snapshot of the requested id
                        self.config['wire'] in ('chat', 'responses') and isinstance(returned_model, str)
                        and returned_model.startswith(self.api_model)):
                    self.transport.interrupt('model_identity_mismatch')
                if isinstance(returned_model, str) and returned_model not in self.returned_model_ids:
                    self.returned_model_ids.append(returned_model)
                if self.config['wire'] == 'gemini':
                    from roboquest.harness.gemini_native import terminal_reason
                    terminal = terminal_reason(payload)
                    break
                if (not isinstance(payload, dict)
                        or not all(isinstance(payload.get(key), str) and payload[key] for key in ('id', 'model'))):
                    raise ValueError('Invalid provider envelope')
                if self.config['wire'] == 'responses':
                    if payload.get('status') not in ('completed', 'incomplete') or not isinstance(payload.get('output'), list):
                        raise ValueError('Invalid Responses envelope')
                elif self.config['wire'] == 'chat':
                    if not isinstance(payload.get('choices'), list) or not payload['choices']:
                        raise ValueError('Invalid Chat Completions envelope')
                elif payload.get('type') != 'message' or payload.get('role') != 'assistant':
                    raise ValueError('Invalid Messages envelope')
                terminal = self._terminal_native_response()
                break
            except GenerationInterrupted as error:
                if (error.reason == 'permanent_provider_error'
                        and self.raw_transport.failure is not None
                        and (self.transport.budget is None or self.transport.attempts < self.transport.budget)
                        and self._restart(observation)):
                    continue
                raise
            except (ValueError, KeyError, TypeError, RuntimeError, AttributeError, IndexError) as error:
                terminal = self._terminal_native_response()
                if terminal is not None:
                    message = None
                    break
                self.transport.interrupt('invalid_provider_response', error_type=type(error).__name__)
        self.decisions += 1
        self.transport.accepted_responses += 1
        payload = self.transport.last_response.json()
        usage_key = 'usageMetadata' if self.config['wire'] == 'gemini' else 'usage'
        self.usage.append(deepcopy(payload.get(usage_key, {})))
        turn = {'decision': self.decisions, 'step': observation['step'],
                'http_attempts': self.transport.attempts-first_attempt,
                'generation_wall_s': time.monotonic()-started, 'model_stop': terminal,
                'command': None, 'validation_error': None}
        if terminal is not None:
            # Terminal/truncated output never executes a possibly partial tool.
            self.turns.append(turn)
            return turn
        self.messages.append(message.raw())
        self.pending_call_ids = [call.id for call in message.tool_calls]
        if not message.tool_calls:
            # A completed response without actions ends the agent run. The task
            # evaluator decides whether this ending satisfies its own contract.
            turn['model_stop'] = 'final_answer'
        elif len(message.tool_calls) != 1:
            turn['validation_error'] = 'Choose exactly one action tool per response.'
        else:
            call = message.tool_calls[0]
            try:
                arguments = json.loads(call.arguments)
                schema = next(t['function']['parameters'] for t in tool_schemas(self.scene)
                              if t['function']['name'] == call.name)
                if (not isinstance(arguments, dict) or set(arguments)-set(schema['properties'])
                        or not set(schema['required']) <= set(arguments)):
                    raise ValueError('invalid tool fields')
                turn['command'] = {'type': call.name, **arguments}
            except (ValueError, TypeError, StopIteration):
                turn['validation_error'] = 'Invalid action tool or arguments; check its schema.'
        self.turns.append(turn)
        return turn

    def feedback(self, accepted, observation, error=None):
        # Feedback cannot accept private score/state; error is a fixed generic
        # message selected by the orchestrator, never a simulator exception.
        clean = public_packet(observation, scene=self.scene, image_size=self.image_size)
        report = {'command_accepted': bool(accepted), 'step': clean['step']}
        if 'remaining_ticks' in clean:      # absent where the budget is never disclosed (spec 1.2)
            report['remaining_ticks'] = clean['remaining_ticks']
        report['error'] = 'Invalid action; check schema and remaining budget.' if error else None
        text = json.dumps(report)
        if self.pending_call_ids:
            self.messages += [{'role': 'tool', 'tool_call_id': call_id, 'content': text}
                              for call_id in self.pending_call_ids]
        else:
            self.messages.append({'role': 'user', 'content': text})
        self.pending_call_ids = []

    def close(self):
        self.client.close()


def run_policy_trial(adapter, policy, *, on_turn=None):
    """Evaluator-owned orchestrator. API/runtime interruptions have no task score."""
    started = time.monotonic()
    physical_wall_s = generation_wall_s = 0.
    native_output_interrupted = False
    result = {'mode': 'model_diagnostic', 'completed': False, 'score_is_official': False,
              'score': None, 'termination': None}
    try:
        observation = adapter.observe()
        while (not observation['episode_ended']
               and (policy.max_decisions is None or policy.decisions < policy.max_decisions)):
            generation_started = time.monotonic()
            try:
                turn = policy.decide(observation)
            finally:
                generation_wall_s += time.monotonic()-generation_started
            if turn['model_stop'] is not None:
                adapter.execute({'type': 'stop'})
                result['termination'] = 'provider_' + turn['model_stop']
                native_output_interrupted = turn['model_stop'] not in ('refusal', 'final_answer')
                result['termination_class'] = ('native_output_interrupted' if native_output_interrupted
                                               else 'agent_finished' if turn['model_stop'] == 'final_answer'
                                               else 'provider_refusal')
                break
            accepted = False
            action_started = time.monotonic()
            if turn['command'] is not None:
                try:
                    observation = adapter.execute(turn['command'])
                    accepted = True
                except (ValueError, KeyError):
                    turn['validation_error'] = 'Invalid action parameters or remaining physical budget.'
            physical_wall_s += time.monotonic()-action_started
            policy.feedback(accepted, observation, turn['validation_error'])
            turn['command_accepted'] = accepted
            turn['end_step'] = observation['step']
            if on_turn is not None:
                on_turn(deepcopy(turn))
        if not adapter.observe()['episode_ended']:
            adapter.execute({'type': 'stop'})
            result['termination'] = result['termination'] or 'model_response_budget'
        result['termination'] = result['termination'] or adapter.termination
        if hasattr(policy, 'check_interruption'):
            policy.check_interruption()
        if not native_output_interrupted:
            score = adapter.score()
            if score.get('scored') is not True:
                raise RuntimeError('Terminal evaluator unavailable')
            result.update(completed=True, score_is_official=True, score=score)
    except GenerationInterrupted as error:
        result.update(termination='api_interrupted', api_interruption=error.as_dict())
    except Exception as error:
        result.update(termination='runtime_interrupted', error_type=type(error).__name__)
    result.update(successful_model_decisions=policy.decisions,
                  generation_attempts=policy.transport.attempts,
                  physical_steps=adapter.steps,
                  simulated_seconds=float(adapter.observe()['sim_time_s']),
                  conversation_restarts=deepcopy(policy.restarts),
                  generation_wall_s=generation_wall_s,
                  http_attempt_wall_s=sum(event['duration_s'] for event in policy.transport.events),
                  physical_execution_wall_s=physical_wall_s,
                  total_wall_s=time.monotonic()-started)
    return result
