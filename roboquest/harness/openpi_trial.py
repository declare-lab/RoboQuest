"""Episode loop for a robot policy served over openpi's WebSocket protocol (``--agent policy``).

Ported from Tej's OpenPI evaluation harness (``scripts/roboquest_openpi_eval.py`` on trunk): every inference sends
the three cameras, the 16-D state and the task's goal (``roboquest/harness/openpi.py``) and receives a chunk of
native 12-D PandaOmron actions at 20 Hz. Each action is executed as one tick (clipped to the controller's range,
torso held; ``--action-filter demos`` applies Tej's sanitizer instead), then the policy is queried again with a
fresh observation, until the physical Submit press or the task's tick budget ends the episode. Recording, scoring and progress are the episode runner's, exactly as for API models.
"""
import time

import numpy as np

from roboquest.harness.openpi import (OpenPIWebsocketClient, PolicyOutputError, PolicyServerError,
                                      make_policy_observation, policy_seed, sanitize_native_action,
                                      validate_policy_response)


ACTION_FILTERS = ('none', 'demos')


def native_action(raw):
    """The policy's action as the controller takes it: every entry clipped to the controller's range [-1, 1], the
    torso held (as for every RoboQuest agent). Arm and base may move in the same tick; the gripper keeps its value
    (the Panda gripper closes on > 0, opens on < 0 and holds on 0); the mode entry chooses how the arm's target
    follows a moving base (> 0) or stays put (<= 0)."""
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape != (12,) or not np.isfinite(raw).all():
        raise PolicyOutputError(f'native policy action must be finite with shape (12,), got {raw.shape}')
    executed = np.clip(raw, -1.0, 1.0)
    executed[10] = 0.0
    held = np.delete(np.arange(12), 10)
    return executed, {'mode': 'base' if executed[11] > 0 else 'arm',
                      'clipped_elements': int(np.count_nonzero(executed[held] != raw[held])),
                      'active_elements': int(held.size),
                      'raw_outside_native_range': int(np.count_nonzero(np.abs(raw) > 1.0))}


def jsonable(value):
    """Server metadata and timings as JSON values (NumPy arrays and scalars become lists and numbers)."""
    if isinstance(value, dict):
        return {str(k.decode() if isinstance(k, bytes) else k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode(errors='replace')
    return value


class OpenPIPolicy:
    """A connection to one policy server plus what the episode records about it."""

    def __init__(self, url, *, timeout_s, image_size, eval_seed, instance_id, replan_every=None,
                 require_subtask=False, action_filter='none'):
        if action_filter not in ACTION_FILTERS:
            raise ValueError(f'action_filter must be one of {ACTION_FILTERS}')
        self.url, self.timeout_s, self.image_size = url, float(timeout_s), int(image_size)
        self.eval_seed, self.instance_id = int(eval_seed), str(instance_id)
        self.replan_every, self.require_subtask = replan_every, bool(require_subtask)
        # none: the controller's range only; demos: Tej's sanitizer, the action space of RoboQuest's
        # demonstrations (arm or base per tick, open/close gripper, the API agents' per-tick speed limits)
        self.filter = native_action if action_filter == 'none' else sanitize_native_action
        self.action_filter = action_filter
        self.client = self.metadata = None
        self.decisions = 0
        self.turns, self.latencies_s, self.subtask_transitions = [], [], []
        self.clipping = dict(clipped_elements=0, active_elements=0, raw_elements_outside_native_range=0)

    def connect(self):
        self.client = OpenPIWebsocketClient(self.url, None, timeout_s=self.timeout_s)
        self.metadata = jsonable(self.client.metadata)
        return self.metadata

    def close(self):
        if self.client is not None:
            self.client.close()


def run_openpi_trial(adapter, policy, *, on_inference=None):
    """Query, execute the chunk tick by tick, repeat until the episode ends; the runner's result fields."""
    started = time.monotonic()
    generation_wall_s = physical_wall_s = 0.
    result = {'mode': 'policy', 'completed': False, 'score_is_official': False, 'score': None, 'termination': None}
    try:
        packet = adapter.observe()
        current_subtask = None
        while not packet['episode_ended']:
            seed = policy_seed(policy.eval_seed, policy.instance_id, policy.decisions)
            request = make_policy_observation(packet, seed, image_size=policy.image_size)
            asked = time.monotonic()
            response = policy.client.infer(request, timeout_s=policy.timeout_s)
            latency_s = time.monotonic() - asked
            generation_wall_s += latency_s
            policy.latencies_s.append(latency_s)
            # any chunk length; validate_policy_response checks the 12-D rows and finite values
            rows = np.shape(response.get('actions')) if isinstance(response, dict) else ()
            if len(rows) != 2 or rows[0] < 1:
                raise PolicyOutputError(f'policy actions have shape {rows}, expected (chunk, 12)')
            raw_chunk, subtask = validate_policy_response(response, action_horizon=rows[0],
                                                          require_subtask=policy.require_subtask)
            policy.decisions += 1
            if subtask != current_subtask:
                policy.subtask_transitions.append({'tick': adapter.steps, 'subtask': subtask})
                current_subtask = subtask
            executed, diagnostics = [], []
            for raw in raw_chunk[:policy.replan_every or len(raw_chunk)]:
                action, info = policy.filter(raw)
                executed.append(action)
                diagnostics.append(info)
                policy.clipping['clipped_elements'] += info['clipped_elements']
                policy.clipping['active_elements'] += info['active_elements']
                policy.clipping['raw_elements_outside_native_range'] += info['raw_outside_native_range']
            start_step = adapter.steps
            acting = time.monotonic()
            for action in executed:
                packet = adapter.act_native(action)
                if packet['episode_ended']:
                    break
            physical_wall_s += time.monotonic() - acting
            turn = {'decision': policy.decisions, 'step': start_step, 'end_step': adapter.steps,
                    'policy_seed': seed, 'latency_s': round(latency_s, 4), 'subtask': subtask,
                    'actions_received': len(raw_chunk), 'actions_executed': adapter.steps - start_step,
                    'clipped_elements': sum(d['clipped_elements'] for d in diagnostics)}
            policy.turns.append(turn)
            if on_inference is not None:
                on_inference(turn, {**turn, 'raw_actions': raw_chunk.tolist(),
                                    'executed_actions': [a.tolist() for a in executed],
                                    'action_diagnostics': diagnostics,
                                    'policy_timing': jsonable(response.get('policy_timing')),
                                    'server_timing': jsonable(response.get('server_timing'))})
        result['termination'] = adapter.termination
        score = adapter.score()
        if score.get('scored') is not True:
            raise RuntimeError('Terminal evaluator unavailable')
        result.update(completed=True, score_is_official=True, score=score)
    except (PolicyOutputError, PolicyServerError) as error:
        # the policy or its server failed, not the task: unscored, like an interrupted API episode
        result.update(termination='policy_error', error_type=type(error).__name__)
    except Exception as error:
        result.update(termination='runtime_interrupted', error_type=type(error).__name__)
    latencies = policy.latencies_s
    result.update(successful_model_decisions=policy.decisions, policy_inferences=policy.decisions,
                  physical_steps=adapter.steps, simulated_seconds=float(adapter.observe()['sim_time_s']),
                  generation_wall_s=generation_wall_s, physical_execution_wall_s=physical_wall_s,
                  total_wall_s=time.monotonic() - started, action_clipping=dict(policy.clipping),
                  subtask_transitions=list(policy.subtask_transitions),
                  inference_latency_s={'mean': float(np.mean(latencies)) if latencies else None,
                                       'median': float(np.median(latencies)) if latencies else None,
                                       'max': float(np.max(latencies)) if latencies else None})
    return result
