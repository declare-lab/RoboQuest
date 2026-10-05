"""Public filesystem action boundary for an evaluated conversational sub-agent.

Only allowlisted observations leave this boundary. The sub-agent is instructed
to use this directory exclusively; this is not an OS sandbox for its other tools.
"""
from copy import deepcopy
import json
from pathlib import Path
import time
from types import SimpleNamespace

from PIL import Image

from roboquest.harness.agent import SYSTEM_PROMPT, public_packet, tool_schemas


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


class SubagentPolicy:
    def __init__(self, public_dir, *, scene, image_size=512, max_decisions=19,
                 idle_timeout_s=900., system_prompt=None):
        self.public_dir = Path(public_dir)
        self.public_dir.mkdir(parents=True, exist_ok=False)
        self.scene, self.image_size = scene, image_size
        self.max_decisions, self.idle_timeout_s = max_decisions, idle_timeout_s
        self.decisions = 0
        self.turns, self.messages, self.restarts = [], [], []
        # No external generation request is made by this adapter.
        self.transport = SimpleNamespace(attempts=0, events=[])
        self._ready_at, self._feedback = None, None
        (self.public_dir/'instructions.txt').write_text((system_prompt or SYSTEM_PROMPT)+'\n')
        atomic_json(self.public_dir/'tools.json', tool_schemas(self.scene))

    def _publish(self, observation):
        clean = public_packet(observation, scene=self.scene, image_size=self.image_size)
        folder = self.public_dir/f'observation-{self.decisions:02d}'
        folder.mkdir(exist_ok=True)
        images = {}
        for camera, rgb in clean.pop('rgb').items():
            path = folder/(camera+'.png')
            Image.fromarray(rgb).save(path)
            images[camera] = str(path)
        packet = {**clean, 'images': images, 'decision_count': self.decisions,
                  'remaining_decisions': (self.max_decisions-self.decisions
                                          if self.max_decisions is not None else None),
                  'next_decision': self.decisions+1 if not clean['episode_ended']
                      and (self.max_decisions is None or self.decisions < self.max_decisions) else None,
                  'feedback': self._feedback}
        atomic_json(folder/'packet.json', packet)
        atomic_json(self.public_dir/'observation.json', packet)
        self.messages.append({'role': 'observation', 'packet': deepcopy(packet)})
        self._ready_at = time.monotonic()
        return packet

    def decide(self, observation):
        self.start(observation)
        number = self.decisions+1
        path = self.public_dir/f'request-{number:02d}.json'
        # Service-owned control channel; never included in policy packets.
        self.check_interruption()
        while not path.is_file():
            self.check_interruption()
            if time.monotonic()-self._ready_at > self.idle_timeout_s:
                raise TimeoutError('Sub-agent action deadline exceeded')
            time.sleep(.1)
        started = time.monotonic()
        command, validation, intent = None, None, None
        try:
            request = json.loads(path.read_text())
            if (not isinstance(request, dict) or set(request)-{'decision', 'command', 'intent'}
                    or type(request.get('decision')) is not int or request['decision'] != number
                    or not isinstance(request.get('command'), dict)):
                raise ValueError()
            command = request['command']
            intent = request.get('intent')
            if intent is not None and (not isinstance(intent, str) or len(intent) > 1000):
                raise ValueError()
            schema = next(t['function']['parameters'] for t in tool_schemas(self.scene)
                          if t['function']['name'] == command.get('type'))
            if (set(command)-{'type'}-set(schema['properties'])
                    or not set(schema['required']) <= set(command)):
                raise ValueError()
        except (ValueError, TypeError, StopIteration):
            command, validation = None, 'Invalid action; check schema and remaining budget.'
        self.decisions = number
        turn = {'decision': number, 'step': observation['step'], 'http_attempts': 0,
                'generation_wall_s': started-self._ready_at,
                'timing_scope': 'Observation-ready to action-file arrival; includes Codex/tool/relay overhead.',
                'model_stop': None, 'command': command, 'intent': intent,
                'validation_error': validation}
        self.turns.append(turn)
        self.messages.append({'role': 'action', 'decision': number,
                              'command': deepcopy(command), 'intent': intent})
        return turn

    def start(self, observation):
        """Publish once so a CLI can attach after a concrete ready signal."""
        if self._ready_at is None:
            self._publish(observation)

    def check_interruption(self):
        if (self.public_dir/'abort.json').exists():
            raise RuntimeError('Episode cancelled by client or operator')

    def feedback(self, accepted, observation, error=None):
        self._feedback = {'command_accepted': bool(accepted), 'step': observation['step']}
        if observation.get('remaining_ticks') is not None:   # absent where the budget is never disclosed
            self._feedback['remaining_ticks'] = observation['remaining_ticks']
        self._feedback['error'] = 'Invalid action; check schema and remaining budget.' if error else None
        if (not observation['episode_ended']
                and (self.max_decisions is None or self.decisions < self.max_decisions)):
            packet = self._publish(observation)
            atomic_json(self.public_dir/f'response-{self.decisions:02d}.json', packet)

    def finish(self, observation):
        packet = self._publish(observation)
        if self.decisions:
            atomic_json(self.public_dir/f'response-{self.decisions:02d}.json', packet)
        atomic_json(self.public_dir/'finished.json', {'episode_ended': True,
                    'decisions': self.decisions, 'score_available_to_policy': False})
