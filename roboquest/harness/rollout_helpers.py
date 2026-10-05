"""Helpers the episode runner uses: the Inspect Robots source pin and the offline (mocked) provider."""
import json
from pathlib import Path
import subprocess


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def inspect_source_pin():
    import inspect_robots
    source = Path(inspect_robots.__file__).resolve().parents[2]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if revision != '7e4d1b7aee1c0d3cfc3a05a7492b9d12cda666f9':
        raise RuntimeError('Inspect source pin mismatch')
    return {'source_root': str(source), 'revision': revision, 'pin_verified': True}



class OfflineNativeSmoke:
    """Reads only public outgoing proprioception; no simulator reference."""
    def __init__(self, model, wire):
        self.model, self.wire, self.calls = model, wire, 0

    def __call__(self, request):
        import httpx
        payload = json.loads(request.content)
        packets = []
        def visit(value):
            if isinstance(value, dict):
                if (value.get('type') in ('text', 'input_text')
                        or self.wire == 'gemini' and isinstance(value.get('text'), str)):
                    try:
                        parsed = json.loads(value.get('text', ''))
                        if isinstance(parsed, dict) and 'proprio' in parsed:
                            packets.append(parsed)
                    except ValueError:
                        pass
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(payload)
        current = packets[-1]['proprio']
        target = list(current['eef_position_world_m'])
        target[2] += .04
        commands = [('arm', {'position_world_m': target, 'ticks': 25}),
                    ('base', {'velocity_body': [-.3, 0, 0], 'ticks': 12}),
                    ('base', {'velocity_body': [0, 0, 0], 'ticks': 10}),
                    ('wait', {'ticks': 10}), ('stop', {})]
        name, arguments = commands[min(self.calls, len(commands)-1)]
        call_id = 'offline_' + str(self.calls)
        self.calls += 1
        if self.wire == 'responses':
            response = {'id': 'resp_'+call_id, 'model': self.model, 'status': 'completed',
                        'output': [{'type': 'function_call', 'id': 'fc_'+call_id,
                                    'call_id': call_id, 'name': name,
                                    'arguments': json.dumps(arguments)}],
                        'usage': {'input_tokens': 0, 'output_tokens': 0}}
        elif self.wire == 'chat':
            response = {'id': 'chatcmpl-'+call_id, 'object': 'chat.completion', 'model': self.model,
                        'choices': [{'index': 0, 'finish_reason': 'tool_calls', 'message': {
                            'role': 'assistant', 'content': None, 'tool_calls': [{
                                'id': call_id, 'type': 'function',
                                'function': {'name': name, 'arguments': json.dumps(arguments)}}]}}],
                        'usage': {'prompt_tokens': 0, 'completion_tokens': 0}}
        elif self.wire == 'gemini':
            response = {'responseId': 'resp_'+call_id, 'modelVersion': self.model,
                        'candidates': [{'index': 0, 'finishReason': 'STOP',
                            'content': {'role': 'model', 'parts': [{
                                'functionCall': {'name': name, 'args': arguments},
                                'thoughtSignature': 'offline-signature'}]}}],
                        'usageMetadata': {'promptTokenCount': 0, 'candidatesTokenCount': 0}}
        else:
            response = {'id': 'msg_'+call_id, 'type': 'message', 'role': 'assistant',
                        'model': self.model, 'stop_reason': 'tool_use',
                        'content': [{'type': 'tool_use', 'id': call_id, 'name': name, 'input': arguments}],
                        'usage': {'input_tokens': 0, 'output_tokens': 0}}
        return httpx.Response(200, json=response)
