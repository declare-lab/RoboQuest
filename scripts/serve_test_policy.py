#!/usr/bin/env python3
"""A stand-in policy server speaking openpi's WebSocket protocol, for checking the wiring of --agent policy.

    bash scripts/run_agent.sh scripts/serve_test_policy.py --port 8000

Like openpi's ``serve_policy``, it sends its metadata when a client connects, then answers every MessagePack
request with a chunk of actions. Each request is checked against what a RoboQuest episode sends (the three
cameras as HxWx3 uint8 images, the 16-D state, the goal prompt); the reply is ``--chunk`` native 12-D actions that
keep the robot still in arm mode with the gripper open, so an episode runs to its tick budget. Ctrl-C stops it.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roboquest.harness.openpi import pack_array, unpack_array  # noqa: E402

EXPECTED = {'observation/image': 3, 'observation/wrist_image': 3, 'observation/right_image': 3,
            'observation/state': 1}


def check(request):
    """Problems with one request, as text (empty when it matches the RoboQuest observation)."""
    problems = []
    for key, ndim in EXPECTED.items():
        value = request.get(key)
        if not isinstance(value, np.ndarray) or value.ndim != ndim:
            problems.append(f'{key}: {type(value).__name__} {getattr(value, "shape", "")}')
        elif ndim == 3 and (value.dtype != np.uint8 or value.shape[2] != 3):
            problems.append(f'{key}: {value.dtype} {value.shape}, expected HxWx3 uint8')
        elif ndim == 1 and value.shape != (16,):
            problems.append(f'{key}: shape {value.shape}, expected (16,)')
    if not isinstance(request.get('prompt'), str) or not request['prompt'].strip():
        problems.append('prompt: missing or empty')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--chunk', type=int, default=20, help='actions per reply (default 20)')
    args = parser.parse_args()
    import msgpack
    from websockets.sync.server import serve

    still = np.zeros((args.chunk, 12), np.float32)
    still[:, 6] = -1.0      # gripper open
    still[:, 11] = -1.0     # arm mode (base still)
    metadata = {'policy': 'roboquest-test-policy', 'action_dim': 12, 'action_horizon': args.chunk}

    def handler(connection):
        connection.send(msgpack.packb(metadata, default=pack_array))
        served = 0
        for message in connection:
            request = msgpack.unpackb(message, object_hook=unpack_array)
            problems = check(request)
            if problems:
                connection.send('request does not match the RoboQuest observation: ' + '; '.join(problems))
                continue
            if served == 0:
                shapes = {k: getattr(v, 'shape', type(v).__name__) for k, v in request.items()}
                print(f'first request ok: {shapes}', flush=True)
            served += 1
            started = time.monotonic()
            connection.send(msgpack.packb({'actions': still, 'server_timing': {
                'infer_ms': (time.monotonic() - started) * 1000}}, default=pack_array))
        print(f'client disconnected after {served} requests', flush=True)

    with serve(handler, args.host, args.port, compression=None, max_size=None) as server:
        print(f'test policy serving on ws://{args.host}:{args.port}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
