"""Submit one numbered native robot action; duplicate submissions never replay it."""
import argparse
import json
import os
from pathlib import Path
import time


def submit(directory, number, command, intent=None, *, timeout_s=180.):
    root = Path(directory)
    request = {'decision': number, 'command': command, 'intent': intent}
    target = root/f'request-{number:02d}.json'
    if target.exists():
        if json.loads(target.read_text()) != request:
            raise ValueError('That decision already has a different action')
    else:
        packet = json.loads((root/'observation.json').read_text())
        if packet['episode_ended'] or packet['next_decision'] != number:
            raise ValueError('Use the current next_decision; the episode may have ended')
        temporary = root/f'.request-{number:02d}-{os.getpid()}.tmp'
        temporary.write_text(json.dumps(request, allow_nan=False)+'\n')
        try:
            os.link(temporary, target)  # Atomic create; never overwrite an action.
        finally:
            temporary.unlink()
    response = root/f'response-{number:02d}.json'
    started = time.monotonic()
    while not response.exists():
        if time.monotonic()-started > timeout_s:
            raise TimeoutError('Action response pending; retry the SAME decision and action to wait safely')
        time.sleep(.1)
    return json.loads(response.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decision', type=int, required=True)
    parser.add_argument('--action', required=True)
    parser.add_argument('--intent')
    args = parser.parse_args()
    print(json.dumps(submit(Path(__file__).resolve().parent, args.decision,
                            json.loads(args.action), args.intent), indent=2))


if __name__ == '__main__':
    main()
