"""Run a task with API or external CLI control through one episode pipeline."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from roboquest.harness.episode_tasks import resolve_task, task_names, reference_status, http_task_scenes
from roboquest.harness.agent import MODEL_CONFIGS
from roboquest.harness.profiles import adapter_parameters, resolve_protocol, completion_protocol

ROOT = Path(__file__).resolve().parents[1]
# --agent policy: one tick budget for every task, 30 minutes of simulated time at 20 Hz (above every task's
# provisional BUDGET_TICKS and about twice the slowest scripted solve)
POLICY_HORIZON = 36_000
# --model <provider>/<model id>: the provider fixes the API format (the harness's wire protocol), the endpoint and
# the variable the key is read from
PROVIDERS = {
    'openai': ('responses', 'https://api.openai.com/v1', 'OPENAI_API_KEY'),
    'openrouter': ('chat', 'https://openrouter.ai/api/v1', 'OPENROUTER_API_KEY'),
    'anthropic': ('messages', 'https://api.anthropic.com/v1', 'ANTHROPIC_API_KEY'),
    'google': ('gemini', None, 'GEMINI_API_KEY'),
    # Google Cloud Vertex AI: the service-account file named by GOOGLE_APPLICATION_CREDENTIALS instead of a key
    'vertex/anthropic': ('messages', None, None),
    'vertex/google': ('gemini', None, None),
    # any OpenAI Chat Completions server at --base-url: vLLM, SGLang, Ollama, Together, Fireworks, a LiteLLM proxy, ...
    'openai-compatible': ('chat', None, 'OPENAI_API_KEY'),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list-tasks', action='store_true')
    parser.add_argument('--task', help='Registered task name, or roboquest_<task>@<instance_id> for one '
                        'development instance of a suite registry; --list-tasks prints the names')
    parser.add_argument('--evaluation-instances', action='store_true', help='Let --task name an instance of the '
                        'evaluation split too: the benchmark evaluation itself, never a development run')
    parser.add_argument('--agent', choices=('api', 'policy', 'codex'), default='codex',
                        help='api: a model through its API (--model); policy: a robot policy served over the '
                        'openpi WebSocket protocol (--policy-url); codex: a public file bridge usable by CLI agents')
    parser.add_argument('--server', help='Run the API agent against the shared HTTP simulator service')
    parser.add_argument('--token-file', type=Path, help='HTTP simulator token; separate from model provider credentials')
    parser.add_argument('--token-env', default='ROBOQUEST_TOKEN')
    parser.add_argument('--simulator-timeout-s', type=float, default=30.)
    parser.add_argument('--simulator-wait-timeout-s', type=float, default=300.)
    parser.add_argument('--model', help='Required for API runs: <provider>/<model id>, the provider one of '
                        + ', '.join(PROVIDERS) + ' (README.md, Connecting a model)')
    parser.add_argument('--effort', help='Reasoning effort to request (default: none sent)')
    parser.add_argument('--max-output-tokens', type=int, help='Output token limit (default: the provider\'s; '
                        '16000 for Anthropic and 32768 for Gemini, which need one)')
    parser.add_argument('--agent-model-label', help='Optional, unverified external CLI model label')
    parser.add_argument('--out', type=Path, help='Fresh private evidence directory; generated if omitted')
    parser.add_argument('--public-dir', type=Path, help='Fresh Codex session directory outside the repository')
    parser.add_argument('--policy-url', help='--agent policy: the policy server, e.g. ws://localhost:8000 '
                        '(openpi serve_policy, or scripts/serve_test_policy.py)')
    parser.add_argument('--policy-timeout-s', type=float, default=180., help='--agent policy: seconds to wait for '
                        'the server to accept the connection, and for each inference')
    parser.add_argument('--policy-seed', type=int, default=0, help='--agent policy: evaluation seed; every '
                        'inference sends a sampling seed derived from it, the instance and the inference index')
    parser.add_argument('--replan-every', type=int, help='--agent policy: execute this many actions of each chunk '
                        'before asking again (default: the whole chunk)')
    parser.add_argument('--require-subtask', action='store_true', help='--agent policy: fail when the server '
                        'returns no predicted subtask (joint subtask and action models)')
    parser.add_argument('--action-filter', choices=('none', 'demos'), default='none', help='--agent policy: none '
                        'executes each action within the controller range (arm and base together allowed); demos '
                        'reshapes it like RoboQuest demonstrations (arm or base per tick, open/close gripper, the '
                        'per-tick speed limits of the API agents)')
    parser.add_argument('--physical-gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, help='Scene seed override. Default: the instance keeps its own '
                        'seed (the scene its gates were run on)')
    parser.add_argument('--image-size', type=int, choices=(256, 512), help='Camera images in pixels (default 512; '
                        '256 for --agent policy, the size the openpi RoboQuest policies are trained on)')
    parser.add_argument('--asset-scene', type=Path)
    parser.add_argument('--shoe-manifest', type=Path)
    parser.add_argument('--horizon', type=int, help='Physical ticks at 20 Hz (default 1000000 for API and CLI '
                        'agents, whose decision cap ends an episode; the task budget for --agent policy)')
    parser.add_argument('--max-decisions', type=int, help='Optional response cap; unlimited by default')
    parser.add_argument('--max-attempts', type=int, help='Optional total API attempt cap')
    parser.add_argument('--max-request-attempts', type=int, default=6)
    parser.add_argument('--retry-backoff-s', type=float, default=0., help='Exponential wait before each retry of '
                        'a transient provider error (429, 5xx, transport); 0 retries at once, the default')
    parser.add_argument('--honor-retry-after', action='store_true', help='Wait at least the delay the provider '
                        'asks for (Retry-After, or the reset time of a per-minute limit) before retrying')
    parser.add_argument('--max-restarts', type=int, default=2)
    parser.add_argument('--request-timeout-s', type=float, default=600.)
    parser.add_argument('--idle-timeout-s', type=float, default=900., help='External agent action deadline')
    parser.add_argument('--base-url', help='Endpoint of an openai-compatible/ model, e.g. http://localhost:8000/v1 '
                        '(the other providers have fixed endpoints)')
    parser.add_argument('--api-key-env', help='Variable the key is read from (default: the '
                        'provider key, e.g. OPENROUTER_API_KEY)')
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument('--env-file', type=Path, help='Private KEY=VALUE file the key (or GOOGLE_APPLICATION_CREDENTIALS) '
                      'is read from; default: the environment')
    auth.add_argument('--key-file', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--auth-python', default='/usr/bin/python3')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare-only', action='store_true', help='Write config/source snapshot without GPU or API use')
    modes.add_argument('--offline', action='store_true', help='Mock provider responses; --server still uses simulator HTTP')
    args = parser.parse_args(argv)
    if args.list_tasks:
        return args
    if not args.task:
        parser.error('--task is required')
    if args.server and (args.agent != 'api' or args.prepare_only):
        parser.error('--server currently requires --agent api and does not support --prepare-only')
    if args.token_file and not args.server:
        parser.error('--token-file requires --server')
    if args.server and (args.task not in http_task_scenes() or args.seed != 0
                       or args.image_size != 512 or args.horizon != 1_000_000
                       or args.asset_scene or args.shoe_manifest):
        parser.error('HTTP service requires a published task, seed 0, 512px and its server-selected assets/horizon')
    if args.agent == 'api' and not args.model:
        parser.error('--agent api requires --model')
    if args.agent == 'api':
        # --model <provider>/<model id>: the provider fixes the API format, the endpoint and the key variable
        args.provider = next((p for p in PROVIDERS if args.model.startswith(p + '/')), None)
        if args.provider is None:
            parser.error(f'--model {args.model}: start it with its provider, one of '
                         + ', '.join(p + '/' for p in PROVIDERS) + ' (for example anthropic/claude-opus-5-5)')
        wire, endpoint, args.provider_key_env = PROVIDERS[args.provider]
        # Anthropic requires an output limit and the native Gemini client needs one (thinking plus answer)
        MODEL_CONFIGS[args.model] = {
            'wire': wire, 'effort': args.effort, 'api_model': args.model[len(args.provider) + 1:],
            'max_output_tokens': args.max_output_tokens or {'messages': 16000, 'gemini': 32768}.get(wire)}
        if args.base_url and args.provider != 'openai-compatible':
            parser.error('--base-url applies to openai-compatible/ models; the other providers have fixed endpoints')
        if args.provider == 'openai-compatible' and not args.base_url and not (args.offline or args.prepare_only):
            parser.error('openai-compatible/ models need --base-url (for example http://localhost:8000/v1)')
        args.base_url = args.base_url or endpoint
        args.vertex = args.provider.startswith('vertex/')
        args.vertex_region = os.environ.get('GOOGLE_CLOUD_LOCATION') or 'global'   # Google Cloud's region variable
    elif args.effort or args.max_output_tokens:
        parser.error('--effort/--max-output-tokens apply to --agent api')
    if args.agent in ('codex', 'policy') and (args.model or args.offline):
        parser.error('--model/--offline apply to --agent api')
    if (args.agent == 'policy') != bool(args.policy_url):
        parser.error('--agent policy needs --policy-url, which applies to it only')
    if args.replan_every is not None and args.replan_every < 1:
        parser.error('--replan-every must be positive')
    # A policy acts on every tick: its episode ends at the Submit press or after POLICY_HORIZON ticks, one value
    # for every task; its cameras default to the 256 px the openpi RoboQuest policies are trained on.
    if args.image_size is None:
        args.image_size = 256 if args.agent == 'policy' else 512
    if args.horizon is None:
        args.horizon = POLICY_HORIZON if args.agent == 'policy' else 1_000_000
    if args.agent == 'codex' and args.max_attempts is not None:
        parser.error('--max-attempts applies to API requests only')
    if args.agent == 'api' and (args.public_dir or args.agent_model_label):
        parser.error('--public-dir/--agent-model-label apply to --agent codex')
    if (args.agent in ('codex', 'policy') or args.offline or args.prepare_only) and any(
            (args.env_file, args.key_file, args.api_key_env)):
        parser.error('Policy, codex, offline and prepare runs do not accept API authentication options')
    import math
    if not math.isfinite(args.idle_timeout_s) or args.idle_timeout_s <= 0:
        parser.error('--idle-timeout-s must be finite and positive')
    if args.physical_gpu < 0:
        parser.error('--physical-gpu must be nonnegative')
    try:
        if args.evaluation_instances and '@' in str(args.task):   # evaluation split only when named
            from roboquest.catalog import bind_instance_profile
            from roboquest.harness.profiles import SCENE_PROFILES
            bind_instance_profile(SCENE_PROFILES, args.task, ('dev', 'eval'))
        args.scene = resolve_task(args.task)
        # a registry instance keeps its own seed (None reaches the adapter), the default instance of a task too
        args.task_parameters = ({} if args.server else adapter_parameters(
            args.scene, asset_scene=args.asset_scene, shoe_manifest=args.shoe_manifest))
        protocol = 'physical_submit_v1' if completion_protocol(args.scene) == 'physical_submit_v1' else 'agent_stop_v1'
        limits = resolve_protocol(protocol, horizon=args.horizon,
            max_decisions=args.max_decisions, max_attempts=args.max_attempts,
            max_request_attempts=args.max_request_attempts, max_restarts=args.max_restarts,
            request_timeout_s=args.request_timeout_s)
        for key, value in limits.items():
            setattr(args, key, value)
    except ValueError as error:
        parser.error(str(error))
    run_id = f'{args.task}-{args.agent}-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex[:8]
    args.out = (args.out or ROOT/'artifacts'/'episodes'/run_id).resolve()
    if args.out.exists():
        parser.error('--out must be a new directory')
    if args.agent == 'codex':
        args.public_dir = (args.public_dir or Path('/tmp')/('roboquest-'+run_id)).resolve()
        if (args.public_dir.exists() or args.public_dir.is_relative_to(ROOT)
                or args.public_dir.is_relative_to(args.out) or args.out.is_relative_to(args.public_dir)):
            parser.error('Public directory must be fresh, outside the repository, and separate from private output')
    args.api_key_env = args.api_key_env or getattr(args, 'provider_key_env', None) or 'OPENAI_API_KEY'
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.list_tasks:
        from roboquest.harness.profiles import SCENE_PROFILES
        print(json.dumps([{'task': name, 'scene': resolve_task(name),
            'version': SCENE_PROFILES[resolve_task(name)]['scene_revision'],
            'reference_status': reference_status(name)}
            for name in task_names()], indent=2))
        return 0
    if args.server:
        raise SystemExit('The shared HTTP simulator service (--server) is not part of the RoboQuest release')
    from roboquest.harness.episode_runner import run_episode
    return run_episode(args, ROOT)


if __name__ == '__main__':
    raise SystemExit(main())
