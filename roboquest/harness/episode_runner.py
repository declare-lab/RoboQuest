"""One evaluator-owned lifecycle and recorder for API and external CLI policies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from roboquest.harness.agent import MODEL_CONFIGS, TableSettingPolicy, api_model, policy_system_prompt, run_policy_trial, tool_schemas
from roboquest.harness.profiles import SCENE_PROFILES, trusted_goal, verify_adapter_goal, completion_protocol


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def freeze_source(root, out):
    from roboquest.harness.rollout_helpers import inspect_source_pin
    sources = sorted((root/'roboquest').rglob('*.py'))
    sources += [root/'scripts'/name for name in (
        'run_episode.py', 'run_episode.sh', 'run_agent.sh', 'run_inspect.sh', 'subagent_action.py')]
    manifest = {}
    for source in sources:
        relative = source.relative_to(root)
        target = out/'source'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    save(out/'source-manifest.json', {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'source_sha256': manifest, 'inspect_pin': inspect_source_pin(),
        'scope': 'Exact local Python source snapshot, including uncommitted changes; private to evaluator.'})


def make_api_policy(args, config, capture):
    import httpx
    from roboquest.harness.agent_runtime import provider_environment, validate_endpoint
    from roboquest.harness.rollout_helpers import OfflineNativeSmoke
    wire = MODEL_CONFIGS[args.model]['wire']
    environment, transport = {}, None
    endpoint = args.base_url
    if args.offline:
        endpoint = 'https://gemini.invalid/v1' if wire == 'gemini' else 'http://offline.invalid/v1'
        transport = httpx.MockTransport(OfflineNativeSmoke(args.model, wire))
    elif args.vertex:
        # Google Cloud: the service-account file named by GOOGLE_APPLICATION_CREDENTIALS (environment or --env-file)
        from roboquest.harness.vertex_auth import VertexServiceAccount
        credentials = provider_environment('GOOGLE_APPLICATION_CREDENTIALS', None, args.env_file)
        auth = VertexServiceAccount(Path(credentials['GOOGLE_APPLICATION_CREDENTIALS']).expanduser(),
                                    auth_python=args.auth_python)
        if wire == 'gemini':       # a Google-published model through the project's Vertex publisher endpoint
            from roboquest.harness.vertex_transport import VertexGeminiTransport
            transport = VertexGeminiTransport(auth.project_id, api_model(args.model),
                region=args.vertex_region, token_provider=auth.token)
            endpoint = 'https://gemini.invalid/v1'
        else:
            from roboquest.harness.vertex_transport import VertexAnthropicTransport
            transport = VertexAnthropicTransport(auth.project_id, api_model(args.model),
                region=args.vertex_region, token_provider=auth.token)
            endpoint = 'https://vertex.invalid/v1'
        auth.token()
        config['vertex_region'] = args.vertex_region
        config['vertex_publisher'] = 'google' if wire == 'gemini' else 'anthropic'
    else:
        environment = provider_environment(args.api_key_env, args.key_file, args.env_file)
        if wire == 'gemini':
            from roboquest.harness.gemini_native import GeminiAPITransport
            transport = GeminiAPITransport(api_model(args.model), api_key=environment[args.api_key_env])
            endpoint = 'https://gemini.invalid/v1'
    validate_endpoint(endpoint)
    config['model_check'] = {'method': 'each_response_before_action', 'extra_generation_attempts': 0}
    from roboquest.harness.agent import OBSERVATION_TEXT_VERSION
    config['observation_text'] = OBSERVATION_TEXT_VERSION
    policy = TableSettingPolicy(args.model, endpoint,
        '' if wire == 'gemini' else environment.get(args.api_key_env, ''),
        transport=transport, capture=capture, scene=args.scene, image_size=args.image_size,
        max_decisions=args.max_decisions, max_attempts=args.max_attempts,
        max_request_attempts=args.max_request_attempts, max_restarts=args.max_restarts,
        timeout_s=args.request_timeout_s, verify_model_identity=True,
        retry_backoff_s=args.retry_backoff_s, honor_retry_after=args.honor_retry_after)
    config['http_timeouts_s'] = policy.client._http.timeout.as_dict()
    return policy


def make_cli_policy(args, root):
    from roboquest.harness.subagent_policy import SubagentPolicy
    policy = SubagentPolicy(args.public_dir, scene=args.scene, image_size=args.image_size,
        max_decisions=args.max_decisions, idle_timeout_s=args.idle_timeout_s,
        system_prompt=policy_system_prompt(args.max_decisions, args.scene))
    shutil.copy2(root/'scripts/subagent_action.py', args.public_dir/'act.py')
    (args.public_dir/'AGENTS.md').write_text(
        '# Robot policy session\n\n'
        'You are the robot policy. Read instructions.txt, tools.json and observation.json. '
        'View the actual camera PNGs referenced by each observation. Choose exactly one '
        'robot action using the current next_decision. Submit with:\n\n'
        "python3 act.py --decision NUMBER --action 'JSON_COMMAND'\n\n"
        'The command is a JSON object with type arm, base, wait or stop and the '
        'parameters in tools.json. After an action, inspect the returned observation '
        'and its images before choosing the next action. On client timeout, retry '
        'the SAME decision and action; never create a new number to replay it. '
        'Continue autonomously until episode_ended is true. Use stop when you decide '
        'the goal is complete if the task uses declared stop. For a physical SUBMIT task, '
        'press its button to finish; stop abandons it and fails. Stopping adds no settling.\n\n'
        'Use only this session directory and its provided action client. Do not read '
        'project files, simulator internals, private artifacts, prior rollouts or '
        'solutions. Do not edit the client or observations, reset the environment, '
        'access other cameras, call outside models, or delegate actions. '
        'This directory contains public observations only; the evaluator keeps the score private.\n')
    return policy


def save_initial(adapter, out):
    import xml.etree.ElementTree as ET
    save(out/'initial-private.json', {'qpos': adapter.initial_qpos.tolist(),
        'qvel': adapter.initial_qvel.tolist(), 'rgb_sha256': adapter.initial_rgb_hashes,
        'reset_actions': adapter.reset_trace})
    xml = adapter.env.sim.model.get_xml()
    (out/'scene.xml').write_text(xml)
    save(out/'task-spec-private.json', adapter.env.task_spec)
    assets = {}
    for element in ET.fromstring(xml).iter():
        if element.get('file'):
            path = Path(element.get('file')).resolve(strict=True)
            assets[str(path)] = {'bytes': path.stat().st_size,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    save(out/'asset-manifest.json', assets)


def run_episode(args, root):
    from roboquest.harness.contract import CAMERAS
    from roboquest.harness.episode_tasks import reference_status
    args.out.mkdir(parents=True, exist_ok=False)
    config = {'pipeline_version': 'roboquest-episode-v1', 'task': args.task, 'scene': args.scene,
        'scene_revision': SCENE_PROFILES[args.scene]['scene_revision'],
        'reference_status': reference_status(args.task),
        'task_parameters': args.task_parameters, 'agent': args.agent, 'seed': args.seed,
        'image_size': args.image_size, 'cameras': list(CAMERAS),
        'completion_protocol': completion_protocol(args.scene),
        'horizon': args.horizon, 'max_decisions': args.max_decisions,
        'mode': 'prepare' if args.prepare_only else 'offline' if args.offline else 'live',
        'score_scope': 'Terminal state only; API/runtime interruptions unscored.',
        'observation_access': 'Allowlisted packets; external CLI filesystem access is instructed, not OS isolated.'}
    if args.agent == 'api':
        config.update(model=args.model, model_settings=MODEL_CONFIGS[args.model],
            max_attempts=args.max_attempts, max_request_attempts=args.max_request_attempts,
            max_restarts=args.max_restarts, request_timeout_s=args.request_timeout_s,
            retry_backoff_s=args.retry_backoff_s, honor_retry_after=args.honor_retry_after,
            image_horizon=2, provider=('offline' if args.offline else args.provider), base_url=args.base_url)
    elif args.agent == 'policy':
        config.update(policy={'url': args.policy_url, 'protocol': 'openpi-websocket-msgpack',
                              'image_size': args.image_size, 'evaluation_seed': args.policy_seed,
                              'replan_every': args.replan_every, 'require_subtask': args.require_subtask,
                              'action_filter': args.action_filter, 'timeout_s': args.policy_timeout_s})
    else:
        config.update(agent_model_label=args.agent_model_label, model_identity_verified=False,
            idle_timeout_s=args.idle_timeout_s, image_history='Public session files retained; CLI context is agent-managed.',
            timing_scope='Observation-to-action interaction time, including CLI reasoning/tools/scheduling.')
    save(args.out/'config.json', config)
    for parameter, filename in (('asset_scene', 'input-asset-scene.xml'),
                                ('shoe_manifest', 'input-shoe-manifest-private.json')):
        if args.task_parameters.get(parameter):
            target = args.out/filename
            shutil.copy2(args.task_parameters[parameter], target)
            config[parameter+'_input'] = {'file': filename,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    save(args.out/'config.json', config)
    freeze_source(root, args.out)
    if args.agent != 'policy':      # a policy gets the goal as its prompt and acts through native actions
        save(args.out/'tool-schemas.json', tool_schemas(args.scene))
        (args.out/'system-prompt.txt').write_text(
            policy_system_prompt(args.max_decisions, args.scene, include_goal=args.agent == 'api')+'\n')
    (args.out/'instruction.txt').write_text(trusted_goal(args.scene)+'\n')
    if args.prepare_only:
        print(json.dumps({'prepared': str(args.out), 'gpu_launched': False, 'http_requests': 0}))
        return 0

    adapter = policy = video = capture = trace_stream = None
    reset_done = False
    started = time.monotonic()
    result = {'completed': False, 'score_is_official': False, 'score': None}
    try:
        import imageio.v2 as imageio
        import numpy as np
        from roboquest.harness.egl import configure_egl
        from roboquest.harness.policy_task_factory import construct_adapter
        if args.agent == 'api':
            from inspect_robots_agent._capture import WireCapture
            capture = WireCapture()
            capture.begin_trial(str(args.out/'provider'), 'trial', 'episode')
            policy = make_api_policy(args, config, capture)
        elif args.agent == 'policy':
            from roboquest.harness.openpi_trial import OpenPIPolicy
            policy = OpenPIPolicy(args.policy_url, timeout_s=args.policy_timeout_s, image_size=args.image_size,
                                  eval_seed=args.policy_seed, instance_id=str(args.task).split('@')[-1],
                                  replan_every=args.replan_every, require_subtask=args.require_subtask,
                                  action_filter=args.action_filter)
            config['policy']['server_metadata'] = policy.connect()
            save(args.out/'config.json', config)
        else:
            policy = make_cli_policy(args, root)
        config['gpu_mapping'] = configure_egl(args.physical_gpu)
        save(args.out/'config.json', config)
        adapter = construct_adapter(args.scene, args.task_parameters, seed=args.seed,
            horizon=args.horizon, image_size=args.image_size, gpu=config['gpu_mapping']['egl_index'])
        # seed None means the registry instance's own seed; the value the scene was built with is recorded
        config['seed_used'] = getattr(adapter, 'seed', args.seed)
        save(args.out/'config.json', config)
        initial = adapter.reset()
        reset_done = True
        verify_adapter_goal(adapter, args.scene)
        save_initial(adapter, args.out)
        if hasattr(adapter.env, 'release_compiler_state'):
            # scene.xml and the asset manifest are on disk: free MuJoCo's ~1 GB compiler state now
            # and after any later hard reset (RoboQuest kitchens, see roboquest/lean.py).
            adapter.env.release_compiler_state()
            adapter.env.release_after_compile = True
        for camera, rgb in initial['rgb'].items():
            imageio.imwrite(args.out/('initial-'+camera+'.png'), rgb)
        video = imageio.get_writer(args.out/'rollout.mp4', fps=10)
        video.append_data(np.concatenate([initial['rgb'][camera] for camera in CAMERAS], axis=1))

        # The private trace goes to disk tick by tick; holding every row until the end grew RAM by ~0.27 MB a tick.
        trace_stream = (args.out/'physical-trace-private.jsonl').open('w')

        def write_trace():
            for row in adapter.trace:
                trace_stream.write(json.dumps(row, allow_nan=False)+'\n')
            adapter.trace.clear()

        def on_step(body):
            write_trace()
            if body.steps % 2 == 0:
                observed = body.observe()
                video.append_data(np.concatenate([observed['rgb'][camera] for camera in CAMERAS], axis=1))
        adapter.callback = on_step

        def on_turn(turn):
            record = deepcopy(turn)
            if args.agent == 'codex':
                record['agent_interaction_wall_s'] = record.pop('generation_wall_s')
                record.pop('http_attempts')
            save(args.out/'progress.json', record)
            folder = args.out/'observations'/f"decision-{turn['decision']:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            for camera, rgb in adapter.observe()['rgb'].items():
                imageio.imwrite(folder/(camera+'.png'), rgb)
            print(json.dumps({'event': 'action', **record}), flush=True)

        if args.agent == 'codex':
            policy.start(initial)
            prompt = 'Follow AGENTS.md and solve the robot task using the public observations and act.py.'
            command = f'codex -C {shlex.quote(str(args.public_dir))} {shlex.quote(prompt)}'
            print(json.dumps({'event': 'ready', 'public_dir': str(args.public_dir),
                              'private_out': str(args.out), 'connect_command': command}), flush=True)
        if args.agent == 'policy':
            from roboquest.harness.openpi_trial import run_openpi_trial
            inference_log = (args.out/'inference.jsonl').open('w')

            def on_inference(turn, record):
                inference_log.write(json.dumps(record, allow_nan=False)+'\n')
                inference_log.flush()
                save(args.out/'progress.json', turn)
                print(json.dumps({'event': 'inference', **turn}), flush=True)
            try:
                result.update(run_openpi_trial(adapter, policy, on_inference=on_inference))
            finally:
                inference_log.close()
        else:
            result.update(run_policy_trial(adapter, policy, on_turn=on_turn))
        if args.offline:
            result.update(score_is_official=False, score_scope='Offline wiring smoke; mock policy, not model performance.')
    except KeyboardInterrupt:
        result.update(completed=False, score_is_official=False, score=None,
                      termination='operator_interrupted')
    except Exception as error:
        # Exception bodies can contain provider/account secrets. Persist only type.
        result.update(completed=False, score_is_official=False, score=None,
                      termination='setup_or_runtime_interrupted', error_type=type(error).__name__)
    finally:
        cleanup_errors = []
        def cleanup(name, action):
            try:
                action()
            except Exception as error:
                cleanup_errors.append({'stage': name, 'error_type': type(error).__name__})
        if adapter is not None:
            if reset_done:
                def stop_adapter():
                    if not adapter.observe()['episode_ended']:
                        adapter.execute({'type': 'stop'})
                cleanup('adapter_stop', stop_adapter)
                if args.agent == 'codex' and policy is not None:
                    cleanup('public_finish', lambda: policy.finish(adapter.observe()))
                cleanup('commands', lambda: save(args.out/'commands.json', adapter.commands))
                def save_trace():
                    if trace_stream is None:
                        with (args.out/'physical-trace-private.jsonl').open('w') as stream:
                            for row in adapter.trace:
                                stream.write(json.dumps(row, allow_nan=False)+'\n')
                        return
                    write_trace()   # rows after the last tick callback (e.g. an evaluation error)
                    trace_stream.close()
                cleanup('physical_trace', save_trace)
                def save_final():
                    for camera, rgb in adapter.observe()['rgb'].items():
                        imageio.imwrite(args.out/('final-'+camera+'.png'), rgb)
                    save(args.out/'final-proprio.json', adapter.observe()['proprio'])
                cleanup('final_observation', save_final)
            cleanup('adapter_close', adapter.close)
        if policy is not None:
            def finish_policy():
                turns = deepcopy(policy.turns)
                if args.agent == 'policy':
                    save(args.out/'turns.json', turns)
                    return
                if args.agent == 'api':
                    result['returned_model_ids'] = policy.returned_model_ids
                    save(args.out/'api-attempts.json', policy.transport.summary())
                    save(args.out/'usage.json', policy.usage)
                    save(args.out/'transcripts.json', {'archived_conversations': policy.archived_messages,
                                                       'current': policy.messages})
                else:
                    for turn in turns:
                        turn['agent_interaction_wall_s'] = turn.pop('generation_wall_s')
                        turn.pop('http_attempts')
                    save(args.out/'public-transcript.json', policy.messages)
                    if not (args.public_dir/'finished.json').exists():
                        save(args.public_dir/'finished.json', {'episode_ended': True,
                            'error': 'Simulator unavailable', 'score_available_to_policy': False})
                    shutil.copytree(args.public_dir, args.out/'public-session')
                save(args.out/'turns.json', turns)
            cleanup('policy_evidence', finish_policy)
            if args.agent in ('api', 'policy'):
                cleanup('policy_close', policy.close)
        if capture is not None:
            cleanup('capture_close', lambda: result.update(wire_capture=capture.end_trial()))
        if video is not None:
            cleanup('video_close', video.close)
        result.update(agent=args.agent, task=args.task, offline=args.offline,
                      whole_process_wall_s=time.monotonic()-started)
        if args.agent == 'codex':
            for old, new in (('successful_model_decisions', 'robot_decisions'),
                             ('generation_wall_s', 'agent_interaction_wall_s'),
                             ('generation_attempts', 'external_model_api_calls')):
                if old in result:
                    result[new] = result.pop(old)
            result.pop('http_attempt_wall_s', None)
        if cleanup_errors:
            result['cleanup_errors'] = cleanup_errors
        save(args.out/'result.json', result)  # Written after video close.
        if result.get('completed'):
            # Progress, the benchmark's progress metric (docs/progress.md), from the finished episode.
            try:
                from roboquest.scoring.progress import write as write_progress
                write_progress(args.out)
            except Exception as error:
                save(args.out/'task-progress.json', {'metric': 'progress', 'error_type': type(error).__name__})
    print(json.dumps({'event': 'finished', 'out': str(args.out),
        'completed': result['completed'], 'termination': result.get('termination'),
        'success': (result.get('score') or {}).get('success')}), flush=True)
    return 0 if result.get('completed') and not result.get('cleanup_errors') else 1
