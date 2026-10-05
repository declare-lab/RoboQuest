"""Public CLI names over the existing evaluator-owned scene registry."""
from roboquest.harness.profiles import SCENE_PROFILES, LIVE_SCENES

# Aliases select existing versioned tasks; do not duplicate their configuration.
TASK_ALIASES = {}


def resolve_task(name):
    scene = TASK_ALIASES.get(name, name)
    if scene not in SCENE_PROFILES and isinstance(scene, str) and '@' in scene:
        scene = _bind_instance(scene)
    if scene not in SCENE_PROFILES:
        raise ValueError('Unknown task: ' + str(name))
    return scene


def _bind_instance(scene):
    """``roboquest_<task>@<instance_id>`` registers a profile bound to that development instance."""
    from roboquest.catalog import resolve_instance_profile
    try:
        profile = resolve_instance_profile(SCENE_PROFILES, scene)
    except KeyError:
        raise ValueError('Unknown task instance: ' + scene) from None
    # Instance addresses are a direct-runner feature; the HTTP catalog stays task-level.
    profile['http_enabled'] = False
    SCENE_PROFILES[scene] = profile
    return scene


def task_names():
    return tuple(dict.fromkeys((*TASK_ALIASES, *SCENE_PROFILES)))


def http_task_scenes():
    """Opt-in publication from the shared registry; aliases have one public name."""
    aliases = {scene: name for name, scene in TASK_ALIASES.items()}
    return {aliases.get(scene, scene): scene for scene, profile in SCENE_PROFILES.items()
            if profile.get('http_enabled', False)}


def http_task_catalog():
    """Only public contract metadata, never factories, assets or evaluator state."""
    catalog = []
    for name, scene in http_task_scenes().items():
        profile = SCENE_PROFILES[scene]
        # The task owns this contract; HTTP must not prescribe a finish button.
        protocol = profile['completion_protocol']
        if not isinstance(protocol, str) or not protocol:
            raise ValueError('HTTP tasks must declare their completion protocol')
        catalog.append({'id': name, 'completion_protocol': protocol,
                        'control_version': profile['control_version'],
                        'scene_revision': profile['scene_revision']})
    return catalog


def reference_status(name):
    scene = resolve_task(name)
    if scene == 'pick_cube':
        return 'pipeline_diagnostic; not_a_benchmark_reference'
    if scene in LIVE_SCENES:
        return 'registered_legacy_physical_reference'
    if scene == 'stamps_v2':
        return 'scripted_native_oracle_verified; physical_submit_v1'
    if scene == 'stamps':
        return 'scripted_native_prototype; shared_agent_adapter_development'
    return 'development_scene; physical_reference_not_accepted_for_legacy_live_runner'
