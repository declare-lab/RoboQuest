"""Trusted diagnostic envelopes and goals; deliberately no simulator imports.

The release registers only the RoboQuest suite tasks (end of file).
"""
from copy import deepcopy
import math
from pathlib import Path

from roboquest.harness.contract import CONTRACT_VERSION
from roboquest.harness.mobile_native_contract import PUBLIC_CONTROL_VERSION

PROTOCOL_PROFILES = {
    'wiring_v1': {'max_decisions': 12, 'max_attempts': 24, 'horizon': 2400},
    'full_counter_v1': {'max_decisions': 48, 'max_attempts': 96, 'horizon': 4000},
    'full_task_v1': {'max_decisions': 48, 'max_attempts': 96, 'horizon': 4000},
    'physical_submit_v1': {'max_decisions': None, 'max_attempts': None, 'horizon': 1_000_000},
    'agent_stop_v1': {'max_decisions': None, 'max_attempts': None, 'horizon': 1_000_000},
}
SCENE_PROFILES = {}
LIVE_SCENES = frozenset()


def trusted_goal(scene='baseline_counter'):
    if scene not in SCENE_PROFILES:
        raise ValueError('Unknown trusted manipulation scene')
    return SCENE_PROFILES[scene]['goal']


def trusted_control_version(scene):
    trusted_goal(scene)
    return SCENE_PROFILES[scene].get('control_version', CONTRACT_VERSION)


def adapter_parameters(scene, variant='cup_deficit', *, asset_scene=None, shoe_manifest=None):
    """Private task construction; family/configuration names never enter packets."""
    trusted_goal(scene)
    profile = SCENE_PROFILES[scene]
    if asset_scene is None and profile.get('default_asset_scene'):
        asset_scene = profile['default_asset_scene']
    if profile.get('requires_asset_scene'):
        if asset_scene is None or not Path(asset_scene).is_file():
            raise ValueError('This scene requires an existing --asset-scene XML')
    elif asset_scene is not None and not profile.get('accepts_asset_scene'):
        raise ValueError('--asset-scene does not apply to this task scene')
    if asset_scene is not None and not Path(asset_scene).is_file():
        raise ValueError('--asset-scene must be an existing XML')
    if profile.get('requires_shoe_manifest'):
        if shoe_manifest is None or not Path(shoe_manifest).is_file():
            raise ValueError('This scene requires an existing --shoe-manifest JSON')
    elif shoe_manifest is not None:
        raise ValueError('--shoe-manifest does not apply to this task scene')
    if 'adapter_parameters' in profile:
        if variant != 'cup_deficit':
            raise ValueError('Table-setting --variant does not apply to this task scene')
        parameters = deepcopy(profile['adapter_parameters'])
        if profile.get('requires_asset_scene') or asset_scene is not None:
            parameters['asset_scene'] = str(Path(asset_scene).resolve())
        if profile.get('requires_shoe_manifest'):
            parameters['shoe_manifest'] = str(Path(shoe_manifest).resolve())
        return parameters
    if variant not in ('cup_deficit', 'plate_deficit', 'visible_deficit'):
        raise ValueError('Unknown table-setting variant')
    return {'variant': variant}


def resolve_protocol(profile='wiring_v1', *, max_decisions=None, max_attempts=None,
                     horizon=None, max_request_attempts=6, max_restarts=2, request_timeout_s=120.):
    if profile not in PROTOCOL_PROFILES:
        raise ValueError('Unknown diagnostic protocol profile')
    envelope = PROTOCOL_PROFILES[profile]
    requested = dict(max_decisions=max_decisions, max_attempts=max_attempts, horizon=horizon)
    for name, value in requested.items():
        limit = envelope[name]
        value = limit if value is None else value
        if limit is None:
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f'{profile} requires {name} to be a positive integer or None')
        elif type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f'{profile} requires {name} in [1,{limit}]')
        requested[name] = value
    if type(max_request_attempts) is not int or not 1 <= max_request_attempts <= 6:
        raise ValueError('At most six immediate attempts per frozen request')
    if type(max_restarts) is not int or not 0 <= max_restarts <= 2:
        raise ValueError('At most two continuation restarts')
    timeout_limit = 600 if profile in ('agent_stop_v1', 'physical_submit_v1') else 120
    if (isinstance(request_timeout_s, bool) or not math.isfinite(request_timeout_s)
            or not 0 < request_timeout_s <= timeout_limit):
        raise ValueError(f'Request timeout must be finite and in (0,{timeout_limit}] seconds')
    return {**requested, 'max_request_attempts': max_request_attempts,
            'max_restarts': max_restarts, 'request_timeout_s': request_timeout_s,
            'protocol_profile': profile, 'profile_envelope': deepcopy(envelope)}


def verify_adapter_goal(adapter, scene):
    """Fail before policy generation if a selected adapter changes its public goal."""
    if adapter.observe()['instruction'] != trusted_goal(scene):
        raise ValueError('Selected adapter and trusted public goal disagree')


def completion_protocol(scene):
    return SCENE_PROFILES[scene].get('completion_protocol', 'legacy_declared_stop_or_horizon')


# RoboQuest suite tasks: one profile per task whose suite registry exists (suite/v1/<task>/registry,
# else suite/v0), bound to the registry's default development instance. JSON only; no simulator imports.
from roboquest.catalog import register_roboquest_profiles as _register_roboquest_profiles  # noqa: E402
ROBOQUEST_PROFILES = tuple(_register_roboquest_profiles(SCENE_PROFILES, control_version=PUBLIC_CONTROL_VERSION))
