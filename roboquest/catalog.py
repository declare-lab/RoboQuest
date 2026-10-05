"""Bridge from suite registries to the policy/HTTP scene catalogue. No simulator imports.

For every task in the manifest whose suite registry exists under ``<suite_dir>/<task>/registry``,
one profile ``roboquest_<task>`` is registered that runs the registry's default development
instance through ``RoboQuestAdapter``. The suite root is ``ROBOQUEST_SUITE_DIR`` when set;
otherwise every task takes the newest default root that holds its registry (``suite/v1`` in the
repository, falling back to ``suite/v0`` until the task's v1 registry is generated). Individual
instances are addressed as ``roboquest_<task>@<instance_id>`` through
:func:`resolve_instance_profile`. Evaluation instances are never listed, and resolve only for a caller
that asks for the evaluation split by name (the benchmark runner's ``--evaluation-instances``).
"""
from copy import deepcopy
import json
import os
from pathlib import Path

from roboquest.manifest import TASKS, SHARED_SOURCES

REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_DIRS = (REPO_ROOT / 'suite' / 'v1', REPO_ROOT / 'suite' / 'v0')     # newest first
DEFAULT_SUITE_DIR = SUITE_DIRS[0]
PROFILE_PREFIX = 'roboquest_'


def suite_dir():
    """The suite root: ``ROBOQUEST_SUITE_DIR`` when set, else the newest default root that exists."""
    override = os.environ.get('ROBOQUEST_SUITE_DIR')
    if override:
        return Path(override).resolve()
    for root in SUITE_DIRS:
        if root.is_dir():
            return root.resolve()
    return SUITE_DIRS[0].resolve()


def registry_dir(task, root=None):
    """``<root>/<task>/registry``. Without ``root`` or an override, the newest default root that holds
    a registry for ``task``: a task keeps its v0 registry until its v1 one is generated."""
    if root is not None:
        return Path(root) / task / 'registry'
    override = os.environ.get('ROBOQUEST_SUITE_DIR')
    if override:
        return Path(override).resolve() / task / 'registry'
    for candidate in SUITE_DIRS:
        if (candidate / task / 'registry' / 'registry.json').is_file():
            return candidate / task / 'registry'
    return SUITE_DIRS[0] / task / 'registry'


def _read_registry(path):
    try:
        return json.loads((Path(path) / 'registry.json').read_text())
    except (OSError, ValueError):
        return None


def dev_instances(task, root=None):
    """Public rows (id, split key, goal) of the development instances of a task's suite registry."""
    payload = _read_registry(registry_dir(task, root))
    if not payload:
        return []
    return [dict(instance_id=r['instance_id'], split_key=r.get('split_key'), goal=r.get('goal'),
                 factors=r.get('factors')) for r in payload['instances'] if r.get('split') == 'dev']


def _profile(task, entry, reg_dir, row, control_version, family_default):
    sources = tuple(entry.get('sources', ())) + tuple(SHARED_SOURCES) + (
        'roboquest/adapter.py', 'roboquest/catalog.py')
    return {
        'goal': row.get('goal') or '', 'family': entry.get('family', family_default),
        'adapter_module': 'roboquest.adapter', 'adapter_class': 'RoboQuestAdapter',
        'adapter_parameters': {'instance': f"{reg_dir}#{row['instance_id']}"},
        'control_version': control_version, 'scene_revision': entry['contract_version'],
        'http_enabled': True, 'completion_protocol': 'physical_submit_v1',
        'roboquest': {'task': task, 'instance_id': row['instance_id'], 'split_key': row.get('split_key'),
                      'registry': str(reg_dir)},
        'extra_sources': sources,
    }


def register_roboquest_profiles(profiles, root=None, control_version='mobile-native-v1'):
    """Add ``roboquest_<task>`` profiles for every task with a suite registry. Returns the names added."""
    added = []
    for task, entry in TASKS.items():
        reg_dir = registry_dir(task, root)
        payload = _read_registry(reg_dir)
        if not payload:
            continue
        default_id = payload.get('summary', {}).get('default_instance')
        rows = {r['instance_id']: r for r in payload['instances']}
        row = rows.get(default_id)
        if row is None or row.get('split') != 'dev':   # never bind a profile to an evaluation instance
            row = next((r for r in payload['instances'] if r.get('split') == 'dev'), None)
        if row is None:
            continue
        name = PROFILE_PREFIX + task
        profiles[name] = _profile(task, entry, reg_dir, row, control_version, 'roboquest')
        added.append(name)
    return added


def resolve_instance_profile(profiles, scene, splits=('dev',)):
    """``roboquest_<task>@<instance_id>`` -> a profile bound to that instance.

    Only instances of the given ``splits`` resolve: development ones by default. The evaluation split is
    reached solely by a caller that names it, which the benchmark runner does for its own evaluation runs
    (``run_episode.py --evaluation-instances``); nothing lists or binds evaluation instances otherwise.
    """
    if '@' not in scene:
        return profiles[scene]
    base, instance_id = scene.split('@', 1)
    if base not in profiles or 'roboquest' not in profiles[base]:
        raise KeyError(scene)
    info = profiles[base]['roboquest']
    payload = _read_registry(info['registry'])
    row = next((r for r in payload['instances'] if r['instance_id'] == instance_id), None) if payload else None
    if row is None or row.get('split') not in splits:
        raise KeyError(f'{scene}: unknown instance, or one outside the {"/".join(splits)} split')
    profile = deepcopy(profiles[base])
    profile['goal'] = row.get('goal') or profile['goal']
    profile['adapter_parameters'] = {'instance': f"{info['registry']}#{instance_id}"}
    profile['roboquest'] = dict(info, instance_id=instance_id, split_key=row.get('split_key'), split=row.get('split'))
    return profile


def bind_instance_profile(profiles, scene, splits=('dev',)):
    """Resolve an instance address and register the bound profile under it, for the direct runner only
    (the HTTP catalogue stays task-level, so the bound profile is never published). Returns the scene."""
    if scene not in profiles:
        try:
            profile = resolve_instance_profile(profiles, scene, splits)
        except KeyError:
            raise ValueError('Unknown task instance: ' + scene) from None
        profile['http_enabled'] = False
        profiles[scene] = profile
    return scene
