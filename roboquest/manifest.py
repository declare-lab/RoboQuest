"""Static task manifest; no simulator imports so catalogs and CLIs stay light.

Each entry names the task module and class, and repeats the contract and
generator versions so registries and public catalogs can be built without
importing RoboCasa. ``tests/test_roboquest.py`` checks the copies agree.
"""
from importlib import import_module

TASKS = {
    'marked_mugs': dict(
        module='roboquest.tasks.marked_mugs', cls='MarkedMugs',
        contract_version='marked-mugs-kitchen-v3', generator_version='v4',
        family='object_inspect', tasks_v0_id=4,
        sources=('roboquest/tasks/marked_mugs.py', 'roboquest/mugs/__init__.py',
                 'roboquest/mugs/geometry.py', 'roboquest/mugs/rules.py',
                 'roboquest/mugs/hiding.py', 'roboquest/observe.py',
                 'roboquest/stand/helpers.py', 'roboquest/scatter.py')),
    'puzzle_box': dict(
        module='roboquest.tasks.puzzle_box', cls='PuzzleBox',
        contract_version='puzzle-box-interlock-kitchen-v2', generator_version='v3',
        family='interactive_testing', tasks_v0_id=9,
        sources=('roboquest/tasks/puzzle_box.py', 'roboquest/puzzle/layouts.py',
                 'roboquest/puzzle/plate.py',
                 'roboquest/puzzle_box_geometry.py', 'roboquest/puzzle_box_layouts.py',
                 'scripts/puzzle_box_physics_gate.py', 'roboquest/oracles/puzzle_box.py',
                 'roboquest/scatter.py')),
    'painted_cubes': dict(
        module='roboquest.tasks.painted_cubes', cls='PaintedCubes',
        contract_version='painted-cubes-kitchen-v3', generator_version='v6',
        family='object_inspect', tasks_v0_id=5,
        sources=('roboquest/tasks/painted_cubes.py', 'roboquest/cubes/marks.py',
                 'roboquest/cubes/bin.py', 'roboquest/cubes/hiding.py',
                 'roboquest/cubes/reach.py', 'roboquest/observe.py')),
    'wobbly_stand': dict(
        module='roboquest.tasks.wobbly_stand', cls='WobblyStand',
        contract_version='wobbly-stand-kitchen-v2', generator_version='v2',
        family='interactive_testing', tasks_v0_id=10,
        sources=('roboquest/tasks/wobbly_stand.py', 'roboquest/stand/__init__.py',
                 'roboquest/stand/geometry.py', 'roboquest/stand/statics.py',
                 'roboquest/stand/bare.py', 'roboquest/stand/helpers.py')),
    'odd_parcel': dict(
        module='roboquest.tasks.odd_parcel', cls='OddParcel',
        contract_version='odd-parcel-kitchen-v3', generator_version='v4',
        family='interactive_testing', tasks_v0_id=8,
        sources=('roboquest/tasks/odd_parcel.py', 'roboquest/balance/__init__.py',
                 'roboquest/balance/answer_box.py', 'roboquest/balance/builder.py',
                 'roboquest/observe.py', 'roboquest/uncover.py',
                 'roboquest/scatter.py')),
    'blackout_search': dict(
        module='roboquest.tasks.blackout_search', cls='BlackoutSearch',
        contract_version='blackout-search-kitchen-v2', generator_version='v3',
        family='search', tasks_v0_id=3,
        sources=('roboquest/tasks/blackout_search.py', 'roboquest/tasks/search_room.py',
                 'roboquest/blackout/__init__.py', 'roboquest/blackout/lamp.py',
                 'roboquest/blackout/darkness.py',
                 'roboquest/blackout/structure.py', 'roboquest/search/layout.py',
                 'roboquest/search/objects.py', 'roboquest/search/rules.py',
                 'roboquest/search/covers.py', 'roboquest/search/visibility.py',
                 'roboquest/closers.py')),
    'search_room': dict(
        module='roboquest.tasks.search_room', cls='SearchRoom',
        contract_version='search-room-kitchen-v2', generator_version='v1.1',
        family='search', tasks_v0_id=1,
        sources=('roboquest/tasks/search_room.py', 'roboquest/search/layout.py',
                 'roboquest/search/objects.py', 'roboquest/search/rules.py',
                 'roboquest/search/covers.py', 'roboquest/closers.py',
                 'roboquest/search/visibility.py', 'roboquest/search/planner.py',
                 'roboquest/search/motor.py', 'roboquest/oracles/search_room.py')),
    'locked_storage': dict(
        module='roboquest.tasks.locked_storage', cls='LockedStorage',
        contract_version='locked-storage-kitchen-v2', generator_version='v3',
        family='search', tasks_v0_id=2,
        sources=('roboquest/tasks/locked_storage.py', 'roboquest/locked/__init__.py',
                 'roboquest/locked/objects.py', 'roboquest/locked/rules.py',
                 'roboquest/locked/structure.py', 'roboquest/locked/heights.py',
                 'roboquest/tasks/search_room.py',
                 'roboquest/search/layout.py', 'roboquest/search/objects.py',
                 'roboquest/search/rules.py', 'roboquest/search/visibility.py',
                 'roboquest/closers.py', 'roboquest/search/motor.py',
                 'roboquest/oracles/locked_storage.py')),
    'stamps': dict(
        module='roboquest.tasks.stamps', cls='Stamps',
        contract_version='stamps-composition-kitchen-v3', generator_version='v3',
        family='interactive_testing', tasks_v0_id=7,
        sources=('roboquest/tasks/stamps.py', 'roboquest/stamps/__init__.py',
                 'roboquest/stamps/rules.py', 'roboquest/stamps/geometry.py',
                 'roboquest/stamps/hiding.py', 'roboquest/oracles/stamps.py',
                 'roboquest/observe.py', 'roboquest/scatter.py')),
    'unfamiliar_containers': dict(
        module='roboquest.tasks.unfamiliar_containers', cls='UnfamiliarContainers',
        contract_version='unfamiliar-containers-kitchen-v3', generator_version='v6',
        family='object_inspect', tasks_v0_id=6,
        sources=('roboquest/tasks/unfamiliar_containers.py',
                 'roboquest/containers/__init__.py',
                 'roboquest/containers/appearance.py',
                 'roboquest/containers/contract.py',
                 'roboquest/containers/hiding.py',
                 'roboquest/containers/mechanism.py',
                 'roboquest/containers/panels.py',
                 'roboquest/containers/placement.py',
                 'roboquest/containers/rules.py',
                 'roboquest/containers/springs.py',
                 'roboquest/closers.py',
                 'roboquest/observe.py',
                 'roboquest/oracles/unfamiliar_containers.py')),
}
SHARED_SOURCES = ('roboquest/kitchen.py', 'roboquest/registry.py',
                  'roboquest/evaluator.py', 'roboquest/adapter.py',
                  'roboquest/manifest.py', 'roboquest/base/submit_button.py',
                  'roboquest/base/scene.py', 'roboquest/base/scene_contracts.py',
                  'roboquest/harness/base_adapter.py', 'roboquest/harness/contract.py',
                  'roboquest/harness/mobile_native_contract.py')


def task_entry(name):
    if name not in TASKS:
        raise ValueError(f'unknown RoboQuest task {name!r}; choose from {sorted(TASKS)}')
    return TASKS[name]


def task_class(name):
    """Import the task's Kitchen subclass (this imports RoboCasa)."""
    entry = task_entry(name)
    return getattr(import_module(entry['module']), entry['cls'])
