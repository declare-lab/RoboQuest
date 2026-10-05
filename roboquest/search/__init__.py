"""Search-task building blocks shared by Search Room (task 1) and its later variants.

Copied from the room-search worktree (`room_search_{layout,objects,planner,motor,rules}`)
and adapted to the RoboQuest base class: compartment discovery over plain fixture
records, the recoloured object pool, the grid base planner, camera-frustum
visibility tests, scoring for a set of targets in a task-frame tray and the
privileged motor used by the oracle. Only ``motor`` imports the simulator stack.
"""
