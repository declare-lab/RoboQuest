"""The room light of Blackout Search: off for the whole episode (torch-only).

There is no wall switch and no timer: the arena light is switched off at reset through
``model.light_active`` and nothing in the task turns it on again. The two helpers here are what the build
and the render gate still need -- finding the arena light(s) so they can be switched off, and switching
them for the room-light half of G3, which proves that a compartment or a cloche (not the darkness) hides
what it should. The lantern's own lights are never touched: a MuJoCo scene with *no* active light renders
flat unlit colour (about 45/255) instead of darkness, so at least one light has to stay on, and the
lantern's two are it.

``empty_kitchen_arena.xml`` ships exactly one arena light, unnamed; the lantern's are named, so "every
light that is not the lantern's" is the arena.
"""


def room_light_ids(model, keep_names=()):
    """Light indices of everything that is not the lantern: the arena light(s) this task switches off."""
    keep = set(keep_names)
    ids = []
    for index in range(int(model.nlight)):
        try:
            name = model.light_id2name(index)
        except Exception:  # noqa: BLE001
            name = None
        if name not in keep:
            ids.append(index)
    return ids


def set_room_light(model, ids, on):
    """Turn the arena lights on or off in the compiled model (the gate's room-light check, never the episode)."""
    for index in ids:
        model.light_active[index] = 1 if on else 0
    return bool(on)
