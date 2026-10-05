"""Chain structure, split keys, hold-outs and reader-pad siting for ``locked_storage``.

**Split key** (generator v3) -- ``d{chain_depth}-{pattern}-{lock colours}``: the depth, the
chain's kind pattern (``drawers`` for an all-drawer chain, ``door`` when the last link is the HingeCabinet
of a double-door instance) and the instance's lock colours sorted and joined with ``+``, e.g.
``d2-drawers-cyan+purple`` or ``d3-door-magenta+orange+white``. The dead-end factor is not in the key.

**Hold-out** -- the class held out for evaluation is *a purple lock somewhere in the instance*
(:data:`HOLDOUT_COLOUR`, :func:`is_holdout`): no development instance uses purple for any lock, every
evaluation instance uses it for at least one, on a lock the structure stream chose (the target's, a
mid-chain link's or the dead end's, so the colour never says which compartment matters). Why this axis:
under the placement rules of 2026-09-24 the chain's kinds are fixed -- rule-1 drawers throughout, or
drawers with one HingeCabinet as the final link of a double-door instance -- and the double-door mix
must appear in both splits at the same proportion, so no kind pattern can be held out; the dead-end
level is a factor; the target object is named in the goal; the lock colour is the one structural choice
of the chain that exists in every cell, on every kitchen and in both door mixes. What development keeps
per depth is :func:`dev_composition_counts`: every kind pattern and every dead-end level, with 4 of the
5 lock colours.

**Reader pads** -- :func:`pad_spot` refuses a compartment as a chain member when the search
machinery finds no clear counter spot above or beside its front, which is what keeps the pad rule
("0.12-0.24 m inside the approachable edge, within 0.35 m laterally of the handle, clear of decor and
of the other task objects") a build-time gate rather than a hope.
"""
import numpy as np

from roboquest.search.layout import disc_clear, front_normal, patch_for_spot, yaw_matrix
from roboquest.locked.objects import PAD_RADIUS_M

CHAIN_DEPTHS = (1, 2, 3)
DEAD_END_LEVELS = (0, 1)
PLACE_KINDS = ('door', 'drawer')
SET_SIZE = 1                 # fixed (contract section 1); recorded in the spec so it can be raised
COMPARTMENT_COUNT = 6        # fixed; the placement table already excludes layouts that cannot host 6

# The held-out class: a purple lock anywhere in the instance (see the module docstring). Exact keys are
# not used (HOLDOUT_KEYS stays empty); `is_holdout` decides from the key's colour part.
HOLDOUT_COLOUR = 'purple'
HOLDOUT_KEYS = ()
KIND_PATTERNS = ('drawers', 'door')      # all rule-1 drawers, or a HingeCabinet as the final link

PAD_LATERAL_MAX_M = .35      # pad centre to the compartment's front centre, along the counter
# Out from that front centre the pad may sit anywhere from the back of the compartment's own counter
# to one base standoff in front of it, so a single stance reaches the lock and its reader. Measured
# with the window open over the 24-instance trial registry, the nearest usable counter spot has a
# median forward of -0.09 m (the counter over the compartment: `sample_counter_spots(edge_only=True)`
# sits 0.12-0.24 m back from the approachable edge) and 46 of 58 pads land inside +-0.3 m.
PAD_FRONT_MIN_M = -.5
PAD_FRONT_MAX_M = .6
PAD_SPACING_M = .18          # between two pads, and from any other task object on the counter


def split_key(chain_depth, double_door, colours):
    pattern = 'door' if double_door else 'drawers'
    return f"d{int(chain_depth)}-{pattern}-" + '+'.join(sorted(colours))


def key_colours(key):
    """The lock colours a split key names (its last dash-separated part)."""
    return tuple(key.rsplit('-', 1)[-1].split('+')) if key else ()


def is_holdout(key):
    """Held out for evaluation: the instance has a lock in :data:`HOLDOUT_COLOUR`."""
    return HOLDOUT_COLOUR in key_colours(key)


def colour_combinations(count, palette):
    """Every sorted ``count``-subset of the palette."""
    from itertools import combinations
    return [tuple(sorted(c)) for c in combinations(sorted(palette), int(count))]


def all_keys(depth, palette=None):
    """Every split key at ``depth`` over both dead-end levels and both kind patterns."""
    from roboquest.locked.objects import COLOUR_NAMES
    palette = COLOUR_NAMES if palette is None else palette
    out = []
    for dead_end in DEAD_END_LEVELS:
        for pattern in KIND_PATTERNS:
            for colours in colour_combinations(int(depth) + int(dead_end), palette):
                out.append(split_key(depth, pattern == 'door', colours))
    return sorted(set(out))


def dev_composition_counts():
    """``{depth: (kept, total)}`` split keys development keeps at each depth (the keys without purple)."""
    counts = {}
    for depth in CHAIN_DEPTHS:
        keys = all_keys(depth)
        counts[depth] = (sum(1 for k in keys if not is_holdout(k)), len(keys))
    return counts


def front_centre(comp):
    """World xy of the middle of a compartment's front face."""
    normal = np.asarray(front_normal(comp['rot']), float)
    return np.asarray(comp['pos'][:2], float) + normal * (float(comp['size'][1]) / 2)


def pad_offsets(comp, xy):
    """(lateral, forward) metres from the compartment's front centre to ``xy``, in the fixture frame."""
    normal = np.asarray(front_normal(comp['rot']), float)
    tangent = (yaw_matrix(comp['rot']) @ np.array([1., 0., 0.]))[:2]
    delta = np.asarray(xy, float) - front_centre(comp)
    return float(np.dot(delta, tangent)), float(np.dot(delta, normal))


def pad_spot(comp, spots, records, exclude=(), keepouts=None, taken=(), radius=PAD_RADIUS_M,
             lateral_max=PAD_LATERAL_MAX_M, front_max=PAD_FRONT_MAX_M, spacing=PAD_SPACING_M,
             rejected=None):
    """The *nearest* counter spot in ``spots`` that can host this compartment's reader pad, or ``None``.

    ``spots`` come from ``search.layout.sample_counter_spots(edge_only=True)``, so each already sits
    0.12-0.24 m inside an approachable counter edge, carries a base standoff and avoids decor. Here
    the spot must additionally lie within ``lateral_max`` of the compartment's front centre along the
    counter and between ``PAD_FRONT_MIN_M`` and ``front_max`` out from it, keep ``spacing`` from the
    spots already ``taken``, and leave the whole pad slab clear of decor and of the ``exclude``
    polygons.

    The candidates are tried nearest-first. Taking them in sampling order instead put the reader
    1.3-5.4 m out (median 2.9 m over the 50 pads of the first trial registry): a long counter run
    across the aisle is still within 0.35 m *laterally* of a cabinet on the opposite wall, so the
    lateral bound alone let the pad land in a different part of the kitchen from its lock. The point
    of the pad is that a robot that has driven to a compartment and read its coloured lock plate can
    see the matching reader from the same stance, so the pad belongs on the counter over the
    compartment or one run around the corner from it. ``rejected`` collects ``(lateral, forward)`` for
    every candidate turned away only by ``front_max``, which is what makes a refusal diagnosable.
    """
    ranked = []
    for spot in spots:
        if spot.get('is_shelf') or spot.get('standoff') is None:
            continue
        lateral, forward = pad_offsets(comp, spot['xy'])
        if abs(lateral) > float(lateral_max) or forward < PAD_FRONT_MIN_M:
            continue      # the pad never goes behind the compartment's own counter run
        ranked.append((float(np.hypot(lateral, forward)), lateral, forward, spot))
    ranked.sort(key=lambda row: row[0])
    for _, lateral, forward, spot in ranked:
        if forward > float(front_max):
            if rejected is not None:
                rejected.append((round(lateral, 4), round(forward, 4)))
            continue
        if any(np.linalg.norm(np.subtract(spot['xy'], t)) < float(spacing) for t in taken):
            continue
        patch = patch_for_spot(records, spot)
        if patch is None or patch.get('is_shelf'):
            continue
        if not disc_clear(spot['xy'], patch, records, radius, exclude=exclude, keepouts=keepouts):
            continue
        return dict(spot, pad_lateral_m=lateral, pad_forward_m=forward)
    return None


def refusal(cid, kind, reason):
    return dict(compartment=cid, kind=kind, reason=str(reason))


def chain_binding(candidates, chain_kinds, dead_end_kind, rng, pad_for, refusals, fits=None):
    """Bind the chain (and the dead end) to real compartments, refusing any that has no pad spot.

    ``candidates`` are the compartments on offer (``{cid: record}``); ``pad_for(comp)`` returns a pad
    spot or ``None``; ``fits(index, comp)`` may veto a compartment for the lock at ``index`` (the
    target has to fit inside the last chain compartment). Returns ``(chain_ids, dead_end_id, pads)``;
    raises ``ValueError`` with the refusal reasons when the layout cannot host the chain, so the
    build gate rejects it.
    """
    used, chain_ids, pads = set(), [], {}
    order = list(chain_kinds) + ([dead_end_kind] if dead_end_kind else [])
    for index, kind in enumerate(order):
        options = [c for c in candidates.values() if c['id'] not in used and c['kind'] == kind
                   and (fits is None or fits(index, c))]
        options.sort(key=lambda c: c['id'])
        if not options:
            raise ValueError(f'no unused {kind} compartment left for lock {index + 1} '
                             f'(refusals: {refusals})')
        picked = None
        while options:
            choice = options[int(rng.integers(len(options)))]
            spot = pad_for(choice)
            if spot is not None:
                picked = (choice, spot)
                break
            refusals.append(refusal(choice['id'], kind, 'no clear reader-pad spot above or beside the front'))
            options = [c for c in options if c['id'] != choice['id']]
        if picked is None:
            raise ValueError(f'every {kind} compartment was refused a reader pad for lock {index + 1} '
                             f'(refusals: {refusals})')
        choice, spot = picked
        used.add(choice['id'])
        pads[choice['id']] = spot
        if index < len(chain_kinds):
            chain_ids.append(choice['id'])
        else:
            return chain_ids, choice['id'], pads
    return chain_ids, None, pads
