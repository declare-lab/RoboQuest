"""Unfamiliar Containers: identical cases, two knobbed panels each, one real, spread over the surface.

Four, five or six identical wooden cases (26 x 26 x 22 cm) stand **sparsely** over the whole free top
of a RoboCasa work surface, each turned to a random quarter of a turn, so that a panel may look toward
the robot, along the surface, or (on an island) away from it. Every case carries **two** knobbed
overlay panels on two distinct faces. One panel is real and opens the case in one of eleven ways
(seven side mechanisms, four top); the other is a **fake** — the same plate and the same knob, screwed
to a solid wall. A decoy case carries two fakes. From outside the two panels are the same panel, so
the only way to learn which is which is to try them, and the cases stand far enough apart that the
robot has to drive from one to the next.

Two, three or four distinct named native items (a lemon, a lime, an orange ...) lie each inside its own
openable case (v1, spec 4.5: nested size ``item_count`` 2, 3, 4 with ``box_count`` 4, 5, 6). The goal
names the items, the bowl and Submit; nothing else. Success is the bowl rule on every item — each
settled inside the bowl, released, supported by the bowl or by another item in it — frozen at the first
physical Submit press.

Placement (generator ``v2``): the cases go anywhere on the work fixture's free top rectangles, decor,
tall fixtures, the bowl and Submit are keep-outs, centres at least 0.55 m apart (relaxed step by step
down to 0.35 m only on a surface that cannot hold them; the achieved spacing is recorded), random
order, random quarter turns (``poses`` stream); a panel, real or fake, may face any side the robot can
reach (island: every side; wall counter: never the wall side); at build time every panel is given a
base stance beyond an accessible edge and a case that cannot be placed with both panels reachable is
not placed (the build fails closed). The bowl and Submit stay at the front middle of the frame.

Passive closers: a subset of the boxes whose mechanism has a natural spring return
(hinged flaps, push doors, slide panels; never twist lids, turn knobs or latched drawers) gets a joint
spring that shuts the panel by itself in twenty to twenty-five seconds (half that to block the hand);
drawn from the ``structure`` stream,
recorded as ``spec['closers']``, applied to the compiled model, never announced.

Appearance (spec 4.9): one case texture, one panel texture and one knob colour per instance from the
``materials`` stream (RoboCasa's wood textures); every case of an instance looks the same.

The sides are named in a *robot frame*: the task frame, turned by 180 degrees on a layout where
RoboCasa starts the robot behind the frame, so "front" always means the aisle the robot starts in
(:mod:`~roboquest.containers.placement`).

Variation
---------
factors
    ``item_count`` {2, 3, 4} (the reported size axis); ``box_count`` = item_count + 2 is derived.
structure
    which boxes hold the items, the holders' mechanisms and the side each real panel faces, the
    held-out class (a third of the mints put one item behind one of the three held-out mechanisms;
    a development instance never shows a held-out mechanism on any box), a decoy, the closers.
nuisance
    the other cases' mechanisms and sides, the placement order, the cases' quarter turns and jitter,
    the placement seed, the item categories, the appearance, the bowl and Submit jitter, the kitchen.
"""
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from scipy.spatial.transform import Rotation

from roboquest import observe as OB
from roboquest.containers import appearance as AP
from roboquest.containers import hiding as H
from roboquest.containers.contract import FACES, SEALED, SIDE_FACES, SIDE_MECHANISMS, TOP_MECHANISMS
from roboquest.containers.mechanism import DARK, ITEM_ASSETS, PANEL, WOOD, add_latch_constraints, add_robocasa_object, latch_released
from roboquest.containers import panels as P
from roboquest.containers import placement as PL
from roboquest.containers.rules import item_in_bowl
from roboquest.containers import springs as SP
from roboquest.kitchen import RoboQuestKitchen, yaw_matrix, yaw_quat_wxyz, _vec
from roboquest.search.layout import base_blockers, floor_bounds_from_walls

CONTRACT_VERSION = 'unfamiliar-containers-kitchen-v3'   # v3: the observability axis (v2: nested sizes)
GENERATOR_VERSION = 'v6'   # v6: closers only where the window they leave the hand is usable (measured; no
                           # closer on slide_side, slower ones on slide_up and slide_lid) and a park spot
                           # for every lift/twist lid's plate (CONTAINERS-CLOSER, 2026-09-22)
                           # v5: stances checked against observe's base footprint, and the pick and prop
                           # stances checked too (v4: slide_side travel 0.245 and the item pool without
                           # the garlic; v3: observability)
COUNT_WORDS = {2: 'both', 3: 'all three', 4: 'all four'}

# ---- the task frame -------------------------------------------------------
# x runs along the front of the work surface, y away from the robot, z up from
# the top. The bowl and the Submit button sit at the front middle, by the
# robot's start; the cases are solved onto the surface at build time.
BOWL_LOCAL = (-.20, -.17)
SUBMIT_LOCAL = (.18, -.19)
PROP_JITTER = .008
FRAME_MARGIN = .01           # the bowl and Submit stay this far inside the footprint band
YAW_JITTER = .06             # rad, on top of the quarter turn
BOWL_CLEARANCE = .05         # the hand over the bowl needs this much beyond the rim
SUBMIT_CLEARANCE = .05
SUBMIT_HALF = (.048, .043)   # plan half extents of the Submit button
TALL_BAND = (.02, .35)       # fixtures reaching into this band above the top block knobs and hands
WALL_PLANE_CLASSES = ('Wall', 'Window', 'WallAccessory', 'Floor', 'Hood')   # in or above the wall: never on the top
PROP_FALLBACK = .45          # the bowl and Submit slide this far to one side only when the middle admits no arrangement
REGION_MIN = (.40, .40)      # smallest top rectangle worth placing on
EDGE_MATCH = .06             # a region edge this close to the fixture's edge is the fixture's edge
FLOOR_MARGIN = .05           # the base stays this far inside the room bounds
LID_PARK_DISTANCE = .45      # a removed top plate is set down this far from its case (teleport and oracle)
OUT_OF_BOX_MARGIN = .03      # an item this far outside its case's interior box counts as out of it
CLOSER_SHARE = .5            # each spring-natural box gets a closer with this probability (at least one)
HOLDOUT_SHARE = 1. / 3.      # share of mints whose structure class is held out (eval by rejection)
DECOY_SHARE = .5             # share of instances with one sealed decoy case
TELEPORT_SETTLE_TICKS = 60
TELEPORT_DROP_TICKS = 40      # per item: the drop onto its own spot comes to rest before the next item
TELEPORT_DROP_ABOVE = .02     # released at most this far above the bowl floor or the pile already in it
TELEPORT_RING_SHARE = 1 / 3.  # each item gets its own spot on a ring of this share of the inner radius
TELEPORT_RESEATS = 3          # an item that ends up outside the bowl is re-seated at most this often
CASE_HALF = PL.CASE_HALF

# Held-out mechanisms (two side, one top), re-derived unchanged at the new size levels. Each pairs an
# action the development split does show (pull, lift) with a precondition or a direction it never
# does: `turn_knob` is the only side mechanism that must be rotated before it pulls, `twist_lid` the
# only top one rotated before it lifts, and `slide_up` the only slide that runs up a face rather than
# across it. Development keeps the whole of the rest of each family (drawer/press_drawer/press_door/
# door/slide_side; lift_lid/slide_lid/flip_lid), so evaluation asks for a composition, not an unseen
# primitive. Every size level appears in both splits: the held-out class is drawn by the structure
# stream at any size.
HOLDOUT_MECHANISMS = ('slide_up', 'turn_knob', 'twist_lid')
DEV_MECHANISMS = tuple(m for m in P.MECHANISMS if m not in HOLDOUT_MECHANISMS)
DEV_SIDE = tuple(m for m in SIDE_MECHANISMS if m not in HOLDOUT_MECHANISMS)
DEV_TOP = tuple(m for m in TOP_MECHANISMS if m not in HOLDOUT_MECHANISMS)
SPLIT_JOIN = '+'

# Layouts whose chosen work surface is an island, measured with this footprint over RoboCasa's
# layouts 1-60 at style 1 (the layout gate's build; v0 knew only 4 and 17, which cost sixteen island
# kitchens at the gate). Only on an island may a panel face away from the robot, because the base has
# to drive round to reach it; build_task re-checks the real fixture and fails closed. KITCHENS below
# overrides this guess for every layout in the v1 pools: it was measured on the surface the build
# actually picks, and layout 35's guess here was wrong.
ISLAND_LAYOUTS = (4, 12, 16, 17, 20, 23, 25, 26, 32, 35, 38, 41, 45, 48, 49, 51, 54, 55)
# Layouts measured to offer no island or counter with a free 1.80 x 0.60 m top (same measurement), so
# the kitchen cannot be built at all. A deny-list rather than an allow-list: widening the layout pool
# needs a build gate (G1), not an edit here.
NO_WORK_SURFACE_LAYOUTS = (1, 3, 21, 28, 31, 34, 36, 37, 40, 43, 44, 47, 50, 52, 57, 58, 60)
# What a kitchen can actually do, one row per layout: the largest size level its work surface arranges
# and whether that surface is an island. Probed 2026-09-21 (style 1, the first mintable seed per size
# level, build to the arrangement) over both v1 pools, after the first v1 registry spent about 25 s per
# candidate discovering the same four capacity failures again and again and three candidates on layout
# 35 failed the island check. A mint whose kitchen is in this table fails the CPU gate when it asks for
# more cases than the row allows, and takes its island flag from the row rather than from the guess in
# ISLAND_LAYOUTS. Unprobed layouts keep the old behaviour: the layout gate and G1 decide.
PROBE_DATE = '2026-09-21'
KITCHENS = {
    # layout: (largest item_count arranged, work surface is an island)
    4: (4, True), 5: (3, False), 9: (2, False), 12: (4, True), 16: (4, True), 17: (4, True),
    20: (3, True), 23: (3, True), 24: (3, False), 25: (4, True), 26: (2, True), 32: (4, True),
    35: (4, True), 38: (4, True), 39: (2, False), 41: (4, True), 45: (4, True), 48: (4, True),
    49: (4, True), 51: (4, True), 54: (4, True), 55: (4, True),
}
LAYOUT_MAX_ITEMS = {layout: row[0] for layout, row in KITCHENS.items()}

# Which surface the build picks is not the layout's business alone: the style dresses the kitchen, and
# on layout 35 at style 7 the island is taken, so the build falls back to counter_2_right_group_1 - a
# wall counter. That is why three layout-35 candidates of the first v1 registry died on the island
# check. Rows here are measured per (layout, style) and win over the layout's row above; probed over
# the ten development styles of layouts 23, 26 and 35 on 2026-09-21.
KITCHEN_STYLES = {(35, 7): False}


def layout_is_island(layout, style=None):
    """Whether the work surface the build picks on this kitchen is an island: the per-style measurement
    where there is one, then the layout's probed row, then the footprint guess in ISLAND_LAYOUTS."""
    layout = int(layout)
    if style and (layout, int(style)) in KITCHEN_STYLES:
        return bool(KITCHEN_STYLES[(layout, int(style))])
    if layout in KITCHENS:
        return bool(KITCHENS[layout][1])
    return layout in ISLAND_LAYOUTS
ITEM_NAMES = tuple(ITEM_ASSETS)


def panel_geom_names(index, face, real):
    """The three geoms a knobbed panel shows: the plate, the knob and its cap.

    A real panel hangs on the moving body and a fake is screwed to the wall, so
    they are built by different calls; the names differ but the shapes and
    materials must not. `real` says which of the case's two panels this is.
    """
    prefix = f'pb{index}'
    if face == 'top':
        stem = prefix if real else f'{prefix}_top'
        plate = f'{prefix}_lid' if real else f'{prefix}_top_plate'
    else:
        stem = prefix if real else f'{prefix}_{face}'
        plate = f'{prefix}_panel' if real else f'{prefix}_{face}_panel'
    return [plate, f'{stem}_knob', f'{stem}_knob_cap']


def bowl_polygon(xy):
    return PL.circle_polygon(xy, P.BOWL_OUTER_RADIUS)


def submit_polygon(xy):
    return PL.rect(xy, 0., SUBMIT_HALF[0], SUBMIT_HALF[1])


def lid_yaw_for(edge, yaw_quarter):
    """Quarter turns of a top plate so that its front (the side its slide runs from, or the edge
    opposite its hinge) faces the accessible edge the base works it from. Invisible from outside (the
    plate is square and its knob central), so it is chosen at build time from the placement."""
    return (PL.SIDES.index(edge) - int(yaw_quarter)) % 4 * PL.QUARTER


def article(name):
    return ('an ' if name[0].lower() in 'aeiou' else 'a ') + name


def item_list(names):
    """'a lemon and a lime' / 'a lemon, a lime and an orange'."""
    words = [article(n) for n in names]
    if len(words) == 1:
        return words[0]
    return ', '.join(words[:-1]) + ' and ' + words[-1]


def box_count_for(item_count):
    return int(item_count) + 2


class UnfamiliarContainers(RoboQuestKitchen):
    TASK_NAME = 'unfamiliar_containers'
    CONTRACT_VERSION = CONTRACT_VERSION
    GENERATOR_VERSION = GENERATOR_VERSION
    # Spec 2.1: the reported axes are the size and the observability level. Six eval cells of nine.
    # There is no ``uncover`` level: the cases are the covers of this task, and no catalogue occluder
    # hides one either (``containers.hiding``), so ``look`` means one case stands in an annex.
    FACTORS = {'item_count': (2, 3, 4), 'observability': H.LEVELS}
    # The footprint is what the base class asks the work surface for: the band the bowl and Submit sit
    # in and the least top a surface must offer. The cases are placed over the fixture's whole free top,
    # not inside the footprint. Kept at the v0 value so the work-fixture choice, the measured
    # ISLAND_LAYOUTS and the layout pools stay valid.
    FOOTPRINT = (1.80, .60)
    HOLDOUT_KEYS = HOLDOUT_MECHANISMS
    # The panel-boxes reel needed 19,142 ticks for eleven boxes with a fixed tour. Provisional (contract
    # C1: the v0 horizon); to be set from the G4 certificates (about three times the oracle's duration).
    HORIZON = 26000
    BUDGET_TICKS = 26000
    submit_prefix = 'uc_submit'

    # ---- generator protocol -------------------------------------------------
    @classmethod
    def goal(cls, spec):
        """Spec 1.3 framing (no clutter sentence: nothing else on the counter is meant to be moved)."""
        names = list(spec['item_names'])
        listed = item_list(names)
        listed = listed[0].upper() + listed[1:]
        middle = f'{listed} are inside them. Put {COUNT_WORDS[len(names)]} in the bowl, then press Submit.'
        return cls.frame_goal(spec, 'boxes', int(spec['box_count']), middle, clutter=False)

    @classmethod
    def sample_spec(cls, streams, factors, seed):
        structure, poses, materials = streams['structure'], streams['poses'], streams['materials']
        item_count = int(factors['item_count'])
        count = box_count_for(item_count)
        island = layout_is_island(factors['kitchen_layout'], factors.get('kitchen_style'))
        sides = PL.reachable_sides(island)
        # Structure: the held-out class, the holders (one box per item), the decoy.
        pool = 'holdout' if float(structure.uniform()) < HOLDOUT_SHARE else 'dev'
        holders = sorted(int(i) for i in structure.choice(count, size=item_count, replace=False))
        held = holders[int(structure.integers(item_count))] if pool == 'holdout' else None
        remaining = [i for i in range(count) if i not in holders]
        decoy_count = 1 if float(structure.uniform()) < DECOY_SHARE else 0
        decoy_ids = sorted(int(i) for i in structure.choice(remaining, size=decoy_count, replace=False)) if decoy_count else []
        order = [int(i) for i in poses.permutation(count)]

        boxes = []
        for index in range(count):
            quarter = int(poses.integers(4))
            yaw = quarter * PL.QUARTER + float(poses.uniform(-YAW_JITTER, YAW_JITTER))
            if index == held:
                mechanisms = HOLDOUT_MECHANISMS
            elif index in decoy_ids:
                mechanisms = (SEALED,)
            else:
                mechanisms = DEV_MECHANISMS
            mechanism, side, dummy_side = cls._draw_panels(structure, mechanisms, sides)
            boxes.append(dict(
                index=index, yaw_quarter=quarter, yaw=round(yaw, 6),
                mechanism=mechanism, side=side, dummy_side=dummy_side,
                face=PL.face_for_side(yaw, side), dummy_face=PL.face_for_side(yaw, dummy_side),
                direction='right' if structure.integers(2) else 'left',
                hinge='left' if structure.integers(2) else 'right'))
        closers = SP.draw_closers(structure, boxes, share=CLOSER_SHARE)
        item_names = [str(n) for n in materials.choice(ITEM_NAMES, size=item_count, replace=False)]
        spec = dict(
            item_count=item_count, box_count=count, holder_pool=pool, boxes=boxes,
            holders=holders, held_out_holder=held, decoys=decoy_ids, decoy_count=decoy_count,
            holder_mechanisms=[boxes[h]['mechanism'] for h in holders],
            closers=closers, work_fixture_is_island=island, kitchen_layout=int(factors['kitchen_layout']),
            kitchen_style=int(factors.get('kitchen_style') or 0),
            item_names=item_names, item_assets=[ITEM_ASSETS[n][0] for n in item_names],
            appearance=AP.draw_appearance(materials),
            placement_order=order, min_spacing=PL.SPACING_LADDER[0],
            layout_seed=int(poses.integers(0, 2 ** 31 - 1)),
            placement_seed=int(poses.integers(0, 2 ** 31 - 1)),
            bowl_xy=[round(BOWL_LOCAL[0] + float(poses.uniform(-PROP_JITTER, PROP_JITTER)), 6),
                     round(BOWL_LOCAL[1] + float(poses.uniform(-PROP_JITTER, PROP_JITTER)), 6)],
            submit_xy=[round(SUBMIT_LOCAL[0] + float(poses.uniform(-PROP_JITTER, PROP_JITTER)), 6),
                       round(SUBMIT_LOCAL[1] + float(poses.uniform(-PROP_JITTER, PROP_JITTER)), 6)],
            footprint=list(cls.FOOTPRINT))
        # Observability (spec 2.2) last of all, from the poses stream after every other draw, so two
        # specs that differ only in the level are identical everywhere else - same kitchen, same
        # mechanisms, same arrangement seeds, same goal - and the axis is the only thing that moved.
        # Which case is hidden is nuisance: a fair coin between the holders and the rest.
        level = str(factors.get('observability', 'visible'))
        spec['observability'] = level
        spec['hidden'] = H.draw_hidden(poses, level, H.box_names(spec),
                                       [f'box{index}' for index in holders])
        return spec

    @classmethod
    def _draw_panels(cls, rng, mechanisms, sides):
        """One real mechanism on a compatible reachable side plus a fake on another reachable side."""
        upright = [s for s in sides if s != 'top']
        options = []
        for candidate in mechanisms:
            if candidate == SEALED:
                options.append(candidate)
            elif candidate in TOP_MECHANISMS:
                if 'top' in sides:
                    options.append(candidate)
            elif upright:
                options.append(candidate)
        if not options:
            raise ValueError(f'no mechanism fits the sides {sides}')
        mechanism = str(rng.choice(sorted(options)))
        if mechanism in TOP_MECHANISMS:
            side = 'top'
        elif mechanism == SEALED:
            side = str(rng.choice(sorted(sides)))
        else:
            side = str(rng.choice(sorted(upright)))
        others = sorted(set(sides) - {side})
        return mechanism, side, str(rng.choice(others))

    @classmethod
    def validate_spec(cls, spec):
        problems = []
        boxes = spec['boxes']
        item_count = int(spec['item_count'])
        if item_count not in cls.FACTORS['item_count']:
            problems.append('item_count outside its levels')
        if len(boxes) != spec['box_count'] or spec['box_count'] != box_count_for(item_count):
            problems.append('box_count does not match the box list or the item count')
        holders = list(spec['holders'])
        if len(holders) != item_count or len(set(holders)) != item_count or any(not 0 <= h < len(boxes) for h in holders):
            problems.append('holders must be item_count distinct boxes')
        if len(spec['decoys']) != spec['decoy_count'] or set(spec['decoys']) & set(holders):
            problems.append('decoys do not match decoy_count or include a holder')
        for h in holders:
            if boxes[h]['mechanism'] == SEALED:
                problems.append(f'holder box {h} is sealed')
        names = list(spec['item_names'])
        if len(names) != item_count or len(set(names)) != item_count or any(n not in ITEM_ASSETS for n in names):
            problems.append('item_names must be item_count distinct known items')
        if sorted(spec['placement_order']) != list(range(len(boxes))):
            problems.append('placement_order is not a permutation of the boxes')
        # Closers only where the window they leave the hand is usable (springs.usable_closer): a closer
        # on a slide_side is a refusal the walk redraws, never a spec silently stripped of it.
        problems.extend(SP.closer_problems(boxes, spec['closers']))
        if spec['holder_pool'] not in ('dev', 'holdout'):
            problems.append('unknown holder pool')
        for box in boxes:
            if box['face'] == box['dummy_face'] or box['side'] == box['dummy_side']:
                problems.append(f"box {box['index']}: the two panels share a face")
            if box['face'] not in FACES or box['dummy_face'] not in FACES:
                problems.append(f"box {box['index']}: unknown face")
            if box['side'] not in PL.SIDES + ('top',) or box['dummy_side'] not in PL.SIDES + ('top',):
                problems.append(f"box {box['index']}: unknown side")
            if box['mechanism'] not in P.MECHANISMS + (SEALED,):
                problems.append(f"box {box['index']}: unknown mechanism {box['mechanism']!r}")
        problems.extend(H.level_problems(spec, H.box_names(spec)))
        return problems

    @classmethod
    def cpu_gate(cls, spec):
        """G0: mechanism/side/face compatibility, the reachable-side rule for the layout's kind of
        surface, the held-out mechanism rule, and the bowl and Submit inside the footprint band without
        touching. The cases' positions are solved on the real surface at build time (G1)."""
        report, problems = {}, []
        boxes = spec['boxes']
        island = bool(spec['work_fixture_is_island'])
        if island != layout_is_island(spec['kitchen_layout'], spec.get('kitchen_style')):
            problems.append('work_fixture_is_island disagrees with the kitchen table')
        if int(spec['kitchen_layout']) in NO_WORK_SURFACE_LAYOUTS:
            problems.append(f"layout {spec['kitchen_layout']} has no free {cls.FOOTPRINT} m work surface")
        capacity = LAYOUT_MAX_ITEMS.get(int(spec['kitchen_layout']))
        if capacity is not None and int(spec['item_count']) > capacity:
            problems.append(f"layout {spec['kitchen_layout']} holds at most {box_count_for(capacity)} cases "
                            f"(item_count {capacity}), not {spec['box_count']}")
        # The annexed case is still counted above: the same kitchen has to serve both levels of a seed,
        # or the level would decide which kitchens the registry may draw.
        hidden_names = [entry['object'] for entry in H.hidden_entries(spec)]
        if any(name not in H.box_names(spec) for name in hidden_names):
            problems.append(f'the hidden list names a case that is not in this instance: {hidden_names}')
        report['observability'] = dict(level=spec.get('observability', 'visible'), hidden=hidden_names)
        reachable = PL.reachable_sides(island)
        for box in boxes:
            index, mechanism = box['index'], box['mechanism']
            if abs(box['yaw'] - box['yaw_quarter'] * PL.QUARTER) > YAW_JITTER + 1e-6:
                problems.append(f'box {index}: yaw {box["yaw"]} is not quarter turn {box["yaw_quarter"]} plus jitter')
            for label, face, side in (('real', box['face'], box['side']), ('fake', box['dummy_face'], box['dummy_side'])):
                if PL.world_side(box['yaw'], face) != side:
                    problems.append(f'box {index}: the {label} panel on face {face} does not look at side {side}')
                if side not in reachable:
                    problems.append(f'box {index}: the {label} panel faces {side}, unreachable on '
                                    f'{"an island" if island else "a counter"} (allowed {reachable})')
            if mechanism in TOP_MECHANISMS and box['side'] != 'top':
                problems.append(f'box {index}: {mechanism} is a top mechanism on side {box["side"]}')
            if mechanism in SIDE_MECHANISMS and box['side'] not in PL.SIDES:
                problems.append(f'box {index}: {mechanism} needs an upright side, not {box["side"]}')
            if box['side'] == box['dummy_side']:
                problems.append(f'box {index}: the two panels share a side')
        holders = list(spec['holders'])
        used = {box['mechanism'] for box in boxes}
        held_out_used = used & set(HOLDOUT_MECHANISMS)
        pool = spec['holder_pool']
        if pool == 'dev' and held_out_used:
            problems.append(f'dev instance uses held-out mechanisms {sorted(held_out_used)}')
        if pool == 'holdout':
            held = spec.get('held_out_holder')
            if held not in holders or boxes[held]['mechanism'] not in HOLDOUT_MECHANISMS:
                problems.append('holdout instance without a held-out holder')
            others = {b['mechanism'] for b in boxes if b['index'] != held} - {SEALED}
            if others & set(HOLDOUT_MECHANISMS):
                problems.append('a box other than the held-out holder uses a held-out mechanism')
        if cls.is_holdout(cls.split_key(spec)) != (pool == 'holdout'):
            problems.append('split key disagrees with the holder pool')
        # The closer window rule (springs.usable_closer; validate_spec refuses it too, so the walk redraws).
        problems.extend(SP.closer_problems(boxes, spec['closers']))

        # The fixed props, in the task frame: inside the footprint band and apart.
        half = np.array(cls.FOOTPRINT) / 2 - FRAME_MARGIN
        bowl, submit = bowl_polygon(spec['bowl_xy']), submit_polygon(spec['submit_xy'])
        for name, polygon in (('bowl', bowl), ('submit', submit)):
            if np.any(np.abs(polygon) > half + 1e-9):
                problems.append(f'{name} outside the footprint')
        if PL.polygons_overlap(bowl, submit, margin=BOWL_CLEARANCE):
            problems.append('the bowl and the Submit button touch')
        report.update(box_count=len(boxes), item_count=int(spec['item_count']), decoys=spec['decoys'],
                      holders=holders, holder_mechanisms=[boxes[h]['mechanism'] for h in holders],
                      closers=list(spec['closers']),
                      closer_windows={int(i): SP.CLOSER_WINDOW_TICKS.get(boxes[i]['mechanism'])
                                      for i in spec['closers'] if 0 <= int(i) < len(boxes)},
                      island=island, mechanisms=sorted(used),
                      sides=sorted({b['side'] for b in boxes} | {b['dummy_side'] for b in boxes}), problems=problems)
        return not problems, report

    @classmethod
    def split_key(cls, spec):
        """The holders' mechanisms, sorted and joined: the structure class the hold-out reads."""
        return SPLIT_JOIN.join(sorted(spec['boxes'][h]['mechanism'] for h in spec['holders']))

    @classmethod
    def is_holdout(cls, split_key):
        return any(part in HOLDOUT_MECHANISMS for part in str(split_key).split(SPLIT_JOIN))

    # ---- the robot frame ----------------------------------------------------
    def _robot_start_anchor(self):
        """RoboCasa's start pose for the robot (world xy, yaw), asked for at build time.

        RoboCasa computes ``init_robot_base_pos_anchor`` only after ``_create_objects``, so the task
        asks its placement function itself; the answer is the same because it depends only on the
        fixtures (the robot stands at the work fixture's front, or on the stool side of a dining
        counter). None when it cannot be computed."""
        import robocasa.utils.env_utils as EnvUtils
        anchor = getattr(self, 'init_robot_base_pos_anchor', None)
        ori = getattr(self, 'init_robot_base_ori_anchor', None)
        if anchor is None:
            had_cfgs = hasattr(self, 'object_cfgs')
            if not had_cfgs or self.object_cfgs is None:
                self.object_cfgs = []
            try:
                anchor, ori = EnvUtils.init_robot_base_pose(self)
            except Exception:  # noqa: BLE001 - the frame then keeps RoboCasa's convention
                return None
            finally:
                if not had_cfgs:
                    del self.object_cfgs
        if anchor is None or ori is None:
            return None
        yaw = float(np.asarray(ori, float).reshape(-1)[-1])
        return [float(anchor[0]), float(anchor[1])], yaw

    def _robot_frame(self):
        """The task frame, turned by 180 degrees when RoboCasa starts the robot behind it (the stool
        side of a dining counter), so that 'front' is the aisle the robot starts in."""
        start = self._robot_start_anchor()
        flipped = False
        if start is not None and self.work:
            heading = start[1] - float(self.work['yaw'])          # pi/2 when the robot faces +y (RoboCasa's front)
            flipped = bool(math.cos(heading - math.pi / 2) < 0.)
        self._flip = -1. if flipped else 1.
        self._flip_yaw = math.pi if flipped else 0.
        self._robot_start = start
        return flipped

    def robot_to_frame_xy(self, xy):
        return [float(self._flip * xy[0]), float(self._flip * xy[1])]

    def frame_to_robot_xy(self, xy):
        return self.robot_to_frame_xy(xy)

    def robot_to_world_xy(self, xy):
        local = self.robot_to_frame_xy(xy)
        return self.frame_to_world([local[0], local[1], 0.])[:2].tolist()

    def robot_yaw_to_world(self, yaw):
        return float(self.work['yaw']) + self._flip_yaw + float(yaw)

    def world_to_robot_xy(self, xy):
        return self.robot_to_frame_xy(OB.world_to_frame_xy(self, xy))

    def world_yaw_to_robot(self, yaw):
        return float(yaw) - float(self.work['yaw']) - self._flip_yaw

    def world_dir_to_robot(self, direction):
        """A plan direction from world axes to the robot frame's (the frame's turn, then its flip)."""
        return self._flip * (H.rot2(-float(self.work['yaw'])) @ np.asarray(direction, float)[:2])

    def _world_box_to_robot_polygon(self, box):
        corners = [(box['x0'], box['y0']), (box['x1'], box['y0']), (box['x1'], box['y1']), (box['x0'], box['y1'])]
        rotation = yaw_matrix(-float(self.work['yaw']))
        origin = np.asarray(self.work['center_world'], float)
        out = []
        for x, y in corners:
            local = rotation @ (np.array([x, y, origin[2]]) - origin)
            out.append([self._flip * local[0], self._flip * local[1]])
        return np.asarray(out)

    # ---- the surface --------------------------------------------------------
    def _surface_regions(self):
        """Every free top rectangle of the work fixture in the robot frame, with its accessible edges."""
        fixture = self.work_fixture
        island = bool(fixture_is_type(fixture, FixtureType.ISLAND))
        cx, cy = float(self.work['center_local'][0]), float(self.work['center_local'][1])
        sx, sy = float(fixture.size[0]), float(fixture.size[1])
        top = [-cx - sx / 2, -cx + sx / 2, -cy - sy / 2, -cy + sy / 2]          # task frame
        if self._flip < 0:
            top = [-top[1], -top[0], -top[3], -top[2]]
        regions = []
        for name, region in sorted(fixture.get_reset_regions(self, top_size=REGION_MIN).items()):
            off, size = region['offset'], region['size']
            rect = [float(off[0] - size[0] / 2 - cx), float(off[0] + size[0] / 2 - cx),
                    float(off[1] - size[1] / 2 - cy), float(off[1] + size[1] / 2 - cy)]
            if self._flip < 0:
                rect = [-rect[1], -rect[0], -rect[3], -rect[2]]
            edges = ['front']
            if island:
                if abs(rect[3] - top[3]) < EDGE_MATCH:
                    edges.append('back')
                if abs(rect[0] - top[0]) < EDGE_MATCH:
                    edges.append('left')
                if abs(rect[1] - top[1]) < EDGE_MATCH:
                    edges.append('right')
            regions.append(dict(name=name, rect=[round(v, 4) for v in rect], edges=tuple(edges)))
        return regions, [round(v, 4) for v in top], island

    def _plan_keepouts(self, top_z):
        """Plan polygons (robot frame) nothing may stand on or reach into: RoboCasa's counter decor and
        every fixture that reaches into the band above the top (fridges, ovens, walls, windows)."""
        keepouts = []
        for decor in self._decor_boxes(top_z):
            keepouts.append((decor['name'], self._world_box_to_robot_polygon(decor)))
        seen = {name for name, _ in keepouts}
        for name, fixture in self.fixtures.items():
            if name in seen or fixture is self.work_fixture or type(fixture).__name__ in WALL_PLANE_CLASSES:
                continue
            if fixture_is_type(fixture, FixtureType.COUNTER):
                continue
            try:
                points = np.asarray(fixture.get_bbox_points(), float)
            except Exception:
                continue
            if points[:, 2].max() < top_z + TALL_BAND[0] or points[:, 2].min() > top_z + TALL_BAND[1]:
                continue
            keepouts.append((name, self._world_box_to_robot_polygon(dict(
                x0=float(points[:, 0].min()), x1=float(points[:, 0].max()),
                y0=float(points[:, 1].min()), y1=float(points[:, 1].max())))))
        return keepouts

    def _fixture_records(self):
        records = {}
        for name, fixture in self.fixtures.items():
            try:
                pos = [float(v) for v in fixture.pos]
            except Exception:
                pos = None
            records[name] = dict(name=name, cls=type(fixture).__name__, pos=pos,
                                 rot=float(getattr(fixture, 'rot', 0.) or 0.),
                                 size=[float(v) for v in getattr(fixture, 'size', [0., 0., 0.])])
        return records

    def floor_map(self):
        """World floor bounds (from the walls; None when they cannot be found) and the plan footprints
        the base cannot drive through, for the stance checks here and the oracle's route planning."""
        records = self._fixture_records()
        try:
            bounds = [float(v) for v in floor_bounds_from_walls(records)]
        except ValueError:
            bounds = None
        blockers = {name: [[float(x), float(y)] for x, y in polygon] for name, polygon in base_blockers(records).items()}
        return dict(bounds=bounds, blockers=blockers)

    def _stance_checker(self, floor):
        """``stance_free(xy, heading)`` in the robot frame: the base's footprint at that stance overlaps
        no fixture footprint and stays inside the room.

        The footprint is :func:`containers.placement.stance_polygon`, the one ``observe.stance_gate``
        tests, not the narrower collision box the knob recipes were tuned against: the build must not
        accept a stance the oracle's floor planner will refuse to drive to (job CONTAINERS-REACH). As
        in that gate the fixtures are cleared by the rectangle shrunk by the touch tolerance and the
        room bounds by the full one."""
        blockers = [PL._Poly(polygon) for polygon in floor['blockers'].values()]
        bounds = floor['bounds']
        inner = None if bounds is None else [bounds[0] + FLOOR_MARGIN, bounds[1] - FLOOR_MARGIN,
                                             bounds[2] + FLOOR_MARGIN, bounds[3] - FLOOR_MARGIN]

        def stance_free(xy, heading):
            world, yaw = self.robot_to_world_xy(xy), self.robot_yaw_to_world(heading)
            if inner is not None and not PL._Poly(PL.stance_polygon(world, yaw, margin=0.)).inside(inner, 0.):
                return False
            polygon = PL._Poly(PL.stance_polygon(world, yaw))
            return not any(polygon.overlaps(b, 0.) for b in blockers)
        return stance_free

    @staticmethod
    def _prop_stance_problems(bowl_xy, submit_xy, stance_free):
        """The bowl and Submit are dropped into and pressed from a stance in the robot's own aisle,
        :data:`containers.placement.PROP_STAND_OFF` in front of them (oracle ``prop_stance``). On a
        kitchen whose aisle is narrow, or whose frame sits close to the opposite run of units, that
        stance is not free floor and the episode cannot finish, so the arrangement is refused here and
        the prop offset (then the seed) is walked on (job CONTAINERS-REACH)."""
        problems = []
        for label, xy in (('bowl', bowl_xy), ('submit', submit_xy)):
            point, heading = PL.prop_stance_for(xy)
            if not stance_free(point, heading):
                problems.append(f'the {label} stance at ({point[0]:+.2f}, {point[1]:+.2f}) is not free floor')
        return problems

    def _stance_world(self, stance):
        if stance is None:
            return None
        out = dict(stance)
        out['xy_world'] = self.robot_to_world_xy(stance['xy'])
        out['heading_world'] = self.robot_yaw_to_world(stance['heading'])
        return out

    # ---- observability (spec 2.2, contract C6) ------------------------------
    def hiding_rng(self):
        """The build-time RNG for the hiding: the instance seed and observe's salt, like every task."""
        seed = int(self.instance.get('seed', 0)) & 0xFFFFFFFF
        return np.random.default_rng([seed, OB.OBSERVE_SALT])

    def _annex_frame(self, region, surface):
        """The annex surface as ``containers.placement`` wants it: an axis-aligned rectangle in the robot
        frame and the accessible edges of that rectangle, named in the robot frame.

        ``surface`` is the whole top region the annex cell was cut out of, so a stance is pushed out to
        the real edge of the surface and not to the inner edge of the cell."""
        corners = [OB.region_to_world(surface, (x, y))
                   for x, y in ((surface['rect'][0], surface['rect'][2]), (surface['rect'][1], surface['rect'][2]),
                                (surface['rect'][1], surface['rect'][3]), (surface['rect'][0], surface['rect'][3]))]
        rect = H.inscribed_rect([self.world_to_robot_xy(c) for c in corners])
        edges = {}
        for edge in OB.accessible_edges(surface):
            direction = H.rot2(float(surface['yaw'])) @ np.asarray(OB.EDGES[edge], float)
            edges[H.edge_label(self.world_dir_to_robot(direction))] = edge
        return rect, tuple(sorted(edges))

    def _annex_panels(self, box, xy_robot, yaw_robot, rect, edges, base, floor, stance_free):
        """Both panels of a case standing at ``xy_robot`` on the annex surface, worked from a stance
        beside it: the placement rules first (an accessible edge the panel looks at, inside the arm's
        depth, the base's footprint on free floor), then the arm's own reach to the *knob*, which is
        what the robot actually has to touch. None when either panel fails."""
        out = {}
        lid_lift = P.SLIDE_LID_LIFT if box['mechanism'] == 'slide_lid' else 0.
        for label, face in (('real', box['face']), ('fake', box['dummy_face'])):
            stance = PL.panel_stance(xy_robot, yaw_robot, face, rect, edges, stance_free)
            if stance is None:
                return None, f'the {label} panel on face {face} is unreachable from the annex surface'
            knob = H.knob_grip_world(base, self.robot_yaw_to_world(yaw_robot), face, lid_lift=lid_lift)
            world = self._stance_world(stance)
            gate = OB.reach_gate(self, knob, floor=floor, at_build=True,
                                 base_pose=(np.asarray(world['xy_world'], float), world['heading_world']))
            if not gate['passed']:
                return None, f"the {label} knob is out of reach from the annex stance: {'; '.join(gate['reasons'])}"
            out[label] = dict(stance=world, knob=[round(float(v), 4) for v in knob],
                              reach={k: gate[k] for k in ('passed', 'ahead_m', 'lateral_m', 'slide_m',
                                                          'residual_lateral_m')})
        return out, None

    def _annex_case(self, box, floor, stance_free, keepouts_world=()):
        """Stand one case off the work surface: out of every policy camera at reset, square to the work
        frame so the placement rules still hold, and with both of its knobs reachable from a stance at
        the annex surface. Raises when no annex on this kitchen can do it (the instance fails at G1).

        The yaw is chosen here rather than taken from the spec, because which way a case has to look to
        be workable depends on the surface it ends up on; the spec's drawn quarter says where the search
        starts, so the draw still shows.

        Feasibility, probed on 2026-09-21 over the six kitchens the brief names (layouts 4, 12, 16, 17
        with islands; 5 and 20 with wide counters), styles 3 and 7, sizes 2/3/4, one seed each: 30 of the
        34 cells in capacity found an annex. Layouts 4, 12, 16 and 20 found one at every size, and the
        two-item cells of 17 and 5 were the only misses. Those misses are the panel draw, not the
        kitchen: what the rejects complain about is 'the real panel on face front is unreachable from
        the annex stance' (5 of 11 regions on 17, 5 of 12 on 5), and the same layout and size annexes
        happily on other seeds - 17 passes G1 at two items on seeds 4 and 1000000. An annex on a wall
        counter costs a longer drive than one on an island (2.2-5.1 m against 1.3-1.5 m).
        """
        rng = self.hiding_rng()
        frusta = OB.expected_camera_frusta(self, at_build=True)
        surfaces = {(s['fixture'], s['region']): s for s in OB.surface_regions(self)}
        work_yaw = float(self.work['yaw'])
        jitter = float(box['yaw']) - int(box['yaw_quarter']) * PL.QUARTER
        radius = H.footprint_radius(jitter)
        regions = OB.annex_regions(self, floor=floor, frusta=frusta, keepouts_world=list(keepouts_world),
                                   object_height=H.CASE_HEIGHT, object_radius=radius)
        # Cells wholly out of the cameras first, in a random order, then the partly hidden ones: a spot in
        # a half-seen cell is only as hidden as the frustum prediction, and G3 has the last word anyway.
        order = sorted(range(len(regions)), key=lambda i: (regions[i]['hidden_fraction'] < .999,
                                                           float(rng.random())))
        rejects, spots, turns = [], 0, 0
        for step in order:
            region = regions[step]
            if not H.is_square(region['yaw'], work_yaw):
                rejects.append(dict(region=region['name'], why='the surface is not square to the work frame'))
                continue
            surface = surfaces.get((region['fixture'], region['region']))
            if surface is None:
                rejects.append(dict(region=region['name'], why='the surface went away between the two passes'))
                continue
            # Which way the case may look depends on the surface, not on the spot, so the turns are
            # settled before a single spot is drawn.
            allowed = []
            for turn in H.quarter_turns(box['yaw_quarter']):
                yaw_robot = turn * PL.QUARTER + jitter
                sides = [PL.world_side(yaw_robot, face) for face in H.panel_faces(box)]
                if all(side in PL.reachable_sides(bool(region['island'])) for side in sides):
                    allowed.append((turn, yaw_robot, sides))
            if not allowed:
                rejects.append(dict(region=region['name'],
                                    why=f"no quarter turn puts both panels on a reachable side of a "
                                        f"{'island' if region['island'] else 'wall'} surface"))
                continue
            rect, edges = self._annex_frame(region, surface)
            why = None
            for attempt in range(H.ANNEX_TRIES):
                placed = OB.place_in_annex(self, region, rng, radius=radius, height=H.CASE_HEIGHT,
                                           frusta=frusta, min_gap=H.CORNER_GAP)
                if placed is None:
                    why = why or 'no spot in this cell is out of every camera'
                    break
                spots += 1
                xy_robot = self.world_to_robot_xy(placed['xy_world'])
                for turn, yaw_robot, sides in allowed:
                    turns += 1
                    base = [float(placed['xy_world'][0]), float(placed['xy_world'][1]), float(region['top_z'])]
                    panels, why = self._annex_panels(box, xy_robot, yaw_robot, rect, edges,
                                                     base, floor, stance_free)
                    if panels is None:
                        continue
                    park = None
                    if box['mechanism'] in P.FREE_LID_MECHANISMS:
                        # The lid park rule on the annex surface: the plate needs a spot beside or behind
                        # the case on that surface with a free top stance within reach (CONTAINERS-CLOSER).
                        blockers = [PL.case_park_block(xy_robot, yaw_robot)] + PL.prop_park_blockers(
                            self.spec['bowl_xy'], self.spec['submit_xy'])
                        found = PL.find_park_spot(xy_robot, panels['real']['stance']['edge'], rect, blockers,
                                                  stance_free)
                        if found is None:
                            why = 'no park spot on the annex surface for the plate'
                            continue
                        park = dict(spot=[float(v) for v in found['spot']],
                                    stance_xy=[float(v) for v in found['stance_xy']],
                                    heading=float(found['heading']), where=found['where'])
                    return dict(box=f"box{box['index']}", index=int(box['index']), mode=H.LOOK_MODE, park=park,
                                region=region['name'], fixture=region['fixture'], kind=region['kind'],
                                island=bool(region['island']), top_z=float(region['top_z']),
                                rect_robot=rect, edges_robot=list(edges),
                                xy_world=list(placed['xy_world']), yaw_world=self.robot_yaw_to_world(yaw_robot),
                                xy_robot=list(xy_robot), yaw_robot=float(yaw_robot), yaw_turn=int(turn),
                                sides=sides, hidden_fraction=float(region['hidden_fraction']),
                                drive_m=float(region['drive_m']), stance=placed['stance'],
                                reach={label: panels[label]['stance'] for label in panels},
                                knobs={label: dict(point=panels[label]['knob'], **panels[label]['reach'])
                                       for label in panels},
                                spots=spots, turns=turns, rejects=rejects)
            rejects.append(dict(region=region['name'], why=why or 'no spot and turn worked'))
        raise ValueError(f"{self.TASK_NAME}: no annex on this kitchen holds case {box['index']} out of every "
                         f"camera within reach of its knobs ({len(regions)} regions, {spots} spots, "
                         f"{turns} turns): {rejects}")

    def _observability_record(self, annex, boxes):
        """The record every task stores under ``task_spec['observability']``, in observe's shape. It is
        filled here rather than by ``observe.apply_observability`` because a case cannot be moved after
        the build: its moving panel is a second top-level body and its item is a third, and posing the
        shell alone would leave both behind. The build stands the case in its annex instead."""
        requested = [dict(entry) for entry in (self.spec.get('hidden') or [])]
        record = dict(level=self.spec.get('observability', 'visible'), requested=requested, realised=[],
                      hidden_bodies=[], covers=[], occluders=[], annex=[], fallbacks=[], problems=[])
        if annex is None:
            record['problems'] = [f"the level asks for {entry['object']} to be hidden and it was not"
                                  for entry in requested]
            return record
        info = boxes[annex['box']]
        record['annex'].append(dict(annex, object=annex['box'], radius=H.CASE_RADIUS, height=H.CASE_HEIGHT,
                                    moved=True))
        record['hidden_bodies'] = [name for name in (info['shell_body'], info['moving_body']) if name]
        record['realised'] = [dict(entry, realised=H.LOOK_MODE, region=annex['region'])
                              for entry in requested]
        return record

    def hiding_role_objects(self):
        """The props the hiding added. The annex adds none - it moves a case that is already in the
        scene - so this is empty; it exists so that every scoring loop can be written the same way in
        every task, and the test that no prop leaks into the score has something to ask.
        """
        return {name for name, entry in self.task_spec.get('objects', {}).items()
                if entry.get('role') in OB.HIDE_MODES + ('occluder', 'cover')}

    # ---- scene protocol -----------------------------------------------------
    def build_task(self):
        self.place_work_frame()
        island = bool(fixture_is_type(self.work_fixture, FixtureType.ISLAND))
        if island != bool(self.spec['work_fixture_is_island']):
            raise ValueError(f"{self.TASK_NAME}: instance expected "
                             f"island={self.spec['work_fixture_is_island']}, the work surface "
                             f"{self.work['fixture']!r} is island={island}")
        world, assets = self.model.worldbody, self.model.asset
        top_z = float(self.work['top_z'])
        frame_yaw = float(self.work['yaw'])
        flipped = self._robot_frame()
        regions, fixture_top, _ = self._surface_regions()
        floor = self.floor_map()
        keepouts = self._plan_keepouts(top_z)
        bounds = None
        if floor['bounds'] is not None:
            room = self._world_box_to_robot_polygon(dict(x0=floor['bounds'][0], x1=floor['bounds'][1],
                                                         y0=floor['bounds'][2], y1=floor['bounds'][3]))
            bounds = [float(room[:, 0].min()), float(room[:, 0].max()), float(room[:, 1].min()), float(room[:, 1].max())]

        # The fixed props at the front middle (robot frame), slid sideways only if decor stands there;
        # the pair moves PROP_FALLBACK to one side only when the middle admits no arrangement of the cases.
        under = [r for r in regions if r['rect'][0] <= 0. <= r['rect'][1]]           # the rectangle under the frame origin
        front = under[0] if under else min(regions, key=lambda r: abs(r['rect'][0] + r['rect'][1]) / 2)
        by_index = {box['index']: box for box in self.spec['boxes']}
        stance_free = self._stance_checker(floor)

        # Observability (spec 2.2, contract C6). The case the level hides is stood in its annex first,
        # before the others are solved onto the work surface: it is built there and its item is put
        # inside it there, so nothing has to be moved afterwards. When the annex is a far part of the
        # work surface itself, the case and the space in front of its knobs become keep-outs, so the
        # arrangement cannot crowd it and the props are settled clear of it.
        annex = None
        annex_park_blockers = []
        if H.hidden_box(self.spec) is not None:
            annex_box = by_index[H.box_index(H.hidden_box(self.spec))]
            annex = self._annex_case(annex_box, floor, stance_free)
            if annex['fixture'] == self.work['fixture']:
                faces = [face for face in H.panel_faces(annex_box) if face not in (None, 'top')]
                for k, polygon in enumerate(H.case_keepout(annex['xy_robot'], annex['yaw_robot'], faces)):
                    keepouts.append((f'annex_{k}', polygon))
                for face in faces:
                    keepouts.append((f'annex_hand_{face}',
                                     PL.knob_work_volume(annex['xy_robot'], annex['yaw_robot'], face)))
                # ... and its block keeps a parked plate away, as the oracle's park search sees it.
                annex_park_blockers.append(PL.case_park_block(annex['xy_robot'], annex['yaw_robot']))
                if annex.get('park') is not None:
                    # ... while its own plate's spot is kept clear of the cases solved next.
                    half = P.LID_HALF + PL.LID_PARK_CLEARANCE
                    keepouts.append(('annex_park', PL.rect(annex['park']['spot'], 0., half, half)))
        polygons = [polygon for _, polygon in keepouts]
        ordered = [by_index[i] for i in self.spec['placement_order']
                   if annex is None or i != annex['index']]
        solved, errors = None, []
        for prop_offset in (0., -PROP_FALLBACK, PROP_FALLBACK):
            try:
                bowl_xy, bowl_shift = PL.settle_prop(
                    bowl_polygon, [self.spec['bowl_xy'][0] + prop_offset, self.spec['bowl_xy'][1]], polygons, front['rect'])
                submit_xy, submit_shift = PL.settle_prop(
                    submit_polygon, [self.spec['submit_xy'][0] + prop_offset, self.spec['submit_xy'][1]],
                    polygons + [bowl_polygon(bowl_xy)], front['rect'])
                blocked = self._prop_stance_problems(bowl_xy, submit_xy, stance_free)
                if blocked:
                    raise PL.PlacementError('; '.join(blocked))
                prop_keepouts = [('bowl', PL.circle_polygon(bowl_xy, P.BOWL_OUTER_RADIUS + BOWL_CLEARANCE)),
                                 ('submit', PL.rect(submit_xy, 0., SUBMIT_HALF[0] + SUBMIT_CLEARANCE,
                                                    SUBMIT_HALF[1] + SUBMIT_CLEARANCE))]
                # The lid park rule (CONTAINERS-CLOSER): every lift/twist lid's plate needs a spot on the
                # surface clear of the cases, the bowl and Submit, with a free top stance within reach; the
                # search walks on past arrangements without one and the build fails when none has one.
                park_blockers = PL.prop_park_blockers(bowl_xy, submit_xy) + list(annex_park_blockers)
                solved = PL.place_boxes(ordered, regions, polygons, np.random.default_rng(int(self.spec['layout_seed'])),
                                        stance_free=stance_free, bounds=bounds,
                                        low_keepouts=[p for _, p in prop_keepouts], park_blockers=park_blockers)
            except PL.PlacementError as error:
                errors.append(f'props at {prop_offset:+.2f}: {error}')
                continue
            keepouts.extend(prop_keepouts)
            break
        if solved is None:
            raise ValueError(f"{self.TASK_NAME}: no arrangement on {self.work['fixture']!r} "
                             f"(regions {[r['rect'] for r in regions]}): " + '; '.join(errors))

        materials = AP.register(assets, self.spec['appearance'])
        boxes, placement, pick_shortfall, park_spots = {}, {}, {}, {}
        closers = set(int(i) for i in self.spec['closers'])
        for box in self.spec['boxes']:
            index = box['index']
            annexed = annex is not None and index == annex['index']
            if annexed:
                # The annexed case: its own surface and its own yaw - the quarter turn that made both of
                # its knobs reachable there - and the stances beside that surface, not the work one.
                xy_robot, yaw_robot, quarter = list(annex['xy_robot']), float(annex['yaw_robot']), annex['yaw_turn']
                reach, region_name = dict(annex['reach']), annex['region']
                sides = (PL.world_side(yaw_robot, box['face']), PL.world_side(yaw_robot, box['dummy_face']))
            else:
                xy_robot, yaw_robot, quarter = solved['positions'][index], float(box['yaw']), box['yaw_quarter']
                reach = {label: self._stance_world(solved['reach'][index][label]) for label in ('real', 'fake')}
                region_name, sides = regions[solved['reach'][index]['region']]['name'], (box['side'], box['dummy_side'])
            for label, stance in reach.items():
                if stance is None:
                    raise ValueError(f'{self.TASK_NAME}: box {index}: the {label} panel is unreachable')
            lid_yaw = lid_yaw_for(reach['real']['edge'], quarter) if box['mechanism'] in TOP_MECHANISMS else 0.
            xy_frame = self.robot_to_frame_xy(xy_robot)
            yaw_frame = float(yaw_robot) + self._flip_yaw
            base = ([annex['xy_world'][0], annex['xy_world'][1], annex['top_z']] if annexed
                    else self.frame_to_world([xy_frame[0], xy_frame[1], 0.]).tolist())
            info = P.build_box(world, index, box['mechanism'], base,
                               yaw=frame_yaw + yaw_frame, face=box['face'],
                               dummy_face=box['dummy_face'], direction=box['direction'],
                               hinge=box['hinge'], lid_yaw=float(lid_yaw))
            bodies = [world.find(f".//body[@name='{info['shell_body']}']")]
            if info['moving_body']:
                bodies.append(world.find(f".//body[@name='{info['moving_body']}']"))
            AP.apply_to_bodies(bodies, materials, self.spec['appearance']['knob_rgb'], _vec(WOOD), _vec(PANEL), _vec(DARK))
            info['frame_local'] = dict(xy=xy_frame, yaw=yaw_frame)
            info['robot_local'] = dict(xy=list(xy_robot), yaw=float(yaw_robot), yaw_quarter=int(quarter))
            info['side'], info['dummy_side'] = sides
            info['reach'] = reach
            info['region'] = region_name
            info['annexed'] = bool(annexed)
            info['closer'] = index in closers
            boxes[f'box{index}'] = info
            # How far short of the crept pick stance the base has to stop on this case, per panel: the
            # mint allows up to PL.DERIVED_STANCE_TOLERANCE of it (the oracle's drives absorb the rest)
            # and the certificate reads it here rather than re-deriving the free floor.
            pick_shortfall[f'box{index}'] = {label: round(float((stance or {}).get('pick_shortfall') or 0.), 4)
                                             for label, stance in reach.items()}
            placement[f'box{index}'] = dict(xy_robot=list(xy_robot), yaw_robot=float(yaw_robot), xy_frame=xy_frame,
                                            yaw_frame=yaw_frame, side=sides[0], dummy_side=sides[1],
                                            lid_yaw=float(lid_yaw), region=region_name, reach=reach,
                                            annexed=bool(annexed))
            if box['mechanism'] in P.FREE_LID_MECHANISMS:
                # Where its plate goes (the park rule): the solver's spot, or the annex's own.
                park = annex['park'] if annexed else solved['park'].get(index)
                if park is None:
                    raise ValueError(f'{self.TASK_NAME}: box {index} ({box["mechanism"]}): no park spot for its plate')
                park_spots[f'box{index}'] = dict(spot_robot=[round(float(v), 4) for v in park['spot']],
                                                 stance_robot=[round(float(v), 4) for v in park['stance_xy']],
                                                 heading_robot=round(float(park['heading']), 5),
                                                 where=str(park['where']))

        rng = np.random.default_rng(int(self.spec['placement_seed']))
        items = []
        for k, (name, holder_index) in enumerate(zip(self.spec['item_names'], self.spec['holders'])):
            holder = boxes[f'box{holder_index}']
            item_asset = P.ITEM_ASSETS[name]
            item_half = np.asarray(item_asset[1], float)
            item_pos, item_yaw = P.item_pose_in(holder, rng, half_z=float(item_half[2]))
            body_name = f'uc_item{k}'
            _, item_geoms = add_robocasa_object(world, assets, body_name,
                                                P.ROBOCASA_ASSETS / item_asset[0], item_pos,
                                                yaw=item_yaw, density='700', friction='1.0 0.01 0.02',
                                                condim='6')
            items.append(dict(index=k, name=name, body_name=body_name, joint_name=body_name + '_joint',
                              geom_names=item_geoms, holder=f'box{holder_index}',
                              source_position=list(item_pos), source_yaw=float(item_yaw), half=item_half.tolist(),
                              source_asset=str(P.ROBOCASA_ASSETS / item_asset[0])))
        bowl_frame = self.robot_to_frame_xy(bowl_xy)
        submit_frame = self.robot_to_frame_xy(submit_xy)
        bowl_pos = self.frame_to_world([bowl_frame[0], bowl_frame[1], P.BOWL_HALF_HEIGHT + .001])
        _, bowl_geoms = add_robocasa_object(world, assets, 'uc_bowl', P.BOWL_ASSET, bowl_pos.tolist(),
                                            free=False, density='800', scale=P.BOWL_SCALE)
        bowl = dict(body_name='uc_bowl', collision_geom_names=bowl_geoms,
                    center_xy=[float(bowl_pos[0]), float(bowl_pos[1])],
                    floor_z=float(top_z + P.BOWL_FLOOR_ABOVE_TOP), rim_z=float(top_z + P.BOWL_RIM_ABOVE_TOP),
                    inner_radius=P.BOWL_INNER_RADIUS, outer_radius=P.BOWL_OUTER_RADIUS, scale=P.BOWL_SCALE,
                    frame_local_xy=bowl_frame, robot_local_xy=list(bowl_xy),
                    source=f'RoboCasa objaverse bowl_1, fixed, meshes scaled {P.BOWL_SCALE:g}x')
        self.add_submit_button(submit_frame)
        self.task_spec['submit_button']['robot_local_xy'] = list(submit_xy)
        self._add_review_camera(regions, fixture_top, top_z)
        if annex is not None and not any(r['name'] == annex['region'] for r in regions):
            # The annex surface joins the placement regions, so that everything which asks a case which
            # rectangle it stands on - the lid park spot, the oracle's route - gets an answer.
            regions.append(dict(name=annex['region'], rect=[round(float(v), 4) for v in annex['rect_robot']],
                                edges=tuple(annex['edges_robot']), kind='annex', fixture=annex['fixture'],
                                top_z=float(annex['top_z'])))
        self.task_spec['observability'] = self._observability_record(annex, boxes)

        self.geometry = dict(boxes=boxes, bowl=bowl, items=items, annex=annex,
                             holders_private=[f"box{h}" for h in self.spec['holders']], top_z=top_z,
                             robot_frame=dict(flipped=flipped, yaw_world=frame_yaw + self._flip_yaw,
                                              origin_world=list(self.work['center_world'])),
                             floor=floor, closers=sorted(f'box{i}' for i in closers))
        self.task_spec.update(
            contract_version=CONTRACT_VERSION, instruction=self.PUBLIC_GOAL, boxes=boxes, bowl=bowl,
            items=items, holders_private=self.geometry['holders_private'], closers=self.geometry['closers'],
            appearance=dict(self.spec['appearance'], materials=materials),
            box_order=sorted(boxes), work_fixture_is_island=island,
            observability_level=self.spec.get('observability', 'visible'),
            hiding_role_ids=sorted(self.hiding_role_objects()),
            frame_origin_world=self.work['center_world'], frame_yaw=frame_yaw,
            robot_frame=self.geometry['robot_frame'], floor=floor,
            placement=dict(regions=regions, fixture_top=fixture_top, island=island,
                           keepouts=[name for name, _ in keepouts], spacing_requested=float(PL.SPACING_LADDER[0]),
                           spacing_m=float(solved['spacing']), ladder_step=int(solved['ladder_step']),
                           search_nodes=int(solved['nodes']), valid_cells=dict(solved['valid_cells']),
                           order=list(self.spec['placement_order']), prop_offset_m=float(prop_offset),
                           bowl_shift_m=bowl_shift, submit_shift_m=submit_shift, placement_errors=errors,
                           room_bounds_robot=bounds, boxes=placement,
                           pick_shortfall_private=pick_shortfall,
                           pick_shortfall_max_m=round(max([v for panels in pick_shortfall.values()
                                                           for v in panels.values()] or [0.]), 4),
                           # The lid park rule's answer per lift/twist case, and how many cells or
                           # arrangements the search walked past for want of one (CONTAINERS-CLOSER).
                           park_spots_private=park_spots, park_rejections=int(solved.get('park_rejections', 0)),
                           # The window each closer leaves the hand, filled in once the model is compiled
                           # (setup_task_references measures it on the built box model).
                           closer_window_ticks={}),
            construction='Identical cases with overlay panels and round knobs on two faces; one panel '
                         'moves on a slide, hinge or free joint and the other is screwed down. Hidden '
                         'guides, a turning tab, a bayonet and two touch latches shape what each panel '
                         'can do; some panels shut again by themselves. Declared rule: a touch latch '
                         f'releases when an inward press loads it beyond {P.LATCH_RELEASE_N:.0f} N.',
            scoring='First physical Submit freezes the score: every named item inside the bowl, resting on '
                    'it or on another item in it, released and settled; no submission fails.')
        self.task_spec['latch_equalities'] = add_latch_constraints(self.model.root, dict(boxes=boxes))
        self.asset_evidence.update(
            boxes='Purpose-made rigid cases, panels, knobs and hidden mechanisms in the task frame; RoboCasa wood textures',
            items='RoboCasa objaverse objects, original meshes/textures/collisions: '
                  + ', '.join(f"{i['name']} ({i['source_asset']})" for i in items),
            bowl=f'RoboCasa objaverse bowl_1, fixed, scaled {P.BOWL_SCALE:g}x ({P.BOWL_ASSET})',
            room='RoboCasa kitchen, layout/style from the instance; render recipe unchanged')

    REVIEW_CAMERA = 'uc_review'

    def _add_review_camera(self, regions, fixture_top, top_z):
        """A fixed top-down camera over the surface (previews and gates only; not a policy camera)."""
        xs = [v for r in regions for v in r['rect'][:2]]
        ys = [v for r in regions for v in r['rect'][2:]]
        centre = self.robot_to_frame_xy([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2])
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.)
        height = .35 * span + 1.6
        pos = self.frame_to_world([centre[0], centre[1], height])
        ET.SubElement(self.model.worldbody, 'camera', name=self.REVIEW_CAMERA, pos=_vec(pos),
                      quat=_vec(yaw_quat_wxyz(float(self.work['yaw']))), fovy='75')
        self.task_spec['review_camera'] = dict(name=self.REVIEW_CAMERA, height_above_top_m=float(height))

    # ---- references ---------------------------------------------------------
    def setup_task_references(self):
        m = self.sim.model
        self.obj_body_id = dict(getattr(self, 'obj_body_id', {}))
        self._item_bid, self._item_geoms, self._geom_item = [], [], {}
        for item in self.geometry['items']:
            bid = m.body_name2id(item['body_name'])
            self.obj_body_id[item['body_name']] = bid
            self._item_bid.append(bid)
            geoms = {m.geom_name2id(g) for g in item['geom_names']}
            self._item_geoms.append(geoms)
            for g in geoms:
                self._geom_item[g] = item['body_name']
        self.obj_body_id['uc_item'] = self._item_bid[0]        # the first item, for tools that expect one
        self._bowl_geoms = {m.geom_name2id(g) for g in self.geometry['bowl']['collision_geom_names']}
        self._box_qadr, self._latch_qadr, self._latches, self._lid_body = {}, {}, {}, {}
        for name, box in self.geometry['boxes'].items():
            mechanism = box['mechanism']
            if mechanism == SEALED:
                continue
            if mechanism in P.FREE_LID_MECHANISMS:
                self._lid_body[name] = m.body_name2id(box['moving_body'])
            else:
                self._box_qadr[name] = m.get_joint_qpos_addr(box['joint_name'])
            if box.get('latch_joint'):
                self._latch_qadr[name] = m.get_joint_qpos_addr(box['latch_joint'])
            if box.get('latch_equality'):
                self._latches[name] = mujoco.mj_name2id(m._model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                                        box['latch_equality'])
        self.latch_released = {name: False for name in self._latches}
        # Passive closers on the compiled model (nothing in the MJCF), then their parameters in the spec, and
        # the window each leaves the hand -- measured on the built box model (springs.closer_windows: the
        # same panels, the same tuning, the kitchen's tick), which fails the build when a window is under
        # springs.CLOSER_WINDOW_MIN_TICKS or off the table the mint drew from.
        self._closers, windows = {}, {}
        for name in self.geometry['closers']:
            box = self.geometry['boxes'][name]
            self._closers[name] = SP.apply_box_closer(self.sim, box)
            windows[name] = SP.closer_window_ticks(box['mechanism'])
            if windows[name] < SP.CLOSER_WINDOW_MIN_TICKS:
                raise ValueError(f"{self.TASK_NAME}: {name} ({box['mechanism']}): its closer leaves the hand "
                                 f"{windows[name]} ticks after the release, under {SP.CLOSER_WINDOW_MIN_TICKS}")
        self.task_spec['closer_parameters'] = dict(self._closers)
        self.task_spec['placement']['closer_window_ticks'] = windows
        self._opened_once = {name: False for name in self.geometry['boxes']}

    def _reset_internal(self):
        super()._reset_internal()
        for name, eq in getattr(self, '_latches', {}).items():
            self.sim.data.eq_active[eq] = 1
            self.latch_released[name] = False
        self._opened_once = {name: False for name in self.geometry['boxes']}

    def _check_latches(self):
        """The one declared rule: a touch latch lets go when the press it
        resists passes LATCH_RELEASE_N beyond its spring preload."""
        d = self.sim.data
        for name, eq in self._latches.items():
            rule = self.geometry['boxes'][name]['latch_rule']
            if d.eq_active[eq] and latch_released(self.sim.model._model, d._data, eq, rule):
                d.eq_active[eq] = 0
                self.latch_released[name] = True

    def _track_openings(self):
        """Remember every box that has stood open at some tick (the collateral check needs it: a closer
        that shuts a never-opened box is not a consequence of anything)."""
        for name, state in self.box_states().items():
            if state.get('open'):
                self._opened_once[name] = True

    def _post_action(self, action):
        self._check_latches()
        self._track_openings()
        return super()._post_action(action)

    # ---- state --------------------------------------------------------------
    def item_state(self, index=0):
        m, d = self.sim.model, self.sim.data
        item = self.geometry['items'][index]
        bid, geoms = self._item_bid[index], self._item_geoms[index]
        supported, robot = set(), []
        for ci, contact in enumerate(d.contact[:d.ncon]):
            a, b = int(contact.geom1), int(contact.geom2)
            if a not in geoms and b not in geoms:
                continue
            other = b if a in geoms else a
            other_name = m.geom_id2name(other) or ''
            if other_name.startswith(('robot', 'gripper', 'mobilebase')):
                robot.append(other_name)
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(m._model, d._data, ci, force)
            normal = np.asarray(contact.frame).reshape(3, 3)[0] * (1 if b in geoms else -1)
            if force[0] > 1e-5 and normal[2] > .5:
                if other in self._bowl_geoms:
                    supported.add('bowl')
                elif other in self._geom_item:
                    supported.add(self._geom_item[other])
                else:
                    supported.add(other_name)
        return dict(name=item['name'], body=item['body_name'],
                    position=np.asarray(d.body_xpos[bid]).tolist(),
                    quaternion_xyzw=Rotation.from_matrix(
                        np.asarray(d.body_xmat[bid]).reshape(3, 3)).as_quat().tolist(),
                    linear_speed=float(np.linalg.norm(np.asarray(d.get_body_xvelp(item['body_name'])))),
                    angular_speed=float(np.linalg.norm(np.asarray(d.get_body_xvelr(item['body_name'])))),
                    supported_by=sorted(supported), robot_contact=bool(robot),
                    grasped=bool(self._check_grasp(self.robots[0].gripper['right'],
                                                   [m.geom_id2name(g) for g in geoms])))

    def item_states(self):
        return [self.item_state(i) for i in range(len(self.geometry['items']))]

    def item_out_of_box(self, index, position=None):
        """The item's centre lies outside its own case's interior box (in the case's face frame, with a
        margin): lifted out, carried away, or riding a pulled-out tray in the open."""
        item = self.geometry['items'][index]
        box = self.geometry['boxes'][item['holder']]
        interior = box['interior']
        p = np.asarray(self.sim.data.body_xpos[self._item_bid[index]] if position is None else position, float)
        local = yaw_matrix(-float(box['frame_yaw'])) @ (p - np.asarray(interior['center'], float))
        return bool(np.any(np.abs(local) > np.asarray(interior['half'], float) + OUT_OF_BOX_MARGIN))

    def box_states(self):
        """Private diagnostic: which cases stand open right now."""
        d = self.sim.data
        states = {}
        holders = self.geometry['holders_private']
        for name, box in self.geometry['boxes'].items():
            if box['mechanism'] == SEALED:
                states[name] = P.box_openness(box)
            elif box['mechanism'] in P.FREE_LID_MECHANISMS:
                states[name] = P.box_openness(box, lid_center=d.body_xpos[self._lid_body[name]])
            else:
                latch = float(d.qpos[self._latch_qadr[name]]) if name in self._latch_qadr else None
                states[name] = P.box_openness(box, joint_value=float(d.qpos[self._box_qadr[name]]),
                                              latch_turn=latch)
            if name in self._latches:
                states[name]['latch_released'] = bool(self.latch_released[name])
            if name in self._closers:
                states[name]['closer'] = True
                states[name]['shut'] = bool(states[name]['joint'] <= self._closers[name]['closed_when'])
            states[name].update(mechanism=box['mechanism'], face=box['face'], side=box['side'],
                                dummy_face=box['dummy_face'], dummy_side=box['dummy_side'],
                                holds_item=name in holders, opened_once=bool(self._opened_once.get(name, False)))
        return states

    def _item_facts(self, states):
        """Per item: the bowl geometry, support, release and rest; then ``in_bowl`` with support by the
        bowl or by another item that is itself in the bowl."""
        bowl = self.geometry['bowl']
        facts = []
        for item, state in zip(self.geometry['items'], states):
            containment = item_in_bowl(state, bowl, float(item['half'][2]))
            facts.append(dict(containment, name=item['name'], body=item['body_name'],
                              supported_by=list(state['supported_by']),
                              supported_by_bowl='bowl' in state['supported_by'],
                              released=bool(not state['robot_contact'] and not state['grasped']),
                              settled=bool(state['linear_speed'] < .03 and state['angular_speed'] < .5),
                              position=state['position']))
        in_bowl = [f['contained'] and f['released'] and f['settled'] and f['supported_by_bowl'] for f in facts]
        changed = True
        while changed:
            changed = False
            for i, f in enumerate(facts):
                if in_bowl[i] or not (f['contained'] and f['released'] and f['settled']):
                    continue
                if any(in_bowl[j] for j, other in enumerate(self.geometry['items'])
                       if j != i and other['body_name'] in f['supported_by']):
                    in_bowl[i] = True
                    changed = True
        for f, flag in zip(facts, in_bowl):
            f['in_bowl'] = bool(flag)
        return facts

    def _current_score(self, states=None):
        facts = self._item_facts(self.item_states())
        success = bool(facts) and all(f['in_bowl'] for f in facts)
        return dict(success=success, items=facts, items_in_bowl=int(sum(f['in_bowl'] for f in facts)),
                    item=facts[0], boxes_private=self.box_states())

    # ---- protocol hooks (contract C1 / C3) --------------------------------------------------
    def task_progress(self):
        """Per item: 0.5 once out of its box, 1 in the bowl; mean over items (spec 1.4)."""
        facts = self._item_facts(self.item_states())
        values = []
        for i, f in enumerate(facts):
            if f['in_bowl']:
                values.append(1.)
            elif self.item_out_of_box(i, f['position']):
                values.append(.5)
            else:
                values.append(0.)
        return float(np.mean(values)) if values else 0.

    def collateral_checks(self):
        """``panel_closed``: a closer shut a panel again with its item still inside (spec 1.5); cost one per
        box. Only boxes that have stood open count, so the reset state is never a consequence."""
        shut = []
        states = self.box_states()
        for name in self.geometry['closers']:
            if not self._opened_once.get(name) or not states[name].get('shut'):
                continue
            inside = [i for i, item in enumerate(self.geometry['items'])
                      if item['holder'] == name and not self.item_out_of_box(i)]
            if inside:
                shut.append(name)
        return {'panel_closed': dict(bad=bool(shut), object=','.join(shut), cost=float(len(shut)))}

    def lid_park_spot(self, name):
        """Where a removed top plate is set down (robot frame): the spot the mint's park rule recorded for
        the case (``placement.park_spots_private``, on its own surface clear of everything, with a free
        stance within reach -- the oracle's spot too); for an instance minted before that rule, beside
        its case along the edge it is worked from on free surface, else in front of the case toward the
        base, where the plate may hang over the edge."""
        placement = self.task_spec['placement']
        recorded = (placement.get('park_spots_private') or {}).get(name)
        if recorded is not None:
            return np.asarray(recorded['spot_robot'], float), str(recorded['where'])
        rows = placement['boxes'][name]
        reach = rows['reach']['real']
        region = next(r for r in placement['regions'] if r['name'] == rows['region'])['rect']
        centre = np.asarray(rows['xy_robot'], float)
        n_edge = PL.normal_of(reach['edge'])
        tangent = np.array([-n_edge[1], n_edge[0]])
        others = [PL.rect(b['xy_robot'], b['yaw_robot'], PL.CASE_HALF + PL.KNOB_OUT, PL.CASE_HALF + PL.KNOB_OUT)
                  for other, b in placement['boxes'].items() if other != name]
        bowl = self.geometry['bowl']['robot_local_xy']
        submit = self.task_spec['submit_button']['robot_local_xy']
        others += [PL.circle_polygon(bowl, P.BOWL_OUTER_RADIUS + .05), PL.rect(submit, 0., .10, .10)]
        for spot in (centre + tangent * LID_PARK_DISTANCE, centre - tangent * LID_PARK_DISTANCE):
            plate = PL.rect(spot, 0., P.LID_HALF + .02, P.LID_HALF + .02)
            if not PL.polygon_inside(plate, region, 0.):
                continue
            if any(PL.polygons_overlap(plate, o, .01) for o in others):
                continue
            return spot, 'beside'
        return centre + n_edge * .42, 'over_the_edge'

    def _teleport_open(self, name):
        """Open one box by state edits: release its latch, turn its tab, move its joint or park its lid."""
        box = self.geometry['boxes'][name]
        mechanism = box['mechanism']
        sim = self.sim
        if mechanism == SEALED:
            raise ValueError(f'{name} is sealed')
        if name in self._latches:
            sim.data.eq_active[self._latches[name]] = 0
            self.latch_released[name] = True
        if mechanism in P.FREE_LID_MECHANISMS:
            spot, where = self.lid_park_spot(name)
            world = self.robot_to_world_xy(spot)
            # The plate spans LID_Z in its body frame; the twist lid's bayonet tabs hang under it.
            drop = P.SLOT_Z[0] + .002 if mechanism == 'twist_lid' else P.LID_Z[0]
            z = float(self.geometry['top_z'] - drop + .002)
            quat = yaw_quat_wxyz(float(box['frame_yaw']))
            sim.data.set_joint_qpos(box['joint_name'], np.array([world[0], world[1], z] + quat))
            sim.forward()
            return dict(parked=where, spot_robot=[float(spot[0]), float(spot[1])])
        if mechanism == 'turn_knob':
            sim.data.qpos[self._latch_qadr[name]] = .8
            sim.forward()
        target = {'drawer': P.DRAWER_TRAVEL, 'press_drawer': P.DRAWER_TRAVEL, 'turn_knob': P.DRAWER_TRAVEL,
                  'door': 1.55, 'press_door': 1.55, 'slide_side': P.SIDE_TRAVEL, 'slide_up': P.UP_TRAVEL,
                  'slide_lid': P.SLIDE_TRAVEL, 'flip_lid': SP.FLIP_OPEN_RAD}[mechanism]
        sim.data.qpos[self._box_qadr[name]] = float(target)
        sim.data.qvel[int(sim.model.jnt_dofadr[sim.model.joint_name2id(box['joint_name'])])] = 0.
        sim.forward()
        return dict(joint=float(target))

    def teleport_solution(self):
        """Bring the scene to a solved state without the robot (contract C3).

        Opens every holder's mechanism by state edits (latches released, tabs turned, joints moved,
        free lids parked beside their case), moves the items into the bowl, settles with zero-action
        steps and evaluates the predicate. Never raises; returns a list of {'step', 'ok', 'detail'}.
        """
        steps = []
        bowl = self.geometry['bowl']
        n = len(self.geometry['items'])
        for item in self.geometry['items']:
            name = item['holder']
            try:
                detail = self._teleport_open(name)
                steps.append(dict(step=f"open {name} ({self.geometry['boxes'][name]['mechanism']})", ok=True, detail=detail))
            except Exception as error:  # noqa: BLE001 - reported, never raised
                steps.append(dict(step=f'open {name}', ok=False, detail=repr(error)))
        zero = np.zeros(self.action_dim)
        try:
            # One item at a time, each on its own spot on a ring inside the bowl and released from just
            # above whatever is already there. Dropping every item on the bowl's axis from above the rim
            # bounced one back out about a fifth of the time at three and four items (v1 registry).
            for k, item in enumerate(self.geometry['items']):
                spot, z = self._bowl_spot(k, n, float(item['half'][2]))
                steps.append(dict(step=f"place {item['name']} in the bowl", ok=True,
                                  detail=self._drop_into_bowl(item, spot, z, zero)))
        except Exception as error:  # noqa: BLE001
            steps.append(dict(step='place items', ok=False, detail=repr(error)))
        try:
            for _ in range(TELEPORT_SETTLE_TICKS):
                self.step(zero)
            steps.append(dict(step='settle', ok=True, detail=dict(ticks=TELEPORT_SETTLE_TICKS)))
        except Exception as error:  # noqa: BLE001
            steps.append(dict(step='settle', ok=False, detail=repr(error)))
        # Re-seat whatever bounced out, as the parcel certificate does: the same spot, the lowest release
        # the pile allows, then settle again. Three tries, and each one is reported.
        for attempt in range(TELEPORT_RESEATS):
            try:
                facts = self._item_facts(self.item_states())
            except Exception as error:  # noqa: BLE001
                steps.append(dict(step='re-seat', ok=False, detail=repr(error)))
                break
            loose = [k for k, fact in enumerate(facts) if not fact['in_bowl']]
            if not loose:
                break
            try:
                for k in loose:
                    item = self.geometry['items'][k]
                    spot, z = self._bowl_spot(k, n, float(item['half'][2]))
                    self._drop_into_bowl(item, spot, z, zero)
                for _ in range(TELEPORT_SETTLE_TICKS):
                    self.step(zero)
                after = self._item_facts(self.item_states())
                steps.append(dict(step=f're-seat {len(loose)} item(s), try {attempt + 1}',
                                  ok=all(after[k]['in_bowl'] for k in loose),
                                  detail=dict(items=[self.geometry['items'][k]['name'] for k in loose],
                                              in_bowl=[bool(after[k]['in_bowl']) for k in loose])))
            except Exception as error:  # noqa: BLE001
                steps.append(dict(step=f're-seat, try {attempt + 1}', ok=False, detail=repr(error)))
                break
        try:
            score = self._current_score()
            steps.append(dict(step='predicate', ok=bool(score['success']),
                              detail=dict(items_in_bowl=score['items_in_bowl'],
                                          items=[{k: v for k, v in f.items() if k != 'position'} for f in score['items']],
                                          progress=self.task_progress())))
        except Exception as error:  # noqa: BLE001
            steps.append(dict(step='predicate', ok=False, detail=repr(error)))
        return steps

    def _bowl_spot(self, index, count, half_z):
        """Where item ``index`` of ``count`` is released (world): its own spot on a ring a third of the
        inner radius out, angles spread evenly, at most ``TELEPORT_DROP_ABOVE`` over the bowl floor or
        over whatever already lies at that height."""
        bowl = self.geometry['bowl']
        radius = 0. if count == 1 else bowl['inner_radius'] * TELEPORT_RING_SHARE
        angle = 2 * math.pi * index / max(count, 1)
        xy = np.asarray(bowl['center_xy'], float) + radius * np.array([math.cos(angle), math.sin(angle)])
        pile = bowl['floor_z']
        for k, other in enumerate(self.geometry['items']):
            position = np.asarray(self.sim.data.body_xpos[self._item_bid[k]], float)
            if k == index or np.linalg.norm(position[:2] - xy) > bowl['inner_radius']:
                continue
            top = float(position[2]) + float(other['half'][2])
            if top > pile and float(position[2]) < bowl['rim_z']:
                pile = top
        return xy, float(pile + half_z + TELEPORT_DROP_ABOVE)

    def _drop_into_bowl(self, item, xy, z, zero):
        """Set one item upright at ``xy``/``z``, let it come to rest, and report where it went."""
        self.sim.data.set_joint_qpos(item['joint_name'], np.array([xy[0], xy[1], z, 1., 0., 0., 0.]))
        self.sim.data.qvel[:] = 0.
        self.sim.forward()
        for _ in range(TELEPORT_DROP_TICKS):
            self.step(zero)
        centre = np.asarray(self.geometry['bowl']['center_xy'], float)
        rest = np.asarray(self.sim.data.body_xpos[self._item_bid[item['index']]], float)
        return dict(xy=[float(xy[0]), float(xy[1])], z=float(z), ticks=TELEPORT_DROP_TICKS,
                    radial_m=round(float(np.linalg.norm(rest[:2] - centre)), 4))

    def task_snapshot(self):
        return dict(items=self.item_states(), boxes=self.box_states())

    # ---- G3 ----------------------------------------------------------------
    def hidden_state_check(self, control='uc_bowl'):
        """The items must change no pixel of the three policy cameras, and the
        bowl must be visible in a fixed camera.

        Hiding every item's visual geoms (alpha 0, no material) must leave the images unchanged; hiding
        ``control`` instead must change them, which proves the toggle itself works (fail-closed, as the
        cube task does).

        Both toggles are rendered **twice** and a pixel counts only when it changed in both passes
        (:func:`hiding.flicker_tolerant_diff`), the rule :func:`observe.visible_in_cameras` and the cube
        task already use: EGL flickers by a handful of pixels on some kitchens, and a single-pass exact
        diff read that flicker as an item leak (layout 5 style 6: 20 changed pixels against 11 that moved
        with nothing toggled). The flicker is reported, never fatal; the single-pass count is kept beside
        it as evidence. The control is held to the same rule, so flicker cannot fake its pass either.
        """
        m = self.sim.model

        def frames():
            return {camera: self.sim.render(width=self.camera_widths[index],
                                            height=self.camera_heights[index],
                                            camera_name=camera).copy()
                    for index, camera in enumerate(self.camera_names)}

        def visual_ids(bodies):
            bids = {m.body_name2id(body) for body in bodies}
            return [g for g in range(m.ngeom)
                    if m.geom_bodyid[g] in bids and not (m.geom_contype[g] or m.geom_conaffinity[g])]

        def hidden(ids):
            rgba = m.geom_rgba[ids].copy()
            matid = m.geom_matid[ids].copy()
            try:
                m.geom_rgba[ids, 3] = 0.
                m.geom_matid[ids] = -1
                return frames()
            finally:
                m.geom_rgba[ids] = rgba
                m.geom_matid[ids] = matid

        frames()   # warm-up: the first frame of a fresh offscreen context is not a valid reference
        base, base_repeat = frames(), frames()
        item_ids = visual_ids([item['body_name'] for item in self.geometry['items']])
        items, items_repeat = hidden(item_ids), hidden(item_ids)
        changed = H.flicker_tolerant_diff({c: (base[c], items[c]) for c in base},
                                          {c: (base_repeat[c], items_repeat[c]) for c in base})
        single = {camera: int(np.any(base[camera] != items[camera], axis=2).sum()) for camera in base}
        unstable = H.unstable_pixels(base, base_repeat)
        control_ids = visual_ids([control])
        control_off, control_off_repeat = hidden(control_ids), hidden(control_ids)
        control_changed = H.flicker_tolerant_diff({c: (base[c], control_off[c]) for c in base},
                                                  {c: (base_repeat[c], control_off_repeat[c]) for c in base})
        hidden_all = not any(changed.values())
        control_visible = any(control_changed.values())
        # ``passed`` is what scripts/roboquest_gate_batch.py reads (without it the gate defaulted to True).
        report = dict(item_changed_pixels=changed, item_changed_pixels_single_pass=single,
                      renderer_unstable_pixels=unstable, control=control,
                      control_changed_pixels=control_changed,
                      hidden=hidden_all, control_visible=control_visible)
        report.update(self.observability_check())
        report['passed'] = bool(hidden_all and control_visible and report['observability_passed'])
        self.task_spec['hidden_state_check'] = report
        return report

    def observability_check(self):
        """The observability half of G3 (spec 2.2, contract C6): the annexed case is invisible in every
        policy camera at reset, the bowl is visible, the case stands where a viewpoint can be driven to,
        and both of its knobs - not its centre - are still within reach from its annex stance.

        The other cases are reported, never gated. On a wide island some of them already fall outside
        every policy camera at reset at the ``visible`` level, so demanding that they all show would fail
        instances the axis never touched.
        """
        record = self.task_spec.get('observability') or {}
        level = self.spec.get('observability', 'visible')
        annex = (record.get('annex') or [None])[0]
        if annex is None:
            problems = list(record.get('problems') or [])
            return dict(observability=level, observability_problems=problems,
                        observability_passed=not problems, annexed=None, hidden_bodies=[],
                        cases_in_no_camera=[], look_reach=None, knob_reach={})
        info = self.geometry['boxes'][annex['object']]
        bowl = self.geometry['bowl']['body_name']
        bodies = [name for name in (info['shell_body'], info['moving_body']) if name]
        others = [box['shell_body'] for name, box in sorted(self.geometry['boxes'].items())
                  if name != annex['object']]
        check = OB.hidden_check(self, bodies, visible=[bowl] + others)
        problems = [f'{name} is visible in a policy camera' for name, ok in check['hidden_ok'].items() if not ok]
        if not check['visible_ok'].get(bowl, True):
            problems.append('the bowl is in no policy camera')
        if not all(check['deterministic'].values()):
            problems.append('the renderer is not deterministic')
        look = OB.look_reachable(self, info['shell_body'], at_build=False,
                                 radius=H.CASE_RADIUS, height=H.CASE_HEIGHT)
        if not look['passed']:
            problems.append(f"{annex['object']}: no reachable viewpoint ({'; '.join(look['reasons'][:2])})")
        knobs = {}
        for label, knob in sorted((annex.get('knobs') or {}).items()):
            stance = (annex.get('reach') or {}).get(label)
            if stance is None:
                problems.append(f'{label} panel: the annex recorded no stance')
                continue
            gate = OB.reach_gate(self, knob['point'], floor=self.geometry['floor'], at_build=False,
                                 base_pose=(np.asarray(stance['xy_world'], float), stance['heading_world']))
            knobs[label] = dict(passed=bool(gate['passed']), point=list(knob['point']),
                                ahead_m=gate['ahead_m'], lateral_m=gate['lateral_m'], slide_m=gate['slide_m'],
                                reasons=gate['reasons'][:2])
            if not gate['passed']:
                problems.append(f"{annex['object']}: the {label} knob is out of reach from its annex "
                                f"stance ({'; '.join(gate['reasons'][:2])})")
        return dict(observability=level, observability_problems=problems,
                    observability_passed=bool(not problems and check['passed'] is not None
                                              and all(check['hidden_ok'].values())),
                    annexed=annex['object'], hidden_bodies=bodies,
                    cases_in_no_camera=sorted(name for name, ok in check['visible_ok'].items()
                                              if not ok and name != bowl),
                    look_reach=dict(passed=bool(look['passed']), surface=look['surface'],
                                    stance=look['stance'], reasons=look['reasons'][:2]),
                    knob_reach=knobs, hidden_pixels=check['pixels'], hidden_flicker=check['flicker'])
