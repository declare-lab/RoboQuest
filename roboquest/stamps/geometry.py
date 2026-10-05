"""MJCF for the stamp-composition props: round self-inking stamps and three paper boards.

Everything is expressed in the task frame of ``RoboQuestKitchen`` (x along the
front of the work surface, y away from the robot, z up from the top), so the
same instance builds on any RoboCasa layout and style.

Two shapes carry the task:

* a **stamp** — a plain round housing (sole disc plus handle) whose die sits
  *inside* the sole. The die is a set of visual-only dot geoms fully enclosed by
  the opaque sole cylinder, so it is the hidden state made physical: nothing
  about the pattern or the stamp's yaw reaches a camera (checked by G3), and the
  pattern is discovered by pressing, exactly as with a real self-inking stamp.
* a **board** — a thin bordered paper whose printed face is a textured plane.
  Its centre and yaw come from the instance (a small jitter about the nominal
  place), so the boards are never exactly aligned with each other.
  Its collider reaches down into the counter top instead of stopping 4 mm above
  it: a free-standing 4 mm slab exposes an underside that traps a pressed stamp
  between the board and the counter (the bug fixed in ``stamps_v2``).

Adapted from ``active_bench/stamps_v2_scene.py``; that module is left untouched.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from roboquest import scatter as S
from roboquest.stamps import rules

# ---- stamp ------------------------------------------------------------------
SOLE_RADIUS = .060            # covers the corner dots (sqrt(2)*CELL_SPACING + DOT_RADIUS = .056)
SOLE_HALF_HEIGHT = .005
HANDLE_RADIUS = .018
HANDLE_HALF_HEIGHT = .037
HANDLE_Z = .047
DIE_Z = .0012                 # the die sits inside the sole disc, never exposed
DIE_HALF_HEIGHT = .0006
SOLE_MASS = .10
HANDLE_MASS = .065
GRASP_OFFSET = (0., 0., .059)
STAMP_HEIGHT = HANDLE_Z + HANDLE_HALF_HEIGHT
SOLE_RGBA = (.24, .27, .30, 1.)
# Housing shells (spec 4.9): one opaque plastic/metal colour per instance, shared by
# every stamp so the housing still says nothing about a particular die. Kept dark
# enough to hide the enclosed die and clear of the functional colours (blue final
# border, green reference border, tan scratch border, red Submit).
SOLE_PALETTE = ((.24, .27, .30, 1.), (.31, .30, .28, 1.), (.20, .24, .27, 1.),
                (.28, .26, .31, 1.), (.26, .30, .28, 1.), (.33, .32, .34, 1.))
# Handle colours are a nuisance draw, assigned independently of the dies, so a
# colour identifies a stamp across a rollout without telling anything about it.
HANDLE_PALETTE = ((.72, .55, .32, 1.), (.30, .45, .68, 1.), (.62, .34, .40, 1.),
                  (.40, .58, .42, 1.), (.66, .62, .30, 1.), (.46, .40, .60, 1.))

# ---- boards -----------------------------------------------------------------
PAPER_THICKNESS = .004
BOARD_SUPPORT_DEPTH = .045    # collider reaches this far below the work surface, into solid counter
BOARD_BORDER = .004
BORDER_RGBA = dict(final=(.10, .20, .60, 1.), reference=(.08, .42, .18, 1.), scratch=(.48, .43, .34, 1.))

# ---- task-frame placement (metres) ------------------------------------------
# Footprint 0.92 x 0.58: x in [-.46, .46], y in [-.29, .29]. A 1.20 m frame (the
# puzzle box's) fits only four of the eight layouts in the registry's default
# pool; this one fits all eight, measured by building each.
#
# The three boards and Submit keep their functional places - the scratch board in
# the middle-left, the reference card above the final board above Submit on the
# right - but each board is drawn from a +/-2 cm box about its nominal centre with
# a +/-5 deg yaw, and the stamps are *scattered* over the free front and side area
# instead of standing in a row. Placement is drawn by
# :mod:`roboquest.scatter`, which re-draws until nothing overlaps, so
# no stamp ever starts on a board (which would print) or on Submit.
BOARD_NOMINAL_XY = dict(scratch=(-.210, .040), final=(.345, -.045), reference=(.345, .153))
BOARD_ORDER = ('scratch', 'final', 'reference')
BOARD_JITTER_XY = .02
BOARD_JITTER_YAW = math.radians(5.)
BOARD_GAP = .006               # clear space between two boards and between a board and Submit
SCRATCH_XY = BOARD_NOMINAL_XY['scratch']
FINAL_XY = BOARD_NOMINAL_XY['final']
REFERENCE_XY = BOARD_NOMINAL_XY['reference']
SUBMIT_XY = (.345, -.198)
SUBMIT_HALF_XY = (.048, .043)
FOOTPRINT = (.92, .58)
# Stamps: anywhere in the front 0.61 m of the frame that is clear of the boards, of
# Submit and of each other. The sole is a disc, so a stamp's footprint is yaw free.
STAMP_REGION_BACK_Y = .03
STAMP_GAP = .015               # clear space between two stamps
STAMP_KEEPOUT_GAP = .02        # clear space from a board or the Submit button


def vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


def frame_rect(footprint=FOOTPRINT):
    """The whole task footprint as a scatter rectangle."""
    return S.rect((0., 0.), (footprint[0] / 2, footprint[1] / 2))


def board_outer_half(name, scratch_cells):
    """Half size of a board including its border: what must not overlap anything."""
    half = board_plan(scratch_cells)[name]['half']
    return (half[0] + BOARD_BORDER, half[1] + BOARD_BORDER)


def board_slots(scratch_cells):
    """Per-board jitter boxes about the nominal centres, in ``BOARD_ORDER``."""
    return [S.rect(BOARD_NOMINAL_XY[name], (BOARD_JITTER_XY, BOARD_JITTER_XY)) for name in BOARD_ORDER]


def submit_shape(xy=SUBMIT_XY):
    return S.rect(xy, SUBMIT_HALF_XY)


def board_shapes(spec):
    """The three boards as oriented boxes in the task frame, in ``BOARD_ORDER``."""
    cells = int(spec['scratch_cells'])
    yaws = spec.get('board_yaw') or {}
    return [S.shape_at(board_outer_half(name, cells), spec['placement'][name], float(yaws.get(name, 0.)))
            for name in BOARD_ORDER]


def stamp_region(footprint=FOOTPRINT):
    """Where the stamps are scattered: the front of the frame, across its whole width."""
    half = (footprint[0] / 2, footprint[1] / 2)
    return S.Rect(-half[0], half[0], -half[1], STAMP_REGION_BACK_Y)


def board_plan(scratch_cells):
    """Task-frame nominal centre and half size of the three boards, keyed by name.

    The nominal centre is the fallback only; the instance's own jittered centre
    travels in ``spec['placement'][name]``.
    """
    scratch = rules.scratch_half(scratch_cells)
    return dict(
        scratch=dict(xy=SCRATCH_XY, half=(scratch, scratch), resolution=rules.SCRATCH_RESOLUTION,
                     inkable=True, gridded=False),
        final=dict(xy=FINAL_XY, half=(rules.PAPER_HALF, rules.PAPER_HALF), resolution=rules.RESOLUTION,
                   inkable=True, gridded=True),
        # The reference card takes ink like any other paper (spec 3: inking it is
        # permanent damage, logged as `reference_inked` collateral). v0 made it
        # ink-proof, which made the check vacuous; nothing in the score reads it.
        reference=dict(xy=REFERENCE_XY, half=(rules.PAPER_HALF, rules.PAPER_HALF), resolution=rules.RESOLUTION,
                       inkable=True, gridded=True))


def placement_problems(spec, footprint=FOOTPRINT):
    """Geometry checks the CPU validator runs without a simulator, on the instance's own
    placement: boards clear of each other and of Submit, stamps clear of the boards, of
    Submit and of each other, everything inside the footprint and the stamps inside their
    region. Returns (problems, report)."""
    frame = frame_rect(footprint)
    cells = int(spec['scratch_cells'])
    boards = board_shapes(spec)
    submit = submit_shape(tuple(spec['placement']['submit']))
    board_check = S.check([(shape.center[0], shape.center[1], shape.yaw) for shape in boards],
                          [shape.half for shape in boards], bounds=frame, min_gap=BOARD_GAP,
                          keepouts=[submit], keepout_gap=BOARD_GAP, names=list(BOARD_ORDER),
                          keepout_names=['submit'])
    stamp_poses = [(xy[0], xy[1], yaw) for xy, yaw in zip(spec['stamp_xy'], spec['initial_yaw_rad'])]
    stamp_check = S.check(stamp_poses, SOLE_RADIUS, bounds=stamp_region(footprint), min_gap=STAMP_GAP,
                          keepouts=boards + [submit], keepout_gap=STAMP_KEEPOUT_GAP,
                          names=[f'stamp_{index}' for index in range(len(stamp_poses))],
                          keepout_names=list(BOARD_ORDER) + ['submit'])
    problems = list(board_check['problems']) + list(stamp_check['problems'])
    if S.region_margin(submit, frame) < 0.:
        problems.append('submit leaves the work footprint')
    report = dict(min_board_gap_m=board_check['min_pair_gap_m'],
                  min_board_submit_gap_m=board_check['min_keepout_gap_m'],
                  min_board_footprint_margin_m=board_check['min_bounds_margin_m'],
                  min_stamp_gap_m=stamp_check['min_pair_gap_m'],
                  min_stamp_keepout_gap_m=stamp_check['min_keepout_gap_m'],
                  min_stamp_region_margin_m=stamp_check['min_bounds_margin_m'],
                  scratch_half_m=rules.scratch_half(cells))
    return problems, report


# ---- MJCF builders ----------------------------------------------------------
def add_stamp(body, name, mask, handle_rgba, sole_rgba=SOLE_RGBA):
    """Sole, handle and the enclosed die. Returns the geom names by role.

    ``body`` already carries the stamp's free joint and its world pose; the
    origin sits at the bottom face of the sole, so the body's z is exactly the
    printing height (``stamps_v2``'s convention, reused by the contact gate).
    """
    collision, visual, die = [], [], []
    for part, z, size, rgba, mass in (
            ('sole', SOLE_HALF_HEIGHT, (SOLE_RADIUS, SOLE_HALF_HEIGHT), sole_rgba, SOLE_MASS),
            ('handle', HANDLE_Z, (HANDLE_RADIUS, HANDLE_HALF_HEIGHT), handle_rgba, HANDLE_MASS)):
        for is_visual in (False, True):
            geom_name = f'{name}_{part}' + ('_visual' if is_visual else '')
            attributes = dict(name=geom_name, type='cylinder', pos=vec((0., 0., z)), size=vec(size),
                              rgba=vec(rgba), friction='1 .01 .001', condim='4', solref='.015 1',
                              solimp='.95 .99 .001', margin='0', group='1' if is_visual else '0')
            attributes.update(dict(contype='0', conaffinity='0', density='0') if is_visual
                              else dict(mass=str(mass)))
            ET.SubElement(body, 'geom', **attributes)
            (visual if is_visual else collision).append(geom_name)
    # The die: one dot per mask cell, inside the opaque sole. Visual only, so it
    # changes no contact, and hidden, so it leaks nothing (gate G3).
    for index, (px, py) in enumerate(rules.points_for_mask(mask)):
        geom_name = f'{name}_die_{index}'
        ET.SubElement(body, 'geom', name=geom_name, type='cylinder', pos=vec((px, py, DIE_Z)),
                      size=vec((rules.DOT_RADIUS, DIE_HALF_HEIGHT)), rgba=vec(np.r_[rules.INK_RGB / 255., 1.]),
                      group='1', contype='0', conaffinity='0', density='0')
        die.append(geom_name)
    return dict(collision_geoms=collision, visual_geoms=visual, die_geoms=die,
                sole_geom=f'{name}_sole', handle_geom=f'{name}_handle')


def add_board(body, assets, name, half, border_rgba, resolution):
    """A bordered paper board whose printed face is a textured plane.

    Returns the geom/texture names; the caller uploads pixels into the texture.
    """
    texture, material = f'{name}_texture', f'{name}_material'
    ET.SubElement(assets, 'texture', name=texture, type='2d', builtin='flat',
                  width=str(int(resolution)), height=str(int(resolution)), rgb1='1 1 1')
    ET.SubElement(assets, 'material', name=material, texture=texture, texrepeat='1 1',
                  texuniform='false', reflectance='0', specular='0', shininess='0', rgba='1 1 1 1')
    outer = (half[0] + BOARD_BORDER, half[1] + BOARD_BORDER)
    ET.SubElement(body, 'geom', name=f'{name}_board', type='box',
                  pos=vec((0., 0., (PAPER_THICKNESS - BOARD_SUPPORT_DEPTH) / 2)),
                  size=vec((outer[0], outer[1], (PAPER_THICKNESS + BOARD_SUPPORT_DEPTH) / 2)),
                  rgba=vec(border_rgba), group='0', friction='1 .01 .001', condim='4',
                  solref='.015 1', solimp='.95 .99 .001', margin='0')
    ET.SubElement(body, 'geom', name=f'{name}_board_visual', type='box',
                  pos=vec((0., 0., PAPER_THICKNESS / 2)),
                  size=vec((outer[0], outer[1], PAPER_THICKNESS / 2)), rgba=vec(border_rgba),
                  group='1', contype='0', conaffinity='0', density='0')
    ET.SubElement(body, 'geom', name=f'{name}_print_surface', type='plane',
                  pos=vec((0., 0., PAPER_THICKNESS + .00008)), size=vec((half[0], half[1], .001)),
                  material=material, group='1', contype='0', conaffinity='0')
    return dict(collision_geom=f'{name}_board', print_geom=f'{name}_print_surface', texture_name=texture)


# Firmer than MuJoCo's default (.02 2) so a light disc does not sink into a thin
# board, but not as stiff as stamps_v2's .002: a RoboCasa counter top is tiled
# from abutting slabs, and a stamp bridging a seam touches two of them at once,
# which a 2 ms time constant resolves into an impulse that launches it (measured
# on layout 4: 11 m/s at reset). Six substeps at the task's 1 ms step are stable.
PAIR_SOLREF = '.006 1'
PAIR_SOLIMP = '.98 .999 .001'


def add_contact_pairs(root, stamp_names, support_geoms):
    """Explicit stamp-to-support contact pairs; returns the pair names.

    Idempotent. RoboCasa re-runs ``_load_model`` when its first robot spawn collides
    (measured on layout 19, a dining-table kitchen: two calls) and hands back the same
    XML tree, so a second pass used to append a second ``<pair>`` under a name MuJoCo
    had already seen and the compile died with ``repeated name ... in pair``. A pair
    that is already there is rewritten in place instead of duplicated; duplicate
    support geoms are collapsed for the same reason.
    """
    contact = root.find('contact')
    if contact is None:
        contact = ET.SubElement(root, 'contact')
    existing = {pair.get('name'): pair for pair in contact.findall('pair')}
    supports, seen = [], set()
    for support in support_geoms:       # order-preserving dedupe
        if support not in seen:
            seen.add(support)
            supports.append(support)
    names = []
    for stamp in stamp_names:
        for part in ('sole', 'handle'):
            for support in supports:
                pair = f'{stamp}_{part}__{support}'
                element = existing.get(pair)
                if element is None:
                    element = ET.SubElement(contact, 'pair', name=pair)
                    existing[pair] = element
                element.set('geom1', f'{stamp}_{part}')
                element.set('geom2', support)
                element.set('condim', '4')
                element.set('friction', '1 1 .01 .001 .001')
                element.set('solref', PAIR_SOLREF)
                element.set('solimp', PAIR_SOLIMP)
                names.append(pair)
    return names
