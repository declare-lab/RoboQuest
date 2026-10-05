"""Identical boxes whose single opening can be on any face.

Third version of the "containers that open differently" task. Every box is
the same 26 x 26 x 22 cm case. One or two of its five faces (the bottom never)
carry an overlay panel: a plate held 3 mm proud of the case face with a round
knob in its middle, exactly like a real overlay drawer front. One of those
panels opens, in one of eleven ways; a second knobbed panel, when present, is
a dummy screwed to a solid wall. From outside, two boxes with knobs on the
same faces are the same box.

Mounting the panel outside the case instead of inside the opening is what lets
a front panel slide sideways or upward: the panel has somewhere to go. It also
means the hidden guides, latches and tabs all live behind the panel, out of
sight while the box is closed.

Face frame: every panel assembly is built in a frame in which its face looks
the way the front looks (outward normal -y, up +z, right +x); the assembly's
body carries the rotation about z that puts it on its real face. The four
upright faces are therefore the same face turned by multiples of 90 degrees
(the footprint is square), and the top panel's slide direction or hinge edge
is one more rotation about z. Motor routines written for the front and the
top work on every face.

Box frame: origin at the work-surface top under the box centre, x right, y into
the box (away from the robot at yaw 0), z up. Every length here is relative to
that origin, so the same box builds on the 0.92 m RoboCasa work surface that
the suite uses and on the 0.80 m table the mechanisms were tuned on.

Copied from the open-containers worktree (``panel_boxes_geometry.py``); the
study-table assembly (``append_panel_boxes``) was dropped because the RoboQuest
task lays the boxes out in its own task frame.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

from roboquest.containers.contract import MECHANISMS, RESPONSE, SEALED, SIDE_FACES, SIDE_MECHANISMS, TOP_MECHANISMS
from roboquest.containers.mechanism import BOWL_ASSET, DARK, ROBOCASA_ASSETS, DOOR_POP_RAD, DOOR_POP_SPRING, GUIDE_FRICTION, ITEM_ASSETS, ITEM_HALF, LATCH_POP_M, LATCH_RELEASE_N, METAL, PANEL, POP_SPRING, WOOD, _cyl, _span, _vec, add_latch_constraints, add_robocasa_object, latch_press_force, latch_released  # noqa: F401

W = D = .26                      # square footprint: the four upright faces are one face turned 90 deg
H = .22
T = .01                          # wall thickness
FLOOR_TOP = .01
WALL_TOP = .205                  # the walls end here; the top plate spans 0.205..0.215
LID_Z = (.205, .215)
LID_HALF = W / 2 - .0015         # the top plate is inset 1.5 mm: no coplanar upright faces, and a
                                 # side panel sliding up passes outside it
HOLE_X = (-.113, .113)           # the opening in a face, 22.6 x 16.5 cm: the hand's housing is 20.4 cm
                                 # across the fingers, and a 20 cm opening jammed it on both jambs
HOLE_Z = (.02, .185)
FACE_Y = (-W / 2, -W / 2 + T)    # the wall of a face, in that face's frame
PANEL_X = (-.128, .128)          # the overlay panel: 1.5 cm larger than the opening all round,
PANEL_Z = (.005, .200)           # 3 mm proud of the case face and 12 mm thick
PANEL_Y = (-W / 2 - .015, -W / 2 - .003)
# The one handle of this task: a round knob with a capped end, the same on
# every face. The cap matters twice over. A bare cylinder is pulled along its
# own axis by friction alone and the pads slide off it; and the 45 degree
# pitched hand carries its housing 2.2 cm behind the pads, so the grip has to
# sit far enough out for the housing to clear the panel.
KNOB_RADIUS, KNOB_SHAFT_HALF = .016, .022
KNOB_CAP_RADIUS, KNOB_CAP_HALF = .024, .004
KNOB_GRIP = .030                 # pinch this far out from the face the knob is mounted on
KNOB_REACH = 2 * KNOB_SHAFT_HALF + 2 * KNOB_CAP_HALF
KNOB_Z = (PANEL_Z[0] + PANEL_Z[1]) / 2
SLIDE_LID_LIFT = .0015           # the sliding top plate hangs on its joint 1.5 mm above the wall tops:
                                 # resting on them, its joint could not relieve the contact and the solver
                                 # clamped it with 46 N, which no reasonable push overcame
TRAY_Z = (.024, .032)
DRAWER_TRAVEL = .19
SIDE_TRAVEL = .245               # a sideways slide takes the panel's inner edge 4 mm past the far jamb
                                 # (PANEL_X[1] - SIDE_TRAVEL = -0.117 < HOLE_X[0]), so the open span is the
                                 # whole 22.6 cm opening, the span a door reach-in passes. At 0.215 the edge
                                 # stopped at -0.087, a 20.0 cm span for a housing 20.4-20.8 cm across the
                                 # fingers, and no hand pose passed (oracle traces, 2026-09-22)
UP_TRAVEL = .16                  # an upward slide uncovers all but the top 2.5 cm; the stiff guides
                                 # still let it creep a few millimetres, so the joint's own stop, not
                                 # the push, decides where it ends
UP_FRICTIONLOSS = 5.             # stiff guides hold the raised panel; MuJoCo's joint friction still creeps
UP_PANEL_MASS = .07              # in proportion to the load, so this panel is a light one (1.1 N with its
                                 # knob): at 0.19 kg it slid 4.6 cm shut over four minutes
DOOR_OPEN_RAD = 1.9
DOOR_PRESS_LEVER = .173          # the press lands 4.5 cm from the knob, this far from the door's hinge edge
TAB_HALF_LENGTH = .045           # turn-knob tab: upright it stands 7 mm behind the wall above the opening
TAB_THICK = .005
RAIL_PROUD = .002                # decorative guide rails on the wall behind a sliding panel

FACE_YAW = {'front': 0., 'right': math.pi / 2, 'back': math.pi, 'left': -math.pi / 2}
FREE_LID_MECHANISMS = ('lift_lid', 'twist_lid')
TRAY_MECHANISMS = ('drawer', 'press_drawer', 'turn_knob')
REACH_IN_MECHANISMS = ('press_door', 'door', 'slide_side', 'slide_up')


# Twist-lid bayonet, in the lid's frame (the numbers follow the v2 lid, shifted
# for the wider case).
SLOT_Z = (.175, .191)            # hidden slot inside the walls, just under the plate
DISK_RADIUS = .107
TAB_REACH = .125
TAB_ROOT = .100
POCKET_Y = (-.045, .045)
SLIDE_TRAVEL = .15               # the top plate slides this far along the top

# The native bowl the items end up in (objaverse bowl_1, placed fixed, meshes scaled uniformly by
# BOWL_SCALE: at its native 12 cm it holds one lemon; two to four need a wider and deeper bowl). The
# radii and heights are the native ones the open-containers rule was written against, scaled.
BOWL_SCALE = 1.6
BOWL_INNER_RADIUS = .05 * BOWL_SCALE
BOWL_OUTER_RADIUS = .062 * BOWL_SCALE
BOWL_HALF_HEIGHT = .0263 * BOWL_SCALE     # the asset's reg_bbox half height: the body sits this high on the top
BOWL_FLOOR_ABOVE_TOP = .011 * BOWL_SCALE  # the inside floor
BOWL_RIM_ABOVE_TOP = .054 * BOWL_SCALE


def _yaw_quat(yaw):
    return _vec((math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)))


def _rot(yaw):
    return np.array([[math.cos(yaw), -math.sin(yaw), 0.], [math.sin(yaw), math.cos(yaw), 0.], [0., 0., 1.]])


def _knob(body, prefix, face_point, axis, outward, mass=.03):
    """A capped round knob standing out of `face_point` along `outward`."""
    face_point = np.asarray(face_point, float)
    normal = np.asarray(outward, float)
    shaft = face_point + normal * KNOB_SHAFT_HALF
    cap = face_point + normal * (2 * KNOB_SHAFT_HALF + KNOB_CAP_HALF)
    return [_cyl(body, f'{prefix}_knob', shaft, KNOB_RADIUS, KNOB_SHAFT_HALF, DARK, axis=axis, mass=mass),
            _cyl(body, f'{prefix}_knob_cap', cap, KNOB_CAP_RADIUS, KNOB_CAP_HALF, DARK, axis=axis, mass=.01)]


def _side_knob(body, prefix, mass=.03):
    return _knob(body, prefix, (0., PANEL_Y[0], KNOB_Z), 'y', (0., -1., 0.), mass=mass)


def _top_knob(body, prefix, mass=.03):
    return _knob(body, prefix, (0., 0., LID_Z[1]), 'z', (0., 0., 1.), mass=mass)


def _knob_points(side, lid_lift=0.):
    """Where the knob's shaft centre and the pinch sit, in the face frame."""
    if side:
        return ((0., PANEL_Y[0] - KNOB_SHAFT_HALF, KNOB_Z), (0., PANEL_Y[0] - KNOB_GRIP, KNOB_Z))
    top = LID_Z[1] + lid_lift
    return ((0., 0., top + KNOB_SHAFT_HALF), (0., 0., top + KNOB_GRIP))


def _overlay_panel(body, prefix, mass=.15, knob=True):
    names = [_span(body, f'{prefix}_panel', PANEL_X, PANEL_Y, PANEL_Z, PANEL, mass=mass)]
    if knob:
        names.extend(_side_knob(body, prefix))
    return names


def _face_wall(body, prefix, hole, **kw):
    """One upright face of the case, solid or with the opening. A sliding top
    plate rides on these wall tops, so they are waxed like its runners: at
    ordinary wood friction the plate's own weight over eight resting contacts
    held it against a 7 N push."""
    if not hole:
        return [_span(body, f'{prefix}_wall', (-W / 2, W / 2), FACE_Y, (FLOOR_TOP, WALL_TOP), WOOD, **kw)]
    names = [_span(body, f'{prefix}_wall_left', (-W / 2, HOLE_X[0]), FACE_Y, (FLOOR_TOP, WALL_TOP), WOOD, **kw),
             _span(body, f'{prefix}_wall_right', (HOLE_X[1], W / 2), FACE_Y, (FLOOR_TOP, WALL_TOP), WOOD, **kw),
             _span(body, f'{prefix}_wall_sill', HOLE_X, FACE_Y, (FLOOR_TOP, HOLE_Z[0]), WOOD, **kw),
             _span(body, f'{prefix}_wall_head', HOLE_X, FACE_Y, (HOLE_Z[1], WALL_TOP), WOOD, **kw)]
    return names


def _guide_rails(body, prefix, direction):
    """Shallow rails on the wall behind a sliding panel. They sit on solid wall
    (never across the opening) and are hidden until the panel moves."""
    y = (FACE_Y[0] - RAIL_PROUD, FACE_Y[0])
    if direction == 'up':
        return [_span(body, f'{prefix}_rail_{side}', x, y, (PANEL_Z[0], PANEL_Z[1]), WOOD)
                for side, x in (('left', (-.125, -.117)), ('right', (.117, .125)))]
    return [_span(body, f'{prefix}_rail_{side}', PANEL_X, y, z, WOOD)
            for side, z in (('low', (.008, .016)), ('high', (.189, .197)))]


def _tray(body, prefix):
    """The drawer box behind the panel; it passes through the opening."""
    x = (HOLE_X[0] + .006, HOLE_X[1] - .006)
    y = (PANEL_Y[1], W / 2 - T - .006)
    top = TRAY_Z[1] + .028
    names = [_span(body, f'{prefix}_tray_floor', x, y, TRAY_Z, WOOD, mass=.08)]
    names.append(_span(body, f'{prefix}_tray_left', (x[0], x[0] + .008), y, (TRAY_Z[1], top), WOOD, mass=.02))
    names.append(_span(body, f'{prefix}_tray_right', (x[1] - .008, x[1]), y, (TRAY_Z[1], top), WOOD, mass=.02))
    names.append(_span(body, f'{prefix}_tray_back', x, (y[1] - .008, y[1]), (TRAY_Z[1], top), WOOD, mass=.02))
    return names


def _turn_tab(moving, prefix):
    """The knob turns on its own axis and carries a slim tab behind the panel.

    Upright the tab stands behind the wall above the opening and any pull
    presses it against that wall; turned about a quarter turn either way it
    lies across the opening, where nothing blocks it, and the panel pulls out
    like a drawer. The tab is 9 cm long, so it clears the head of the opening
    after about 24 degrees, within what the wrist manages on a pinched knob.
    """
    latch = ET.SubElement(moving, 'body', name=f'{prefix}_latch', pos='0 0 0')
    ET.SubElement(latch, 'joint', name=f'{prefix}_turn', type='hinge', axis='0 1 0',
                  pos=_vec((0., PANEL_Y[0], KNOB_Z)), range=_vec((-1.65, 1.65)), limited='true',
                  damping='0.05', frictionloss='0.15')
    names = _side_knob(latch, prefix)
    # Shaft through the panel (its own parent, so they never collide) into the
    # opening, and the tab just behind the wall.
    names.append(_cyl(latch, f'{prefix}_shaft', (0., (PANEL_Y[0] + FACE_Y[1]) / 2, KNOB_Z), .008,
                      abs(FACE_Y[1] - PANEL_Y[0]) / 2, DARK, axis='y', mass=.01))
    names.append(_span(latch, f'{prefix}_tab', (-TAB_THICK, TAB_THICK), (FACE_Y[1] + .002, FACE_Y[1] + .012),
                       (KNOB_Z, KNOB_Z + TAB_HALF_LENGTH * 2), WOOD, mass=.02))
    return names


def _lid_plate(body, prefix, mechanism, cues=False):
    """The top panel: the same inset plate on every box, with its knob."""
    friction = dict(friction=GUIDE_FRICTION) if mechanism == 'slide_lid' else {}
    names = [_span(body, f'{prefix}_lid', (-LID_HALF, LID_HALF), (-LID_HALF, LID_HALF), LID_Z, PANEL,
                   mass=.20, **friction)]
    if mechanism == 'twist_lid':
        turner = ET.SubElement(body, 'body', name=f'{prefix}_turner', pos='0 0 0')
        ET.SubElement(turner, 'joint', name=f'{prefix}_turn', type='hinge', axis='0 0 1',
                      pos=_vec((0., 0., LID_Z[1])), range=_vec((-1.6, 1.6)), limited='true',
                      damping='0.02', frictionloss='0.05')
        names.extend(_top_knob(turner, prefix))
        # Bayonet: a hidden disk under the plate whose two tabs sit in short
        # wall pockets at the closed angle, so the plate only lifts 3 mm.
        names.append(_cyl(turner, f'{prefix}_disk', (0., 0., SLOT_Z[0] + .007), DISK_RADIUS, .006, WOOD, mass=.10))
        for side, sign in (('left', -1.), ('right', 1.)):
            tab_x = (sign * TAB_ROOT, sign * TAB_REACH) if sign > 0 else (sign * TAB_REACH, sign * TAB_ROOT)
            names.append(_span(turner, f'{prefix}_tab_{side}', tab_x, (-.008, .008),
                               (SLOT_Z[0] + .002, SLOT_Z[1] - .003), WOOD, mass=.01))
    else:
        names.extend(_top_knob(body, prefix))
    if mechanism == 'slide_lid':
        # A web inboard of each wall carries a foot that runs in the hidden
        # wall slot: lifting is blocked, sliding is free.
        for side, sign in (('left', -1.), ('right', 1.)):
            web = (sign * .113, sign * .1185) if sign > 0 else (sign * .1185, sign * .113)
            foot = (sign * .113, sign * .125) if sign > 0 else (sign * .125, sign * .113)
            # The webs stop 1 cm short of the back wall at full travel.
            names.append(_span(body, f'{prefix}_web_{side}', web, (-.115, -.04), (.188, LID_Z[0]), WOOD,
                               mass=.01, friction=GUIDE_FRICTION))
            names.append(_span(body, f'{prefix}_tongue_{side}', foot, (-.115, -.04), (.177, .188), WOOD,
                               mass=.02, friction=GUIDE_FRICTION))
    return names


def _side_walls_for_lid(shell, prefix, mechanism):
    """Slots inside the upright walls for the lids that need them, in the lid's
    own frame: the sliding lid's tongues run a full-depth slot, the twist lid's
    tabs sit in short pockets."""
    names = []
    bearing = dict(friction=GUIDE_FRICTION) if mechanism == 'slide_lid' else {}
    for side, (x0, x1) in (('left', (-W / 2, -W / 2 + T)), ('right', (W / 2 - T, W / 2))):
        outer = x0 if side == 'left' else x1
        skin = (outer, outer + .003) if side == 'left' else (outer - .003, outer)
        slot_y = (-W / 2, W / 2) if mechanism == 'slide_lid' else POCKET_Y
        if mechanism == 'twist_lid':
            names.append(_span(shell, f'{prefix}_{side}_front', (x0, x1), (-W / 2, POCKET_Y[0]),
                               (FLOOR_TOP, WALL_TOP), WOOD))
            names.append(_span(shell, f'{prefix}_{side}_rear', (x0, x1), (POCKET_Y[1], W / 2),
                               (FLOOR_TOP, WALL_TOP), WOOD))
        names.append(_span(shell, f'{prefix}_{side}_lower', (x0, x1), slot_y, (FLOOR_TOP, SLOT_Z[0]), WOOD, **bearing))
        names.append(_span(shell, f'{prefix}_{side}_skin', skin, slot_y, SLOT_Z, WOOD, **bearing))
        names.append(_span(shell, f'{prefix}_{side}_upper', (x0, x1), slot_y, (SLOT_Z[1], WALL_TOP), WOOD, **bearing))
    return names


def build_box(world, index, mechanism, base, yaw=0., face='front', dummy_face=None, direction='right',
              hinge='left', lid_yaw=0.):
    """Append one box and return its evaluator metadata.

    `face` is the face whose panel opens (one of the four upright faces, or
    `top`); `dummy_face` is another face carrying an identical fixed panel, or
    None. `direction` is the sideways slide's direction, `hinge` the door's
    hinge edge, `lid_yaw` the top panel's slide direction or hinge edge.
    World metadata is returned next to the same points as offsets in the
    panel's own face frame (`local`), whose yaw is `frame_yaw`.
    """
    if mechanism not in MECHANISMS + (SEALED,):
        raise ValueError(mechanism)
    side = mechanism in SIDE_MECHANISMS or (mechanism == SEALED and face != 'top')
    if side and face not in SIDE_FACES:
        raise ValueError(f'{mechanism} needs an upright face, not {face}')
    if mechanism in TOP_MECHANISMS and face != 'top':
        raise ValueError(f'{mechanism} is a top mechanism')
    if dummy_face is not None and dummy_face == face:
        raise ValueError('the dummy panel needs its own face')
    prefix = f'pb{index}'
    base = [float(v) for v in base]
    yaw = float(yaw)
    frame_yaw = yaw + (FACE_YAW[face] if side else float(lid_yaw))
    rotation = _rot(frame_yaw)
    local = {}

    def at(key, offset):
        local[key] = [float(v) for v in offset]
        return (np.asarray(base) + rotation @ np.asarray(offset, float)).tolist()

    lid_mechanism = mechanism if mechanism in TOP_MECHANISMS else None
    lid_lift = SLIDE_LID_LIFT if mechanism == 'slide_lid' else 0.
    shell = ET.SubElement(world, 'body', name=f'{prefix}_shell', pos=_vec(base), quat=_yaw_quat(yaw))
    names = [_span(shell, f'{prefix}_floor', (-W / 2, W / 2), (-W / 2, W / 2), (0., FLOOR_TOP), WOOD)]
    panels = {face: mechanism}
    if dummy_face is not None:
        panels[dummy_face] = 'dummy'
    # The sliding and twisting lids need slots inside two opposite walls; those walls are cut in the
    # lid's frame (turned by ``lid_yaw``) and built once, below, so the plain walls of the two box
    # faces they land on are left out here: the box's left and right faces when the lid frame is the
    # box frame or turned by a half turn, its front and back faces at a quarter turn.
    slot_faces = ()
    if lid_mechanism in ('slide_lid', 'twist_lid'):
        slot_faces = ('left', 'right') if int(round(float(lid_yaw) / (math.pi / 2))) % 2 == 0 else ('front', 'back')
    for wall_face in SIDE_FACES:
        # Each upright wall is built in its own face frame, inside the shell.
        wall = ET.SubElement(shell, 'body', name=f'{prefix}_face_{wall_face}', pos='0 0 0',
                             quat=_yaw_quat(FACE_YAW[wall_face]))
        opens_here = wall_face == face and side and mechanism != SEALED
        if wall_face not in slot_faces:
            names.extend(_face_wall(wall, f'{prefix}_{wall_face}', opens_here,
                                    **(dict(friction=GUIDE_FRICTION) if lid_mechanism == 'slide_lid' else {})))
            if opens_here and mechanism in ('slide_side', 'slide_up'):
                names.extend(_guide_rails(wall, f'{prefix}_{wall_face}', 'up' if mechanism == 'slide_up' else 'side'))
        # A fixed knobbed panel (the dummy, or the sealed box's own) goes on any face, slotted or not:
        # it hangs 3 mm proud of the face, outside the slotted wall's skin.
        if panels.get(wall_face) in (None, 'dummy') or mechanism == SEALED:
            if wall_face in panels:
                # A fixed panel: the dummy, or the sealed box's own panel.
                names.extend(_overlay_panel(wall, f'{prefix}_{wall_face}', mass=None))
    if lid_mechanism in ('slide_lid', 'twist_lid'):
        slotted = ET.SubElement(shell, 'body', name=f'{prefix}_face_slots', pos='0 0 0', quat=_yaw_quat(lid_yaw))
        names.extend(_side_walls_for_lid(slotted, prefix, lid_mechanism))
    if lid_mechanism is None:
        # The top plate is part of the case; it carries a knob when the top is
        # the dummy face.
        top = ET.SubElement(shell, 'body', name=f'{prefix}_face_top', pos='0 0 0', quat=_yaw_quat(lid_yaw))
        names.append(_span(top, f'{prefix}_top_plate', (-LID_HALF, LID_HALF), (-LID_HALF, LID_HALF), LID_Z, PANEL))
        if dummy_face == 'top' or (mechanism == SEALED and face == 'top'):
            names.extend(_top_knob(top, f'{prefix}_top'))

    info = dict(mechanism=mechanism, response=RESPONSE[mechanism], face=face, dummy_face=dummy_face,
                shell_body=f'{prefix}_shell', base=base, yaw=yaw, frame_yaw=frame_yaw, shell_geoms=names,
                moving_body=None, joint_name=None, latch_equality=None, local=local,
                direction=direction if mechanism == 'slide_side' else None,
                hinge=hinge if mechanism in ('door', 'press_door') else None,
                lid_yaw=float(lid_yaw) if not side else None,
                knob_center=at('knob_center', _knob_points(side, lid_lift)[0]),
                knob_grip=at('knob_grip', _knob_points(side, lid_lift)[1]), lid_lift=lid_lift,
                interior=dict(center=at('interior_center', (0., 0., FLOOR_TOP)), half=[.11, .11, .19]),
                pick='top' if not side else 'reach_in')
    if dummy_face is not None:
        dummy_yaw = yaw + (FACE_YAW[dummy_face] if dummy_face != 'top' else float(lid_yaw))
        centre, grip = _knob_points(dummy_face != 'top')
        info['dummy_knob_center'] = (np.asarray(base) + _rot(dummy_yaw) @ np.asarray(centre)).tolist()
        info['dummy_knob_grip'] = (np.asarray(base) + _rot(dummy_yaw) @ np.asarray(grip)).tolist()
        info['dummy_frame_yaw'] = dummy_yaw
    if mechanism == SEALED:
        return info

    moving = ET.SubElement(world, 'body', name=f'{prefix}_moving', pos=_vec([base[0], base[1], base[2] + lid_lift]),
                           quat=_yaw_quat(frame_yaw))
    info['moving_body'] = f'{prefix}_moving'
    if mechanism in ('drawer', 'press_drawer'):
        joint = ET.SubElement(moving, 'joint', name=f'{prefix}_slide', type='slide', axis='0 -1 0',
                              range=_vec((0., DRAWER_TRAVEL)), limited='true', damping='6', frictionloss='0.6')
        info['joint_name'] = f'{prefix}_slide'
        info['moving_geoms'] = _overlay_panel(moving, prefix) + _tray(moving, prefix)
        if mechanism == 'press_drawer':
            joint.set('range', _vec((-.006, DRAWER_TRAVEL)))
            info['latch_equality'] = f'{prefix}_latch'
            info['latch_rule'] = dict(sign=1., preload=POP_SPRING * LATCH_POP_M, threshold=LATCH_RELEASE_N, unit='N')
            info['pop_tendon'] = f'{prefix}_pop'
        info['interior'] = dict(center=at('interior_center', (0., -.01, TRAY_Z[1])), half=[.09, .09, .14])
        info['open_when'] = dict(joint_above=.10)
        info['pick'] = 'tray'
    elif mechanism == 'turn_knob':
        ET.SubElement(moving, 'joint', name=f'{prefix}_slide', type='slide', axis='0 -1 0',
                      range=_vec((0., DRAWER_TRAVEL)), limited='true', damping='6', frictionloss='0.6')
        info['joint_name'] = f'{prefix}_slide'
        info['moving_geoms'] = _overlay_panel(moving, prefix, knob=False) + _tray(moving, prefix)
        info['latch_body'] = f'{prefix}_latch'
        info['latch_joint'] = f'{prefix}_turn'
        info['latch_geoms'] = _turn_tab(moving, prefix)
        info['interior'] = dict(center=at('interior_center', (0., -.01, TRAY_Z[1])), half=[.09, .09, .14])
        info['open_when'] = dict(joint_above=.10)
        info['pick'] = 'tray'
    elif mechanism in ('door', 'press_door'):
        sign = -1. if hinge == 'left' else 1.
        hinge_x = sign * PANEL_X[1]
        ET.SubElement(moving, 'joint', name=f'{prefix}_hinge', type='hinge',
                      axis='0 0 -1' if sign < 0 else '0 0 1',
                      pos=_vec((hinge_x, PANEL_Y[1], 0.)), range=_vec((-.02, DOOR_OPEN_RAD)), limited='true',
                      damping='0.6', frictionloss='0.25')
        info['joint_name'] = f'{prefix}_hinge'
        info['moving_geoms'] = _overlay_panel(moving, prefix)
        info['hinge_world'] = at('hinge_world', (hinge_x, PANEL_Y[1], 0.))
        info['hinge_sign'] = sign
        if mechanism == 'press_door':
            info['latch_equality'] = f'{prefix}_latch'
            info['pop_tendon'] = f'{prefix}_pop'
            info['pop'] = dict(stiffness=DOOR_POP_SPRING, dead_band_from=DOOR_POP_RAD)
            info['latch_rule'] = dict(sign=1., preload=DOOR_POP_SPRING * DOOR_POP_RAD,
                                      threshold=LATCH_RELEASE_N * DOOR_PRESS_LEVER, unit='N.m')
        info['interior'] = dict(center=at('interior_center', (0., -.03, FLOOR_TOP)), half=[.10, .08, .16])
        info['open_when'] = dict(joint_above=1.0)
    elif mechanism == 'slide_side':
        sign = 1. if direction == 'right' else -1.
        ET.SubElement(moving, 'joint', name=f'{prefix}_slide', type='slide',
                      axis='1 0 0' if sign > 0 else '-1 0 0', range=_vec((0., SIDE_TRAVEL)), limited='true',
                      damping='3', frictionloss='0.5')
        info['joint_name'] = f'{prefix}_slide'
        info['moving_geoms'] = _overlay_panel(moving, prefix)
        info['slide_sign'] = sign
        info['interior'] = dict(center=at('interior_center', (0., -.03, FLOOR_TOP)), half=[.10, .08, .16])
        info['open_when'] = dict(joint_above=.19)
    elif mechanism == 'slide_up':
        # Stiff friction constraint, like the flip lid's stay hinge: with the
        # default soft one the raised panel crept back down over the following
        # half minute and closed its own opening.
        ET.SubElement(moving, 'joint', name=f'{prefix}_slide', type='slide', axis='0 0 1',
                      range=_vec((0., UP_TRAVEL)), limited='true', damping='3',
                      frictionloss=str(UP_FRICTIONLOSS), solreffriction='0.002 1',
                      solimpfriction='0.9999 0.9999 0.0001')
        info['joint_name'] = f'{prefix}_slide'
        info['moving_geoms'] = _overlay_panel(moving, prefix, mass=UP_PANEL_MASS)
        info['interior'] = dict(center=at('interior_center', (0., -.03, FLOOR_TOP)), half=[.10, .08, .16])
        info['open_when'] = dict(joint_above=.13)
    elif mechanism == 'lift_lid':
        ET.SubElement(moving, 'freejoint', name=f'{prefix}_free')
        info['joint_name'] = f'{prefix}_free'
        info['moving_geoms'] = _lid_plate(moving, prefix, mechanism)
        info['open_when'] = dict(lid_offset_beyond=.14, lid_center_above=base[2] + LID_Z[0] + .08)
    elif mechanism == 'twist_lid':
        ET.SubElement(moving, 'freejoint', name=f'{prefix}_free')
        info['joint_name'] = f'{prefix}_free'
        info['moving_geoms'] = _lid_plate(moving, prefix, mechanism)
        info['handle_body'] = f'{prefix}_turner'
        info['handle_joint'] = f'{prefix}_turn'
        info['open_when'] = dict(lid_offset_beyond=.14, lid_center_above=base[2] + LID_Z[0] + .08)
    elif mechanism == 'slide_lid':
        ET.SubElement(moving, 'joint', name=f'{prefix}_lidslide', type='slide', axis='0 1 0',
                      range=_vec((0., SLIDE_TRAVEL)), limited='true', damping='3', frictionloss='0.5')
        info['joint_name'] = f'{prefix}_lidslide'
        info['moving_geoms'] = _lid_plate(moving, prefix, mechanism)
        info['open_when'] = dict(joint_above=.11)
    elif mechanism == 'flip_lid':
        ET.SubElement(moving, 'joint', name=f'{prefix}_flip', type='hinge', axis='-1 0 0',
                      pos=_vec((0., LID_HALF, LID_Z[0])), range=_vec((0., 1.95)), limited='true',
                      damping='0.5', frictionloss='0.6', solreffriction='0.004 1',
                      solimpfriction='0.99 0.999 0.001')
        info['joint_name'] = f'{prefix}_flip'
        info['moving_geoms'] = _lid_plate(moving, prefix, mechanism)
        info['hinge_world'] = at('hinge_world', (0., LID_HALF, LID_Z[0]))
        info['open_when'] = dict(joint_above=1.2)
    if mechanism in TOP_MECHANISMS:
        # The sliding lid uncovers the near part of the interior, so its item sits there.
        y_off = -.05 if mechanism == 'slide_lid' else 0.
        info['interior'] = dict(center=at('interior_center', (0., y_off, FLOOR_TOP)), half=[.10, .10, .18])
    return info


def item_pose_in(info, rng, half_z=None):
    """Seeded item pose inside a box, in world coordinates: jittered about the
    interior centre, resting on the floor or the tray, its long axis across the
    opening so the demonstration's pinch fits through. `half_z` is the item's
    own half height (the lemon's when it is not given)."""
    interior = info['interior']
    mechanism = info['mechanism']
    jitter = rng.uniform(-.015, .015, size=2) * np.array([.55, 1.])
    if mechanism in REACH_IN_MECHANISMS:
        jitter = rng.uniform(-.012, .012, size=2)
    local_yaw = float(rng.uniform(-math.pi / 3, math.pi / 3))
    frame_yaw = float(info['frame_yaw'])
    rotation = _rot(frame_yaw)[:2, :2]
    xy = np.asarray(interior['center'][:2]) + rotation @ np.asarray(jitter)
    half_z = float(ITEM_HALF[2] if half_z is None else half_z)
    return ([float(xy[0]), float(xy[1]), float(interior['center'][2] + half_z + .0015)],
            frame_yaw + local_yaw)


def box_openness(info, joint_value=None, lid_center=None, latch_turn=None):
    """Private diagnostic: is this box physically open right now?"""
    mechanism = info['mechanism']
    if mechanism == SEALED:
        return dict(open=False)
    rule = info['open_when']
    if mechanism in FREE_LID_MECHANISMS:
        center = np.asarray(lid_center, dtype=float)
        rest = np.asarray(info['base'], dtype=float)
        lifted = center[2] >= rule['lid_center_above']
        moved = float(np.linalg.norm(center[:2] - rest[:2])) >= rule['lid_offset_beyond']
        return dict(lid_center=center.tolist(), open=bool(lifted or moved))
    result = dict(joint=float(joint_value), open=bool(joint_value >= rule['joint_above']))
    if mechanism == 'turn_knob':
        result['latch_turn'] = float(latch_turn) if latch_turn is not None else None
    return result
