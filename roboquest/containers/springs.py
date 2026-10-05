"""Passive closers on the panel boxes (spec 4.5).

A closer is a joint spring plus damper on a panel whose spring return is natural: a hinged flap
(``door``), a push door (``press_door``) or a slide panel (``slide_side``, ``slide_up``, ``slide_lid``).
Never a twist lid, a turn knob or a latched drawer, which stay put by design, and never a free lid. The
flip lid is left out as well: with a spring return a top lid has nowhere to rest open, so a one-armed
robot could only hold it (the oracle lifts it by the knob, drives round and pushes its edge over; the
lid was shut again before the push), whereas a side panel that swings or slides shut merely leans on
the arm that is already in the opening. Which boxes get one is drawn per instance from the ``structure`` stream
(:mod:`~roboquest.tasks.unfamiliar_containers`) and stored as ``spec['closers']``; nothing
about it is written to the MJCF or the goal, so a closer box looks exactly like the others.

The closer is applied to the compiled model after ``_setup_references`` with
:func:`roboquest.closers.apply_closer` (the shared recipe), then tuned per mechanism kind:
the joint's dry friction is lowered so the spring closes the panel fully instead of stalling part way
(the shared recipe only raises the spring to a multiple of the friction, which leaves a door a third
open), and the damping is set from the measured closing profile. The numbers in :data:`TUNING` were
measured with :func:`closing_ticks` on the built kitchen (see the task handoff). Spec 4.5 asked for
about ten seconds; spec 6 asks for the timing to be set against the oracle's open-to-retract duration.

**The window a closer leaves** (job CONTAINERS-CLOSER, 2026-09-22, on CONTAINERS-PICK's measurements).
A panel with a closer is only worth minting when the opening stays passable for the hand after the
knob is let go: a one-armed robot cannot hold the panel and reach in with the same hand. The
measured hand is 0.208 m across the fingers, so a ``slide_side`` panel must stand at 0.230 of its
0.245 m travel for the housing to pass, and at any closing rate a policy could still open against
the 1.5 cm it leaves lasts 18-33 ticks (0.46-0.84 mm/tick measured); the shortest reach-in after
the release is 86-140 ticks. So :data:`ENTRY_WHEN` names, per kind, the joint value below which the
opening is shut to the hand's *entry* (not the collateral check's :data:`CLOSED_WHEN`), the window
is *measured* on the built box model -- the panel set to its open travel, released, the ticks counted
until the joint falls below the entry value (:func:`measure_closer_window`) -- and a kind whose
window is under :data:`CLOSER_WINDOW_MIN_TICKS` never gets a closer (:func:`usable_closer`;
:func:`draw_closers` draws only among usable kinds and ``validate_spec`` refuses a closer anywhere
else, so a candidate that breaks the rule is redrawn, never patched). Under the tuning here that
leaves the doors, the sliding lid and the upward slide with closers (the last two slowed so that
their windows clear the minimum with the item at the near end of its jitter) and takes the closer
off the sideways slide, which no closing rate can carry: its window would need 200 ticks for
1.5 cm, a full close of 160 s. The per-kind windows are :data:`CLOSER_WINDOW_TICKS`, locked to the
measurement by the tests; the build records the measured window per panel in the spec's private
placement block (``closer_window_ticks``) for the reach probe and the oracle.
"""
import functools
import xml.etree.ElementTree as ET

import numpy as np

from roboquest import closers
from roboquest.containers import panels as P
from roboquest.containers.contract import TOP_MECHANISMS

CLOSER_MECHANISMS = ('door', 'press_door', 'slide_side', 'slide_up', 'slide_lid')
CLOSE_TIME_S = 20.
FLIP_OPEN_RAD = 1.95                # the flip lid's joint limit: leaning back past the vertical
# The fully open joint value each mechanism is measured from: the joint's far limit, except the doors,
# which the oracle pushes to 1.75 rad (at the 1.9 rad limit a door can meet the neighbouring dummy knob).
OPEN_QPOS = {'door': 1.75, 'press_door': 1.75, 'slide_side': P.SIDE_TRAVEL,
             'slide_up': P.UP_TRAVEL, 'slide_lid': P.SLIDE_TRAVEL, 'flip_lid': FLIP_OPEN_RAD}
# Below this joint value the opening is shut to the hand again (the door's free edge less than 12 cm
# out, a slide covering more than half the opening): the collateral check's notion of "closed".
CLOSED_WHEN = {'door': .5, 'press_door': .5, 'slide_side': .095, 'slide_up': .065, 'slide_lid': .055,
               'flip_lid': .6}
# Spring rest position past the closed limit: far past it, so the spring is preloaded and its force
# stays nearly constant over the travel (a door closer, not a rubber band). The joint range stops the
# panel at its closed limit. A constant force closes at a constant speed, which keeps the opening
# usable for about a third of the closing time (an exponential return covers half the opening in a
# tenth of it), and the preload is what the hand feels when it opens the panel.
REST_PAST = {'door': 1.5, 'press_door': 1.5, 'flip_lid': 1.5, 'slide_side': .15, 'slide_up': .15,
             'slide_lid': .15}
# Per-mechanism tuning applied on top of the shared recipe: the joint's dry friction (N or N.m), and
# the spring stiffness and damping written to the model (None keeps the recipe's value). Measured on
# the built kitchen (containers handoff, closer study): shut to the hand / fully closed from fully open.
# The upward slide and the sliding lid were slowed on 2026-09-22 (CONTAINERS-CLOSER) so that the window
# they leave the hand (below) clears CLOSER_WINDOW_MIN_TICKS: slide_up damping 70 -> 200, slide_lid
# 80 -> 140. The sideways slide keeps its tuning for the record; no closer is drawn on it any more.
TUNING = {
    'door': dict(frictionloss=.0, stiffness=.15, damping=3.9),        # 0.22-0.51 N.m over the travel
    'press_door': dict(frictionloss=.0, stiffness=.15, damping=3.9),  # rests ajar against its pop spring
    'slide_side': dict(frictionloss=.0, stiffness=2.0, damping=47.),  # 0.3-0.7 N
    'slide_up': dict(frictionloss=.9, stiffness=1.0, damping=200.),   # gravity (1.1 N) does most of the closing
    'slide_lid': dict(frictionloss=.0, stiffness=2.0, damping=140.),
}

# ---- the window a closer leaves the hand (CONTAINERS-CLOSER, 2026-09-22) ------------------------------
# The joint value below which the opening is shut to the hand's *entry*, per kind (RESIDUE.md of job
# CONTAINERS-PICK, measured on the compiled hand: housing 0.208 across the fingers, 0.064 across the
# slide, its full width 0.026-0.07 above the eef; flange r 0.044 from 0.09):
#   slide_side  the panel's inner edge must be past the housing's side on the travel side by 4 mm with
#               the hand 6 mm toward the far jamb: PANEL_X 0.128 + housing half 0.104 - 0.006 + 0.004 =
#               0.230 of the 0.245 m travel (oracle ``slide_side_needed``).
#   slide_up    the panel's bottom edge must be above the 20-degree pitched housing's top at the face
#               plane, entering level at the item's height: for the tallest item of the pool (the lime,
#               half height 0.0352) 0.1256 of the 0.16 m rise (oracle ``slide_up_needed``); 0.126.
#   slide_lid   the housing at the hover must fit under the lid's edge: hand y = item_y - 0.03 (the far
#               bias), edge past it by the housing's half thickness 0.032 and the 4 mm margin, and the
#               item sits at y -0.05 +- 0.015 in the lid frame (``panels.item_pose_in``), so the near end
#               of its jitter needs the lid at -0.035 - 0.03 + 0.036 + LID_HALF 0.1285 = 0.0995 of the
#               0.15 m travel (oracle ``slide_lid_can_enter`` / ``slide_lid_entry_y``); 0.100.
#   door        the door swings away from the opening: a housing already in front of the face stops its
#               return and holds it, so the reach-in fails only once the free edge is within 12 cm of
#               the face -- the collateral check's own 0.5 rad (CLOSED_WHEN).
ENTRY_WHEN = {'door': .5, 'press_door': .5, 'slide_side': .230, 'slide_up': .126, 'slide_lid': .100}
# The least window (20 Hz ticks from the release until the joint falls below ENTRY_WHEN) a closer may
# leave. Measured need: the shortest reach-in after letting go of a slid panel's knob is 86-140 ticks
# (back off the knob 0.15, creep 0.12, entry pose, level entry: the panel read 86 ticks after the push;
# or 92 ticks to the entry pose plus about 50 from there to the face plane), and the sliding lid's
# quick pick reached its hover 117-129 ticks after the push. 200
# ticks (10 s) is that path with the slack of one retried waypoint; a policy that opens a box and
# reaches in at once is never blocked, one that goes away for longer finds the box shut again.
CLOSER_WINDOW_MIN_TICKS = 200
TICK_S = .05                 # the kitchen's control tick (robocasa control_freq 20)
PHYSICS_TIMESTEP = .002      # robosuite's simulation timestep, the kitchen's too (asserted by the build test)
SUBSTEPS = int(round(TICK_S / PHYSICS_TIMESTEP))
WINDOW_MAX_TICKS = 2400      # a window is measured out to two minutes of sim time
# The windows measured on the built box model with :func:`closer_windows` at the tuning above
# (2026-09-22, CONTAINERS-CLOSER; ticks from the release at OPEN_QPOS until the joint falls below
# ENTRY_WHEN; the tests re-measure them and the build test compares them with the built kitchen):
CLOSER_WINDOW_TICKS = {'door': 253, 'press_door': 253, 'slide_side': 19, 'slide_up': 289, 'slide_lid': 256}
# (door 4.94 mrad/tick; slide_side 0.79 mm/tick -- 0.46-0.84 measured in the kitchen; slide_up 0.118 mm/tick at
# damping 200 against 0.337 at the old 70, the kitchen's 0.34; slide_lid 0.195 at 140 against 0.342 at the old
# 80, the kitchen's 0.34. Shut to the hand (CLOSED_WHEN): door 258, slide_side 225, slide_up 864, slide_lid 534
# ticks. A slide_side closer damped to 500 would still leave only 194 ticks and take 2389 to shut.)
CLOSER_WINDOW_TOLERANCE = .10   # a re-measurement may differ from the table by this share (solver settings)
USABLE_CLOSER_MECHANISMS = tuple(m for m in CLOSER_MECHANISMS if CLOSER_WINDOW_TICKS[m] >= CLOSER_WINDOW_MIN_TICKS)


def has_closer_mechanism(mechanism):
    """A spring-natural mechanism (the design's vocabulary): the closer *could* be fitted."""
    return mechanism in CLOSER_MECHANISMS


def usable_closer(mechanism):
    """A spring-natural mechanism whose closer leaves the hand a usable window: the closer *is* fitted."""
    return has_closer_mechanism(mechanism) and CLOSER_WINDOW_TICKS[mechanism] >= CLOSER_WINDOW_MIN_TICKS


def draw_closers(rng, boxes, share=.5):
    """Which boxes get a closer: every box with a usable closer mechanism with probability ``share``, and
    at least one whenever such a box exists. ``boxes`` are spec rows with ``index`` and ``mechanism``;
    returns sorted indices. Deterministic in ``rng`` (the structure stream)."""
    candidates = [int(b['index']) for b in boxes if usable_closer(b['mechanism'])]
    if not candidates:
        return []
    chosen = [i for i in candidates if float(rng.uniform()) < share]
    if not chosen:
        chosen = [candidates[int(rng.integers(len(candidates)))]]
    return sorted(chosen)


def closer_problems(boxes, closers):
    """Spec problems with a closer list: a closer on a box without a spring-natural mechanism, on one
    whose kind leaves the hand too short a window, or none at all where a usable one exists."""
    problems = []
    for index in closers:
        if not 0 <= index < len(boxes) or not has_closer_mechanism(boxes[index]['mechanism']):
            problems.append(f'closer on box {index}, which has no spring-natural mechanism')
        elif not usable_closer(boxes[index]['mechanism']):
            mechanism = boxes[index]['mechanism']
            problems.append(f'closer on box {index}: a {mechanism} closer leaves the hand '
                            f'{CLOSER_WINDOW_TICKS[mechanism]} ticks after the release, under the '
                            f'{CLOSER_WINDOW_MIN_TICKS} it needs')
    if any(usable_closer(b['mechanism']) for b in boxes) and not closers:
        problems.append('a mechanism with a usable closer is present but no box has a closer')
    return problems


def rest_qpos(mechanism):
    return -float(REST_PAST[mechanism])


def closed_threshold(mechanism):
    return float(CLOSED_WHEN[mechanism])


def entry_threshold(mechanism):
    return float(ENTRY_WHEN[mechanism])


def apply_box_closer(sim, box, close_time_s=CLOSE_TIME_S, tuning=None):
    """Turn the box's panel joint into a self-closing one. Returns the parameters written (for the spec)."""
    mechanism = box['mechanism']
    if not has_closer_mechanism(mechanism):
        raise ValueError(f'{mechanism} takes no closer')
    joint = box['joint_name']
    params = closers.apply_closer(sim, joint, rest_qpos(mechanism), close_time_s=close_time_s,
                                  open_qpos=OPEN_QPOS[mechanism])
    model = sim.model
    jid = model.joint_name2id(joint)
    dadr = int(model.jnt_dofadr[jid])
    tune = dict(TUNING[mechanism])
    if tuning and mechanism in tuning:
        tune.update(tuning[mechanism])
    if tune.get('frictionloss') is not None:
        model.dof_frictionloss[dadr] = float(tune['frictionloss'])
    if tune.get('stiffness') is not None:
        model.jnt_stiffness[jid] = float(tune['stiffness'])
    if tune.get('damping') is not None:
        model.dof_damping[dadr] = float(tune['damping'])
    params.update(mechanism=mechanism, stiffness=float(model.jnt_stiffness[jid]),
                  damping=float(model.dof_damping[dadr]), frictionloss=float(model.dof_frictionloss[dadr]),
                  closed_when=closed_threshold(mechanism), entry_when=entry_threshold(mechanism),
                  recipe_stiffness=params['stiffness'], recipe_damping=params['damping'])
    return params


def _release_latch(sim, box):
    """A latched panel is held shut by its equality until pressed; a measurement starts released."""
    if box.get('latch_equality'):
        import mujoco
        eq = mujoco.mj_name2id(sim.model._model, mujoco.mjtObj.mjOBJ_EQUALITY, box['latch_equality'])
        sim.data.eq_active[eq] = 0


def closing_ticks(env, box, closed_tol=None, max_ticks=1200):
    """Measure how many 20 Hz ticks the panel needs to shut from fully open (set directly), stepping the
    env with a zero action. ``closed_tol`` defaults to the closed threshold, so the measurement means
    "the opening is shut to the hand" (the joint within CLOSED_WHEN of its closed limit, where the
    preloaded spring rests it); a second measurement with a tight tolerance gives the full settling.
    Returns (ticks or None, final joint value)."""
    mechanism = box['mechanism']
    tol = closed_threshold(mechanism) if closed_tol is None else float(closed_tol)
    _release_latch(env.sim, box)
    zero = np.zeros(env.action_dim)
    return closers.closing_time(env.sim, box['joint_name'], lambda: env.step(zero), OPEN_QPOS[mechanism], tol,
                                max_ticks=max_ticks)


def measure_closer_window(sim, box, tick, max_ticks=WINDOW_MAX_TICKS):
    """The window a closer leaves the hand, measured: the panel set to its open travel (:data:`OPEN_QPOS`)
    and released, ``tick()`` advancing one control tick, the ticks counted until the joint falls below
    :data:`ENTRY_WHEN`. Returns ``dict(ticks, open, entry, final, rate_m_per_tick)``; ``ticks`` is None when
    the panel never comes below the entry value within ``max_ticks`` (a window longer than that)."""
    mechanism = box['mechanism']
    model, data = sim.model, sim.data
    jid = model.joint_name2id(box['joint_name'])
    qadr, dadr = int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    open_qpos, entry = float(OPEN_QPOS[mechanism]), entry_threshold(mechanism)
    _release_latch(sim, box)
    data.qpos[qadr] = open_qpos
    data.qvel[dadr] = 0.
    sim.forward()
    ticks = None
    for count in range(1, int(max_ticks) + 1):
        tick()
        if float(data.qpos[qadr]) < entry:
            ticks = count
            break
    final = float(data.qpos[qadr])
    rate = (open_qpos - entry) / ticks if ticks else (open_qpos - final) / max(1, int(max_ticks))
    return dict(ticks=ticks, open=open_qpos, entry=entry, final=round(final, 5), rate_m_per_tick=round(rate, 6))


def box_model_xml(mechanisms=CLOSER_MECHANISMS):
    """A bare MuJoCo model holding one box per mechanism (the task's own builder, a metre apart), with the
    kitchen's physics timestep: the closers' joints see exactly the panels the kitchen builds."""
    root = ET.Element('mujoco', model='panel_box_closers')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep=str(PHYSICS_TIMESTEP))
    world = ET.SubElement(root, 'worldbody')
    infos = {}
    for k, mechanism in enumerate(mechanisms):
        face = 'top' if mechanism in TOP_MECHANISMS else 'front'
        infos[mechanism] = P.build_box(world, k, mechanism, [1.5 * k, 0., 0.], yaw=0., face=face)
    P.add_latch_constraints(root, dict(boxes=infos))
    return ET.tostring(root, encoding='unicode'), infos


@functools.lru_cache(maxsize=4)
def closer_windows(mechanisms=CLOSER_MECHANISMS, tuning_items=None):
    """Measure the window every closer kind leaves the hand, on the built box model (:func:`box_model_xml`)
    with the closers applied exactly as the kitchen applies them, stepping :data:`SUBSTEPS` physics steps
    per tick. ``tuning_items`` is an optional frozen ``((mechanism, ((key, value), ...)), ...)`` override of
    :data:`TUNING` for studies. Cached per process: the boxes are the same in every kitchen, so the
    windows are too (the build test checks that on a built kitchen). Returns ``{mechanism: dict(ticks,
    open, entry, final, rate_m_per_tick, shut_ticks)}``, ``shut_ticks`` being the collateral check's
    "shut to the hand" (the joint within CLOSED_WHEN of the limit)."""
    import mujoco
    from robosuite.utils.binding_utils import MjSim
    tuning = {m: dict(kv) for m, kv in tuning_items} if tuning_items else None
    xml, infos = box_model_xml(tuple(mechanisms))
    sim = MjSim.from_xml_string(xml)
    model, data = sim.model._model, sim.data._data

    def tick():
        mujoco.mj_step(model, data, nstep=SUBSTEPS)

    out = {}
    for mechanism in mechanisms:
        box = infos[mechanism]
        params = apply_box_closer(sim, box, tuning=tuning)
        row = measure_closer_window(sim, box, tick)
        shut = closers.closing_time(sim, box['joint_name'], tick, OPEN_QPOS[mechanism], closed_threshold(mechanism),
                                    max_ticks=WINDOW_MAX_TICKS)
        row.update(shut_ticks=shut[0], damping=params['damping'], stiffness=params['stiffness'],
                   frictionloss=params['frictionloss'])
        out[mechanism] = row
    return out


def closer_window_ticks(mechanism):
    """The window the built box model gives ``mechanism`` now (cached measurement); raises when it disagrees
    with :data:`CLOSER_WINDOW_TICKS` by more than :data:`CLOSER_WINDOW_TOLERANCE`, so a tuning change that
    was not carried into the table fails the build instead of minting an unusable closer."""
    measured = closer_windows()[mechanism]['ticks']
    expected = CLOSER_WINDOW_TICKS[mechanism]
    if measured is None or abs(measured - expected) > CLOSER_WINDOW_TOLERANCE * expected:
        raise ValueError(f'{mechanism}: the built box model leaves the hand {measured} ticks after the release, '
                         f'the table says {expected}; re-measure CLOSER_WINDOW_TICKS with closer_windows()')
    return int(measured)
