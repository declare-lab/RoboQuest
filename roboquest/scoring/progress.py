"""Progress, the benchmark's progress metric (docs/progress.md), scored from a finished episode directory.

Reads result.json (the score frozen at the first Submit press, or the end state), task-spec-private.json and 1 Hz
lines of the episode's physical-trace-private.jsonl (every 20th 20 Hz line plus the last). One function per task
returns dict(P, stages, deduction, flags, ceiling, goal_ever, ...); ``write`` stores the result as
task-progress.json beside result.json.

Thresholds:
* puzzle: lid stage = lid joint max over the trace >= 0.5 x open_at; item out = item centre displaced
  > 0.30 m from its source at any sampled step, or item_in_tray on any sampled line.
* stand: q from the final tilt and the snapshot's initial_tilt_deg; q = 0 unless score released and settled
  and the stand itself has no robot contact / grasp at the end; q = 1 if score['level']; q < 0.01 -> 0
  (the task's own progress deadband).
* odd parcel: B = parcels in the box (score box_contents) that are released; the 0.1 deduction when the balance
  base stands more than 2 cm from its reset pose. Parcels left on the pans are a flag.
* marked mugs: per vessel the frozen score's placement_credit (1 / 0.5 / 0 from the footprint's area fraction on
  its own pad, see mugs/rules.py) times 1 with its own two balls inside, else 0.5.
* search family: opened = the target's compartment reached 15 % of its joint range on a sampled line
  (env OPEN_FRACTION); out = the env's out-of-hiding rule on a sampled line; cloche: out counts 2/3, no 1/3
  stage; open counter: no 1/3 stage.
  Placed = per_target in_tray, supported, upright at the press and the object neither grasped nor touched.
  Deduction "non-target object on the tray" = any '<name>:unwanted_tray_overlap' error in the frozen score
  (the success predicate's own test; it covers covers/cloches, tokens and the lamp, not only tray_contents).
* containers: opened = boxes_private[holder]['opened_once']; out = the env's item_out_of_box rule
  (unfamiliar_containers.py: centre outside the holder's interior box + 0.03 m in the box frame) on any sampled
  line, recomputed from spec geometry; in bowl = score.
* locked storage: effective links = chain links with locks['locked'][cid] false in the final snapshot, or the
  whole chain once the target has been out.
* stamps: per-cell grades of the printed dots (roboquest/scoring/stamps_grading.py) on the final board's ink events of the frozen
  line; strays = on-paper dots outside target cells + 1 if off-grid ink > 35 px.
"""
import json
import math
import os
import sys

from roboquest.scoring import stamps_grading as DR  # noqa: E402

DEDUCTION = .1
TRACE = 'physical-trace-private.jsonl'
SAMPLE_EVERY = 20          # 20 Hz trace -> 1 Hz lines


# ---------------------------------------------------------------- 1 Hz trace lines
def _rnd(v):
    if isinstance(v, float):
        return round(v, 4)
    if isinstance(v, list):
        return [_rnd(x) for x in v]
    if isinstance(v, dict):
        return {k: _rnd(x) for k, x in v.items()}
    return v


def slim_record(d):
    """Score and task snapshot kept, per object only pose, velocity, grasp and contact flags and the ids of its
    supports; floats rounded to 4 decimals."""
    snap = d.get('snapshot') or {}
    objs = snap.get('objects')
    if isinstance(objs, dict):
        out = {}
        for name, o in objs.items():
            if not isinstance(o, dict):
                out[name] = o
                continue
            keep = {k: o.get(k) for k in ('position', 'quaternion_xyzw', 'linear_velocity', 'angular_velocity', 'linear_speed',
                                          'angular_speed', 'grasped', 'robot_contact', 'fixed', 'supported_pads', 'rotation') if k in o}
            sc = o.get('support_contacts')
            if isinstance(sc, list):
                keep['supports'] = sorted({str(c.get('support_id')) for c in sc if isinstance(c, dict)})
            out[name] = keep
        snap = dict(snap, objects=out)
    return _rnd(dict(step=d.get('step'), sim_time=d.get('sim_time'), score=d.get('score'), snapshot=snap))


_LINES = {}


def _one_hz(ep):
    """Every SAMPLE_EVERY-th trace line, plus the last line (marked final) when it was not sampled itself."""
    out, last = [], None
    with open(os.path.join(ep, TRACE), 'rb') as f:
        for i, line in enumerate(f):
            if i % SAMPLE_EVERY == 0:
                out.append(json.loads(json.dumps(slim_record(json.loads(line)))))
                last = None
            else:
                last = line
    if last is not None:
        try:
            out.append(json.loads(json.dumps(dict(slim_record(json.loads(last)), final=True))))
        except ValueError as e:
            out.append(dict(final=True, error=str(e)[:100]))
    return out


def lines(ep):
    key = os.path.abspath(ep)
    if key not in _LINES:
        _LINES.clear()
        _LINES[key] = _one_hz(ep)
    yield from _LINES[key]


def load(ep, name):
    with open(os.path.join(ep, name)) as f:
        return json.load(f)


def frozen(ep):
    return load(ep, 'result.json')['score']


def _goal(x):
    """The live predicate on a trace line; the final line carries the frozen score (no current_goal_satisfied
    key), where success stands for it."""
    s = x.get('score') or {}
    return bool(s.get('current_goal_satisfied') or (x.get('final') and s.get('success')))


def clip01(x):
    return max(0., min(1., x))


# ---------------------------------------------------------------- stamps
def _dot(spec, events, target, prescribed):
    imps = [(e['stamp'], [float(v) for v in (e.get('translation_xy_m') or [e.get('x_m'), e.get('y_m')])],
             float(e.get('yaw_rad', 0.)), e.get('step')) for e in events if e.get('board') == 'final']
    centres = DR.dot_centres(spec, imps)
    _, c = DR.grade(centres, target, False, prescribed)
    return c


def _stamps_P(c, n_target, off_grid):
    strays = c['stray'] + (1 if off_grid > DR.MAX_OFF_GRID_PIXELS else 0)
    clean = c['perfect'] + c['design_multiple']
    imperfect = c['partial'] + c['overpainted']
    den = n_target + strays
    P = (clean + .5 * imperfect) / den if den else 0.
    ceiling = (clean + c['missing'] + .5 * imperfect) / den if den else 0.
    return P, ceiling, strays


def stamps(ep):
    spec = load(ep, 'task-spec-private.json')
    sc = frozen(ep)
    target = sc.get('target_cells') or spec.get('target')
    n_target = sum(bool(v) for row in target for v in row)
    prescribed = DR.prescribed_counts(spec)
    goal_ever = goal_ever_coarse = pattern_ever = False
    last = None
    for x in lines(ep):
        last = x
        s = x.get('score') or {}
        goal_ever_coarse |= _goal(x)
        ev = (x.get('snapshot') or {}).get('ink_events') or []
        if not any(e.get('board') == 'final' for e in ev):
            continue
        c = _dot(spec, ev, target, prescribed)
        P_l, _, _ = _stamps_P(c, n_target, int(s.get('off_grid_ink_pixels') or 0))
        if P_l >= 1. - 1e-9:
            pattern_ever = True
            st = s.get('stamps') or {}
            if st and all(v.get('released') and v.get('stable') and v.get('supported') for v in st.values()):
                goal_ever = True
    ev = (last.get('snapshot') or {}).get('ink_events') or []
    c = _dot(spec, ev, target, prescribed)
    off_grid = int(sc.get('off_grid_ink_pixels') or 0)
    P, ceiling, strays = _stamps_P(c, n_target, off_grid)
    st = sc.get('stamps') or {}
    put_down = bool(st) and all(v.get('released') and v.get('stable') and v.get('supported') for v in st.values())
    return dict(P=clip01(P), ceiling=clip01(ceiling), deduction=False,
                stages=dict(n_target=n_target, strays=strays, off_grid_px=off_grid, **c),
                flags=dict(stamps_put_down=put_down, off_grid_over_35=off_grid > DR.MAX_OFF_GRID_PIXELS),
                goal_ever=goal_ever, goal_ever_dot=goal_ever,
                pattern_ever_dot=pattern_ever, goal_ever_coarse=goal_ever_coarse)


# ---------------------------------------------------------------- wobbly stand
STAND_DEADBAND = .01   # PROGRESS_DEADBAND of wobbly_stand.py


def wobbly_stand(ep):
    sc = frozen(ep)
    goal_ever = False
    init = None
    last = None
    ball_off_ever = False
    levelled_passive_ever = False
    for x in lines(ep):
        last = x
        s = x.get('score') or {}
        goal_ever |= _goal(x)
        sn = x.get('snapshot') or {}
        if init is None and sn.get('initial_tilt_deg') is not None:
            init = float(sn['initial_tilt_deg'])
        if ((sn.get('collateral') or {}).get('ball_off_stand') or {}).get('bad'):
            ball_off_ever = True
        if s.get('level') and s.get('released') and s.get('settled'):
            levelled_passive_ever = True
    stand = (last.get('snapshot') or {}).get('stand') or {}
    stand_touched = bool(stand.get('robot_contact') or stand.get('grasped'))
    passive = bool(sc.get('released') and sc.get('settled') and not stand_touched)
    tilt = float(sc['tilt_deg'])
    if not passive:
        q = 0.
    elif sc.get('level'):
        q = 1.
    else:
        q = clip01(1. - tilt / init) if init and init > 1e-9 else 0.
        if q < STAND_DEADBAND:
            q = 0.
    b = 1. if sc.get('ball_on_top') else 0.
    P = q * (1. + b) / 2.
    return dict(P=clip01(P), ceiling=None, deduction=False,
                stages=dict(q=round(q, 4), b=b, tilt_deg=tilt, initial_tilt_deg=init, level=bool(sc.get('level')),
                            released=bool(sc.get('released')), settled=bool(sc.get('settled')),
                            stand_touched_end=stand_touched),
                flags=dict(ball_off_ever=ball_off_ever, ball_on_top_end=bool(sc.get('ball_on_top')),
                           levelled_passive_ever=levelled_passive_ever,
                           ball_off_after_level=bool(levelled_passive_ever and not sc.get('ball_on_top'))),
                goal_ever=goal_ever)


# ---------------------------------------------------------------- odd parcel
BALANCE_TOL_M = .02


def odd_parcel(ep):
    spec = load(ep, 'task-spec-private.json')
    sc = frozen(ep)
    goal_ever = any(_goal(x) for x in lines(ep))
    odd = set(spec['odd_parcels_private'])
    unreleased = {e.split(':')[0] for e in sc.get('errors', []) if e.endswith(':not_released')}
    B = set(sc.get('box_contents') or []) - unreleased
    J = len(B & odd) / len(B | odd) if (B | odd) else 0.
    pans = sc.get('parcels_on_pans') or {}
    on_pans = sorted(n for side in pans.values() for n in side)
    moved = float(sc.get('balance_displacement_m') or 0.) > BALANCE_TOL_M
    ded = moved
    P = max(0., J - DEDUCTION * ded)
    return dict(P=clip01(P), ceiling=None, deduction=ded,
                stages=dict(jaccard=round(J, 4), boxed_released=sorted(B), odd=sorted(odd),
                            boxed_unreleased=sorted(set(sc.get('box_contents') or []) & unreleased)),
                flags=dict(parcels_on_pans=on_pans, balance_moved=moved,
                           balance_displacement_m=sc.get('balance_displacement_m')),
                goal_ever=goal_ever)


# ---------------------------------------------------------------- puzzle box
ITEM_OUT_M = .30


def puzzle_box(ep):
    spec = load(ep, 'task-spec-private.json')
    sc = frozen(ep)
    parts = spec['parts']
    bolts = sorted([k for k, v in parts.items() if v.get('kind') == 'bolt'], key=lambda k: int(k.split('_')[1]))
    open_at = float(parts['lid']['open_at'])
    src = (spec.get('item') or {}).get('source_position')
    lid_max = 0.
    disp_max = 0.
    in_tray_ever = False
    goal_ever = False
    relocks = 0
    prev = None
    for x in lines(ep):
        s = x.get('score') or {}
        p = s.get('parts') or {}
        goal_ever |= _goal(x)
        in_tray_ever |= bool(s.get('item_in_tray'))
        lj = (p.get('lid') or {}).get('joint_m')
        if lj is not None:
            lid_max = max(lid_max, float(lj))
        ip = ((x.get('snapshot') or {}).get('item') or {}).get('position')
        if ip and src:
            disp_max = max(disp_max, math.dist(ip, src))
        rel = {b: (p.get(b) or {}).get('released') is True for b in bolts}
        if prev is not None:
            relocks += sum(1 for b in bolts if prev[b] and not rel[b])
        prev = rel
    pe = sc.get('parts') or {}
    released_end = [b for b in bolts if (pe.get(b) or {}).get('released') is True]
    placed = bool(sc.get('item_in_tray') and sc.get('item_supported_by_tray') and sc.get('item_released'))
    out_ever = disp_max > ITEM_OUT_M or in_tray_ever or placed
    lid_half = lid_max >= .5 * open_at
    stage = 3 if placed else 2 if out_ever else 1 if lid_half else 0
    nb = len(bolts) if stage >= 2 else len(released_end)
    P = (nb + stage / 3.) / (len(bolts) + 1)
    contact = sorted(k for k, v in pe.items() if v.get('robot_contact'))
    return dict(P=clip01(P), ceiling=None, deduction=False,
                stages=dict(chain_length=len(bolts), bolts_released_end=len(released_end), bolts_counted=nb,
                            item_stage=stage, lid_joint_max=round(lid_max, 4), lid_open_at=open_at,
                            item_max_displacement_m=round(disp_max, 4), item_in_tray_ever=in_tray_ever),
                flags=dict(robot_contact_at_press=contact, relocks_1hz=relocks),
                goal_ever=goal_ever)


# ---------------------------------------------------------------- marked mugs
def marked_mugs(ep):
    sc = frozen(ep)
    goal_ever = any(_goal(x) for x in lines(ep))
    vals, per = [], {}
    for name, v in sorted((sc.get('vessels') or {}).items()):
        g = float(v['placement_credit']) * (1. if v.get('original_pair_retained') else .5)
        vals.append(g)
        per[name] = g
    P = sum(vals) / len(vals) if vals else 0.
    return dict(P=clip01(P), ceiling=None, deduction=False, stages=dict(per_vessel=per),
                flags=dict(wrong_pad=sum(1 for v in (sc.get('vessels') or {}).values() if v.get('wrong_pad')),
                           all_objects_released=bool(sc.get('all_objects_released'))),
                goal_ever=goal_ever)


# ---------------------------------------------------------------- painted cubes
def painted_cubes(ep):
    sc = frozen(ep)
    goal_ever = any(_goal(x) for x in lines(ep))
    cubes = sc.get('cubes') or {}
    n = len(cubes)
    right = sum(1 for c in cubes.values() if c.get('captured_by') and c['captured_by'] == c.get('correct_bin'))
    wrong = sum(1 for c in cubes.values() if c.get('captured_by') and c['captured_by'] != c.get('correct_bin'))
    cap = sum(1 for c in cubes.values() if c.get('captured_by'))
    return dict(P=right / n if n else 0., ceiling=1. - wrong / n if n else 0., deduction=False,
                stages=dict(n=n, right=right, wrong=wrong),
                flags=dict(coverage=cap / n if n else 0.), goal_ever=goal_ever)


# ---------------------------------------------------------------- unfamiliar containers
OUT_OF_BOX_MARGIN = .03


def _item_out(box, p):
    c = box['interior']['center']
    h = box['interior']['half']
    yaw = -float(box['frame_yaw'])
    cs, sn = math.cos(yaw), math.sin(yaw)
    dx, dy, dz = p[0] - c[0], p[1] - c[1], p[2] - c[2]
    local = (cs * dx - sn * dy, sn * dx + cs * dy, dz)
    return any(abs(local[i]) > h[i] + OUT_OF_BOX_MARGIN for i in range(3))


def unfamiliar_containers(ep):
    spec = load(ep, 'task-spec-private.json')
    sc = frozen(ep)
    items = spec['items']
    names = [it['name'] for it in items]
    out_ever = {n: False for n in names}
    goal_ever = False
    live_err = 0.
    for x in lines(ep):
        s = x.get('score') or {}
        goal_ever |= _goal(x)
        pos = {i['name']: i['position'] for i in ((x.get('snapshot') or {}).get('items') or [])}
        inb = {i['name']: i.get('in_bowl') for i in (s.get('items') or [])}
        vals = []
        for it in items:
            p = pos.get(it['name'])
            o = bool(p) and _item_out(spec['boxes'][it['holder']], p)
            out_ever[it['name']] |= o
            vals.append(1. if inb.get(it['name']) else .5 if o else 0.)
        if s.get('progress') is not None and vals and not x.get('final'):
            live_err = max(live_err, abs(sum(vals) / len(vals) - float(s['progress'])))
    fin = {i['name']: i for i in (sc.get('items') or [])}
    boxes = sc.get('boxes_private') or {}
    per, vals, detail = {}, [], {}
    for it in items:
        n = it['name']
        if fin.get(n, {}).get('in_bowl'):
            st = 3
        elif out_ever[n]:
            st = 2
        elif (boxes.get(it['holder']) or {}).get('opened_once'):
            st = 1
        else:
            st = 0
        per[n] = st
        pos_end = fin.get(n, {}).get('position')
        detail[n] = dict(holder=it['holder'], opened_once=bool((boxes.get(it['holder']) or {}).get('opened_once')),
                         out_ever=out_ever[n], out_end=bool(pos_end) and _item_out(spec['boxes'][it['holder']], pos_end),
                         in_bowl=bool(fin.get(n, {}).get('in_bowl')))
        vals.append(st / 3.)
    P = sum(vals) / len(vals) if vals else 0.
    return dict(P=clip01(P), ceiling=None, deduction=False,
                stages=dict(per_item=per, items=detail, live_progress_max_abs_err=round(live_err, 4)),
                flags=dict(items_released=all(fin.get(n, {}).get('released') for n in names),
                           out_then_back_in=[n for n in names if out_ever[n] and not fin.get(n, {}).get('in_bowl')
                                             and not _item_out(spec['boxes'][items[names.index(n)]['holder']],
                                                               fin.get(n, {}).get('position') or [0, 0, 0])]),
                goal_ever=goal_ever)


# ---------------------------------------------------------------- search family
OPEN_FRACTION = .15
FLOOR_Z = .15


def _comp_fraction(comp, joints):
    fr = 0.
    for j in comp['joints']:
        if j in joints:
            lo, hi = comp['joint_ranges'][j]
            ext = max(abs(lo), abs(hi))
            if ext > 0:
                fr = max(fr, min(1., abs(float(joints[j])) / ext))
    return fr


def _to_local(pos, rot, world):
    c, s = math.cos(rot), math.sin(rot)
    dx, dy, dz = world[0] - pos[0], world[1] - pos[1], world[2] - pos[2]
    return (c * dx + s * dy, -s * dx + c * dy, dz)


def _inside_compartment(comp, joints, world, margin=.03):
    local = _to_local(comp['pos'], comp['rot'], world)
    slide = float(joints.get(comp['joints'][0], 0.)) if comp['kind'] == 'drawer' else 0.
    reg = comp['region']
    ox, oy, oz = reg['offset_local']
    sx, sy = reg['size_xy']
    x, y, z = local[0], local[1] - slide, local[2]
    return (abs(x - ox) <= sx / 2 + margin and abs(y - oy) <= sy / 2 + margin
            and oz - margin <= z <= oz + float(reg['height']) + margin)


def _under_cover(position, cover_position, radius, height, margin=.02):
    d = math.hypot(position[0] - cover_position[0], position[1] - cover_position[1])
    z0 = cover_position[2]
    return d <= radius + margin and z0 - margin <= position[2] <= z0 + height + margin


def _out_of_hiding(spec, place, objects, cj):
    pos = objects['obj_' + place['object']]['position']
    if place.get('compartment'):
        cid = place['compartment']
        return not _inside_compartment(spec['compartments'][cid], cj.get(cid, {}), pos)
    if place.get('cover'):
        cov = spec['covers'][place['cover']]
        return not _under_cover(pos, objects[place['cover']]['position'], cov['radius'], cov['height'])
    spot = place.get('spot')
    if spot is None:
        return False
    return math.hypot(pos[0] - spot['xy'][0], pos[1] - spot['xy'][1]) > .15 or pos[2] > float(spot['top_z']) + .10


def _on_floor(o):
    return any(str(s).startswith('floor') for s in o.get('supports', [])) or o['position'][2] < FLOOR_Z


def _search(ep, locked=False):
    spec = load(ep, 'task-spec-private.json')
    sc = frozen(ep)
    comps = spec['compartments']
    targets = [(k, p) for k, p in sorted(spec['places'].items()) if p['role'] == 'target']
    tn = ['obj_' + p['object'] for _, p in targets]
    opened = {n: False for n in tn}
    out = {n: False for n in tn}
    floor_ever = {n: False for n in tn}
    goal_ever = False
    lamp_picked = lamp_floor_ever = False
    last = None
    for x in lines(ep):
        last = x
        s = x.get('score') or {}
        goal_ever |= _goal(x)
        sn = x.get('snapshot') or {}
        cj = {cid: c['joints'] for cid, c in (sn.get('compartments') or {}).items()}
        objs = sn.get('objects') or {}
        for (k, p), n in zip(targets, tn):
            if p.get('compartment') and _comp_fraction(comps[p['compartment']], cj.get(p['compartment'], {})) >= OPEN_FRACTION:
                opened[n] = True
            if n in objs:
                if _out_of_hiding(spec, p, objs, cj):
                    out[n] = True
                if _on_floor(objs[n]):
                    floor_ever[n] = True
        lamp = sn.get('lamp') or {}
        lamp_picked |= bool(lamp.get('picked') or lamp.get('picks'))
        if 'work_lamp' in objs:
            o = objs['work_lamp']
            lamp_floor_ever |= any(str(q).startswith('floor') for q in o.get('supports', [])) or o['position'][2] < .30
    fsn = last.get('snapshot') or {}
    fobj = fsn.get('objects') or {}
    fcj = {cid: c['joints'] for cid, c in (fsn.get('compartments') or {}).items()}
    per = sc.get('per_target') or {}
    stages = {}
    for (k, p), n in zip(targets, tn):
        pt = per.get(n) or {}
        o = fobj.get(n) or {}
        placed = bool(pt.get('in_tray') and pt.get('supported') and pt.get('upright')
                      and not o.get('grasped') and not o.get('robot_contact'))
        if placed:
            st = 3
        elif out[n]:
            st = 2
        elif p.get('compartment') and opened[n]:
            st = 1
        else:
            st = 0
        stages[n] = dict(kind=p['kind'], stage=st, opened=opened[n] if p.get('compartment') else None,
                         out_ever=out[n], placed=placed, on_tray_end=bool(pt.get('in_tray') and pt.get('supported')),
                         upright_end=bool(pt.get('upright')), floor_ever=floor_ever[n],
                         floor_end=bool(o) and _on_floor(o),
                         out_end=bool(o) and _out_of_hiding(spec, p, fobj, fcj))
    overlap = sorted({e.split(':')[0] for e in sc.get('errors', []) if e.endswith(':unwanted_tray_overlap')})
    tray_nontarget = [n for n in (sc.get('tray_contents') or []) if n not in tn]
    ded = bool(overlap)
    flags = dict(tray_overlap_nontarget=overlap, tray_contents_nontarget=tray_nontarget,
                 targets_floor_ever=[n for n in tn if floor_ever[n]],
                 targets_back_in_hiding=[n for n in tn if out[n] and not stages[n]['out_end'] and not stages[n]['placed']])
    res = dict(deduction=ded, flags=flags, goal_ever=goal_ever, ceiling=None)
    if not locked:
        base = sum(v['stage'] for v in stages.values()) / (3. * len(stages)) if stages else 0.
        res.update(P=clip01(base - DEDUCTION * ded), stages=dict(per_target=stages, base=round(base, 4)))
        if spec.get('lamp') is not None or 'work_lamp' in fobj:
            flags.update(lamp_picked=lamp_picked, lamp_floor_ever=lamp_floor_ever)
        return res
    chain = list(spec['chain'])
    t = stages[tn[0]]
    locks = (fsn.get('locks') or {}).get('locked') or {}
    if t['out_ever'] or t['placed']:
        eff = len(chain)
    else:
        eff = sum(1 for c in chain if locks.get(c) is False)
    base = (eff + t['stage'] / 3.) / (len(chain) + 1)
    dead = spec.get('dead_end')
    ul = (fsn.get('locks') or {}).get('unlocked_ever') or []
    flags.update(relocks=sum(((fsn.get('locks') or {}).get('relocks') or {}).values()),
                 dead_end_unlocked=(dead in ul) if dead else None,
                 unlocked_ever_chain=sum(1 for c in chain if c in ul))
    res.update(P=clip01(base - DEDUCTION * ded),
               stages=dict(per_target=stages, chain_depth=len(chain), effective_links=eff, base=round(base, 4)))
    return res


def search_room(ep):
    return _search(ep)


def blackout_search(ep):
    return _search(ep)


def locked_storage(ep):
    return _search(ep, locked=True)


TASKS = dict(stamps=stamps, wobbly_stand=wobbly_stand, odd_parcel=odd_parcel, puzzle_box=puzzle_box,
             marked_mugs=marked_mugs, painted_cubes=painted_cubes, unfamiliar_containers=unfamiliar_containers,
             search_room=search_room, blackout_search=blackout_search, locked_storage=locked_storage)


def task_of(ep):
    """The task name from the episode's config (``roboquest_<task>@<instance>``)."""
    scene = load(ep, 'config.json')['task']
    return scene.split('@', 1)[0][len('roboquest_'):]


def score(ep, task=None):
    return TASKS[task or task_of(ep)](ep)


def write(ep):
    """Score one finished episode and write task-progress.json beside its result.json."""
    result = dict(metric='progress', spec='docs/progress.md', **score(ep))
    with open(os.path.join(ep, 'task-progress.json'), 'w') as f:
        json.dump(result, f, indent=1, default=str)
        f.write('\n')
    return result


if __name__ == '__main__':
    for ep in sys.argv[1:]:
        print(json.dumps(score(ep), default=str))
