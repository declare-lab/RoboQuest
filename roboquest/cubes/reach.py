"""Mint-time reach rules for painted_cubes (job CUBES-MINT-REACH).

Reserving room for a cube is not enough: the robot also has to *pick it up* and drop it in a bin, and
:func:`observe.reach_gate` gates that with the top-down envelope
``REACH_MIN_AHEAD_M <= ahead <= GRASP_REACH_M`` and ``|lateral| <= LATERAL_REACH_M`` after the base has
slid along the counter front. Until these rules the draw knew nothing about the envelope, so the scatter's
back row was not clipped to the achievable reach and a bin could stand beside a counter the base cannot
slide along. Measured by CUBES-ORACLE-2 on the 54 evaluation instances (one headless build each,
2026-09-22, reach table ``tmp/cubes_oracle2/reach_table.md``): 20 of the 54 were unreachable by
construction, 16 with a cube 5-99 mm beyond ``GRASP_REACH_M`` ahead and 8 on layout 10, whose floor allows
only +0.11 m of slide against the 0.18 m a bin at frame x +-0.43 needs (four ids are in both families).

A third rule came from the same job's carry probe: over the counter a wall cabinet, shelf or hood hangs
0.46-0.47 m above the top on layouts 7, 8 and 56, and the hand crossing at 0.48 m with a cube stops
against it - on layout 7 the arm ended 0.26 m short of the mouth (``tmp/cubes_oracle2/bin_reach.json``,
``cab_main_main_group_door_g1`` against ``robot0_link7``), while the bins whose corridor is clear were
reached. CUBES-ORACLE-2's per-geom headroom table agrees exactly (``tmp/cubes_oracle2/headroom.jsonl``:
38 of 108 bins under a fixture, all on layouts 7, 8 and 56, 0.026-0.150 m over the mouth plane against
0.161 m needed). Neither a stance nor a carry height fixes it, because the mouth itself stands under the
fixture, and moving the bins forward does not either: sweeping the bin y from -0.10 to +0.20 in a build
per kitchen (``tmp/cubes_mint_reach/biny.jsonl``, 30 evaluation and 42 development kitchens) the
clearance never changes, because the unit overhangs the whole 0.64 m depth of the frame.

What does fix it is *where on the counter the frame stands*. Scoring every surface and every sideways
offset the frame search would try (``tmp/cubes_mint_reach/frames.jsonl``) shows layouts 7 and 8 and
development layout 50 have places with nothing overhead - 4, 11 and 11 of them - that the search misses
because it only asks for the largest decor-free region and never looks up, while layout 56 has none at
all, on any of its five evaluation styles. So the corridor is a *frame-placement* rule at build time
(:func:`corridor_blockers`, used by ``PaintedCubes._find_frame_offset``, which refuses an offset whose
corridors are blocked exactly as it refuses one that stands in the decor) and a kitchen rule at mint time
(:func:`corridor_problems`, for the kitchens where no place can ever be clear).

Three facts the build knows and the spec does not are measured here rather than guessed:

* **the base depth.** The base stands at task-frame (0, ``-depth``) facing +y with its front at the work
  surface edge on every instance, so it can never advance and ``ahead`` is one subtraction. The measured
  depth is 0.475 m everywhere except layout 10, where it is 0.525 m. An unmeasured layout takes the deeper
  of the two, :data:`DEFAULT_BASE_DEPTH_M`, so a guess is never optimistic.
* **the lateral slide the floor allows**, the contiguous run of ``observe.stance_gate``-passing stances
  through the reset one. It varies with the layout, the style and (layouts 7 and 56) with what else stands
  on the floor, so :data:`SLIDE_RUN_M` keeps the worst run measured for each (layout, style) pair and
  :data:`LAYOUT_SLIDE_RUN_M` the worst over a layout's measured styles, which is what an unmeasured style
  of a measured layout gets. A layout nothing was measured on is not refused here: the build gate is still
  the authority, and refusing every unmeasured kitchen would empty the development pool.
* **the air over the bin corridors**, the lowest fixture underside over the strip from the front of the
  work surface to each bin mouth, measured with ``headroom.overhead_boxes`` over *every* place the frame
  search could put the frame. A kitchen is refused only when no place clears the carry height, which of
  the measured kitchens is layout 56 alone; :data:`BIN_CORRIDOR_M` keeps its best measurement and
  :data:`MEASURED_CLEAR_LAYOUTS` the layouts measured to have a clear place. A layout nothing was
  measured on is not refused: the build's own frame search is the authority there.

Both rules are candidate refusals, not spec validity: ``sample_spec`` raises and ``mint_grid`` walks on to
the next seed, exactly like the mint-time cover walk. The instances minted before the rules existed stay
valid at their own generator version.
"""
from roboquest import observe as OB

CUBE_REACH_MARGIN_M = .02      # slack on the grasp envelope, as the cover band keeps .03 for a knob
# Measured base depth (metres in front of the frame origin) per layout; the deeper default for the rest.
DEFAULT_BASE_DEPTH_M = .525
BASE_DEPTH_M = {7: .475, 8: .475, 10: .525, 51: .475, 53: .475, 56: .475}
# Worst (tightest) measured slide run per (layout, style): (lo, hi) in metres along the counter front.
SLIDE_RUN_M = {
    (7, 11): (-.58, 1.20), (7, 12): (-.50, 1.20), (7, 13): (-1.20, .42), (7, 15): (-.51, 1.20),
    (8, 11): (-1.20, 1.18), (8, 12): (-1.20, 1.18), (8, 13): (-1.20, 1.18), (8, 14): (-1.20, 1.19),
    (8, 15): (-1.20, .98),
    (10, 11): (-1.20, .11), (10, 13): (-1.20, .11), (10, 14): (-1.20, .11),
    (51, 11): (-1.20, 1.20), (51, 12): (-1.20, 1.20), (51, 13): (-1.20, 1.20), (51, 14): (-1.20, 1.20),
    (53, 11): (-.60, 1.20), (53, 12): (-.60, 1.20), (53, 13): (-.60, 1.20), (53, 14): (-.60, 1.20),
    (53, 15): (-.60, 1.20),
    (56, 12): (-.77, 1.12), (56, 13): (-.46, 1.20), (56, 14): (-.36, 1.20), (56, 15): (-.77, 1.12),
}
LAYOUT_SLIDE_RUN_M = {layout: (max(lo for (l, _), (lo, _hi) in SLIDE_RUN_M.items() if l == layout),
                               min(hi for (l, _), (_lo, hi) in SLIDE_RUN_M.items() if l == layout))
                      for layout, _style in SLIDE_RUN_M}
# Rule (c), the carry corridor. Air the hand needs over the work top to carry a cube to a bin mouth:
# the mouth stands .32 m over the top, the transit adds MOUTH_TRANSIT_M + CUBE_HALF_M (.06) and the hand
# rides .10 m over the cube, so the gripper crosses at .48 m and the wrist another .05 m over that.
BIN_CARRY_NEED_M = .53
BIN_CARRY_RADIUS_M = .10       # the cube in the hand plus the fingers; the headroom gate pads it by HAND_PAD_M
CLEAR_M = float('inf')         # nothing hangs over that corridor at all
# Kitchens where *no* place on any work surface carries a cube to both mouths: the value is the best any
# candidate place reaches (frames.jsonl, all 31 places of each style). Layout 56 is the only one measured.
BIN_CORRIDOR_M = {(56, 11): .46, (56, 12): .46, (56, 13): .46, (56, 14): .46, (56, 15): .46}
# Layouts measured to have at least one clear place, so the frame search can find it: the evaluation pool
# less 56 (frames.jsonl for 7 and 8, biny.jsonl for the rest) and every development layout (biny.jsonl,
# styles 3 and 8; layout 50 needs the search, its default place is blocked and 11 others are not).
MEASURED_CLEAR_LAYOUTS = (7, 8, 10, 12, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 29, 31, 33, 38, 39, 43,
                          45, 48, 49, 50, 51, 53)
LAYOUT_BIN_CORRIDOR_M = {layout: min(v for (l, _), v in BIN_CORRIDOR_M.items() if l == layout)
                         for layout, _style in BIN_CORRIDOR_M}


def base_depth_m(layout_id):
    """How far in front of the frame origin the base stands on this layout (measured, else the deeper one)."""
    return float(BASE_DEPTH_M.get(int(layout_id), DEFAULT_BASE_DEPTH_M))


def slide_run_m(layout_id, style_id):
    """The lateral slide run the floor allows, or ``None`` when nothing was measured on this layout."""
    run = SLIDE_RUN_M.get((int(layout_id), int(style_id))) or LAYOUT_SLIDE_RUN_M.get(int(layout_id))
    return (float(run[0]), float(run[1])) if run else None


def bin_corridor_m(layout_id, style_id):
    """Clear air over the best bin corridor this kitchen offers, or ``None`` when nothing was measured."""
    value = BIN_CORRIDOR_M.get((int(layout_id), int(style_id)))
    if value is None:                       # an unmeasured style keeps its layout's worst, as rule (b) does
        value = LAYOUT_BIN_CORRIDOR_M.get(int(layout_id))
    if value is None and int(layout_id) in MEASURED_CLEAR_LAYOUTS:
        value = CLEAR_M
    return None if value is None else float(value)


def corridor_problems(layout_id, style_id, need=BIN_CARRY_NEED_M):
    """Rule (c): somewhere in this kitchen the hand can carry a cube over the counter to each bin mouth.

    A wall cabinet, shelf or hood hanging over the counter stops the wrist well short of the mouth however
    the base stands (measured: the arm stops 0.26 m short under a 0.47 m corridor), and neither a carry
    height nor a bin moved forward helps, since the unit overhangs the whole frame. The build's frame
    search (:func:`corridor_blockers`) moves the frame to a place with clear air; this refusal is for the
    kitchens where it has nowhere to move, so the seed walk skips them without paying for a build.
    Nothing is refused on a layout nothing was measured on, as in :func:`slide_problems`.
    """
    available = bin_corridor_m(layout_id, style_id)
    if available is None or available >= float(need):
        return []
    return [f'no place on layout {int(layout_id)}/{int(style_id)} carries a cube to the bins: '
            f'{available:.2f} m of air over the best carry corridor, {float(need):.2f} m needed '
            f'(the hand crosses at {BIN_CARRY_NEED_M - .05:.2f} m with the cube)']


def corridor_blockers(boxes, to_world, top_z, bin_spots, front_y,
                      need=BIN_CARRY_NEED_M, radius=BIN_CARRY_RADIUS_M):
    """Build-time rule (c): the overhead fixtures hanging into a bin's carry corridor at this frame place.

    ``boxes`` are ``headroom.overhead_boxes`` for the surface, ``to_world`` maps a task-frame xy to world
    xy for the *candidate* placement (not necessarily the one the frame ends up at), ``bin_spots`` are the
    bin mouths in the task frame and ``front_y`` the frame's front edge. The corridor is the strip the hand
    carries the cube along - the front of the surface straight back to the mouth - plus the column over the
    mouth itself, both taken as the hand's disc: the same volume :func:`headroom.headroom_gate` uses for a
    cover. Returns the blocking fixture names (empty when every corridor is clear), so the caller can put
    them in a refusal message.
    """
    from roboquest import headroom as HR
    names = []
    for spot in bin_spots:
        mouth = to_world((float(spot[0]), float(spot[1])))
        front = to_world((float(spot[0]), float(front_y)))
        for gate in (HR.path_headroom_gate(boxes, HR.path_points(front, mouth), radius, 0., top_z),
                     HR.headroom_gate(boxes, mouth, radius, 0., top_z)):
            available = gate['available_m']
            if available is not None and available < float(need) and gate['fixture'] not in names:
                names.append(gate['fixture'])
    return names


def ahead_m(xy, layout_id):
    """How far ahead of the base a task-frame point lies (``observe.reach_gate``'s ``ahead``)."""
    return float(xy[1]) + base_depth_m(layout_id)


def max_y_m(layout_id):
    """The deepest task-frame y a pickable prop may stand at on this layout."""
    return OB.GRASP_REACH_M - CUBE_REACH_MARGIN_M - base_depth_m(layout_id)


def reach_zone(zone, layout_id):
    """The part of the cube zone the robot can pick from on this layout: the zone clipped to :func:`max_y_m`.

    Clipping the draw is what keeps rule (a) from throwing seeds away - the back row is scattered where the
    arm can reach instead of being minted and refused - and :func:`reach_problems` stays the fail-closed
    check on the result.
    """
    return OB.S.Rect(zone.x0, zone.x1, zone.y0, min(float(zone.y1), max_y_m(layout_id)))


def ahead_problems(props, layout_id, margin=CUBE_REACH_MARGIN_M):
    """Rule (a): every pickable prop stands inside the grasp envelope ahead of the base, with a margin."""
    limit, problems = OB.GRASP_REACH_M - float(margin), []
    for name, xy in props:
        ahead = ahead_m(xy, layout_id)
        if ahead > limit:
            problems.append(f'{name}: {ahead:.3f} m ahead of the base on layout {int(layout_id)} is beyond '
                            f'{limit:.3f} m (grasp reach {OB.GRASP_REACH_M:.2f} m less a {margin * 1e3:.0f} mm margin)')
        elif ahead < OB.REACH_MIN_AHEAD_M:
            problems.append(f'{name}: {ahead:.3f} m ahead of the base is inside the '
                            f'{OB.REACH_MIN_AHEAD_M:.2f} m minimum')
    return problems


def slide_problems(props, layout_id, style_id):
    """Rule (b): a stance the floor allows brings every prop within ``LATERAL_REACH_M`` of the base.

    The base slides along the counter front, so a prop at task-frame x needs a stance in
    ``[x - LATERAL_REACH_M, x + LATERAL_REACH_M]`` inside the measured run. Nothing is refused on a layout
    no run was measured on.
    """
    run = slide_run_m(layout_id, style_id)
    if run is None:
        return []
    lo, hi = run
    problems = []
    for name, xy in props:
        want_lo, want_hi = float(xy[0]) - OB.LATERAL_REACH_M, float(xy[0]) + OB.LATERAL_REACH_M
        if max(lo, want_lo) > min(hi, want_hi):
            residual = min(abs(want_lo - hi), abs(lo - want_hi))
            problems.append(f'{name}: at frame x {float(xy[0]):+.3f} it needs the base between '
                            f'{want_lo:+.2f} and {want_hi:+.2f} m along the counter, outside the '
                            f'{lo:+.2f}..{hi:+.2f} m the floor allows on layout {int(layout_id)}/'
                            f'{int(style_id)} ({residual:.2f} m short)')
    return problems


def reach_problems(props, layout_id, style_id, margin=CUBE_REACH_MARGIN_M):
    """Both rules over ``(name, xy)`` pairs in the task frame: the cubes and the two bins."""
    props = [(str(name), (float(xy[0]), float(xy[1]))) for name, xy in props]
    return ahead_problems(props, layout_id, margin) + slide_problems(props, layout_id, style_id)
