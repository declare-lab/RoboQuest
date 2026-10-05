# RoboQuest scoring

Every episode gets two scores: **success** (0 or 1) and **progress** `P` (0 to 1). Both are frozen at the first
press of the physical Submit button, or taken from the final state when the episode ends without one.

- **Success**: the task's goal state holds at a valid Submit press. The task code evaluates it when Submit is
  pressed; it is `score.success` in the episode's `result.json`. An episode that never submits does not succeed.
- **Progress**: a per-task staged measure of how much of the goal state the task's own objects reach, computed
  from the finished episode (`result.json`, `task-spec-private.json` and the simulator trace
  `physical-trace-private.jsonl`) by `roboquest/scoring/progress.py`, which the episode runner runs after every
  episode and stores as `task-progress.json`. Every success has P = 1.

## Common rules

- P describes the state of the task's own objects, staged the way the success predicate is staged. Means
  (opening a lid, lifting the lantern, putting a stamp down) are not credited.
- Every stage counts only in a passive state: released, supported, settled, no robot contact where the
  predicate says so. An object held over its destination is not at its destination.
- No caps. P = 1 means the goal state is fully satisfied at the frozen moment; success additionally needs the
  task's side conditions and a valid physical Submit.
- One tidiness deduction of 0.1, in four tasks: Search Room, Blackout Search and Locked Storage (a non-target
  object on the tray) and Odd Parcel (the balance moved). Every other side condition is a flag, not a term.
  P is clipped to [0, 1].
- A ceiling (the highest P still reachable) is reported for painted cubes and stamps, where mistakes are
  irreversible.
- Three-stage object rule for the fetch tasks (search room, blackout search, locked storage, containers):
  hiding place opened 1/3, object out of its hiding place 2/3, object placed 3/3. "Out" is sticky: once an
  object has been out, it never drops below 2/3 again (a closer shutting it back in, or a fall to the floor,
  is a flag). A cloche target counts 2/3 as soon as the cloche is lifted. An open-counter target has no
  "opened" stage.
- Chain rule for the chain tasks (puzzle box, locked storage): each lock or bolt is one share and the object's
  stages together are one share; once the object is out, the chain shares are held at full.

## Per task

| Task | P | 0.1 deduction | Flags |
|---|---|---|---|
| stamps | sum of target-cell grades / (target cells + stray dots). Cell grade: 1 clean and no more than the design's k dots, 0.5 a dot crossing the cell edge or more than k dots, 0 empty. Stray dots = dots on the paper outside target cells, plus one if off-grid ink exceeds 35 px. Cell grades from the printed dots, as in `roboquest/scoring/stamps_grading.py`. Untouched scene = 0. | none | stamps put down. Ceiling = (clean + empty cells + 0.5 x imperfect cells) / (target cells + strays): strays and imperfect cells are permanent, empty cells can still be filled. |
| wobbly stand | q x (1 + b) / 2. q = fraction of the initial tilt removed while the stand is released, settled and untouched (0 while held or touched; 1 once within the level threshold; below 0.01 counts as 0). b = 1 if the ball is on top (no stillness requirement), else 0. | none | ball fell off / put back. |
| odd parcel | Jaccard(B, O) - 0.1 x [balance moved]; B = parcels released inside the box, O = the odd set. Floor 0. | the balance moved (its base more than 2 cm from its reset pose) | parcels left on the pans. |
| puzzle box | (chain bolts released + item stage / 3) / (chain length + 1). Item stage: lid slid past half its travel 1, item out of the cavity 2, item in the tray (supported, released) 3. Once the item is out, bolts count as all released; while it is inside, a relocked bolt does not count. Decoys never count. | none | robot contact with any part at the press; relocks. |
| marked mugs | mean over vessels of placement credit x ball factor (below). Ball factor 1 with its own two balls inside, else 0.5. | none | wrong-pad count; all objects released. |
| search room | mean over targets of the three-stage rule (compartment opened 1/3 by the environment's 15 % joint rule; out of hiding 2/3; placed on the tray, supported, upright 3/3). | a non-target object on the tray | targets dropped to the floor; closer re-shut a target. |
| painted cubes | cubes captured in their correct bin / cube count. | none | ceiling = 1 - wrong captures / cube count; coverage = captured / cube count. |
| unfamiliar containers | mean over items: box opened once 1/3, item out of its box 2/3, item in the bowl 3/3. | none | items released. |
| locked storage | (effective unlocked chain links + target stage / 3) / (chain depth + 1). Effective = currently passable (or all, once the target is out). Target stage as search room. Dead-end compartment never counts. | a non-target object on the tray | relocks, dead end opened. |
| blackout search | as search room. No lantern term. | a non-target object on the tray | lantern picked up / on the floor. |

## Odd Parcel: success

The goal is "Put every parcel that weighs differently in the box and nothing else, leave the balance where it
stands, then press Submit." Success: Submit pressed, the box holds exactly the odd parcels (every collision
corner of a boxed parcel inside the interior + 5 mm), every parcel released and at rest, and the balance base within
2 cm of its reset pose. A parcel left on a balance pan is reported as a flag (`<parcel>:left_on_pan` in
`score.flags`) and neither fails the episode nor costs progress.

## Marked Mugs: success and placement credit

A vessel's footprint is the disc of radius 3.95 cm around its interior centre. Its placement credit comes from
f, the area fraction of that disc lying on the pad of its label colour (the pad rectangle, half size + 2 mm),
computed exactly:

- f >= 3/4: credit 1; 1/2 <= f < 3/4: credit 0.5; f < 1/2: credit 0;
- credit also needs the vessel upright, released and stable (its own balls at rest too), supported from below by
  its own pad, and not standing on a wrong pad; otherwise it is 0.

Success: Submit pressed, every object released, and every vessel with credit 1 and its own two balls inside.
The rules are `classify_placements` and `placement_credit` in `roboquest/mugs/rules.py`; each vessel's row in
`score.vessels` records its `pad_area_fraction` and `placement_credit`.

## Analysis metrics in `task-progress.json`

Besides `P`, each task's result carries its stages, flags, `deduction`, `ceiling` (painted cubes and stamps) and
`goal_ever` (whether the live goal predicate held on any sampled line of the trace), for analyses such as goal
states reached but not submitted, side conditions failed, and progress lost after a mistake.
