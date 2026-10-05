"""Panel-box geometry for the ``unfamiliar_containers`` task.

Identical wooden cases carrying knobbed overlay panels on two of their five
faces. One panel opens, in one of eleven ways (seven on an upright face, four
on the top); the other is a fake — the same plate and the same knob, screwed to
a solid wall. From outside the two are the same panel.

The modules are copies of the "panel boxes" version in the ``open-containers``
worktree (that worktree is read-only source and is never modified):

* :mod:`contract` — mechanism and face vocabulary (no simulator import);
* :mod:`mechanism` — primitive geoms, materials, RoboCasa asset import and the
  one declared rule (a touch latch releases at 6 N of inward press);
* :mod:`panels` — one box, built in its own frame at any base pose and yaw;
* :mod:`rules` — the bowl containment test used for scoring.
"""
