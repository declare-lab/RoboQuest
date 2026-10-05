"""Marked-vessel props: RoboCasa mug or bowl clones, coloured balls, underside labels and pads.

Adapted from the 0.80 m study-room task (``active_bench.inspect_cups_*``) for
RoboCasa work surfaces: everything is expressed in the RoboQuest task frame
(x along the front of the surface, y away from the robot, z up from the top),
so the same spec builds on any layout and style. The mug asset, its sixteen
hollow collision hulls, the tuned ball/mug/pad contact parameters and the
placement rule are unchanged from the study room; only the frame, the palette
sizes and the pad geometry were re-derived for the 0.92 m surface.

v1 (spec 4.6) adds the **bowl**: RoboCasa's ``bowl_11`` visual mesh with a
procedural hollow collision shell in place of its convex hulls, so two balls
settle inside it and the balls-inside rule is a single truncated-cone test that
covers both vessel types. The package keeps its ``mugs`` name because the task
is still ``marked_mugs``.
"""
