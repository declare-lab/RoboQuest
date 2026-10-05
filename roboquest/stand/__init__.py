"""Wobbly stand prop for the RoboQuest ``wobbly_stand`` task.

``geometry`` builds the stand, the shim plates and the ball as MJCF elements
(dressed with RoboCasa wood textures) and can export a bare MuJoCo model for
contact tuning; ``statics`` is the rigid-body resting-pose model (which legs
support the stand, how far the top tilts, which foot hangs) that the CPU gate
and the scene placement share. Neither module imports RoboCasa.
"""
