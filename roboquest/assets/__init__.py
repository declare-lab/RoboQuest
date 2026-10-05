"""Procedural props for RoboQuest v1 (spec 4.10, build brief contract C4): dome covers with a
knob, a wide cup with a handle, the odd-parcel answer box, and RoboCasa-texture materials.

Every builder appends ``xml.etree`` elements under ``parent`` (a body element), names every geom
``{name}_*``, builds hollow shells from thin boxes, keeps visuals opaque and marks the grasp
feature with a site. With ``free=True`` the body gets a free joint, so ``parent`` must then be
the worldbody (MuJoCo allows free joints on top-level bodies only): pass the world pose through
``pos`` and ``yaw`` (``env.frame_to_world(...)`` and ``env.work['yaw']``). Fixed props (the answer
box, or ``free=False``) can hang under a frame body from ``add_frame_body``.
"""
from roboquest.assets import materials
from roboquest.assets.box import answer_box
from roboquest.assets.cloche import cloche
from roboquest.assets.cup import wide_cup

__all__ = ['answer_box', 'cloche', 'materials', 'wide_cup']
