"""Blackout Search parts: the lantern, the room light (off), the structure rules.

Kept out of ``tasks/blackout_search.py`` so the pieces that have no simulator in them (``structure``,
``darkness``) are testable on the CPU, and so the lantern reads like the rest of the procedural assets.
The wall switch and the lighting timer of the 2026-09-22 contract are gone (torch-only).
"""
from roboquest.blackout.darkness import room_light_ids, set_room_light
from roboquest.blackout.lamp import build_lamp, aim_yaw, beam_direction, glow_material
from roboquest.blackout.structure import COMPARTMENT_COUNTS, FACTORS, GOAL_TEMPLATE, HIDING_FLOOR_MIN_M, HIDING_KINDS, HOLDOUT_KEYS, SET_SIZES, admissible_hiding_place, capacity_problems, draw_structure, progress_weights

__all__ = ['build_lamp', 'aim_yaw', 'beam_direction', 'glow_material', 'room_light_ids', 'set_room_light',
           'COMPARTMENT_COUNTS', 'FACTORS', 'GOAL_TEMPLATE', 'HIDING_FLOOR_MIN_M', 'HIDING_KINDS', 'HOLDOUT_KEYS',
           'SET_SIZES', 'admissible_hiding_place', 'capacity_problems', 'draw_structure', 'progress_weights']
