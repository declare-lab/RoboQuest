"""Lock hardware, lock rule, chain structure and placement rules for the ``locked_storage`` task (RoboQuest v1.1)."""
from roboquest.locked.heights import DOOR_TARGET_DEPTH_MAX_M, FLOOR_MIN_M, LAYOUT_PLACES, RULES, allowed_compartment, allowed_place, door_is_hinge, drawer_is_rule1, layout_places, objects_hideable, placement_problems, places_from_rows
from roboquest.locked.objects import COLOUR_NAMES, LOCK_COLOURS, PAD_RADIUS_M, PAD_SIZE_M, PLATE_SIZE_M, TOKEN_MASS_KG, TOKEN_SIZE_M, lock_plate, pad_name, plate_name, reader_pad, token, token_name
from roboquest.locked.rules import LOCK_SLACK_RAD, PAD_HEIGHT_TOL_M, PAD_SPEED_MAX, LockRule, closer_candidates, is_closed, locked_range, token_on_pad
from roboquest.locked.structure import CHAIN_DEPTHS, COMPARTMENT_COUNT, DEAD_END_LEVELS, HOLDOUT_COLOUR, HOLDOUT_KEYS, KIND_PATTERNS, SET_SIZE, all_keys, chain_binding, colour_combinations, dev_composition_counts, is_holdout, key_colours, pad_offsets, pad_spot, split_key

__all__ = [
    'DOOR_TARGET_DEPTH_MAX_M', 'FLOOR_MIN_M', 'LAYOUT_PLACES', 'RULES', 'allowed_compartment', 'allowed_place',
    'door_is_hinge', 'drawer_is_rule1', 'layout_places', 'objects_hideable', 'placement_problems', 'places_from_rows',
    'COLOUR_NAMES', 'LOCK_COLOURS', 'PAD_RADIUS_M', 'PAD_SIZE_M', 'PLATE_SIZE_M', 'TOKEN_MASS_KG',
    'TOKEN_SIZE_M', 'lock_plate', 'pad_name', 'plate_name', 'reader_pad', 'token', 'token_name',
    'LOCK_SLACK_RAD', 'PAD_HEIGHT_TOL_M', 'PAD_SPEED_MAX', 'LockRule', 'closer_candidates', 'is_closed',
    'locked_range', 'token_on_pad',
    'CHAIN_DEPTHS', 'COMPARTMENT_COUNT', 'DEAD_END_LEVELS', 'HOLDOUT_COLOUR', 'HOLDOUT_KEYS', 'KIND_PATTERNS',
    'SET_SIZE', 'all_keys', 'chain_binding', 'colour_combinations', 'dev_composition_counts', 'is_holdout',
    'key_colours', 'pad_offsets', 'pad_spot', 'split_key',
]
