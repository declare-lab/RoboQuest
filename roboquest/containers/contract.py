"""Mechanism and face vocabulary of the unfamiliar-containers boxes.

Copied from the open-containers worktree (``panel_boxes_contract.py``, the
formal "panel boxes" version). Every box is the same case carrying one or two
overlay panels with round knobs; exactly one panel can open, in one of eleven
ways, and a second knobbed panel is a dummy screwed to a solid wall. Which
face opens, how, and which knob is a dummy are evaluator/oracle privileges.
No simulator import here.
"""
# Mechanisms of a panel on one of the four upright faces.
SIDE_MECHANISMS = ('drawer', 'press_drawer', 'press_door', 'door', 'turn_knob', 'slide_side', 'slide_up')
# Mechanisms of the top panel.
TOP_MECHANISMS = ('lift_lid', 'slide_lid', 'twist_lid', 'flip_lid')
MECHANISMS = SIDE_MECHANISMS + TOP_MECHANISMS
SEALED = 'sealed'
SIDE_FACES = ('front', 'right', 'back', 'left')
FACES = SIDE_FACES + ('top',)

OPENING_ACTION = {
    'drawer': 'pull the knob straight out; the tray slides with it',
    'press_drawer': 'press the panel in firmly; a latch releases, a spring pops it out, then pull',
    'press_door': 'press the panel in firmly; a latch releases and a spring cracks it ajar on a side hinge; swing it open by the knob',
    'door': 'pull the knob; the panel swings open on a hidden hinge along one of its side edges',
    'turn_knob': 'turn the knob about a quarter turn (either way) to free a hidden tab behind the opening, then pull',
    'slide_side': 'push the panel sideways along the face',
    'slide_up': 'push the panel straight up the face; stiff guides hold it wherever it is left',
    'lift_lid': 'lift the loose top panel off',
    'slide_lid': 'push the top panel along the top; hidden tongues under a rail block lifting',
    'twist_lid': 'turn the knob about a quarter turn (either way), then lift; a hidden bayonet under the lid blocks a straight lift',
    'flip_lid': 'lift the edge of the top panel opposite its hidden hinge',
    SEALED: 'cannot be opened',
}

RESPONSE = {
    'drawer': 'pull', 'press_drawer': 'press_then_pull', 'press_door': 'press_then_swing',
    'door': 'pull_swings', 'turn_knob': 'turn_then_pull', 'slide_side': 'push_sideways',
    'slide_up': 'push_up', 'lift_lid': 'lift', 'slide_lid': 'push_along_top',
    'twist_lid': 'twist_then_lift', 'flip_lid': 'lift_arc', SEALED: 'none',
}
