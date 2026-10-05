"""The lock rule of ``locked_storage`` (contract v1.1 section 2), as pure state so it can be tested on fakes.

``token_on_pad`` is geometry only: the token body centre inside the pad rectangle in xy, its bottom
within :data:`PAD_HEIGHT_TOL_M` of the pad top, and its speed below :data:`PAD_SPEED_MAX`. The lock
itself is a three-line state machine evaluated every tick:

* token on its pad  -> unlocked;
* else closed (within ``closers.CLOSED_FRACTION`` of the closed limit) -> locked;
* else (open, token away) -> unchanged: a bolt cannot engage while the door stands open, so an open
  compartment stays open until it is closed again.

The task turns ``locked`` into physics by collapsing the compartment's joint range to
``closed +- LOCK_SLACK_RAD`` and restores RoboCasa's range to unlock.
"""
import math

from roboquest.closers import CLOSED_FRACTION

PAD_HEIGHT_TOL_M = .03      # the token's bottom may sit this far above (or below) the pad top
PAD_SPEED_MAX = .05         # m/s: a token still being carried does not read
PAD_XY_MARGIN_M = .0        # the centre must be inside the rectangle itself
LOCK_SLACK_RAD = 1e-4       # the collapsed joint range half-width (MuJoCo needs range[0] < range[1])

EVENT_KINDS = ('unlock', 'relock', 'token_on_pad', 'token_off_pad')


def pad_local(pad, position):
    """``position`` (world xyz) in the pad's own frame: x along the slab's long axis, y across it."""
    dx = float(position[0]) - float(pad['centre'][0])
    dy = float(position[1]) - float(pad['centre'][1])
    yaw = float(pad.get('yaw', 0.))
    cos, sin = math.cos(-yaw), math.sin(-yaw)
    return (dx * cos - dy * sin, dx * sin + dy * cos, float(position[2]))


def token_on_pad(pad, position, bottom_z, speed, margin=PAD_XY_MARGIN_M):
    """Is the token resting on this reader pad? See the module docstring."""
    x, y, _ = pad_local(pad, position)
    hx, hy = pad['half_xy']
    if abs(x) > float(hx) + float(margin) or abs(y) > float(hy) + float(margin):
        return False
    if abs(float(bottom_z) - float(pad['top_z'])) > PAD_HEIGHT_TOL_M:
        return False
    return float(speed) <= PAD_SPEED_MAX


def is_closed(open_fraction, closed_fraction=CLOSED_FRACTION):
    """Closed within ``closed_fraction`` of the travel (the same threshold the passive closers use)."""
    return float(open_fraction) <= float(closed_fraction)


class LockRule:
    """Per-compartment lock state for the locked compartments of one instance.

    ``compartment_ids`` are the locked compartments (chain plus dead end). Every one starts closed
    and locked. :meth:`update` takes ``{cid: (token_on_pad, closed)}`` and returns the events that
    fired this tick as ``(kind, cid)`` pairs, in a stable order.
    """

    def __init__(self, compartment_ids):
        self.ids = list(compartment_ids)
        self.locked = {cid: True for cid in self.ids}
        self.on_pad = {cid: False for cid in self.ids}
        self.unlocked_ever = set()
        self.relocks = {cid: 0 for cid in self.ids}

    def update(self, observations):
        events = []
        for cid in self.ids:
            on_pad, closed = observations[cid]
            on_pad, closed = bool(on_pad), bool(closed)
            if on_pad != self.on_pad[cid]:
                events.append(('token_on_pad' if on_pad else 'token_off_pad', cid))
                self.on_pad[cid] = on_pad
            if on_pad:
                want = False
            elif closed:
                want = True
            else:
                want = self.locked[cid]     # open and unattended: the bolt cannot engage
            if want != self.locked[cid]:
                if want:
                    self.relocks[cid] += 1
                    events.append(('relock', cid))
                else:
                    self.unlocked_ever.add(cid)
                    events.append(('unlock', cid))
                self.locked[cid] = want
        return events

    def summary(self):
        return dict(locked={cid: bool(v) for cid, v in self.locked.items()},
                    on_pad={cid: bool(v) for cid, v in self.on_pad.items()},
                    unlocked_ever=sorted(self.unlocked_ever),
                    relocks={cid: int(v) for cid, v in self.relocks.items()})


def locked_range(closed_qpos, slack=LOCK_SLACK_RAD):
    """The collapsed joint range that holds a compartment shut, the way ``_disable_other_fixtures``
    locks every other fixture statically."""
    return (float(closed_qpos) - float(slack), float(closed_qpos) + float(slack))


def closer_candidates(chain_ids, dead_end_id, compartment_ids):
    """Compartments a passive closer may be placed on (contract section 2).

    Every free compartment and every chain compartment *except the last*. Never the last chain
    compartment (it holds the target) and never the dead end (it holds the dead-end distractor):
    a closer there would shut the prize in while the robot fetches the token. A closer on a
    mid-chain compartment only re-locks it with nothing lost, which is intended.
    """
    banned = set()
    if chain_ids:
        banned.add(chain_ids[-1])
    if dead_end_id is not None:
        banned.add(dead_end_id)
    return [cid for cid in sorted(compartment_ids) if cid not in banned]
