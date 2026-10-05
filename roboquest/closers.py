"""Passive closers: a joint spring that shuts a compartment door, drawer or container panel by itself.

Search room and the container mechanisms draw, per instance, which joints get one (spec 4.4 / 4.5).
The closer is applied to the compiled model after ``_setup_references``; nothing is written to the
MJCF, so the appearance is unchanged and the goal never mentions it.

The spring rest position sits slightly past the closed limit so the door shuts firmly; the joint
range stops it there. Stiffness and damping are sized for the wanted closing time from fully open
by the joint's slow pole: an over-damped joint with inertia ``I`` (``dof_M0``), damping ``d`` and
stiffness ``k`` settles as ``exp(-s t)`` with ``s = (d - sqrt(d^2 - 4 k I)) / (2 I)``, so asking for
``s = ln(1 / CLOSED_FRACTION) / close_time_s`` gives ``k = s (d - I s)``. The joint's own damping is
kept when it already makes the joint over-damped at that pole (RoboCasa gives hinges 2 N m s/rad and
drawers 10 N s/m, which dominates a weak spring), otherwise it is raised to ``2 zeta I s`` so the
door never bounces off its closed limit. Verified in the search room (2026-09-21): drawers and
doors close from fully open within a few per cent of the asked time. The joint's ``frictionloss`` can
still stall a weak spring, so the spring force at full open is raised to at least ``friction_factor``
times the friction loss.
"""
import math

CLOSED_FRACTION = .02   # "closed": within this fraction of the travel from the limit (9 mm on a 0.45 m drawer, 1.8 deg on a door)


def closed_qpos(model, jid, rest_qpos):
    """Where the joint actually comes to rest: the spring rest, clipped to the joint range (the rest is set
    slightly past the closed limit, so the limit stops the joint short of it)."""
    lo, hi = (float(v) for v in model.jnt_range[jid])
    if int(model.jnt_limited[jid]) and hi > lo:
        return min(max(float(rest_qpos), lo), hi)
    return float(rest_qpos)


def design(inertia, joint_damping, close_time_s, zeta=1.2, closed_fraction=CLOSED_FRACTION):
    """(stiffness, damping, mode) for the slow-pole design described in the module docstring."""
    sigma = math.log(1. / float(closed_fraction)) / float(close_time_s)
    damping = max(float(joint_damping), 2. * float(zeta) * float(inertia) * sigma)
    stiffness = sigma * (damping - float(inertia) * sigma)
    return stiffness, damping, ('joint_damping' if damping == float(joint_damping) else 'raised_damping')


def apply_closer(sim, joint_name, rest_qpos, close_time_s=10.0, zeta=1.2, friction_factor=3.0, open_qpos=None):
    """Turn ``joint_name`` into a self-closing joint whose rest position is ``rest_qpos``.

    ``open_qpos`` is the fully open position the closing time is designed from (defaults to the far end of the
    joint range); ``zeta`` is the least damping ratio when the damping has to be raised. Returns the
    parameters written, for the spec / gate report.
    """
    model = sim.model
    jid = model.joint_name2id(joint_name)
    qadr = int(model.jnt_qposadr[jid])
    dadr = int(model.jnt_dofadr[jid])
    inertia = max(float(model.dof_M0[dadr]), 1e-4)
    if open_qpos is None:
        lo, hi = (float(v) for v in model.jnt_range[jid])
        open_qpos = hi if abs(hi - rest_qpos) > abs(lo - rest_qpos) else lo
    travel = abs(float(open_qpos) - float(rest_qpos))
    joint_damping = float(model.dof_damping[dadr])
    stiffness, damping, mode = design(inertia, joint_damping, close_time_s, zeta)
    friction = float(model.dof_frictionloss[dadr])
    if friction > 0 and travel > 0 and stiffness < friction_factor * friction / travel:
        stiffness = friction_factor * friction / travel
        damping = max(damping, 2. * zeta * math.sqrt(stiffness * inertia))
        mode = 'friction_floor'
    model.jnt_stiffness[jid] = stiffness
    model.dof_damping[dadr] = damping
    model.qpos_spring[qadr] = float(rest_qpos)
    return dict(joint=joint_name, stiffness=round(float(stiffness), 5), damping=round(float(damping), 4),
                rest_qpos=float(rest_qpos), open_qpos=float(open_qpos), closed_qpos=closed_qpos(model, jid, rest_qpos),
                inertia=round(inertia, 6), frictionloss=friction, joint_damping=joint_damping,
                close_time_s=float(close_time_s), design=mode, closed_tol=float(CLOSED_FRACTION * travel))


def closing_time(sim, joint_name, env_step, open_qpos, closed_tol, max_ticks=600):
    """Measure how many ticks the joint needs to come within ``closed_tol`` of the position it can actually rest
    at (the closed limit when the spring rest lies past it) after being set to ``open_qpos``; ``env_step`` is a
    zero-action step callable. Returns (ticks, final_qpos); ticks is None when it never arrives."""
    model, data = sim.model, sim.data
    jid = model.joint_name2id(joint_name)
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr] = float(open_qpos)
    data.qvel[int(model.jnt_dofadr[jid])] = 0.0
    sim.forward()
    rest = closed_qpos(model, jid, float(model.qpos_spring[qadr]))
    for tick in range(1, max_ticks + 1):
        env_step()
        if abs(float(data.qpos[qadr]) - rest) <= closed_tol:
            return tick, float(data.qpos[qadr])
    return None, float(data.qpos[qadr])
