"""Public robot controls for any RoboQuest task; episodes end at the first physical Submit."""
from copy import deepcopy

from roboquest.harness.mobile_native_contract import PUBLIC_CONTROL_VERSION
from roboquest.harness.base_adapter import TableSettingAdapter
from roboquest.evaluator import first_submit_evaluator
from roboquest.manifest import task_class
from roboquest.registry import load_instance


RESET_TICKS = 10   # the fixed initialization steps TableSettingAdapter.reset takes before step 1


def check_seed(instance, seed, allow_foreign_seed=False):
    """A registry instance is built with its own seed: the adapter's ``seed`` reseeds the kitchen's
    decor and fixture sampling before ``env.reset``, so a runner seed silently leaves the gated
    scene (built with another seed, the task objects match to 1e-5, but the decor clearance, utensil
    racks, window groups and stools do not, and an instance may not build at all).
    ``None`` means the instance seed; anything else must equal it unless the caller says so."""
    if seed is None or allow_foreign_seed:
        return
    if int(seed) != int(instance['seed']):
        raise ValueError(
            f"RoboQuest instance {instance.get('instance_id', '?')} is built with its own seed "
            f"{instance['seed']}; a runner seed of {seed} would resample the kitchen's decor and fixtures "
            f"and leave the gated scene. Pass seed=None (the default) or allow_foreign_seed=True on purpose.")

# Spec 1.2: the budget is fixed per task and is never stated to the policy. Any packet field
# that counts the episode down states it, so every key naming one is dropped (see observe).
TIME_REMAINING_TERMS = ('remaining', 'tick', 'horizon', 'budget')


class RoboQuestAdapter(TableSettingAdapter):
    def __init__(self, instance, seed=None, image_size=512, horizon=None, gpu=0,
                 near_clip_m=.001, env=None, allow_foreign_seed=False, render_on_demand=False,
                 terminal_only_tick_snapshots=False):
        instance = load_instance(instance)
        cls = task_class(instance['task'])
        check_seed(instance, seed, allow_foreign_seed)
        # Spec 1.2: the task's own fixed budget is the episode length, the same for every instance.
        horizon = int(cls.BUDGET_TICKS if horizon is None else horizon)
        if env is None:
            from roboquest.kitchen import make_env
            env = make_env(cls, instance, image_size=image_size, gpu=gpu,
                           horizon=horizon + RESET_TICKS, camera_obs=not render_on_demand)
        self.instance = instance
        self.task_class = cls
        self.terminal_only_tick_snapshots = bool(terminal_only_tick_snapshots)
        super().__init__(seed=int(instance['seed']) if seed is None else seed, image_size=image_size,
                         horizon=horizon, gpu=gpu, near_clip_m=near_clip_m, env=env,
                         evaluator_factory=first_submit_evaluator(cls.CONTRACT_VERSION),
                         instruction=cls.goal(instance['spec']), control_version=PUBLIC_CONTROL_VERSION)

    def _tick_evaluator_snapshot(self, step):
        """Avoid rebuilding full private geometry on ordinary VLA ticks.

        RoboQuest's environment handles the physical button during ``env.step``
        and freezes the exact task score at that instant.  Before a submission
        or timeout the adapter only needs a truthful nonterminal marker.  A
        complete snapshot is still captured at every terminal tick and by
        :meth:`score`, preserving final scoring and audit evidence.
        """
        if not self.terminal_only_tick_snapshots:
            return super()._tick_evaluator_snapshot(step)
        if getattr(self.env, 'submission', None) is not None or bool(getattr(self.env, 'timed_out', False)):
            return super()._tick_evaluator_snapshot(step)
        return {
            'step': int(step),
            'score': {
                'success': False,
                'submitted': False,
                'outcome': None,
                'termination': 'awaiting_submit',
            },
            'submission': None,
            'roboquest': {'events': deepcopy(getattr(self.env, 'events', []))},
        }

    def reset(self):
        """Line the scene's timeout up with the policy's budget.

        The fixed initialization steps run on the environment's own tick
        counter, which the adapter's step count does not include, so the
        scene's horizon is moved to the tick where the last charged step lands:
        the episode then ends in ``timeout`` (spec 1.1) exactly when the budget
        is used, and never one tick earlier.
        """
        observation = super().reset()
        self.env.horizon = int(self.env.timestep) + int(self.horizon)
        return observation

    def observe(self):
        """The public packet with no remaining-time field of any kind (spec 1.2).

        ``TableSettingAdapter`` publishes ``remaining_ticks``; here the episode
        horizon *is* the task's fixed budget, so that field would state the
        budget at step 0. The policy still sees elapsed time (``step``,
        ``sim_time_s``) but never a countdown. The key is removed rather than
        nulled or defaulted, so no value of it can be read as the budget.
        """
        packet = super().observe()
        for key in [key for key in packet
                    if any(term in key.lower() for term in TIME_REMAINING_TERMS)]:
            del packet[key]
        return packet

    def _physical_termination(self, snapshot):
        return 'physical_submit' if snapshot['score']['submitted'] else None

    def outcome(self):
        """The episode's outcome per spec 1.1: success, wrong_submit, timeout, or None while running."""
        return self.env.evaluate_success()['outcome']
