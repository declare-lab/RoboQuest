"""Trusted runtime adapter construction; import simulator modules only after EGL setup."""
from importlib import import_module

from roboquest.harness.profiles import SCENE_PROFILES, trusted_goal


def construct_adapter(scene, parameters, *, seed, horizon, image_size, gpu):
    trusted_goal(scene)
    selection = SCENE_PROFILES[scene]
    adapter_class = getattr(import_module(selection['adapter_module']), selection['adapter_class'])
    controls = dict(seed=seed, horizon=horizon, image_size=image_size, gpu=gpu)
    parameters = dict(parameters)
    if 'scene_factory' not in selection:
        return adapter_class(**parameters, **controls)
    # Caddy's accepted adapter accepts an explicit environment rather than an
    # asset path. Preserve that implementation; construct its existing scene here.
    module, name = selection['scene_factory']
    factory = getattr(import_module(module), name)
    env = factory(**parameters, **{**controls, 'horizon':horizon+20})
    parameters.pop('asset_scene')
    try:
        return adapter_class(env=env, **parameters, **controls)
    except Exception:
        env.close()
        raise
