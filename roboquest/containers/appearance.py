"""Per-instance appearance of the panel boxes (spec 4.9): case and panel textures, knob colour.

Every case in an instance shares one appearance (the cases must stay identical to each other; that is
the task), drawn from the ``materials`` stream at mint time and recorded in the spec, so the CPU-side
generator never touches the file system. The textures are the wood pool of
:mod:`roboquest.assets.materials` (RoboCasa's own wood grains and planks) and are
registered on the model through that module's ``register`` at build time, with its wood finish.
Functional colours are untouched (the Submit button stays red).
"""
import xml.etree.ElementTree as ET  # noqa: F401  (the bodies passed in are etree elements)

from roboquest.assets import materials as M

WOOD_TEXTURES = tuple(M.TEXTURES['wood'])
KNOB_COLOURS = ((.20, .20, .22), (.12, .12, .13), (.35, .28, .24), (.72, .73, .75), (.55, .45, .30),
                (.60, .60, .62), (.30, .22, .18), (.42, .42, .45))


def draw_appearance(rng):
    """One appearance for every case of an instance (plain JSON): case texture, panel texture (a
    different wood, so the overlay plates read as plates) and knob colour. Three draws of ``rng``."""
    case = WOOD_TEXTURES[M.draw(rng, len(WOOD_TEXTURES))]
    others = [t for t in WOOD_TEXTURES if t != case]
    panel = others[M.draw(rng, len(others))]
    knob = [float(v) for v in KNOB_COLOURS[M.draw(rng, len(KNOB_COLOURS))]]
    return dict(case_texture=case, panel_texture=panel, knob_rgb=knob)


def register(assets, appearance):
    """Register the instance's two wood textures on the model assets; returns the material names."""
    return {role: M.register(assets, 'wood', appearance[f'{role}_texture']) for role in ('case', 'panel')}


def _rgba_key(text):
    return tuple(round(float(v), 3) for v in text.split())


def apply_to_bodies(bodies, materials, knob_rgb, wood_rgba, panel_rgba, knob_rgba):
    """Rewrite the visual geoms under ``bodies``: case wood gets the case material, the overlay plates
    and the top plate the panel material, the knobs the drawn colour. Real and fake panels are rewritten
    by the same rule, so they stay identical. Returns the number of geoms changed."""
    wood, panel, knob = _rgba_key(wood_rgba), _rgba_key(panel_rgba), _rgba_key(knob_rgba)
    changed = 0
    knob_text = ' '.join(f'{v:g}' for v in knob_rgb) + ' 1'
    for body in bodies:
        for geom in body.iter('geom'):
            if geom.get('group') != '1' or geom.get('rgba') is None:
                continue
            key = _rgba_key(geom.get('rgba'))
            if key == wood:
                geom.set('material', materials['case'])
                del geom.attrib['rgba']
            elif key == panel:
                geom.set('material', materials['panel'])
                del geom.attrib['rgba']
            elif key == knob:
                geom.set('rgba', knob_text)
            else:
                continue
            changed += 1
    return changed
