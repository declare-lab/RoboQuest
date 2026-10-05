"""Per-instance appearance for the procedural props from RoboCasa's shipped textures (spec 4.9).

``pick(rng, kind, assets)`` draws one texture of ``kind`` (wood, metal, paper, fabric, plastic
or ceramic) from the pools below, registers it once on the model's ``<asset>`` element as a
cube-mapped texture plus a material with the kind's finish, and returns the material name.
``rgba(rng, kind)`` is the texture-free fallback, a plain colour of the kind. Both take exactly
one draw from ``rng`` (the instance's ``materials`` stream), so a scene stays reproducible
whichever of the two a builder ends up using. Functional colours (Submit red, lid blue and
yellow, borders, pads) are never drawn here, and the pools leave out the saturated blue for
the same reason.

The pools are files under ``robocasa/models/assets/textures`` of the RoboCasa checkout in
``ROBOQUEST_RUNTIME``: the 25 wood grains and planks, the 4 metals plus the brushed steel,
the flat colours and plasters for paper and plastic, printed tiles and dyed flats for fabric,
the glaze, tiles and marbles for ceramic.
"""
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from roboquest.assets._mjcf import draw

RUNTIME_ROOT = Path(os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime')))
TEXTURE_ROOT = RUNTIME_ROOT / 'src/robocasa/robocasa/models/assets/textures'
KINDS = ('wood', 'metal', 'paper', 'fabric', 'plastic', 'ceramic')
PREFIX = 'rq'

_WOODS = ('bamboo', 'dark_wood_parquet', 'dark_wood_planks', 'dark_wood_planks_2', 'gray_wood_grain',
          'gray_wood_planks', 'greenheart_wood_grain', 'light_wood_parquet', 'light_wood_planks',
          'light_wood_planks_2', 'light_wood_planks_long', 'red_wood_planks', 'varnished_wood_planks',
          'walnut_wood_grain', 'warm_wood_grain', 'warm_wood_grain_2', 'warm_wood_parquet', 'warm_wood_planks',
          'wood_grain_1', 'wood_grain_2', 'wood_grain_3', 'wood_planks', 'wood_planks_2', 'wood_planks_3',
          'wood_planks_wall')
TEXTURES = {
    'wood': tuple(f'wood/{name}.png' for name in _WOODS),
    'metal': ('metals/brass.png', 'metals/bright_metal.png', 'metals/brighter_metal.png', 'metals/metal.png',
              'steel-brushed.png'),
    'paper': ('flat/cream.png', 'flat/warm_white.png', 'flat/warm_white_2.png', 'flat/plaster.png',
              'flat/light_gray.png', 'flat/lighter_gray.png', 'flat/white.png', 'cream-plaster.png'),
    'fabric': ('tiles/blue_dots.png', 'tiles/checkers.png', 'tiles/flower_ceramic.png', 'flat/blue_gray.png',
               'flat/green.png', 'flat/light_blue.png', 'flat/light_green.png', 'flat/dark_gray.png',
               'flat/cream.png'),
    'plastic': ('flat/black.png', 'flat/blue_gray.png', 'flat/dark_gray.png', 'flat/gray.png', 'flat/green.png',
                'flat/light_blue.png', 'flat/light_gray.png', 'flat/light_green.png', 'flat/white.png',
                'flat/cream.png'),
    'ceramic': ('ceramic.png', 'tiles/flower_ceramic.png', 'tiles/white_tiles.png',
                'marble/marble.png', 'marble/marble_2.png', 'marble/marble_3.png', 'marble/marble_4.png',
                'marble/marble_5.png', 'marble/marble_6.png', 'marble/dark_marble.png', 'flat/white.png',
                'flat/warm_white.png'),
}
FINISH = {
    'wood': dict(specular='0.2', shininess='0.15', reflectance='0'),
    'metal': dict(specular='0.8', shininess='0.7', reflectance='0.25'),
    'paper': dict(specular='0.05', shininess='0.02', reflectance='0'),
    'fabric': dict(specular='0.02', shininess='0.01', reflectance='0'),
    'plastic': dict(specular='0.5', shininess='0.4', reflectance='0.05'),
    'ceramic': dict(specular='0.7', shininess='0.6', reflectance='0.1'),
}
PALETTES = {
    'wood': ((.60, .44, .28, 1.), (.45, .30, .18, 1.), (.72, .56, .36, 1.), (.35, .22, .13, 1.),
             (.66, .50, .32, 1.), (.52, .38, .24, 1.)),
    'metal': ((.72, .73, .75, 1.), (.55, .56, .58, 1.), (.80, .68, .40, 1.), (.40, .41, .43, 1.),
              (.85, .86, .88, 1.)),
    'paper': ((.85, .72, .52, 1.), (.93, .90, .82, 1.), (.78, .64, .44, 1.), (.96, .95, .90, 1.),
              (.88, .80, .66, 1.)),
    'fabric': ((.20, .30, .55, 1.), (.60, .20, .20, 1.), (.25, .45, .30, 1.), (.50, .45, .35, 1.),
               (.35, .30, .45, 1.), (.70, .60, .40, 1.)),
    'plastic': ((.90, .90, .90, 1.), (.15, .15, .15, 1.), (.15, .55, .55, 1.), (.30, .60, .35, 1.),
                (.45, .30, .55, 1.), (.80, .75, .65, 1.), (.50, .50, .52, 1.)),
    'ceramic': ((.95, .95, .93, 1.), (.85, .90, .95, 1.), (.93, .88, .80, 1.), (.75, .80, .85, 1.),
                (.60, .65, .70, 1.), (.98, .98, .98, 1.)),
}


def _kind(kind):
    if kind not in KINDS:
        raise ValueError(f'material kind must be one of {KINDS}, not {kind!r}')
    return kind


def texture_path(relative):
    path = TEXTURE_ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f'RoboCasa texture missing: {path}')
    return path


def material_name(kind, relative):
    return f'{PREFIX}_{_kind(kind)}_{Path(relative).stem}'


def register(assets, kind, relative, name=None):
    """Register one RoboCasa texture and its material once on ``assets``; returns the material name."""
    if not hasattr(assets, 'find'):
        raise TypeError('assets must be the <asset> element of the model')
    name = name or material_name(kind, relative)
    if assets.find(f"material[@name='{name}']") is None:
        ET.SubElement(assets, 'texture', name=name + '_tex', type='cube', file=str(texture_path(relative)))
        ET.SubElement(assets, 'material', name=name, texture=name + '_tex', texuniform='true', texrepeat='1 1',
                      **FINISH[_kind(kind)])
    return name


def pick(rng, kind, assets, name=None):
    """Draw a texture of ``kind`` with one draw of ``rng``, register it on ``assets`` and return the
    material name. Deterministic for a given rng state; the same texture is registered once."""
    pool = TEXTURES[_kind(kind)]
    return register(assets, kind, pool[draw(rng, len(pool))], name)


def rgba(rng, kind):
    """A plain colour of ``kind`` with one draw of ``rng``: the fallback when no assets element is at hand."""
    palette = PALETTES[_kind(kind)]
    return tuple(palette[draw(rng, len(palette))])


def dress(rng, kind, assets=None, material=None):
    """(material, rgba) for a builder's visuals: the material given, else a texture registered on
    ``assets`` when there is one, else a plain colour. One draw of ``rng`` in the last two cases."""
    if material is not None:
        return material, None
    if assets is not None:
        return pick(rng, kind, assets), None
    return None, rgba(rng, kind)
