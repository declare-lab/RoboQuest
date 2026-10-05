"""Object skin for the search pool objects.

The pool objects are native RoboCasa/objaverse meshes whose own textures are dropped (they carry brand
artwork). The skin keeps each object's colour name and gives every visual geom a material:

* can / tin body (``label``): generic, text-free label texture in the pool colour (white band, emblem rings),
  brushed-metal ends on the can's own UV layout; metallic finish;
* tin lid (``silver``): brushed silver; water-bottle cap (``black``): black;
* water-bottle body (``speckle``): muted pool colour, satin finish, fine neutral speckle texture;
* mug (``ceramic``): two-tone glazed ceramic, muted pool colour outside and on the handle, cream inside and
  rim, unglazed beige foot (texture painted per face class on the mug's UV layout).

The PNGs in ``skin/`` are generated, text-free textures. Muted colours stay inside the scripted
oracle's HSV ranges for their colour names (search/motor.py COLOUR_HSV). Render only: geoms keep their mesh,
group, contype/conaffinity, size and density, so masses, inertias and contacts are unchanged. No simulator
imports here."""
from pathlib import Path
import xml.etree.ElementTree as ET

SKIN_DIR = Path(__file__).with_name('skin')
# finish per part: specular, shininess, reflectance
FINISH = {'label': (1., .9, .1), 'silver': (1., .9, .1), 'black': (.7, .8, 0.), 'speckle': (.35, .4, 0.),
          'ceramic': (.5, .6, 0.)}
BLACK = (.03, .03, .03, 1.)
SILVER = (.78, .79, .81, 1.)
WHITE = (1., 1., 1., 1.)
MUTED = {'blue': (.15, .28, .68, 1.), 'red': (.72, .14, .12, 1.), 'green': (.16, .52, .24, 1.),
         'yellow': (.85, .68, .15, 1.)}   # bottle and mug colours (HSV inside COLOUR_HSV for the same name)
LABELLED = ('can', 'tin')            # categories with a label texture on their body mesh
CAP_MATERIALS = {'Black_plast.001': 'black', 'Blik': 'silver'}   # native material of the bottle cap / tin lid
BODY_PART = {'can': 'label', 'tin': 'label', 'bottle': 'speckle', 'mug': 'ceramic'}


def texture_path(category, colour):
    """The texture a category's body part uses (bottle: one neutral speckle tinted by the geom colour)."""
    if category == 'bottle':
        return SKIN_DIR / 'speckle.png'
    return SKIN_DIR / f'{category}_{colour}.png'


def part_of(category, native_material):
    """Which skin a visual geom gets, from the native material it had before the flat recolour."""
    native = native_material or ''
    for key, part in CAP_MATERIALS.items():
        if native == key or native.endswith('_' + key):
            return part
    return BODY_PART[category]


def part_rgba(part, colour):
    if part == 'black':
        return BLACK
    if part == 'silver':
        return SILVER
    if part == 'speckle':
        return MUTED[colour]
    return WHITE                      # label and ceramic carry their colour in the texture


def _rgba(values):
    return ' '.join(f'{float(v):g}' for v in values)


def skin_geom(obj, geom, entry, native_material):
    """Give one visual geom of a pool object its skin; registers the material (and texture) on the object's
    own <asset> the first time it is needed. Returns the part name."""
    part = part_of(entry['category'], native_material)
    prefix = obj.naming_prefix
    material = f'{prefix}skin_{part}'
    if obj.asset.find(f"material[@name='{material}']") is None:
        spec, shin, refl = FINISH[part]
        attrs = dict(name=material, specular=f'{spec:g}', shininess=f'{shin:g}', reflectance=f'{refl:g}')
        if part in ('label', 'speckle', 'ceramic'):
            texture = f'{prefix}skin_{part}_tex'
            ET.SubElement(obj.asset, 'texture', name=texture, type='2d',
                          file=str(texture_path(entry['category'], entry['colour'])))
            attrs['texture'] = texture
        ET.SubElement(obj.asset, 'material', **attrs)
    geom.set('material', material)
    geom.set('rgba', _rgba(part_rgba(part, entry['colour'])))
    return part
