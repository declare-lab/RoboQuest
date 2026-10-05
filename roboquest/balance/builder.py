"""Two-pan beam balance as MJCF (ElementTree), dressed with RoboCasa textures.

Mechanism (all ordinary MuJoCo joints, nothing scripted):

* a fixed base plate and pillar; the **beam** rotates on a hinge at the pillar
  top (axis +y of the balance frame, the beam lying along x) with joint-range
  **stops** at +-``stop_deg``, a torsional **flexure** (joint stiffness, zero
  rest angle) and viscous damping;
* each **pan** hangs from a beam end on its own hinge (same axis) like a
  pendulum, so the load always acts vertically through the hinge and the reading
  does not depend on where a parcel sits on the pan;
* a **pointer** fixed to the beam hangs below the pivot and sweeps over a dial
  plate on the pillar's front face (five ticks: the stops, half way, level).

Sign convention: a positive beam angle lowers the +x pan ("right" when the
balance frame's +y points away from the robot). The flexure makes the reading
graded, ``theta = dm * g * arm / k`` up to the stops (see
:func:`tilt_for_mass_difference`); the stops are reached for differences above
roughly ``k * tan(stop) / (g * arm)``.

The instrument stays under 0.40 m tall (beam ends at the stop) because RoboCasa
wall cabinets start 0.46 m above the counter top.

Geoms follow the suite convention: every mass-carrying geom is a group-0
collision geom (robosuite's ``inertiagrouprange="0 0"``) with a group-1 visual
twin that carries the material, because robosuite's offscreen renderer hides
group 0.
"""
from dataclasses import dataclass, asdict, replace
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

ROBOCASA_TEXTURES = (Path(os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime')))
                     / 'src/robocasa/robocasa/models/assets/textures')
GRAVITY = 9.81


def _vec(values):
    return ' '.join(f'{float(v):.9g}' for v in values)


@dataclass(frozen=True)
class BalanceDesign:
    """Dimensions (m), masses (kg) and joint parameters of the balance."""
    arm: float = .18                   # pivot to pan hinge
    beam_half: tuple = (.195, .008, .006)
    beam_drop: float = .012            # beam bar centre below the pivot (CoM under the pivot)
    pivot_z: float = .34               # hinge height above the base plane
    hanger_height: float = .28         # pan hinge to pan floor top
    pan_half: tuple = (.13, .08, .004)  # pan floor: 26 x 16 cm, 8 mm thick (three 7 cm parcels along x)
    rim_height: float = .02
    rim_thickness: float = .004
    rod_radius: float = .004
    hanger_back: float = .076          # hanger rods stand on the back rim line (+y)
    hanger_span: float = .11           # half distance between the two hanger rods (x)
    stop_deg: float = 12.
    pan_swing_deg: float = 60.         # pan hinge range, only a safety against flips
    beam_stiffness: float = 1.8        # N m / rad, torsional flexure at the pivot
    # The pans stay vertical while the beam turns, so each pan damper also damps the
    # beam: effective beam damping = beam_damping + 2 * pan_damping (tuned in the rig).
    beam_damping: float = .25          # N m s / rad
    pan_damping: float = .30
    base_half: tuple = (.05, .08, .006)
    pillar_half: tuple = (.015, .015)
    pointer_length: float = .12        # hangs below the pivot
    pointer_half_width: float = .003
    plate_half: tuple = (.045, .002, .03)
    mass_beam: float = .25
    mass_pointer: float = .01
    mass_pan_floor: float = .20
    mass_rim: float = .01              # each of four rims
    mass_rod: float = .02              # each hanger rod, spar and arm
    friction_pan: str = '1.0 0.005 0.0001'
    friction_metal: str = '0.6 0.005 0.0001'
    textures: tuple = (('frame', 'metals/metal.png'), ('beam', 'metals/brass.png'),
                       ('pan', 'metals/bright_metal.png'), ('plate', 'flat/warm_white.png'))

    @property
    def stop_rad(self):
        return math.radians(self.stop_deg)

    @property
    def pan_inner_half(self):
        return (self.pan_half[0] - self.rim_thickness, self.pan_half[1] - self.rim_thickness)

    @property
    def pan_floor_z(self):
        """Pan floor top above the base plane when the beam is level."""
        return self.pivot_z - self.hanger_height

    @property
    def max_height(self):
        """Highest point of the instrument (a beam end block at the stop)."""
        return self.pivot_z + .012 + self.arm * math.sin(self.stop_rad) + self.rod_radius

    @property
    def half_width(self):
        """Half extent along x: outer pan edge or hanger spar, whichever is wider."""
        return self.arm + max(self.pan_half[0], self.hanger_span + self.rod_radius)


DEFAULT_DESIGN = BalanceDesign()

# Per-instance appearance (spec 4.9): the instrument's metals vary, the reading does not.
# The pointer stays dark and the dial's centre tick stays red - those are functional.
METAL_TEXTURES = ('metals/metal.png', 'metals/brass.png', 'metals/bright_metal.png',
                  'metals/brighter_metal.png')
PLATE_TEXTURES = ('flat/warm_white.png', 'flat/white.png', 'flat/warm_white_2.png',
                  'flat/light_gray.png', 'flat/cream.png')
TEXTURE_KEYS = ('frame', 'beam', 'pan', 'plate')


def sample_textures(rng):
    """Draw one metal per structural key and a light dial plate. Returns ``{key: relative path}``.

    ``rng`` is the instance's ``materials`` stream; the result belongs in the spec, so the
    build stays a pure function of the instance.
    """
    chosen = {key: METAL_TEXTURES[int(rng.integers(0, len(METAL_TEXTURES)))]
              for key in ('frame', 'beam', 'pan')}
    chosen['plate'] = PLATE_TEXTURES[int(rng.integers(0, len(PLATE_TEXTURES)))]
    return chosen


def textures_are_known(chosen):
    """True when ``chosen`` names one texture per key from the allowed palettes."""
    if set(chosen or ()) != set(TEXTURE_KEYS):
        return False
    return (all(chosen[key] in METAL_TEXTURES for key in ('frame', 'beam', 'pan'))
            and chosen['plate'] in PLATE_TEXTURES)


def with_textures(chosen, design=DEFAULT_DESIGN):
    """``design`` dressed in ``chosen`` (a ``{key: relative path}`` map); geometry untouched."""
    return replace(design, textures=tuple((key, chosen[key]) for key in TEXTURE_KEYS))


def tilt_for_mass_difference(delta_kg, design=DEFAULT_DESIGN):
    """Static beam angle (deg) for a mass difference between the pans, clipped at the stops."""
    angle = math.degrees(delta_kg * GRAVITY * design.arm / design.beam_stiffness)
    return max(-design.stop_deg, min(design.stop_deg, angle))


def beam_stiffness_for(delta_kg, tilt_deg, arm=DEFAULT_DESIGN.arm):
    """Flexure stiffness giving ``tilt_deg`` for a ``delta_kg`` difference at ``arm``."""
    return delta_kg * GRAVITY * arm / math.radians(tilt_deg)


def _pair(body, name, geom_type, pos, size, material=None, rgba=None, mass=None, friction=None, fromto=None):
    """Collision geom (group 0, carries the mass) plus its visual twin (group 1, carries the material)."""
    common = dict(name=name, type=geom_type, size=_vec(size))
    if fromto is not None:
        common['fromto'] = _vec(fromto)
    else:
        common['pos'] = _vec(pos)
    collision = dict(common, group='0', contype='1', conaffinity='1', rgba='0.5 0.5 0.5 1',
                     solref='0.006 1', solimp='0.98 0.999 0.001', margin='0', condim='3')
    if mass is not None:
        collision['mass'] = f'{mass:.9g}'
    if friction is not None:
        collision['friction'] = friction
    ET.SubElement(body, 'geom', **collision)
    visual = dict(common, name=name + '_visual', group='1', contype='0', conaffinity='0', density='0')
    if material is not None:
        visual['material'] = material
    if rgba is not None:
        visual['rgba'] = _vec(rgba)
    ET.SubElement(body, 'geom', **visual)
    return name


def _add_materials(assets, prefix, design):
    names = {}
    for key, relative in design.textures:
        path = ROBOCASA_TEXTURES / relative
        if not path.is_file():
            raise FileNotFoundError(f'RoboCasa texture missing: {path}')
        tex, mat = f'{prefix}_tex_{key}', f'{prefix}_mat_{key}'
        ET.SubElement(assets, 'texture', name=tex, type='cube', file=str(path))
        shiny = key in ('beam', 'pan', 'frame')
        ET.SubElement(assets, 'material', name=mat, texture=tex, texuniform='true', texrepeat='1 1',
                      specular='0.5' if shiny else '0.1', shininess='0.4' if shiny else '0.05',
                      reflectance='0.15' if shiny else '0')
        names[key] = mat
    return names


def add_balance(parent, assets, contact=None, prefix='balance', design=DEFAULT_DESIGN):
    """Append the balance under ``parent`` (a body element whose origin is on the support plane).

    ``assets`` receives the RoboCasa textures/materials; ``contact`` (the model's
    ``<contact>`` element) receives exclusions between the fixed frame and the
    moving parts. They are needed because the frame is welded to the world, so
    MuJoCo's parent-child contact filter does not cover the axle in its fork.
    Returns metadata for the evaluator and gates: joint names, geom names per
    part, the pan floor and inner extents, the pointer tip, in the balance frame.
    """
    d = design
    mats = _add_materials(assets, prefix, d)
    if contact is not None:
        for moving in ('beam', 'pan_left', 'pan_right'):
            ET.SubElement(contact, 'exclude', name=f'{prefix}_x_{moving}', body1=f'{prefix}_base',
                          body2=f'{prefix}_{moving}')
    geoms = {'frame': [], 'beam': [], 'pan_left': [], 'pan_right': []}
    base = ET.SubElement(parent, 'body', name=f'{prefix}_base', pos='0 0 0')
    # Fixed frame: plate, pillar, fork holding the axle, dial plate on the pillar's front face.
    geoms['frame'].append(_pair(base, f'{prefix}_plate', 'box', (0, 0, d.base_half[2]), d.base_half,
                                material=mats['frame'], mass=.6, friction=d.friction_metal))
    pillar_top = d.pivot_z - .02
    pillar_z0 = 2 * d.base_half[2]
    geoms['frame'].append(_pair(base, f'{prefix}_pillar', 'box', (0, 0, (pillar_z0 + pillar_top) / 2),
                                (d.pillar_half[0], d.pillar_half[1], (pillar_top - pillar_z0) / 2),
                                material=mats['frame'], mass=.4, friction=d.friction_metal))
    for side, sy in (('front', -1), ('back', 1)):
        geoms['frame'].append(_pair(base, f'{prefix}_fork_{side}', 'box', (0, sy * .018, d.pivot_z - .006),
                                    (.02, .003, .018), material=mats['frame'], mass=.02, friction=d.friction_metal))
    plate_y = -(d.pillar_half[1] + d.plate_half[1])
    plate_z = d.pivot_z - d.pointer_length + .005
    geoms['frame'].append(_pair(base, f'{prefix}_tick_plate', 'box', (0, plate_y, plate_z), d.plate_half,
                                material=mats['plate'], mass=.02, friction=d.friction_metal))
    tip_radius = d.pointer_length
    ticks = []
    for deg in (-d.stop_deg, -d.stop_deg / 2, 0., d.stop_deg / 2, d.stop_deg):
        x = tip_radius * math.sin(math.radians(deg))
        z = d.pivot_z - tip_radius * math.cos(math.radians(deg)) + .004
        rgba = (.75, .08, .06, 1) if deg == 0 else (.08, .08, .08, 1)
        name = f'{prefix}_tick_{int(deg + d.stop_deg)}'
        ET.SubElement(base, 'geom', name=name, type='box', pos=_vec((x, plate_y - d.plate_half[1] - .0005, z)),
                      size=_vec((.0012 if deg else .0018, .0005, .012)), rgba=_vec(rgba), group='1',
                      contype='0', conaffinity='0', density='0')
        ticks.append(dict(deg=deg, x=x, z=z, geom=name))
    # Beam on the pivot hinge; its bar hangs slightly below the axle so the CoM is under the pivot.
    beam = ET.SubElement(base, 'body', name=f'{prefix}_beam', pos=_vec((0, 0, d.pivot_z)))
    ET.SubElement(beam, 'joint', name=f'{prefix}_beam_hinge', type='hinge', axis='0 1 0',
                  range=_vec((-d.stop_rad, d.stop_rad)), limited='true', stiffness=f'{d.beam_stiffness:.9g}',
                  damping=f'{d.beam_damping:.9g}', springref='0', armature='0.0002')
    geoms['beam'].append(_pair(beam, f'{prefix}_beam_bar', 'box', (0, 0, -d.beam_drop), d.beam_half,
                               material=mats['beam'], mass=d.mass_beam, friction=d.friction_metal))
    geoms['beam'].append(_pair(beam, f'{prefix}_axle', 'cylinder', None, (.008, .012), material=mats['beam'],
                               mass=.02, friction=d.friction_metal, fromto=(0, -.012, 0, 0, .012, 0)))
    for side, sx in (('left', -1), ('right', 1)):
        geoms['beam'].append(_pair(beam, f'{prefix}_end_{side}', 'box', (sx * d.arm, 0, 0), (.012, .012, .012),
                                   material=mats['beam'], mass=.03, friction=d.friction_metal))
    pointer_y = plate_y - d.plate_half[1] - .004 - d.pointer_half_width
    geoms['beam'].append(_pair(beam, f'{prefix}_pointer', 'box', (0, pointer_y, -d.pointer_length / 2),
                               (d.pointer_half_width, d.pointer_half_width, d.pointer_length / 2),
                               rgba=(.03, .03, .03, 1), mass=d.mass_pointer, friction=d.friction_metal))
    ET.SubElement(beam, 'site', name=f'{prefix}_pointer_tip', pos=_vec((0, pointer_y, -d.pointer_length)),
                  size='0.002', rgba='0 0 0 0', group='5')
    pans = {}
    for side, sx in (('left', -1), ('right', 1)):
        pan = ET.SubElement(beam, 'body', name=f'{prefix}_pan_{side}', pos=_vec((sx * d.arm, 0, 0)))
        ET.SubElement(pan, 'joint', name=f'{prefix}_pan_{side}_hinge', type='hinge', axis='0 1 0',
                      range=_vec((-math.radians(d.pan_swing_deg), math.radians(d.pan_swing_deg))), limited='true',
                      damping=f'{d.pan_damping:.9g}', armature='0.0001')
        key = f'pan_{side}'
        r = d.rod_radius
        floor_z = -d.hanger_height
        rim_top = floor_z + d.rim_height
        geoms[key].append(_pair(pan, f'{prefix}_{key}_arm', 'capsule', None, (r, .01), material=mats['pan'],
                                mass=d.mass_rod, friction=d.friction_metal, fromto=(0, .014, 0, 0, d.hanger_back, 0)))
        geoms[key].append(_pair(pan, f'{prefix}_{key}_spar', 'capsule', None, (r, .01), material=mats['pan'],
                                mass=d.mass_rod, friction=d.friction_metal,
                                fromto=(-d.hanger_span, d.hanger_back, 0, d.hanger_span, d.hanger_back, 0)))
        for rod, rx in (('rod_left', -d.hanger_span), ('rod_right', d.hanger_span)):
            geoms[key].append(_pair(pan, f'{prefix}_{key}_{rod}', 'capsule', None, (r, .01), material=mats['pan'],
                                    mass=d.mass_rod, friction=d.friction_metal,
                                    fromto=(rx, d.hanger_back, 0, rx, d.hanger_back, rim_top - .004)))
        geoms[key].append(_pair(pan, f'{prefix}_{key}_floor', 'box', (0, 0, floor_z - d.pan_half[2]), d.pan_half,
                                material=mats['pan'], mass=d.mass_pan_floor, friction=d.friction_pan))
        hx, hy = d.pan_half[0], d.pan_half[1]
        t, hz = d.rim_thickness / 2, d.rim_height / 2
        rims = {'rim_front': ((0, -hy + t, floor_z + hz), (hx, t, hz)),
                'rim_back': ((0, hy - t, floor_z + hz), (hx, t, hz)),
                'rim_left': ((-hx + t, 0, floor_z + hz), (t, hy - 2 * t, hz)),
                'rim_right': ((hx - t, 0, floor_z + hz), (t, hy - 2 * t, hz))}
        for rim, (pos, half) in rims.items():
            geoms[key].append(_pair(pan, f'{prefix}_{key}_{rim}', 'box', pos, half, material=mats['pan'],
                                    mass=d.mass_rim, friction=d.friction_pan))
        ET.SubElement(pan, 'site', name=f'{prefix}_{key}_floor_site', pos=_vec((0, 0, floor_z)), size='0.002',
                      rgba='0 0 0 0', group='5')
        pans[side] = dict(body=f'{prefix}_pan_{side}', joint=f'{prefix}_pan_{side}_hinge',
                          floor_site=f'{prefix}_{key}_floor_site', hinge_local=[sx * d.arm, 0., d.pivot_z],
                          floor_local_z=floor_z, inner_half_xy=list(d.pan_inner_half), rim_height=d.rim_height)
    return dict(prefix=prefix, design=asdict(d), beam_joint=f'{prefix}_beam_hinge', beam_body=f'{prefix}_beam',
                base_body=f'{prefix}_base', pointer_tip_site=f'{prefix}_pointer_tip', pans=pans, geoms=geoms,
                ticks=ticks, stop_deg=d.stop_deg, pan_floor_z_level=d.pan_floor_z, max_height=d.max_height,
                half_width=d.half_width, pan_capacity_hint='three 7 x 10 x 5 cm parcels side by side along x',
                sign_convention='positive beam angle lowers the +x (right) pan')
