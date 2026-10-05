"""Purpose-built two-pan beam balance for RoboQuest (MJCF builder, no simulator imports).

RoboCasa has no weighing instrument, so the balance is a purpose-built prop
dressed with RoboCasa textures. See :mod:`.builder` for the geometry and the
tuned joint parameters, :mod:`.answer_box` for the v1 column answer box, and
``docs/handoffs/roboquest-odd-parcel.md`` for the measurements behind them.
"""
from roboquest.balance.answer_box import answer_box, interior_xy
from roboquest.balance.builder import METAL_TEXTURES, PLATE_TEXTURES, TEXTURE_KEYS, BalanceDesign, DEFAULT_DESIGN, ROBOCASA_TEXTURES, add_balance, beam_stiffness_for, sample_textures, textures_are_known, tilt_for_mass_difference, with_textures

__all__ = ['BalanceDesign', 'DEFAULT_DESIGN', 'METAL_TEXTURES', 'PLATE_TEXTURES', 'ROBOCASA_TEXTURES',
           'TEXTURE_KEYS', 'add_balance', 'answer_box', 'beam_stiffness_for', 'interior_xy',
           'sample_textures', 'textures_are_known', 'tilt_for_mass_difference', 'with_textures']
