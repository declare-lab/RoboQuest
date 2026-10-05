"""Reduced RoboCasa work area using its native drawer and PandaOmron.

The clue is a colored physical insert attached inside the moving drawer. Only
its color changes between paired variants; masses, geometry and RNG do not.
"""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.kitchen.atomic.kitchen_drawer import OpenDrawer
from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.models.scenes.kitchen_arena import KitchenArena
from robocasa.utils import camera_utils as CamUtils
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import array_to_string

CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
COLORS = {"red": (0.85, 0.035, 0.025, 1), "blue": (0.025, 0.08, 0.85, 1)}
INSTRUCTION = (
    "Open the drawer and inspect the colored insert inside. Move only the "
    "matching colored cube onto the green destination. Release it and leave "
    "the other cube outside the destination."
)


def scene_config():
    # Texture shipped in the source checkout; no replacement downloaded assets.
    tex = str(Path(__file__).resolve().parents[2] / "assets/textures/source_marble.png")
    layout = {
        "work": {
            "fixtures": [
                dict(
                    name="counter",
                    type="counter",
                    size=[1.4, 0.5, 0.9],
                    pos=[-0.10, 0, 0.45],
                    top_texture=tex,
                    base_texture=tex,
                    base_color=[0.45, 0.48, 0.5, 1],
                    overhang=0,
                    hollow=[False, True],
                ),
                dict(
                    name="drawer",
                    type="drawer",
                    size=[0.32, 0.36, 0.18],
                    pos=[0.36, -0.07, 0.99],
                    texture=tex,
                    panel_type="slab",
                    handle_type="bar",
                    handle_config={"texture": tex},
                ),
                dict(
                    name="floor",
                    type="floor",
                    size=[2, 2, 0.02],
                    pos=[0, -0.7, 0],
                    texture=tex,
                ),
            ]
        }
    }
    return layout, {"cabinet": "default", "counter": "default", "floor": "default"}


class HiddenClueDrawer(OpenDrawer):
    """Native OpenDrawer mechanism, deterministic custom scene, conditional goal."""

    instruction = INSTRUCTION

    def __init__(self, clue="red", **kwargs):
        if clue not in COLORS:
            raise ValueError("clue must be red or blue")
        self._clue = clue
        self.destination = np.array([-0.48, -0.12, 0.904])
        super().__init__(drawer_id="drawer_work", **kwargs)

    def _setup_model(self):
        self.layout_id = "active_bench_reduced_v1"
        self.style_id = "source_marble_v1"
        self._curr_gen_fixtures = None
        for robot in self.robots:
            robot.init_qpos = (
                -0.01612974,
                -1.03446714,
                -0.02397936,
                -2.27550888,
                0.03932365,
                1.51639493,
                0.69615947,
            )
            robot.init_torso_qpos = np.array([0.0])
        layout, style = scene_config()
        self.mujoco_arena = KitchenArena(
            deepcopy(layout), deepcopy(style), rng=self.rng
        )
        self.mujoco_arena.set_origin([0, 0, 0])
        CamUtils.set_cameras(self)
        self.fixture_cfgs = self.mujoco_arena.get_fixture_cfgs()
        self.fixtures = {cfg["name"]: cfg["model"] for cfg in self.fixture_cfgs}
        # Native sliding drawer; the insert is attached to its inner box.
        drawer = self.fixtures["drawer_work"]
        inner = drawer.get_obj().find(".//body[@name='drawer_work_inner_box']")
        if inner is None:
            raise RuntimeError("Native drawer inner body changed upstream")
        ET.SubElement(
            inner,
            "geom",
            name="hidden_insert",
            type="box",
            pos="0 -0.08 -0.025",
            size="0.085 0.055 0.004",
            rgba=array_to_string(COLORS[self._clue]),
            contype="0",
            conaffinity="0",
            mass="0.001",
            group="1",
        )
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=[r.robot_model for r in self.robots],
            mujoco_objects=list(self.fixtures.values()),
            enable_multiccd=True,
            enable_sleeping_islands=False,
        )
        # Single-sample rendering avoids sub-LSB multisample resolve differences
        # between independent EGL contexts in strict matched-image diagnostics.
        visual = self.model.root.find("visual")
        if visual is None:
            visual = ET.SubElement(self.model.root, "visual")
        quality = visual.find("quality")
        if quality is None:
            quality = ET.SubElement(visual, "quality")
        quality.set("offsamples", "0")
        ET.SubElement(
            self.model.worldbody,
            "geom",
            name="destination",
            type="box",
            pos=array_to_string(self.destination),
            size="0.07 0.07 0.004",
            rgba="0.12 0.65 0.15 1",
            contype="1",
            conaffinity="1",
            group="1",
        )

    def _setup_kitchen_references(self):
        Kitchen._setup_kitchen_references(self)
        self.drawer = self.register_fixture_ref("drawer", dict(id="drawer_work"))
        self.init_robot_base_ref = self.drawer
        self.drawer_side = "right"

    def _place_robot(self):
        self.init_robot_base_pos_anchor = np.array([0.0, -0.72, 0.0])
        self.init_robot_base_ori_anchor = np.array([0.0, 0.0, np.pi / 2])
        self.drawer_side = "right"
        return True

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        for color in COLORS:
            obj = BoxObject(
                name=f"{color}_cube",
                size=[0.022] * 3,
                rgba=COLORS[color],
                density=500,
                friction=[1.0, 0.005, 0.0001],
            )
            self.objects[obj.name] = obj
            self.model.merge_objects([obj])

    def _reset_internal(self):
        super()._reset_internal()
        # RNG use is identical for both clue variants.
        jitter = self.rng.uniform(-0.012, 0.012, size=(2, 2))
        for i, color in enumerate(COLORS):
            pos = np.array([-0.12 - i * 0.16, -0.14, 0.924])
            pos[:2] += jitter[i]
            self.sim.data.set_joint_qpos(
                self.objects[f"{color}_cube"].joints[0], np.r_[pos, [1, 0, 0, 0]]
            )
        self.sim.forward()

    def get_ep_meta(self):
        # Privileged upstream metadata stays within the evaluator.
        meta = Kitchen.get_ep_meta(self)
        meta["lang"] = self.instruction
        return meta

    def _check_success(self):
        # See adapter score() for the stricter released/stable terminal criterion.
        if not hasattr(self, "obj_body_id") or not self.obj_body_id:
            return False
        positions = {
            c: self.sim.data.body_xpos[self.obj_body_id[f"{c}_cube"]] for c in COLORS
        }
        inside = {
            c: np.all(np.abs(p[:2] - self.destination[:2]) < 0.044)
            and abs(p[2] - (self.destination[2] + 0.026)) < 0.012
            for c, p in positions.items()
        }
        return bool(
            inside[self._clue] and not inside["blue" if self._clue == "red" else "red"]
        )
