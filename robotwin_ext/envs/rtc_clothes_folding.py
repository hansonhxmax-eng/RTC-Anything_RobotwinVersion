"""Experimental fold-like rigid-panel proxy for RTC Anything.

RoboTwin's SAPIEN configuration here has no cloth solver. One padded panel
is lifted and placed onto another; this does not reproduce cloth dynamics.
Do not use results as the real-robot clothes-folding success rate.
"""

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import ArmTag, create_box, create_visual_box


class rtc_clothes_folding(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.base_panel = create_box(
            self, sapien.Pose([0.13, -0.04, 0.76]),
            half_size=[0.055, 0.06, 0.019], color=(0.15, 0.32, 0.82),
            is_static=True, name="rtc_garment_fixed_panel",
        )
        self.fold_panel = create_box(
            self, sapien.Pose([0.28, -0.04, 0.76]),
            half_size=[0.055, 0.06, 0.019], color=(0.25, 0.45, 0.93),
            name="rtc_garment_fold_panel",
        )
        self.fold_line = create_visual_box(
            self, sapien.Pose([0.205, -0.04, 0.741]),
            half_size=[0.003, 0.067, 0.001], color=(0.8, 0.8, 0.8),
            name="rtc_fold_line",
        )
        self.add_prohibit_area(self.fold_panel, padding=0.07)

    def play_once(self):
        arm = ArmTag("right")
        self.move(self.grasp_actor(self.fold_panel, arm_tag=arm, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm, z=0.10))
        self.move(self.move_by_displacement(arm, x=-0.15))
        self.move(self.move_by_displacement(arm, z=-0.062))
        self.move(self.open_gripper(arm))
        self.move(self.back_to_origin(arm))
        self.info["info"] = {"{A}": "rigid garment panel", "{B}": "folded panel"}
        return self.info

    def check_success(self):
        base = self.base_panel.get_pose().p
        flap = self.fold_panel.get_pose().p
        return bool(
            np.linalg.norm(flap[:2] - base[:2]) < 0.055
            and 0.025 < flap[2] - base[2] < 0.07
            and self.is_right_gripper_open()
        )
