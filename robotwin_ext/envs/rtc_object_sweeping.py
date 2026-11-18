"""Experimental physical brush sweep of five rigid objects.

The colored collection zone is a dustpan proxy. This prototype verifies
actual rigid-body movement; the full pan-to-basket transfer is still pending.
Do not compare this proxy's score directly with the real-robot task.
"""

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import ArmTag, create_box, create_visual_box


class rtc_object_sweeping(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        self.dustpan_zone = create_visual_box(
            self, sapien.Pose([0.22, 0.18, 0.741]),
            half_size=[0.175, 0.085, 0.001], color=(0.12, 0.7, 0.28),
            name="rtc_dustpan_zone",
        )
        self.objects = []
        for i, x in enumerate((0.10, 0.16, 0.22, 0.28, 0.34)):
            obj = create_box(
                self, sapien.Pose([x, -0.10, 0.759]),
                half_size=[0.017, 0.017, 0.018],
                color=(0.85, 0.16 + 0.1 * i, 0.1),
                name=f"rtc_sweep_object_{i}",
            )
            self.objects.append(obj)

        self.brush = create_box(
            self, sapien.Pose([0.34, -0.19, 0.76]),
            half_size=[0.025, 0.023, 0.019], color=(0.12, 0.32, 0.88),
            name="rtc_brush_proxy",
        )
        self.add_prohibit_area(self.brush, padding=0.07)

    def play_once(self):
        arm = ArmTag("right")
        self.move(self.grasp_actor(self.brush, arm_tag=arm, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm, z=0.10))
        for obj in self.objects:
            target_x = float(obj.get_pose().p[0])
            brush_xyz = self.brush.get_pose().p.copy()
            self.move(self.move_by_displacement(
                arm, x=target_x - float(brush_xyz[0]),
                y=-0.17 - float(brush_xyz[1]),
            ))
            brush_z = float(self.brush.get_pose().p[2])
            self.move(self.move_by_displacement(arm, z=0.76 - brush_z))
            # The object must be moved by contact with the held brush.
            # Short contact motions help localize a planning failure and
            # avoid asking the arm to traverse the entire zone at once.
            for _ in range(3):
                self.move(self.move_by_displacement(arm, y=0.105))
            self.move(self.move_by_displacement(arm, z=0.10))

        self.move(self.open_gripper(arm))
        self.move(self.back_to_origin(arm))
        self.info["info"] = {"{A}": "five swept blocks", "{B}": "brush proxy"}
        return self.info

    def check_success(self):
        return bool(
            all(
                0.04 < obj.get_pose().p[0] < 0.40
                and 0.095 < obj.get_pose().p[1] < 0.27
                and 0.74 < obj.get_pose().p[2] < 0.84
                for obj in self.objects
            )
            and self.is_right_gripper_open()
        )
