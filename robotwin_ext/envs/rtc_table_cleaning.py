"""Approximate RTC table cleaning in RoboTwin's rigid-body simulator.

The towel is a thin rigid body; milk is a grid of visible, removable patches.
The success condition checks object placement and actual towel coverage.
This is a simulation proxy, not liquid or fabric physics.
"""

from ._base_task import Base_Task
from .utils import Action, ArmTag, create_box, create_visual_box
import numpy as np
import sapien


class rtc_table_cleaning(Base_Task):
    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)

    def load_actors(self):
        # A painted collection zone is a rigid-simulator proxy for the
        # physical waste bucket in the original real-robot task.
        self.tray = create_visual_box(
            self, sapien.Pose([-0.35, 0, 0.7405]),
            half_size=[0.12, 0.17, 0.0005],
            color=(0.12, 0.65, 0.24), name="rtc_collection_zone",
        )
        self.items = []
        for index, (x, y) in enumerate(((-0.25, -0.17), (-0.14, -0.18), (-0.08, -0.04))):
            obj = create_box(
                self, sapien.Pose([x, y, 0.76]),
                half_size=[0.019, 0.019, 0.019],
                color=(0.2 + 0.2 * index, 0.4, 0.8 - 0.2 * index),
                name=f"rtc_clean_item_{index}",
            )
            self.items.append((obj, ArmTag("left")))
            self.add_prohibit_area(obj, padding=0.05)

        self.towel = create_box(
            self, sapien.Pose([0.24, -0.17, 0.76]),
            half_size=[0.025, 0.025, 0.019], color=(0.1, 0.45, 0.85),
            name="rtc_towel_proxy",
        )
        self.add_prohibit_area(self.towel, padding=0.07)
        self.milk = []
        for index, (x, y) in enumerate(
            ( (0.12, -0.06), (0.19, -0.06), (0.26, -0.06),
              (0.12, 0.01), (0.19, 0.01), (0.26, 0.01) )
        ):
            marker = create_visual_box(
                self, sapien.Pose([x, y, 0.741]),
                half_size=[0.025, 0.025, 0.001],
                color=(0.94, 0.95, 0.8), name=f"rtc_milk_{index}",
            )
            self.milk.append({"xy": np.array([x, y]), "entity": marker, "clean": False})

    def play_once(self):
        for obj, arm in self.items:
            self.move(self.grasp_actor(obj, arm_tag=arm, pre_grasp_dis=0.08))
            self.move(self.move_by_displacement(arm, z=0.12))
            # Drop into the marked collection zone from above.
            self.move((arm, [Action("left", "move", [-0.35, -0.1, 0.93,
                                                   0.65, -0.25, 0.25, 0.65])]))
            self.move(self.open_gripper(arm))
            self.move(self.back_to_origin(arm))

        arm = ArmTag("right")
        self.move(self.grasp_actor(self.towel, arm_tag=arm, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm, z=0.10))
        # Move through free space before lowering to the tabletop; a direct
        # diagonal motion at table height is blocked by the robot geometry.
        self.move(self.move_by_displacement(arm, x=-0.12, y=0.11))
        self.move(self.move_by_displacement(arm, z=-0.10))
        # Each marker must be visited at table height with the towel in hand.
        # Keep the grasp orientation and move the end effector by the measured
        # displacement from towel center to the selected patch.
        for spot in self.milk:
            towel_position = self.towel.get_pose().p.copy()
            displacement = spot["xy"] - towel_position[:2]
            self.move(self.move_by_displacement(
                arm, x=float(displacement[0]), y=float(displacement[1]),
                z=float(0.76 - towel_position[2]),
            ))
            self.check_success()
        self.move(self.open_gripper(arm))
        self.move(self.back_to_origin(arm))
        self.info["info"] = {"{A}": "three blocks", "{B}": "rigid towel proxy"}
        return self.info

    def check_success(self):
        towel = self.towel.get_pose().p
        held = not self.robot.is_right_gripper_open()
        for spot in self.milk:
            if not spot["clean"] and held and 0.74 <= towel[2] <= 0.80 and np.linalg.norm(towel[:2] - spot["xy"]) < 0.04:
                spot["clean"] = True
                spot["entity"].set_pose(sapien.Pose([3, 3, -1]))
        objects_collected = all(
            abs(obj.get_pose().p[0] + 0.35) < 0.12
            and abs(obj.get_pose().p[1]) < 0.17
            and 0.74 < obj.get_pose().p[2] < 0.86
            for obj, _ in self.items
        )
        return objects_collected and all(spot["clean"] for spot in self.milk)
