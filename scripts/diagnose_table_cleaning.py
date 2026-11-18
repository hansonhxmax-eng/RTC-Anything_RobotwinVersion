"""Run exactly one table-cleaning expert seed and report its failing action.

Run from the RoboTwin root, with the same environment as collect_data.py.
This is a diagnostic and deliberately does not save demonstrations.
"""

import argparse
import sys
from pathlib import Path


class StageFailure(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, default=Path("/workspace/RoboTwin"))
    parser.add_argument("--task-config", default="rtc_table_smoke_v2")
    parser.add_argument("--seed", type=int, default=0)
    opts = parser.parse_args()
    root = opts.robotwin_root.resolve()
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    import os

    os.chdir(root)
    from test_render import Sapien_TEST
    import collect_data

    Sapien_TEST()

    def run_one(task, args):
        args["need_plan"] = True
        count = 0
        try:
            task.setup_demo(now_ep_num=0, seed=opts.seed, **args)
            original_move = task.move

            def tracked_move(actions, *move_args, **move_kwargs):
                nonlocal count
                count += 1
                print(f"[MOVE {count:02d}] begin", flush=True)
                result = original_move(actions, *move_args, **move_kwargs)
                print(
                    f"[MOVE {count:02d}] result={result!r} "
                    f"plan_success={task.plan_success}",
                    flush=True,
                )
                if not task.plan_success:
                    raise StageFailure(f"planning failed at move {count}")
                return result

            task.move = tracked_move
            task.play_once()
            print(f"[FINAL] plan_success={task.plan_success}", flush=True)
            for index, (obj, _) in enumerate(task.items):
                print(f"[ITEM {index}] xyz={obj.get_pose().p.tolist()}", flush=True)
            print(f"[TOWEL] xyz={task.towel.get_pose().p.tolist()}", flush=True)
            print(f"[MILK] clean={[spot['clean'] for spot in task.milk]}", flush=True)
            print(f"[FINAL] check_success={task.check_success()}", flush=True)
            print(f"[FINAL] planned_joints_legal={task.planned_joints_legal()}", flush=True)
        except StageFailure as exc:
            print(f"[DIAGNOSIS] {exc}", flush=True)
        finally:
            if hasattr(task, "scene"):
                task.close_env()

    collect_data.run = run_one
    collect_data.main("rtc_table_cleaning", opts.task_config)


if __name__ == "__main__":
    main()
