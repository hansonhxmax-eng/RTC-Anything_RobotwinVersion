"""Run one RoboTwin expert seed; report stage, objects, and final checks."""

import argparse
import os
import sys
from pathlib import Path


class StageFailure(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, default=Path("/workspace/RoboTwin"))
    parser.add_argument("--task-name", required=True,
                        choices=("rtc_object_sweeping", "rtc_clothes_folding", "rtc_table_cleaning"))
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = args.robotwin_root.resolve()
    sys.path[:0] = [str(root), str(root / "scripts")]
    os.chdir(root)

    import collect_data
    # Do not call test_render.Sapien_TEST: it catches rendering exceptions and
    # exits with code 0, hiding the actual GPU/Vulkan failure from the runner.
    final_result = {"ok": False}

    def report(task):
        for label in ("objects", "items", "brush", "towel", "base_panel", "fold_panel"):
            value = getattr(task, label, None)
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for index, obj in enumerate(values):
                if isinstance(obj, tuple):
                    obj = obj[0]
                print(f"[{label} {index}] xyz={obj.get_pose().p.tolist()}", flush=True)
        if hasattr(task, "milk"):
            print(f"[milk] clean={[spot['clean'] for spot in task.milk]}", flush=True)

    def one_seed(task, config):
        config["need_plan"] = True
        counter = 0
        try:
            task.setup_demo(now_ep_num=0, seed=args.seed, **config)
            original_move = task.move

            def tracked_move(actions, *extras, **kwargs):
                nonlocal counter
                counter += 1
                print(f"[MOVE {counter:02d}] begin", flush=True)
                result = original_move(actions, *extras, **kwargs)
                print(f"[MOVE {counter:02d}] result={result!r} "
                      f"plan_success={task.plan_success}", flush=True)
                if not task.plan_success:
                    raise StageFailure(f"planning failed at move {counter}")
                return result

            task.move = tracked_move
            task.play_once()
            report(task)
            success = bool(task.check_success())
            legal = task.planned_joints_legal()
            print(f"[FINAL] success={success} legal={legal}", flush=True)
            final_result["ok"] = success and legal[0]
        except StageFailure as error:
            print(f"[DIAGNOSIS] {error}", flush=True)
            report(task)
        finally:
            if hasattr(task, "scene"):
                task.close_env()

    collect_data.run = one_seed
    collect_data.main(args.task_name, args.task_config)
    if not final_result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
