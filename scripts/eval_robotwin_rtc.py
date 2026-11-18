"""Run RoboTwin's official evaluator with asynchronous RTC policy inference.

Run this with RoboTwin's simulation environment. RoboTwin and its XPolicyLab
submodule remain separate checkouts; no simulator files are patched.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--policy-name", default="Pi_0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--env-cfg-type", default="arx_x5")
    parser.add_argument("--ckpt-setting", required=True)
    parser.add_argument("--action-type", choices=("joint", "ee"), default="joint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-num", type=int, default=1)
    parser.add_argument("--request-interval", type=int, default=8)
    parser.add_argument("--first-action-timeout", type=float, default=60)
    args = parser.parse_args()

    root = args.robotwin_root.resolve()
    if not (root / "scripts/eval_policy_xpolicylab.py").is_file():
        parser.error("--robotwin-root must point to a RoboTwin checkout with XPolicyLab evaluation")
    # RoboTwin imports load relative asset paths during module import.
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from test_render import Sapien_TEST
    import eval_policy_xpolicylab as evaluator
    from robotwin_rtc import RTCModelClient

    original_builder = evaluator.build_policy_client

    def build_rtc_client(usr_args):
        return RTCModelClient(
            original_builder(usr_args),
            request_interval=args.request_interval,
            first_action_timeout=args.first_action_timeout,
        )

    evaluator.build_policy_client = build_rtc_client
    Sapien_TEST()
    evaluator.main({
        "task_name": args.task_name,
        "policy_name": args.policy_name,
        "host": args.host,
        "port": args.port,
        "protocol": "ws",
        "root_dir": str(root),
        "seed": args.seed,
        "task_config": args.task_config,
        "env_cfg_type": args.env_cfg_type,
        "ckpt_setting": args.ckpt_setting,
        "action_type": args.action_type,
        "test_num": args.test_num,
    })


if __name__ == "__main__":
    main()
