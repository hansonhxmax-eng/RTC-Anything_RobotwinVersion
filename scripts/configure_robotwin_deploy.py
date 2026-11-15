"""Point XPolicyLab's π0 policy server at the newly trained RoboTwin run."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from prepare_robotwin_pi0 import CONFIG_NAME


def replace_key(source: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?m)^({re.escape(key)}:\s*)[^\n]*$")
    if len(pattern.findall(source)) != 1:
        raise ValueError(f"Expected exactly one {key} key in deploy.yml")
    return pattern.sub(lambda match: match.group(1) + value, source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+", args.run_name) or not re.fullmatch(r"[\w.-]+", args.repo_id):
        parser.error("Run name and repo ID must be simple local names")
    if args.checkpoint_step < 1:
        parser.error("Checkpoint step must be positive")
    policy_dir = args.robotwin_root.resolve() / "XPolicyLab/policy/Pi_0"
    checkpoint = policy_dir / "checkpoints" / args.run_name / str(args.checkpoint_step)
    if not (checkpoint / "params").is_dir() or not (checkpoint / "assets" / args.repo_id / "norm_stats.json").is_file():
        parser.error(f"Checkpoint params or matching normalization assets missing: {checkpoint}")
    path = policy_dir / "deploy.yml"
    original = path.read_text(encoding="utf-8")
    updated = original
    for key, value in (("train_config_name", CONFIG_NAME), ("repo_id", args.repo_id),
                       ("checkpoint_num", str(args.checkpoint_step))):
        updated = replace_key(updated, key, value)
    if original != updated:
        backup = path.with_name("deploy.yml.before_rtc_robotwin")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        path.write_text(updated, encoding="utf-8")
    print(f"Policy server now selects {checkpoint} with dataset assets {args.repo_id}")


if __name__ == "__main__":
    main()
