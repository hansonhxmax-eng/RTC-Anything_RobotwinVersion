"""Train a new RoboTwin π0 LoRA run; never overwrite an existing checkpoint.

Prepare a *new* local LeRobot dataset and compute norm stats first, as shown
in ROBOTWIN.md. This script supports resuming the same run after interruption.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from prepare_robotwin_pi0 import CONFIG_NAME


def training_command(args: argparse.Namespace, checkpoint_dir: Path) -> list[str]:
    command = ["uv", "run", "scripts/train.py", CONFIG_NAME,
               f"--exp-name={args.run_name}", f"--checkpoint-dir-override={checkpoint_dir}",
               f"--seed={args.seed}"]
    if args.resume:
        command.append("--resume")
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="New local LeRobot dataset ID")
    parser.add_argument("--run-name", required=True, help="New experiment/checkpoint directory name")
    parser.add_argument("--gpus", default="0", help="Comma-separated visible GPU IDs, e.g. 0 or 0,1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume this run from its latest checkpoint")
    args = parser.parse_args()
    root = args.robotwin_root.resolve()
    policy_dir = root / "XPolicyLab/policy/Pi_0"
    openpi_dir = policy_dir / "openpi"
    if not (openpi_dir / "scripts/train.py").is_file():
        parser.error("OpenPI not found; initialize the RoboTwin XPolicyLab submodule")
    if not args.repo_id or "/" in args.repo_id or args.run_name in {".", ".."} or "/" in args.run_name:
        parser.error("Use simple nonempty repo IDs and run names without slashes")
    gpus = args.gpus.split(",")
    if not gpus or any(not item.isdigit() for item in gpus) or len(set(gpus)) != len(gpus):
        parser.error("--gpus must be unique numeric IDs, e.g. 0 or 0,1")
    os.environ["OPENPI_DATA_REPO_ID"] = args.repo_id
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    dataset_root = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
    if not (dataset_root / args.repo_id / "meta/info.json").is_file():
        parser.error(f"LeRobot dataset missing: {dataset_root / args.repo_id}")
    # Both OpenPI training and policy serving load stats from the run's assets.
    norm_stats = openpi_dir / "assets" / CONFIG_NAME / args.repo_id / "norm_stats.json"
    if not norm_stats.is_file():
        parser.error(f"Normalization missing: {norm_stats}; run compute_norm_stats.py first")
    checkpoint_dir = policy_dir / "checkpoints" / args.run_name
    config_path = openpi_dir / "src/openpi/training/config.py"
    if not config_path.is_file() or f'name="{CONFIG_NAME}"' not in config_path.read_text(encoding="utf-8"):
        parser.error("Run prepare_robotwin_pi0.py before training")
    if args.resume:
        if not checkpoint_dir.is_dir() or not any(p.is_dir() and p.name.isdigit() for p in checkpoint_dir.iterdir()):
            parser.error(f"No finalized numeric checkpoint under {checkpoint_dir}; cannot resume")
    elif checkpoint_dir.exists():
        parser.error(f"Checkpoint directory already exists: {checkpoint_dir}; choose a new run name or --resume")
    command = training_command(args, checkpoint_dir)
    print("Run:", " ".join(command), flush=True)
    print("Dataset:", args.repo_id, "GPUs:", args.gpus, flush=True)
    subprocess.run(command, cwd=openpi_dir, env=os.environ.copy(), check=True)


if __name__ == "__main__":
    main()
