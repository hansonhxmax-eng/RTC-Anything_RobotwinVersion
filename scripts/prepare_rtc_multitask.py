"""Validate and convert the three RTC simulation suites into one LeRobot dataset.

This does not synthesize demonstrations. Each task must first have a RoboTwin
environment and successful expert rollouts collected with collect_data.py.
The upstream converter deletes an existing target dataset, so this wrapper
refuses to run when that directory already exists.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess


TASKS = ("rtc_clothes_folding", "rtc_object_sweeping", "rtc_table_cleaning")


def check_sources(root: Path, minimum_episodes: int) -> dict[str, int]:
    counts = {}
    missing = []
    for task in TASKS:
        environment = root / "envs" / f"{task}.py"
        folder = root / "data" / "demo_clean" / task / "aloha_agilex"
        data = folder / "data"
        instructions = folder / "instruction"
        episodes = sorted(data.glob("episode_*.hdf5"))
        if not environment.is_file():
            missing.append(f"{task}: environment missing: {environment}")
        if len(episodes) < minimum_episodes:
            missing.append(f"{task}: {len(episodes)}/{minimum_episodes} HDF5 demonstrations in {data}")
        else:
            # The training task prompt must describe the corresponding suite.
            # All selected episodes need an instruction rather than an empty
            # generic prompt; the converter samples from their HDF5 metadata.
            for episode in episodes[:minimum_episodes]:
                instruction = instructions / f"{episode.stem}.json"
                if not instruction.is_file():
                    missing.append(f"{task}: instruction missing: {instruction}")
                    break
                try:
                    prompts = json.loads(instruction.read_text(encoding="utf-8"))
                    if not prompts.get("seen"):
                        missing.append(f"{task}: empty seen instructions: {instruction}")
                        break
                except (ValueError, OSError) as exc:
                    missing.append(f"{task}: cannot read {instruction}: {exc}")
                    break
        counts[task] = len(episodes)
    if missing:
        raise ValueError("Three-suite dataset is not ready:\n" + "\n".join(missing))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="rtc_three_suites_clean_v1")
    parser.add_argument("--episodes-per-task", type=int, default=50)
    args = parser.parse_args()
    if args.episodes_per_task < 1:
        parser.error("--episodes-per-task must be positive")
    if not args.repo_id.replace("_", "").isalnum():
        parser.error("--repo-id must contain only letters, numbers and underscores")

    root = args.robotwin_root.resolve()
    converter = root / "XPolicyLab" / "scripts" / "transform_lerobot_v30_format.py"
    python = root / "XPolicyLab" / "policy" / "Pi_0" / "openpi" / ".venv" / "bin" / "python"
    if not converter.is_file() or not python.is_file():
        parser.error("XPolicyLab v3 converter or OpenPI Python 3.11 is missing")
    dataset_home = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache/huggingface/lerobot"))
    destination = dataset_home / args.repo_id
    if destination.exists():
        parser.error(f"Refusing to overwrite existing dataset: {destination}")
    try:
        counts = check_sources(root, args.episodes_per_task)
    except ValueError as exc:
        parser.error(str(exc))
    print("Validated demonstrations:", counts, flush=True)
    command = [
        str(python), str(converter),
        *(f"demo_clean.{task}.aloha_agilex" for task in TASKS),
        "--repo_id", args.repo_id, "--max_episode", str(args.episodes_per_task),
    ]
    subprocess.run(command, cwd=root, check=True)
    metadata = destination / "meta" / "info.json"
    if not metadata.is_file():
        raise RuntimeError(f"Converted dataset metadata missing: {metadata}")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    expected = len(TASKS) * args.episodes_per_task
    if int(info.get("total_episodes", -1)) != expected:
        raise RuntimeError(f"Dataset contains {info.get('total_episodes')} episodes; expected {expected}")
    print(f"Ready for normalization and training: {destination} ({expected} episodes)")


if __name__ == "__main__":
    main()
