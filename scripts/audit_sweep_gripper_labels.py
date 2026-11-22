"""Audit right-gripper commands in raw and converted RTC sweeping demonstrations."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import h5py
import numpy as np
import pyarrow.parquet as pq


def summarize(label, episodes, tail_frames):
    print(f"\n{label}: {len(episodes)} episodes")
    print("episode frames ever_closed reopened_after_close final_open tail_open final_value")
    closed_count = reopened_count = final_count = 0
    for name, values in episodes:
        values = np.asarray(values, dtype=float)
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
            raise ValueError(f"{label}/{name}: invalid gripper sequence")
        closed = np.flatnonzero(values < 0.2)
        reopened = bool(len(closed) and np.any(values[closed[0] + 1:] > 0.8))
        final_open = bool(values[-1] > 0.8)
        tail = values[-tail_frames:]
        closed_count += bool(len(closed))
        reopened_count += reopened
        final_count += final_open
        print(f"{name} {len(values)} {bool(len(closed))} {reopened} "
              f"{final_open} {np.count_nonzero(tail > 0.8)}/{len(tail)} {values[-1]:.5f}")
    print(f"{label} SUMMARY episodes={len(episodes)} closed={closed_count} "
          f"reopened={reopened_count} final_open={final_count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--lerobot", type=Path, required=True)
    parser.add_argument("--tail-frames", type=int, default=50)
    args = parser.parse_args()
    if args.tail_frames < 1:
        parser.error("--tail-frames must be positive")

    raw_paths = sorted(args.raw.glob("episode_*.hdf5"))
    if not raw_paths:
        parser.error(f"No HDF5 episodes in {args.raw}")
    raw = []
    for path in raw_paths:
        with h5py.File(path, "r") as episode:
            raw.append((path.stem, episode["joint_action/right_gripper"][:].reshape(-1)))
    summarize("RAW", raw, args.tail_frames)

    prompts = set()
    for path in (args.raw.parent / "instruction").glob("episode_*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        prompts.update(str(prompt).strip().casefold() for prompt in data.get("seen", []))

    tasks = pq.read_table(args.lerobot / "meta/tasks.parquet").to_pandas()
    if "task" in tasks.columns:
        names = tasks["task"]
    elif "__index_level_0__" in tasks.columns:
        names = tasks["__index_level_0__"]
    else:
        names = tasks.index
    sweep_pattern = re.compile(r"\bsweep\w*\b|\b(?:target|collection|marked) (?:area|zone)\b")
    sweep_ids = {
        int(idx) for idx, name in zip(tasks["task_index"], names)
        if str(name).strip().casefold() in prompts
        or sweep_pattern.search(str(name).casefold())
    }
    if not sweep_ids:
        raise ValueError("No sweeping task IDs found in LeRobot metadata")

    frames = defaultdict(list)
    for path in sorted((args.lerobot / "data").rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        columns = ["episode_index", "frame_index", "task_index", "action"]
        for batch in parquet.iter_batches(columns=columns):
            data = batch.to_pydict()
            for episode, frame, task, action in zip(*(data[k] for k in columns)):
                if task in sweep_ids:
                    if len(action) != 14:
                        raise ValueError(f"Expected 14 action dimensions, got {len(action)} in {path}")
                    frames[episode].append((frame, action[-1]))
    if not frames:
        raise ValueError(f"No label rows for sweeping task IDs {sorted(sweep_ids)}")
    converted = [(str(episode), [value for _, value in sorted(rows)])
                 for episode, rows in sorted(frames.items())]
    summarize("LEROBOT", converted, args.tail_frames)
    if len(raw) != len(converted):
        raise ValueError("Raw and converted episode counts differ; check task matching")


if __name__ == "__main__":
    main()
