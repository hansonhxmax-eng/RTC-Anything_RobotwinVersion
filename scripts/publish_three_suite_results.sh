#!/usr/bin/env bash
set -euo pipefail

repo=/workspace/RTC-Anything_RobotwinVersion
robotwin=/workspace/RoboTwin
base=/workspace/rtc_three_suite_training
branch=results/rtc-three-suites
cd "$repo"
git fetch origin feature/rtc-three-task-simulation
source_ref=$(git rev-parse FETCH_HEAD)
worktree=$(mktemp -d /tmp/rtc-results.XXXXXX)
git worktree add --detach "$worktree" "$source_ref"
echo "Preparing results in $worktree"
/opt/conda/envs/RoboTwin/bin/python - "$robotwin" "$base" "$worktree" <<'PY'
from pathlib import Path
import re
import shutil
import sys

root, base, worktree = map(Path, sys.argv[1:])
original = base / "latest"
folding = base / "sweep-diagnostic-folding-latest"
sweeping = base / "sweep-async-completion-latest"
out = worktree / "results" / "rtc-three-suites"
out.mkdir(parents=True, exist_ok=True)
ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
rate = re.compile(r"Success rate:\s*(\d+)/(\d+).*?current seed:\s*(\d+)")

def log_text(path):
    return ansi.sub("", path.read_text(encoding="utf-8", errors="replace"))

def outcomes(path):
    return [tuple(map(int, row)) for row in rate.findall(log_text(path))]

def video_dir(log):
    paths = re.findall(r"Data has been saved to\s+(.+?/_result\.txt)", log_text(log))
    if not paths:
        raise SystemExit(f"No video output path in {log}")
    p = Path(paths[-1])
    return (p if p.is_absolute() else root / p).parent

def video_for(directory, episode):
    # Official evaluator uses zero-based filenames. The log increments the
    # completed-episode counter after recording each video.
    p = directory / f"episode{episode - 1}.mp4"
    if not p.is_file() or p.stat().st_size == 0:
        raise SystemExit(f"Missing or empty video: {p}")
    return p

def success_example(log):
    prior = 0
    for successes, episode, seed in outcomes(log):
        if successes > prior:
            return video_for(video_dir(log), episode), seed
        prior = successes
    raise SystemExit(f"No successful episode found in {log}")

tasks = [
    ("rtc_table_cleaning", "Table cleaning", original, original),
    ("rtc_object_sweeping", "Object sweeping", sweeping, sweeping),
    ("rtc_clothes_folding", "Clothes folding", folding, folding),
]
readme = [
    "# RoboTwin π0 + RTC simulation results",
    "",
    "Asynchronous RTC evaluation of the JAX π0 checkpoint at step 29999.",
    "Each task was evaluated for 100 episodes. Results are for the RoboTwin simulation tasks.",
    "",
    "| Task | Successes | Success rate | Successful rollout |",
    "|---|---:|---:|---|",
]
for task, name, log_home, metrics_home in tasks:
    log = log_home / f"{task}_eval100.log"
    rows = outcomes(log)
    if len(rows) != 100 or rows[-1][1] != 100:
        raise SystemExit(f"Expected 100 completed episodes in {log}")
    metrics = metrics_home / "metrics" / task
    dest = out / task
    dest.mkdir(exist_ok=True)
    for filename in ("eval_success.csv", "eval_success.png", "summary.json", "train_loss.csv", "train_loss.png"):
        shutil.copy2(metrics / filename, dest / filename)
    video_log = sweeping / "remaining_sweeps.log" if task == "rtc_object_sweeping" else log
    video, seed = success_example(video_log)
    video_name = f"success_seed_{seed}.mp4"
    shutil.copy2(video, dest / video_name)
    successes = rows[-1][0]
    readme.append(f"| {name} | {successes}/100 | {successes}% | [Seed {seed}]({task}/{video_name}) |")

sweep_log = original / "rtc_object_sweeping_eval100.log"
sweep_dir = root / "eval_result/rtc_object_sweeping/Pi_0/demo_clean/rtc_three_suites_pi0_30k/2026-11-22_08-33-33_808238"
readme.extend(["", "## Sweeping failure examples", ""])
prior = 0
failures = 0
for successes, episode, seed in outcomes(sweep_log):
    if successes == prior:
        name = f"failure_seed_{seed}.mp4"
        shutil.copy2(video_for(sweep_dir, episode), out / "rtc_object_sweeping" / name)
        readme.append(f"- [Seed {seed}](rtc_object_sweeping/{name})")
        failures += 1
        if failures == 2:
            break
    prior = successes
if failures != 2:
    raise SystemExit("Expected two original sweeping failures")
readme.extend(["", "The sweeping score combines the original 18 episodes and 82 continued episodes."])
(out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
PY

git -C "$worktree" add -- results/rtc-three-suites
git -C "$worktree" -c user.name=hansonhxmax-eng -c user.email=hansonhxmax-eng@users.noreply.github.com \
  commit -m "Publish RoboTwin asynchronous three-task evaluation results"
git -C "$worktree" -c credential.helper= -c "credential.helper=cache --timeout=1800" \
  push origin "HEAD:refs/heads/$branch"
git worktree remove "$worktree"
echo "Published results on $branch"
