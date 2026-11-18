"""Plot training loss and official RoboTwin episode success rates from logs.

The training curve is *training loss*, not held-out validation loss. Episode
success is a separate simulation metric and should never be labeled loss.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


LOSS = re.compile(r"Step\s+(\d+):[^\n]*?\bloss=([0-9.eE+-]+)")
SUCCESS = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9.]+)%")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_training(text: str) -> list[tuple[int, float]]:
    by_step = {int(step): float(loss) for step, loss in LOSS.findall(text)}
    return sorted(by_step.items())


def parse_evaluation(text: str) -> list[tuple[int, int, float]]:
    plain_text = ANSI_ESCAPE.sub("", text)
    return [(int(total), int(wins), float(rate)) for wins, total, rate in SUCCESS.findall(plain_text)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--eval-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    train = parse_training(args.train_log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n"))
    evals = parse_evaluation(args.eval_log.read_text(encoding="utf-8", errors="replace").replace("\r", "\n"))
    if not train or not evals:
        parser.error("Logs must contain 'Step N: loss=...' and 'Success rate: X/Y => Z%' lines")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_loss.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "train_loss"))
        writer.writerows(train)
    with (args.output_dir / "eval_success.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("episode", "success_count", "cumulative_success_percent"))
        writer.writerows(evals)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "last_training_step_logged": train[-1][0],
        "episodes": evals[-1][0],
        "successes": evals[-1][1],
        "success_rate_percent": evals[-1][2],
        "training_loss_is_validation_loss": False,
    }, indent=2) + "\n", encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, x, y, xlabel, ylabel in (
        ("train_loss.png", [v[0] for v in train], [v[1] for v in train], "Training step", "Training loss"),
        ("eval_success.png", [v[0] for v in evals], [v[2] for v in evals], "Episode", "Cumulative success rate (%)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / name, dpi=160)
        plt.close(fig)
    print(f"Saved curves and CSV to {args.output_dir}; final success {evals[-1][1]}/{evals[-1][0]}")


if __name__ == "__main__":
    main()
