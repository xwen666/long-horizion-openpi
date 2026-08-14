#!/usr/bin/env python3
"""Create deterministic episode-level train/validation/test splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from planner_dataset_utils import (
    DEFAULT_PLANNER_DATASET,
    choose_dataset_path,
    episode_id,
    global_task,
    load_records,
)


def split_dataset(
    dataset_path: Path,
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    records = load_records(dataset_path)
    episodes: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        episodes[episode_id(record)].append(record)
    episode_ids = sorted(episodes)
    random.Random(seed).shuffle(episode_ids)
    train_end = int(len(episode_ids) * train_ratio)
    val_end = train_end + int(len(episode_ids) * val_ratio)
    assignments = {
        "train": episode_ids[:train_end],
        "val": episode_ids[train_end:val_end],
        "test": episode_ids[val_end:],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_path": str(dataset_path.resolve()),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "episodes": {},
        "samples": {},
        "tasks": {},
        "episode_overlap": {},
    }
    assigned_sets = {name: set(values) for name, values in assignments.items()}
    for left_name, left in assigned_sets.items():
        for right_name, right in assigned_sets.items():
            if left_name < right_name:
                summary["episode_overlap"][f"{left_name}_{right_name}"] = len(left & right)

    for split_name, split_episode_ids in assignments.items():
        split_records = [record for episode in split_episode_ids for record in episodes[episode]]
        with (output_dir / f"{split_name}.jsonl").open("w") as handle:
            for record in split_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        task_counts = Counter(global_task(record) for record in split_records)
        summary["episodes"][split_name] = len(split_episode_ids)
        summary["samples"][split_name] = len(split_records)
        summary["tasks"][split_name] = dict(task_counts)

    summary["total_episodes"] = len(episode_ids)
    summary["total_samples"] = len(records)
    (output_dir / "split_statistics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote episode-level splits to {output_dir.resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PLANNER_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("robocasa/processed/composite_subtasks/splits"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    split_dataset(
        choose_dataset_path(args.dataset),
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )


if __name__ == "__main__":
    main()
