#!/usr/bin/env python3
"""Create a small debug planner dataset without splitting episodes."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from planner_dataset_utils import DEFAULT_PLANNER_DATASET, choose_dataset_path, episode_id, load_records


def create_subset(dataset_path: Path, output_path: Path, target_samples: int, seed: int) -> dict:
    records = load_records(dataset_path)
    episodes = defaultdict(list)
    for record in records:
        episodes[episode_id(record)].append(record)
    episode_ids = list(episodes)
    random.Random(seed).shuffle(episode_ids)
    selected = []
    selected_episodes = []
    for current_episode in episode_ids:
        selected_episodes.append(current_episode)
        selected.extend(episodes[current_episode])
        if len(selected) >= target_samples:
            break
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in selected:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    stats = {
        "input": str(dataset_path.resolve()),
        "output": str(output_path.resolve()),
        "requested_samples": target_samples,
        "samples": len(selected),
        "episodes": len(selected_episodes),
        "seed": seed,
        "episode_ids": selected_episodes,
    }
    output_path.with_suffix(output_path.suffix + ".summary.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PLANNER_DATASET)
    parser.add_argument("--output", type=Path, default=Path("robocasa/processed/composite_subtasks/debug_1000.jsonl"))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    create_subset(choose_dataset_path(args.dataset), args.output, args.num_samples, args.seed)


if __name__ == "__main__":
    main()
