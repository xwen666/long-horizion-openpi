"""Compute RoboCasa state/action normalization stats without decoding videos.

Usage:
    .venv/bin/python tools/compute_robocasa_norm_stats.py --config-name pi05_robocasa_status
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as config_lib


def _episode_files(root: pathlib.Path, episode_indices: tuple[int, ...]) -> list[pathlib.Path]:
    wanted = set(episode_indices)
    files = []
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        episode_index = int(path.stem.removeprefix("episode_"))
        if episode_index in wanted:
            files.append(path)
    missing = wanted - {
        int(path.stem.removeprefix("episode_")) for path in files
    }
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} episode parquet files under {root}: {sorted(missing)[:10]}")
    return files


def main(config_name: str = "pi05_robocasa_status") -> None:
    config = config_lib.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    if not data_config.source_roots or not data_config.source_episodes:
        raise ValueError("This script expects a multi-source RoboCasa config with explicit episode splits.")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    total_frames = 0
    total_episodes = 0
    for source_root, episodes in zip(data_config.source_roots, data_config.source_episodes, strict=True):
        root = pathlib.Path(source_root)
        files = _episode_files(root, tuple(episodes))
        total_episodes += len(files)
        for parquet_path in tqdm.tqdm(files, desc=root.parent.parent.name, unit="episode"):
            table = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
            state = np.stack(table["observation.state"].to_numpy())
            actions = np.stack(table["action"].to_numpy())
            if state.ndim != 2 or actions.ndim != 2:
                raise ValueError(f"Unexpected shapes in {parquet_path}: state={state.shape}, actions={actions.shape}")
            stats["state"].update(state)
            stats["actions"].update(actions)
            total_frames += len(state)

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    output_path = config.assets_dirs / data_config.repo_id
    normalize.save(output_path, norm_stats)
    print(f"Wrote {output_path / 'norm_stats.json'}")
    print(f"episodes={total_episodes} frames={total_frames}")
    for key, value in norm_stats.items():
        print(f"{key}: dim={len(value.mean)} mean={value.mean} std={value.std}")


if __name__ == "__main__":
    tyro.cli(main)
