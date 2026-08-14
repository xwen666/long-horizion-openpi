"""Fast norm stats computation - reads parquet directly, skips video decoding.

Usage:
    OPENPI_GRASP_REPO_ID=grasp_200 \
    OPENPI_GRASP_DATASET_ROOT=/cc/openpi/grasp_200 \
    OPENPI_GRASP_SPLITS_PATH=/cc/openpi/grasp_splits/splits_200.json \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python scripts/compute_norm_stats_fast.py --config-name pi05_grasp
"""

import json
import os
import pathlib

import numpy as np
import pandas as pd
import tqdm
import tyro

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.transforms as _transforms
from openpi.transforms import DeltaActions, make_bool_mask


def main(config_name: str):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    root = pathlib.Path(data_config.root)

    # Load episode split
    if data_config.splits_path:
        splits_path = pathlib.Path(os.path.expandvars(data_config.splits_path)).expanduser()
        splits = json.loads(splits_path.read_text())
        train_episodes = set(splits["train"])
        print(f"Train episodes: {len(train_episodes)}")
    else:
        train_episodes = None

    # Read all per-episode parquet files
    print("Reading parquet data...")
    data_dir = root / "data/chunk-000"
    files = sorted(data_dir.glob("episode_*.parquet"))

    states = []
    actions = []
    for f in tqdm.tqdm(files, desc="Reading parquets"):
        ep_idx = int(f.stem.split("_")[1])
        if train_episodes is not None and ep_idx not in train_episodes:
            continue
        df = pd.read_parquet(f)

        s = np.stack(df["observation.state"].values)  # (N, 7)
        a = np.stack(df["action"].values)  # (N, 7)

        # Apply DeltaActions if enabled: mask [T,T,T,T,T,T,F] means
        # joints 0-5 become delta relative to current state, gripper stays absolute
        if getattr(data_config, "use_delta_joint_actions", False):
            mask = np.array(make_bool_mask(6, -1))  # (7,) = [T,T,T,T,T,T,F]
            dims = len(mask)
            a[:, :dims] -= np.where(mask, s[:, :dims], 0)

        states.append(s)
        actions.append(a)

    state = np.concatenate(states, axis=0)
    action = np.concatenate(actions, axis=0)
    print(f"Total frames: {len(state)}")
    print(f"State shape: {state.shape}, Action shape: {action.shape}")

    # Compute running stats
    print("Computing stats...")
    stats = {
        "state": normalize.RunningStats(),
        "actions": normalize.RunningStats(),
    }
    batch_size = 4096
    for i in tqdm.tqdm(range(0, len(state), batch_size), desc="Stats"):
        stats["state"].update(state[i:i + batch_size])
        stats["actions"].update(action[i:i + batch_size])

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)

    print("Done!")
    for k in ["state", "actions"]:
        ns = norm_stats[k]
        print(f"\n{k}:")
        print(f"  mean: {ns.mean}")
        print(f"  std:  {ns.std}")
        print(f"  q01:  {ns.q01}")
        print(f"  q99:  {ns.q99}")


if __name__ == "__main__":
    tyro.cli(main)
