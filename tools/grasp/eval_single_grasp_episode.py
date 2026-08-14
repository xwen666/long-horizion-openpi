from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Force CPU inference for one-off evaluation to avoid JAX restoring the model on all visible GPUs.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import dataclasses
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.policies import policy_config
from openpi.training import config as _config


CHECKPOINT_DIR = Path("/cc/openpi/checkpoints/pi05_grasp_200_strict_lora/pi05_grasp_200lora_lora/19999")
DATASET_ROOT = Path("/cc/openpi/grasp_200")
SPLITS_PATH = Path("/cc/openpi/grasp_200lora_splits/splits.json")
OUTPUT_DIR = Path("/cc/openpi/outputs/grasp_eval_19999")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one grasp episode and plot predicted vs ground-truth actions.")
    parser.add_argument("--episode-id", type=int, default=None, help="Episode id to evaluate. Defaults to the first test episode.")
    parser.add_argument("--frame-idx", type=int, default=None, help="Frame index within the episode. Defaults to an automatic choice.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory to write plots and metrics.")
    return parser.parse_args()


def load_episode_dataframe(episode_id: int) -> pd.DataFrame:
    episode_path = DATASET_ROOT / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
    return pq.read_table(episode_path).to_pandas()


def load_prompt_map() -> dict[int, str]:
    tasks_path = DATASET_ROOT / "meta" / "tasks.jsonl"
    mapping = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        mapping[int(item["task_index"])] = item["task"]
    return mapping


def choose_frame(df: pd.DataFrame, horizon: int) -> int:
    max_start = len(df) - horizon
    if max_start <= 0:
        return 0
    return min(10, max_start)


def build_observation(df: pd.DataFrame, frame_idx: int, prompt: str) -> dict:
    row = df.iloc[frame_idx]
    return {
        "observation.images.front": row["observation.images.front"],
        "observation.images.wrist": row["observation.images.wrist"],
        "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
        "prompt": prompt,
    }


def build_ground_truth(df: pd.DataFrame, frame_idx: int, horizon: int) -> np.ndarray:
    actions = []
    for idx in range(frame_idx, frame_idx + horizon):
        actions.append(np.asarray(df.iloc[idx]["action"], dtype=np.float32))
    return np.stack(actions, axis=0)


def to_numpy(value):
    if isinstance(value, dict):
        return {k: to_numpy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_numpy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_numpy(v) for v in value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def main() -> None:
    os.environ.setdefault("OPENPI_DATA_HOME", "/cc/openpi")
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    episode_id = int(splits["test"][0] if args.episode_id is None else args.episode_id)

    prompt_map = load_prompt_map()
    df = load_episode_dataframe(episode_id)

    train_config = _config.get_config("pi05_grasp_200_strict_lora")
    train_data = dataclasses.replace(
        train_config.data,
        repo_id="grasp_200lora",
        root=str(DATASET_ROOT),
        splits_path=str(SPLITS_PATH),
        default_prompt="grasp the object",
        assets=dataclasses.replace(
            train_config.data.assets,
            assets_dir="/cc/openpi/outputs/openpi_assets/pi05_grasp_low_mem_finetune",
            asset_id="grasp",
        ),
    )
    eval_data = dataclasses.replace(
        train_config.eval_data,
        repo_id="grasp_200lora",
        root=str(DATASET_ROOT),
        splits_path=str(SPLITS_PATH),
        default_prompt="grasp the object",
        assets=dataclasses.replace(
            train_config.eval_data.assets,
            assets_dir="/cc/openpi/outputs/openpi_assets/pi05_grasp_low_mem_finetune",
            asset_id="grasp",
        ),
    )
    train_config = dataclasses.replace(train_config, data=train_data, eval_data=eval_data)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    policy = policy_config.create_trained_policy(
        train_config,
        CHECKPOINT_DIR,
        repack_transforms=data_config.repack_transforms,
        sample_kwargs={"num_steps": 10},
        default_prompt=None,
        norm_stats=None,
    )

    horizon = train_config.model.action_horizon
    frame_idx = choose_frame(df, horizon) if args.frame_idx is None else args.frame_idx

    task_index = int(df.iloc[frame_idx].get("task_index", 0)) if "task_index" in df.columns else 0
    prompt = prompt_map.get(task_index, "grasp the object")

    metadata = lerobot_dataset.LeRobotDatasetMetadata("grasp_200lora", root=DATASET_ROOT)
    delta_timestamps = {"action": [t / metadata.fps for t in range(horizon)]}
    dataset = lerobot_dataset.LeRobotDataset(
        "grasp_200lora",
        root=DATASET_ROOT,
        episodes=None,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )
    episode_indices = torch.stack(dataset.hf_dataset["episode_index"]).numpy()
    matching = np.where(episode_indices == episode_id)[0]
    if len(matching) == 0:
        raise ValueError(f"Episode {episode_id} not found in test dataset view")
    dataset_idx = int(matching[min(frame_idx, len(matching) - 1)])
    sample = to_numpy(dataset[dataset_idx])

    obs = {
        "observation.images.front": sample["observation.images.front"],
        "observation.images.wrist": sample["observation.images.wrist"],
        "observation.state": np.asarray(sample["observation.state"], dtype=np.float32),
        # RepackTransform expects this key even though inference does not use it.
        "action": np.zeros((horizon, train_config.model.action_dim), dtype=np.float32),
        "prompt": sample.get("task", prompt),
    }
    gt = build_ground_truth(df, frame_idx, horizon)
    pred = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)

    error = pred - gt
    mae_per_dim = np.mean(np.abs(error), axis=0)
    rmse_per_dim = np.sqrt(np.mean(np.square(error), axis=0))
    mae_overall = float(np.mean(np.abs(error)))
    rmse_overall = float(np.sqrt(np.mean(np.square(error))))

    dim_names = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    ]

    report = {
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "episode_id": episode_id,
        "frame_idx": frame_idx,
        "dataset_idx": dataset_idx,
        "prompt": prompt,
        "horizon": horizon,
        "mae_overall": mae_overall,
        "rmse_overall": rmse_overall,
        "mae_per_dim": {name: float(mae_per_dim[i]) for i, name in enumerate(dim_names)},
        "rmse_per_dim": {name: float(rmse_per_dim[i]) for i, name in enumerate(dim_names)},
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    t = np.arange(horizon)
    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(dim_names):
        ax = axes[i]
        ax.plot(t, gt[:, i], label="ground truth", linewidth=2)
        ax.plot(t, pred[:, i], label="predicted", linewidth=2)
        ax.set_title(f"{name} | MAE={mae_per_dim[i]:.4f}")
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    axes[0].legend()
    fig.suptitle(f"Episode {episode_id} frame {frame_idx} | overall MAE={mae_overall:.4f}, RMSE={rmse_overall:.4f}")
    fig.tight_layout()
    fig.savefig(output_dir / "pred_vs_gt.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(dim_names):
        ax = axes[i]
        ax.plot(t, error[:, i], label="pred - gt", linewidth=2)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{name} error | RMSE={rmse_per_dim[i]:.4f}")
        ax.grid(True, alpha=0.3)
    axes[-1].axis("off")
    fig.suptitle(f"Prediction Error | Episode {episode_id} frame {frame_idx}")
    fig.tight_layout()
    fig.savefig(output_dir / "error_curves.png", dpi=160)
    plt.close(fig)

    np.savez(
        output_dir / "arrays.npz",
        ground_truth=gt,
        predicted=pred,
        error=error,
    )

    print(json.dumps(report, indent=2))
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
