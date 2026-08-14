from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch
import tqdm

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.policies import policy_config
from openpi.training import config as _config


CHECKPOINT_DIR = Path("/cc/openpi/checkpoints/pi05_grasp_200_standard_lora/pi05_grasp_200lora_lora/19999")
DATASET_ROOT = Path("/cc/openpi/grasp_200")
SPLITS_PATH = Path("/cc/openpi/grasp_200lora_splits/splits.json")
OUTPUT_DIR = Path("/cc/openpi/outputs/grasp_open_loop_eval_19999")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grasp checkpoint in open loop on the dataset split.")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sample-stride", type=int, default=30)
    parser.add_argument("--max-samples", type=int, default=40)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


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


def load_prompt_map() -> dict[int, str]:
    tasks_path = DATASET_ROOT / "meta" / "tasks.jsonl"
    mapping = {}
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        mapping[int(item["task_index"])] = item["task"]
    return mapping


def build_config(split: str):
    train_config = _config.get_config("pi05_grasp_200_standard_lora")
    common_data = dataclasses.replace(
        train_config.data,
        repo_id="grasp_200lora",
        root=str(DATASET_ROOT),
        splits_path=str(SPLITS_PATH),
        assets=dataclasses.replace(
            train_config.data.assets,
            assets_dir="/cc/openpi/assets/pi05_grasp",
            asset_id="grasp_200",
        ),
    )
    eval_data = None
    if train_config.eval_data is not None:
        eval_data = dataclasses.replace(
            train_config.eval_data,
            repo_id="grasp_200lora",
            root=str(DATASET_ROOT),
            splits_path=str(SPLITS_PATH),
            assets=dataclasses.replace(
                train_config.eval_data.assets,
                assets_dir="/cc/openpi/assets/pi05_grasp",
                asset_id="grasp_200",
            ),
        )
    train_config = dataclasses.replace(train_config, data=common_data, eval_data=eval_data)
    data_factory = common_data if split == "train" or eval_data is None else eval_data
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    policy = policy_config.create_trained_policy(
        train_config,
        CHECKPOINT_DIR,
        repack_transforms=data_config.repack_transforms,
        sample_kwargs={"num_steps": 10},
        default_prompt=None,
        norm_stats=None,
    )
    return train_config, data_config, policy


def main() -> None:
    os.environ.setdefault("OPENPI_DATA_HOME", "/cc/openpi")
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config, data_config, policy = build_config(args.split)
    prompt_map = load_prompt_map()

    metadata = lerobot_dataset.LeRobotDatasetMetadata("grasp_200lora", root=DATASET_ROOT)
    horizon = train_config.model.action_horizon
    action_dim = 7
    delta_timestamps = {"action": [t / metadata.fps for t in range(horizon)]}
    dataset = lerobot_dataset.LeRobotDataset(
        "grasp_200lora",
        root=DATASET_ROOT,
        episodes=None if data_config.episodes is None else list(data_config.episodes),
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    indices = list(range(0, len(dataset), max(1, args.sample_stride)))
    if args.max_samples is not None:
        indices = indices[: args.max_samples]

    abs_per_dim = np.zeros(action_dim, dtype=np.float64)
    sq_per_dim = np.zeros(action_dim, dtype=np.float64)
    count_per_dim = np.zeros(action_dim, dtype=np.float64)
    abs_per_horizon = np.zeros(horizon, dtype=np.float64)
    sq_per_horizon = np.zeros(horizon, dtype=np.float64)
    count_per_horizon = np.zeros(horizon, dtype=np.float64)
    abs_total = 0.0
    sq_total = 0.0
    count_total = 0.0
    episode_ids: list[int] = []

    for idx in tqdm.tqdm(indices, desc=f"open-loop {args.split}"):
        sample = to_numpy(dataset[idx])
        gt_action = np.asarray(sample["action"][:horizon, :action_dim], dtype=np.float32)
        valid_mask = np.ones(gt_action.shape[:1], dtype=bool)
        if "action_is_pad" in sample:
            valid_mask = ~np.asarray(sample["action_is_pad"][:horizon], dtype=bool).reshape(-1)

        task_index = int(sample.get("task_index", 0))
        prompt = sample.get("task") or prompt_map.get(task_index)
        obs = {
            "observation.images.front": sample["observation.images.front"],
            "observation.images.wrist": sample["observation.images.wrist"],
            "observation.state": np.asarray(sample["observation.state"], dtype=np.float32),
            "action": np.zeros((horizon, train_config.model.action_dim), dtype=np.float32),
            "prompt": prompt,
        }

        pred_action = np.asarray(policy.infer(obs)["actions"][:horizon, :action_dim], dtype=np.float32)
        error = pred_action - gt_action
        abs_error = np.abs(error)
        sq_error = np.square(error)
        mask = valid_mask[:, None].astype(np.float64)

        abs_per_dim += (abs_error * mask).sum(axis=0)
        sq_per_dim += (sq_error * mask).sum(axis=0)
        count_per_dim += mask.sum(axis=0)
        abs_per_horizon += (abs_error * mask).sum(axis=1)
        sq_per_horizon += (sq_error * mask).sum(axis=1)
        count_per_horizon += valid_mask.astype(np.float64) * action_dim
        abs_total += float((abs_error * mask).sum())
        sq_total += float((sq_error * mask).sum())
        count_total += float(mask.sum() * action_dim)
        episode_ids.append(int(sample["episode_index"]))

    mae_per_dim = abs_per_dim / np.maximum(count_per_dim, 1)
    rmse_per_dim = np.sqrt(sq_per_dim / np.maximum(count_per_dim, 1))
    mae_per_horizon = abs_per_horizon / np.maximum(count_per_horizon, 1)
    rmse_per_horizon = np.sqrt(sq_per_horizon / np.maximum(count_per_horizon, 1))
    joint_names = [f"joint_{i}" for i in range(1, 7)] + ["gripper"]

    report = {
        "checkpoint_dir": str(CHECKPOINT_DIR),
        "split": args.split,
        "num_observations": len(indices),
        "sample_stride": args.sample_stride,
        "episodes_sampled": episode_ids,
        "mae_overall": float(abs_total / max(count_total, 1.0)),
        "rmse_overall": float(np.sqrt(sq_total / max(count_total, 1.0))),
        "mae_per_joint": {name: float(mae_per_dim[i]) for i, name in enumerate(joint_names)},
        "rmse_per_joint": {name: float(rmse_per_dim[i]) for i, name in enumerate(joint_names)},
        "mae_per_horizon": [float(v) for v in mae_per_horizon],
        "rmse_per_horizon": [float(v) for v in rmse_per_horizon],
    }

    (output_dir / f"open_loop_{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {output_dir}")


if __name__ == "__main__":
    main()
