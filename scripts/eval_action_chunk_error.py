#!/usr/bin/env python
"""Evaluate OpenPI action-chunk prediction error on a LeRobot split."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
import tqdm

import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.policies import policy_config
from openpi.training import config as _config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_name", help="OpenPI config name, e.g. pi05_grasp")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint step dir, e.g. checkpoints/pi05_grasp/run/1000")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--output-dir", default="eval_openpi_action_chunk_error")
    parser.add_argument("--sample-stride", type=int, default=30, help="Evaluate every Nth frame.")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=None, help="Compare only the first N action steps.")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    return parser.parse_args()


def to_numpy(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: to_numpy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_numpy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_numpy(v) for v in value)
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy") and not isinstance(value, np.ndarray):
        return value.numpy()
    return value


def get_data_config(train_config: _config.TrainConfig, split: str) -> _config.DataConfig:
    data_factory = train_config.data
    if split == "test" and train_config.eval_data is not None:
        data_factory = train_config.eval_data
    return data_factory.create(train_config.assets_dirs, train_config.model)


def safe_div(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return numer / np.maximum(denom, 1)


def write_csvs(report: dict[str, Any], output_dir: Path) -> None:
    with (output_dir / "per_joint_error.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["joint", "mae", "rmse"])
        writer.writeheader()
        for joint, mae in report["mae_per_joint"].items():
            writer.writerow({"joint": joint, "mae": mae, "rmse": report["rmse_per_joint"][joint]})

    with (output_dir / "per_horizon_error.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["horizon_step", "mae", "rmse"])
        writer.writeheader()
        for i, (mae, rmse) in enumerate(zip(report["mae_per_horizon"], report["rmse_per_horizon"], strict=True)):
            writer.writerow({"horizon_step": i, "mae": mae, "rmse": rmse})


def build_report(
    *,
    args: argparse.Namespace,
    data_config: _config.DataConfig,
    joint_names: list[str],
    total_samples: int,
    abs_per_dim: np.ndarray,
    sq_per_dim: np.ndarray,
    count_per_dim: np.ndarray,
    abs_per_horizon: np.ndarray,
    sq_per_horizon: np.ndarray,
    count_per_horizon: np.ndarray,
    abs_total: float,
    sq_total: float,
    count_total: float,
) -> dict[str, Any]:
    mae_per_dim = safe_div(abs_per_dim, count_per_dim)
    rmse_per_dim = np.sqrt(safe_div(sq_per_dim, count_per_dim))
    mae_per_horizon = safe_div(abs_per_horizon, count_per_horizon)
    rmse_per_horizon = np.sqrt(safe_div(sq_per_horizon, count_per_horizon))

    return {
        "config_name": args.config_name,
        "checkpoint_dir": str(Path(args.checkpoint_dir).expanduser().resolve()),
        "dataset_repo_id": data_config.repo_id,
        "dataset_root": data_config.root,
        "split": args.split,
        "episodes": list(data_config.episodes) if data_config.episodes is not None else None,
        "sample_stride": args.sample_stride,
        "max_samples": args.max_samples,
        "num_inference_steps": args.num_inference_steps,
        "horizon": len(mae_per_horizon),
        "num_observations": total_samples,
        "mae_overall": float(abs_total / max(count_total, 1.0)),
        "rmse_overall": float(np.sqrt(sq_total / max(count_total, 1.0))),
        "mae_per_joint": {name: float(mae_per_dim[i]) for i, name in enumerate(joint_names)},
        "rmse_per_joint": {name: float(rmse_per_dim[i]) for i, name in enumerate(joint_names)},
        "mae_per_horizon": [float(v) for v in mae_per_horizon],
        "rmse_per_horizon": [float(v) for v in rmse_per_horizon],
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = _config.get_config(args.config_name)
    data_config = get_data_config(train_config, args.split)
    if data_config.repo_id is None:
        raise ValueError("Data config must have repo_id set.")

    root = Path(os.path.expandvars(data_config.root)).expanduser() if data_config.root is not None else None
    metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, root=root)
    delta_timestamps = {
        key: [t / metadata.fps for t in range(train_config.model.action_horizon)]
        for key in data_config.action_sequence_keys
    }

    dataset_kwargs = {}
    if data_config.video_backend is not None:
        dataset_kwargs["video_backend"] = data_config.video_backend
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=root,
        episodes=None if data_config.episodes is None else list(data_config.episodes),
        delta_timestamps=delta_timestamps,
        **dataset_kwargs,
    )

    policy = policy_config.create_trained_policy(
        train_config,
        Path(args.checkpoint_dir).expanduser(),
        repack_transforms=data_config.repack_transforms,
        sample_kwargs={"num_steps": args.num_inference_steps},
        default_prompt=None,
    )

    action_dim = metadata.features["action"]["shape"][0]
    joint_names = metadata.features["action"].get("names") or [f"action_{i}" for i in range(action_dim)]
    horizon = min(args.horizon or train_config.model.action_horizon, train_config.model.action_horizon)

    abs_per_dim = np.zeros(action_dim, dtype=np.float64)
    sq_per_dim = np.zeros(action_dim, dtype=np.float64)
    count_per_dim = np.zeros(action_dim, dtype=np.float64)
    abs_per_horizon = np.zeros(horizon, dtype=np.float64)
    sq_per_horizon = np.zeros(horizon, dtype=np.float64)
    count_per_horizon = np.zeros(horizon, dtype=np.float64)
    abs_total = 0.0
    sq_total = 0.0
    count_total = 0.0
    total_samples = 0

    indices = range(0, len(dataset), max(1, args.sample_stride))
    if args.max_samples is not None:
        indices = list(indices)[: args.max_samples]

    for idx in tqdm.tqdm(indices, desc=f"chunk error {args.split}"):
        sample = to_numpy(dataset[idx])
        gt_action = np.asarray(sample["action"][:horizon, :action_dim], dtype=np.float32)
        valid_mask = np.ones(gt_action.shape[:1], dtype=bool)
        if "action_is_pad" in sample:
            valid_mask = ~np.asarray(sample["action_is_pad"][:horizon], dtype=bool).reshape(-1)

        prediction = policy.infer(sample)
        pred_action = np.asarray(prediction["actions"][:horizon, :action_dim], dtype=np.float32)
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
        total_samples += 1

    jax.clear_caches()
    report = build_report(
        args=args,
        data_config=data_config,
        joint_names=list(joint_names),
        total_samples=total_samples,
        abs_per_dim=abs_per_dim,
        sq_per_dim=sq_per_dim,
        count_per_dim=count_per_dim,
        abs_per_horizon=abs_per_horizon,
        sq_per_horizon=sq_per_horizon,
        count_per_horizon=count_per_horizon,
        abs_total=abs_total,
        sq_total=sq_total,
        count_total=count_total,
    )
    report_path = output_dir / f"action_chunk_error_{args.split}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_csvs(report, output_dir)
    logging.info(
        "split=%s observations=%d horizon=%d MAE=%.6f RMSE=%.6f",
        args.split,
        total_samples,
        horizon,
        report["mae_overall"],
        report["rmse_overall"],
    )
    logging.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
