from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

from openpi.policies import policy_config
from openpi.training import config as _config


CHECKPOINT_DIR = Path("/cc/openpi/checkpoints/pi05_grasp_200_strict_lora/pi05_grasp_200lora_lora/19999")
DATASET_ROOT = Path("/cc/openpi/grasp_200")
SPLITS_PATH = Path("/cc/openpi/grasp_200lora_splits/splits.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GT vs predicted grasp episode video.")
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def load_fk_class():
    spec = importlib.util.spec_from_file_location(
        "piper_fk", Path("/cc/starVLA/piper_sdk/piper_sdk/kinematics/piper_fk.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.C_PiperForwardKinematics


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


def fk_points(fk_solver, state7: np.ndarray) -> np.ndarray:
    joints = state7[:6].astype(np.float64)
    poses = fk_solver.CalFK(joints.tolist())
    pts = np.zeros((7, 3), dtype=np.float64)
    for i in range(6):
        pts[i + 1] = np.asarray(poses[i][:3], dtype=np.float64) / 1000.0
    return pts


def next_state_from_action(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    nxt = state.copy()
    nxt[:6] = state[:6] + action[:6]
    nxt[6] = action[6]
    return nxt


def simulate_pred_states(initial_state: np.ndarray, pred_actions: np.ndarray) -> np.ndarray:
    states = []
    state = initial_state.copy()
    for action in pred_actions:
        state = next_state_from_action(state, action)
        states.append(state.copy())
    return np.stack(states, axis=0)


def build_policy():
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
    return policy, train_config


def plot_arm(ax, pts: np.ndarray, color: str, ee_traj: np.ndarray | None = None):
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, linewidth=4, alpha=0.9)
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color="orange", s=55, edgecolors="peru", linewidths=0.8)
    if ee_traj is not None and len(ee_traj) > 1:
        ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], color="#e91e63", linestyle="--", linewidth=1.5, alpha=0.55)


def style_axis(ax):
    ax.set_xlim(-0.3, 0.4)
    ax.set_ylim(-0.5, 0.25)
    ax.set_zlim(-0.15, 0.58)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=28, azim=-62)
    ax.grid(True, alpha=0.35)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    prompt_map = load_prompt_map()
    metadata = lerobot_dataset.LeRobotDatasetMetadata("grasp_200lora", root=DATASET_ROOT)
    horizon = 20
    delta_timestamps = {"action": [t / metadata.fps for t in range(horizon)]}
    dataset = lerobot_dataset.LeRobotDataset(
        "grasp_200lora",
        root=DATASET_ROOT,
        episodes=None,
        delta_timestamps=delta_timestamps,
        video_backend="pyav",
    )

    episode_indices = torch.stack(dataset.hf_dataset["episode_index"]).numpy()
    matching = np.where(episode_indices == args.episode_id)[0]
    if len(matching) == 0:
        raise ValueError(f"Episode {args.episode_id} not found")

    C_PiperForwardKinematics = load_fk_class()
    fk_solver = C_PiperForwardKinematics()
    policy, train_config = build_policy()

    prompt = None
    frames = []
    max_frames = min(args.max_frames, len(matching))

    for local_i in range(max_frames):
        dataset_idx = int(matching[local_i])
        sample = to_numpy(dataset[dataset_idx])
        if prompt is None:
            task_index = int(sample.get("task_index", 0))
            prompt = sample.get("task") or prompt_map.get(task_index, "grasp the object")

        obs = {
            "observation.images.front": sample["observation.images.front"],
            "observation.images.wrist": sample["observation.images.wrist"],
            "observation.state": np.asarray(sample["observation.state"], dtype=np.float32),
            "action": np.zeros((train_config.model.action_horizon, train_config.model.action_dim), dtype=np.float32),
            "prompt": prompt,
        }

        gt_state = np.asarray(sample["observation.state"], dtype=np.float32)
        gt_actions = np.asarray(sample["action"], dtype=np.float32)
        pred_actions = np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
        pred_future_states = simulate_pred_states(gt_state, pred_actions)

        gt_arm = fk_points(fk_solver, gt_state)
        pred_arm = fk_points(fk_solver, gt_state)
        gt_traj = np.stack([fk_points(fk_solver, next_state_from_action(gt_state, a))[-1] for a in gt_actions], axis=0)
        pred_traj = np.stack([fk_points(fk_solver, s)[-1] for s in pred_future_states], axis=0)

        fig = plt.figure(figsize=(12, 5), dpi=100)
        fig.suptitle(f"Episode {args.episode_id}: {prompt}", fontsize=13, fontweight="bold")

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")

        ax1.set_title("Ground Truth State", fontsize=14, fontweight="bold")
        ax2.set_title("OpenPI Predicted Action", fontsize=14, fontweight="bold")
        ax1.text2D(0.02, 0.94, f"Frame {local_i}/{max_frames - 1}", transform=ax1.transAxes, fontsize=10)
        ax2.text2D(0.02, 0.94, f"Frame {local_i}/{max_frames - 1}", transform=ax2.transAxes, fontsize=10)

        plot_arm(ax1, gt_arm, color="#4CAF50", ee_traj=gt_traj)
        plot_arm(ax2, pred_arm, color="#9C27B0", ee_traj=pred_traj)
        style_axis(ax1)
        style_axis(ax2)

        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        frames.append(frame)
        plt.close(fig)

    imageio.mimsave(args.output, frames, fps=args.fps)
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
