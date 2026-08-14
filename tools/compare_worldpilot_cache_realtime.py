#!/usr/bin/env python3
"""Compare one offline WorldPilot cache sample with online Cosmos Policy output.

The comparison uses the same LIBERO dataset frame for both paths:

  offline: cosmos_cache/episode_XXXXXX.npz
  online:  the realtime Cosmos Policy worker, with the frame images/state

It is intended to diagnose cache/model distribution mismatches and camera-order
mistakes before running a long simulator evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq

from openpi.policies.cosmos_realtime import CosmosRealtimeClient
from openpi.policies.cosmos_realtime import CosmosRealtimeConfig
from openpi.shared import paths

DEFAULT_DATA_ROOT = pathlib.Path(paths.configured_path("OPENPI_LIBERO_LEROBOT_ROOT", "datasets/libero_lerobot"))
DEFAULT_CACHE_ROOT = pathlib.Path(
    paths.configured_path(
        "OPENPI_WORLDPILOT_LIBERO_CACHE_ROOT",
        "cosmos_cache/WorldPilot-LIBERO-precompute/cosmos_cache",
    )
)
DEFAULT_COSMOS_ROOT = pathlib.Path(paths.configured_path("COSMOS_REPO", "cosmos-predict2.5"))
DEFAULT_POLICY_ROOT = pathlib.Path(
    paths.configured_path("COSMOS_POLICY_ROOT", "cosmos_checkpoints/Cosmos-Policy-LIBERO-Predict2-2B")
)
DEFAULT_POLICY_VAE = pathlib.Path(
    os.environ.get(
        "COSMOS_POLICY_VAE_PATH",
        str(
            pathlib.Path.home()
            / ".cache/huggingface/hub/models--nvidia--Cosmos-Predict2.5-2B/"
            "snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/tokenizer.pth"
        ),
    )
)


def _suite_dir(suite: str) -> str:
    return f"{suite}_no_noops_1.0.0_lerobot"


def _episode_file(root: pathlib.Path, episode: int, suffix: str) -> pathlib.Path:
    matches = sorted(root.glob(f"data/chunk-*/episode_{episode:06d}.parquet"))
    if not matches:
        raise FileNotFoundError(f"No parquet file found for episode {episode} under {root / 'data'}")
    return matches[0]


def _video_file(root: pathlib.Path, episode: int, camera: str) -> pathlib.Path:
    matches = sorted(root.glob(f"videos/chunk-*/{camera}/episode_{episode:06d}.mp4"))
    if not matches:
        raise FileNotFoundError(f"No {camera} video found for episode {episode} under {root / 'videos'}")
    return matches[0]


def _read_video_frame(path: pathlib.Path, frame_index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        decoded_index = 0
        for frame in container.decode(video=0):
            if decoded_index == frame_index:
                return frame.to_ndarray(format="rgb24")
            decoded_index += 1
    raise IndexError(f"Frame {frame_index} is out of range for {path} (decoded {decoded_index} frames)")


def _load_task_prompt(root: pathlib.Path, task_index: int) -> str:
    tasks_path = root / "meta" / "tasks.jsonl"
    for line in tasks_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["task_index"]) == task_index:
            return str(record["task"])
    raise KeyError(f"task_index={task_index} not found in {tasks_path}")


def _stats(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(value.shape),
        "dtype": str(array.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "l2": float(np.linalg.norm(value.reshape(-1))),
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32).reshape(-1)
    right = np.asarray(right, dtype=np.float32).reshape(-1)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(np.dot(left, right) / denominator) if denominator > 0 else 0.0


def _per_view_stats(name: str, value: np.ndarray) -> dict[str, Any]:
    return {f"{name}[{index}]": _stats(view) for index, view in enumerate(value)}


def _print_stats(title: str, value: np.ndarray) -> None:
    print(title)
    for name, stats in _per_view_stats(title, value).items():
        print(
            f"  {name}: shape={stats['shape']} dtype={stats['dtype']} "
            f"min={stats['min']:.6f} max={stats['max']:.6f} "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} l2={stats['l2']:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--episode", type=int, default=11)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--dataset-root", type=pathlib.Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-root", type=pathlib.Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--cosmos-root", type=pathlib.Path, default=DEFAULT_COSMOS_ROOT)
    parser.add_argument("--policy-root", type=pathlib.Path, default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--policy-vae", type=pathlib.Path, default=DEFAULT_POLICY_VAE)
    parser.add_argument("--gpu", default="0", help="CUDA device used by the temporary Cosmos worker")
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--proprio-mode",
        choices=("cache", "realtime"),
        default="cache",
        help="Use zero proprio to match the released WorldPilot cache, or pass the dataset state.",
    )
    parser.add_argument("--output-json", type=pathlib.Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    dataset_dir = args.dataset_root / _suite_dir(args.suite)
    cache_dir = args.cache_root / _suite_dir(args.suite)
    episode_path = _episode_file(dataset_dir, args.episode, "parquet")
    cache_path = cache_dir / f"episode_{args.episode:06d}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)

    table = pq.read_table(episode_path, columns=["observation.state", "task_index"])
    if args.frame < 0 or args.frame >= table.num_rows:
        raise IndexError(f"frame={args.frame} is out of range for {episode_path} with {table.num_rows} rows")
    state = np.asarray(table["observation.state"][args.frame].as_py(), dtype=np.float32)
    dataset_state = state.copy()
    cosmos_state: np.ndarray | None = state
    if args.proprio_mode == "cache":
        # The released WorldPilot precompute calls Cosmos with proprio=None;
        # Cosmos Policy then substitutes a zero 9D proprio vector.
        cosmos_state = None
    task_index = int(table["task_index"][args.frame].as_py())
    prompt = _load_task_prompt(dataset_dir, task_index)
    images = {
        "image": _read_video_frame(_video_file(dataset_dir, args.episode, "observation.images.image"), args.frame),
        "wrist_image": _read_video_frame(
            _video_file(dataset_dir, args.episode, "observation.images.wrist_image"), args.frame
        ),
    }

    with np.load(cache_path, allow_pickle=False) as cache:
        if "future_image_latents" not in cache or "action_chunk" not in cache:
            raise ValueError(f"{cache_path} must contain future_image_latents and action_chunk")
        if args.frame >= cache["future_image_latents"].shape[0]:
            raise IndexError(
                f"frame={args.frame} is out of range for cached future latents with "
                f"{cache['future_image_latents'].shape[0]} frames"
            )
        offline_latent = np.asarray(cache["future_image_latents"][args.frame], dtype=np.float32).reshape(2, -1)
        offline_action = np.asarray(cache["action_chunk"][args.frame], dtype=np.float32)

    policy_root = args.policy_root
    config = CosmosRealtimeConfig(
        cosmos_python=str(args.cosmos_root / ".venv/bin/python"),
        worker_script=str(paths.repo_root() / "tools/cosmos_realtime_worker.py"),
        cosmos_repo=str(args.cosmos_root),
        cosmos_vae_path=str(args.policy_vae),
        cosmos_policy_checkpoint=str(policy_root / "Cosmos-Policy-LIBERO-Predict2-2B.pt"),
        cosmos_policy_dataset_stats=str(policy_root / "libero_dataset_statistics.json"),
        cosmos_policy_text_embeddings=str(policy_root / "libero_t5_embeddings.pkl"),
        cosmos_policy_num_steps=args.num_steps,
        cosmos_policy_chunk_size=16,
        action_prior_dim=7,
        latent_dim=16 * 28 * 28,
        worker_cuda_visible_devices=args.gpu,
        worker_log_path="/tmp/worldpilot_cache_realtime_compare_worker.log",
    )
    client = CosmosRealtimeClient(config)
    try:
        online = client.infer(
            images=images,
            instruction=prompt,
            views=["image", "wrist_image"],
            state=cosmos_state,
        )
    finally:
        client.close()

    online_latent = np.asarray(online["latent"], dtype=np.float32)
    online_action = np.asarray(online["action_prior"], dtype=np.float32)
    if online_latent.shape != offline_latent.shape:
        raise ValueError(f"Latent shape mismatch: offline={offline_latent.shape} online={online_latent.shape}")
    if online_action.shape != offline_action.shape:
        raise ValueError(f"Action shape mismatch: offline={offline_action.shape} online={online_action.shape}")

    result: dict[str, Any] = {
        "suite": args.suite,
        "episode": args.episode,
        "frame": args.frame,
        "proprio_mode": args.proprio_mode,
        "dataset_state": dataset_state.tolist(),
        "cosmos_state": None if cosmos_state is None else cosmos_state.tolist(),
        "task_index": task_index,
        "prompt": prompt,
        "cache_path": str(cache_path),
        "offline_latent": _per_view_stats("offline_latent", offline_latent),
        "online_latent": _per_view_stats("online_latent", online_latent),
        "offline_action_prior": _stats(offline_action),
        "online_action_prior": _stats(online_action),
        "latent_cosine_same_order": [_cosine(offline_latent[i], online_latent[i]) for i in range(2)],
        "latent_cosine_swapped_order": [_cosine(offline_latent[i], online_latent[1 - i]) for i in range(2)],
        "latent_mse_same_order": [float(np.mean((offline_latent[i] - online_latent[i]) ** 2)) for i in range(2)],
        "latent_mse_swapped_order": [float(np.mean((offline_latent[i] - online_latent[1 - i]) ** 2)) for i in range(2)],
        "action_prior_mse": float(np.mean((offline_action - online_action) ** 2)),
        "action_prior_mae": float(np.mean(np.abs(offline_action - online_action))),
        "action_prior_per_dim_mse": np.mean((offline_action - online_action) ** 2, axis=0).tolist(),
        "action_prior_per_dim_corr": [
            float(np.corrcoef(offline_action[:, dim], online_action[:, dim])[0, 1])
            for dim in range(offline_action.shape[1])
        ],
        "offline_action_prior_head": offline_action[:4].tolist(),
        "online_action_prior_head": online_action[:4].tolist(),
        "online_view_ids": np.asarray(online["view_ids"]).tolist(),
        "online_time_ids": np.asarray(online["time_ids"]).tolist(),
    }

    print(f"sample: {args.suite} episode={args.episode} frame={args.frame} task_index={task_index}")
    print(f"prompt: {prompt}")
    print(f"cache: {cache_path}")
    _print_stats("offline future_image_latents", offline_latent)
    _print_stats("online future_image_latents", online_latent)
    print("latent cosine, same view:     ", result["latent_cosine_same_order"])
    print("latent cosine, swapped view:  ", result["latent_cosine_swapped_order"])
    print("latent mse, same view:        ", result["latent_mse_same_order"])
    print("latent mse, swapped view:     ", result["latent_mse_swapped_order"])
    print("offline action_prior:", _stats(offline_action))
    print("online action_prior: ", _stats(online_action))
    print(f"action prior mse={result['action_prior_mse']:.8f} mae={result['action_prior_mae']:.8f}")
    print("action prior per-dim mse:", result["action_prior_per_dim_mse"])
    print("action prior per-dim corr:", result["action_prior_per_dim_corr"])
    print("offline action prior first4:\n", np.array2string(offline_action[:4], precision=6))
    print("online action prior first4:\n", np.array2string(online_action[:4], precision=6))
    print("online view_ids:", result["online_view_ids"], "time_ids:", result["online_time_ids"])

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
