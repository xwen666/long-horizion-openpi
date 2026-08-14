from __future__ import annotations

import dataclasses
import logging
import os
import socket
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import tyro

from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    config: str = "pi05_grasp_200_strict_lora"
    checkpoint_dir: str = "/cc/openpi/checkpoints/pi05_grasp_200_strict_lora/pi05_grasp_200lora_lora/19999"
    port: int = 8000
    host: str = "0.0.0.0"
    default_prompt: str | None = None

    dataset_root: str = "/cc/openpi/grasp_200"
    splits_path: str = "/cc/openpi/grasp_200lora_splits/splits.json"
    repo_id: str = "grasp_200lora"
    assets_dir: str = "/cc/openpi/outputs/openpi_assets/pi05_grasp_low_mem_finetune"
    asset_id: str = "grasp"


def build_policy(args: Args):
    cfg = _config.get_config(args.config)
    train_data = dataclasses.replace(
        cfg.data,
        repo_id=args.repo_id,
        root=args.dataset_root,
        splits_path=args.splits_path,
        default_prompt="grasp the object",
        assets=dataclasses.replace(cfg.data.assets, assets_dir=args.assets_dir, asset_id=args.asset_id),
    )
    eval_data = dataclasses.replace(
        cfg.eval_data,
        repo_id=args.repo_id,
        root=args.dataset_root,
        splits_path=args.splits_path,
        default_prompt="grasp the object",
        assets=dataclasses.replace(cfg.eval_data.assets, assets_dir=args.assets_dir, asset_id=args.asset_id),
    )
    cfg = dataclasses.replace(cfg, data=train_data, eval_data=eval_data)
    return _policy_config.create_trained_policy(cfg, Path(args.checkpoint_dir), default_prompt=args.default_prompt)


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    policy = build_policy(args)
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Serving policy on %s:%d (host=%s ip=%s)", args.host, args.port, hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            **policy.metadata,
            "action_horizon": 20,
            "action_threshold": 0.5,
            "checkpoint_dir": args.checkpoint_dir,
        },
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
