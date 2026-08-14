"""Training registrations for the local Cosmos latent-fusion experiment."""

from __future__ import annotations

import os
from typing import Any

from openpi.models import pi05_cosmos_config
from openpi.shared import paths
from openpi.training import weight_loaders
import openpi.training.optimizer as _optimizer


def get_configs() -> list[Any]:
    """Return the Cosmos plug config without enlarging the core config module."""
    # Imported lazily to avoid a cycle while config.py is building its registry.
    from openpi.training import config as _config

    repo_id = os.environ.get("OPENPI_COSMOS_PLUG_REPO_ID", "move_aloha_bottles_box_basket_openpi")
    dataset_root = os.environ.get(
        "OPENPI_COSMOS_PLUG_DATASET_ROOT",
        str(paths.repo_root() / "datasets/move_aloha_bottles_box_basket_openpi"),
    )
    cache_dir = os.environ.get(
        "OPENPI_COSMOS_PLUG_CACHE_DIR",
        str(
            paths.repo_root()
            / "cosmos_cache/move_aloha_bottles_box_basket_openpi_predict_video2world_k16_future_slot"
        ),
    )
    latent_dim = int(os.environ.get("OPENPI_COSMOS_PLUG_LATENT_DIM", "15360"))
    num_latent_tokens = int(os.environ.get("OPENPI_COSMOS_PLUG_NUM_LATENT_TOKENS", "2"))
    max_future_steps = int(os.environ.get("OPENPI_COSMOS_PLUG_MAX_FUTURE_STEPS", "17"))
    max_views = int(os.environ.get("OPENPI_COSMOS_PLUG_MAX_VIEWS", "2"))

    return [
        _config.TrainConfig(
            name="pi05_cosmos",
            model=pi05_cosmos_config.Pi05CosmosConfig(
                pi05=True,
                action_dim=14,
                action_horizon=16,
                discrete_state_input=False,
                paligemma_variant="gemma_2b_lora",
                action_expert_variant="gemma_300m_lora",
                use_cosmos_latent_steering=True,
                cosmos_latent_dim=latent_dim,
                vlm_width=2048,
                num_cosmos_latent_tokens=num_latent_tokens,
                max_future_steps=max_future_steps,
                max_views=max_views,
                steering_num_heads=8,
            ),
            data=_config.LeRobotGraspDataConfig(
                repo_id=repo_id,
                assets=_config.AssetsConfig(
                    assets_dir=os.environ.get(
                        "OPENPI_COSMOS_PLUG_ASSETS_DIR",
                        str(paths.repo_root() / "assets/pi05_absolute"),
                    ),
                    asset_id="absolute",
                ),
                root=dataset_root,
                default_prompt=_config.GRASP_PROMPT,
                action_dim=14,
                use_delta_joint_actions=False,
                output_drop_first_n_dims=2,
                prepend_zero_dims=True,
                video_backend="pyav",
                image_key="image",
                wrist_image_key="wrist_image",
                state_key="state",
                actions_key="actions",
                cosmos_latent_cache_dir=cache_dir,
                base_config=_config.DataConfig(prompt_from_task=True),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                _config.PI05_BASE_PARAMS,
                missing_regex=(
                    ".*lora.*|dynamics_encoder/.*|latent_steering/.*|"
                    "cosmos_time_embed/.*|cosmos_view_embed/.*"
                ),
            ),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=4e-5,
                decay_steps=10_000,
                decay_lr=1e-5,
            ),
            optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
            freeze_filter=_config.cosmos_steering_lora_freeze_filter(),
            ema_decay=None,
            num_train_steps=10_000,
            batch_size=64,
            num_workers=8,
            log_interval=20,
            save_interval=1000,
            max_checkpoints_to_keep=None,
            eval_interval=0,
            eval_num_batches=16,
            keep_period=1000,
            wandb_enabled=False,
        )
    ]
