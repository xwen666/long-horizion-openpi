"""WorldPilot-style Cosmos/WAM training registrations for LIBERO."""

from __future__ import annotations

import os
import pathlib
from typing import Any

from openpi.models import pi05_cosmos_config
from openpi.shared import paths
from openpi.training import weight_loaders
import openpi.training.optimizer as _optimizer

_SUITES = (
    "libero_spatial_no_noops_1.0.0_lerobot",
    "libero_object_no_noops_1.0.0_lerobot",
    "libero_goal_no_noops_1.0.0_lerobot",
    "libero_10_no_noops_1.0.0_lerobot",
)


def _paths() -> tuple[str, str, int, int]:
    dataset_root = os.environ.get(
        "OPENPI_LIBERO_LEROBOT_ROOT",
        str(paths.repo_root() / "datasets/libero_lerobot"),
    )
    cache_root = os.environ.get(
        "OPENPI_WORLDPILOT_LIBERO_CACHE_ROOT",
        str(paths.repo_root() / "cosmos_cache/WorldPilot-LIBERO-precompute/cosmos_cache"),
    )
    latent_dim = int(os.environ.get("OPENPI_WORLDPILOT_LIBERO_LATENT_DIM", str(16 * 28 * 28)))
    num_latent_tokens = int(os.environ.get("OPENPI_WORLDPILOT_LIBERO_NUM_LATENT_TOKENS", "2"))
    return dataset_root, cache_root, latent_dim, num_latent_tokens


def _model(num_latent_tokens: int, latent_dim: int) -> pi05_cosmos_config.Pi05CosmosConfig:
    return pi05_cosmos_config.Pi05CosmosConfig(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        action_loss_mask=(True,) * 7 + (False,) * 25,
        discrete_state_input=False,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        use_cosmos_latent_steering=True,
        cosmos_latent_dim=latent_dim,
        vlm_width=2048,
        num_cosmos_latent_tokens=num_latent_tokens,
        max_future_steps=17,
        max_views=2,
        steering_num_heads=8,
        use_action_steering=True,
        action_prior_encoder_type="query_pool",
        action_prior_encoder_hidden_dim=256,
        action_prior_encoder_layers=2,
        action_prior_encoder_heads=4,
        wam_action_horizon=16,
        policy_action_horizon=16,
        action_prior_source_to_policy=tuple(range(7)),
        action_prior_gripper_indices=(6,),
        continuous_action_interpolation="linear",
        gripper_interpolation="nearest",
        wam_condition_dropout=0.3,
        vision_prior_dropout=0.3,
        action_prior_dropout=0.3,
        strict_action_prior_shapes=True,
    )


def _loader(_config: Any) -> weight_loaders.CheckpointWeightLoader:
    return weight_loaders.CheckpointWeightLoader(
        _config.PI05_BASE_PARAMS,
        missing_regex=(
            "dynamics_encoder/.*|latent_steering/.*|action_prior_encoder/.*|"
            "cosmos_time_embed/.*|cosmos_view_embed/.*"
        ),
    )


def _schedule(_config: Any) -> Any:
    return _optimizer.CosineDecaySchedule(
        warmup_steps=1_000,
        peak_lr=2e-5,
        decay_steps=40_000,
        decay_lr=2e-6,
    )


def _single_config(_config: Any, config_name: str, suite_name: str, *, latent_dim: int, tokens: int) -> Any:
    dataset_root, cache_root, _, _ = _paths()
    return _config.TrainConfig(
        name=config_name,
        model=_model(tokens, latent_dim),
        data=_config.LeRobotLiberoCosmosDataConfig(
            repo_id=suite_name,
            root=str(pathlib.Path(dataset_root) / suite_name),
            video_backend="pyav",
            extra_delta_transform=False,
            cosmos_latent_cache_dir=str(pathlib.Path(cache_root) / suite_name),
            image_key="observation.images.image",
            wrist_image_key="observation.images.wrist_image",
            state_key="observation.state",
            action_key="action",
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=_loader(_config),
        lr_schedule=_schedule(_config),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=30_000,
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


def _combined_config(_config: Any, *, latent_dim: int, tokens: int) -> Any:
    dataset_root, cache_root, _, _ = _paths()
    return _config.TrainConfig(
        name="pi05_cosmos_libero_all",
        model=_model(tokens, latent_dim),
        data=_config.LeRobotLiberoCosmosCombinedDataConfig(
            repo_id="libero_worldpilot_all",
            suite_names=_SUITES,
            root=dataset_root,
            cosmos_latent_cache_root=cache_root,
            video_backend="pyav",
            extra_delta_transform=False,
            base_config=_config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=_loader(_config),
        lr_schedule=_schedule(_config),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        num_train_steps=40_000,
        batch_size=16,
        num_workers=8,
        log_interval=20,
        save_interval=2000,
        max_checkpoints_to_keep=None,
        eval_interval=0,
        eval_num_batches=16,
        keep_period=2000,
        wandb_enabled=False,
    )


def get_configs() -> list[Any]:
    """Return per-suite and combined WorldPilot-style LIBERO configs."""
    from openpi.training import config as _config

    _, _, latent_dim, tokens = _paths()
    return [
        _single_config(_config, "pi05_cosmos_libero_spatial", _SUITES[0], latent_dim=latent_dim, tokens=tokens),
        _single_config(_config, "pi05_cosmos_libero_object", _SUITES[1], latent_dim=latent_dim, tokens=tokens),
        _single_config(_config, "pi05_cosmos_libero_goal", _SUITES[2], latent_dim=latent_dim, tokens=tokens),
        _single_config(_config, "pi05_cosmos_libero_10", _SUITES[3], latent_dim=latent_dim, tokens=tokens),
        _combined_config(_config, latent_dim=latent_dim, tokens=tokens),
    ]
