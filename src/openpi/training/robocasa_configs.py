"""JAX status-head configurations for RoboCasa composite subtasks."""

from __future__ import annotations

import os
from typing import Any

from openpi.models import pi0_config
from openpi.shared import paths
from openpi.training import weight_loaders
import openpi.training.optimizer as _optimizer


_ROBOCASA_TASKS = (
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CuttingToolSelection",
    "DeliverStraw",
    "GarnishPancake",
    "GatherTableware",
    "GetToastedBread",
    "HeatKebabSandwich",
    "KettleBoiling",
    "LoadDishwasher",
    "MakeIceLemonade",
    "PackIdenticalLunches",
    "PanTransfer",
    "PortionHotDogs",
    "PreSoakPan",
    "PrepareCoffee",
    "RecycleBottlesByType",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SeparateFreezerRack",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WaffleReheat",
    "WashFruitColander",
    "WashLettuce",
    "WeighIngredients",
)


def _paths() -> tuple[str, str, str, str]:
    repo_root = paths.repo_root()
    dataset_root = os.environ.get(
        "OPENPI_ROBOCASA_ROOT", str(repo_root / "robocasa/datasets/v1.0/target/composite")
    )
    split_root = os.environ.get(
        "OPENPI_ROBOCASA_SPLIT_ROOT", str(repo_root / "robocasa/processed/composite_subtasks/splits")
    )
    assets_root = os.environ.get("OPENPI_ROBOCASA_ASSETS_ROOT", str(repo_root / "assets"))
    return dataset_root, split_root, assets_root, ""


def _model(*, status_only_trainable: bool) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        discrete_state_input=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        enable_status_head=True,
        status_hidden_dim=512,
        status_num_layers=2,
        status_num_heads=8,
        status_ffn_dim=2048,
        status_dropout=0.1,
        status_loss_weight=1.0,
        status_only_trainable=status_only_trainable,
        # RoboCasa stores 12 action channels. The remaining 20 channels are
        # padding for the pi0.5 interface and must not affect action loss.
        action_loss_mask=(True,) * 12 + (False,) * 20,
    )


def _data(
    _config: Any,
    *,
    split: str,
    dataset_root: str,
    split_root: str,
    assets_root: str,
) -> Any:
    return _config.LeRobotRoboCasaCompletionDataConfig(
        repo_id="robocasa_composite_status",
        assets=_config.AssetsConfig(
            assets_dir=os.path.join(assets_root, "pi05_robocasa_status"),
            asset_id="robocasa_composite_status",
        ),
        root=dataset_root,
        task_names=_ROBOCASA_TASKS,
        split_jsonl=os.path.join(split_root, f"{split}.jsonl"),
        video_backend="pyav",
        completion_positive_window=0.1,
        completion_hard_negative_window=0.1,
        completion_balance=True,
        completion_positive_ratio=0.5,
        completion_include_subtask_prompt=True,
        action_key="action",
        state_key="observation.state",
        base_image_key="observation.images.robot0_agentview_left",
        wrist_image_key="observation.images.robot0_eye_in_hand",
        right_image_key="observation.images.robot0_agentview_right",
        base_config=_config.DataConfig(prompt_from_task=False),
    )


def _config(_config: Any, *, name: str, status_only_trainable: bool, peak_lr: float) -> Any:
    dataset_root, split_root, assets_root, _ = _paths()
    model = _model(status_only_trainable=status_only_trainable)
    return _config.TrainConfig(
        name=name,
        model=model,
        data=_data(
            _config,
            split="train",
            dataset_root=dataset_root,
            split_root=split_root,
            assets_root=assets_root,
        ),
        eval_data=_data(
            _config,
            split="val",
            dataset_root=dataset_root,
            split_root=split_root,
            assets_root=assets_root,
        ),
        assets_base_dir=assets_root,
        weight_loader=weight_loaders.CheckpointWeightLoader(_config.PI05_BASE_PARAMS),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=peak_lr,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        num_train_steps=20_000,
        batch_size=32,
        num_workers=8,
        log_interval=20,
        save_interval=1000,
        eval_interval=1000,
        eval_num_batches=32,
        keep_period=1000,
        max_checkpoints_to_keep=5,
        wandb_enabled=False,
        policy_metadata={"status_threshold": model.done_threshold},
    )


def get_configs() -> list[Any]:
    from openpi.training import config as _config_module

    return [
        _config(
            _config_module,
            name="pi05_robocasa_status",
            status_only_trainable=True,
            peak_lr=1e-4,
        ),
        _config(
            _config_module,
            name="pi05_robocasa_joint",
            status_only_trainable=False,
            peak_lr=2.5e-5,
        ),
    ]
