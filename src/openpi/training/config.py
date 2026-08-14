"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import json
import logging
import os
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.grasp_policy as grasp_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.shared.download as _download
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.normalize as _normalize
import openpi.shared.paths as _paths
import openpi.training.cosmos_configs as _cosmos_configs
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.robocasa_configs as _robocasa_configs
import openpi.training.weight_loaders as weight_loaders
import openpi.training.worldpilot_configs as _worldpilot_configs
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Optional local dataset root. When set, LeRobot will read the dataset from this directory instead of the HF cache.
    root: str | None = None
    # Optional subset of episode indices to load from a local/HF LeRobot dataset.
    episodes: Sequence[int] | None = None
    # Optional video decoder backend passed through to LeRobotDataset.
    video_backend: str | None = None
    # Optional multi-source LeRobot datasets. When set, data_loader concatenates these local roots.
    source_repo_ids: Sequence[str] = ()
    source_roots: Sequence[str] = ()
    source_episodes: Sequence[Sequence[int]] = ()
    source_cache_dirs: Sequence[str] = ()
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Optional subtask completion labels for the JAX status branch. None keeps the
    # existing data path unchanged. Labels use the last positive window of each segment.
    completion_positive_window: float | None = None
    completion_hard_negative_window: float = 0.1
    completion_balance: bool = True
    completion_positive_ratio: float = 0.5
    completion_include_subtask_prompt: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


def _expand_env_path(path: str | None) -> str | None:
    if path is None:
        return None
    return str(pathlib.Path(os.path.expandvars(path)).expanduser())


def _load_episode_split(splits_path: str | None, split: str) -> tuple[int, ...] | None:
    if splits_path is None:
        return None
    path = pathlib.Path(os.path.expandvars(splits_path)).expanduser()
    with path.open("r", encoding="utf-8") as f:
        splits = json.load(f)
    if split not in splits:
        raise ValueError(f"Split '{split}' not found in {path}. Available splits: {sorted(splits)}")
    return tuple(int(ep) for ep in splits[split])


def _load_episode_indices_from_cache(cache_dir: str) -> tuple[int, ...]:
    path = pathlib.Path(cache_dir)
    episode_indices = []
    for cache_path in sorted(path.glob("episode_*.npz")):
        try:
            episode_indices.append(int(cache_path.stem.removeprefix("episode_")))
        except ValueError as exc:
            raise ValueError(f"Unexpected Cosmos cache filename: {cache_path}") from exc
    if not episode_indices:
        raise FileNotFoundError(f"No episode_*.npz files found in Cosmos cache directory: {cache_dir}")
    return tuple(episode_indices)


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotRoboCasaCompletionDataConfig(DataConfigFactory):
    """RoboCasa composite LeRobot sources with subtask completion supervision.

    Each source is expected to expose the RoboCasa annotation columns
    ``episode_index``, ``frame_index`` and ``subtask_idx``. The optional
    ``task_roots`` argument is useful when each task has a different date or
    LeRobot directory layout.
    """

    root: str = tyro.MISSING
    task_names: Sequence[str] = ()
    task_roots: Sequence[str] = ()
    lerobot_subdir: str = "20250809/lerobot"
    video_backend: str | None = "pyav"
    default_prompt: str | None = None
    completion_positive_window: float = 0.1
    completion_hard_negative_window: float = 0.1
    completion_balance: bool = True
    completion_positive_ratio: float = 0.5
    completion_include_subtask_prompt: bool = False
    split_jsonl: str | None = None
    action_key: str = "action"
    state_key: str = "observation.state"
    base_image_key: str = "observation.images.robot0_agentview_left"
    wrist_image_key: str = "observation.images.robot0_eye_in_hand"
    right_image_key: str = "observation.images.robot0_agentview_right"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.task_names:
            raise ValueError("LeRobotRoboCasaCompletionDataConfig requires task_names.")
        if self.task_roots and len(self.task_roots) != len(self.task_names):
            raise ValueError("task_roots must have the same length as task_names when provided.")

        root = pathlib.Path(_expand_env_path(self.root) or "")
        source_roots = tuple(_expand_env_path(path) or path for path in self.task_roots) if self.task_roots else tuple(
            self._find_task_root(root, task) for task in self.task_names
        )
        source_episodes = self._load_split_episodes(self.split_jsonl, self.task_names)
        repack_structure = {
            "observation/image": self.base_image_key,
            "observation/wrist_image": self.wrist_image_key,
            "observation/right_image": self.right_image_key,
            "observation/state": self.state_key,
            "actions": self.action_key,
            "prompt": "prompt",
            "episode_index": "episode_index",
            "frame_index": "frame_index",
        }
        if getattr(model_config, "enable_status_head", False):
            repack_structure["done_label"] = "done_label"
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        data_transforms = _transforms.Group(inputs=[robocasa_policy.RoboCasaInputs()])
        base_config = self.create_base_config(assets_dirs, model_config)
        return dataclasses.replace(
            base_config,
            source_repo_ids=tuple(self.task_names),
            source_roots=source_roots,
            video_backend=self.video_backend or os.environ.get("OPENPI_VIDEO_BACKEND") or None,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(self.action_key,),
            completion_positive_window=(self.completion_positive_window if model_config.enable_status_head else None),
            completion_hard_negative_window=self.completion_hard_negative_window,
            completion_balance=self.completion_balance,
            completion_positive_ratio=self.completion_positive_ratio,
            completion_include_subtask_prompt=self.completion_include_subtask_prompt,
            source_episodes=source_episodes,
        )

    def _find_task_root(self, root: pathlib.Path, task_name: str) -> str:
        task_root = root / task_name
        candidates = sorted(task_root.glob("*/lerobot"))
        if len(candidates) == 1:
            return str(candidates[0])
        fallback = task_root / self.lerobot_subdir
        if fallback.exists():
            return str(fallback)
        if not candidates:
            raise FileNotFoundError(f"No RoboCasa LeRobot directory found for task {task_name}: {task_root}")
        raise ValueError(f"Expected one dated LeRobot directory for {task_name}, found: {candidates}")

    @staticmethod
    def _load_split_episodes(
        split_jsonl: str | None, task_names: Sequence[str]
    ) -> tuple[tuple[int, ...], ...]:
        if split_jsonl is None:
            return ()
        episodes = {task_name: set() for task_name in task_names}
        path = pathlib.Path(_expand_env_path(split_jsonl) or split_jsonl)
        with path.open("r", encoding="utf-8") as split_file:
            for line in split_file:
                if not line.strip():
                    continue
                record = json.loads(line)
                task_name = record.get("task_name") or record.get("dataset_task")
                if task_name in episodes:
                    episodes[task_name].add(int(record["episode_index"]))
        return tuple(tuple(sorted(episodes[task_name])) for task_name in task_names)


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoCosmosDataConfig(DataConfigFactory):
    """LIBERO LeRobot data config with precomputed Cosmos/WAM latent cache."""

    root: str | None = None
    episodes: Sequence[int] | None = None
    video_backend: str | None = None
    extra_delta_transform: bool = False
    cosmos_latent_cache_dir: str | None = None
    filter_episodes_to_cache: bool = True
    image_key: str = "image"
    wrist_image_key: str = "wrist_image"
    state_key: str = "state"
    action_key: str = "actions"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_structure = {
            "observation/image": self.image_key,
            "observation/wrist_image": self.wrist_image_key,
            "observation/state": self.state_key,
            "actions": self.action_key,
            "prompt": "prompt",
        }
        if self.cosmos_latent_cache_dir is not None:
            repack_structure["episode_index"] = "episode_index"
            repack_structure["frame_index"] = "frame_index"

        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
        data_inputs: list[_transforms.DataTransformFn] = []
        if self.cosmos_latent_cache_dir is not None:
            data_inputs.append(
                _transforms.AttachCosmosLatent(
                    _expand_env_path(self.cosmos_latent_cache_dir),
                    attach_action_prior=getattr(model_config, "use_action_steering", False),
                    expected_action_horizon=getattr(model_config, "wam_action_horizon", None)
                    or model_config.action_horizon,
                    expected_action_dim=(
                        len(getattr(model_config, "action_prior_source_to_policy", ()))
                        if getattr(model_config, "action_prior_source_to_policy", None) is not None
                        else None
                    ),
                    expected_action_normalized=getattr(model_config, "action_prior_normalization", "normalized")
                    == "normalized",
                    strict_action_prior_shapes=getattr(model_config, "strict_action_prior_shapes", True),
                    strict_cache_alignment=getattr(model_config, "strict_cache_alignment", True),
                )
            )
        data_inputs.append(libero_policy.LiberoInputs(model_type=model_config.model_type))

        data_transforms = _transforms.Group(
            inputs=data_inputs,
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)
        cache_dir = _expand_env_path(self.cosmos_latent_cache_dir)
        episodes = tuple(self.episodes) if self.episodes is not None else None
        if cache_dir is not None and self.filter_episodes_to_cache:
            cache_episodes = _load_episode_indices_from_cache(cache_dir)
            episodes = tuple(ep for ep in episodes if ep in cache_episodes) if episodes is not None else cache_episodes

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            root=_expand_env_path(self.root),
            episodes=episodes,
            video_backend=self.video_backend or os.environ.get("OPENPI_VIDEO_BACKEND") or None,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(self.action_key,),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoCosmosCombinedDataConfig(DataConfigFactory):
    """Combined LIBERO suites with per-suite precomputed WorldPilot Cosmos/WAM cache."""

    suite_names: Sequence[str] = ()
    root: str | None = None
    cosmos_latent_cache_root: str | None = None
    video_backend: str | None = None
    extra_delta_transform: bool = False
    image_key: str = "observation.images.image"
    wrist_image_key: str = "observation.images.wrist_image"
    state_key: str = "observation.state"
    action_key: str = "action"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        if not self.suite_names:
            raise ValueError("LeRobotLiberoCosmosCombinedDataConfig requires at least one suite name.")

        root = pathlib.Path(_expand_env_path(self.root) or "")
        cache_root = pathlib.Path(_expand_env_path(self.cosmos_latent_cache_root) or "")
        source_roots = tuple(str(root / suite_name) for suite_name in self.suite_names)
        source_cache_dirs = tuple(str(cache_root / suite_name) for suite_name in self.suite_names)
        source_episodes = tuple(_load_episode_indices_from_cache(cache_dir) for cache_dir in source_cache_dirs)

        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": self.image_key,
                        "observation/wrist_image": self.wrist_image_key,
                        "observation/state": self.state_key,
                        "actions": self.action_key,
                        "prompt": "prompt",
                        "episode_index": "episode_index",
                        "frame_index": "frame_index",
                        "cosmos_cache_dir": "cosmos_cache_dir",
                    }
                )
            ]
        )
        data_inputs: list[_transforms.DataTransformFn] = [
            _transforms.AttachCosmosLatent(
                attach_action_prior=getattr(model_config, "use_action_steering", False),
                expected_action_horizon=getattr(model_config, "wam_action_horizon", None)
                or model_config.action_horizon,
                expected_action_dim=(
                    len(getattr(model_config, "action_prior_source_to_policy", ()))
                    if getattr(model_config, "action_prior_source_to_policy", None) is not None
                    else None
                ),
                expected_action_normalized=getattr(model_config, "action_prior_normalization", "normalized")
                == "normalized",
                strict_action_prior_shapes=getattr(model_config, "strict_action_prior_shapes", True),
                strict_cache_alignment=getattr(model_config, "strict_cache_alignment", True),
            ),
            libero_policy.LiberoInputs(model_type=model_config.model_type),
        ]
        data_transforms = _transforms.Group(
            inputs=data_inputs,
            outputs=[libero_policy.LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            source_repo_ids=tuple(self.suite_names),
            source_roots=source_roots,
            source_episodes=source_episodes,
            source_cache_dirs=source_cache_dirs,
            video_backend=self.video_backend or os.environ.get("OPENPI_VIDEO_BACKEND") or None,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(self.action_key,),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotGraspDataConfig(DataConfigFactory):
    """Config for the local Piper grasp LeRobot dataset."""

    root: str | None = None
    splits_path: str | None = None
    split: Literal["train", "val", "test"] = "train"
    default_prompt: str | None = None
    use_delta_joint_actions: bool = True
    action_dim: int = 7
    output_drop_first_n_dims: int = 0
    prepend_zero_dims: bool = False
    prepend_zero_count: int = 2
    delta_action_mask_dims: Sequence[int] | None = None
    image_key: str = "observation.images.front"
    wrist_image_key: str = "observation.images.wrist"
    state_key: str = "observation.state"
    action_key: str = "action"
    actions_key: str | None = None
    video_backend: str | None = None
    cosmos_latent_cache_dir: str | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        base_config = self.create_base_config(assets_dirs, model_config)
        action_key = self.actions_key or self.action_key

        repack_structure: dict[str, Any] = {
            "observation/image": self.image_key,
            "observation/wrist_image": self.wrist_image_key,
            "observation/state": self.state_key,
            "actions": action_key,
        }
        if base_config.prompt_from_task:
            repack_structure["prompt"] = "prompt"
        if self.cosmos_latent_cache_dir is not None:
            repack_structure["episode_index"] = "episode_index"
            repack_structure["frame_index"] = "frame_index"

        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
        data_inputs: list[_transforms.DataTransformFn] = [grasp_policy.GraspInputs(model_type=model_config.model_type)]
        if self.prepend_zero_dims:
            data_inputs.append(grasp_policy.PrependZeroDims(self.prepend_zero_count))
        if self.cosmos_latent_cache_dir is not None:
            data_inputs.append(_transforms.AttachCosmosLatent(_expand_env_path(self.cosmos_latent_cache_dir)))
        data_transforms = _transforms.Group(
            inputs=data_inputs,
            outputs=[
                grasp_policy.GraspOutputs(
                    action_dim=self.action_dim,
                    drop_first_n_dims=self.output_drop_first_n_dims,
                )
            ],
        )

        if self.use_delta_joint_actions:
            # Piper actions are absolute joint/gripper targets in the LeRobot dataset. Train joint deltas
            # relative to the current state and keep the gripper absolute, matching the common pi0 setup.
            delta_action_mask = _transforms.make_bool_mask(*(self.delta_action_mask_dims or (6, -1)))
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        video_backend = self.video_backend or os.environ.get("OPENPI_VIDEO_BACKEND") or None

        return dataclasses.replace(
            base_config,
            root=_expand_env_path(self.root),
            episodes=_load_episode_split(self.splits_path, self.split),
            video_backend=video_backend,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(action_key,),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)
    # Optional held-out data to evaluate during training.
    eval_data: tyro.conf.Suppress[DataConfigFactory | None] = None

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 20_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # How often (in steps) to evaluate eval_data. Set to 0 to disable.
    eval_interval: int = 0
    # Number of held-out batches to average for eval loss.
    eval_num_batches: int = 8
    # If true, evaluate EMA weights when EMA is enabled.
    eval_use_ema: bool = True
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # Maximum number of recent checkpoints to keep. Set to None to keep every saved checkpoint.
    max_checkpoints_to_keep: int | None = 1

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_GRASP_REPO_ID = os.environ.get("OPENPI_GRASP_REPO_ID", "grasp")
_GRASP_DATASET_ROOT = os.environ.get("OPENPI_GRASP_DATASET_ROOT", "/cc/openpi/grasp")
_GRASP_SPLITS_PATH = os.environ.get(
    "OPENPI_GRASP_SPLITS_PATH", "/cc/openpi/grasp_splits/splits.json"
)
_GRASP_PROMPT = os.environ.get("OPENPI_GRASP_PROMPT", "First put the cylinder into the white box, then put the cube into the brown box.")
_PI05_BASE_PARAMS = os.environ.get(
    "OPENPI_PI05_BASE_PARAMS",
    str(_paths.repo_root() / "openpi-assets/checkpoints/pi05_base/params"),
)

# Public accessors used by modular config registries. The underscored names
# remain for compatibility with the original config definitions below.
GRASP_PROMPT = _GRASP_PROMPT
PI05_BASE_PARAMS = _PI05_BASE_PARAMS


def _grasp_action_expert_strict_lora_freeze_filter() -> Filter:
    """Freeze everything except action-expert LoRA and small action heads."""
    trainable = nnx.Any(
        nnx_utils.PathRegex("PaliGemma/llm/.*_1/.*lora.*"),
        nnx_utils.PathRegex("action_in_proj/.*"),
        nnx_utils.PathRegex("time_mlp_in/.*"),
        nnx_utils.PathRegex("time_mlp_out/.*"),
        nnx_utils.PathRegex("action_out_proj/.*"),
    )
    return nnx.Not(trainable)


def _grasp_action_expert_balanced_lora_freeze_filter() -> Filter:
    """Freeze everything except action-expert LoRA plus pi05 action-side adapters."""
    trainable = nnx.Any(
        nnx_utils.PathRegex("PaliGemma/llm/.*_1/.*lora.*"),
        nnx_utils.PathRegex("action_in_proj/.*"),
        nnx_utils.PathRegex("time_mlp_in/.*"),
        nnx_utils.PathRegex("time_mlp_out/.*"),
        nnx_utils.PathRegex("action_out_proj/.*"),
        nnx_utils.PathRegex("PaliGemma/llm/layers/attn_1/kv_einsum/.*"),
        nnx_utils.PathRegex("PaliGemma/llm/layers/attn_1/q_einsum/.*"),
        nnx_utils.PathRegex("PaliGemma/llm/layers/mlp_1/.*"),
    )
    return nnx.Not(trainable)


def _cosmos_steering_lora_freeze_filter() -> Filter:
    """Freeze everything except Cosmos steering modules, action heads, and pi0.5 LoRA."""
    trainable = nnx.Any(
        nnx_utils.PathRegex("dynamics_encoder/.*"),
        nnx_utils.PathRegex("latent_steering/.*"),
        nnx_utils.PathRegex("cosmos_time_embed/.*"),
        nnx_utils.PathRegex("cosmos_view_embed/.*"),
        nnx_utils.PathRegex("PaliGemma/llm/.*lora.*"),
        nnx_utils.PathRegex("action_in_proj/.*"),
        nnx_utils.PathRegex("time_mlp_in/.*"),
        nnx_utils.PathRegex("time_mlp_out/.*"),
        nnx_utils.PathRegex("action_out_proj/.*"),
        nnx_utils.PathRegex("action_prior_encoder/.*"),
    )
    return nnx.Not(trainable)


def cosmos_steering_lora_freeze_filter() -> Filter:
    """Return the trainable filter for the modular Cosmos/WAM configs."""
    return _cosmos_steering_lora_freeze_filter()


_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    # WorldPilot-style LIBERO configs are registered in their own module.
    *_worldpilot_configs.get_configs(),
    # JAX-only RoboCasa subtask completion config, independent of Cosmos/WAM.
    *_robocasa_configs.get_configs(),
    #
    # Fine-tuning local Piper grasp dataset.
    #
    TrainConfig(
        name="pi05_grasp",
        model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=50, discrete_state_input=False),
        data=LeRobotGraspDataConfig(
            repo_id=_GRASP_REPO_ID,
            root=_GRASP_DATASET_ROOT,
            splits_path=_GRASP_SPLITS_PATH,
            split="train",
            default_prompt=_GRASP_PROMPT,
            base_config=DataConfig(prompt_from_task=True),
        ),
        eval_data=LeRobotGraspDataConfig(
            repo_id=_GRASP_REPO_ID,
            root=_GRASP_DATASET_ROOT,
            splits_path=_GRASP_SPLITS_PATH,
            split="test",
            default_prompt=_GRASP_PROMPT,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3e-5,
            decay_steps=30_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        num_train_steps=10_000,
        batch_size=32,
        num_workers=2,
        log_interval=20,
        save_interval=1000,
        eval_interval=1000,
        eval_num_batches=8,
        keep_period=5000,
        wandb_enabled=False,
    ),
    # The local grasp Cosmos plug is also kept separate from the official registry.
    *_cosmos_configs.get_configs(),

    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
