import bisect
from collections.abc import Iterator, Sequence
import logging
import math
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        for name in ("completion_labels", "completion_hard_negative", "completion_sampling_weights"):
            if hasattr(dataset, name):
                setattr(self, name, getattr(dataset, name))

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class ConcatenatedLeRobotDataset(Dataset):
    def __init__(self, datasets: Sequence[Dataset], *, cache_dirs: Sequence[str] = ()):
        if not datasets:
            raise ValueError("ConcatenatedLeRobotDataset requires at least one dataset.")
        if cache_dirs and len(cache_dirs) != len(datasets):
            raise ValueError("cache_dirs must have the same length as datasets.")
        self._datasets = tuple(datasets)
        self._cache_dirs = tuple(cache_dirs)
        self._cumulative_sizes = []
        total = 0
        for dataset in self._datasets:
            total += len(dataset)
            self._cumulative_sizes.append(total)
        if all(hasattr(dataset, "completion_labels") for dataset in self._datasets):
            self.completion_labels = np.concatenate(
                [np.asarray(dataset.completion_labels) for dataset in self._datasets]
            )
            self.completion_hard_negative = np.concatenate(
                [np.asarray(dataset.completion_hard_negative) for dataset in self._datasets]
            )

    def __getitem__(self, index: SupportsIndex):
        idx = index.__index__()
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        source_index = bisect.bisect_right(self._cumulative_sizes, idx)
        previous_size = 0 if source_index == 0 else self._cumulative_sizes[source_index - 1]
        sample = dict(self._datasets[source_index][idx - previous_size])
        if "prompt" not in sample and "task" in sample:
            sample["prompt"] = sample["task"]
        if self._cache_dirs:
            sample["cosmos_cache_dir"] = self._cache_dirs[source_index]
        return sample

    def __len__(self) -> int:
        return self._cumulative_sizes[-1]


class CompletionLabelDataset(Dataset):
    """Adds segment-tail completion labels using LeRobot annotation columns."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        positive_window: float,
        hard_negative_window: float,
        include_subtask_prompt: bool = False,
    ) -> None:
        if not 0.0 < positive_window <= 1.0:
            raise ValueError("completion_positive_window must be in (0, 1]")
        if hard_negative_window < 0.0:
            raise ValueError("completion_hard_negative_window must be non-negative")
        self._dataset = dataset
        self._include_subtask_prompt = include_subtask_prompt
        self._subtask_prompts = self._build_subtask_prompts(dataset) if include_subtask_prompt else None
        self.completion_labels, self.completion_hard_negative = self._build_labels(
            dataset, positive_window, hard_negative_window
        )

    @staticmethod
    def _column(dataset, name: str) -> np.ndarray | None:
        table = getattr(dataset, "hf_dataset", None)
        if table is None or name not in table.column_names:
            return None
        return np.asarray(table[name])

    @classmethod
    def _build_subtask_prompts(cls, dataset: Dataset) -> np.ndarray:
        subtask_name = cls._column(dataset, "annotation.human.subtask_name")
        if subtask_name is None:
            raise ValueError(
                "Subtask prompts require the RoboCasa annotation column annotation.human.subtask_name."
            )
        task_mapping = getattr(getattr(dataset, "meta", None), "tasks", {})
        task_index = cls._column(dataset, "task_index")
        if task_index is None:
            raise ValueError("Subtask prompts require the LeRobot task_index column.")
        prompts = []
        for task_id, global_task_id in zip(subtask_name.reshape(-1), task_index.reshape(-1), strict=True):
            global_task = task_mapping.get(int(global_task_id), str(int(global_task_id)))
            subtask = task_mapping.get(int(task_id), str(int(task_id)))
            prompts.append(f"{global_task}. Current subtask: {subtask}")
        return np.asarray(prompts, dtype=object)

    @classmethod
    def _build_labels(
        cls, dataset: Dataset, positive_window: float, hard_negative_window: float
    ) -> tuple[np.ndarray, np.ndarray]:
        episode = cls._column(dataset, "episode_index")
        frame = cls._column(dataset, "frame_index")
        subtask = cls._column(dataset, "subtask_idx")
        if episode is None or frame is None or subtask is None:
            raise ValueError(
                "Completion labels require LeRobot columns episode_index, frame_index, and subtask_idx. "
                "Use the RoboCasa composite dataset or provide an explicit completion annotation transform."
            )
        episode = episode.astype(np.int64, copy=False)
        frame = frame.astype(np.int64, copy=False)
        subtask = subtask.astype(np.int64, copy=False)

        # Prefer a real per-frame completion annotation if the dataset contains one.
        for name in ("subtask_done", "annotation.human.subtask_done", "next.subtask_done"):
            explicit = cls._column(dataset, name)
            if explicit is not None:
                labels = explicit.astype(np.float32, copy=False).reshape(-1)
                return labels, np.zeros_like(labels, dtype=np.bool_)

        starts: dict[tuple[int, int], int] = {}
        ends: dict[tuple[int, int], int] = {}
        for ep, st, fr in zip(episode, subtask, frame, strict=True):
            key = (int(ep), int(st))
            starts[key] = min(starts.get(key, int(fr)), int(fr))
            ends[key] = max(ends.get(key, int(fr)), int(fr))

        progress = np.empty(frame.shape[0], dtype=np.float32)
        for index, (ep, st, fr) in enumerate(zip(episode, subtask, frame, strict=True)):
            key = (int(ep), int(st))
            progress[index] = (int(fr) - starts[key]) / max(ends[key] - starts[key], 1)

        threshold = 1.0 - positive_window
        labels = (progress >= threshold).astype(np.float32)
        hard_threshold = max(0.0, threshold - hard_negative_window)
        hard_negative = (progress >= hard_threshold) & ~labels.astype(bool)
        return labels, hard_negative

    def __getitem__(self, index: SupportsIndex):
        item_index = index.__index__()
        sample = dict(self._dataset[item_index])
        sample["done_label"] = np.asarray(self.completion_labels[item_index], dtype=np.float32)
        if self._subtask_prompts is not None:
            sample["prompt"] = self._subtask_prompts[item_index]
        return sample

    def __len__(self) -> int:
        return len(self._dataset)


class CompletionBalancedDistributedSampler(torch.utils.data.Sampler[int]):
    """Weighted, replacement sampler that works with PyTorch DDP."""

    def __init__(self, weights: np.ndarray, *, num_replicas: int, rank: int, seed: int = 0) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.num_samples = math.ceil(len(self.weights) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.multinomial(self.weights, self.total_size, replacement=True, generator=generator)
        return iter(indices[self.rank : self.total_size : self.num_replicas].tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def _completion_sampling_weights(dataset: Dataset, positive_ratio: float) -> np.ndarray | None:
    labels = getattr(dataset, "completion_labels", None)
    if labels is None:
        return None
    labels = np.asarray(labels, dtype=np.float32)
    hard_negative = np.asarray(
        getattr(dataset, "completion_hard_negative", np.zeros_like(labels, dtype=np.bool_)), dtype=np.bool_
    )
    positive = labels > 0.5
    negative = ~positive
    if positive.sum() == 0 or negative.sum() == 0:
        raise ValueError(
            f"Completion sampler needs both classes, got positive={int(positive.sum())} "
            f"negative={int(negative.sum())}"
        )
    positive_ratio = float(positive_ratio)
    if not 0.0 < positive_ratio < 1.0:
        raise ValueError("completion_positive_ratio must be in (0, 1)")

    weights = np.zeros_like(labels, dtype=np.float64)
    weights[positive] = positive_ratio / positive.sum()
    hard = negative & hard_negative
    easy = negative & ~hard_negative
    if hard.any():
        weights[hard] = (1.0 - positive_ratio) * 0.4 / hard.sum()
        if easy.any():
            weights[easy] = (1.0 - positive_ratio) * 0.6 / easy.sum()
        else:
            weights[hard] = (1.0 - positive_ratio) / hard.sum()
    else:
        weights[negative] = (1.0 - positive_ratio) / negative.sum()
    return weights


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    include_completion_labels: bool = True,
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)
    if data_config.source_roots:
        return _create_multi_source_lerobot_dataset(
            data_config, action_horizon, include_completion_labels=include_completion_labels
        )

    dataset_root = pathlib.Path(os.path.expandvars(data_config.root)).expanduser() if data_config.root is not None else None
    if dataset_root is not None:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_root)
    dataset_kwargs = {}
    if data_config.video_backend is not None:
        dataset_kwargs["video_backend"] = data_config.video_backend
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=dataset_root,
        episodes=None if data_config.episodes is None else list(data_config.episodes),
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        **dataset_kwargs,
    )

    _fix_episode_data_index(dataset, data_config.episodes)

    if include_completion_labels and data_config.completion_positive_window is not None:
        dataset = CompletionLabelDataset(
            dataset,
            positive_window=data_config.completion_positive_window,
            hard_negative_window=data_config.completion_hard_negative_window,
            include_subtask_prompt=data_config.completion_include_subtask_prompt,
        )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def _create_multi_source_lerobot_dataset(
    data_config: _config.DataConfig, action_horizon: int, *, include_completion_labels: bool = True
) -> Dataset:
    if not data_config.source_repo_ids or len(data_config.source_repo_ids) != len(data_config.source_roots):
        raise ValueError("source_repo_ids and source_roots must be set and have matching lengths.")
    if data_config.source_episodes and len(data_config.source_episodes) != len(data_config.source_roots):
        raise ValueError("source_episodes must have the same length as source_roots.")
    if data_config.source_cache_dirs and len(data_config.source_cache_dirs) != len(data_config.source_roots):
        raise ValueError("source_cache_dirs must have the same length as source_roots.")

    datasets = []
    for source_index, (source_repo_id, source_root) in enumerate(zip(data_config.source_repo_ids, data_config.source_roots, strict=True)):
        dataset_root = pathlib.Path(os.path.expandvars(source_root)).expanduser()
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(source_repo_id, root=dataset_root)
        dataset_kwargs = {}
        if data_config.video_backend is not None:
            dataset_kwargs["video_backend"] = data_config.video_backend
        episodes = None
        if data_config.source_episodes:
            episodes = tuple(data_config.source_episodes[source_index])
        dataset = lerobot_dataset.LeRobotDataset(
            source_repo_id,
            root=dataset_root,
            episodes=None if episodes is None else list(episodes),
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
            },
            **dataset_kwargs,
        )
        _fix_episode_data_index(dataset, episodes)
        if include_completion_labels and data_config.completion_positive_window is not None:
            dataset = CompletionLabelDataset(
                dataset,
                positive_window=data_config.completion_positive_window,
                hard_negative_window=data_config.completion_hard_negative_window,
                include_subtask_prompt=data_config.completion_include_subtask_prompt,
            )
        datasets.append(dataset)

    return ConcatenatedLeRobotDataset(datasets, cache_dirs=data_config.source_cache_dirs)


def _fix_episode_data_index(dataset, episodes: Sequence[int] | None) -> None:
    # Fix episode_data_index for non-contiguous episode indices.
    # When episodes are filtered (e.g. [0, 1, 2, 4, 5, ..., 199]), episode_data_index
    # is indexed by position (0..N-1), but the LeRobot dataset's __getitem__ uses the raw
    # episode_index from hf_dataset to access it. We expand the tensors to cover the max
    # episode index so that _get_query_indices works correctly.
    if episodes is None:
        return
    eps = sorted(episodes)
    if not eps:
        return
    max_ep = eps[-1]
    new_from = torch.full((max_ep + 1,), -1, dtype=torch.long)
    new_to = torch.full((max_ep + 1,), -1, dtype=torch.long)
    for i, ep in enumerate(eps):
        new_from[ep] = dataset.episode_data_index["from"][i]
        new_to[ep] = dataset.episode_data_index["to"][i]
    dataset.episode_data_index = {"from": new_from, "to": new_to}


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(
        data_config,
        action_horizon,
        model_config,
        include_completion_labels=data_config.completion_positive_window is not None,
    )
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch" or data_config.completion_positive_window is not None:
        completion_weights = (
            _completion_sampling_weights(dataset, data_config.completion_positive_ratio)
            if data_config.completion_positive_window is not None and data_config.completion_balance
            else None
        )
        if framework == "pytorch" and torch.distributed.is_initialized():
            if completion_weights is not None:
                sampler = CompletionBalancedDistributedSampler(
                    completion_weights,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    seed=seed,
                )
            else:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=torch.distributed.get_world_size(),
                    rank=torch.distributed.get_rank(),
                    shuffle=shuffle,
                    drop_last=True,
                )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        elif completion_weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(
                torch.as_tensor(completion_weights, dtype=torch.double),
                num_samples=len(dataset),
                replacement=True,
            )
            local_batch_size = batch_size
        else:
            if framework == "pytorch":
                local_batch_size = batch_size
            else:
                local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    if data_config.completion_positive_window is not None:
        labels = np.asarray(getattr(dataset, "completion_labels", []), dtype=np.float32)
        hard = np.asarray(getattr(dataset, "completion_hard_negative", []), dtype=np.bool_)
        if labels.size:
            logging.info(
                "completion_sampling: positive_count=%d negative_count=%d hard_negative_count=%d "
                "positive_ratio=%.4f balance=%s positive_window=%.3f",
                int((labels > 0.5).sum()),
                int((labels <= 0.5).sum()),
                int(hard.sum()),
                float((labels > 0.5).mean()),
                data_config.completion_balance,
                data_config.completion_positive_window,
            )
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._data_loader.sampler, "set_epoch"):
            self._data_loader.sampler.set_epoch(epoch)

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

    def set_epoch(self, epoch: int) -> None:
        """Forward epoch updates to distributed samplers when present."""
        if hasattr(self._data_loader, "set_epoch"):
            self._data_loader.set_epoch(epoch)
