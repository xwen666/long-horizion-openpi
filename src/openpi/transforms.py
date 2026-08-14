from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
import pathlib
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats
_logger = logging.getLogger("openpi")
_REPORTED_COSMOS_CACHE_PATHS: set[str] = set()


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class AttachCosmosLatent(DataTransformFn):
    """Attach precomputed Cosmos latent tokens from an npz cache."""

    cache_dir: str | pathlib.Path | None = None
    future_image_key: str = "future_image_latents"
    future_time_id: int = 16
    attach_action_prior: bool = False
    action_prior_key: str = "action_chunk"
    expected_action_horizon: int | None = None
    expected_action_dim: int | None = None
    expected_action_normalized: bool = True
    strict_action_prior_shapes: bool = True
    strict_cache_alignment: bool = True

    def __call__(self, data: DataDict) -> DataDict:
        if "episode_index" not in data or "frame_index" not in data:
            raise ValueError('AttachCosmosLatent requires "episode_index" and "frame_index" in the sample.')

        episode_index = _as_scalar_int(data["episode_index"], "episode_index")
        frame_index = _as_scalar_int(data["frame_index"], "frame_index")
        cache_dir_value = data.get("cosmos_cache_dir", self.cache_dir)
        if cache_dir_value is None:
            raise ValueError('AttachCosmosLatent requires either "cache_dir" or "cosmos_cache_dir" in the sample.')
        cache_dir = pathlib.Path(_as_scalar_str(cache_dir_value, "cosmos_cache_dir"))
        cache_path = cache_dir / f"episode_{episode_index:06d}" / f"frame_{frame_index:06d}.npz"
        episode_cache_path = cache_dir / f"episode_{episode_index:06d}.npz"
        action_prior = None
        loaded_cache_path = cache_path if cache_path.exists() else episode_cache_path
        cache_keys: tuple[str, ...] = ()

        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as cache:
                cache_keys = tuple(cache.files)
                required_keys = ("latent", "mask", "time_ids", "view_ids")
                missing_keys = [key for key in required_keys if key not in cache]
                if missing_keys:
                    raise ValueError(f"Cosmos latent cache {cache_path} is missing keys: {missing_keys}")

                latent = np.asarray(cache["latent"], dtype=np.float32)
                mask = np.asarray(cache["mask"], dtype=np.bool_)
                time_ids = np.asarray(cache["time_ids"], dtype=np.int32)
                view_ids = np.asarray(cache["view_ids"], dtype=np.int32)
                if self.attach_action_prior:
                    action_key = self.action_prior_key if self.action_prior_key in cache else "action_prior"
                    if action_key not in cache:
                        if self.strict_action_prior_shapes:
                            raise ValueError(f"Cosmos action-prior cache {cache_path} is missing key {self.action_prior_key!r}")
                    else:
                        action_prior = np.asarray(cache[action_key], dtype=np.float32)
        elif episode_cache_path.exists():
            with np.load(episode_cache_path, allow_pickle=False) as cache:
                cache_keys = tuple(cache.files)
                if self.future_image_key not in cache:
                    raise ValueError(
                        f"Cosmos episode cache {episode_cache_path} is missing key {self.future_image_key!r}"
                    )

                episode_latents = cache[self.future_image_key]
                if frame_index < 0 or frame_index >= episode_latents.shape[0]:
                    raise IndexError(
                        f"frame_index={frame_index} is out of range for {episode_cache_path} "
                        f"with {episode_latents.shape[0]} frames"
                    )
                latent = np.asarray(episode_latents[frame_index], dtype=np.float32)
                if self.attach_action_prior:
                    action_key = self.action_prior_key if self.action_prior_key in cache else "action_prior"
                    if action_key not in cache:
                        if self.strict_action_prior_shapes:
                            raise ValueError(
                                f"Cosmos episode cache {episode_cache_path} is missing key {self.action_prior_key!r}"
                            )
                    else:
                        episode_action_prior = np.asarray(cache[action_key], dtype=np.float32)
                        if episode_action_prior.ndim != 3:
                            raise ValueError(
                                f"Cosmos action prior must have shape [T, H, A], got {episode_action_prior.shape} "
                                f"in {episode_cache_path}"
                            )
                        if frame_index >= episode_action_prior.shape[0]:
                            message = (
                                f"frame_index={frame_index} is out of range for action prior in {episode_cache_path} "
                                f"with {episode_action_prior.shape[0]} frames"
                            )
                            if self.strict_cache_alignment:
                                raise IndexError(message)
                        else:
                            action_prior = np.asarray(episode_action_prior[frame_index], dtype=np.float32)

            # WorldPilot LIBERO cache stores one npz per episode:
            # future_image_latents[t] has shape [views, channels, height, width].
            if latent.ndim == 3:
                latent = latent[None, ...]
            latent = latent.reshape((latent.shape[0], -1))
            mask = np.ones((latent.shape[0],), dtype=np.bool_)
            time_ids = np.full((latent.shape[0],), self.future_time_id, dtype=np.int32)
            view_ids = np.arange(latent.shape[0], dtype=np.int32)
        else:
            raise FileNotFoundError(
                f"Cosmos latent cache file not found for episode_index={episode_index}, "
                f"frame_index={frame_index}. Tried {cache_path} and {episode_cache_path}."
            )

        if latent.ndim != 2:
            raise ValueError(f"Cosmos latent must have shape [N, C], got {latent.shape} in {cache_path}")
        expected_shape = (latent.shape[0],)
        for name, value in (("mask", mask), ("time_ids", time_ids), ("view_ids", view_ids)):
            if value.shape != expected_shape:
                raise ValueError(f"Cosmos {name} must have shape {expected_shape}, got {value.shape} in {cache_path}")

        result = {
            **data,
            "cosmos_latent": latent,
            "cosmos_latent_mask": mask,
            "cosmos_time_ids": time_ids,
            "cosmos_view_ids": view_ids,
        }

        if self.attach_action_prior and action_prior is not None:
            if action_prior.ndim != 2:
                raise ValueError(f"Cosmos action prior must have shape [H, A], got {action_prior.shape} in {loaded_cache_path}")
            if self.expected_action_horizon is not None and action_prior.shape[0] != self.expected_action_horizon:
                message = (
                    f"Cosmos action prior horizon mismatch in {loaded_cache_path}: "
                    f"expected {self.expected_action_horizon}, got {action_prior.shape[0]}"
                )
                if self.strict_action_prior_shapes:
                    raise ValueError(message)
                _logger.warning(message)
            if self.expected_action_dim is not None and action_prior.shape[1] != self.expected_action_dim:
                message = (
                    f"Cosmos action prior dimension mismatch in {loaded_cache_path}: "
                    f"expected {self.expected_action_dim}, got {action_prior.shape[1]}"
                )
                if self.strict_action_prior_shapes:
                    raise ValueError(message)
                _logger.warning(message)
            if not np.isfinite(action_prior).all():
                raise ValueError(f"Cosmos action prior contains non-finite values in {loaded_cache_path}")
            if self.expected_action_normalized and np.max(np.abs(action_prior)) > 2.0:
                raise ValueError(
                    f"Cosmos action prior in {loaded_cache_path} is outside the expected normalized range: "
                    f"min={action_prior.min()}, max={action_prior.max()}. "
                    "Do not pass raw or delta actions without an explicit adapter."
                )
            result.update(
                wam_action_prior=action_prior,
                wam_action_prior_mask=np.ones((action_prior.shape[0],), dtype=np.bool_),
                wam_action_prior_valid=np.asarray(1, dtype=np.bool_),
            )
            cache_key = str(loaded_cache_path)
            if cache_key not in _REPORTED_COSMOS_CACHE_PATHS:
                _REPORTED_COSMOS_CACHE_PATHS.add(cache_key)
                _logger.info(
                    "WorldPilot action cache: path=%s keys=%s action_prior_shape=%s dtype=%s "
                    "min=%.5f max=%.5f mean=%.5f std=%.5f",
                    loaded_cache_path,
                    cache_keys,
                    action_prior.shape,
                    action_prior.dtype,
                    float(action_prior.min()),
                    float(action_prior.max()),
                    float(action_prior.mean()),
                    float(action_prior.std()),
                )

        return result


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def _as_scalar_int(value, name: str) -> int:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar, got shape {value.shape}")
    return int(value.reshape(()).item())


def _as_scalar_str(value, name: str) -> str:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"{name} must be a scalar, got shape {value.shape}")
    item = value.reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
