"""WorldPilot-style action-prior alignment and trajectory encoding.

The WAM action trajectory is an external, frozen condition.  This module only
turns it into one token for the pi0.5 action expert; it does not predict or
modify the final action trajectory.
"""

from collections.abc import Sequence
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


def _validate_mapping(source_dim: int, policy_dim: int, source_to_policy: Sequence[int] | None) -> tuple[int, ...]:
    if source_to_policy is None:
        if source_dim != policy_dim:
            raise ValueError(
                "WAM action dimension does not match policy action dimension: "
                f"source={source_dim}, policy={policy_dim}. Provide an explicit source_to_policy mapping; "
                "implicit truncation or zero-padding is forbidden."
            )
        return tuple(range(source_dim))

    mapping = tuple(int(index) for index in source_to_policy)
    if len(mapping) != source_dim:
        raise ValueError(
            f"source_to_policy must contain one target index per source channel: "
            f"source_dim={source_dim}, mapping_len={len(mapping)}"
        )
    if len(set(mapping)) != len(mapping):
        raise ValueError(f"source_to_policy contains duplicate target indices: {mapping}")
    if any(index < 0 or index >= policy_dim for index in mapping):
        raise ValueError(f"source_to_policy indices must be in [0, {policy_dim}), got {mapping}")
    return mapping


def _resample_axis(values: jax.Array, target_horizon: int, *, nearest: bool) -> jax.Array:
    """Resamples axis 1 while preserving all other axes and the input dtype."""
    source_horizon = values.shape[1]
    if source_horizon <= 0:
        raise ValueError("WAM action prior must contain at least one time step.")
    if target_horizon <= 0:
        raise ValueError(f"target_horizon must be positive, got {target_horizon}")
    if source_horizon == target_horizon:
        return values
    if source_horizon == 1:
        return jnp.broadcast_to(values, values.shape[:1] + (target_horizon,) + values.shape[2:])

    positions = jnp.linspace(0.0, source_horizon - 1.0, target_horizon, dtype=jnp.float32)
    if nearest:
        indices = jnp.rint(positions).astype(jnp.int32)
        return jnp.take(values, indices, axis=1)

    lower = jnp.floor(positions).astype(jnp.int32)
    upper = jnp.ceil(positions).astype(jnp.int32)
    weight = (positions - lower.astype(jnp.float32)).astype(values.dtype)
    lower_values = jnp.take(values, lower, axis=1)
    upper_values = jnp.take(values, upper, axis=1)
    return lower_values * (1 - weight)[None, :, None] + upper_values * weight[None, :, None]


def align_action_prior(
    action_prior: at.Float[at.Array, "b h a"],
    *,
    target_horizon: int,
    policy_action_dim: int,
    source_to_policy: Sequence[int] | None = None,
    gripper_indices: Sequence[int] = (),
    continuous_interpolation: str = "linear",
    gripper_interpolation: str = "nearest",
    action_prior_mask: at.Bool[at.Array, "b h"] | None = None,
) -> tuple[at.Float[at.Array, "b k a"], at.Bool[at.Array, "b k"]]:
    """Aligns a WAM action trajectory to the policy action space.

    ``source_to_policy`` maps source channel ``i`` to policy channel
    ``source_to_policy[i]``.  The mapping is mandatory whenever the channel
    counts differ. Unmapped policy channels are intentionally zero because the
    caller explicitly declared that they are not represented by WAM.
    """
    action_prior = jnp.asarray(action_prior)
    if action_prior.ndim != 3:
        raise ValueError(f"action_prior must have shape [B, H, A], got {action_prior.shape}")
    if continuous_interpolation != "linear":
        raise ValueError(f"Unsupported continuous_interpolation={continuous_interpolation!r}; expected 'linear'.")
    if gripper_interpolation != "nearest":
        raise ValueError(f"Unsupported gripper_interpolation={gripper_interpolation!r}; expected 'nearest'.")

    source_dim = action_prior.shape[-1]
    mapping = _validate_mapping(source_dim, policy_action_dim, source_to_policy)
    gripper_indices = tuple(int(index) for index in gripper_indices)
    if any(index < 0 or index >= source_dim for index in gripper_indices):
        raise ValueError(f"gripper_indices must be source-channel indices in [0, {source_dim}), got {gripper_indices}")

    aligned_source = _resample_axis(action_prior, target_horizon, nearest=False)
    if gripper_indices:
        nearest_source = _resample_axis(action_prior, target_horizon, nearest=True)
        aligned_source = aligned_source.at[..., list(gripper_indices)].set(nearest_source[..., list(gripper_indices)])

    if action_prior_mask is None:
        aligned_mask = jnp.ones((action_prior.shape[0], target_horizon), dtype=jnp.bool_)
    else:
        action_prior_mask = jnp.asarray(action_prior_mask, dtype=jnp.bool_)
        if action_prior_mask.shape != action_prior.shape[:2]:
            raise ValueError(
                f"action_prior_mask must have shape {action_prior.shape[:2]}, got {action_prior_mask.shape}"
            )
        aligned_mask = _resample_axis(action_prior_mask[:, :, None], target_horizon, nearest=True)[..., 0]

    if source_dim == policy_action_dim and mapping == tuple(range(source_dim)):
        aligned = aligned_source
    else:
        aligned = jnp.zeros((*aligned_source.shape[:-1], policy_action_dim), dtype=action_prior.dtype)
        aligned = aligned.at[..., jnp.asarray(mapping)].set(aligned_source)
    return aligned, aligned_mask


@dataclasses.dataclass(eq=True, init=False)
class ActionPriorAligner:
    """Configuration-bound, parameter-free action horizon adapter."""

    target_horizon: int
    policy_action_dim: int
    source_to_policy: tuple[int, ...] | None
    gripper_indices: tuple[int, ...]
    continuous_interpolation: str
    gripper_interpolation: str

    def __init__(
        self,
        *,
        target_horizon: int,
        policy_action_dim: int,
        source_to_policy: Sequence[int] | None = None,
        gripper_indices: Sequence[int] = (),
        continuous_interpolation: str = "linear",
        gripper_interpolation: str = "nearest",
    ):
        self.target_horizon = int(target_horizon)
        self.policy_action_dim = int(policy_action_dim)
        self.source_to_policy = None if source_to_policy is None else tuple(source_to_policy)
        self.gripper_indices = tuple(gripper_indices)
        self.continuous_interpolation = continuous_interpolation
        self.gripper_interpolation = gripper_interpolation

    def __call__(
        self,
        action_prior: at.Float[at.Array, "b h a"],
        action_prior_mask: at.Bool[at.Array, "b h"] | None = None,
    ) -> tuple[at.Float[at.Array, "b k a"], at.Bool[at.Array, "b k"]]:
        return align_action_prior(
            action_prior,
            target_horizon=self.target_horizon,
            policy_action_dim=self.policy_action_dim,
            source_to_policy=self.source_to_policy,
            gripper_indices=self.gripper_indices,
            continuous_interpolation=self.continuous_interpolation,
            gripper_interpolation=self.gripper_interpolation,
            action_prior_mask=action_prior_mask,
        )


class WorldPilotActionEncoder(nnx.Module):
    """Compresses [B, K, A] into one learned action-expert-width token.

    A learned query performs masked temporal pooling after a two-layer
    per-step MLP. This is intentionally trajectory-level: the action expert
    receives one token, never K WAM tokens.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int,
        action_expert_width: int,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        if hidden_dim <= 0 or action_expert_width <= 0:
            raise ValueError("Action encoder dimensions must be positive.")
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.hidden_dim = hidden_dim
        self.action_expert_width = action_expert_width
        self.input_proj = nnx.Linear(action_dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.input_norm = nnx.LayerNorm(hidden_dim, dtype=dtype, rngs=rngs)
        self.step_mlp_in = nnx.Linear(hidden_dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.step_mlp_out = nnx.Linear(hidden_dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.position_embed = nnx.Embed(action_horizon, hidden_dim, dtype=dtype, rngs=rngs)
        self.query_embed = nnx.Embed(1, hidden_dim, dtype=dtype, rngs=rngs)
        self.output_proj = nnx.Linear(hidden_dim, action_expert_width, dtype=dtype, rngs=rngs)
        self.output_norm = nnx.LayerNorm(action_expert_width, dtype=dtype, rngs=rngs)
        self.action_prior_type_embed = nnx.Embed(1, action_expert_width, dtype=dtype, rngs=rngs)

    def __call__(
        self,
        aligned_action_prior: at.Float[at.Array, "b k a"],
        action_prior_mask: at.Bool[at.Array, "b k"] | None = None,
    ) -> at.Float[at.Array, "b 1 d"]:
        if aligned_action_prior.ndim != 3:
            raise ValueError(f"aligned_action_prior must have shape [B, K, A], got {aligned_action_prior.shape}")
        batch_size, horizon, action_dim = aligned_action_prior.shape
        if horizon != self.action_horizon or action_dim != self.action_dim:
            raise ValueError(
                "Action encoder input shape mismatch: "
                f"expected [B, {self.action_horizon}, {self.action_dim}], got {aligned_action_prior.shape}"
            )
        if action_prior_mask is None:
            action_prior_mask = jnp.ones((batch_size, horizon), dtype=jnp.bool_)
        elif action_prior_mask.shape != (batch_size, horizon):
            raise ValueError(
                f"action_prior_mask must have shape {(batch_size, horizon)}, got {action_prior_mask.shape}"
            )

        positions = jnp.arange(horizon, dtype=jnp.int32)[None, :]
        tokens = self.input_norm(self.input_proj(aligned_action_prior))
        tokens = tokens + self.position_embed(jnp.broadcast_to(positions, (batch_size, horizon)))
        tokens = tokens + self.step_mlp_out(nnx.swish(self.step_mlp_in(tokens)))

        query = self.query_embed(jnp.zeros((batch_size, 1), dtype=jnp.int32))[:, 0]
        scores = jnp.einsum("bkd,bd->bk", tokens, query) / jnp.sqrt(jnp.asarray(self.hidden_dim, dtype=tokens.dtype))
        scores = jnp.where(action_prior_mask, scores, jnp.asarray(-1e4, dtype=scores.dtype))
        weights = jax.nn.softmax(scores, axis=-1)
        weights = jnp.where(action_prior_mask, weights, jnp.zeros_like(weights))
        weights = weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), jnp.asarray(1e-6, dtype=weights.dtype))
        pooled = jnp.einsum("bk,bkd->bd", weights, tokens)

        token = self.output_norm(self.output_proj(nnx.swish(pooled)))[:, None, :]
        return token + self.action_prior_type_embed(jnp.zeros((batch_size, 1), dtype=jnp.int32))
