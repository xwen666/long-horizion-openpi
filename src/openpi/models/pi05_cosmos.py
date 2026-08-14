import logging

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import cosmos_latent_steering
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import worldpilot_action_steering
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


class Pi05Cosmos(pi0.Pi0):
    def __init__(self, config, rngs: nnx.Rngs):
        super().__init__(config, rngs)

        paligemma_width = _gemma.get_config(config.paligemma_variant).width
        if config.vlm_width != paligemma_width:
            raise ValueError(
                f"config.vlm_width ({config.vlm_width}) must match PaliGemma width ({paligemma_width}) "
                f"for variant {config.paligemma_variant!r}."
        )

        dtype = jnp.dtype(config.dtype) if config.dtype is not None else None
        self.use_cosmos_latent_steering = config.use_cosmos_latent_steering and config.use_vision_latent_steering
        self.use_action_steering = config.use_action_steering
        self.cosmos_latent_dim = config.cosmos_latent_dim
        self.vlm_width = config.vlm_width
        self.max_future_steps = config.max_future_steps
        self.max_views = config.max_views
        self.cosmos_condition_dropout = config.cosmos_condition_dropout
        self.vision_prior_dropout = config.vision_prior_dropout
        self.action_prior_dropout = config.action_prior_dropout
        self.strict_action_prior_shapes = config.strict_action_prior_shapes
        self.debug_shapes = config.debug_shapes

        self.dynamics_encoder = cosmos_latent_steering.DynamicsEncoder(
            config.cosmos_latent_dim,
            config.vlm_width,
            dtype=dtype,
            rngs=rngs,
        )
        self.latent_steering = cosmos_latent_steering.LatentSteeringBlock(
            config.vlm_width,
            config.steering_num_heads,
            dtype=dtype,
            rngs=rngs,
        )
        self.cosmos_time_embed = nnx.Embed(config.max_future_steps, config.vlm_width, dtype=dtype, rngs=rngs)
        self.cosmos_view_embed = nnx.Embed(config.max_views, config.vlm_width, dtype=dtype, rngs=rngs)

        if self.use_action_steering:
            action_expert_width = _gemma.get_config(config.action_expert_variant).width
            self.action_prior_aligner = worldpilot_action_steering.ActionPriorAligner(
                target_horizon=config.policy_action_horizon or config.action_horizon,
                policy_action_dim=config.action_dim,
                source_to_policy=config.action_prior_source_to_policy,
                gripper_indices=config.action_prior_gripper_indices,
                continuous_interpolation=config.continuous_action_interpolation,
                gripper_interpolation=config.gripper_interpolation,
            )
            self.action_prior_encoder = worldpilot_action_steering.WorldPilotActionEncoder(
                action_dim=config.action_dim,
                action_horizon=config.policy_action_horizon or config.action_horizon,
                hidden_dim=config.action_prior_encoder_hidden_dim,
                action_expert_width=action_expert_width,
                dtype=dtype,
                rngs=rngs,
            )

    def build_cosmos_tokens(
        self, observation: _model.Observation
    ) -> tuple[at.Float[at.Array, "b n d"], at.Bool[at.Array, "b n"]]:
        if observation.cosmos_latent is None:
            raise ValueError("build_cosmos_tokens() requires observation.cosmos_latent.")

        cosmos_latent = observation.cosmos_latent
        if cosmos_latent.ndim != 3:
            raise ValueError(f"cosmos_latent must have shape [B, N, C], got {cosmos_latent.shape}")
        if cosmos_latent.shape[-1] != self.cosmos_latent_dim:
            raise ValueError(
                f"cosmos_latent last dimension must be {self.cosmos_latent_dim}, got {cosmos_latent.shape[-1]}"
            )

        batch_size, latent_len, _ = cosmos_latent.shape
        cosmos_tokens = self.dynamics_encoder(cosmos_latent)

        if observation.cosmos_time_ids is None:
            time_ids = jnp.zeros((batch_size, latent_len), dtype=jnp.int32)
        else:
            time_ids = observation.cosmos_time_ids
        if observation.cosmos_view_ids is None:
            view_ids = jnp.zeros((batch_size, latent_len), dtype=jnp.int32)
        else:
            view_ids = observation.cosmos_view_ids
        if observation.cosmos_latent_mask is None:
            cosmos_mask = jnp.ones((batch_size, latent_len), dtype=jnp.bool_)
        else:
            cosmos_mask = observation.cosmos_latent_mask

        expected_id_shape = cosmos_latent.shape[:2]
        for name, value in (
            ("cosmos_time_ids", time_ids),
            ("cosmos_view_ids", view_ids),
            ("cosmos_latent_mask", cosmos_mask),
        ):
            if value.shape != expected_id_shape:
                raise ValueError(f"{name} must have shape {expected_id_shape}, got {value.shape}")

        time_ids = jnp.clip(time_ids.astype(jnp.int32), 0, self.max_future_steps - 1)
        view_ids = jnp.clip(view_ids.astype(jnp.int32), 0, self.max_views - 1)
        cosmos_tokens = cosmos_tokens + self.cosmos_time_embed(time_ids) + self.cosmos_view_embed(view_ids)
        return cosmos_tokens, cosmos_mask.astype(jnp.bool_)

    def apply_cosmos_latent_steering(
        self,
        observation: _model.Observation,
        prefix_tokens: at.Float[at.Array, "b s d"],
        *,
        train: bool = False,
        dropout_rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b s d"]:
        if not self.use_cosmos_latent_steering or observation.cosmos_latent is None:
            return prefix_tokens
        if prefix_tokens.ndim != 3:
            raise ValueError(f"prefix_tokens must have shape [B, S, D], got {prefix_tokens.shape}")
        if prefix_tokens.shape[-1] != self.vlm_width:
            raise ValueError(f"prefix token width must be {self.vlm_width}, got {prefix_tokens.shape[-1]}")

        cosmos_tokens, cosmos_mask = self.build_cosmos_tokens(observation)
        if train and self.vision_prior_dropout > 0.0:
            if dropout_rng is None:
                raise ValueError("Cosmos condition dropout requires dropout_rng during training.")
            keep_prob = 1.0 - self.vision_prior_dropout
            if keep_prob <= 0.0:
                keep_condition = jnp.zeros((cosmos_tokens.shape[0], 1), dtype=jnp.bool_)
            else:
                keep_condition = jax.random.bernoulli(dropout_rng, keep_prob, (cosmos_tokens.shape[0], 1))
            cosmos_mask = jnp.logical_and(cosmos_mask, keep_condition)
            cosmos_tokens = jnp.where(keep_condition[:, :, None], cosmos_tokens, jnp.zeros_like(cosmos_tokens))
        steered_prefix_tokens = self.latent_steering(prefix_tokens, cosmos_tokens, cosmos_mask)
        if steered_prefix_tokens.shape != prefix_tokens.shape:
            raise ValueError(
                "Cosmos latent steering must preserve prefix token shape: "
                f"before={prefix_tokens.shape}, after={steered_prefix_tokens.shape}"
            )

        if self.debug_shapes:
            logger.info(
                "Cosmos latent steering shapes: prefix_tokens=%s cosmos_latent=%s cosmos_tokens=%s "
                "steered_prefix_tokens=%s",
                prefix_tokens.shape,
                observation.cosmos_latent.shape,
                cosmos_tokens.shape,
                steered_prefix_tokens.shape,
            )

        return steered_prefix_tokens

    def build_action_prior_token(
        self,
        observation: _model.Observation,
        *,
        train: bool = False,
        dropout_rng: at.KeyArrayLike | None = None,
    ) -> tuple[at.Float[at.Array, "b 1 d"] | None, at.Bool[at.Array, "b 1"] | None]:
        """Builds one action-prior token, or returns no condition for fallback."""
        if not self.use_action_steering:
            return None, None
        if observation.wam_action_prior is None:
            if self.strict_action_prior_shapes:
                raise ValueError("Action steering is enabled but observation.wam_action_prior is missing.")
            return None, None

        # WAM/cache is an external frozen condition. Stop gradients at the
        # interface while keeping gradients through the trainable encoder.
        prior = jax.lax.stop_gradient(observation.wam_action_prior)
        if prior.ndim != 3:
            raise ValueError(f"wam_action_prior must have shape [B, H, A], got {prior.shape}")
        batch_size = prior.shape[0]
        if observation.wam_action_prior_mask is None:
            prior_mask = jnp.ones(prior.shape[:2], dtype=jnp.bool_)
        else:
            prior_mask = jnp.asarray(observation.wam_action_prior_mask, dtype=jnp.bool_)
        if prior_mask.shape != prior.shape[:2]:
            raise ValueError(f"wam_action_prior_mask must have shape {prior.shape[:2]}, got {prior_mask.shape}")
        if observation.wam_action_prior_valid is None:
            prior_valid = jnp.ones((batch_size,), dtype=jnp.bool_)
        else:
            prior_valid = jnp.asarray(observation.wam_action_prior_valid, dtype=jnp.bool_)
        if prior_valid.shape != (batch_size,):
            raise ValueError(f"wam_action_prior_valid must have shape {(batch_size,)}, got {prior_valid.shape}")

        aligned_prior, aligned_mask = self.action_prior_aligner(prior, prior_mask)
        action_prior_token = self.action_prior_encoder(aligned_prior, aligned_mask)
        prior_valid = jnp.logical_and(prior_valid, jnp.any(aligned_mask, axis=-1))

        keep = jnp.ones((batch_size, 1), dtype=jnp.bool_)
        if train and self.action_prior_dropout > 0.0:
            if dropout_rng is None:
                raise ValueError("Action-prior dropout requires dropout_rng during training.")
            keep = jax.random.bernoulli(dropout_rng, 1.0 - self.action_prior_dropout, (batch_size, 1))
        prior_valid = jnp.logical_and(prior_valid[:, None], keep)
        action_prior_token = jnp.where(prior_valid[:, :, None], action_prior_token, jnp.zeros_like(action_prior_token))

        if self.debug_shapes:
            jax.debug.print(
                "WorldPilot action steering: raw={raw} aligned={aligned} token={token} keep_ratio={ratio} prior_norm={norm}",
                raw=prior.shape,
                aligned=aligned_prior.shape,
                token=action_prior_token.shape,
                ratio=jnp.mean(keep.astype(jnp.float32)),
                norm=jnp.mean(jnp.linalg.norm(action_prior_token.astype(jnp.float32), axis=-1)),
            )
        return action_prior_token, prior_valid

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # Keep the disabled ablation bit-for-bit aligned with the base Pi0.5
        # implementation, including its RNG split. This makes the baseline
        # comparable to the steering variants under the same seed.
        if not self.use_cosmos_latent_steering and not self.use_action_steering:
            return super().compute_loss(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng, vision_dropout_rng, action_dropout_rng = jax.random.split(rng, 5)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_tokens = self.apply_cosmos_latent_steering(
            observation, prefix_tokens, train=train, dropout_rng=vision_dropout_rng
        )
        action_prior_token, action_prior_mask = self.build_action_prior_token(
            observation, train=train, dropout_rng=action_dropout_rng
        )
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time, action_prior_token, action_prior_mask
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return self.action_loss(v_t, u_t)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        if not self.use_cosmos_latent_steering and not self.use_action_steering:
            return super().sample_actions(rng, observation, num_steps=num_steps, noise=noise)

        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_tokens = self.apply_cosmos_latent_steering(observation, prefix_tokens)
        action_prior_token, action_prior_mask = self.build_action_prior_token(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                action_prior_token,
                action_prior_mask,
            )
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = jnp.broadcast_to(prefix_mask[:, None, :], (batch_size, suffix_tokens.shape[1], prefix_tokens.shape[1]))
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
