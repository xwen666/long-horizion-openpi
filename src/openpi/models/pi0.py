import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import optax
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.status_head import DoneHead
from openpi.models.status_head import StatusEncoder
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        if config.action_loss_mask is not None and len(config.action_loss_mask) != config.action_dim:
            raise ValueError(
                f"action_loss_mask must have length action_dim={config.action_dim}, "
                f"got {len(config.action_loss_mask)}"
            )
        self.action_loss_mask = config.action_loss_mask
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        dtype = jnp.dtype(config.dtype) if config.dtype is not None else None
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.status_encoder = None
        self.done_head = None
        if config.enable_status_head:
            self.status_encoder = StatusEncoder(
                vla_dim=paligemma_config.width,
                status_dim=config.status_hidden_dim,
                num_layers=config.status_num_layers,
                num_heads=config.status_num_heads,
                ffn_dim=config.status_ffn_dim,
                dropout=config.status_dropout,
                dtype=dtype,
                rngs=rngs,
            )
            self.done_head = DoneHead(
                status_dim=config.status_hidden_dim,
                dropout=config.status_dropout,
                dtype=dtype,
                rngs=rngs,
            )
            logger.info(
                "Enabled JAX status branch: prefix_dim=%d status_dim=%d status_only_trainable=%s",
                paligemma_config.width,
                config.status_hidden_dim,
                config.status_only_trainable,
            )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def action_loss(self, prediction: at.Float[at.Array, "b h d"], target: at.Float[at.Array, "b h d"]):
        """Returns per-step action loss, ignoring padded action channels when configured."""
        squared_error = jnp.square(prediction - target)
        if self.action_loss_mask is None:
            return jnp.mean(squared_error, axis=-1)

        mask = jnp.asarray(self.action_loss_mask, dtype=squared_error.dtype)
        valid_dim_count = jnp.maximum(jnp.sum(mask), jnp.asarray(1.0, dtype=squared_error.dtype))
        return jnp.sum(squared_error * mask, axis=-1) / valid_dim_count

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
        action_prior_token: at.Float[at.Array, "b 1 emb"] | None = None,
        action_prior_mask: at.Bool[at.Array, "b 1"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None

        if action_prior_token is not None:
            if action_prior_token.ndim != 3 or action_prior_token.shape[1] != 1:
                raise ValueError(f"action_prior_token must have shape [B, 1, D], got {action_prior_token.shape}")
            if action_prior_token.shape[0] != noisy_actions.shape[0] or action_prior_token.shape[-1] != action_expert_tokens.shape[-1]:
                raise ValueError(
                    "action_prior_token must share batch and action-expert width with noisy actions: "
                    f"prior={action_prior_token.shape}, actions={action_expert_tokens.shape}"
                )
            if action_prior_mask is None:
                action_prior_mask = jnp.ones((noisy_actions.shape[0], 1), dtype=jnp.bool_)
            if action_prior_mask.shape != (noisy_actions.shape[0], 1):
                raise ValueError(
                    f"action_prior_mask must have shape {(noisy_actions.shape[0], 1)}, got {action_prior_mask.shape}"
                )
            tokens.append(action_prior_token)
            input_mask.append(action_prior_mask.astype(jnp.bool_))
            # Start a new block: actions can attend to the prior, but the prior
            # cannot attend to the noisy action tokens.
            ar_mask += [True]

        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _compute_action_loss_and_prefix(
        self,
        rngs: tuple[at.KeyArrayLike, at.KeyArrayLike, at.KeyArrayLike],
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "b s d"], at.Bool[at.Array, "b s"]]:
        preprocess_rng, noise_rng, time_rng = rngs
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # One big forward pass of prefix + suffix at once. The status-only path
        # below deliberately avoids this action-expert computation.
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return self.action_loss(v_t, u_t), prefix_out, prefix_mask

    def _compute_prefix_output(
        self,
        preprocess_rng: at.KeyArrayLike | None,
        observation: _model.Observation,
        *,
        train: bool,
    ) -> tuple[at.Float[at.Array, "b s d"], at.Bool[at.Array, "b s"]]:
        """Run only the pi0.5 prefix/VLM stream for completion prediction."""
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), _ = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        return prefix_out, prefix_mask

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        action_loss, _, _ = self._compute_action_loss_and_prefix(
            (preprocess_rng, noise_rng, time_rng), observation, actions, train=train
        )
        return action_loss

    @at.typecheck
    def compute_loss_with_status(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
        include_action_loss: bool = True,
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, ""], at.Float[at.Array, "b 1"]]:
        """Compute completion loss, optionally including the action-expert loss.

        Status-only training does not need noisy actions or the action expert. Keeping
        that path separate is important because the completion head is meant to be
        trained on top of the pi0.5 base prefix, independently of Cosmos/WAM and the
        action-learning objective.
        """
        if self.status_encoder is None or self.done_head is None:
            raise RuntimeError("compute_loss_with_status requires config.enable_status_head=True")
        if observation.done_label is None:
            raise ValueError("Status training requires observation.done_label from the DataLoader.")

        preprocess_rng, noise_rng, time_rng, status_rng = jax.random.split(rng, 4)
        if include_action_loss:
            action_loss, prefix_out, prefix_mask = self._compute_action_loss_and_prefix(
                (preprocess_rng, noise_rng, time_rng), observation, actions, train=train
            )
        else:
            prefix_out, prefix_mask = self._compute_prefix_output(preprocess_rng, observation, train=train)
            action_loss = jnp.zeros(
                (observation.state.shape[0], self.action_horizon), dtype=prefix_out.dtype
            )
        encoder_rng, head_rng = jax.random.split(status_rng)
        status = self.status_encoder(
            prefix_out,
            prefix_mask,
            train=train,
            dropout_rng=encoder_rng,
        )
        done_logit = self.done_head(status, train=train, dropout_rng=head_rng)
        done_label = observation.done_label.astype(done_logit.dtype).reshape(-1)
        done_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(done_logit.squeeze(-1), done_label))
        return action_loss, done_loss, done_logit

    def predict_done(self, observation: _model.Observation) -> at.Float[at.Array, "b 1"]:
        """Predict current subtask completion probability from observation and prompt."""
        if self.status_encoder is None or self.done_head is None:
            raise RuntimeError("predict_done requires config.enable_status_head=True")
        prefix_out, prefix_mask = self._compute_prefix_output(None, observation, train=False)
        status = self.status_encoder(prefix_out, prefix_mask, train=False)
        return jax.nn.sigmoid(self.done_head(status, train=False))

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
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
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
