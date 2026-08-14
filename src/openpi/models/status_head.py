"""JAX/NNX subtask-completion head for the frozen pi0.5 prefix."""

import jax
import jax.numpy as jnp
from flax import nnx

from openpi.shared import array_typing as at


class StatusEncoder(nnx.Module):
    """Compresses VLA prefix tokens into one completion-status representation."""

    def __init__(
        self,
        vla_dim: int,
        status_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        if status_dim <= 0 or num_layers <= 0 or num_heads <= 0 or ffn_dim <= 0:
            raise ValueError("Status encoder dimensions must be positive.")
        if status_dim % num_heads != 0:
            raise ValueError(f"status_dim={status_dim} must be divisible by num_heads={num_heads}.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_proj = nnx.Linear(vla_dim, status_dim, dtype=dtype, rngs=rngs)
        self.status_token = nnx.Param(jax.random.normal(rngs.params(), (1, 1, status_dim)) * 0.02)
        self.output_norm = nnx.LayerNorm(status_dim, dtype=dtype, rngs=rngs)
        self.num_layers = num_layers

        for layer_index in range(num_layers):
            setattr(
                self,
                f"attention_{layer_index}",
                nnx.MultiHeadAttention(
                    num_heads=num_heads,
                    in_features=status_dim,
                    qkv_features=status_dim,
                    out_features=status_dim,
                    dtype=dtype,
                    dropout_rate=0.0,
                    rngs=rngs,
                ),
            )
            setattr(self, f"attention_norm_{layer_index}", nnx.LayerNorm(status_dim, dtype=dtype, rngs=rngs))
            setattr(self, f"ffn_in_{layer_index}", nnx.Linear(status_dim, ffn_dim, dtype=dtype, rngs=rngs))
            setattr(self, f"ffn_out_{layer_index}", nnx.Linear(ffn_dim, status_dim, dtype=dtype, rngs=rngs))
            setattr(self, f"ffn_norm_{layer_index}", nnx.LayerNorm(status_dim, dtype=dtype, rngs=rngs))
            setattr(self, f"attention_dropout_{layer_index}", nnx.Dropout(dropout, rngs=rngs))
            setattr(self, f"ffn_dropout_{layer_index}", nnx.Dropout(dropout, rngs=rngs))

    def __call__(
        self,
        prefix_out: at.Float[at.Array, "b m d"],
        prefix_valid_mask: at.Bool[at.Array, "b m"],
        *,
        train: bool = False,
        dropout_rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b d"]:
        if prefix_out.ndim != 3:
            raise ValueError(f"prefix_out must have shape [B, M, D], got {prefix_out.shape}")
        if prefix_valid_mask.shape != prefix_out.shape[:2]:
            raise ValueError(
                f"prefix_valid_mask must have shape {prefix_out.shape[:2]}, got {prefix_valid_mask.shape}"
            )
        if train and dropout_rng is None and any(
            getattr(self, f"attention_dropout_{i}").rate > 0.0 for i in range(self.num_layers)
        ):
            raise ValueError("Status dropout requires dropout_rng during training.")

        x = self.input_proj(prefix_out)
        batch_size = x.shape[0]
        status_token = jnp.broadcast_to(self.status_token.value, (batch_size, 1, x.shape[-1])).astype(x.dtype)
        x = jnp.concatenate([x, status_token], axis=1)
        valid_mask = jnp.concatenate(
            [prefix_valid_mask.astype(jnp.bool_), jnp.ones((batch_size, 1), dtype=jnp.bool_)], axis=1
        )
        attention_mask = valid_mask[:, None, :, None] & valid_mask[:, None, None, :]
        dropout_rngs = nnx.Rngs(dropout=dropout_rng) if train and dropout_rng is not None else None

        for layer_index in range(self.num_layers):
            attention = getattr(self, f"attention_{layer_index}")
            attention_norm = getattr(self, f"attention_norm_{layer_index}")
            ffn_in = getattr(self, f"ffn_in_{layer_index}")
            ffn_out = getattr(self, f"ffn_out_{layer_index}")
            ffn_norm = getattr(self, f"ffn_norm_{layer_index}")
            attention_dropout = getattr(self, f"attention_dropout_{layer_index}")
            ffn_dropout = getattr(self, f"ffn_dropout_{layer_index}")

            attention_out = attention(x, mask=attention_mask, deterministic=not train, decode=False)
            attention_out = attention_dropout(attention_out, deterministic=not train, rngs=dropout_rngs)
            x = attention_norm(x + attention_out)

            ffn_out_value = ffn_out(nnx.gelu(ffn_in(x)))
            ffn_out_value = ffn_dropout(ffn_out_value, deterministic=not train, rngs=dropout_rngs)
            x = ffn_norm(x + ffn_out_value)

        return self.output_norm(x[:, -1, :])


class DoneHead(nnx.Module):
    """Maps the status representation to one completion logit."""

    def __init__(
        self,
        status_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        self.input_proj = nnx.Linear(status_dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.output_proj = nnx.Linear(hidden_dim, 1, dtype=dtype, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(
        self,
        status: at.Float[at.Array, "b d"],
        *,
        train: bool = False,
        dropout_rng: at.KeyArrayLike | None = None,
    ) -> at.Float[at.Array, "b 1"]:
        if train and self.dropout.rate > 0.0 and dropout_rng is None:
            raise ValueError("Done-head dropout requires dropout_rng during training.")
        hidden = nnx.gelu(self.input_proj(status))
        hidden = self.dropout(
            hidden,
            deterministic=not train,
            rngs=nnx.Rngs(dropout=dropout_rng) if train and dropout_rng is not None else None,
        )
        return self.output_proj(hidden)
