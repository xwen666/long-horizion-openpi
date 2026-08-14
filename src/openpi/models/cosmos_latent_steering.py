import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


class DynamicsEncoder(nnx.Module):
    """Projects Cosmos future-scene latents into the VLM hidden width."""

    def __init__(
        self,
        cosmos_latent_dim: int,
        vlm_width: int,
        *,
        hidden_dim: int | None = None,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        hidden_dim = hidden_dim or vlm_width
        self.input_proj = nnx.Linear(cosmos_latent_dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.output_proj = nnx.Linear(hidden_dim, vlm_width, dtype=dtype, rngs=rngs)
        self.cosmos_latent_dim = cosmos_latent_dim
        self.vlm_width = vlm_width

    def __call__(self, z: at.Float[at.Array, "b n c"]) -> at.Float[at.Array, "b n d"]:
        # z: [B, N, cosmos_latent_dim] -> [B, N, vlm_width]
        z = self.input_proj(z)
        z = nnx.swish(z)
        z = self.output_proj(z)
        return nnx.swish(z)


class LatentSteeringBlock(nnx.Module):
    """Residual cross-attention from pi0.5 prefix tokens to Cosmos latent tokens."""

    def __init__(
        self,
        vlm_width: int,
        num_heads: int,
        *,
        dtype: jnp.dtype | None = None,
        rngs: nnx.Rngs,
    ):
        if vlm_width % num_heads != 0:
            raise ValueError(f"vlm_width ({vlm_width}) must be divisible by num_heads ({num_heads})")

        self.q_proj = nnx.Linear(vlm_width, vlm_width, dtype=dtype, rngs=rngs)
        self.k_proj = nnx.Linear(vlm_width, vlm_width, dtype=dtype, rngs=rngs)
        self.v_proj = nnx.Linear(vlm_width, vlm_width, dtype=dtype, rngs=rngs)
        self.out_proj = nnx.Linear(vlm_width, vlm_width, dtype=dtype, rngs=rngs)
        self.vlm_width = vlm_width
        self.num_heads = num_heads
        self.head_dim = vlm_width // num_heads

    def __call__(
        self,
        h: at.Float[at.Array, "b s d"],
        z: at.Float[at.Array, "b n d"],
        z_mask: at.Bool[at.Array, "b n"] | None = None,
    ) -> at.Float[at.Array, "b s d"]:
        # h: [B, S, D] provides Q. z: [B, N, D] provides K/V.
        batch_size, seq_len, width = h.shape
        latent_len = z.shape[1]
        if width != self.vlm_width or z.shape[-1] != self.vlm_width:
            raise ValueError(
                f"LatentSteeringBlock expected hidden width {self.vlm_width}, "
                f"got h.shape={h.shape}, z.shape={z.shape}"
            )

        q = self.q_proj(h).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(z).reshape(batch_size, latent_len, self.num_heads, self.head_dim)
        v = self.v_proj(z).reshape(batch_size, latent_len, self.num_heads, self.head_dim)

        # [B, S, H, Dh] -> [B, H, S, Dh], [B, N, H, Dh] -> [B, H, N, Dh]
        q = jnp.swapaxes(q, 1, 2)
        k = jnp.swapaxes(k, 1, 2)
        v = jnp.swapaxes(v, 1, 2)

        logits = jnp.einsum("bhsd,bhnd->bhsn", q, k) / math.sqrt(self.head_dim)
        original_z_mask = z_mask
        if z_mask is not None:
            # z_mask: [B, N] -> [B, 1, 1, N]
            z_mask = z_mask[:, None, None, :]
            logits = jnp.where(z_mask, logits, jnp.asarray(-1e9, dtype=logits.dtype))

        weights = jax.nn.softmax(logits, axis=-1)
        if z_mask is not None:
            weights = jnp.where(z_mask, weights, jnp.zeros_like(weights))

        delta_h = jnp.einsum("bhsn,bhnd->bhsd", weights, v)
        # [B, H, S, Dh] -> [B, S, H, Dh] -> [B, S, D]
        delta_h = jnp.swapaxes(delta_h, 1, 2).reshape(batch_size, seq_len, self.vlm_width)
        delta_h = self.out_proj(delta_h)
        if original_z_mask is not None:
            has_condition = jnp.any(original_z_mask, axis=-1)[:, None, None]
            delta_h = jnp.where(has_condition, delta_h, jnp.zeros_like(delta_h))

        # Residual steering keeps h shape unchanged: [B, S, D] -> [B, S, D].
        steered_h = h + delta_h
        return steered_h.astype(h.dtype)


class CosmosLatentProjector(nnx.Module):
    """Compatibility wrapper around DynamicsEncoder."""

    def __init__(self, cosmos_latent_dim: int, vlm_width: int, *, dtype: jnp.dtype | None = None, rngs: nnx.Rngs):
        self.encoder = DynamicsEncoder(cosmos_latent_dim, vlm_width, dtype=dtype, rngs=rngs)

    def __call__(self, z: at.Float[at.Array, "b n c"]) -> at.Float[at.Array, "b n d"]:
        return self.encoder(z)
