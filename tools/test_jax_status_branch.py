"""Small CPU smoke test for the JAX pi0.5 status branch."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from openpi.models import pi0_config
from openpi.models.status_head import DoneHead
from openpi.models.status_head import StatusEncoder


def main() -> None:
    rng = jax.random.key(0)
    encoder = StatusEncoder(2048, 512, 2, 8, 2048, 0.1, rngs=nnx.Rngs(rng))
    head = DoneHead(512, 256, 0.1, rngs=nnx.Rngs(rng))
    prefix = jax.random.normal(jax.random.key(1), (4, 11, 2048), dtype=jnp.float32)
    prefix_mask = jnp.ones((4, 11), dtype=jnp.bool_)
    labels = jnp.asarray([0.0, 1.0, 0.0, 1.0])

    def loss_fn(encoder, head):
        status = encoder(prefix, prefix_mask, train=True, dropout_rng=jax.random.key(2))
        logits = head(status, train=True, dropout_rng=jax.random.key(3)).squeeze(-1)
        return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, labels))

    loss, (encoder_grads, head_grads) = nnx.value_and_grad(loss_fn, argnums=(0, 1))(encoder, head)
    encoder_norm = optax.global_norm(encoder_grads)
    head_norm = optax.global_norm(head_grads)
    if not float(encoder_norm) > 0.0 or not float(head_norm) > 0.0:
        raise AssertionError(f"Expected non-zero status gradients, got encoder={encoder_norm}, head={head_norm}")

    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        enable_status_head=True,
        status_only_trainable=True,
    )
    print(f"loss={float(loss):.6f}")
    print(f"encoder_grad_norm={float(encoder_norm):.6f}")
    print(f"head_grad_norm={float(head_norm):.6f}")
    print(f"freeze_filter={config.get_freeze_filter()}")


if __name__ == "__main__":
    main()
