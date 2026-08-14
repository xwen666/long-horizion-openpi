import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import pi0_config
from openpi.models import pi05_cosmos_config
from openpi.models import worldpilot_action_steering as steering


def test_align_action_prior_interpolates_continuous_and_keeps_gripper_nearest():
    prior = jnp.asarray([[[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]], dtype=jnp.float32)
    aligned, mask = steering.align_action_prior(
        prior,
        target_horizon=5,
        policy_action_dim=2,
        gripper_indices=(1,),
    )

    np.testing.assert_allclose(aligned[0, :, 0], [0.0, 0.5, 1.0, 1.5, 2.0])
    np.testing.assert_allclose(aligned[0, :, 1], [0.0, 0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(mask, np.ones((1, 5), dtype=bool))


def test_align_action_prior_requires_explicit_channel_mapping():
    with pytest.raises(ValueError, match="explicit source_to_policy mapping"):
        steering.align_action_prior(
            jnp.zeros((2, 4, 7), dtype=jnp.float32),
            target_horizon=16,
            policy_action_dim=32,
        )

    aligned, _ = steering.align_action_prior(
        jnp.ones((2, 4, 7), dtype=jnp.float32),
        target_horizon=16,
        policy_action_dim=32,
        source_to_policy=tuple(range(7)),
    )
    assert aligned.shape == (2, 16, 32)
    np.testing.assert_allclose(aligned[..., 7:], 0.0)


def test_action_encoder_returns_one_action_expert_token():
    encoder = steering.WorldPilotActionEncoder(
        action_dim=8,
        action_horizon=16,
        hidden_dim=32,
        action_expert_width=64,
        rngs=nnx.Rngs(0),
    )
    prior = jax.random.normal(jax.random.key(1), (3, 16, 8))
    token = encoder(prior)
    assert token.shape == (3, 1, 64)
    assert len(nnx.state(encoder, nnx.Param).flat_state()) > 0


def test_pi05_suffix_inserts_prior_before_noisy_actions():
    config = pi0_config.Pi0Config(
        pi05=True,
        action_dim=8,
        action_horizon=4,
        max_token_len=8,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=2)
    noisy_actions = jnp.zeros((2, 4, 8), dtype=jnp.float32)
    prior_token = jnp.ones((2, 1, 64), dtype=jnp.float32)
    suffix, mask, ar_mask, _ = model.embed_suffix(
        observation,
        noisy_actions,
        jnp.ones((2,), dtype=jnp.float32),
        prior_token,
        jnp.ones((2, 1), dtype=jnp.bool_),
    )
    assert suffix.shape == (2, 5, 64)
    assert mask.shape == (2, 5)
    # pi0.5 has no explicit state token: [one prior token; first action block; rest].
    np.testing.assert_array_equal(ar_mask, [True, True, False, False, False])


def test_disabled_cosmos_model_uses_original_pi05_rng_and_objective():
    kwargs = {
        "pi05": True,
        "action_dim": 8,
        "action_horizon": 4,
        "max_token_len": 8,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
    }
    base = pi0_config.Pi0Config(**kwargs).create(jax.random.key(0))
    disabled = pi05_cosmos_config.Pi05CosmosConfig(
        **kwargs,
        cosmos_latent_dim=8,
        vlm_width=64,
        use_cosmos_latent_steering=False,
        use_vision_latent_steering=False,
        use_action_steering=False,
    ).create(jax.random.key(0))
    observation = pi0_config.Pi0Config(**kwargs).fake_obs(batch_size=2)
    actions = jax.random.normal(jax.random.key(1), (2, 4, 8))
    loss_base = base.compute_loss(jax.random.key(2), observation, actions, train=False)
    loss_disabled = disabled.compute_loss(jax.random.key(2), observation, actions, train=False)
    np.testing.assert_array_equal(loss_base, loss_disabled)

    noise = jax.random.normal(jax.random.key(3), (2, 4, 8))
    sample_base = base.sample_actions(jax.random.key(4), observation, num_steps=2, noise=noise)
    sample_disabled = disabled.sample_actions(jax.random.key(4), observation, num_steps=2, noise=noise)
    np.testing.assert_array_equal(sample_base, sample_disabled)
