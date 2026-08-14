import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi05CosmosConfig(pi0_config.Pi0Config):
    pi05: bool = True

    use_cosmos_latent_steering: bool = True
    cosmos_latent_dim: int = 15360
    vlm_width: int = 2048
    num_cosmos_latent_tokens: int | None = None
    max_future_steps: int = 8
    max_views: int = 4
    steering_num_heads: int = 8
    cosmos_condition_dropout: float = 0.3
    use_vision_latent_steering: bool = True
    use_action_steering: bool = False
    action_prior_encoder_type: str = "query_pool"
    action_prior_encoder_hidden_dim: int = 256
    action_prior_encoder_layers: int = 2
    action_prior_encoder_heads: int = 4
    wam_action_horizon: int | None = None
    policy_action_horizon: int | None = None
    action_prior_source_to_policy: tuple[int, ...] | None = None
    action_prior_gripper_indices: tuple[int, ...] = ()
    continuous_action_interpolation: str = "linear"
    gripper_interpolation: str = "nearest"
    wam_condition_dropout: float = 0.3
    vision_prior_dropout: float = 0.3
    action_prior_dropout: float = 0.3
    action_prior_normalization: str = "normalized"
    strict_action_prior_shapes: bool = True
    strict_cache_alignment: bool = True
    debug_shapes: bool = False

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05:
            raise ValueError("Pi05CosmosConfig requires pi05=True.")
        for name, value in (
            ("cosmos_condition_dropout", self.cosmos_condition_dropout),
            ("wam_condition_dropout", self.wam_condition_dropout),
            ("vision_prior_dropout", self.vision_prior_dropout),
            ("action_prior_dropout", self.action_prior_dropout),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0, 1).")
        if self.action_prior_encoder_type != "query_pool":
            raise ValueError("Only action_prior_encoder_type='query_pool' is currently implemented.")
        if self.action_prior_normalization != "normalized":
            raise ValueError(
                "Only normalized WAM action priors are supported. Add an explicit action-space adapter before "
                "using raw or delta actions."
            )
        if self.action_prior_encoder_hidden_dim <= 0:
            raise ValueError("action_prior_encoder_hidden_dim must be positive.")
        if self.action_prior_encoder_layers != 2:
            raise ValueError("The current encoder implements exactly two temporal MLP layers.")
        if self.action_prior_encoder_heads <= 0:
            raise ValueError("action_prior_encoder_heads must be positive.")
        if self.wam_action_horizon is not None and self.wam_action_horizon <= 0:
            raise ValueError("wam_action_horizon must be positive when provided.")
        if self.policy_action_horizon is not None and self.policy_action_horizon != self.action_horizon:
            raise ValueError(
                f"policy_action_horizon={self.policy_action_horizon} must match action_horizon={self.action_horizon}."
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng: at.KeyArrayLike):
        from openpi.models.pi05_cosmos import Pi05Cosmos

        return Pi05Cosmos(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        observation_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        num_latent_tokens = self.num_cosmos_latent_tokens or (self.max_future_steps * self.max_views)
        action_prior_source_dim = (
            len(self.action_prior_source_to_policy)
            if self.action_prior_source_to_policy is not None
            else self.action_dim
        )
        with at.disable_typechecking():
            observation_spec = dataclasses.replace(
                observation_spec,
                cosmos_latent=jax.ShapeDtypeStruct(
                    [batch_size, num_latent_tokens, self.cosmos_latent_dim], jnp.float32
                ),
                cosmos_latent_mask=jax.ShapeDtypeStruct([batch_size, num_latent_tokens], jnp.bool_),
                cosmos_time_ids=jax.ShapeDtypeStruct([batch_size, num_latent_tokens], jnp.int32),
                cosmos_view_ids=jax.ShapeDtypeStruct([batch_size, num_latent_tokens], jnp.int32),
                wam_action_prior=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.wam_action_horizon or self.action_horizon, action_prior_source_dim],
                        jnp.float32,
                    )
                    if self.use_action_steering
                    else None
                ),
                wam_action_prior_mask=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.wam_action_horizon or self.action_horizon],
                        jnp.bool_,
                    )
                    if self.use_action_steering
                    else None
                ),
                wam_action_prior_valid=(
                    jax.ShapeDtypeStruct([batch_size], jnp.bool_) if self.use_action_steering else None
                ),
            )
        return observation_spec, action_spec
