import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import cosmos_realtime
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.shared import paths
from openpi.training import config as _config
import openpi.transforms as transforms


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


class CosmosCacheMode(enum.Enum):
    """How online policy serving should provide Cosmos/WAM latents."""

    none = "none"
    realtime = "realtime"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # For pi0.5+Cosmos/WAM policies, attach online frozen WAM latents at inference time.
    cosmos_cache_mode: CosmosCacheMode = CosmosCacheMode.none

    # Cosmos worker options. The worker runs in the Cosmos Predict venv, separate from this OpenPI process.
    cosmos_python: str = paths.configured_path("COSMOS_PYTHON", "cosmos-predict2.5/.venv/bin/python")
    cosmos_worker_script: str = paths.configured_path("COSMOS_WORKER_SCRIPT", "tools/cosmos_realtime_worker.py")
    cosmos_repo: str = paths.configured_path("COSMOS_REPO", "cosmos-predict2.5")
    cosmos_checkpoint: str = paths.configured_path(
        "COSMOS_CHECKPOINT",
        "cosmos_checkpoints/Cosmos-Predict2.5-2B/base/post-trained/"
        "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
    )
    cosmos_model: str = "2B/post-trained"
    cosmos_vae_path: str | None = None
    cosmos_config_file: str | None = None
    cosmos_latent_source: str = "predict_video2world"
    cosmos_resolution: str = "224,224"
    cosmos_guidance: float = 7.0
    cosmos_num_steps: int = 5
    cosmos_seed: int = 1
    cosmos_num_input_frames: int = 1
    cosmos_num_output_frames: int = 77
    cosmos_offload_diffusion_model: bool = False
    cosmos_offload_text_encoder: bool = False
    cosmos_offload_tokenizer: bool = False
    cosmos_future_slot_index: int = 1
    cosmos_target_step: int = 16
    cosmos_latent_dim: int = 12544
    cosmos_worker_cuda_visible_devices: str | None = None
    cosmos_worker_log_path: str = "/tmp/openpi_cosmos_realtime_worker.log"
    cosmos_allow_dummy_latents: bool = False
    cosmos_policy_checkpoint: str | None = None
    cosmos_policy_config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    cosmos_policy_config_file: str = (
        "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"
    )
    cosmos_policy_dataset_stats: str | None = None
    cosmos_policy_text_embeddings: str | None = None
    cosmos_policy_num_steps: int = 5
    cosmos_policy_chunk_size: int = 16
    cosmos_policy_action_dim: int = 7
    cosmos_policy_use_proprio: bool = False
    cosmos_allow_wam_fallback: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            train_config = _config.get_config(args.policy.config)
            if args.cosmos_cache_mode == CosmosCacheMode.realtime:
                repo_id = getattr(train_config.data, "repo_id", "libero_worldpilot_all") or "libero_worldpilot_all"
                train_config = dataclasses.replace(
                    train_config,
                    data=_config.LeRobotLiberoDataConfig(
                        repo_id=repo_id,
                        extra_delta_transform=False,
                        base_config=_config.DataConfig(prompt_from_task=True),
                    ),
                )
                realtime_config = cosmos_realtime.CosmosRealtimeConfig(
                    cosmos_python=args.cosmos_python,
                    worker_script=args.cosmos_worker_script,
                    cosmos_repo=args.cosmos_repo,
                    cosmos_checkpoint=args.cosmos_checkpoint,
                    cosmos_model=args.cosmos_model,
                    cosmos_vae_path=args.cosmos_vae_path,
                    cosmos_config_file=args.cosmos_config_file,
                    cosmos_latent_source=args.cosmos_latent_source,
                    cosmos_resolution=args.cosmos_resolution,
                    cosmos_guidance=args.cosmos_guidance,
                    cosmos_num_steps=args.cosmos_num_steps,
                    cosmos_seed=args.cosmos_seed,
                    cosmos_num_input_frames=args.cosmos_num_input_frames,
                    cosmos_num_output_frames=args.cosmos_num_output_frames,
                    cosmos_offload_diffusion_model=args.cosmos_offload_diffusion_model,
                    cosmos_offload_text_encoder=args.cosmos_offload_text_encoder,
                    cosmos_offload_tokenizer=args.cosmos_offload_tokenizer,
                    future_slot_index=args.cosmos_future_slot_index,
                    target_step=args.cosmos_target_step,
                    latent_dim=args.cosmos_latent_dim,
                    worker_cuda_visible_devices=args.cosmos_worker_cuda_visible_devices,
                    worker_log_path=args.cosmos_worker_log_path,
                    allow_dummy_latents=args.cosmos_allow_dummy_latents,
                    cosmos_policy_checkpoint=args.cosmos_policy_checkpoint,
                    cosmos_policy_config=args.cosmos_policy_config,
                    cosmos_policy_config_file=args.cosmos_policy_config_file,
                    cosmos_policy_dataset_stats=args.cosmos_policy_dataset_stats,
                    cosmos_policy_text_embeddings=args.cosmos_policy_text_embeddings,
                    cosmos_policy_num_steps=args.cosmos_policy_num_steps,
                    cosmos_policy_chunk_size=args.cosmos_policy_chunk_size,
                    action_prior_dim=args.cosmos_policy_action_dim,
                    cosmos_policy_use_proprio=args.cosmos_policy_use_proprio,
                    allow_wam_fallback=args.cosmos_allow_wam_fallback,
                )
                return _policy_config.create_trained_policy(
                    train_config,
                    args.policy.dir,
                    repack_transforms=transforms.Group(),
                    extra_input_transforms=(cosmos_realtime.AttachRealtimeCosmosLatent(realtime_config),),
                    default_prompt=args.default_prompt,
                )

            return _policy_config.create_trained_policy(train_config, args.policy.dir, default_prompt=args.default_prompt)
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
