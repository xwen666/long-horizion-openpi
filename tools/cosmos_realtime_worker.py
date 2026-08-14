#!/usr/bin/env python3
"""Persistent JSONL worker for real-time frozen Cosmos/WAM latent extraction.

This process is intentionally separate from the OpenPI policy server.  The
Cosmos Predict stack currently lives in its own Python environment, so the
policy server sends observations over stdin and receives WAM latent tokens over
stdout.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any

import numpy as np

_OPENPI_ROOT = Path(__file__).resolve().parents[1]
if str(_OPENPI_ROOT) not in sys.path:
    sys.path.insert(0, str(_OPENPI_ROOT))

from tools.build_cosmos_latent_cache import _DEFAULT_COSMOS_REPO  # noqa: E402
from tools.build_cosmos_latent_cache import _DEFAULT_FUTURE_SLOT_INDEX  # noqa: E402
from tools.build_cosmos_latent_cache import _DEFAULT_LATENT_FRAME_REPEAT  # noqa: E402
from tools.build_cosmos_latent_cache import _DEFAULT_TARGET_STEP  # noqa: E402
from tools.build_cosmos_latent_cache import CosmosLatentExtractorConfig  # noqa: E402
from tools.build_cosmos_latent_cache import FrozenCosmosLatentExtractor  # noqa: E402
from tools.build_cosmos_latent_cache import _parse_resolution  # noqa: E402
from tools.build_cosmos_latent_cache import run_cosmos_encoder_or_predictor  # noqa: E402

_DEFAULT_LIBERO_COSMOS_RESOLUTION = (224, 224)
_DEFAULT_LIBERO_FLATTENED_DIM = 16 * 28 * 28


def _encode_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_array(value: str) -> np.ndarray:
    raw = base64.b64decode(value.encode("ascii"))
    with io.BytesIO(raw) as buffer:
        return np.load(buffer, allow_pickle=False)


def _make_extractor(args: argparse.Namespace) -> FrozenCosmosLatentExtractor | None:
    if args.allow_dummy_latents:
        return None

    return FrozenCosmosLatentExtractor(
        CosmosLatentExtractorConfig(
            source=args.cosmos_latent_source,
            latent_dim=args.latent_dim,
            cosmos_repo=args.cosmos_repo,
            cosmos_vae_path=args.cosmos_vae_path,
            cosmos_checkpoint=args.cosmos_checkpoint,
            cosmos_model=args.cosmos_model,
            cosmos_config_file=args.cosmos_config_file,
            resolution=args.cosmos_resolution,
            guidance=args.cosmos_guidance,
            num_steps=args.cosmos_num_steps,
            seed=args.cosmos_seed,
            future_slot_index=args.future_slot_index,
            current_repeat_frames=args.current_repeat_frames,
            future_repeat_frames=args.future_repeat_frames,
            num_input_frames=args.cosmos_num_input_frames,
            num_output_frames=args.cosmos_num_output_frames,
            offload_diffusion_model=args.cosmos_offload_diffusion_model,
            offload_text_encoder=args.cosmos_offload_text_encoder,
            offload_tokenizer=args.cosmos_offload_tokenizer,
        )
    )


class CosmosPolicyActionProvider:
    """Optional online Cosmos Policy action provider.

    The provider is kept in the Cosmos Python process. OpenPI receives the
    normalized [16, 7] action chunk over the same JSONL request used for the
    future-scene latent, so one policy decision performs one WAM query.
    """

    def __init__(self, args: argparse.Namespace):
        if args.cosmos_vae_path:
            # The released Cosmos Policy checkpoint does not bundle the Wan
            # VAE. Allow the OpenPI launcher to provide an already downloaded
            # compatible local VAE without touching the policy checkpoint.
            os.environ["COSMOS_POLICY_VAE_PATH"] = args.cosmos_vae_path
        from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import get_action
        from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import get_model
        from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import (
            init_t5_text_embeddings_cache,
        )
        from cosmos_predict2._src.predict2.cosmos_policy.experiments.robot.cosmos_utils import load_dataset_stats

        self._get_action = get_action
        self._action_dim = args.cosmos_policy_action_dim
        self._latent_dim = args.latent_dim
        self._cfg = SimpleNamespace(
            suite="libero",
            config=args.cosmos_policy_config,
            ckpt_path=args.cosmos_policy_checkpoint,
            config_file=args.cosmos_policy_config_file,
            use_wrist_image=True,
            num_wrist_images=1,
            use_third_person_image=True,
            num_third_person_images=1,
            use_proprio=True,
            normalize_proprio=True,
            unnormalize_actions=False,
            trained_with_image_aug=True,
            use_jpeg_compression=True,
            flip_images=True,
            use_variance_scale=False,
            shift=5,
            chunk_size=args.cosmos_policy_chunk_size,
        )
        self._model, _ = get_model(self._cfg)
        stats_path = args.cosmos_policy_dataset_stats
        if not stats_path:
            raise ValueError("--cosmos-policy-dataset-stats is required when Cosmos Policy is enabled.")
        self._dataset_stats = load_dataset_stats(stats_path)
        if args.cosmos_policy_text_embeddings:
            init_t5_text_embeddings_cache(args.cosmos_policy_text_embeddings)
        logging.info("Loaded online Cosmos Policy action provider from %s", args.cosmos_policy_checkpoint)

    @staticmethod
    def _adapt_openpi_libero_state(state: np.ndarray) -> np.ndarray:
        """Maps OpenPI's [xyz, axis-angle, gripper2] state to Cosmos [gripper2, xyz, quat]."""
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape == (9,):
            return state
        if state.shape != (8,):
            raise ValueError(f"Expected LIBERO state with 8 or 9 values, got {state.shape}")
        from scipy.spatial.transform import Rotation

        position = state[:3]
        quaternion_xyzw = Rotation.from_rotvec(state[3:6]).as_quat().astype(np.float32)
        gripper = state[6:8]
        return np.concatenate([gripper, position, quaternion_xyzw], axis=0)

    def infer(
        self,
        *,
        current_images: dict[str, np.ndarray],
        state: np.ndarray | None,
        instruction: str,
        num_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Public WorldPilot precompute/eval passes proprio=None. Cosmos Policy
        # then uses a zero 9D proprio vector, which must match the cache during
        # realtime evaluation of a policy trained on that cache.
        if state is None:
            state = np.zeros((9,), dtype=np.float32)
        observation = {
            "wrist_image": current_images["wrist_image"],
            "primary_image": current_images["image"],
            "proprio": self._adapt_openpi_libero_state(state),
        }
        result = self._get_action(
            self._cfg,
            self._model,
            self._dataset_stats,
            observation,
            instruction,
            seed=1,
            randomize_seed=False,
            num_denoising_steps_action=num_steps,
            # The generated latent sequence still contains the future image
            # slots. Disabling decoding avoids an unnecessary VAE round trip.
            generate_future_state_and_value_in_parallel=False,
        )
        action_prior = np.asarray(result["actions"], dtype=np.float32)
        expected_shape = (self._cfg.chunk_size, self._action_dim)
        if action_prior.ndim != 2 or action_prior.shape != expected_shape:
            raise ValueError(f"Cosmos Policy action prior must have shape {expected_shape}, got {action_prior.shape}")

        generated = result["generated_latent"]
        if generated.ndim != 5:
            raise ValueError(f"Cosmos Policy generated latent must have shape [B, C, T, H, W], got {generated.shape}")

        def extract_slot(name: str) -> np.ndarray:
            slot = result["latent_indices"][name]
            if hasattr(slot, "detach"):
                slot = slot.detach().cpu()
            slot = int(np.asarray(slot).reshape(-1)[0])
            if slot < 0 or slot >= generated.shape[2]:
                raise ValueError(f"Cosmos Policy {name}={slot} is out of range for latent shape {tuple(generated.shape)}")
            token = generated[:, :, slot : slot + 1].reshape(generated.shape[0], -1)
            token = token[0].detach().float().cpu().numpy().astype(np.float32)
            if token.shape != (self._latent_dim,):
                raise ValueError(
                    f"Cosmos Policy {name} latent has shape {token.shape}, expected {(self._latent_dim,)}"
                )
            return token

        # The released WorldPilot LIBERO cache convention is wrist-first,
        # followed by the primary camera. Keep the realtime output in the
        # same order as future_image_latents[t] from that cache.
        future_scene_latent = np.stack(
            [extract_slot("future_wrist_image_latent_idx"), extract_slot("future_image_latent_idx")], axis=0
        )
        return action_prior, future_scene_latent


def _handle_request(
    request: dict[str, Any],
    *,
    extractor: FrozenCosmosLatentExtractor | None,
    action_provider: CosmosPolicyActionProvider | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    start = time.perf_counter()
    views = list(request["views"])
    current_images = {view: _decode_array(request["images"][view]) for view in views}
    instruction = str(request.get("instruction", ""))
    state = _decode_array(request["state"]) if request.get("state") is not None else None
    action_prior = None
    if action_provider is not None:
        # One Cosmos Policy query supplies both future-scene slots and the
        # anticipated action trajectory for this OpenPI action chunk.
        action_prior, latent = action_provider.infer(
            current_images=current_images,
            state=state,
            instruction=instruction,
            num_steps=args.cosmos_policy_num_steps,
        )
    else:
        latent = run_cosmos_encoder_or_predictor(
            current_images=current_images,
            future_images={},
            instruction=instruction,
            views=views,
            extractor=extractor,
            latent_dim=args.latent_dim,
            allow_dummy_latents=args.allow_dummy_latents,
        )
    latent = np.asarray(latent, dtype=np.float32)
    if latent.shape != (len(views), args.latent_dim):
        raise ValueError(f"Expected latent shape {(len(views), args.latent_dim)}, got {latent.shape}")
    elapsed = time.perf_counter() - start
    logging.info(
        "Cosmos realtime request_id=%s views=%d source=%s num_steps=%d elapsed=%.3fs",
        request.get("request_id"),
        len(views),
        args.cosmos_latent_source,
        args.cosmos_num_steps,
        elapsed,
    )

    response = {
        "ok": True,
        "request_id": request.get("request_id"),
        "latent": _encode_array(latent),
        "mask": _encode_array(np.ones((len(views),), dtype=np.bool_)),
        "time_ids": _encode_array(np.full((len(views),), args.target_step, dtype=np.int32)),
        "view_ids": _encode_array(np.arange(len(views), dtype=np.int32)),
    }
    if action_prior is not None:
        response["action_prior"] = _encode_array(action_prior)
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-latent-source", choices=("predict_video2world", "current_vae"), default="predict_video2world")
    parser.add_argument("--cosmos-repo", default=os.environ.get("COSMOS_REPO", _DEFAULT_COSMOS_REPO))
    parser.add_argument("--cosmos-vae-path", default=None)
    parser.add_argument("--cosmos-checkpoint", default=None)
    parser.add_argument("--cosmos-model", default="2B/post-trained")
    parser.add_argument("--cosmos-config-file", default=None)
    parser.add_argument("--cosmos-resolution", type=_parse_resolution, default=_DEFAULT_LIBERO_COSMOS_RESOLUTION)
    parser.add_argument("--cosmos-guidance", type=float, default=7.0)
    parser.add_argument("--cosmos-num-steps", type=int, default=5)
    parser.add_argument("--cosmos-seed", type=int, default=1)
    parser.add_argument("--cosmos-num-input-frames", type=int, default=1)
    parser.add_argument("--cosmos-num-output-frames", type=int, default=77)
    parser.add_argument("--cosmos-offload-diffusion-model", action="store_true")
    parser.add_argument("--cosmos-offload-text-encoder", action="store_true")
    parser.add_argument("--cosmos-offload-tokenizer", action="store_true")
    parser.add_argument("--future-slot-index", type=int, default=_DEFAULT_FUTURE_SLOT_INDEX)
    parser.add_argument("--current-repeat-frames", type=int, default=_DEFAULT_LATENT_FRAME_REPEAT)
    parser.add_argument("--future-repeat-frames", type=int, default=_DEFAULT_LATENT_FRAME_REPEAT)
    parser.add_argument("--target-step", type=int, default=_DEFAULT_TARGET_STEP)
    parser.add_argument("--latent-dim", type=int, default=_DEFAULT_LIBERO_FLATTENED_DIM)
    parser.add_argument("--allow-dummy-latents", action="store_true")
    parser.add_argument("--cosmos-policy-checkpoint", default=None)
    parser.add_argument("--cosmos-policy-config", default="cosmos_predict2_2b_480p_libero__inference_only")
    parser.add_argument(
        "--cosmos-policy-config-file",
        default="cosmos_predict2/_src/predict2/cosmos_policy/config/config.py",
    )
    parser.add_argument("--cosmos-policy-dataset-stats", default=None)
    parser.add_argument("--cosmos-policy-text-embeddings", default=None)
    parser.add_argument("--cosmos-policy-num-steps", type=int, default=5)
    parser.add_argument("--cosmos-policy-chunk-size", type=int, default=16)
    parser.add_argument("--cosmos-policy-action-dim", type=int, default=7)
    args = parser.parse_args()

    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr

    def send(response: dict[str, Any]) -> None:
        protocol_stdout.write(json.dumps(response) + "\n")
        protocol_stdout.flush()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    logging.info("Starting Cosmos realtime latent worker: source=%s latent_dim=%d", args.cosmos_latent_source, args.latent_dim)
    action_provider = CosmosPolicyActionProvider(args) if args.cosmos_policy_checkpoint else None
    # Cosmos Policy already returns the future image latent slots and the
    # anticipated action chunk from one diffusion sample. Do not load a second
    # Cosmos Predict model in that mode.
    extractor = None if action_provider is not None else _make_extractor(args)
    send({"ok": True, "ready": True})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("shutdown"):
                send({"ok": True, "shutdown": True})
                return
            response = _handle_request(request, extractor=extractor, action_provider=action_provider, args=args)
        except Exception as exc:  # pragma: no cover - exercised through subprocess integration
            response = {
                "ok": False,
                "request_id": request.get("request_id") if "request" in locals() else None,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            logging.error("Cosmos realtime request failed: %s", response["traceback"])
        send(response)


if __name__ == "__main__":
    main()
