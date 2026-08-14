"""Runtime Cosmos/WAM latent provider for policy inference."""

from __future__ import annotations

import atexit
import base64
import contextlib
import dataclasses
import io
import json
import logging
import os
import pathlib
import subprocess
import threading
from typing import Any

import numpy as np

from openpi import transforms
from openpi.shared import paths

_logger = logging.getLogger("openpi")

_DEFAULT_COSMOS_PYTHON = paths.configured_path("COSMOS_PYTHON", "cosmos-predict2.5/.venv/bin/python")
_DEFAULT_WORKER = paths.configured_path("COSMOS_WORKER_SCRIPT", "tools/cosmos_realtime_worker.py")
_DEFAULT_COSMOS_REPO = paths.configured_path("COSMOS_REPO", "cosmos-predict2.5")
_DEFAULT_COSMOS_CHECKPOINT = paths.configured_path(
    "COSMOS_CHECKPOINT",
    "cosmos_checkpoints/Cosmos-Predict2.5-2B/base/post-trained/"
    "81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt",
)


def _prepend_library_paths(env: dict[str, str], paths: list[pathlib.Path]) -> None:
    existing = [value for value in env.get("LD_LIBRARY_PATH", "").split(":") if value]
    additions = [str(path) for path in paths if path.exists()]
    if additions:
        env["LD_LIBRARY_PATH"] = ":".join(additions + existing)


def _encode_array(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _decode_array(value: str) -> np.ndarray:
    raw = base64.b64decode(value.encode("ascii"))
    with io.BytesIO(raw) as buffer:
        return np.load(buffer, allow_pickle=False)


@dataclasses.dataclass(frozen=True)
class CosmosRealtimeConfig:
    """Configuration for the external real-time Cosmos/WAM worker."""

    cosmos_python: str = _DEFAULT_COSMOS_PYTHON
    worker_script: str = _DEFAULT_WORKER
    cosmos_repo: str = _DEFAULT_COSMOS_REPO
    cosmos_checkpoint: str = _DEFAULT_COSMOS_CHECKPOINT
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
    cosmos_policy_checkpoint: str | None = None
    cosmos_policy_config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    cosmos_policy_config_file: str = (
        "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"
    )
    cosmos_policy_dataset_stats: str | None = None
    cosmos_policy_text_embeddings: str | None = None
    cosmos_policy_num_steps: int = 5
    cosmos_policy_chunk_size: int = 16
    action_prior_dim: int = 7
    # The released WorldPilot cache was generated with proprio=None, which
    # Cosmos Policy represents as a zero 9D proprio vector. Keep realtime
    # inference on that distribution unless explicitly requested otherwise.
    cosmos_policy_use_proprio: bool = False
    allow_wam_fallback: bool = False
    future_slot_index: int = 1
    target_step: int = 16
    latent_dim: int = 12544
    worker_cuda_visible_devices: str | None = None
    worker_log_path: str = "/tmp/openpi_cosmos_realtime_worker.log"
    allow_dummy_latents: bool = False


class CosmosRealtimeClient:
    def __init__(self, config: CosmosRealtimeConfig):
        self._config = config
        self._process: subprocess.Popen[str] | None = None
        self._stderr_file = None
        self._lock = threading.Lock()
        self._next_request_id = 0
        atexit.register(self.close)

    def _command(self) -> list[str]:
        cfg = self._config
        cmd = [
            cfg.cosmos_python,
            cfg.worker_script,
            "--cosmos-latent-source",
            cfg.cosmos_latent_source,
            "--cosmos-repo",
            cfg.cosmos_repo,
            "--cosmos-checkpoint",
            cfg.cosmos_checkpoint,
            "--cosmos-model",
            cfg.cosmos_model,
            "--cosmos-resolution",
            cfg.cosmos_resolution,
            "--cosmos-guidance",
            str(cfg.cosmos_guidance),
            "--cosmos-num-steps",
            str(cfg.cosmos_num_steps),
            "--cosmos-seed",
            str(cfg.cosmos_seed),
            "--cosmos-num-input-frames",
            str(cfg.cosmos_num_input_frames),
            "--cosmos-num-output-frames",
            str(cfg.cosmos_num_output_frames),
            "--future-slot-index",
            str(cfg.future_slot_index),
            "--target-step",
            str(cfg.target_step),
            "--latent-dim",
            str(cfg.latent_dim),
        ]
        if cfg.cosmos_policy_checkpoint:
            cmd.extend(["--cosmos-policy-checkpoint", cfg.cosmos_policy_checkpoint])
            cmd.extend(["--cosmos-policy-config", cfg.cosmos_policy_config])
            cmd.extend(["--cosmos-policy-config-file", cfg.cosmos_policy_config_file])
            if cfg.cosmos_policy_dataset_stats:
                cmd.extend(["--cosmos-policy-dataset-stats", cfg.cosmos_policy_dataset_stats])
            if cfg.cosmos_policy_text_embeddings:
                cmd.extend(["--cosmos-policy-text-embeddings", cfg.cosmos_policy_text_embeddings])
            cmd.extend(["--cosmos-policy-num-steps", str(cfg.cosmos_policy_num_steps)])
            cmd.extend(["--cosmos-policy-chunk-size", str(cfg.cosmos_policy_chunk_size)])
        if cfg.cosmos_vae_path:
            cmd.extend(["--cosmos-vae-path", cfg.cosmos_vae_path])
        if cfg.cosmos_config_file:
            cmd.extend(["--cosmos-config-file", cfg.cosmos_config_file])
        if cfg.cosmos_offload_diffusion_model:
            cmd.append("--cosmos-offload-diffusion-model")
        if cfg.cosmos_offload_text_encoder:
            cmd.append("--cosmos-offload-text-encoder")
        if cfg.cosmos_offload_tokenizer:
            cmd.append("--cosmos-offload-tokenizer")
        if cfg.allow_dummy_latents:
            cmd.append("--allow-dummy-latents")
        return cmd

    def _ensure_started(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        cfg = self._config
        env = os.environ.copy()
        env["COSMOS_REPO"] = cfg.cosmos_repo
        if cfg.worker_cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = cfg.worker_cuda_visible_devices
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        cosmos_venv = pathlib.Path(cfg.cosmos_python).expanduser().parent.parent
        cuda_lib_dirs = sorted(cosmos_venv.glob("lib/python*/site-packages/nvidia/cu13/lib"))
        cuda_lib_dirs.extend(sorted(cosmos_venv.glob("lib/python*/site-packages/nvidia/cuda_runtime/lib")))
        # Transformer Engine searches CUDA_HOME for libnvrtc before consulting
        # ldconfig. The Cosmos venv ships CUDA 13 libraries under nvidia/cu13,
        # but the host image does not expose that directory as /usr/local/cuda.
        if cuda_lib_dirs:
            env.setdefault("CUDA_HOME", str(cuda_lib_dirs[0].parent))
        _prepend_library_paths(
            env,
            [
                pathlib.Path(cfg.cosmos_repo) / ".ffmpeg6" / "lib",
                *cuda_lib_dirs,
            ],
        )

        log_path = pathlib.Path(cfg.worker_log_path).expanduser()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_file = log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            self._command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
            env=env,
            cwd=cfg.cosmos_repo,
        )

        ready_line = self._process.stdout.readline() if self._process.stdout is not None else ""
        if not ready_line:
            raise RuntimeError(f"Cosmos realtime worker failed to start. See {log_path}")
        ready = json.loads(ready_line)
        if not ready.get("ok") or not ready.get("ready"):
            raise RuntimeError(f"Cosmos realtime worker did not report ready: {ready}. See {log_path}")

    def infer(
        self,
        *,
        images: dict[str, np.ndarray],
        instruction: str,
        views: list[str],
        state: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        with self._lock:
            self._ensure_started()
            assert self._process is not None
            assert self._process.stdin is not None
            assert self._process.stdout is not None

            request_id = self._next_request_id
            self._next_request_id += 1
            request = {
                "request_id": request_id,
                "instruction": instruction,
                "views": views,
                "images": {view: _encode_array(images[view]) for view in views},
            }
            if state is not None:
                request["state"] = _encode_array(np.asarray(state))
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()

            response_line = self._process.stdout.readline()
            if not response_line:
                raise RuntimeError(f"Cosmos realtime worker exited unexpectedly with code {self._process.poll()}")
            response = json.loads(response_line)
            if response.get("request_id") != request_id:
                raise RuntimeError(f"Cosmos realtime worker returned mismatched response: {response}")
            if not response.get("ok"):
                raise RuntimeError(
                    "Cosmos realtime worker failed: "
                    f"{response.get('error')}\n{response.get('traceback', '')}"
                )
            result = {
                "latent": _decode_array(response["latent"]).astype(np.float32),
                "mask": _decode_array(response["mask"]).astype(np.bool_),
                "time_ids": _decode_array(response["time_ids"]).astype(np.int32),
                "view_ids": _decode_array(response["view_ids"]).astype(np.int32),
            }
            if response.get("action_prior") is not None:
                result["action_prior"] = _decode_array(response["action_prior"]).astype(np.float32)
            return result

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(json.dumps({"shutdown": True}) + "\n")
                    process.stdin.flush()
            except Exception:
                pass
            with contextlib.suppress(Exception):
                process.terminate()
        self._process = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None


@dataclasses.dataclass
class AttachRealtimeCosmosLatent(transforms.DataTransformFn):
    """Attach real-time frozen WAM/Cosmos latent tokens to an online policy input."""

    config: CosmosRealtimeConfig
    image_key: str = "observation/image"
    wrist_image_key: str = "observation/wrist_image"
    # LIBERO's OpenPI image is rotated 180 degrees for the VLA input, while
    # Cosmos Policy expects the vertically flipped simulator image. Callers
    # that have both versions can provide these optional keys.
    cosmos_image_key: str = "observation/cosmos_image"
    cosmos_wrist_image_key: str = "observation/cosmos_wrist_image"
    prompt_key: str = "prompt"
    views: tuple[str, str] = ("image", "wrist_image")

    def __post_init__(self):
        self._client = CosmosRealtimeClient(self.config)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        images = {
            self.views[0]: np.asarray(data.get(self.cosmos_image_key, data[self.image_key])),
            self.views[1]: np.asarray(data.get(self.cosmos_wrist_image_key, data[self.wrist_image_key])),
        }
        instruction = data.get(self.prompt_key, "")
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8")
        elif not isinstance(instruction, str):
            instruction = str(np.asarray(instruction).item())

        try:
            result = self._client.infer(
                images=images,
                instruction=instruction,
                views=list(self.views),
                state=(data.get("observation/state") if self.config.cosmos_policy_use_proprio else None),
            )
        except Exception:
            if not self.config.allow_wam_fallback:
                raise
            _logger.exception("Realtime Cosmos/WAM failed; using zero-condition fallback for this policy query.")
            return self._fallback(data)
        latent = result["latent"]
        expected_shape = (len(self.views), self.config.latent_dim)
        if latent.shape != expected_shape:
            raise ValueError(f"Realtime Cosmos latent has shape {latent.shape}, expected {expected_shape}")
        output = {
            **data,
            "cosmos_latent": latent,
            "cosmos_latent_mask": result["mask"],
            "cosmos_time_ids": result["time_ids"],
            "cosmos_view_ids": result["view_ids"],
        }
        if "action_prior" in result:
            output.update(
                wam_action_prior=result["action_prior"],
                wam_action_prior_mask=np.ones((result["action_prior"].shape[0],), dtype=np.bool_),
                wam_action_prior_valid=np.asarray(1, dtype=np.bool_),
            )
        elif self.config.allow_wam_fallback:
            output.update(**self._zero_action_prior())
        return output

    def _zero_action_prior(self) -> dict[str, np.ndarray]:
        """Return an explicitly invalid prior so strict model code can fall back safely."""
        return {
            "wam_action_prior": np.zeros(
                (self.config.cosmos_policy_chunk_size, self.config.action_prior_dim), dtype=np.float32
            ),
            "wam_action_prior_mask": np.zeros((self.config.cosmos_policy_chunk_size,), dtype=np.bool_),
            "wam_action_prior_valid": np.zeros((), dtype=np.bool_),
        }

    def _fallback(self, data: dict[str, Any]) -> dict[str, Any]:
        """Disable both online WAM conditions while preserving model input shapes."""
        view_count = len(self.views)
        return {
            **data,
            "cosmos_latent": np.zeros((view_count, self.config.latent_dim), dtype=np.float32),
            "cosmos_latent_mask": np.zeros((view_count,), dtype=np.bool_),
            "cosmos_time_ids": np.full((view_count,), self.config.target_step, dtype=np.int32),
            "cosmos_view_ids": np.arange(view_count, dtype=np.int32),
            **self._zero_action_prior(),
        }
