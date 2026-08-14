#!/usr/bin/env python3
"""Build an offline frozen Cosmos/WAM latent cache for a local LeRobot dataset.

The cache schema is consumed by ``openpi.transforms.AttachCosmosLatent``.  The
default path follows the Cosmos Policy / WorldPilot setup for action chunks:
K=16, one future state target per sample, and one fixed future-image latent slot
per camera view.

This script does not sample several natural-video future timesteps.  In the VAE
debug path it uses latent-frame injection: ``o_t`` is repeated four times and
``o_{t+16}`` is repeated four times, so temporal compression maps them to two
independent latent frames.  The cache stores only the fixed future slot.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import dataclasses
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_COSMOS_REPO = os.environ.get("COSMOS_REPO", str(_WORKSPACE_ROOT / "cosmos-predict2.5"))
_DEFAULT_COSMOS_VAE_PATH = os.environ.get(
    "COSMOS_WAN2PT1_VAE_PATH",
    str(_WORKSPACE_ROOT / "cosmos_checkpoints/Cosmos-Predict2.5-2B/tokenizer.pth"),
)
_DEFAULT_COSMOS_RESOLUTION = (192, 320)
_DEFAULT_TARGET_STEP = 16
_DEFAULT_LATENT_FRAME_REPEAT = 4
_DEFAULT_FUTURE_SLOT_INDEX = 1
_WAN2PT1_LATENT_CHANNELS = 16
_WAN2PT1_SPATIAL_COMPRESSION = 8
_DEFAULT_WAN2PT1_FLATTENED_DIM = (
    _WAN2PT1_LATENT_CHANNELS
    * (_DEFAULT_COSMOS_RESOLUTION[0] // _WAN2PT1_SPATIAL_COMPRESSION)
    * (_DEFAULT_COSMOS_RESOLUTION[1] // _WAN2PT1_SPATIAL_COMPRESSION)
)


def _parse_csv_ints(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return result


def _parse_csv_strings(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("expected at least one comma-separated view name")
    return result


def _parse_resolution(value: str) -> tuple[int, int]:
    parts = _parse_csv_ints(value)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected resolution as H,W, e.g. 192,320")
    return parts[0], parts[1]


def _to_int(value: Any) -> int:
    value = np.asarray(value)
    if value.size != 1:
        raise ValueError(f"expected scalar value, got shape {value.shape}")
    return int(value.reshape(()).item())


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _get_sample_value(sample: dict[str, Any], key: str) -> Any:
    if key in sample:
        return sample[key]

    current: Any = sample
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Key {key!r} not found in sample")
        current = current[part]
    return current


def _first_existing_path(paths: list[str | None]) -> str | None:
    for path in paths:
        if not path:
            continue
        expanded = Path(path).expanduser()
        if expanded.exists():
            return str(expanded)
    return None


def _prepare_cosmos_imports(cosmos_repo: str | None) -> None:
    repo = Path(cosmos_repo or os.environ.get("COSMOS_REPO", _DEFAULT_COSMOS_REPO)).expanduser()
    if repo.exists():
        repo_str = str(repo.resolve())
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        cuda_home = os.environ.get("CUDA_HOME")
        bundled_cuda = repo / ".venv" / "lib" / "python3.13" / "site-packages" / "nvidia" / "cu13"
        if cuda_home is None and bundled_cuda.exists():
            os.environ["CUDA_HOME"] = str(bundled_cuda)

        cuda_lib = Path(os.environ["CUDA_HOME"]) / "lib" if os.environ.get("CUDA_HOME") else None
        if cuda_lib is not None and cuda_lib.exists():
            ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
            paths = ld_library_path.split(":") if ld_library_path else []
            if str(cuda_lib) not in paths:
                os.environ["LD_LIBRARY_PATH"] = f"{cuda_lib}{':' + ld_library_path if ld_library_path else ''}"


def _resolve_cosmos_vae_path(explicit_path: str | None) -> str:
    path = _first_existing_path(
        [
            explicit_path,
            os.environ.get("COSMOS_WAN2PT1_VAE_PATH"),
            os.environ.get("COSMOS_TOKENIZER_PATH"),
            (
                str(Path(os.environ["COSMOS_CHECKPOINTS_DIR"]) / "Cosmos-Predict2.5-2B" / "tokenizer.pth")
                if os.environ.get("COSMOS_CHECKPOINTS_DIR")
                else None
            ),
            _DEFAULT_COSMOS_VAE_PATH,
        ]
    )
    if path is None:
        raise FileNotFoundError(
            "Could not find Cosmos Wan2.1 VAE/tokenizer checkpoint. Pass --cosmos-vae-path, "
            "or set COSMOS_WAN2PT1_VAE_PATH/COSMOS_CHECKPOINTS_DIR."
        )
    return path


def _validate_target_args(args: argparse.Namespace) -> None:
    if args.target_step <= 0:
        raise ValueError(f"--target-step must be positive, got {args.target_step}.")
    if args.future_slot_index < 0:
        raise ValueError(f"--future-slot-index must be non-negative, got {args.future_slot_index}.")
    if args.current_repeat_frames <= 0 or args.future_repeat_frames <= 0:
        raise ValueError(
            "--current-repeat-frames and --future-repeat-frames must be positive; "
            f"got {args.current_repeat_frames} and {args.future_repeat_frames}."
        )


def _cache_file_matches(path: Path, *, expected_tokens: int, latent_dim: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            required_keys = ("latent", "mask", "time_ids", "view_ids")
            if any(key not in cache for key in required_keys):
                return False
            latent = np.asarray(cache["latent"])
            if latent.shape != (expected_tokens, latent_dim):
                return False
            expected_id_shape = (expected_tokens,)
            return all(np.asarray(cache[key]).shape == expected_id_shape for key in ("mask", "time_ids", "view_ids"))
    except Exception:
        return False


def _write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    *,
    dataset_dir: Path,
    repo_id: str,
    status: str,
    generated: int,
    skipped: int,
    initial_max_samples: int | None,
) -> None:
    manifest = {
        "schema_version": 1,
        "status": status,
        "source": args.cosmos_latent_source,
        "worldpilot_aligned": args.cosmos_latent_source == "predict_video2world",
        "dataset_dir": str(dataset_dir),
        "repo_id": repo_id,
        "target_step": args.target_step,
        "views": list(args.views),
        "latent_dim": args.latent_dim,
        "num_tokens": len(args.views),
        "cosmos_model": args.cosmos_model,
        "cosmos_checkpoint": args.cosmos_checkpoint,
        "cosmos_resolution": list(args.cosmos_resolution),
        "cosmos_guidance": args.cosmos_guidance,
        "cosmos_num_steps": args.cosmos_num_steps,
        "cosmos_seed": args.cosmos_seed,
        "future_slot_index": args.future_slot_index,
        "current_repeat_frames": args.current_repeat_frames,
        "future_repeat_frames": args.future_repeat_frames,
        "generated_files": generated,
        "skipped_existing_files": skipped,
        "max_samples": initial_max_samples,
        "notes": (
            "Each cache entry stores only the fixed future-image latent slot for target_step. "
            "The future_vae source is an oracle/debug mode; predict_video2world is the deployment-style source."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


@dataclasses.dataclass(frozen=True)
class CosmosLatentExtractorConfig:
    source: str
    latent_dim: int
    cosmos_repo: str | None
    cosmos_vae_path: str | None
    cosmos_checkpoint: str | None
    cosmos_model: str
    cosmos_config_file: str | None
    resolution: tuple[int, int]
    guidance: float
    num_steps: int
    seed: int
    future_slot_index: int
    current_repeat_frames: int
    future_repeat_frames: int
    num_input_frames: int
    num_output_frames: int
    offload_diffusion_model: bool
    offload_text_encoder: bool
    offload_tokenizer: bool


class FrozenCosmosLatentExtractor:
    """Frozen Cosmos/WAM feature extractor used by the offline cache builder."""

    def __init__(self, config: CosmosLatentExtractorConfig):
        self.config = config
        _prepare_cosmos_imports(config.cosmos_repo)

        try:
            import torch
            import torch.nn.functional as torch_f
            import torchvision.transforms.functional as tv_f
        except Exception as exc:  # pragma: no cover - depends on the runtime env
            raise RuntimeError("PyTorch/torchvision are required for real Cosmos latent extraction.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("Cosmos Wan2.1 VAE extraction requires a CUDA GPU in the current environment.")

        self.torch = torch
        self.torch_f = torch_f
        self.tv_f = tv_f
        self.device = torch.device("cuda")
        self.vae_path = _resolve_cosmos_vae_path(config.cosmos_vae_path)
        self.vae = self._load_vae()
        self.predictor = self._load_predictor() if config.source == "predict_video2world" else None

    def _load_vae(self):
        try:
            from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import Wan2pt1VAEInterface
        except Exception as exc:  # pragma: no cover - depends on the Cosmos env
            raise RuntimeError(
                "Failed to import Cosmos Wan2.1 tokenizer. Run with the Cosmos Predict environment, "
                "or set COSMOS_REPO/CUDA_HOME/LD_LIBRARY_PATH so transformer_engine can find libnvrtc."
            ) from exc

        logging.info("Loading frozen Cosmos Wan2.1 VAE from %s", self.vae_path)
        return Wan2pt1VAEInterface(
            vae_pth=self.vae_path,
            s3_credential_path="",
            load_mean_std=False,
        )

    def _load_predictor(self):
        if not self.config.cosmos_checkpoint:
            raise ValueError("--cosmos-checkpoint is required when --cosmos-latent-source=predict_video2world")

        try:
            from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
            from cosmos_predict2.config import MODEL_CHECKPOINTS
            from cosmos_predict2.config import MODEL_KEYS
        except Exception as exc:  # pragma: no cover - depends on the Cosmos env
            raise RuntimeError("Failed to import Cosmos Predict video2world inference code.") from exc

        model_key = MODEL_KEYS[self.config.cosmos_model]
        checkpoint = MODEL_CHECKPOINTS[model_key]
        config_file = self.config.cosmos_config_file or "cosmos_predict2/_src/predict2/configs/video2world/config.py"
        logging.info(
            "Loading frozen Cosmos Predict model=%s experiment=%s checkpoint=%s",
            self.config.cosmos_model,
            checkpoint.experiment,
            self.config.cosmos_checkpoint,
        )
        return Video2WorldInference(
            experiment_name=checkpoint.experiment,
            ckpt_path=self.config.cosmos_checkpoint,
            s3_credential_path="",
            config_file=config_file,
            offload_diffusion_model=self.config.offload_diffusion_model,
            offload_text_encoder=self.config.offload_text_encoder,
            offload_tokenizer=self.config.offload_tokenizer,
        )

    def _image_to_tensor(self, image: np.ndarray):
        torch = self.torch
        image = np.asarray(image)
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"expected image with shape [H,W,C] or [C,H,W], got {image.shape}")
        if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.moveaxis(image, 0, -1)
        if image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)
        if image.shape[-1] != 3:
            raise ValueError(f"expected RGB image with 3 channels, got {image.shape}")

        tensor = torch.as_tensor(image, device=self.device)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255.0
        else:
            tensor = tensor.float()
            if float(tensor.max().detach().cpu()) > 2.0:
                tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0).permute(2, 0, 1).contiguous()

        target_h, target_w = self.config.resolution
        _, orig_h, orig_w = tensor.shape
        scale = max(target_w / orig_w, target_h / orig_h)
        resized_h = int(np.ceil(scale * orig_h))
        resized_w = int(np.ceil(scale * orig_w))
        tensor = self.tv_f.resize(tensor, [resized_h, resized_w], antialias=True)
        tensor = self.tv_f.center_crop(tensor, [target_h, target_w])
        return tensor * 2.0 - 1.0

    def _encode_image_tokens(self, images: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        if not images:
            return np.zeros((0, self.config.latent_dim), dtype=np.float32)

        video = torch.stack([self._image_to_tensor(image) for image in images], dim=0).unsqueeze(2)
        with torch.inference_mode():
            latents = self.vae.encode(video).float()
            tokens = self._latent_to_tokens(latents)
        return tokens.detach().cpu().numpy().astype(np.float32)

    def _encode_injected_future_slot_tokens(
        self,
        *,
        current_images: dict[str, np.ndarray],
        future_images: dict[str, np.ndarray],
        views: list[str],
    ) -> np.ndarray:
        torch = self.torch
        tokens_by_view = []
        for view in views:
            current = self._image_to_tensor(current_images[view])
            future = self._image_to_tensor(future_images[view])
            frames = [current] * self.config.current_repeat_frames + [future] * self.config.future_repeat_frames
            video = torch.stack(frames, dim=1).unsqueeze(0)
            with torch.inference_mode():
                latents = self.vae.encode(video).float()
                if self.config.future_slot_index >= latents.shape[2]:
                    raise ValueError(
                        f"future_slot_index={self.config.future_slot_index} is out of range for injected latent "
                        f"shape {tuple(latents.shape)}. With four current and four future frames, the expected "
                        "future slot is 1."
                    )
                token = self._latent_to_tokens(
                    latents[:, :, self.config.future_slot_index : self.config.future_slot_index + 1]
                )
            tokens_by_view.append(token.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(tokens_by_view, axis=0)

    def _latent_to_tokens(self, latents):
        features = latents.flatten(start_dim=1).float()
        logging.debug("Cosmos raw latent shape=%s flattened_dim=%d", tuple(latents.shape), features.shape[-1])
        if features.shape[-1] == self.config.latent_dim:
            return features
        raise ValueError(
            f"Raw flattened Cosmos latent dim is {features.shape[-1]}, but --latent-dim is "
            f"{self.config.latent_dim}. For true flattened Wan2.1 VAE tokens, set --latent-dim to "
            "16 * latent_T * (H/8) * (W/8). With the default 192,320 resolution and one latent frame, "
            f"that is {_DEFAULT_WAN2PT1_FLATTENED_DIM}."
        )

    def _predict_future_tokens(
        self,
        *,
        current_images: dict[str, np.ndarray],
        instruction: str,
        views: list[str],
    ) -> np.ndarray:
        if self.predictor is None:
            raise RuntimeError("predict_video2world source was requested but the predictor was not loaded.")

        tokens_by_view = []
        for view in views:
            current = self._image_to_tensor(current_images[view])
            model_required_frames = self.predictor.model.tokenizer.get_pixel_num_frames(
                self.predictor.model.config.state_t
            )
            video_input = self.torch.zeros(
                1,
                3,
                model_required_frames,
                self.config.resolution[0],
                self.config.resolution[1],
                dtype=self.torch.uint8,
                device=self.device,
            )
            current_uint8 = ((current + 1.0) * 127.5).clamp(0, 255).to(self.torch.uint8)
            num_current_frames = min(self.config.current_repeat_frames, model_required_frames)
            video_input[:, :, :num_current_frames] = current_uint8[:, None]
            with self.torch.inference_mode():
                data_batch = self.predictor._get_data_batch_input(  # noqa: SLF001 - Cosmos Predict has no public batch helper.
                    video=video_input,
                    prompt=instruction,
                    num_conditional_frames=self.config.num_input_frames,
                )

                if self.predictor.offload_text_encoder and self.predictor.model.text_encoder is not None:
                    if (
                        hasattr(self.predictor.model.text_encoder, "model")
                        and self.predictor.model.text_encoder.model is not None
                    ):
                        self.predictor.model.text_encoder.model = self.predictor.model.text_encoder.model.to("cpu")
                    self.torch.cuda.empty_cache()

                if self.predictor.offload_tokenizer:
                    if (
                        hasattr(self.predictor.model.tokenizer, "encoder")
                        and self.predictor.model.tokenizer.encoder is not None
                    ):
                        self.predictor.model.tokenizer.encoder = self.predictor.model.tokenizer.encoder.to("cuda")
                    self.torch.cuda.empty_cache()

                if self.predictor.offload_diffusion_model:
                    self.predictor.model.net = self.predictor.model.net.to("cuda")
                    if hasattr(self.predictor.model, "conditioner") and self.predictor.model.conditioner is not None:
                        self.predictor.model.conditioner = self.predictor.model.conditioner.to("cuda")
                    self.torch.cuda.empty_cache()

                generate_samples = (
                    self.predictor.model.generate_samples_from_batch_lora
                    if getattr(self.predictor.model.config, "use_lora", False)
                    else self.predictor.model.generate_samples_from_batch
                )
                latent = generate_samples(
                    data_batch,
                    n_sample=1,
                    guidance=self.config.guidance,
                    seed=self.config.seed,
                    is_negative_prompt=True,
                    num_steps=self.config.num_steps,
                )

                if self.predictor.offload_diffusion_model:
                    self.predictor.model.net = self.predictor.model.net.to("cpu")
                    if hasattr(self.predictor.model, "conditioner") and self.predictor.model.conditioner is not None:
                        self.predictor.model.conditioner = self.predictor.model.conditioner.to("cpu")

                if self.predictor.offload_tokenizer:
                    if (
                        hasattr(self.predictor.model.tokenizer, "encoder")
                        and self.predictor.model.tokenizer.encoder is not None
                    ):
                        self.predictor.model.tokenizer.encoder = self.predictor.model.tokenizer.encoder.to("cpu")
                    self.torch.cuda.empty_cache()

                if self.predictor.offload_text_encoder and self.predictor.model.text_encoder is not None:
                    if (
                        hasattr(self.predictor.model.text_encoder, "model")
                        and self.predictor.model.text_encoder.model is not None
                    ):
                        self.predictor.model.text_encoder.model = self.predictor.model.text_encoder.model.to("cuda")
                    self.torch.cuda.empty_cache()

            if isinstance(latent, list):
                latent = self.torch.cat(latent, dim=3)

            if self.config.future_slot_index >= latent.shape[2]:
                raise ValueError(
                    f"future_slot_index={self.config.future_slot_index} is out of range for predicted latent "
                    f"shape {tuple(latent.shape)}."
                )
            token = self._latent_to_tokens(
                latent[:, :, self.config.future_slot_index : self.config.future_slot_index + 1]
            )
            tokens_by_view.append(token.detach().cpu().numpy().astype(np.float32))

        return np.concatenate(tokens_by_view, axis=0)

    def __call__(
        self,
        *,
        current_images: dict[str, np.ndarray],
        future_images: dict[str, np.ndarray],
        instruction: str,
        views: list[str],
    ) -> np.ndarray:
        if self.config.source == "future_vae":
            return self._encode_injected_future_slot_tokens(
                current_images=current_images,
                future_images=future_images,
                views=views,
            )
        if self.config.source == "current_vae":
            images = [current_images[view] for view in views]
            return self._encode_image_tokens(images)
        if self.config.source == "predict_video2world":
            return self._predict_future_tokens(
                current_images=current_images,
                instruction=instruction,
                views=views,
            )
        raise ValueError(f"Unsupported Cosmos latent source: {self.config.source}")


def run_cosmos_encoder_or_predictor(
    *,
    current_images: dict[str, np.ndarray],
    future_images: dict[str, np.ndarray],
    instruction: str,
    views: list[str],
    extractor: FrozenCosmosLatentExtractor | None,
    latent_dim: int,
    allow_dummy_latents: bool,
) -> np.ndarray:
    """Return one fixed future-slot Cosmos token per view with shape ``[N, C]``."""
    num_tokens = len(views)
    if allow_dummy_latents:
        return np.zeros((num_tokens, latent_dim), dtype=np.float32)
    if extractor is None:
        raise RuntimeError("A FrozenCosmosLatentExtractor is required unless --allow-dummy-latents is set.")
    latent = extractor(
        current_images=current_images,
        future_images=future_images,
        instruction=instruction,
        views=views,
    )
    if latent.shape != (num_tokens, latent_dim):
        raise ValueError(
            f"Cosmos extractor returned {latent.shape}, expected {(num_tokens, latent_dim)}. "
            "If you changed --latent-dim, keep the Pi05CosmosConfig cosmos_latent_dim in sync."
        )
    return latent


def _instruction_for_sample(sample: dict[str, Any], tasks: dict[int, str]) -> str:
    for key in ("task", "prompt", "instruction"):
        if key in sample:
            value = sample[key]
            if isinstance(value, bytes):
                return value.decode("utf-8")
            if not isinstance(value, str):
                value = np.asarray(value).item()
            return str(value)
    if "task_index" in sample:
        return tasks.get(_to_int(sample["task_index"]), "")
    return ""


class _SequentialVideoReader:
    def __init__(self, path: Path, *, max_cache_frames: int = 256):
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("The local dataset reader requires PyAV (`av`) to decode mp4 videos.") from exc

        self._av = av
        self._path = path
        self._max_cache_frames = max_cache_frames
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._open()

    def _open(self) -> None:
        self._container = self._av.open(str(self._path))
        self._stream = self._container.streams.video[0]
        self._frames = self._container.decode(self._stream)
        self._next_index = 0

    def _restart(self) -> None:
        self._container.close()
        self._cache.clear()
        self._open()

    def get(self, frame_index: int) -> np.ndarray:
        frame_index = int(frame_index)
        if frame_index in self._cache:
            self._cache.move_to_end(frame_index)
            return self._cache[frame_index]

        # The cache builder walks forward through each episode and only looks a few frames ahead.
        # If a caller jumps backward beyond the small cache, restart decoding instead of doing
        # approximate H264 seeking.
        if frame_index < self._next_index:
            self._restart()

        for frame in self._frames:
            image = frame.to_ndarray(format="rgb24")
            self._cache[self._next_index] = image
            self._cache.move_to_end(self._next_index)
            while len(self._cache) > self._max_cache_frames:
                self._cache.popitem(last=False)
            if self._next_index == frame_index:
                self._next_index += 1
                return image
            self._next_index += 1

        raise IndexError(f"Frame {frame_index} is out of range for video {self._path}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class _LocalLeRobotDatasetReader:
    """Minimal local LeRobot v3 reader for parquet metadata plus mp4 videos.

    This fallback is intentionally small and exists so the Cosmos venv can build
    caches without installing LeRobot/OpenPI into it.
    """

    def __init__(self, dataset_dir: Path, views: list[str]):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("The local dataset reader requires pyarrow to read LeRobot parquet files.") from exc

        self.root = dataset_dir
        tables = [pq.read_table(path) for path in sorted((dataset_dir / "data").glob("**/*.parquet"))]
        if not tables:
            raise FileNotFoundError(f"No parquet files found under {dataset_dir / 'data'}")
        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        self.episode_indices = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        self.frame_indices = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
        self.global_indices = (
            np.asarray(table["index"].to_pylist(), dtype=np.int64)
            if "index" in table.column_names
            else np.arange(table.num_rows, dtype=np.int64)
        )
        self.task_indices = (
            np.asarray(table["task_index"].to_pylist(), dtype=np.int64)
            if "task_index" in table.column_names
            else np.zeros(table.num_rows, dtype=np.int64)
        )
        self.tasks = {
            int(row["task_index"]): str(row["task"])
            for row in _load_jsonl(dataset_dir / "meta" / "tasks.jsonl")
        }
        self._videos = {view: _SequentialVideoReader(self._resolve_single_video(view)) for view in views}

    def _resolve_single_video(self, view: str) -> Path:
        files = sorted((self.root / "videos" / view).glob("**/*.mp4"))
        if len(files) != 1:
            raise NotImplementedError(
                f"The local fallback reader currently expects exactly one mp4 for view {view!r}; found {len(files)}. "
                "Use --dataset-reader lerobot for multi-file LeRobot datasets."
            )
        return files[0]

    def get_sample(self, index: int, views: list[str]) -> dict[str, Any]:
        video_frame_index = int(self.global_indices[index])
        task_index = int(self.task_indices[index])
        sample = {
            "episode_index": int(self.episode_indices[index]),
            "frame_index": int(self.frame_indices[index]),
            "task_index": task_index,
            "task": self.tasks.get(task_index, ""),
        }
        for view in views:
            sample[view] = self._videos[view].get(video_frame_index)
        return sample


class _LeRobotDatasetReader:
    def __init__(self, repo_id: str, dataset_dir: Path):
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        self.metadata = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_dir)
        self.dataset = lerobot_dataset.LeRobotDataset(repo_id, root=dataset_dir)
        self.tasks = self.metadata.tasks
        self.episode_indices = np.asarray([_to_int(value) for value in self.dataset.hf_dataset["episode_index"]])
        self.frame_indices = np.asarray([_to_int(value) for value in self.dataset.hf_dataset["frame_index"]])

    def get_sample(self, index: int, views: list[str]) -> dict[str, Any]:
        del views
        return self.dataset[int(index)]


def _make_dataset_reader(args: argparse.Namespace, dataset_dir: Path, repo_id: str):
    if args.dataset_reader in ("auto", "lerobot"):
        try:
            reader = _LeRobotDatasetReader(repo_id, dataset_dir)
            logging.info("Using LeRobot dataset reader.")
            return reader
        except ImportError:
            if args.dataset_reader == "lerobot":
                raise
            logging.info("LeRobot is not installed in this environment; using local parquet/mp4 reader.")

    reader = _LocalLeRobotDatasetReader(dataset_dir, args.views)
    logging.info("Using local parquet/mp4 dataset reader.")
    return reader


def build_cache(args: argparse.Namespace) -> None:
    _validate_target_args(args)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    repo_id = args.repo_id or dataset_dir.name
    expected_tokens = len(args.views)
    initial_max_samples = args.max_samples
    remaining_samples = args.max_samples
    generated = 0
    skipped = 0

    dataset = _make_dataset_reader(args, dataset_dir, repo_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(
        output_dir,
        args,
        dataset_dir=dataset_dir,
        repo_id=repo_id,
        status="in_progress",
        generated=generated,
        skipped=skipped,
        initial_max_samples=initial_max_samples,
    )
    extractor = None
    if not args.allow_dummy_latents:
        extractor = FrozenCosmosLatentExtractor(
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

    episode_indices = dataset.episode_indices
    frame_indices = dataset.frame_indices
    needs_future_images = args.cosmos_latent_source == "future_vae" and not args.allow_dummy_latents
    stop_after_limit = False

    for episode_index in sorted(np.unique(episode_indices).tolist()):
        if stop_after_limit:
            break
        dataset_indices = np.where(episode_indices == episode_index)[0]
        order = np.argsort(frame_indices[dataset_indices])
        dataset_indices = dataset_indices[order]
        local_frames = frame_indices[dataset_indices]
        last_pos = len(dataset_indices) - 1

        for pos, dataset_index in enumerate(dataset_indices):
            if remaining_samples is not None and remaining_samples <= 0:
                stop_after_limit = True
                break
            frame_index = int(local_frames[pos])
            episode_dir = output_dir / f"episode_{episode_index:06d}"
            save_path = episode_dir / f"frame_{frame_index:06d}.npz"
            if not args.overwrite_existing and _cache_file_matches(
                save_path,
                expected_tokens=expected_tokens,
                latent_dim=args.latent_dim,
            ):
                logging.info(
                    "episode=%06d frame=%06d latent_shape=(%d, %d) skipped_existing=%s",
                    episode_index,
                    frame_index,
                    expected_tokens,
                    args.latent_dim,
                    save_path,
                )
                skipped += 1
                if remaining_samples is not None:
                    remaining_samples -= 1
                continue

            sample = dataset.get_sample(int(dataset_index), args.views)
            instruction = _instruction_for_sample(sample, dataset.tasks)
            current_images = {view: _to_numpy(_get_sample_value(sample, view)) for view in args.views}

            target_pos = pos + args.target_step
            in_bounds = target_pos <= last_pos
            target_pos = min(target_pos, last_pos)
            if needs_future_images:
                target_sample = dataset.get_sample(int(dataset_indices[target_pos]), args.views)
                future_images = {view: _to_numpy(_get_sample_value(target_sample, view)) for view in args.views}
            else:
                future_images = {}
            mask = [in_bounds] * len(args.views)
            time_ids = [args.target_step] * len(args.views)
            view_ids = list(range(len(args.views)))

            latent = run_cosmos_encoder_or_predictor(
                current_images=current_images,
                future_images=future_images,
                instruction=instruction,
                views=args.views,
                extractor=extractor,
                latent_dim=args.latent_dim,
                allow_dummy_latents=args.allow_dummy_latents,
            )
            latent = np.asarray(latent, dtype=np.float32)
            if latent.shape != (expected_tokens, args.latent_dim):
                raise ValueError(
                    f"run_cosmos_encoder_or_predictor returned {latent.shape}, "
                    f"expected {(expected_tokens, args.latent_dim)}"
                )

            episode_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save_path,
                latent=latent,
                mask=np.asarray(mask, dtype=np.bool_),
                time_ids=np.asarray(time_ids, dtype=np.int32),
                view_ids=np.asarray(view_ids, dtype=np.int32),
            )
            logging.info(
                "episode=%06d frame=%06d latent_shape=%s saved=%s",
                episode_index,
                frame_index,
                latent.shape,
                save_path,
            )
            generated += 1
            if remaining_samples is not None:
                remaining_samples -= 1

    status = "partial" if stop_after_limit else "complete"
    _write_manifest(
        output_dir,
        args,
        dataset_dir=dataset_dir,
        repo_id=repo_id,
        status=status,
        generated=generated,
        skipped=skipped,
        initial_max_samples=initial_max_samples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--target-step",
        type=int,
        default=_DEFAULT_TARGET_STEP,
        help=(
            "Single action-chunk future target. Defaults to 16, so each sample uses observation[t+16] "
            "as the future state target."
        ),
    )
    parser.add_argument(
        "--views",
        type=_parse_csv_strings,
        required=True,
        help="Comma-separated LeRobot image keys, e.g. observation.images.front,observation.images.wrist",
    )
    parser.add_argument("--cosmos-checkpoint", default=None)
    parser.add_argument(
        "--cosmos-latent-source",
        choices=("future_vae", "current_vae", "predict_video2world"),
        default="predict_video2world",
        help=(
            "predict_video2world runs frozen Cosmos Predict from the current observation plus instruction "
            "and stores the fixed predicted future-image WAM slot. "
            "future_vae encodes real future dataset frames with the frozen Cosmos VAE; "
            "current_vae encodes the current observation for a pipeline smoke-test token; "
            "future_vae/current_vae are debugging modes, not the final WorldPilot-aligned path."
        ),
    )
    parser.add_argument("--cosmos-repo", default=os.environ.get("COSMOS_REPO", _DEFAULT_COSMOS_REPO))
    parser.add_argument("--cosmos-vae-path", default=None)
    parser.add_argument("--cosmos-model", default="2B/post-trained")
    parser.add_argument("--cosmos-config-file", default=None)
    parser.add_argument("--cosmos-resolution", type=_parse_resolution, default=_DEFAULT_COSMOS_RESOLUTION)
    parser.add_argument("--cosmos-guidance", type=float, default=7.0)
    parser.add_argument("--cosmos-num-steps", type=int, default=10)
    parser.add_argument("--cosmos-seed", type=int, default=1)
    parser.add_argument(
        "--future-slot-index",
        type=int,
        default=_DEFAULT_FUTURE_SLOT_INDEX,
        help=(
            "Fixed future-image latent slot to read. For Cosmos Policy latent-frame injection "
            "[o_t x4, o_t+K x4], the future slot is 1."
        ),
    )
    parser.add_argument("--current-repeat-frames", type=int, default=_DEFAULT_LATENT_FRAME_REPEAT)
    parser.add_argument("--future-repeat-frames", type=int, default=_DEFAULT_LATENT_FRAME_REPEAT)
    parser.add_argument("--cosmos-num-input-frames", type=int, default=1)
    parser.add_argument("--cosmos-num-output-frames", type=int, default=77)
    parser.add_argument("--cosmos-offload-diffusion-model", action="store_true")
    parser.add_argument("--cosmos-offload-text-encoder", action="store_true")
    parser.add_argument("--cosmos-offload-tokenizer", action="store_true")
    parser.add_argument("--repo-id", default=None, help="Defaults to the dataset directory name.")
    parser.add_argument(
        "--dataset-reader",
        choices=("auto", "lerobot", "local"),
        default="auto",
        help="auto uses LeRobot if installed, otherwise a local parquet/mp4 reader for this dataset format.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=_DEFAULT_WAN2PT1_FLATTENED_DIM,
        help=(
            "Flattened Cosmos latent dimension. Default is the true Wan2.1 VAE flattened dim "
            f"for 192x320 single-frame latents: {_DEFAULT_WAN2PT1_FLATTENED_DIM}."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for smoke tests.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate existing cache files instead of skipping shape-compatible .npz files.",
    )
    parser.add_argument(
        "--allow-dummy-latents",
        action="store_true",
        help="Write zero latents for dataloader smoke tests. Do not use for real training.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_cache(args)


if __name__ == "__main__":
    main()
