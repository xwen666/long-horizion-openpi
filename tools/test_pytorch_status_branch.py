"""Small CPU/GPU smoke test for the PyTorch π0.5 status branch."""

import importlib.util
from pathlib import Path
import tempfile
import types

import safetensors.torch
import torch

from openpi.models import model as model_lib
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.status_head import DoneScheduler
from openpi.models_pytorch.status_head import StatusEncoder

_TRAIN_MODULE_SPEC = importlib.util.spec_from_file_location("openpi_train_pytorch", "scripts/train_pytorch.py")
_TRAIN_MODULE = importlib.util.module_from_spec(_TRAIN_MODULE_SPEC)
assert _TRAIN_MODULE_SPEC.loader is not None
_TRAIN_MODULE_SPEC.loader.exec_module(_TRAIN_MODULE)
load_model_weights_compatible = _TRAIN_MODULE.load_model_weights_compatible


def make_observation(batch_size: int, device: torch.device) -> model_lib.Observation:
    image = {
        key: torch.randn(batch_size, 3, 32, 32, device=device)
        for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    }
    masks = {key: torch.ones(batch_size, dtype=torch.bool, device=device) for key in image}
    return model_lib.Observation(
        images=image,
        image_masks=masks,
        state=torch.randn(batch_size, 32, device=device),
        tokenized_prompt=torch.randint(0, 128, (batch_size, 32), device=device),
        tokenized_prompt_mask=torch.ones(batch_size, 32, dtype=torch.bool, device=device),
        done_label=torch.tensor([0.0, 1.0], device=device)[:batch_size],
    )


def main() -> None:
    # Some hosts expose a CUDA device while the installed wheel lacks its
    # architecture. Set OPENPI_STATUS_TEST_DEVICE=cpu to force a CPU smoke test.
    requested_device = __import__("os").environ.get("OPENPI_STATUS_TEST_DEVICE")
    device = torch.device(requested_device or ("cuda" if torch.cuda.is_available() else "cpu"))
    config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=4,
        max_token_len=32,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        enable_status_head=True,
        pytorch_compile_mode=None,
    )
    model = PI0Pytorch(config).to(device)
    model.gradient_checkpointing_disable()
    status_probe = StatusEncoder(vla_dim=2048)
    assert status_probe(torch.randn(2, 8, 2048), torch.ones(2, 8, dtype=torch.bool)).shape == (2, 512)

    # The repository's dummy Gemma language width is 64 while the production
    # PaliGemma image tower has a fixed 2048-wide projection. Stub only the
    # image/text embedding boundary so the smoke test remains architecture-sized
    # without downloading a production checkpoint.
    def fake_embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        batch_size = lang_tokens.shape[0]
        prefix = torch.randn(batch_size, 8, 64, device=lang_tokens.device)
        valid = torch.ones(batch_size, 8, dtype=torch.bool, device=lang_tokens.device)
        attention = torch.zeros(batch_size, 8, dtype=torch.bool, device=lang_tokens.device)
        return prefix, valid, attention

    model.embed_prefix = types.MethodType(fake_embed_prefix, model)
    model.train()
    observation = make_observation(2, device)
    actions = torch.randn(2, config.action_horizon, config.action_dim, device=device)

    outputs = model(observation, actions, return_details=True)
    assert outputs["z_status"].shape == (2, 512), outputs["z_status"].shape
    assert outputs["done_logit"].shape == (2, 1), outputs["done_logit"].shape
    assert outputs["action_loss"].shape == actions.shape, outputs["action_loss"].shape
    outputs["done_loss"].backward()

    status_grads = [p.grad for n, p in model.named_parameters() if n.startswith(("status_encoder.", "done_head."))]
    base_grads = [p.grad for n, p in model.named_parameters() if not n.startswith(("status_encoder.", "done_head."))]
    assert any(g is not None and torch.any(g != 0) for g in status_grads)
    assert all(g is None for g in base_grads)

    model.eval()
    sampled_actions = model.sample_actions(device, observation, num_steps=2)
    assert sampled_actions.shape == actions.shape
    done_prob = model.predict_done(observation)
    assert done_prob.shape == (2, 1)
    assert torch.all((done_prob >= 0) & (done_prob <= 1))

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "status.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = PI0Pytorch(config).to(device)
        restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    scheduler = DoneScheduler(threshold=0.8, consecutive_steps=3)
    assert not scheduler.update(0.9)
    assert not scheduler.update(0.9)
    assert scheduler.update(0.9)
    scheduler.reset()

    disabled_config = Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=4,
        max_token_len=32,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        enable_status_head=False,
        pytorch_compile_mode=None,
    )
    disabled_model = PI0Pytorch(disabled_config).to(device)
    disabled_model.embed_prefix = types.MethodType(fake_embed_prefix, disabled_model)
    disabled_output = disabled_model(observation, actions)
    assert isinstance(disabled_output, torch.Tensor)
    assert disabled_output.shape == actions.shape

    with tempfile.TemporaryDirectory() as directory:
        legacy_checkpoint = Path(directory) / "legacy.safetensors"
        legacy_model = PI0Pytorch(disabled_config)
        safetensors.torch.save_model(legacy_model, legacy_checkpoint)
        load_model_weights_compatible(restored, legacy_checkpoint)

    print(f"status smoke test passed on {device}: z_status={tuple(outputs['z_status'].shape)}")


if __name__ == "__main__":
    main()
