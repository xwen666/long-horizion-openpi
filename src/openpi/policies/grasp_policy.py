import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class GraspInputs(transforms.DataTransformFn):
    """Inputs for the local Piper grasp LeRobot dataset.

    Expected keys after repacking:
    - observation/image: third-person RGB image
    - observation/wrist_image: wrist RGB image
    - observation/state: 7D joint + gripper state
    - actions: [action_horizon, 7] joint + gripper actions during training
    - prompt: language instruction
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        for key in ("episode_index", "frame_index"):
            if key in data:
                inputs[key] = data[key]

        return inputs


@dataclasses.dataclass(frozen=True)
class PrependZeroDims(transforms.DataTransformFn):
    num_dims: int = 2

    def __call__(self, data: dict) -> dict:
        for key in ("state", "actions"):
            if key not in data:
                continue
            value = np.asarray(data[key])
            zeros = np.zeros((*value.shape[:-1], self.num_dims), dtype=value.dtype)
            data[key] = np.concatenate([zeros, value], axis=-1)
        return data


@dataclasses.dataclass(frozen=True)
class GraspOutputs(transforms.DataTransformFn):
    action_dim: int = 7
    drop_first_n_dims: int = 0

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        return {"actions": actions[:, self.drop_first_n_dims : self.action_dim]}
