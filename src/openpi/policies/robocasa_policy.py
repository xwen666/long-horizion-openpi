"""RoboCasa-specific data transforms for the standard π0.5 input contract."""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    """Convert RoboCasa's three camera views to the standard π0.5 inputs."""

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": _parse_image(data["observation/image"]),
                "left_wrist_0_rgb": _parse_image(data["observation/wrist_image"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_image"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        for key in ("actions", "prompt", "done_label", "episode_index", "frame_index"):
            if key in data:
                inputs[key] = data[key]
        return inputs
