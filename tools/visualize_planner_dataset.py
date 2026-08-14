#!/usr/bin/env python3
"""Create offline HTML pages for manually inspecting planner samples."""

from __future__ import annotations

import argparse
import base64
import html
import io
import random
from pathlib import Path
from typing import Any

from planner_dataset_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLANNER_DATASET,
    choose_dataset_path,
    history,
    image_references,
    load_records,
    normalized_record,
    resolve_media_path,
    target,
)


def _decode_image(ref: dict[str, Any], data_root: Path) -> tuple[str | None, str | None]:
    path = resolve_media_path(ref, data_root)
    if path is None or not path.exists():
        return None, f"Missing image/video: {path}"
    frame_index = int(ref.get("frame_index") or 0)
    try:
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            from PIL import Image

            image = Image.open(path).convert("RGB")
        else:
            import av

            container = av.open(str(path))
            stream = container.streams.video[0]
            image = None
            for index, frame in enumerate(container.decode(stream)):
                if index == frame_index:
                    image = frame.to_image().convert("RGB")
                    break
            container.close()
            if image is None:
                return None, f"Frame {frame_index} unavailable in {path}"
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}", None
    except Exception as exc:  # keep one broken sample from stopping the batch
        return None, f"Decode failed for {path} frame {frame_index}: {exc}"


def _text(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _sample_html(index: int, record: dict[str, Any], data_root: Path) -> str:
    normalized = normalized_record(record)
    images_html = []
    for camera, ref in image_references(record).items():
        data_url, error = _decode_image(ref, data_root)
        body = f'<img src="{data_url}" alt="{_text(camera)}" />' if data_url else f'<div class="error">{_text(error)}</div>'
        images_html.append(f"<figure><figcaption>{_text(camera)}</figcaption>{body}</figure>")

    history_items = "".join(
        f"<li><b>{_text(item.get('skill'))}</b> "
        f"<span class=stage>{_text(item.get('stage'))}</span>: "
        f"{_text(item.get('instruction'))}</li>"
        for item in history(record)
    ) or "<li><em>No completed subtasks</em></li>"
    target_record = target(record)
    previous = normalized["previous_result"]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Planner sample {index:05d}</title>
<style>
body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 24px auto; padding: 0 24px; color: #202124; }}
h1 {{ font-size: 24px; }} h2 {{ margin-top: 28px; font-size: 18px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
.meta {{ color: #5f6368; }} .images {{ display: flex; flex-wrap: wrap; gap: 16px; }}
figure {{ margin: 0; width: min(31%, 360px); min-width: 260px; }} figcaption {{ font-weight: 600; margin-bottom: 6px; }}
img {{ width: 100%; aspect-ratio: 1; object-fit: contain; background: #f1f3f4; border: 1px solid #ddd; }}
.error {{ padding: 24px 8px; background: #fff4f4; color: #b00020; min-height: 80px; }}
.stage {{ color: #6b4f00; }} pre {{ background: #f6f8fa; padding: 12px; overflow: auto; }}
</style>
</head>
<body>
<h1>Planner sample {index:05d}</h1>
<p class="meta">Sample ID: {_text(record.get('sample_id', normalized['episode_id']))}<br />Episode: {_text(normalized['episode_id'])}</p>
<h2>Global Task</h2><p>{_text(normalized['global_task'])}</p>
<h2>Current Observation</h2><div class="images">{''.join(images_html)}</div>
<h2>Completed History</h2><ol>{history_items}</ol>
<p><b>Previous result:</b> {_text(previous)}</p>
<h2>Ground Truth Next Subtask</h2>
<p><b>Instruction:</b> {_text(target_record.get('instruction'))}</p>
<p><b>Skill:</b> {_text(target_record.get('skill'))}<br /><b>Stage:</b> {_text(target_record.get('stage'))}</p>
<h2>Raw Record</h2><pre>{_text(__import__('json').dumps(record, ensure_ascii=False, indent=2))}</pre>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PLANNER_DATASET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/visualization"))
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    records = load_records(choose_dataset_path(args.dataset))
    if not records:
        raise ValueError("Planner dataset is empty")
    randomizer = random.Random(args.seed)
    selected = randomizer.sample(records, min(args.num_samples, len(records)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(selected):
        path = args.output_dir / f"sample_{index:05d}.html"
        path.write_text(_sample_html(index, record, args.data_root))
    print(f"Wrote {len(selected)} offline HTML samples to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
