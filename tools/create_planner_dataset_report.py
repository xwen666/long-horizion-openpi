#!/usr/bin/env python3
"""Build the human-readable planner dataset report from generated artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _bullet_map(values: dict, limit: int | None = None) -> str:
    items = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        items = items[:limit]
    return "\n".join(f"- `{key}`: {value}" for key, value in items) or "- None"


def build(stats_path: Path, quality_path: Path, split_path: Path, dataset_path: Path, output_path: Path) -> None:
    stats = _load(stats_path, {})
    quality = _load(quality_path, {})
    split = _load(split_path, {})
    first = None
    if dataset_path.exists():
        with dataset_path.open() as handle:
            first = json.loads(next(handle))

    checks = quality.get("checks", {})
    lines = [
        "# RoboCasa Planner Dataset Report",
        "",
        "## 1. Dataset Overview",
        "",
        f"- Dataset: `{stats.get('dataset_path', dataset_path)}`",
        f"- Planner samples: **{stats.get('planner_samples', 0)}**",
        f"- Episodes: **{stats.get('episodes', 0)}**",
        f"- Global tasks: **{stats.get('unique_global_tasks', 0)}**",
        f"- Average samples per episode: **{stats.get('avg_samples_per_episode', 0):.3f}**",
        f"- Samples per episode range: **{stats.get('min_samples_per_episode', 0)} - {stats.get('max_samples_per_episode', 0)}**",
        "",
        "## 2. Sample Schema",
        "",
        "The source manifest keeps generic RoboCasa camera names and stores images as `video_path + frame_index` references.",
        "The target is one semantic next subtask; no action chunk, joint state, or low-level command is included.",
        "",
        "```text",
        "global_task",
        "images: camera_name -> {video_path, frame_index}",
        "completed_subtask_history: [{subtask_idx, instruction}]",
        "next_subtask_instruction",
        "next_subtask_name",
        "next_subtask_stage",
        "```",
        "",
        "See [planner_dataset_schema.md](planner_dataset_schema.md) for the complete inspected schema.",
        "",
        "## 3. Statistics",
        "",
        f"- Missing front images: **{stats.get('image', {}).get('missing_front', 0)}**",
        f"- Missing wrist images: **{stats.get('image', {}).get('missing_wrist', 0)}**",
        f"- Invalid image paths: **{stats.get('image', {}).get('invalid_path', 0)}**",
        f"- Raw `previous_result` fields: **{stats.get('planner_samples', 0) - stats.get('missing_raw_previous_result', 0)}**",
        f"- Derived previous result fields: **{stats.get('missing_raw_previous_result', 0)}**",
        "",
        "## 4. Task and Skill Distribution",
        "",
        "### Global tasks",
        "",
        _bullet_map(stats.get("task_frequency", {}), limit=32),
        "",
        "### Atomic skills",
        "",
        _bullet_map(stats.get("atomic_skill_distribution", {})),
        "",
        "### Stages",
        "",
        _bullet_map(stats.get("stage_distribution", {})),
        "",
        "## 5. History Length Distribution",
        "",
        _bullet_map({int(key): value for key, value in stats.get("history_length_distribution", {}).items()}),
        "",
        "## 6. Example Sample",
        "",
        "```json",
        json.dumps(first, ensure_ascii=False, indent=2) if first is not None else "null",
        "```",
        "",
        "## 7. Data Quality Issues",
        "",
    ]
    if quality:
        for name, check in checks.items():
            lines.append(f"- `{name}`: **{check.get('status', 'unknown')}** ({check.get('count', check.get('wrong_boundary_frames', check.get('episodes_with_errors', 0)))})")
        lines.append("")
        lines.append(f"- Total issue records: **{len(quality.get('issues', []))}** (capped at 500 in the report JSON).")
        lines.append(f"- Issue counts: `{quality.get('issue_counts', {})}`")
    else:
        lines.append("Quality report not found; run `python tools/check_planner_dataset.py`.")
    lines += [
        "",
        "## 8. Train/Val/Test Split",
        "",
    ]
    if split:
        for split_name in ("train", "val", "test"):
            lines.append(f"- `{split_name}` episodes: **{split.get('episodes', {}).get(split_name, 0)}**, samples: **{split.get('samples', {}).get(split_name, 0)}**")
        lines.append(f"- Episode overlap: `{split.get('episode_overlap', {})}`")
    else:
        lines.append("Split statistics not found; run `python tools/split_planner_dataset.py`.")
    lines += [
        "",
        "## 9. Qwen3-VL Training Format",
        "",
        "The converter writes system/user/assistant messages. The user message contains front and wrist images followed by the planner prompt. The assistant content is a JSON string with:",
        "",
        "```json",
        '{"mode":"execute", "instruction":"...", "atomic_skill":"...", "stage":"..."}',
        "```",
        "",
        "Generate a processor-ready version with actual JPEG frames using:",
        "",
        "```bash",
        "python tools/convert_to_qwen3vl_format.py --image-mode frame",
        "```",
        "",
        "No Qwen3-VL training is started by this pipeline.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Wrote report: {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=Path("outputs/planner_dataset_stats.json"))
    parser.add_argument("--quality", type=Path, default=Path("outputs/data_quality_report.json"))
    parser.add_argument("--split", type=Path, default=Path("robocasa/processed/composite_subtasks/splits/split_statistics.json"))
    parser.add_argument("--dataset", type=Path, default=Path("robocasa/processed/composite_subtasks/annotations.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/planner_dataset_report.md"))
    args = parser.parse_args()
    build(args.stats, args.quality, args.split, args.dataset, args.output)


if __name__ == "__main__":
    main()
