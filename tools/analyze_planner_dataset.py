#!/usr/bin/env python3
"""Inspect and summarize a high-level planner dataset.

The tool discovers the dataset format from the input suffix and supports JSON,
JSONL, Parquet, and HDF5 through :mod:`planner_dataset_utils`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from planner_dataset_utils import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PLANNER_DATASET,
    choose_dataset_path,
    image_references,
    normalized_record,
    resolve_media_path,
)


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    return type(value).__name__


def _field_schema(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for record in records:
        for key, value in record.items():
            entry = fields.setdefault(key, {"types": Counter(), "present": 0, "missing": 0})
            entry["types"][_type_name(value)] += 1
            entry["present"] += 1
    for entry in fields.values():
        entry["missing"] = len(records) - entry["present"]
        entry["types"] = dict(entry["types"])
    return fields


def _write_schema_doc(
    path: Path,
    dataset_path: Path,
    records: list[dict[str, Any]],
    stats: dict[str, Any],
    first_record: dict[str, Any] | None,
) -> None:
    fields = _field_schema(records)
    normalized_fields = {
        "episode_id": "string",
        "global_task": "string",
        "images": "object: camera name -> {path, frame_index}",
        "history": "array of {instruction, skill, stage, optional subtask_idx}",
        "previous_result": "string (derived success/not_applicable when absent in source)",
        "target": "object: instruction, skill, stage, optional subtask_idx/frame_index",
        "observation_timestep": "integer/float or null",
        "target_timestep": "integer/float or null",
    }
    lines = [
        "# Planner Dataset Schema",
        "",
        "## Dataset Path",
        "",
        f"- Input file: `{dataset_path.resolve()}`",
        f"- Data root for image/video references: `{stats['data_root']}`",
        f"- Format: `{dataset_path.suffix.lower().lstrip('.')}`",
        "",
        "## Dataset Size",
        "",
        f"- Planner samples: **{stats['planner_samples']}**",
        f"- Unique episodes: **{stats['episodes']}**",
        f"- Unique global tasks: **{stats['unique_global_tasks']}**",
        "",
        "## Source Sample Format",
        "",
        "The source records are not rewritten by this inspection step. Their top-level fields are:",
        "",
        "| Field | Types observed | Missing |",
        "| --- | --- | ---: |",
    ]
    for key, entry in fields.items():
        types = ", ".join(f"{name} ({count})" for name, count in entry["types"].items())
        lines.append(f"| `{key}` | {types} | {entry['missing']} |")
    lines += [
        "",
        "## Normalized Planner Fields",
        "",
        "The tools normalize aliases into the following fields:",
        "",
        "| Field | Meaning |",
        "| --- | --- |",
    ]
    for key, meaning in normalized_fields.items():
        lines.append(f"| `{key}` | {meaning} |")
    lines += [
        "",
        "## Image Fields",
        "",
        f"- Missing front image: **{stats['image']['missing_front']}**",
        f"- Missing wrist image: **{stats['image']['missing_wrist']}**",
        f"- Invalid image/video paths: **{stats['image']['invalid_path']}**",
        "- RoboCasa camera mapping: `agentview_left/right` is treated as front and `eye_in_hand` as wrist.",
        "- Image records keep `video_path + frame_index`; no image is copied during manifest inspection.",
        "",
        "## Text, History, and Target Fields",
        "",
        "- `global_task` is the complete composite task instruction.",
        "- `history` contains already completed semantic subtasks.",
        "- `target.instruction` is exactly one next semantic subtask.",
        "- `target.skill` and `target.stage` come from `next_subtask_name` and `next_subtask_stage` when present.",
        "- The current RoboCasa manifest has no raw `previous_result`; the tools derive `success` for samples with completed history and `not_applicable` for the first subtask.",
        "",
        "## Missing and Invalid Samples",
        "",
        f"- Empty global task: **{stats['invalid']['empty_global_task']}**",
        f"- Empty target instruction: **{stats['invalid']['empty_target']}**",
        f"- Repeated target instruction in history (warning): **{stats['warnings']['repeated_target_instruction']}**",
        f"- Observation is after target transition: **{stats['invalid']['image_leakage']}**",
        f"- Missing episode id: **{stats['invalid']['missing_episode_id']}**",
        "",
        "## Complete Sample",
        "",
        "```json",
        json.dumps(first_record, ensure_ascii=False, indent=2) if first_record is not None else "null",
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def analyze(dataset_path: Path, data_root: Path, schema_path: Path, stats_path: Path) -> dict[str, Any]:
    from planner_dataset_utils import episode_id, global_task, history, target

    records = [dict(record) for record in __import__("planner_dataset_utils").load_records(dataset_path)]
    normalized = [normalized_record(record) for record in records]
    episodes = Counter(item["episode_id"] for item in normalized)
    tasks = Counter(item["global_task"] for item in normalized)
    instructions = Counter(str(item["target"].get("instruction", "")) for item in normalized)
    skills = Counter(str(item["target"].get("skill", "")) for item in normalized)
    stages = Counter(str(item["target"].get("stage", "")) for item in normalized)
    history_lengths = Counter(str(len(item["history"])) for item in normalized)

    invalid_counts = Counter()
    invalid_examples: dict[str, list[str]] = defaultdict(list)
    missing_raw_previous_result = 0
    image_stats = Counter()
    for record, item in zip(records, normalized):
        sample_id = str(record.get("sample_id", item["episode_id"]))
        if not item["global_task"].strip():
            invalid_counts["empty_global_task"] += 1
            invalid_examples["empty_global_task"].append(sample_id)
        instruction = str(item["target"].get("instruction", "") or "").strip()
        if not instruction:
            invalid_counts["empty_target"] += 1
            invalid_examples["empty_target"].append(sample_id)
        if any(instruction == str(entry.get("instruction", "")).strip() for entry in item["history"]):
            invalid_counts["repeated_target_instruction"] += 1
            invalid_examples["repeated_target_instruction"].append(sample_id)
        obs_t = item["observation_timestep"]
        target_t = item["target_timestep"]
        if obs_t is not None and target_t is not None and float(obs_t) > float(target_t):
            invalid_counts["image_leakage"] += 1
            invalid_examples["image_leakage"].append(sample_id)
        if item["episode_id"].startswith("<missing_"):
            invalid_counts["missing_episode_id"] += 1
            invalid_examples["missing_episode_id"].append(sample_id)
        if item["previous_result_derived"]:
            missing_raw_previous_result += 1

        role_images = item["role_images"]
        for role in ("front", "wrist"):
            ref = role_images.get(role)
            if ref is None:
                image_stats[f"missing_{role}"] += 1
                continue
            path = resolve_media_path(ref, data_root)
            if path is None or not path.exists():
                image_stats["invalid_path"] += 1

    stats = {
        "dataset_path": str(dataset_path.resolve()),
        "data_root": str(data_root.resolve()),
        "planner_samples": len(records),
        "episodes": len(episodes),
        "avg_samples_per_episode": len(records) / len(episodes) if episodes else 0.0,
        "min_samples_per_episode": min(episodes.values()) if episodes else 0,
        "max_samples_per_episode": max(episodes.values()) if episodes else 0,
        "unique_global_tasks": len(tasks),
        "task_frequency": dict(tasks),
        "subtask_instructions": dict(instructions),
        "atomic_skill_distribution": dict(skills),
        "stage_distribution": dict(stages),
        "history_length_distribution": dict(history_lengths),
        "image": {
            "missing_front": image_stats["missing_front"],
            "missing_wrist": image_stats["missing_wrist"],
            "invalid_path": image_stats["invalid_path"],
        },
        "missing_raw_previous_result": missing_raw_previous_result,
        "invalid": {
            **{key: invalid_counts[key] for key in ("empty_global_task", "empty_target", "image_leakage", "missing_episode_id")},
            "examples": {key: values[:20] for key, values in invalid_examples.items()},
        },
        "warnings": {
            "repeated_target_instruction": invalid_counts["repeated_target_instruction"],
            "repeated_target_examples": invalid_examples.get("repeated_target_instruction", [])[:20],
        },
        "sample_format": {
            "episode_id": "string",
            "global_task": "string",
            "images": "camera -> {path, frame_index}",
            "completed_subtask_history": "list[object]",
            "next_subtask_instruction": "string",
            "next_subtask_name": "string",
            "next_subtask_stage": "string",
        },
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    _write_schema_doc(schema_path, dataset_path, records, stats, records[0] if records else None)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("\nComplete first sample:")
    print(json.dumps(records[0] if records else None, ensure_ascii=False, indent=2))
    print(f"\nWrote statistics: {stats_path}")
    print(f"Wrote schema: {schema_path}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_PLANNER_DATASET)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--stats-output", type=Path, default=Path("outputs/planner_dataset_stats.json"))
    parser.add_argument("--schema-output", type=Path, default=Path("docs/planner_dataset_schema.md"))
    args = parser.parse_args()
    dataset_path = choose_dataset_path(args.dataset)
    analyze(dataset_path, args.data_root, args.schema_output, args.stats_output)


if __name__ == "__main__":
    main()
