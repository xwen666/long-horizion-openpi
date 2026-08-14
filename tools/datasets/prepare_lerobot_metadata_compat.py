#!/usr/bin/env python3
"""Create jsonl metadata sidecars for local LeRobot datasets.

Some OpenPI-pinned LeRobot builds load v3 datasets but still look for the older
`meta/tasks.jsonl` / `meta/episodes.jsonl` sidecars before falling back to the
Hub. This script derives those files from the v3 parquet metadata so local,
offline datasets do not trigger a HuggingFace download.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq

        return pq.read_table(path).to_pylist()
    except ImportError:
        import polars as pl

        return pl.read_parquet(path).to_dicts()


def jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_jsonl(path: Path, records: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        print(f"[metadata] exists: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
    print(f"[metadata] wrote {path} ({len(records)} records)")


def make_tasks_jsonl(meta_dir: Path, *, overwrite: bool) -> None:
    tasks_jsonl = meta_dir / "tasks.jsonl"
    tasks_parquet = meta_dir / "tasks.parquet"
    if not tasks_parquet.exists():
        print(f"[metadata] missing {tasks_parquet}, skipping tasks.jsonl")
        return

    records = []
    for fallback_index, row in enumerate(read_parquet_rows(tasks_parquet)):
        task_index = int(row.get("task_index", fallback_index))
        task = row.get("task") or row.get("__index_level_0__")
        if task is None:
            string_values = [v for v in row.values() if isinstance(v, str)]
            task = string_values[0] if string_values else str(task_index)
        records.append({"task_index": task_index, "task": str(task)})

    records.sort(key=lambda r: r["task_index"])
    write_jsonl(tasks_jsonl, records, overwrite=overwrite)


def split_episode_record(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    episode: dict[str, Any] = {}
    stats: dict[str, dict[str, Any]] = {}

    for key, value in row.items():
        if key.startswith("stats/"):
            _, feature, stat_name = key.split("/", 2)
            stats.setdefault(feature, {})[stat_name] = value
        elif key.startswith("meta/episodes/"):
            continue
        else:
            episode[key] = value

    episode_index = int(episode["episode_index"])
    episode_stats = {"episode_index": episode_index, "stats": stats} if stats else None
    return episode, episode_stats


def make_episode_jsonl(meta_dir: Path, *, overwrite: bool) -> None:
    episodes_dir = meta_dir / "episodes"
    if not episodes_dir.exists():
        print(f"[metadata] missing {episodes_dir}, skipping episodes jsonl")
        return

    episode_records: list[dict[str, Any]] = []
    episode_stats_records: list[dict[str, Any]] = []
    for parquet_path in sorted(episodes_dir.glob("chunk-*/file-*.parquet")):
        for row in read_parquet_rows(parquet_path):
            episode, episode_stats = split_episode_record(row)
            episode_records.append(episode)
            if episode_stats is not None:
                episode_stats_records.append(episode_stats)

    episode_records.sort(key=lambda r: int(r["episode_index"]))
    episode_stats_records.sort(key=lambda r: int(r["episode_index"]))

    write_jsonl(meta_dir / "episodes.jsonl", episode_records, overwrite=overwrite)
    if episode_stats_records:
        write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats_records, overwrite=overwrite)


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    meta_dir = dataset_root / "meta"
    if not meta_dir.exists():
        raise FileNotFoundError(f"Missing metadata directory: {meta_dir}")

    make_tasks_jsonl(meta_dir, overwrite=args.overwrite)
    make_episode_jsonl(meta_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
