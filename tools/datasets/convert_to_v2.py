"""Convert data2_grasp_200 from LeRobot v3.0 to v2.1 format matching /cc/openpi/grasp.

Steps:
1. Split monolithic parquet into per-episode parquet files
2. Extract per-episode videos from chunk videos (AV1 -> H.264)
3. Generate episodes.jsonl, tasks.jsonl, episodes_stats.jsonl
4. Rewrite info.json in v2.1 format
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

SRC = Path("/cc/openpi/data2_grasp_200")
DST = Path("/cc/openpi/grasp_200")
EPISODE_CHUNK = 0
DATA_CHUNK = 0


def _to_json(val):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(val, np.ndarray):
        if val.dtype == np.dtype("O"):
            return [_to_json(v) for v in val.flat]
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, list):
        return [_to_json(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_json(v) for k, v in val.items()}
    return val


def main():
    if DST.exists():
        print(f"Removing existing output: {DST}")
        shutil.rmtree(DST)

    DST.mkdir(parents=True)

    # 1. Read source data
    print("Reading source data...")
    df = pd.read_parquet(SRC / "data/chunk-000/file-000.parquet")
    ep = pd.read_parquet(SRC / "meta/episodes/chunk-000/file-000.parquet")
    tasks = pd.read_parquet(SRC / "meta/tasks.parquet")
    info = json.loads((SRC / "meta/info.json").read_text())
    stats = json.loads((SRC / "meta/stats.json").read_text())

    total_episodes = len(ep)
    total_frames = len(df)
    print(f"Episodes: {total_episodes}, Frames: {total_frames}")

    # 2. Build output directories
    (DST / "data/chunk-000").mkdir(parents=True)
    (DST / "meta").mkdir(parents=True)

    # 3. Write per-episode parquet files
    print("Writing per-episode parquet files...")
    for ep_idx in range(total_episodes):
        ep_row = ep.iloc[ep_idx]
        start = int(ep_row["dataset_from_index"])
        end = int(ep_row["dataset_to_index"])
        ep_df = df.iloc[start:end].copy()
        ep_df["episode_index"] = ep_idx
        path = DST / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
        ep_df.to_parquet(path, index=False)
        if (ep_idx + 1) % 20 == 0:
            print(f"  {ep_idx + 1}/{total_episodes} parquet files written")

    # 4. Extract per-episode videos (AV1 -> H.264)
    video_keys = ["observation.images.front", "observation.images.wrist"]
    for vk in video_keys:
        (DST / f"videos/{vk}/chunk-000").mkdir(parents=True)
    print("\nExtracting and re-encoding videos (this may take a while)...")
    for ep_idx in range(total_episodes):
        ep_row = ep.iloc[ep_idx]
        for vk in video_keys:
            src_chunk = int(ep_row[f"videos/{vk}/chunk_index"])
            src_file = int(ep_row[f"videos/{vk}/file_index"])
            start_ts = float(ep_row[f"videos/{vk}/from_timestamp"])
            end_ts = float(ep_row[f"videos/{vk}/to_timestamp"])
            src_path = SRC / f"videos/{vk}/chunk-{src_chunk:03d}/file-{src_file:03d}.mp4"
            dst_path = DST / f"videos/{vk}/chunk-000/episode_{ep_idx:06d}.mp4"

            if not src_path.exists():
                print(f"  WARNING: Source video not found: {src_path}")
                continue

            # ffmpeg: seek and re-encode segment
            cmd = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{start_ts:.3f}",
                "-to", f"{end_ts:.3f}",
                "-i", str(src_path),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                "-vf", "fps=30",
                str(dst_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR episode {ep_idx} {vk}: {result.stderr.strip()}")
                sys.exit(1)

        if (ep_idx + 1) % 10 == 0:
            print(f"  {ep_idx + 1}/{total_episodes} videos done")

    # 5. Create episodes.jsonl
    print("\nCreating metadata files...")
    episodes_jsonl = []
    episodes_stats_jsonl = []
    task_set = set()
    for ep_idx in range(total_episodes):
        ep_row = ep.iloc[ep_idx]
        episode_tasks = ep_row["tasks"]
        if isinstance(episode_tasks, np.ndarray):
            episode_tasks = episode_tasks.tolist()
        elif isinstance(episode_tasks, list):
            pass
        else:
            episode_tasks = [str(episode_tasks)]

        # Collect all tasks for tasks.jsonl
        for t in episode_tasks:
            task_set.add(t)

        episodes_jsonl.append(
            json.dumps({"episode_index": ep_idx, "tasks": episode_tasks, "length": int(ep_row["length"])})
        )

        # Extract per-episode stats
        ep_stats = {}
        for col in ep.columns:
            if col.startswith("stats/"):
                stat_name = col.replace("stats/", "")
                val = ep_row[col]
                ep_stats[stat_name] = _to_json(val)
        episodes_stats_jsonl.append(json.dumps({"episode_index": ep_idx, "stats": ep_stats}))

    (DST / "meta/episodes.jsonl").write_text("\n".join(episodes_jsonl) + "\n")
    (DST / "meta/episodes_stats.jsonl").write_text("\n".join(episodes_stats_jsonl) + "\n")

    # 6. Create tasks.jsonl
    # Build task_index -> task string mapping
    task_index_map = {}
    for task_str, row in tasks.iterrows():
        task_index_map[int(row["task_index"])] = task_str
    tasks_jsonl = []
    for ti in sorted(task_index_map.keys()):
        tasks_jsonl.append(json.dumps({"task_index": ti, "task": task_index_map[ti]}))
    (DST / "meta/tasks.jsonl").write_text("\n".join(tasks_jsonl) + "\n")

    # 7. Copy stats.json (global stats same)
    shutil.copy(SRC / "meta/stats.json", DST / "meta/stats.json")

    # 8. Write info.json in v2.1 format
    info_v2 = {
        "codebase_version": "v2.1",
        "robot_type": info["robot_type"],
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_index_map),
        "chunks_size": info.get("chunks_size", 1000),
        "data_files_size_in_mb": info.get("data_files_size_in_mb", 100),
        "video_files_size_in_mb": info.get("video_files_size_in_mb", 200),
        "fps": info["fps"],
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        "features": info["features"],
    }
    # Update video codec info to h264
    for feat_key, feat_val in info_v2["features"].items():
        if isinstance(feat_val, dict) and feat_val.get("dtype") == "video":
            feat_val["info"]["video.codec"] = "h264"
            feat_val["info"]["video.pix_fmt"] = "yuv420p"

    (DST / "meta/info.json").write_text(json.dumps(info_v2, indent=4) + "\n")

    print(f"\nConversion complete. Output: {DST}")
    print(f"  Episodes: {total_episodes}")
    print(f"  Frames:   {total_frames}")
    print(f"  Tasks:    {len(task_index_map)}")
    print(f"\nVerifying...")
    verify(DST, total_episodes)


def verify(dst: Path, total_episodes: int):
    """Quick verification of the output."""
    info = json.loads((dst / "meta/info.json").read_text())
    assert info["codebase_version"] == "v2.1"
    assert info["total_episodes"] == total_episodes

    for ep_idx in range(total_episodes):
        assert (dst / f"data/chunk-000/episode_{ep_idx:06d}.parquet").exists()
        assert (dst / f"videos/observation.images.front/chunk-000/episode_{ep_idx:06d}.mp4").exists()
        assert (dst / f"videos/observation.images.wrist/chunk-000/episode_{ep_idx:06d}.mp4").exists()

    assert (dst / "meta/episodes.jsonl").exists()
    assert (dst / "meta/tasks.jsonl").exists()
    assert (dst / "meta/episodes_stats.jsonl").exists()
    assert (dst / "meta/stats.json").exists()

    print("  All files verified OK.")


if __name__ == "__main__":
    main()
