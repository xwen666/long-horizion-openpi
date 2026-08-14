# Planner Dataset Schema

## Dataset Path

- Input file: `/cc/openpi_wam/robocasa/processed/composite_subtasks/annotations.jsonl`
- Data root for image/video references: `/cc/openpi_wam/robocasa/datasets/v1.0/target/composite`
- Format: `jsonl`

## Dataset Size

- Planner samples: **84401**
- Unique episodes: **16181**
- Unique global tasks: **958**

## Source Sample Format

The source records are not rewritten by this inspection step. Their top-level fields are:

| Field | Types observed | Missing |
| --- | --- | ---: |
| `sample_id` | str (84401) | 0 |
| `dataset_task` | str (84401) | 0 |
| `global_task` | str (84401) | 0 |
| `task_name` | str (84401) | 0 |
| `episode_index` | int (84401) | 0 |
| `frame_index` | int (84401) | 0 |
| `subtask_idx` | int (84401) | 0 |
| `next_subtask_instruction` | str (84401) | 0 |
| `next_subtask_name` | str (84401) | 0 |
| `next_subtask_stage` | str (84401) | 0 |
| `completed_subtask_history` | array (84401) | 0 |
| `images` | object (84401) | 0 |
| `source` | object (84401) | 0 |
| `terminal` | bool (84401) | 0 |

## Normalized Planner Fields

The tools normalize aliases into the following fields:

| Field | Meaning |
| --- | --- |
| `episode_id` | string |
| `global_task` | string |
| `images` | object: camera name -> {path, frame_index} |
| `history` | array of {instruction, skill, stage, optional subtask_idx} |
| `previous_result` | string (derived success/not_applicable when absent in source) |
| `target` | object: instruction, skill, stage, optional subtask_idx/frame_index |
| `observation_timestep` | integer/float or null |
| `target_timestep` | integer/float or null |

## Image Fields

- Missing front image: **0**
- Missing wrist image: **0**
- Invalid image/video paths: **0**
- RoboCasa camera mapping: `agentview_left/right` is treated as front and `eye_in_hand` as wrist.
- Image records keep `video_path + frame_index`; no image is copied during manifest inspection.

## Text, History, and Target Fields

- `global_task` is the complete composite task instruction.
- `history` contains already completed semantic subtasks.
- `target.instruction` is exactly one next semantic subtask.
- `target.skill` and `target.stage` come from `next_subtask_name` and `next_subtask_stage` when present.
- The current RoboCasa manifest has no raw `previous_result`; the tools derive `success` for samples with completed history and `not_applicable` for the first subtask.

## Missing and Invalid Samples

- Empty global task: **0**
- Empty target instruction: **0**
- Repeated target instruction in history (warning): **5249**
- Observation is after target transition: **0**
- Missing episode id: **0**

## Complete Sample

```json
{
  "sample_id": "ArrangeBreadBasket/episode_000000/subtask_000",
  "dataset_task": "ArrangeBreadBasket",
  "global_task": "Open the cabinet, pick up the bread from the cabinet and place it in the basket. Then move the basket to the dining counter.",
  "task_name": "ArrangeBreadBasket",
  "episode_index": 0,
  "frame_index": 0,
  "subtask_idx": 0,
  "next_subtask_instruction": "open the cabinet door above",
  "next_subtask_name": "OpenCabinet",
  "next_subtask_stage": "execute",
  "completed_subtask_history": [],
  "images": {
    "observation.images.robot0_eye_in_hand": {
      "video_path": "ArrangeBreadBasket/20250809/lerobot/videos/chunk-000/observation.images.robot0_eye_in_hand/episode_000000.mp4",
      "frame_index": 0
    },
    "observation.images.robot0_agentview_left": {
      "video_path": "ArrangeBreadBasket/20250809/lerobot/videos/chunk-000/observation.images.robot0_agentview_left/episode_000000.mp4",
      "frame_index": 0
    },
    "observation.images.robot0_agentview_right": {
      "video_path": "ArrangeBreadBasket/20250809/lerobot/videos/chunk-000/observation.images.robot0_agentview_right/episode_000000.mp4",
      "frame_index": 0
    }
  },
  "source": {
    "parquet_path": "ArrangeBreadBasket/20250809/lerobot/data/chunk-000/episode_000000.parquet",
    "row_index": 0
  },
  "terminal": false
}
```

