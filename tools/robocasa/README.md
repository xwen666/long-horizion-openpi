# RoboCasa Composite 子任务标注

`prepare_composite_subtask_data.py` 用于检查 RoboCasa LeRobot 数据的真实
annotation 字段，并将 32 个 composite task 转成高层子任务指令数据。

## 真实字段

parquet 中的 annotation 不是字符串，而是整数索引：

```text
annotation.human.task_description  -> 全局任务文本
annotation.human.task_name         -> composite task 名称
annotation.human.subtask           -> 当前子任务指令
annotation.human.subtask_name      -> 原子任务类型，例如 PickPlaceCounterToCounter
annotation.human.subtask_stage     -> 当前阶段，例如 pick/place/navigate
subtask_idx                        -> 子任务在当前 trajectory 中的顺序
```

整数通过同一数据集的 `meta/tasks.jsonl` 解码。不要把 `subtask_name` 当成
自然语言指令；训练标签使用 `annotation.human.subtask` 解码后的文本。

## 检查一个 trajectory

在仓库根目录执行：

```bash
python tools/robocasa/prepare_composite_subtask_data.py inspect \
  --data-root /cc/openpi_wam/robocasa/datasets/v1.0/target/composite \
  --task ArrangeBreadBasket \
  --episode 0
```

脚本会打印 parquet 的实际 annotation 列、索引对应的文字和所有
`subtask_idx` 切换帧。

## 生成转换结果

默认遍历已经下载的全部 32 个 composite task 和 16,181 个 episode：

```bash
python tools/robocasa/prepare_composite_subtask_data.py convert \
  --data-root /cc/openpi_wam/robocasa/datasets/v1.0/target/composite \
  --output-dir /cc/openpi_wam/robocasa/processed/composite_subtasks
```

输出：

```text
robocasa/processed/composite_subtasks/
├── annotations.jsonl
└── summary.json
```

默认每个连续 `subtask_idx` 段只取首帧，因此不会把约 1,276 万帧全部展开
成超大的 JSONL。最终样本结构为：

```json
{
  "global_task": "...",
  "images": {
    "observation.images.robot0_eye_in_hand": {
      "video_path": "ArrangeBreadBasket/20250809/lerobot/videos/...mp4",
      "frame_index": 218
    }
  },
  "completed_subtask_history": [
    {
      "subtask_idx": 0,
      "instruction": "open the cabinet door above",
      "skill": "OpenCabinet",
      "stage": "execute",
      "result": "success"
    }
  ],
  "next_subtask_instruction": "pick up the bread from the cabinet"
}
```

`video_path` 默认是相对于 `--data-root` 的路径；训练数据加载器需要把它和
`/cc/openpi_wam/robocasa/datasets/v1.0/target/composite` 拼接，然后从
`frame_index` 解码当前图像。脚本不会复制或修改原始视频。

如果确实需要每一帧一条记录，可以使用：

```bash
python tools/robocasa/prepare_composite_subtask_data.py convert \
  --sample-mode every-frame \
  --output-dir /cc/openpi_wam/robocasa/processed/composite_subtasks_every_frame
```

这会生成约 1,276 万条记录，通常不建议作为第一版 Qwen3-VL 训练输入。
