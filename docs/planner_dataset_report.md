# RoboCasa Planner 数据集报告

## 1. 数据集概览

- 数据集：`/cc/openpi_wam/robocasa/processed/composite_subtasks/annotations.jsonl`
- Planner 样本数：**84401**
- Episode 数：**16181**
- 全局任务变体数：**958**
- 每个 episode 的平均样本数：**5.216**
- 每个 episode 的样本数范围：**2 - 15**

## 2. 样本结构

源 manifest 保留通用的 RoboCasa 相机名称，并使用 `video_path + frame_index` 引用图像。
每个目标都是一个语义层面的下一个子任务，不包含 action chunk、关节状态或低层控制指令。

```text
global_task
images: camera_name -> {video_path, frame_index}
completed_subtask_history: [{subtask_idx, instruction}]
next_subtask_instruction
next_subtask_name
next_subtask_stage
```

完整的检查结果请参见 [planner_dataset_schema.md](planner_dataset_schema.md)。

## 3. 统计信息

- 缺失 front 图像：**0**
- 缺失 wrist 图像：**0**
- 无效图像路径：**0**
- 原始 `previous_result` 字段：**0**
- 派生的 previous result 字段：**84401**

## 4. 任务和技能分布

### 全局任务（保留原始英文标注）

- `Place one bun and one sausage from the bowl on each plate.`: 5001
- `Pick up the kebab skewer and baguette bread, and place them inside the toaster oven. Close the toaster oven door and start by setting the timer.`: 3962
- `Grab a lemon wedge from the fridge and one ice cube from the ice bowl, and put them in the glass of lemonade.`: 3579
- `Move the plastic bottles in the middle to the plastics group, and the glass bottles in the middle to the glass group.`: 3118
- `Open the cabinet, pick up the bread from the cabinet and place it in the basket. Then move the basket to the dining counter.`: 2968
- `Pick up the knife from the drawer and place it on the cutting board. Then place the meat from the plate to the cutting board.`: 2531
- `Start the toaster. Once the lever pops up, take the bread to the plate on the dining counter.`: 2530
- `Pick the pan and sponge and place them into the sink. Then turn on the water.`: 2505
- `Pick up the cup and bowl from the counter, place them in the dishwasher, and close the dishwasher door.`: 2505
- `Open the microwave, place the bowl with waffle inside the microwave, then close the microwave door and turn it on.`: 2503
- `Pick the kettle from the counter and place it on the tray. Then pick the mug from the cabinet and place it on the tray. Then close the cabinet doors.`: 2500
- `Gather all objects into one cabinet and sort the glasses and bowls to opposite sides.`: 2116
- `From the different types of pastries on the counter, select a croissant and place it on the cutting board. Then retrieve a jar of jam from the cabinet and place it alongside the croissant on the cutting board.`: 2114
- `Pick up the bowls on the counter and stack them on top of one another in the open cabinet. Place the smaller bowl on top of the larger bowl.`: 2087
- `Put the shaker and condiment bottle from the counter next to their counterparts in the cabinet.`: 2052
- `Take a straw from the drawer in front and place it inside the glass cup on the dining counter.`: 2016
- `Take the strawberry from the fridge and place it on top of the pancake, located on the dining counter.`: 2012
- `Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button.`: 1542
- `Pick up the pan and dump the vegetables in it onto the plate. Then return the pan to the stove.`: 1530
- `Pick the kettle from the counter and place it on a stove burner. Then turn the burner on.`: 1503
- `Wash the lettuce in the sink by running water over it.`: 1503
- `Pick up the sponge from the counter and clean the cutting board by briefly scrubbing or pressing down on the cutting board. Once finished, release the sponge.`: 1036
- `Turn on the sink and manuever the spout to wash all locations of the sink basin.`: 1018
- `Place one lemon wedge and one chicken drumstick in each tupperware on the nearby counter, to pack two identical lunches.`: 300
- `Place one lemon and one chicken drumstick in each tupperware on the nearby counter, to pack two identical lunches.`: 299
- `Place one cucumber and one chicken drumstick in each tupperware on the nearby counter, to pack two identical lunches.`: 297
- `Pick the lime from the sink and place it in the bowl. Then pick the bowl and place it in the microwave. Then close the microwave door and press the start button.`: 294
- `Pick the canned food and place it on the digital scale for weighing, and close the cabinet.`: 270
- `Pick the chicken drumstick and lemon from their plates and place them in the bowl. Then put the bowl in the fridge.`: 259
- `Pick the broccoli from the sink and place it in the bowl. Then pick the bowl and place it in the microwave. Then close the microwave door and press the start button.`: 257
- `Place one sweet potato and one chicken drumstick in each tupperware on the nearby counter, to pack two identical lunches.`: 254
- `Place one avocado and one chicken drumstick in each tupperware on the nearby counter, to pack two identical lunches.`: 253

### 原子技能

- `PickPlaceCounterToCounter`: 14220
- `NavigateKitchen`: 12267
- `PickPlaceFridgeToCounter`: 6034
- `PickPlaceCounterToSink`: 5792
- `PickPlaceCounterToCabinet`: 4112
- `PickPlaceCabinetToCounter`: 4042
- `PickPlaceCounterToStove`: 4008
- `PickPlaceDrawerToCounter`: 3018
- `PickPlaceCounterToMicrowave`: 2022
- `TurnOnSinkFaucet`: 2018
- `PickPlaceCounterToDishwasher`: 2004
- `PickPlaceCounterToFreezer`: 2004
- `PickPlaceCounterToToasterOven`: 1998
- `CoffeeSetupMug`: 1028
- `PickPlaceSinkToCounter`: 1022
- `PickPlaceCabinetToCabinet`: 1016
- `OpenFridge`: 1015
- `PickPlaceToasterToCounter`: 1012
- `TurnOnToaster`: 1012
- `CloseMicrowave`: 1011
- `TurnOnMicrowave`: 1011
- `PickPlaceCounterToFridge`: 1006
- `CloseCabinet`: 1004
- `PickPlaceCabinetToStove`: 1002
- `TurnOnStove`: 1002
- `SlideToasterOvenRack`: 958
- `TurnSinkSpout`: 652
- `StartCoffeeMachine`: 514
- `Pick`: 510
- `Place`: 510
- `Tilt`: 510
- `OpenDrawer`: 504
- `PickUpSponge`: 504
- `ScrubCuttingBoard`: 504
- `CloseToasterOvenDoor`: 503
- `TurnOnToasterOven`: 503
- `CloseDishwasher`: 501
- `PickUpSpatula`: 501
- `StirVegetables`: 501
- `OpenMicrowave`: 500
- `OpenCabinet`: 410
- `MoveMugsAside`: 133
- `MoveBowlAside`: 3

### 阶段

- `pick`: 29185
- `place`: 28180
- `execute`: 13247
- `navigate`: 12267
- `tilt`: 510
- `press`: 506
- `wait`: 506

## 5. History 长度分布

- `0`: 16181
- `1`: 16181
- `2`: 14679
- `3`: 12109
- `4`: 9117
- `5`: 5955
- `6`: 3688
- `7`: 1933
- `8`: 899
- `9`: 786
- `10`: 713
- `11`: 592
- `12`: 548
- `13`: 526
- `14`: 494

## 6. 样本示例

下面的任务指令和技能名称保留数据集原始标注，用于和源数据逐项对照。

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

## 7. 数据质量问题

- `boundary_correctness`（边界正确性）：**pass** (0)
- `empty_target`（目标为空）：**pass** (0)
- `history_leakage`（History 重复指令）：**warning** (5249)
- `image_leakage`（图像时间泄漏）：**pass** (0)
- `sequence_consistency`（序列一致性）：**pass** (0)
- `invalid_image_path`（无效图像路径）：**pass** (0)

- 问题记录数：**500**（report JSON 最多记录 500 条具体问题）。
- 问题计数：`{'repeated_target_instruction': 5249}`

## 8. Train/Val/Test 划分

- `train` episode 数：**12944**，样本数：**67530**
- `val` episode 数：**1618**，样本数：**8423**
- `test` episode 数：**1619**，样本数：**8448**
- Episode 重叠情况：`{'train_val': 0, 'test_train': 0, 'test_val': 0}`

## 9. Qwen3-VL 训练格式

转换器会生成 system/user/assistant 消息。user 消息包含 front 和 wrist 图像，后面跟随 planner prompt；assistant 内容是一个 JSON 字符串，格式如下：

```json
{"mode":"execute", "instruction":"...", "atomic_skill":"...", "stage":"..."}
```

使用下面的命令生成包含真实 JPEG 帧、可供 processor 使用的版本：

```bash
python tools/convert_to_qwen3vl_format.py --image-mode frame
```

本 pipeline 不会启动 Qwen3-VL 训练。
