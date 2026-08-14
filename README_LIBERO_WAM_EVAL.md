# LIBERO 上使用 OpenPI + WAM/Cosmos 的两种模式

本文档对应当前 checkpoint：

```bash
/cc/openpi_wam/checkpoints/pi05_cosmos_libero_all/libero_all_wam_full_4gpu_bs4_worldpilot_dropout/37500
```

## 环境说明

这里有两个 Python 虚拟环境，不要混用：

```text
OpenPI/LIBERO 环境：
/cc/openpi_wam/.venv

Cosmos/WAM 环境：
/cc/openpi_wam/cosmos-predict2.5/.venv
```

训练、policy server、LIBERO eval 都在 OpenPI 环境里运行：

```bash
cd /cc/openpi_wam
source .venv/bin/activate
```

实时 WAM 推理不是在当前 shell 里手动 `source` Cosmos 环境，而是由
`scripts/serve_policy.py` 自动启动一个 Cosmos worker 子进程，并通过下面这个参数指定
Cosmos 的 Python：

```bash
--cosmos-python=/cc/openpi_wam/cosmos-predict2.5/.venv/bin/python
```

如果你想单独调试 Cosmos worker，再进入 Cosmos 环境：

```bash
cd /cc/openpi_wam/cosmos-predict2.5
source .venv/bin/activate
```

## 模式 1：训练时使用预存 cache

训练不要实时跑 WAM。训练数据 loader 读取已经生成好的 `.npz` latent cache：

```python
TrainConfig(
    name="pi05_cosmos_libero_all",
    data=LeRobotLiberoCosmosCombinedDataConfig(
        root="/cc/openpi_wam/datasets/libero_lerobot",
        cosmos_latent_cache_root="/cc/openpi_wam/cosmos_cache/WorldPilot-LIBERO-precompute/cosmos_cache",
        ...
    ),
)
```

实际调用链是：

```text
LeRobot sample
  -> RepackTransform
  -> AttachCosmosLatent
  -> LiberoInputs
  -> Normalize
  -> model transforms
```

`AttachCosmosLatent` 会按 `episode_index/frame_index/cosmos_cache_dir` 读取 cache，并产生：

```text
cosmos_latent       [2, 12544]
cosmos_latent_mask  [2]
cosmos_time_ids     [2]  # 默认 16
cosmos_view_ids     [2]  # image=0, wrist_image=1
```

WorldPilot cache 的实际 episode 文件还包含：

```text
future_image_latents  [T, 2, 16, 28, 28] float16
action_chunk          [T, 16, 7]          float16
value                 [T]                 float32
```

训练配置中的 `action_dim=32` 是 OpenPI π0.5 的 padded policy action space；
Action Steering 使用显式映射 `source_to_policy=(0,1,2,3,4,5,6)`，所以 cache 的
7 维 LIBERO action 不会被静默截断或补齐。它会先对齐到 `[B,16,32]`，再压成一个
`[B,1,D_action_expert]` token。

这 7 维不是关节绝对角度，而是 LIBERO 的末端执行器动作表示：
`[x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper]`。输入 state 是
`[x, y, z, axis_angle1, axis_angle2, axis_angle3, gripper, gripper]`；在线送给
Cosmos Policy 时，worker 会把 axis-angle state 显式转换成其要求的
`[gripper2, xyz, quaternion_xyzw]`，不会把两种动作语义混用。

继续训练示例：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=4,5,7,0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 \
python scripts/train.py pi05_cosmos_libero_all \
  --exp-name=libero_all_wam_full_4gpu_bs4_worldpilot_dropout \
  --resume \
  --fsdp-devices=4 \
  --batch-size=16 \
  --num-workers=8 \
  --model.cosmos-condition-dropout=0.3
```

## 模式 2：推理/eval 时实时预测 WAM latent

推理时不读 `.npz`。OpenPI policy server 会启动一个常驻 Cosmos worker 子进程。
配置 Cosmos Policy checkpoint 后，worker 每次 policy decision 只调用一次 Cosmos
Policy；同一个 generated latent sequence 同时提供 future scene latent 和 16 步
anticipated action，不会再额外加载第二个 Cosmos Predict 模型：

```text
LIBERO observation
  -> AttachRealtimeCosmosLatent
  -> frozen Cosmos Policy(o_t, zero-proprio, instruction)
  -> future image latent slots + normalized action chunk [16,7]
  -> LiberoInputs
  -> Normalize
  -> pi0.5 + WAM steering policy
```

这里的实时 WAM 是按 action chunk 粒度调用的：每次 policy server 预测一个
16-step action chunk 时调用一次 WAM，然后 LIBERO 环境执行这个 chunk。不是每个
environment step 都调用一次 WAM。

注意：公开 WorldPilot 的 LIBERO cache 生成和官方 eval 都将 `proprio=None` 传给
Cosmos Policy，服务端实际使用全零 9D proprio。为了让实时 WAM 与训练时的离线
cache 保持同一分布，本仓库默认也使用 zero-proprio。只有明确传入
`--cosmos-policy-use-proprio` 时，才会把 OpenPI 的实时 state 传给 Cosmos；
这会改变 action-prior 和 future latent，不能与当前公开 cache 做逐点复现。

启动 policy server：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=6 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/serve_policy.py \
  --port=8000 \
  --cosmos-cache-mode=realtime \
  --cosmos-python=/cc/openpi_wam/cosmos-predict2.5/.venv/bin/python \
  --cosmos-worker-cuda-visible-devices=7 \
  --cosmos-resolution=224,224 \
  --cosmos-num-steps=5 \
  --cosmos-guidance=7.0 \
  --cosmos-latent-dim=12544 \
  --cosmos-policy-checkpoint=/cc/openpi_wam/cosmos_checkpoints/Cosmos-Policy-LIBERO-Predict2-2B \
  --cosmos-policy-config=cosmos_predict2_2b_480p_libero__inference_only \
  --cosmos-policy-config-file=cosmos_predict2/_src/predict2/cosmos_policy/config/config.py \
  --cosmos-policy-dataset-stats=/cc/openpi_wam/cosmos_checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
  --cosmos-policy-text-embeddings=/cc/openpi_wam/cosmos_checkpoints/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
  --cosmos-policy-num-steps=5 \
  --cosmos-policy-chunk-size=16 \
  --cosmos-policy-action-dim=7 \
  policy:checkpoint \
  --policy.config=pi05_cosmos_libero_all \
  --policy.dir=/cc/openpi_wam/checkpoints/pi05_cosmos_libero_all/libero_all_wam_full_4gpu_bs4_worldpilot_dropout/37500
```

然后另开一个终端先跑标准 LIBERO eval。标准 LIBERO 的正式口径是每个 suite
`10 tasks x 50 init states = 500 rollouts`。单个 suite 命令如下：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python examples/libero/main.py \
  --args.host=127.0.0.1 \
  --args.port=8000 \
  --args.libero-mode=standard \
  --args.task-suite-name=libero_spatial \
  --args.task-start=0 \
  --args.max-tasks=10 \
  --args.num-trials-per-task=50 \
  --args.replan-steps=16 \
  --args.no-save-videos \
  --args.results-path=outputs/eval/libero_standard_37500/results/libero_spatial.jsonl \
  --args.video-out-path=outputs/eval/libero_standard_37500/videos/libero_spatial
```

四个标准 LIBERO suite 全量评测是 `2000` 条 rollout。可以用脚本排队执行：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

tmux new-session -d -s eval_libero_standard_37500 \
  /cc/openpi_wam/scripts/run_libero_standard_eval_37500.sh
tmux attach -t eval_libero_standard_37500
```

确认标准 LIBERO 跑通后，再切到 LIBERO-plus 扰动任务：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python examples/libero/main.py \
  --args.host=127.0.0.1 \
  --args.port=8000 \
  --args.libero-mode=plus \
  --args.task-suite-name=libero_spatial \
  --args.task-start=0 \
  --args.max-tasks=10 \
  --args.num-trials-per-task=1 \
  --args.replan-steps=16 \
  --args.video-out-path=data/libero/videos/pi05_cosmos_37500_liberoplus_spatial
```

LIBERO-plus 的 `libero_spatial/object/goal/10` 是几千个扰动任务，建议先用
`--args.max-tasks=10` 做小切片，确认没问题后再扩大。

`--replan-steps=16` 会执行完整 action chunk，每 16 个 env step 重新预测一次，
所以 WAM 也是每 16 个 env step 调用一次。若只想测试闭环更频繁重规划，可以临时改小；
例如每个环境 step 都重新用当前 observation 调一次 WAM 和 policy：

```bash
--replan-steps=1
```

对于上面的正式 WorldPilot 路径，真正控制 action/future latent 采样的是
`--cosmos-policy-num-steps=5`。`--cosmos-num-steps` 和 `--cosmos-guidance` 只在没有
配置 `--cosmos-policy-checkpoint`、退回 Cosmos Predict `predict_video2world` 兼容路径时
生效。Cosmos Predict 兼容路径只提供视觉 latent，不提供 anticipated action；在训练配置
`use_action_steering=True` 时，正式评测必须提供 Cosmos Policy checkpoint。

如果只是调试服务故障回退，可以显式增加：

```bash
--cosmos-allow-wam-fallback
```

此时 worker 不可用或没有返回 action prior 时，模型收到的是全 masked 的视觉条件和
`wam_action_prior_valid=False` 的零 prior，等价于 baseline/vision-only 条件；这不应
用于报告 full Action Steering 成绩。

## GPU 分配建议

建议 OpenPI policy 和 Cosmos worker 分不同 GPU：

```text
CUDA_VISIBLE_DEVICES=6                 # OpenPI / JAX policy server
--cosmos-worker-cuda-visible-devices=7 # Cosmos Policy/PyTorch worker
```

如果你使用的是没有 Cosmos Policy checkpoint 的 Predict-only 兼容路径，显存紧张时可以
给 worker 加 offload：

```bash
--cosmos-offload-diffusion-model \
--cosmos-offload-text-encoder \
--cosmos-offload-tokenizer
```

worker 日志默认在：

```bash
/tmp/openpi_cosmos_realtime_worker.log
```

## 离线 cache 与实时 WAM 对齐检查

正式跑长时间 eval 前，先用同一个 LIBERO 帧比较两条路径：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python tools/compare_worldpilot_cache_realtime.py \
  --suite=libero_spatial \
  --episode=11 \
  --frame=0 \
  --gpu=0 \
  --num-steps=5 \
  --proprio-mode=cache \
  --output-json=outputs/diagnostics/cache_realtime_compare.json
```

`--proprio-mode=cache` 是公开 WorldPilot 的设置：Cosmos 输入使用 `proprio=None`
（服务端内部为全零 9D proprio）。`--proprio-mode=realtime` 才会把数据集里的
真实 state 传入 Cosmos，适合专门测量 state 分布偏移，不适合直接和当前公开
cache 做逐点比较。

检查重点：

- `latent cosine, same view` 应明显高于 `swapped view`，确认 camera 顺序没有反；
- 每个 view 的 mean/std/l2 应接近；
- action-prior 的 shape 应为 `[16, 7]`，MSE 应接近 0，per-dim correlation 应接近 1；
- 默认实时路径已经按 cache convention 使用 zero-proprio。
- 标准 LIBERO simulator eval 会同时提供两套图像：OpenPI 继续使用原来的 180°
  旋转图，Cosmos 使用仅垂直翻转图，以匹配公开 precompute 的输入方向。

当前公开 cache 的 camera 顺序是 `future_image_latents[t] = [wrist, primary]`，实时
worker 已按同样顺序输出。公开 cache 与本地 LeRobot 数据集的 episode 数量也需要
单独审计；训练配置会按实际存在的 `episode_*.npz` 过滤缺失 episode。

## 训练/推理路径检查

训练时 `AttachCosmosLatent` 读取 cache 中同一个 `episode_index/frame_index` 的
`future_image_latents[t]` 与 `action_chunk[t]`，并在首次读取时打印 cache path、keys、
shape、dtype 和统计值。推理时 `AttachRealtimeCosmosLatent` 每次接收一个 observation
调用一次 worker；action chunk 内的 16 个 environment steps 不会重复调用 WAM。

Action prior 的最终路径是：

```text
[B,16,7] cache/online Cosmos Policy action chunk
  -> ActionPriorAligner [B,16,32]
  -> WorldPilotActionEncoder [B,1,D_action_expert]
  -> pi0.5 suffix: [one prior token ; noisy action tokens]
```

它不参与 flow 初始化、prior residual 相加或 action imitation loss；训练目标仍是
原始 π0.5 flow-matching action objective。

## Smoke test

只测试 server 与 worker 通信，不跑真实 WAM：

```bash
cd /cc/openpi_wam
source .venv/bin/activate

python scripts/serve_policy.py \
  --port=8000 \
  --cosmos-cache-mode=realtime \
  --cosmos-python=/cc/openpi_wam/cosmos-predict2.5/.venv/bin/python \
  --cosmos-allow-dummy-latents \
  --cosmos-allow-wam-fallback \
  policy:checkpoint \
  --policy.config=pi05_cosmos_libero_all \
  --policy.dir=/cc/openpi_wam/checkpoints/pi05_cosmos_libero_all/libero_all_wam_full_4gpu_bs4_worldpilot_dropout/37500
```

真实评测不要加 `--cosmos-allow-dummy-latents`。
