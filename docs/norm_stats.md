# 归一化统计量

训练前使用 [`scripts/compute_norm_stats.py`](../scripts/compute_norm_stats.py) 或 [`scripts/compute_norm_stats_fast.py`](../scripts/compute_norm_stats_fast.py) 生成数据集的 state/action 统计量。

统计量属于数据集和机器人配置，不应提交到公共仓库。训练配置通过 `AssetsConfig` 指向本地 `assets/<asset_id>/norm_stats.json`，并会将使用的统计量复制到 checkpoint 的 `assets` 目录中，保证之后评估可复现。
