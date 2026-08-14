# 数据集工具

这里放只用于数据整理和 LeRobot 兼容性的脚本，不属于 OpenPI 运行时核心代码。

- `convert_to_v2.py`：将旧的抓取数据转换为 LeRobot v2.1 目录格式。脚本内仍保留原始实验路径，使用前请检查并修改 `SRC`、`DST`。
- `prepare_lerobot_metadata_compat.py`：从 v3 parquet 元数据补生成 `tasks.jsonl` 和 `episodes.jsonl`，用于离线读取本地数据集。

LIBERO、Move-Aloha 和抓取数据本体不放入 Git；请通过 `configs/env.example` 配置本机路径。
