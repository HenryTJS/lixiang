#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""shm 包：多源连续损伤度(D) 在线评估与流式数据基础设施

正式方法 = 连续损伤度 D(t) + 分级预警（见 damage_index.py）。
四阶段划分(stage_divider)体系已弃用(2026-09-06)。

模块结构：
    shm/
    ├── __init__.py          # 导出核心接口
    ├── config.py            # 全局常量/路径（EXT_BLOCK_PTS, CHUNK_SIZE 等）
    ├── data_loader.py       # DataLoader：多源数据加载（StreamSimulator 依赖）
    ├── streaming.py         # ChunkedDataReader + StreamSimulator（逐点流式）
    └── damage_index.py      # OnlineDamageIndex：连续损伤度 D(t) + 分级预警

用法：
    from shm.streaming import StreamSimulator
    from shm.damage_index import OnlineDamageIndex
    from shm.data_loader import DataLoader
"""
from .config import BASE_DIR, DEFAULT_GROUPS, CHUNK_SIZE
from .data_loader import DataLoader
from .streaming import ChunkedDataReader, StreamSimulator
from .damage_index import OnlineDamageIndex

__all__ = [
    'BASE_DIR', 'DEFAULT_GROUPS', 'CHUNK_SIZE',
    'DataLoader', 'ChunkedDataReader', 'StreamSimulator', 'OnlineDamageIndex',
]
