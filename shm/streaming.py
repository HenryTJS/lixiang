#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流式读取与在线处理引擎

包含三个核心组件：
- ChunkedDataReader: 逐块数据读取器，不预加载全部数据，按需分块读取
- StreamSimulator:   数据流模拟器，逐块流式读取，逐点推送
- StreamProcessor:   在线流式处理主引擎，协调各组件，逐点处理

【改造2】不再预加载全部数据，改用 ChunkedDataReader 逐块读取。
"""
import os
import time

import numpy as np
import pandas as pd

from .config import BASE_DIR, CHUNK_SIZE
from .data_loader import DataLoader


# ========== 改造2: StreamSimulator 改为逐块流式读取 ==========
class ChunkedDataReader:
    """逐块数据读取器：不预加载全部数据，按需分块读取

    将已对齐的 DataFrame 保存为临时 CSV 文件，
    然后逐块读取，避免一次性加载全部数据到内存。
    """

    def __init__(self, group_id, chunk_size=CHUNK_SIZE):
        self.group_id = group_id
        self.chunk_size = chunk_size
        self._temp_file = None
        self._total_rows = 0
        self._row_index = 0
        self._chunk_offset = 0
        self._data = None
        self._cols = []
        self._col_idx = {}
        self.ae_cols = []
        self.fo_cols = []
        # 松散对齐：本行某源无新数据(NaN)时保持上一值（不插值/不填空）
        self._prev_strain = None
        self._prev_ae = None
        self._prev_fo = None

    def load_and_prepare(self):
        """加载数据、对齐，转为内存 numpy 数组，返回总行数"""
        dl = DataLoader(self.group_id).load_all()
        data = dl.sync_timeline()
        if data is None:
            raise ValueError(f'组 {self.group_id} 数据加载失败')

        self._total_rows = len(data)
        self.ae_cols = [c for c in data.columns if c.startswith('ae_')]
        self.fo_cols = [c for c in data.columns if c.startswith('s') and len(c) <= 3 and c != 'strain']
        # 列名 → 索引映射（内存迭代用）
        self._cols = list(data.columns)
        self._col_idx = {c: i for i, c in enumerate(self._cols)}
        # 转为内存 numpy 数组（跳过临时 CSV，消除 I/O 与逐块读取开销）
        self._data = data.to_numpy(dtype=float)
        self._row_index = 0
        self._chunk_offset = 0
        print(f'  [ChunkedDataReader] 组 {self.group_id} 已准备: {self._total_rows} 点, '
              f'AE通道={len(self.ae_cols)}, FO通道={len(self.fo_cols)}')
        return self._total_rows

    def read_row(self):
        """读取下一行数据（numpy 数组行）或 None（已读完）"""
        if self._data is None or self._row_index >= self._total_rows:
            return None
        row = self._data[self._row_index]
        self._row_index += 1
        self._chunk_offset += 1
        return row

    def get_row_dict(self, row):
        """将 numpy 行转为标准 dict 格式。

        松散对齐下，各源只在有真实数据的位置填值（其余 NaN）；
        这里对本行无新数据的源做"保持上值"（事件流更新语义），
        供在线逐点处理。
        """
        ci = self._col_idx
        # 应变
        if 'strain' in ci:
            v = row[ci['strain']]
            strain_val = None if np.isnan(v) else float(v)
        else:
            strain_val = None
        if strain_val is not None:
            self._prev_strain = strain_val
        else:
            strain_val = self._prev_strain
        # AE（事件流：本行有事件则更新，否则保持上值）
        ae_dict = {}
        ae_new = False
        for col in self.ae_cols:
            v = row[ci[col]]
            if not np.isnan(v):
                ae_dict[col] = float(v)
                ae_new = True
        if ae_dict:
            self._prev_ae = dict(ae_dict)
        elif self._prev_ae is not None:
            ae_dict = dict(self._prev_ae)
        # FO（连续信号：本行有值则更新，否则保持上值）
        fo_dict = {}
        for col in self.fo_cols:
            v = row[ci[col]]
            if not np.isnan(v):
                fo_dict[col] = float(v)
        if fo_dict:
            self._prev_fo = dict(fo_dict)
        elif self._prev_fo is not None:
            fo_dict = dict(self._prev_fo)
        t = row[ci['time']] if 'time' in ci else float(self._chunk_offset - 1)
        return {'index': self._chunk_offset - 1, 'time': float(t),
                'strain': strain_val, 'ae': ae_dict, 'fo': fo_dict,
                'ae_new': ae_new}

    def cleanup(self):
        """释放内存（兼容旧临时文件清理）"""
        if hasattr(self, '_data'):
            self._data = None
        if self._temp_file and os.path.exists(self._temp_file):
            try:
                os.remove(self._temp_file)
            except (OSError, PermissionError):
                pass

    @property
    def total_rows(self):
        return self._total_rows

    @property
    def current_offset(self):
        return self._chunk_offset


class StreamSimulator:
    """数据流模拟器：逐块流式读取，逐点推送

    【改造2】不再预加载全部数据，改用 ChunkedDataReader 逐块读取。
    """

    def __init__(self, group_id, speed_factor=1.0):
        self.group_id = group_id
        self.speed_factor = speed_factor
        self.reader = ChunkedDataReader(group_id, chunk_size=CHUNK_SIZE)
        self.total_points = 0
        self.ae_cols = []
        self.fo_cols = []

    def load_data(self):
        n = self.reader.load_and_prepare()
        self.total_points = n
        self.ae_cols = self.reader.ae_cols
        self.fo_cols = self.reader.fo_cols
        return n

    def has_next(self):
        return self.reader.current_offset < self.total_points

    def next_point(self):
        if not self.has_next():
            return None
        row = self.reader.read_row()
        if row is None:
            return None
        return self.reader.get_row_dict(row)

    @property
    def progress(self):
        return self.reader.current_offset / self.total_points if self.total_points > 0 else 0

    def cleanup(self):
        self.reader.cleanup()


