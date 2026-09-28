#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据加载器：统一加载应变、声发射、光纤数据，并进行时间同步

仅负责加载和对齐，不做全局归一化（归一化由 OnlineNormalizer 在线完成）。
"""
import os
import numpy as np
import pandas as pd

from .config import BASE_DIR


def ts_to_seconds(t_series):
    """10Hz 时间戳统一换算为秒（0.1s 计数单位 → ×0.1）。

    所有数据采样率均为 10Hz：光纤 Time Stamp 相邻差≈1.0 为 0.1s 计数；
    应变有两种（dt≈0.1 已是秒 / dt≈1.0 需 ×0.1）。按中位间隔自动判断。
    """
    t = pd.to_numeric(t_series if isinstance(t_series, pd.Series) else pd.Series(t_series),
                      errors='coerce').values.astype(float)
    valid = t[~np.isnan(t)]
    if len(valid) < 2:
        return t
    dt = float(np.median(np.diff(valid)))
    if dt >= 0.2:
        return t * 0.1
    return t


class DataLoader:
    """数据加载器：统一加载应变、声发射、光纤数据，并进行时间同步"""

    def __init__(self, group_id):
        self.group_id = str(group_id).zfill(3)
        self.data_dir = os.path.join(BASE_DIR, self.group_id)
        self.strain = self.ae = self.fo = self.synced_data = None

    def _detect_encoding(self, file_path):
        for enc in ['utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1']:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    f.readline()
                return enc
            except (UnicodeDecodeError, UnicodeError):
                continue
        return 'latin-1'

    def _find_file(self, name_patterns):
        for p in name_patterns:
            fp = os.path.join(self.data_dir, p)
            if os.path.exists(fp):
                return fp
        return None

    def _unify_columns(self, df, col_map_rules):
        df.columns = df.columns.str.strip()
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace(' ', '').replace('(', '').replace(')', '').replace('（', '').replace('）', '').replace('_', '')
            for pattern, target in col_map_rules:
                if pattern in cl:
                    col_map[col] = target
                    break
        for i, (pattern, target) in enumerate([('time', 'time'), ('strain', 'strain')]):
            if target not in col_map.values() and len(df.columns) > i:
                col_map[df.columns[i]] = target
        return df.rename(columns=col_map)

    def load_strain(self):
        fp = self._find_file([f'{self.group_id}应变.csv', f'{self.group_id}应变.xlsx'])
        if fp is None:
            print(f'  [警告] 未找到应变文件: {self.data_dir}')
            return None
        if fp.endswith('.csv'):
            encoding = self._detect_encoding(fp)
            with open(fp, 'r', encoding=encoding) as f:
                sep = '\t' if '\t' in f.readline() else ','
            df = pd.read_csv(fp, sep=sep, encoding=encoding, engine='python')
        else:
            df = pd.read_excel(fp)
        df = self._unify_columns(df, [
            (c, 'time') for c in ['时间', 'time', 'timestamp']
        ] + [
            (c, 'strain') for c in ['应变', 'strain', '幅值']
        ])
        df = df[['time', 'strain']].copy()
        df['time'] = pd.to_numeric(df['time'], errors='coerce')
        df['strain'] = pd.to_numeric(df['strain'], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        df['time'] = ts_to_seconds(df['time'])
        self.strain = df
        return df

    def load_ae(self):
        fp = os.path.join(self.data_dir, f'{self.group_id}声发射.csv')
        if not os.path.exists(fp):
            print(f'  [警告] 未找到声发射文件: {fp}')
            return None
        df = pd.read_csv(fp, encoding='utf-8-sig', engine='python')
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        self.ae = df
        return df

    def load_fo(self):
        fp = self._find_file([f'{self.group_id}光纤.csv', f'{self.group_id}光纤.xlsx'])
        if fp is None:
            print(f'  [警告] 未找到光纤文件: {self.data_dir}')
            return None
        df = pd.read_csv(fp, encoding='utf-8-sig', engine='python') if fp.endswith('.csv') else pd.read_excel(fp)
        df.columns = df.columns.str.strip()
        col_map = {}
        for col in df.columns:
            cl = col.lower().replace(' ', '').replace('(', '').replace(')', '').replace('（', '').replace('）', '').replace('_', '')
            if any(t in cl for t in ['时间', 'time', 'timestamp']):
                col_map[col] = 'time'
            elif cl.startswith('s') and len(col.strip()) <= 3:
                col_map[col] = col.strip()
            elif 'fiber' in cl and 's' in cl:
                col_map[col] = col.split('_')[-1]
        if 'time' not in col_map.values() and len(df.columns) > 0:
            col_map[df.columns[0]] = 'time'
        # 其余未映射通道列统一命名 s1..sN
        # （兼容 strain/strain1 等非标准光纤列名，否则在线系统检测不到 FO 通道）
        ch_idx = 1
        for col in df.columns:
            if col not in col_map:
                col_map[col] = f's{ch_idx}'
                ch_idx += 1
        df = df.rename(columns=col_map)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna().reset_index(drop=True)
        if 'time' in df.columns:
            df['time'] = ts_to_seconds(df['time'])
        self.fo = df
        return df

    def load_all(self):
        print(f'\n>>> 加载组 {self.group_id} 数据')
        self.load_strain(); self.load_ae(); self.load_fo()
        return self

    def sync_timeline(self):
        """松散对齐：统一时间头尾，各源保留原始点（不插值、不填空）。

        主时钟 = 应变时间戳网格（无应变时用光纤）；
        光纤按精确时间戳匹配（同为 10Hz 网格），无匹配置 NaN；
        声发射事件均匀映射到主网格，仅在事件位置填值，其余置 NaN。
        在线流式读取时对 NaN 用"保持上值"（见 ChunkedDataReader.get_row_dict）。
        """
        if self.strain is None and self.fo is None:
            return None
        base = self.strain if self.strain is not None else self.fo
        timeline = base[['time']].copy().sort_values('time').drop_duplicates('time').reset_index(drop=True)
        timeline['time'] = timeline['time'].astype(float).round(3)  # 消除×0.1浮点误差
        result = timeline.copy()

        # 应变（时间基准）
        if self.strain is not None:
            s = self.strain.copy().sort_values('time').drop_duplicates('time')
            s['time'] = s['time'].astype(float).round(3)
            result = result.merge(s, on='time', how='left')

        # 光纤：精确时间戳匹配（同 10Hz 网格），无匹配为 NaN，不做最近邻/插值
        if self.fo is not None:
            fo = self.fo.copy().sort_values('time').drop_duplicates('time')
            fo['time'] = fo['time'].astype(float).round(3)
            result = result.merge(fo, on='time', how='left', suffixes=('', '_fo'))

        # 声发射：事件均匀映射到主网格，仅在事件位置填值，其余为 NaN
        if self.ae is not None and len(self.ae) > 0:
            grid = result['time'].values
            n_grid = len(grid)
            ae_t = np.linspace(grid.min(), grid.max(), len(self.ae))
            idx = np.clip(np.searchsorted(grid, ae_t), 0, n_grid - 1)
            ae_sub = self.ae.copy()
            for c in ae_sub.columns:
                ae_sub[c] = pd.to_numeric(ae_sub[c], errors='coerce')
            for c in ae_sub.columns:
                arr = np.full(n_grid, np.nan)
                arr[idx] = ae_sub[c].values  # 同一网格点多事件 → 后者覆盖
                result[f'ae_{c}'] = arr

        self.synced_data = result
        return result
