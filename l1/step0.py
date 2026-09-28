"""
Step 0: 数据预处理 — 将原始 SHM 数据转换为 CSV 格式
=====================================================

本脚本将 ReMAP H2020 项目中第 1 组试验 (TU Delft Campaign) 的原始 SHM 数据
（AE、FBG、LUNA DFOS）转换为标准 CSV 格式，便于后续分析和建模。

数据模态:
  - AE (声发射): Vallen AMSY-6, 4×VS900-M, .pridb/.tradb 格式
  - FBG (光纤光栅): Micron Optics sm130, Sensors.*.txt (ENLIGHT 格式)
  - LUNA DFOS (分布式光纤传感): LUNA ODiSI-B, .txt 矩阵格式

输出:
  每个试件根目录下生成 3 个 CSV 文件:
  - {specimen}/{specimen}声发射.csv   (AE 数据)
  - {specimen}/{specimen}光纤.csv     (FBG 光纤光栅数据)
  - {specimen}/{specimen}分布式应变.csv     (LUNA DFOS 数据)

  说明: 光纤/分布式应变 CSV 的第一列 timestamp 由绝对时间转为相对开始的
  秒数 (每路各自的首条记录 = 0)。

依赖:
  pip install pandas numpy vallenae

作者: Roo
日期: 2026-07-29
"""

import pandas as pd
import numpy as np
import os
import warnings
import vallenae as vae
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================
# 全局配置
# ============================================================
DATA_ROOT = '.'  # 当前工作目录 (e:/TU-Delft)

# 1st TU Delft Campaign 试件 (Paper 1: Broer et al. 2022)
SPECIMENS_TUDELFT = ['L1-03', 'L1-04', 'L1-05', 'L1-09']

# AE 传感器配置 (4个 VS900-M 传感器)
AE_SENSOR_POSITIONS = {
    1: (145, 190),   # Ch1: (145, 190) mm
    2: (145, 20),    # Ch2: (145, 20) mm
    3: (20, 50),     # Ch3: (20, 50) mm
    4: (20, 220),    # Ch4: (20, 220) mm
}

AE_WAVE_VELOCITIES = {
    'longitudinal': 5586.59,  # m/s
    'lateral': 4054.05,       # m/s
}


def to_relative_seconds(timestamps):
    """将绝对时间戳序列转换为相对开始的秒数（该路首条记录 = 0）。

    参数:
        timestamps: datetime64 数组 / datetime 序列（需已按时间升序排列）

    返回:
        np.ndarray: 相对开始的秒数 (float)；无法解析的位置为 NaN
    """
    ts = pd.to_datetime(pd.Series(timestamps), errors='coerce')
    t0 = ts.dropna().iloc[0]
    return (ts - t0).dt.total_seconds().values


# ============================================================
# 1. LUNA DFOS → CSV
# ============================================================

def load_luna(filepath):
    """
    加载 LUNA ODiSI-B DFOS 数据。

    参数:
        filepath: .txt 文件路径

    返回:
        timestamps: 时间戳数组 (pd.DatetimeIndex)
        positions:  空间位置数组 (mm)
        strain_map: 应变矩阵 (n_timestamps × n_positions), 单位 με
    """
    if not os.path.exists(filepath):
        print(f'  [警告] 文件不存在: {filepath}')
        return None, None, None

    print(f'  [LUNA] 加载: {os.path.basename(filepath)}')

    # 跳过4行文件头, 读取数据
    df = pd.read_csv(filepath, sep='\t', skiprows=4, header=0,
                     encoding='utf-8', engine='python')

    time_col = df.columns[0]

    # 提取位置信息 (mm)
    positions = df.columns[1:].astype(float).values

    # 提取时间戳
    timestamps = pd.to_datetime(df[time_col], errors='coerce')

    # 提取应变矩阵 (με)
    strain_cols = df.columns[1:]
    strain_map = df[strain_cols].values.astype(float)

    n_times, n_pos = strain_map.shape
    print(f'    时间点: {n_times}, 空间位置: {n_pos}')
    print(f'    时间范围: {timestamps.min()} ~ {timestamps.max()}')

    return timestamps, positions, strain_map


def save_luna_to_csv(timestamps, positions, strain_map, output_path):
    """
    将 LUNA 数据保存为 CSV 文件。

    CSV 格式:
      - 第1列: timestamp (相对开始的秒数, 本文件首条=0)
      - 第2~N列: 各空间位置的应变值 (με), 列名为位置值 (mm)
    """
    if timestamps is None or strain_map is None:
        print(f'  [跳过] 无数据可保存')
        return False

    # 构建 DataFrame: 绝对时间戳 → 相对开始的秒数
    df_out = pd.DataFrame(strain_map, columns=[f'{p:.2f}mm' for p in positions])
    df_out.insert(0, 'timestamp', to_relative_seconds(timestamps))

    # 保存 CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  [保存] {output_path}')
    print(f'    形状: {df_out.shape}, 大小: {os.path.getsize(output_path) / 1e6:.2f} MB')
    return True


def process_luna_specimen(specimen_name, folder_path):
    """处理单个试件的所有 LUNA 文件，合并为一个 CSV。

    同一试件可能有多个 LUNA 文件 (如 L1-04-1.txt, L1-04-2.txt)，
    它们代表同一试件全流程的不同时间段，空间位置 (列) 一致，
    因此按时间顺序合并为一个 CSV 文件。
    """
    luna_dir = os.path.join(folder_path, 'LUNA')
    if not os.path.exists(luna_dir):
        print(f'  [跳过] LUNA 目录不存在: {luna_dir}')
        return

    txt_files = sorted([f for f in os.listdir(luna_dir)
                        if f.endswith('.txt') and f != 'LUNASetup.txt'])

    if not txt_files:
        print(f'  [跳过] 无 LUNA 数据文件')
        return

    # 合并所有文件的 DataFrame
    frames = []
    positions_ref = None
    for txt_file in txt_files:
        filepath = os.path.join(luna_dir, txt_file)
        timestamps, positions, strain_map = load_luna(filepath)

        if timestamps is None:
            continue

        # 检查空间位置是否一致
        if positions_ref is None:
            positions_ref = positions
        elif not np.allclose(positions_ref, positions):
            print(f'  [警告] {txt_file} 的空间位置与首个文件不一致，跳过该文件')
            continue

        df = pd.DataFrame(strain_map, columns=[f'{p:.2f}mm' for p in positions])
        df.insert(0, 'timestamp', timestamps)
        frames.append(df)

    if not frames:
        print(f'  [跳过] 无有效的 LUNA 数据')
        return

    # 合并、按时间排序、去重
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    # 输出: {specimen}分布式应变.csv (放在试件根目录)
    output_path = os.path.join(folder_path, f'{specimen_name}分布式应变.csv')
    save_luna_to_csv(df_all['timestamp'].values, positions_ref, df_all.iloc[:, 1:].values, output_path)


# ============================================================
# 2. FBG → CSV
# ============================================================

def load_fbg_txt(filepath):
    """
    加载 Micron Optics sm130 导出的 .txt FBG 数据文件。

    参数:
        filepath: .txt 文件路径

    返回:
        timestamps: datetime 数组
        fbg_data: FBG 应变矩阵 (n_samples × 20), 单位 με
        column_names: 传感器名称列表
    """
    if not os.path.exists(filepath):
        print(f'  [警告] 文件不存在: {filepath}')
        return None, None, None

    print(f'  [FBG] 加载: {os.path.basename(filepath)}')

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    # 自动查找列名行 (包含 'Timestamp' 的行)
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('Timestamp\t'):
            header_idx = i
            break

    if header_idx is None:
        print('    [错误] 未找到列名行 (Timestamp)')
        return None, None, None

    # 解析列名
    header_line = lines[header_idx].strip()
    column_names = header_line.split('\t')
    n_channels = len(column_names) - 1

    # 解析数据
    data_lines = lines[header_idx + 1:]
    timestamps = []
    fbg_values = []

    for line in data_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != n_channels + 1:
            continue

        # 解析时间戳 (荷兰语格式: dd-MM-yyyy HH:mm:ss.ffffff)
        try:
            ts = datetime.strptime(parts[0], '%d-%m-%Y %H:%M:%S.%f')
        except ValueError:
            try:
                ts = datetime.strptime(parts[0], '%d-%m-%Y %H:%M:%S')
            except ValueError:
                continue

        # 解析应变值 (十进制逗号 -> 点)
        try:
            values = [float(v.replace(',', '.')) for v in parts[1:]]
        except ValueError:
            continue

        timestamps.append(ts)
        fbg_values.append(values)

    if not fbg_values:
        print('    [错误] 无法解析任何数据行')
        return None, None, None

    timestamps = np.array(timestamps)
    fbg_data = np.array(fbg_values, dtype=float)

    print(f'    数据行数: {len(timestamps)}')
    print(f'    传感器: {column_names[1:]}')

    return timestamps, fbg_data, column_names[1:]


def save_fbg_to_csv(timestamps, fbg_data, column_names, output_path):
    """
    将 FBG 数据保存为 CSV 文件。

    CSV 格式:
      - 第1列: timestamp (相对开始的秒数, 本文件首条=0)
      - 第2~21列: 20个 FBG 传感器的应变值 (με)
    """
    if timestamps is None or fbg_data is None:
        print(f'  [跳过] 无数据可保存')
        return False

    # 构建 DataFrame: 绝对时间戳 → 相对开始的秒数
    df_out = pd.DataFrame(fbg_data, columns=column_names)
    df_out.insert(0, 'timestamp', to_relative_seconds(timestamps))

    # 保存 CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  [保存] {output_path}')
    print(f'    形状: {df_out.shape}, 大小: {os.path.getsize(output_path) / 1e6:.2f} MB')
    return True


def process_fbg_specimen(specimen_name, folder_path):
    """处理单个试件的所有 FBG 文件，合并为一个 CSV。

    同一试件可能有多个 Sensors.*.txt 文件，它们代表同一试件
    全流程的不同时间段 (可能时间有重叠)，传感器列一致，
    因此合并并按时间戳去重后输出为一个 CSV 文件。
    """
    fbg_dir = os.path.join(folder_path, 'FBG')
    if not os.path.exists(fbg_dir):
        print(f'  [跳过] FBG 目录不存在: {fbg_dir}')
        return

    # 查找 Sensors.*.txt 文件
    txt_files = sorted([f for f in os.listdir(fbg_dir)
                        if f.startswith('Sensors.') and f.endswith('.txt')])

    if not txt_files:
        print(f'  [跳过] 无 FBG 数据文件')
        return

    # 合并所有文件的 DataFrame
    frames = []
    column_names_ref = None
    for fbg_file in txt_files:
        filepath = os.path.join(fbg_dir, fbg_file)
        timestamps, fbg_data, column_names = load_fbg_txt(filepath)

        if timestamps is None:
            continue

        # 检查传感器列是否一致
        if column_names_ref is None:
            column_names_ref = column_names
        elif column_names != column_names_ref:
            print(f'  [警告] {fbg_file} 的传感器列与首个文件不一致，跳过该文件')
            continue

        df = pd.DataFrame(fbg_data, columns=column_names)
        df.insert(0, 'timestamp', pd.to_datetime(timestamps))
        frames.append(df)

    if not frames:
        print(f'  [跳过] 无有效的 FBG 数据')
        return

    # 合并、按时间排序、去重
    df_all = pd.concat(frames, ignore_index=True)
    df_all = df_all.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    # 输出: {specimen}光纤.csv (放在试件根目录)
    output_path = os.path.join(folder_path, f'{specimen_name}光纤.csv')
    save_fbg_to_csv(df_all['timestamp'].values, df_all.iloc[:, 1:].values,
                    column_names_ref, output_path)


# ============================================================
# 3. AE → CSV
# ============================================================

def load_ae_vallenae(filepath_pridb):
    """
    使用 vallenae 加载 AE 数据，并将振幅转换为 dB 单位。

    参数:
        filepath_pridb: .pridb 文件路径

    返回:
        df_ae: DataFrame 包含 AE hit 参数 (amplitude 已转换为 dB)
    """
    if not os.path.exists(filepath_pridb):
        print(f'  [警告] 文件不存在: {filepath_pridb}')
        return None

    print(f'  [AE] 加载: {os.path.basename(filepath_pridb)}')

    db = vae.io.PriDatabase(filepath_pridb)
    df_ae = db.read_hits()

    # 振幅转换: V → dB (20 * log10(V / 1µV))
    if 'amplitude' in df_ae.columns:
        V_ref = 1e-6
        df_ae['amplitude'] = 20 * np.log10(df_ae['amplitude'] / V_ref)
        if 'threshold' in df_ae.columns:
            df_ae['threshold'] = 20 * np.log10(df_ae['threshold'] / V_ref)

    print(f'    AE hits: {len(df_ae)}')
    if len(df_ae) > 0:
        print(f'    通道: {sorted(df_ae["channel"].unique())}')
        print(f'    时间范围: {df_ae["time"].min():.1f} ~ {df_ae["time"].max():.1f} s')

    db.close()
    return df_ae


def save_ae_to_csv(df_ae, output_path):
    """
    将 AE hit 参数保存为 CSV 文件。

    CSV 格式:
      - 包含所有 vallenae 返回的 hit 参数字段
      - amplitude 和 threshold 已转换为 dB 单位
    """
    if df_ae is None or len(df_ae) == 0:
        print(f'  [跳过] 无 AE 数据可保存')
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_ae.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'  [保存] {output_path}')
    print(f'    形状: {df_ae.shape}, 大小: {os.path.getsize(output_path) / 1e6:.2f} MB')
    return True


def process_ae_specimen(specimen_name, folder_path):
    """处理单个试件的所有 AE 文件，合并为一个 CSV。

    ⚠️ **多段 `.pridb` 的会话缝合**：`.pridb` 的 `time` 是**各次采集自己的起算秒数**
    （采集中断重开会从 ~0 重算），不是墙钟。旧实现 `concat` → `drop_duplicates('time')`
    → `sort_values('time')` 假设两段"时间有重叠⇒有重复"，但实测 L1-04 两段事件
    **零重合**（是两段独立会话），去重无效、排序交错 ⇒ AE 只覆盖 cycle 61.3%、
    且后期高活跃段被错放到 cycle 110k–161k（与论文"前 240k cycles 几乎无 AE"矛盾）。
    现交由 `ae_io.plan()` 判定「按 Time 合并」还是「按会话顺序缝合」，见
    `ae_io` 模块 docstring 与 docs/details.md §17.11。
    """
    ae_dir = os.path.join(folder_path, 'AE')
    if not os.path.exists(ae_dir):
        print(f'  [跳过] AE 目录不存在: {ae_dir}')
        return

    import sys as _sys
    _h = os.path.dirname(os.path.abspath(__file__))
    if _h not in _sys.path:
        _sys.path.insert(0, _h)
    import ae_io

    df_all, pl = ae_io.read_hits_vallenae(specimen_name)
    if df_all is None or len(df_all) == 0:
        print(f'  [跳过] 无有效的 AE 数据')
        return
    print(f'    mode={pl["mode"]}  hits={len(df_all):,}  '
          f'time={df_all["time"].min():.1f}~{df_all["time"].max():.1f} s  '
          f'通道: {sorted(df_all["channel"].unique())}')

    # 振幅转换: V → dB (20 * log10(V / 1µV))
    V_ref = 1e-6
    if 'amplitude' in df_all.columns:
        df_all['amplitude'] = 20 * np.log10(df_all['amplitude'] / V_ref)
    if 'threshold' in df_all.columns:
        df_all['threshold'] = 20 * np.log10(df_all['threshold'] / V_ref)

    # 输出: {specimen}声发射.csv (放在试件根目录)
    output_path = os.path.join(folder_path, f'{specimen_name}声发射.csv')
    save_ae_to_csv(df_all, output_path)


# ============================================================
# 4. 主处理流程
# ============================================================

def print_header(title):
    """打印格式化的章节标题。"""
    print()
    print('=' * 60)
    print(title)
    print('=' * 60)
    print()


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Step0: 原始 → CSV')
    ap.add_argument('--only', default='all',
                    choices=['all', 'ae', 'luna', 'fbg'],
                    help='只处理某一类数据（默认全部）。'
                         '重生成 {gid}声发射.csv 用 --only ae')
    a = ap.parse_args()

    print('=' * 60)
    print('Step 0: 数据预处理 — 原始数据 → CSV')
    print('=' * 60)
    print(f'数据根目录: {os.path.abspath(DATA_ROOT)}')
    print(f'处理范围: {a.only}')
    print()

    # ============================================================
    # 处理 1st TU Delft Campaign 试件 (L1-03, L1-04, L1-05)
    # ============================================================
    print_header('1st TU Delft Campaign (Paper 1: Broer et al. 2022)')

    for specimen in SPECIMENS_TUDELFT:
        print_header(f'--- 试件: {specimen} ---')

        # LUNA DFOS
        if a.only in ('all', 'luna'):
            print('[LUNA DFOS]')
            process_luna_specimen(specimen, specimen)

        # FBG
        if a.only in ('all', 'fbg'):
            print('[FBG]')
            process_fbg_specimen(specimen, specimen)

        # AE
        if a.only in ('all', 'ae'):
            print('[AE]')
            process_ae_specimen(specimen, specimen)

    # ============================================================
    # 处理摘要
    # ============================================================
    print_header('处理摘要')

    total_csv_size = 0
    csv_count = 0

    for specimen in SPECIMENS_TUDELFT:
        for suffix in ['声发射.csv', '光纤.csv', '分布式应变.csv']:
            fp = os.path.join(specimen, f'{specimen}{suffix}')
            if os.path.exists(fp):
                size_mb = os.path.getsize(fp) / 1e6
                total_csv_size += size_mb
                csv_count += 1
                print(f'  {fp}  ({size_mb:.2f} MB)')

    print()
    print(f'  共生成 {csv_count} 个 CSV 文件')
    print(f'  总大小: {total_csv_size:.2f} MB')
    print()
    print('=' * 60)
    print('数据预处理完成!')
    print('=' * 60)


if __name__ == '__main__':
    main()
