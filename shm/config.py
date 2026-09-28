#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局配置（精简）：连续损伤度 D(t) 体系的必要常量

四阶段划分(stage_divider)体系已弃用(2026-09-06)，相关配置一并移除。
保留被 shm 模块实际引用的：BASE_DIR / CHUNK_SIZE / EXT_BLOCK_PTS / DEFAULT_GROUPS。
"""
import os

# 项目根 = shm 的上级；主样本数据/产物在 <项目根>\main\ 下（2026-09-10 结构重组）
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(_PROJ, 'main')

# 主样本组（5 组，正式口径）
DEFAULT_GROUPS = ['016', '017', '018', '019', '020']

CHUNK_SIZE = 1000            # 逐块数据读取大小(点)

EXT_BLOCK_PTS = 500          # 损伤度 D 分块点数（≈每块结算一次块级统计）

# 声发射为主源，其余为辅助源（漏组可选参与融合）
PRIMARY_SOURCE = 'ae'
AUX_SOURCES = ('strain', 'fo')

# 光纤证据 e_fo 默认参数（仅当 sources 含 'fo' 时生效）
#   e_fo = clip((跨通道去共模极差 / 运行中位 - 1) / fo_ratio_gain, 0, 1) × fo_w
FO_PARAMS = {'fo_w': 0.6, 'fo_ratio_gain': 1.0, 'fo_min_blocks': 20}

# ============ 级别层异源分级（逐组闸门） ============
# 设计: D/L1 仍由声发射单源决定（5/5 精准, 误差 3.0%）；
#       L2/L3 额外要求“刚度损失率”达标 —— 从而把“检测”与“退化确认”在时间上分开。
#   刚度损失率 = |块均值| 相对校准段 [stiff_cal_lo, stiff_cal_hi) 低分位的相对增长
#   （载荷控制疲劳下 应变幅值 ∝ 1/刚度，故该相对增长即刚度损失率）。
# 依据 main/grade_compare.py 实测（results/grade_*.csv）:
#   016 应变通道可自动判别为"已解调", 刚度损失率 0.10@59% / 0.50@87%, 闸门有效;
#   018 修正幅值口径(块 std)后刚度损失 0.10 要到 95.7% 才达 → **闸门有害, 关闭**;
#   017 前 57% 寿命幅值单调降(硬化)、019 刚度信号早于 AE、020 幅值单调降 → 均关闭。
# 注: 分级为**可选扩展**; 项目主线交付是"损伤预警"(见 README §4.1)。
GRADE_GATE = {
    '016': {'strain_mode': 'stiff', 'lvl2_stiff': 0.10, 'lvl3_stiff': 0.50},
    '017': {},
    '018': {},
    '019': {},
    '020': {},
}
