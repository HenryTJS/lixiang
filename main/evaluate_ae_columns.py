# -*- coding: utf-8 -*-
"""AE 多列证据评估（形状/比值列 · 单源消融口径）

背景
----
损伤度 D 的 AE 证据以往**只用 `ae_Peak` 一列**（25 个 AE 列里另外 24 列从未参与，
仅存在于 `aligned/*.csv`）。逐列证据筛查（2026-09-20）显示：
  - **无量纲比值列**（MarginFactor=Peak/RootMAV、PeakFactor=Peak/RMS、
    ImpulseFactor=Peak/MAV）在 9/9 组（5 主样本 + 023~026）上趋势为正；
  - **Kurtosis / Skewness** 尾部抬升显著（Kurtosis 在 025 上 1.7~3.1 倍）；
  - 原始量级列（RMS/Std/Variance/MeanSquare/MAV/RootMAV）趋势**反向**（ρ≈-0.11~-0.22），
    频域 6 列无一致趋势 → 均不采用。
故新增可选证据 `e_shape`（见 `shm/damage_index.py`），本脚本量化其效果与代价。

口径
----
- **采样率恒为 10 Hz**（全组，数据方口径）；注意部分组的原始「时间列」是**整数计数器**而非秒
  （见 `main/candidate_check.timebase()`）——**不要从该列反推采样率**。
  `EXT_BLOCK_PTS=500` = 50 s = 250 个 5 Hz 载荷循环，组间一致。
- **AE 单源**（`abl='no_strain'`）：risk = e_ae，隔离形状证据本身的作用；
- 块内事件特征取 `shape_q` 分位（默认 95，即"块内最显著事件"），
  以固定校准段 [shape_cal_lo, shape_cal_hi) 块值的 `shape_base_pct` 分位为基线，
  `e_shape = clip((q/base - 1)/shape_gain, 0, 1) × shape_w`（纯因果，零未来信息）；
- **权重约定**：`shape_w` 默认 0.6，与既有辅助证据 `estrain_w=0.6` 同约定。
  实测 w=1.0 时 e_shape 单独即可把 risk 顶满 1.0 → 9 组 D_end 全部饱和≈1.0，
  跨试件区分度归零；w=0.6 则 025 由 D_end 0.202→0.824 而其余组基本不动。
- 只有 016~020 有 b2（离线参考锚），023~026 **无 b2**，故后者只按**形态**判读
  （D_end 量级、t_warn 是否存在/是否过早、尾部单调性），不计分级。

用法
----
    python evaluate_ae_columns.py                      # 默认：Kurtosis/MarginFactor × q{95,50} × w{1.0,0.6}
    python evaluate_ae_columns.py --cols Kurtosis --shape-w 0.6 --shape-q 95
    python evaluate_ae_columns.py --cols Peak,Kurtosis,MarginFactor,Skewness
    python evaluate_ae_columns.py --groups 021,022,023,024,025,026,027 \
        --cols PeakFactor,Kurtosis,MarginFactor --shape-w 0.6 \
        --out main/results/ae_column_eval_ext.csv

⚠️ PowerShell 会把未加引号的 `021,022` 当**数字**传递 → 前导零被吃掉（gid 变 int，
   pivot 匹配不上）。本脚本已对 `--groups` 做 `zfill(3)` 兜底；传参时仍建议加引号。

输出：results/ae_column_eval.csv（长表 cfg×gid）+ 控制台三张 pivot 表。
注意：023/024/025 数据中**无光纤文件**（与本脚本无关，AE 单源口径不依赖光纤）。
"""
import os
import sys
import io
import argparse
import contextlib

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CWD0 = os.getcwd()                                # 调用时的目录(脚本内会 chdir)
sys.path.insert(0, os.path.dirname(ROOT))          # 项目根(含 shm)
sys.path.insert(0, ROOT)                           # main/(含 eval_common)
os.chdir(ROOT)                                     # main/

from shm.streaming import ChunkedDataReader        # noqa: E402
from shm.damage_index import OnlineDamageIndex     # noqa: E402
from eval_common import per_group_metrics          # noqa: E402

DEF_GROUPS = ['016', '017', '018', '019', '020', '023', '024', '025', '026']
DEF_COLS = ['Kurtosis', 'MarginFactor']
RES = os.path.join(ROOT, 'results')


def load_arrays(gid):
    """一次取出逐点 (strain, peak, {列: 值})，语义与 StreamSimulator 一致。

    strain 为松散对齐下的"保持上值"(ffill)；peak/形状值仅在**有新事件**的行给出，
    其余为 NaN → 与在线逐点 update 的入参完全一致。
    """
    rd = ChunkedDataReader(gid)
    with contextlib.redirect_stdout(io.StringIO()):
        rd.load_and_prepare()
    data, ci, n = rd._data, rd._col_idx, rd._data.shape[0]
    ae = rd.ae_cols
    if 'strain' in ci:
        st = pd.Series(data[:, ci['strain']]).ffill().to_numpy()
    else:
        st = np.full(n, np.nan)
    mask = np.isfinite(data[:, [ci[c] for c in ae]]).any(axis=1)   # 有 AE 事件的行
    pk = np.full(n, np.nan)
    pv = data[:, ci['ae_Peak']] if 'ae_Peak' in ci else np.zeros(n)
    pk[mask] = np.where(np.isfinite(pv[mask]), pv[mask], 0.0)
    cols = {}
    for c in ae:
        if c == 'ae_Peak':
            continue
        v = data[:, ci[c]]
        a = np.full(n, np.nan)
        a[mask] = np.where(np.isfinite(v[mask]), v[mask], np.nan)
        cols[c[len('ae_'):]] = a
    rd.cleanup()
    return st, pk, cols


def run_cfg(gid, col, abl, ov):
    """跑一个配置 → 逐点 D。col=None 即基线(仅 Peak)。"""
    st, pk, cols = load_arrays(gid)
    p = {'abl': abl}
    p.update(ov)
    if col:
        p['shape_col'] = col
    di = OnlineDamageIndex(p)
    arr = cols.get(col) if col else None
    n = len(st)
    d = np.empty(n, dtype=np.float32)
    for i in range(n):
        a = pk[i]
        b = arr[i] if arr is not None else None
        if b is not None and not np.isfinite(b):
            b = None
        d[i] = di.update(st[i], None if not np.isfinite(a) else float(a), None, b)
    return d, float(di._last_e_shape)


def main():
    ap = argparse.ArgumentParser(description='AE 多列(形状/比值)证据评估')
    ap.add_argument('--groups', default=','.join(DEF_GROUPS))
    ap.add_argument('--cols', default=','.join(DEF_COLS))
    ap.add_argument('--shape-q', default='95,50', help='块内事件分位(逗号分隔)')
    ap.add_argument('--shape-w', default='1.0,0.6', help='辅助证据权重(逗号分隔)')
    ap.add_argument('--abl', default='no_strain', help='消融模式; no_strain=AE 单源')
    ap.add_argument('--out', default=os.path.join(RES, 'ae_column_eval.csv'),
                    help='输出 CSV；相对路径按**调用时**目录解析(脚本内会 chdir 到 main/)')
    a = ap.parse_args()
    if not os.path.isabs(a.out):                   # 相对路径 → 按调用目录
        a.out = os.path.join(CWD0, a.out)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    groups = [g.strip().zfill(3) for g in a.groups.split(',') if g.strip()]
    cols = [c.strip() for c in a.cols.split(',') if c.strip()]
    qs = [float(x) for x in a.shape_q.split(',') if x.strip()]
    ws = [float(x) for x in a.shape_w.split(',') if x.strip()]
    cfgs = [('base', None, {})]
    for c in cols:
        for q in qs:
            for w in ws:
                cfgs.append((f'{c[:2]}q{q:g}w{w:g}', c,
                             {'shape_q': q, 'shape_w': w}))

    print(f'组 {len(groups)} 个 × 配置 {len(cfgs)} 个 (abl={a.abl})')
    rows = []
    for gid in groups:
        for name, col, ov in cfgs:
            d, esh_end = run_cfg(gid, col, a.abl, ov)
            m = per_group_metrics(gid, d)
            rows.append(dict(cfg=name, col=(col or '-'), shape_q=ov.get('shape_q', np.nan),
                             shape_w=ov.get('shape_w', np.nan), gid=gid,
                             D_end=m['D_end'], t25=m['t25'], t55=m['t55'],
                             t85=m['t85'], t_warn=m['t_warn'],
                             tail_mono=m['tail_mono'], err=m['err'],
                             grade=m['grade'], esh_end=esh_end))
        print(f'  {gid} 完成 ({len(cfgs)} 配置)')
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False, float_format='%.4f')

    order = [c[0] for c in cfgs]
    for key in ['t85', 't_warn', 'D_end']:
        print(f'\n===== {key} =====')
        pv = df.pivot(index='gid', columns='cfg', values=key).reindex(
            index=groups, columns=order)
        print(pv.round(3).to_string())
    print('\n===== 主样本 5 组 A-预警分级(仅 016~020 有 b2, 离线锚) =====')
    sub = df[df['gid'].isin(['016', '017', '018', '019', '020'])]
    pv = sub.pivot(index='cfg', columns='gid', values='grade').reindex(order)
    pv['全部A'] = (pv == 'A').all(axis=1)
    print(pv.to_string())
    print(f'\nsaved {os.path.abspath(a.out)}')


if __name__ == '__main__':
    main()
