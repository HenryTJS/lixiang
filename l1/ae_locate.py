# -*- coding: utf-8 -*-
"""AE 事件定位：TDOA + 各向异性走时反演（L1 加筋条板）。

为什么必须用各向异性模型
------------------------
PDF 明确给出两个方向的声速，差异达 60%：
    沿加筋条纵向 (along stiffener)   vy = 6426~6729 m/s
    横向穿过加筋条 (through stiffener) vx = 2933~4289 m/s（多数 ~4064）
用单一声速会把定位结果系统性压向某一侧。

几何（13 组 PDF 实测一致）
--------------------------
传感器（skin-side, from bottom left, 单位 mm）::

    Y(长, 沿加筋条, 243mm)
    ^    S4(20,220)              S1(145,190)
    |
    |    S3(20,50)               S2(145,20)
    +-------------------------------> X(宽, 165mm)

走时模型: ``t(P->S) = hypot((xP-xS)/vx, (yP-yS)/vy)``

流程
----
1. 读 SetType=2 的 ``(Time, Chan, Amp)``；``Time/1e7`` → 秒
2. 单次扫描聚类：相邻 hit 时间差 ``<= WIN_US`` 归为一簇；同簇内每通道只取一次
3. 对 ``>=3`` 通道的簇做 TDOA 定位：以最早通道为参考消除事件绝对时刻，
   粗网格搜索 → 局部精化
4. 输出 ``results/_l1_loc_{gid}.npz``

用法
----
    python l1/ae_locate.py --groups L1-49 --max-events 30000
    python l1/ae_locate.py                      # 全部 GROUPS
"""
import argparse
import glob
import io
import os
import re
import sqlite3
import sys

import numpy as np

# ⚠️ Windows 控制台默认 GBK，输出含 µ/→ 等字符会 UnicodeEncodeError 直接崩。
# 与 ae_raf.py / step0_v2.py / ae_shape_features.py 保持同一写法。
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')

GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54',
          'L1-55', 'L1-56', 'L1-59', 'L1-60']

# --- 几何（13 组 PDF 一致） -------------------------------------------------
SENSORS = {1: (145.0, 190.0), 2: (145.0, 20.0),
           3: (20.0, 50.0), 4: (20.0, 220.0)}      # mm, skin-side
LX, LY = 165.0, 243.0                               # 宽(X) x 长(Y) [mm]
VX_DEF, VY_DEF = 4064.0, 6659.0                     # 缺省横向/纵向声速 [m/s]

TIME_BASE = 1e7                                     # .pridb Time 单位: 100 ns
SETTYPE_HIT = 2

# 聚类窗口：实测扫描（L1-49, min_ch=4）
#   15µs -> rms 3.25 (100% 可用) ; 20µs -> 3.86 (100%) ; 30µs -> 4.73 (94%)
#   40µs -> 12.21 (64%) ; 60µs -> 21.99 (33%)   ← 窗口过大把不同事件并簇，残差暴涨
# 取 20 µs：残差接近模型误差，且质心与冲击点最吻合。
WIN_US = 20.0
RMS_GOOD = 5.0                                      # 可信定位的残差上限 [µs]
# 参与 TDOA 的最少通道数。**必须 4**（= 论文“four AE sensors”）：
# min_ch=3 时 TDOA 在 2D 下是**恰定方程**（2 方程 2 未知）⇒ 残差 rms 恒 ≈0
# （实测 L1-51 rms 中位 0.16 µs、L1-52 0.18 µs），既无冗余度也无定位质量度量，
# RMS_GOOD 筛选失去意义。min_ch=4 才有 1 个自由度可算 rms。
# （本文件头的窗长扫描也是按 min_ch=4 标定的。）
MIN_CH = 4

# 冲击点参考（**仅 L1-49 已核实可用，其余组慎用**）。
#
# 2026-09-15 逐组核对 13 份 PDF + 实测定位质心（见 `l1/impact_truth_check.py`
# 与 `l1/results/l1_impact_truth.csv`），结论：
#   1. PDF 的冲击位置描述**逐组不同**（至少 5 类）：
#        right edge x top / right edge x bottom / left edge x top /
#        left edge x bottom / "skin near right, 2.5cm" / "centre, 8.25cm" …
#   2. 用**任何统一坐标约定**（含 180° 翻转）都无法让 PDF 描述与实测质心同时自洽；
#      且 PDF 的 left/right 标注与实测 X 质心**无相关性**。
#   3. 实测 X 质心 = 46,49,53,61,75,76,78,79,81 mm，呈**双峰**，
#      既非同一真值下的散射、也非 PDF 描述所能解释。
#   → 故 (50,80) 只对 L1-49 自洽（偏差 9 mm）；
#      **"距冲击点"不得当作 13 组统一的定位准确性指标**，
#      定位结论只能用组内**重复性**（σ），不能用它与真值的偏差。
# 原始文字：L1-49 "stiffener (right). 5 cm from right edge x 8 cm from top"
# 换算到 skin-side：两面视角翻转 → X=165-115=50, Y=80。
IMPACT = (50.0, 80.0)
IMPACT_VERIFIED_GROUPS = ('L1-49',)      # 仅该组偏离 <10 mm

# 各组声速（横向 vx, 纵向 vy；PDF 实测，缺失用缺省）
# 2026-09-15 补齐 L1-50（PDF 有值却漏登记，曾回落缺省导致 vy 偏 3%）。
# 数据集 B（L1-03/04/05/09）PDF 虽有值，但原文注明
#   "1 manual PZT recording performed after specimen failure, but considering
#    sensor failure, these measurements might not be useful"
# → **不可信，故意不登记**（且这 4 组不在 GROUPS 里，不影响本脚本）。
VEL = {
    'L1-49': (4064.0, 6426.0), 'L1-50': (4077.2, 6470.6),
    'L1-51': (4090.0, 6535.0), 'L1-52': (4289.0, 6729.0),
    'L1-54': (4049.0, 6529.0), 'L1-55': (2933.0, 6623.0),
    'L1-56': (4022.0, 6659.0), 'L1-59': (4000.0, 6729.0),
    'L1-60': (4116.0, 6550.0),
}


# --------------------------------------------------------------------------
def read_hits(gid):
    """读全部 AE 文件的 (Time[s], Chan, Amp)。

    ⚠️ **第 0 列已是「缝合后的秒数」**（不再是 100 ns tick）——多段 `.pridb`
    的会话缝合由 `ae_io` 统一处理（见 `ae_io` 模块 docstring / details §17.11）。
    """
    import ae_io
    a, _pl = ae_io.read_hits(gid, cols=('Time', 'Chan', 'Amp'))
    return a


def cluster_runs(t_s, win_s):
    """单次扫描：相邻时间差 > win_s 处切分，返回 (start, end) 索引数组。"""
    brk = np.nonzero(np.diff(t_s) > win_s)[0] + 1
    starts = np.concatenate(([0], brk))
    ends = np.concatenate((brk, [t_s.size]))
    return starts, ends


def make_tmap(gx, gy, vx, vy):
    """预计算走时表 -> (nch, ny, nx) [µs]。"""
    T = []
    for ch in sorted(SENSORS):
        sx, sy = SENSORS[ch]
        # 注意单位: mm/(m/s) = 1e-3 s , 故 ×1e3 得 µs
        d = np.hypot((gx[None, :] - sx) / vx, (gy[:, None] - sy) / vy)
        T.append(d * 1e3)
    return np.stack(T)


def locate(t_us, ch, tmap, gx, gy):
    """TDOA 最小二乘：以最早通道为参考。返回 (x, y, rms_us)。

    t_us: 该簇各通道到时 [µs]（已按时间升序）
    ch:   对应通道号（1-based）
    """
    k = len(ch)
    dt = t_us - t_us[0]
    pred = tmap[np.asarray(ch) - 1]              # (k, ny, nx)
    pred = pred - pred[0][None, :, :]
    res = pred - dt[:, None, None]
    cost = np.sqrt(np.mean(res * res, axis=0))   # (ny, nx) [µs]
    j = int(np.argmin(cost))
    iy, ix = np.unravel_index(j, (gy.size, gx.size))
    return float(gx[ix]), float(gy[iy]), float(cost[iy, ix])


def run(gid, win_us=WIN_US, step=2.0, min_ch=MIN_CH, max_events=None, seed=0):
    vx, vy = VEL.get(gid, (VX_DEF, VY_DEF))
    gx = np.arange(0.0, LX + 1e-9, step)
    gy = np.arange(0.0, LY + 1e-9, step)
    tmap = make_tmap(gx, gy, vx, vy)

    a = read_hits(gid)
    t_s = a[:, 0]                       # ae_io 已换算为缝合后的秒数
    chan = a[:, 1].astype(np.int32)
    amp = a[:, 2]
    print(f'  [{gid}] hits={t_s.size:,}  通道={sorted(set(chan.tolist()))}  '
          f'vx={vx:.0f} vy={vy:.0f} m/s  跨度={t_s[-1] - t_s[0]:.0f}s')

    starts, ends = cluster_runs(t_s, win_us * 1e-6)
    print(f'  簇数={starts.size:,}（窗口 {win_us:.0f} µs）')

    # 每簇：每通道取一次（保留该通道内最先到的 hit）
    # 向量化：以 (簇 id, 通道) 为键，取首个索引
    cid = np.repeat(np.arange(starts.size), ends - starts)
    order = np.lexsort((t_s, chan, cid))
    c_s, ch_s = cid[order], chan[order]
    first = np.ones(order.size, dtype=bool)
    first[1:] = (c_s[1:] != c_s[:-1]) | (ch_s[1:] != ch_s[:-1])
    sel = order[first]
    nch = np.bincount(cid[sel], minlength=starts.size)
    ok = np.nonzero(nch >= min_ch)[0]
    print(f'  >={min_ch} 通道的簇 = {ok.size:,}'
          f'（占 {100.0 * ok.size / max(starts.size, 1):.1f}%）')

    if max_events and ok.size > max_events:
        rng = np.random.default_rng(seed)
        ok = np.sort(rng.choice(ok, max_events, replace=False))
        print(f'  抽样定位 {max_events:,} 个事件')

    sel_c, sel_ch = cid[sel], chan[sel]
    pos = np.searchsorted(ok, sel_c)
    m = (pos < ok.size) & (ok[np.clip(pos, 0, ok.size - 1)] == sel_c)

    sel2 = sel[m]
    grp_c, grp_ch = sel_c[m], sel_ch[m]
    xs, ys, rms, cych, nchv = [], [], [], [], []
    bnd = np.nonzero(np.diff(grp_c))[0] + 1
    st = np.concatenate(([0], bnd))
    en = np.concatenate((bnd, [grp_c.size]))
    for i0, i1 in zip(st, en):
        tt = t_s[sel2[i0:i1]]
        cc = grp_ch[i0:i1]
        o = np.argsort(tt)
        # locate() 与 tmap 的单位都是 µs，这里必须由秒换算，否则 dt 会被 tmap 淹没
        x, y, r = locate(tt[o] * 1e6, cc[o], tmap, gx, gy)
        xs.append(x); ys.append(y); rms.append(r); nchv.append(len(cc))
        cych.append(float(tt[o][0]))     # 事件时刻，秒
    xs = np.asarray(xs); ys = np.asarray(ys)
    rms = np.asarray(rms); nchv = np.asarray(nchv); cych = np.asarray(cych)

    good = rms <= RMS_GOOD
    print(f'  定位 {xs.size:,} 个  rms中位={np.median(rms):.2f} µs  '
          f'rms<={RMS_GOOD:.0f}µs 的 {100.0 * good.mean():.1f}%'
          f'  通道数分布={dict(zip(*np.unique(nchv, return_counts=True)))}')
    if good.sum() > 10:
        d = np.hypot(xs[good] - IMPACT[0], ys[good] - IMPACT[1])
        print(f'    可信定位质心 = ({xs[good].mean():.0f}, {ys[good].mean():.0f}) mm  '
              f'σ = ({xs[good].std():.0f}, {ys[good].std():.0f}) mm')
        if gid in IMPACT_VERIFIED_GROUPS:
            print(f'    （与冲击点 {IMPACT[0]:.0f},{IMPACT[1]:.0f} 比对  '
                  f'距冲击点中位={np.median(d):.0f} mm — 该组真值已核实）')
        else:
            print(f'    ⚠️ 距冲击点中位={np.median(d):.0f} mm —— **该数字不可用**：'
                  f'本组 PDF 的位置描述与实测质心不自洽，'
                  f'真值未核实（见文件头 IMPACT 注释）')
        print(f'    x[{xs[good].min():.0f},{xs[good].max():.0f}] '
              f'y[{ys[good].min():.0f},{ys[good].max():.0f}]')

    out = os.path.join(RES, f'_l1_loc_{gid}.npz')
    np.savez_compressed(out, x=xs, y=ys, rms_us=rms, nch=nchv, t_s=cych,
                        vx=vx, vy=vy, win_us=win_us, step=step)
    print(f'  已存 {out}')
    return dict(x=xs, y=ys, rms=rms, nch=nchv, t=cych)


def plot_map(gid, rms_max=RMS_GOOD):
    """空间定位云：左=密度+传感器几何，右=按时间 5 等分事件数的质心迁移。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fp = os.path.join(RES, f'_l1_loc_{gid}.npz')
    if not os.path.exists(fp):
        print(f'  [skip] {gid}: 缺 {fp}')
        return
    z = np.load(fp)
    x, y, r, t = z['x'], z['y'], z['rms_us'], z['t_s']
    m = r <= rms_max
    if m.sum() < 20:
        print(f'  [skip] {gid}: 可信定位仅 {m.sum()} 个')
        return
    xs, ys, ts = x[m], y[m], t[m]

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.4))

    a0 = ax[0]
    a0.hist2d(xs, ys, bins=[22, 26], range=[[0, LX], [0, LY]],
              cmap='viridis', cmin=1)
    for ch, (sx, sy) in SENSORS.items():
        a0.plot(sx, sy, 'r^', ms=11, mec='k', zorder=5)
        a0.text(sx + 4, sy + 4, f'S{ch}', color='r', fontsize=9, weight='bold')
    a0.plot(*IMPACT, 'w*', ms=20, mec='k', mew=1.0, zorder=6)
    a0.text(IMPACT[0] + 5, IMPACT[1] - 13, 'impact (est.)', color='w', fontsize=8)
    a0.set_xlabel('X [mm]  (across stiffener)')
    a0.set_ylabel('Y [mm]  (along stiffener)')
    a0.set_title(f'{gid}  AE localization density (rms<={rms_max:.0f}us, n={m.sum():,})')
    a0.set_xlim(0, LX); a0.set_ylim(0, LY); a0.set_aspect('equal')

    a1 = ax[1]
    a1.plot(xs, ys, '.', ms=2, color='0.78', zorder=1)
    q = np.percentile(ts, [0, 20, 40, 60, 80, 100])
    cm = plt.cm.plasma(np.linspace(0.05, 0.95, 5))
    for i in range(5):
        mm = (ts >= q[i]) & (ts <= q[i + 1])
        if mm.sum() < 5:
            continue
        a1.plot(xs[mm], ys[mm], '.', ms=4, color=cm[i])
        a1.plot(xs[mm].mean(), ys[mm].mean(), 'o', ms=13, mfc='none',
                mec=cm[i], mew=2.2, zorder=5)
        a1.annotate(str(i + 1), (xs[mm].mean(), ys[mm].mean()),
                    color='k', fontsize=9, ha='center', va='center', zorder=6)
    for ch, (sx, sy) in SENSORS.items():
        a1.plot(sx, sy, 'r^', ms=11, mec='k', zorder=5)
    a1.plot(*IMPACT, 'k*', ms=20, zorder=6)
    a1.set_xlabel('X [mm]')
    a1.set_ylabel('Y [mm]')
    a1.set_title('centroid migration (1->5, time quintiles)')
    a1.set_xlim(0, LX); a1.set_ylim(0, LY); a1.set_aspect('equal')

    fig.tight_layout()
    out = os.path.join(FIG, f'l1_loc_{gid}.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  已存 {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS))
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--win', type=float, default=WIN_US, help='聚类窗口 [µs]')
    ap.add_argument('--step', type=float, default=2.0, help='网格步长 [mm]')
    ap.add_argument('--min-ch', type=int, default=MIN_CH)
    ap.add_argument('--max-events', type=int, default=None)
    a = ap.parse_args()
    for g in [s.strip() for s in a.groups.split(',')]:
        print(f'--- {g} ---')
        run(g, win_us=a.win, step=a.step, min_ch=a.min_ch,
            max_events=a.max_events)
        if a.plot:
            plot_map(g)


if __name__ == '__main__':
    main()
