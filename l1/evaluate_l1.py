# -*- coding: utf-8 -*-
"""第一步验证：公开数据集 L1-03/04/05 (Broer et al.2022 / ReMAP TU-Delft)
AE(事件级) + FBG(光纤应变)

关键方法学结论(2026-09-09, 与用户确认):
  L1 = 冲击后疲劳(impact-induced)工况, 与 016-020 服役期渐进损伤形态不同:
  - 早期(冲击后)即有大量 AE 扩展 → "先建健康基线再判损伤型"的原方法不适用
  - 块级(5000cycle)诊断发现: FBG 应变在健康段平稳、损伤后持续单调偏离(方向随试件,
    如 L1-03 变负/晚期加速, L1-04 缓慢变正, L1-05 末端变负) → 是单调可靠损伤指标
  - AE: 早期冲击扩展能量高, 后期需"损伤型/脱粘事件"筛选才单调
→ 新方法主体: FBG 应变相对自身健康基线【绝对偏离累计】; AE 作辅助(扩展期活跃)

本脚本 = 第一步"单调性+预警时机"验证:
  1. 分块(5000cycle): 算每块 FBG 应变均值 → 健康基线(前 ~25%)→ 偏离信号 dev[k]
  2. HI[k] = 应变绝对偏离的 EWMA/单调累积, 归一化到 [0,1] (首过阈给预警 cycle)
  3. 输出: HI 曲线 + 达 0.25/0.5/0.7 的 cycle + 与论文检测周期对齐 + 尾段单调性

注: LUNA 分布式应变 = 第二步融合, 本脚本不接入。
用法: python evaluate_l1.py [--groups ...] [--mode block|hi|plot]
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根(含 shm)
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # l1/

ROOT = os.path.dirname(os.path.abspath(__file__))   # <项目根>\l1
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, ROOT)                            # 使 l1_meta 可导入
from l1_meta import load_meta                       # noqa: E402
GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09']
META = {g: load_meta(g) for g in GROUPS}
GAP_S = 300.0        # FBG 测量块间隔阈值
CYCS_PER_BLK = 5000


def fbg_strain_cols(df):
    """全部 FBG 应变通道(表头自适应): fbg* 与 b*; 排除 FBG_A*/FBG_B*(波长列)。"""
    return [c for c in df.columns
            if (c.startswith('fbg') and c[3:].isdigit())
            or (c.startswith('b') and c[1:].isdigit())]


def load_fbg(gid):
    """读 FBG; 应变 = **全部 FBG 应变通道均值**(自适应从表头取)。
    返回 (t, strain_series)。"""
    fp = os.path.join(ROOT, gid, f'{gid}光纤.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig')
    t = df['timestamp'].to_numpy(float)
    s = df[fbg_strain_cols(df)].mean(axis=1, skipna=True).to_numpy(float)
    return t, s


def fbg_blocks(gid):
    """返回 (cyc, 块应变均值, mid_t)。步长 ~5000 cycle(FBG 停机测量块)。"""
    t, s = load_fbg(gid)
    nf = META[gid]['n_f']
    gap = np.where(np.diff(t) > GAP_S)[0]
    bounds = np.concatenate([[0], gap + 1, [len(t)]])
    mid_t, mean_s = [], []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 500:
            continue
        mid_t.append(0.5 * (t[a] + t[b - 1]))
        mean_s.append(float(np.nanmean(s[a:b])))
    mid_t = np.array(mid_t)
    mean_s = np.array(mean_s)
    nblk = len(mean_s)
    cyc = (np.arange(nblk) + 0.5) / nblk * nf
    return cyc, mean_s, mid_t


def load_ae_per_block(gid, cyc, mid_t):
    """AE 事件按块聚合: 能量和 / 大事件数(≥85dB)。返回与 cyc 等长。"""
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig', usecols=['time', 'energy', 'amplitude'])
    ta = df['time'].to_numpy(float)
    ea = df['energy'].to_numpy(float)
    aa = df['amplitude'].to_numpy(float)
    nf = META[gid]['n_f']
    nblk = len(cyc)
    # AE time → cycle: 用块中心时间线性映射 (超出边界外推)
    cyc_ae = np.interp(ta, mid_t, cyc, left=-1e9, right=nf * 2)
    e_sum = np.zeros(nblk)
    big = np.zeros(nblk)
    half = CYCS_PER_BLK / 2.0
    for i in range(nblk):
        lo, hi = cyc[i] - half, cyc[i] + half
        m = (cyc_ae >= lo) & (cyc_ae < hi)
        e_sum[i] = ea[m].sum()
        big[i] = np.sum(aa[m] >= 85)
    return e_sum, big


def hi_from_strain(mean_s, base_frac=0.25, alpha=0.5, drop=0.15):
    """应变绝对偏离 → HI[0,1]。
    方案A(2026-09-12): base = **首个应变测量**(论文 Level-4 "相对首测"口径);
    dev = |strain - base| (µε); HI: 升快(追 dev 归一)降慢(记忆)。
    归一尺度: 全窗 90 分位 dev (相对尺度), 得 0~1 单调近似。
    """
    mean_s = np.asarray(mean_s, float)
    n = len(mean_s)
    base = float(mean_s[0])          # 方案A: 基准 = 首个应变测量
    dev = np.abs(mean_s - base)
    scale = float(np.percentile(dev, 90)) if np.percentile(dev, 90) > 1e-9 else 1.0
    hi = np.zeros(n)
    for k in range(n):
        target = min(dev[k] / scale, 1.0)
        if target > hi[k - 1] if k > 0 else True:
            hi[k] = hi[k - 1] + alpha * (target - hi[k - 1]) if k > 0 else target
        else:
            hi[k] = hi[k - 1] * (1.0 - drop) if k > 0 else target
    hi = np.clip(hi, 0, 1)
    return hi, base, dev, scale


def analyze(gid):
    cyc, mean_s, mid_t = fbg_blocks(gid)
    nf = META[gid]['n_f']
    hi, base, dev, scale = hi_from_strain(mean_s)
    e_sum, big = load_ae_per_block(gid, cyc, mid_t)
    # 单调性(尾 30% 块上升占比)
    tail = hi[int(len(hi) * 0.7):]
    mono = float(np.mean(np.diff(tail) >= 0)) * 100 if len(tail) > 2 else np.nan

    def first_th(th):
        idx = np.where(hi >= th)[0]
        return float(cyc[idx[0]]) if len(idx) else np.nan

    row = dict(gid=gid, n_f=nf, n_blk=len(hi), base=round(base, 1),
               strain_first=round(float(mean_s[0]), 1), strain_last=round(float(mean_s[-1]), 1),
               dev_scale=round(scale, 1),
               HI_end=round(float(hi[-1]), 3), tail_mono=round(mono, 1),
               warn25=round(first_th(.25), 1) if not np.isnan(first_th(.25)) else np.nan,
               warn50=round(first_th(.5), 1) if not np.isnan(first_th(.5)) else np.nan,
               warn70=round(first_th(.7), 1) if not np.isnan(first_th(.7)) else np.nan)
    print(f'\n[{gid}] 块={len(hi)} 基线应变={row["base"]} 末应变={row["strain_last"]} '
          f'(漂移{row["strain_last"]-row["base"]:+.0f}µε)')
    print(f'  HI_end={row["HI_end"]} 尾段单调上升占比={row["tail_mono"]}%')
    print(f'  HI 达 0.25@{row["warn25"]}  0.50@{row["warn50"]}  0.70@{row["warn70"]} cycle '
          f'(n_f={nf})')
    print('  论文参考:')
    for lab, c in META[gid]['refs']:
        print(f'    {lab}: {c}')
    # 存
    np.savez_compressed(os.path.join(RES, f'_l1_hi_{gid}.npz'),
                        cyc=cyc, strain=mean_s, hi=hi, dev=dev,
                        ae_energy=e_sum, ae_big=big, nf=nf)
    return row


def plot(gid):
    z = np.load(os.path.join(RES, f'_l1_hi_{gid}.npz'))
    cyc, strain, hi, e_sum, big = z['cyc'], z['strain'], z['hi'], z['ae_energy'], z['ae_big']
    nf = float(z['nf'])
    x = cyc / 1000
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, strain, 'o-', color='tab:green', lw=1.2)
    axes[0].set_ylabel('FBG 应变(µε)')
    axes[0].set_title(f'{gid}  应变漂移 + HI (虚线=论文检测, 黑虚线=失效 n_f={nf/1000:.0f}k)')
    axes[1].plot(x, hi, 'o-', color='k', lw=1.3)
    for th, c in [(0.25, '#e07b00'), (0.5, '#d62728'), (0.7, '#8b0000')]:
        axes[1].axhline(th, color=c, lw=0.7, ls='--', alpha=0.6)
    axes[1].set_ylabel('HI (应变偏离归一)')
    axes[1].set_ylim(-0.02, 1.02)
    axes[2].bar(x, big, width=3.5, color='tab:red')
    axes[2].set_ylabel('AE 大事件/块')
    axes[2].set_xlabel('cycle (k)')
    for ax in axes:
        for lab, c in META[gid]['refs']:
            if c <= nf * 1.1:
                ax.axvline(c / 1000, color='grey', ls=':', lw=0.8)
        ax.axvline(nf / 1000, color='k', ls='--', lw=1)
    fig.tight_layout()
    out = os.path.join(FIG, f'l1_hi_{gid}.png')
    fig.savefig(out, dpi=120)
    print('已存:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(META))
    ap.add_argument('--mode', default='hi', choices=['hi', 'plot'])
    a = ap.parse_args()
    groups = a.groups.split(',')
    if a.mode == 'hi':
        rows = [analyze(g) for g in groups]
        pd.DataFrame(rows).to_csv(os.path.join(RES, 'l1_hi_metrics.csv'),
                                  index=False, encoding='utf-8-sig')
        for g in groups:
            plot(g)
    elif a.mode == 'plot':
        for g in groups:
            plot(g)


if __name__ == '__main__':
    main()
