# -*- coding: utf-8 -*-
"""L1 第二批试件（L1-49..L1-56，**无 FBG**）的连续损伤度 D(t)

与 evaluate_l1_degree.py 的差异（那套依赖 FBG 块应变）：
  - 应变证据改用 **DFOS 脚部应变**（ODiSi-B 左/右脚空间段均值），
    取自 step0_v2.py 的段级 `{gid}分布式应变.csv`
  - cycle 锚改用 **DFOS 测量段**（`l1_time_align.dfos_cycles`，按寿命均匀校准），
    因第二批 `段数×500/n_f` 实测 0.26~1.68（不恒为 1，PDF 亦注明 PA 测量会重置计数）
  - AE 事件按段聚合（段内峰值 sqrt(energy)），AE 时间经 markers 墙钟映射到 cycle

⚠️ 口径说明（2026-09-14 修订）：
  初版用段级**中位**分布，丢失了段内 valley/peak 行的区分。现已由 `step0_v2.py`
  额外输出 `{gid}分布式应变_peak.csv`（段内**压缩峰值载荷行** = 两脚应变均值最负的行），
  本脚本优先用它 → 应变证据与第一批 FBG 的「块级循环幅值」语义一致。

⚠️ AE 失效自适应（2026-09-16 新增）：
  L1-54/56/60 的 AE 属「整体高位且平稳」型，不存在显著超背景的突发事件
  （喂入引擎的 loge 中位数达 11.5，而损伤型判据要求 > 背景 p60 + lift(2.0)）
  → e_ae 恒 ≈ 0 → D 被应变辅证权重上限（estrain_w=0.6）锁死，D_end 仅 0.53~0.60。
  现启用 `OnlineDamageIndex(ae_auto=True)`：维护最近 20 个**已结算块**的 e_ae 窗口，
  若窗口内最大值仍 < 0.1，则判定 AE 源无信息，把应变权重提升至 1.0（应变升为主证）。
  纯因果（零未来信息）；窗口按**块数**而非寿命比例 → 总块数 < 20 的短寿命试件永不触发。
  实测：6 个 AE 正常组结果逐位不变，仅 L1-54/56/60 的 D_end 恢复 0.963/0.970/0.898，
  且预警时刻（t25/t55/t85）完全不变。

⚠️ 已回退的尝试（2026-09-16，实测无效）：
  曾尝试「块长按寿命比例统一（blk_pts ≈ nb/100）」+「c0 基线重定义（_c0_cycle）」
  来消除阶梯与过早临危。实测反效果：块长统一后 t85 反而**更早**（L1-49 68.6%→25.1%），
  L1-60 的 D_end 退化（0.898→0.580）；c0 只落在 2~14% 寿命（本批 AE 自冲击后即活跃，
  「先静默后回升」判据不成立）。根因不在时间尺度，而在**指标语义**：
  `risk = max(e_ae, e_strain)` 是「块内是否越界」的即时探测器 → 天然早报，且块越细越早。
  故两项改动已全部回退；L1 的指标改用文献口径（见 evaluate_l1_hi_ae.py）。

用法:
  python evaluate_l1_degree_v2.py
  python evaluate_l1_degree_v2.py --groups L1-52,L1-54
输出: results/l1_degree_v2.csv + figures/l1_degree_v2_<gid>.png
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.dirname(ROOT))          # 项目根(含 shm)
sys.path.insert(0, ROOT)
from shm.damage_index import OnlineDamageIndex     # noqa: E402
from l1_meta import load_meta                      # noqa: E402
import l1_time_align as ta                         # noqa: E402

GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
          'L1-59', 'L1-60']


def load_dfos_feet(gid, use_peak=True):
    """脚部应变矩阵（段 × 脚位点）。

    use_peak=True 优先用 `{gid}分布式应变_peak.csv`（段内**压缩峰值载荷行**，
    与第一批 FBG 的「块级循环幅值」口径一致）；不存在时回退到段内中位版本。
    """
    fpk = os.path.join(ROOT, gid, f'{gid}分布式应变_peak.csv')
    fp = fpk if (use_peak and os.path.exists(fpk)) \
        else os.path.join(ROOT, gid, f'{gid}分布式应变.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, encoding='utf-8-sig')
    pos = np.array([float(c[:-2]) for c in df.columns[1:]])
    M = df.iloc[:, 1:].to_numpy(float)
    m = load_meta(gid)
    fl, fr = m['foot_L'], m['foot_R']
    mL = (pos >= fl[0]) & (pos <= fl[1]) if fl else np.zeros_like(pos, bool)
    mR = (pos >= fr[0]) & (pos <= fr[1]) if fr else np.zeros_like(pos, bool)
    both = mL | mR
    if both.sum() == 0:
        both = np.ones_like(pos, bool)              # 退路: 全部位置
    return M[:, both]


def group_series(gid, with_di=False):
    """返回 (cyc_grid, strain, peak, D)。

    with_di=True 时追加证据层 (risk, eae, est) —— 供看板导出器复用同一口径，
    避免在导出器里重写一遍 D 的在线循环（默认 False，行为与原先完全一致）。

    注意口径：`OnlineDamageIndex` 的块级结算按 **EXT_BLOCK_PTS=500 点** 进行，
    而旧流程约定 **1 点 = 10 cycles**（500 点 = 5000 cycles = 1 块）。
    故这里把 **段级** DFOS/AE 重采样到 10-cycle 网格，
    否则按"1 段 1 点"喂入会导致 500 段才结算 1 块（D 恒为 0）。
    """
    nf = load_meta(gid)['n_f']
    feet = load_dfos_feet(gid)
    if feet is None:
        return None
    # 含 BI（冲击前, cyc=−1）与 AI（冲击后）的完整时间轴；**只用 AI 段**建寿命网格
    cyc_all, keep = ta.dfos_cycles(gid, nf, include_bi=True)
    if cyc_all.size == 0 or cyc_all.size != len(feet):
        return None
    cyc_all = cyc_all[keep]
    feet = feet[keep]
    is_ai = cyc_all >= 0
    if int(is_ai.sum()) < 3:
        return None
    cyc_seg = cyc_all[is_ai]
    strain_seg = np.nanmean(feet[is_ai], axis=1)
    ok = np.isfinite(strain_seg)
    if ok.sum() < 3:
        return None

    CYCS_PER_PT = 10
    nb = int(nf / CYCS_PER_PT) + 1
    cyc_grid = np.arange(nb) * CYCS_PER_PT
    strain = np.interp(cyc_grid, cyc_seg[ok], strain_seg[ok],
                       left=strain_seg[ok][0], right=strain_seg[ok][-1])

    peak = np.zeros(nb)
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    if os.path.exists(fp):
        ae = pd.read_csv(fp, encoding='utf-8-sig')
        if len(ae):
            ac = ta.time_to_cycle(gid, ae['time'].to_numpy(float), nf,
                                  clip_out_of_life=True)
            e = np.sqrt(np.maximum(ae['energy'].to_numpy(float), 0.0))
            ok = np.isfinite(ac)                     # 剔除冲击前(BI)事件
            ac, e = ac[ok], e[ok]
            bi = np.clip((ac / CYCS_PER_PT).astype(int), 0, nb - 1)
            np.maximum.at(peak, bi, e)

    # 块长保持引擎默认 EXT_BLOCK_PTS=500 点（=5000 cycle）；
    # 「按寿命比例缩放块长」已实测无效并回退，详见模块 docstring。
    di = OnlineDamageIndex({'ae_auto': True})   # 应变失效时自动升主证
    D = np.zeros(nb)
    risk = np.zeros(nb) if with_di else None
    eae = np.zeros(nb) if with_di else None
    est = np.zeros(nb) if with_di else None
    for i in range(nb):
        pk = float(peak[i]) if peak[i] > 0 else None
        D[i] = di.update(float(strain[i]), pk)
        if with_di:
            risk[i] = di.risk
            eae[i] = di._last_e_ae          # 证据层(与 D 同源)
            est[i] = di._last_e_strain
    if with_di:
        return cyc_grid, strain, peak, D, risk, eae, est
    return cyc_grid, strain, peak, D


def metrics(gid, cyc, D, nf):
    def fg(th):
        j = np.where(D >= th)[0]
        return float(cyc[j[0]]) if len(j) else np.nan
    t25, t55, t85 = fg(.25), fg(.55), fg(.85)
    pct = lambda t: round(t / nf * 100, 1) if np.isfinite(t) else None   # noqa: E731
    return dict(gid=gid, n_f=nf, n_seg=len(cyc), D_end=round(float(D[-1]), 3),
                t25=round(t25) if np.isfinite(t25) else None,
                t55=round(t55) if np.isfinite(t55) else None,
                t85=round(t85) if np.isfinite(t85) else None,
                t25_pct=pct(t25), t55_pct=pct(t55), t85_pct=pct(t85))


def plot(gid, cyc, D, nf):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(cyc / 1000, D, 'k-', lw=1.6, label='D(t)')
    for th, c in [(0.25, 'tab:green'), (0.55, 'tab:orange'), (0.85, 'tab:red')]:
        ax.axhline(th, color=c, ls=':', lw=0.8)
    ax.axvline(nf / 1000, color='k', ls='--', lw=1, label='n_f')
    ax.set_xlim(0, nf / 1000 * 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('cycle (k)'); ax.set_ylabel('D'); ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    ax.set_title(f'{gid}  D(t) — AE + DFOS 脚部应变（第二批, 无 FBG）')
    fig.tight_layout()
    fp = os.path.join(FIG, f'l1_degree_v2_{gid}.png')
    fig.savefig(fp, dpi=115)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS))
    a = ap.parse_args()
    rows = []
    for g in a.groups.split(','):
        g = g.strip()
        r = group_series(g)
        if r is None:
            print(f'  [{g}] 缺数据，跳过'); continue
        cyc, strain, peak, D = r
        m = metrics(g, cyc, D, load_meta(g)['n_f'])
        rows.append(m)
        print(f'  [{g}] 点={len(cyc)} D_end={m["D_end"]} '
              f'达0.25@{m["t25"]}({m["t25_pct"]}%) 0.55@{m["t55"]}({m["t55_pct"]}%) '
              f'0.85@{m["t85"]}({m["t85_pct"]}%)')
        plot(g, cyc, D, load_meta(g)['n_f'])
    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(RES, 'l1_degree_v2.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print('\n已存:', out)
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
