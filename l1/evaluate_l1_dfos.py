# -*- coding: utf-8 -*-
"""DFOS(分布式应变/LUNA) 逐块验证 L1-03/04/05
DFOS 每 FBG 块(~5000cycle) 有 ~14 行空间应变快照(0~4924mm 加强筋脚)。
方法(论文式): 比较每块应变分布 vs 健康基准块(前若干块)的偏离 →
  脱粘/失稳使局部应变分布出现峰或整体重分布 → 偏离信号 → HI
输出: 每块偏离指标 + HI + 对齐论文检测点
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))   # <项目根>\l1
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

sys.path.insert(0, ROOT)                            # 使 l1_meta 可导入
from l1_meta import load_meta                       # noqa: E402
# 第一批（有 FBG）: L1-03/04/05/09；第二批（无 FBG, AE+DFOS）: L1-49..L1-56
GROUPS_V1 = ['L1-03', 'L1-04', 'L1-05', 'L1-09']
GROUPS_V2 = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
             'L1-59', 'L1-60']
GROUPS = GROUPS_V1 + GROUPS_V2
META = {g: load_meta(g) for g in GROUPS}

# --- DFOS 空间去尖峰(光纤掉点)标定, 2026-09-12 ---
# 全分辨率网格间距 0.650 mm; 本底残差 p50≈2.5 / p99≈28 µε, 与掉点完全分离。
# 阈值 = max(绝对非物理跳变, k×局部稳健尺度)。实测 |resid|>3000 µε 命中:
#   L1-03 0 点 / L1-04 7 点 / L1-05 5 点 / L1-09 263 点(≈ 降采样后 17 个位置)。
DESPIKE_ABS = 3000.0          # µε — 单点跨越 0.65 mm 的跳变超过此值即非物理
DESPIKE_K = 8.0               # 相对局部稳健尺度(1.4826×MAD)的倍数; 与 ABS 取大

# FBG 测量块划分阈值(s): 旧组(有 FBG)相邻时间戳间隔 > 该值视为新块(每块 ≈5000 cycles)
GAP_S = 300.0


def load_dfos(gid):
    fp = os.path.join(ROOT, gid, f'{gid}分布式应变.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig')
    t = df['timestamp'].to_numpy(float)
    pos = np.array([float(c[:-2]) for c in df.columns[1:]])
    M = df.iloc[:, 1:].to_numpy(float)
    return t, pos, M


def load_fbg_blocks(gid):
    """测量块时间区间 [[t0,t1],...]（相对秒）。

    旧组（有 FBG）：用 FBG 时间戳 >GAP_S 的间隔划分块（每块 ≈5000 cycles）。
    第二批（L1-49..L1-56，**无 FBG**）：用 DFOS 自己的测量段
    （`{gid}dfos_anchor.csv`，由 step0_v2.py 生成），丢弃行数 <30 的碎片段。
    实测段结构：每段 ≈120 行 @1 Hz（≈119 s），段间 ≈373 s。
    """
    fp = os.path.join(ROOT, gid, f'{gid}光纤.csv')
    if not os.path.exists(fp):
        afp = os.path.join(ROOT, gid, f'{gid}dfos_anchor.csv')
        if not os.path.exists(afp):
            print(f'  [{gid}] 无 FBG 也无 dfos_anchor，跳过')
            return np.zeros((0, 2))
        a = pd.read_csv(afp, encoding='utf-8-sig')
        a = a[a['n_rows'] >= 30].reset_index(drop=True)
        t0 = a['rel_s'].to_numpy(float)
        return np.column_stack([t0, t0 + a['seg_span_s'].to_numpy(float)])
    fb = pd.read_csv(fp, encoding='utf-8-sig')
    tf = fb['timestamp'].to_numpy(float)
    gap = np.where(np.diff(tf) > GAP_S)[0]
    bounds = np.concatenate([[0], gap + 1, [len(tf)]])
    blk = []
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 500:
            continue
        blk.append((tf[a], tf[b - 1]))
    return np.array(blk)


def distribute_rows_to_blocks(td, blk):
    """DFOS 行 → FBG 块索引; 返回每行所属块(-1=窗间/后)"""
    n = len(td)
    out = np.full(n, -1)
    for i, (t0, t1) in enumerate(blk):
        m = (td >= t0) & (td <= t1)
        out[m] = i
    return out


def block_profile(M, rows_in_blk):
    """每块: 空间应变分布的代表曲线 = 块内行中位(逐空间点)。"""
    rows = np.where(rows_in_blk)[0]
    if len(rows) == 0:
        return None
    return np.nanmedian(M[rows], axis=0)


def drift_feature(prof, base_prof, pos):
    """块分布 vs 健康基线的偏离。脱粘特征: 局部出现应变峰/凹陷。
    指标: 1) RMSE 全分布差; 2) 局部峰强度(某空间窗内 max|dev|)。
    返回 (rmse, local_peak)。"""
    dev = prof - base_prof
    # 去两端噪声(只在加强筋脚有效区)
    ok = np.isfinite(dev)
    rmse = float(np.sqrt(np.nanmean(dev[ok] ** 2))) if ok.sum() > 10 else np.nan
    # 局部峰: 用滑窗找最强局部偏离(脱粘产生峰)
    w = 100  # ~65mm
    from numpy.lib.stride_tricks import sliding_window_view as sw
    if ok.sum() > 2 * w:
        v = np.where(np.isnan(dev), 0.0, dev)
        sq = sw(np.abs(v), w).mean(axis=1)
        local = float(np.max(sq)) if len(sq) else np.nan
    else:
        local = np.nan
    return rmse, local


def _medfilt_row(row, win=9):
    """一维中值滤波(edge 填充), 不依赖 scipy。"""
    from numpy.lib.stride_tricks import sliding_window_view as sw
    w = int(win) | 1                       # 强制奇数窗
    if row.size < w:
        return row.copy()
    ext = np.pad(row, w // 2, mode='edge')
    return np.median(sw(ext, w), axis=1)


def despike_row(row, win=9, abs_th=DESPIKE_ABS, k=DESPIKE_K):
    """单条空间分布的**去尖峰**(光纤掉点检测与修复)。

    依据: 掉点是**孤立空间尖峰**(L1-09 单点跳 1.6 万 µε, 邻点仅几个 µε),
    而真实损伤是**平滑空间梯度** → 用"与局部中位偏差 > max(abs_th, k×1.4826·MAD)"判定,
    命中点用局部中位替换。返回 (清洗后行, 修复点数)。
    """
    r = np.asarray(row, float)
    finite = np.isfinite(r)
    if finite.sum() < 10:
        return r.copy(), 0
    fill = r.copy()
    if not finite.all():                   # NaN 先沿位置插值(仅用于算中位)
        idx = np.arange(r.size)
        fill = np.interp(idx, idx[finite], r[finite])
    med = _medfilt_row(fill, win)
    resid = fill - med
    sc = float(np.median(np.abs(resid))) * 1.4826
    m = np.abs(resid) > max(abs_th, k * sc)
    out = r.copy()
    if m.any():
        out[m] = med[m]
    return out, int(m.sum())


def despike_profiles(prof, **kw):
    """批量去尖峰(逐条块分布)。返回 (清洗后二维数组, 总修复点数)。"""
    out = np.asarray(prof, float).copy()
    n = 0
    for b in range(out.shape[0]):
        out[b], c = despike_row(out[b], **kw)
        n += c
    return out, n


def analyze(gid, base_mode='bi'):
    nf = META[gid]['n_f']
    td, pos, M = load_dfos(gid)
    blk = load_fbg_blocks(gid)
    nblk = len(blk)
    row_blk = distribute_rows_to_blocks(td, blk)

    # ---- 阶段掩码（第二批有 BI=冲击前 / AI=冲击后；第一批无 phase → 全按 AI）----
    has_fbg = os.path.exists(os.path.join(ROOT, gid, f'{gid}光纤.csv'))
    bi_mask = np.zeros(nblk, bool)
    afp = os.path.join(ROOT, gid, f'{gid}dfos_anchor.csv')
    if (not has_fbg) and os.path.exists(afp):
        aa = pd.read_csv(afp, encoding='utf-8-sig')
        if 'phase' in aa.columns:
            aa = aa[aa['n_rows'] >= 30].reset_index(drop=True)   # 与 blk 同序
            if len(aa) == len(blk):
                bi_mask = aa['phase'].astype(str).str.upper().to_numpy() == 'BI'

    # ---- 寿命锚：只在 **AI（冲击后）** 块上均匀分布；BI 块 cyc = −1 ----
    n_ai = nblk - int(bi_mask.sum())
    if bi_mask.any() and n_ai > 0:
        blk_cyc = np.full(nblk, -1.0)
        blk_cyc[~bi_mask] = (np.arange(n_ai) + 0.5) / n_ai * nf
    else:
        blk_cyc = (np.arange(nblk) + 0.5) / nblk * nf

    # ---- 健康基准 ----
    # 方案A(2026-09-12，第一批): **首个测量块**（论文"相对首测"口径）
    # 第二批（2026-09-14）: `--base bi`（默认）= **BI 冲击前段**（物理真健康态，
    #   适合冲击影响评估/脱粘监测）；`--base post` = 冲击后前 10 段
    #   （排除冲击本身造成的偏离，适合“疲劳累积”视角）。
    if has_fbg:
        base_rows = np.isin(row_blk, [0])
    elif base_mode == 'bi' and bi_mask.any():
        base_rows = np.isin(row_blk, np.where(bi_mask)[0])
        print(f'  [{gid}] 健康基准 = BI(冲击前) {int(bi_mask.sum())} 段')
    else:
        cand = np.where(~bi_mask)[0][:10] if bi_mask.any() else np.arange(10)
        base_rows = np.isin(row_blk, cand)
        print(f'  [{gid}] 健康基准 = 冲击后前 10 段')
    if base_rows.sum() == 0:
        base_rows = row_blk >= 0                      # 退路: 全部
    base_prof = np.nanmedian(M[base_rows], axis=0)
    base_nan0 = int(np.isnan(base_prof).sum())          # 清洗前的原始缺失数(数据质量指标)
    # 基准自身也去尖峰(否则基线的掉点会污染全部块的偏离)
    base_prof, n_fix_base = despike_row(base_prof)
    # 基准就是块0分布时, 其修复已含在下面的逐块统计里 → 避免重复计数
    n_fix_base = 0 if bool((row_blk == 0).any()) else n_fix_base
    # 每块特征(先去尖峰再算偏离)
    rmse, local = [], []
    n_fix = 0
    for k in range(nblk):
        prof = block_profile(M, row_blk == k)
        if prof is None:
            rmse.append(np.nan); local.append(np.nan); continue
        prof, nfx = despike_row(prof)
        n_fix += nfx
        r, l = drift_feature(prof, base_prof, pos)
        rmse.append(r); local.append(l)
    rmse = np.array(rmse); local = np.array(local)
    # HI: 用 local(局部峰, 脱粘特征) 归一; 健康期应≈0
    #   2026-09-14: **BI（冲击前）块与"冲击后"基准不同源** → 必须排除出 HI 评估，
    #   否则 BI 块相对冲击后基准的偏离会让单调包络在 BI 段立即锁定，峰达时刻退化为 cyc=−1。
    ref = local[~bi_mask] if bi_mask.any() else local
    p90 = float(np.nanpercentile(ref, 90)) if np.isfinite(ref).any() else 0.0
    scale = p90 if p90 > 0 else 1.0
    hi = np.clip(local / scale, 0, 1)
    if bi_mask.any():
        hi[bi_mask] = np.nan
    # 单调累积降噪（仅对有效块）
    hi2 = np.full(nblk, np.nan)
    last = 0.0
    for k in range(nblk):
        if not np.isfinite(hi[k]):
            continue
        last = max(last, hi[k])
        hi2[k] = last
    hi = hi2
    valid = np.isfinite(hi)
    tail = hi[valid][int(valid.sum() * 0.6):] if valid.any() else np.array([])
    mono = float(np.mean(np.diff(tail) >= 0)) * 100 if len(tail) > 2 else np.nan

    def first_th(th):
        idx = np.where(hi >= th)[0]
        return float(blk_cyc[idx[0]]) if len(idx) else np.nan

    row = dict(gid=gid, n_f=nf, n_blk=nblk, base_prof_nan=base_nan0,
               spike_fix=round(float(n_fix + n_fix_base)),
               rmse_end=round(float(rmse[-1]), 1), local_scale=round(scale, 1),
               HI_end=round(float(hi[-1]), 3), tail_mono=round(mono, 1),
               warn25=round(first_th(.25), 1) if not np.isnan(first_th(.25)) else np.nan,
               warn50=round(first_th(.5), 1) if not np.isnan(first_th(.5)) else np.nan,
               warn70=round(first_th(.7), 1) if not np.isnan(first_th(.7)) else np.nan)
    print(f'\n[{gid}] 块={nblk}  local_scale={scale:.0f}  HI_end={hi[-1]:.2f}  尾段单调={mono:.0f}%  '
          f'去尖峰={int(n_fix + n_fix_base)}点')
    print(f'  HI 达0.25@{row["warn25"]}  0.50@{row["warn50"]}  0.70@{row["warn70"]} (n_f={nf})')
    print('  论文参考: ' + ', '.join(f'{l}@{c}' for l, c in META[gid]['refs']))
    np.savez_compressed(os.path.join(RES, f'_l1_dfos_hi_{gid}.npz'),
                        cyc=blk_cyc, hi=hi, rmse=rmse, local=local,
                        pos=pos, base_prof=base_prof, nf=nf)
    return row


def plot(gid):
    z = np.load(os.path.join(RES, f'_l1_dfos_hi_{gid}.npz'))
    cyc, hi, rmse, local, pos = z['cyc'], z['hi'], z['rmse'], z['local'], z['pos']
    nf = float(z['nf'])
    x = cyc / 1000
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(x, local, 'o-', color='tab:blue', lw=1.2, label='局部峰偏离')
    axes[0].plot(x, rmse, 's--', color='tab:orange', lw=0.9, label='RMSE')
    axes[0].set_ylabel('DFOS 分布偏离'); axes[0].legend()
    axes[0].set_title(f'{gid}  DFOS(分布式应变) 脱粘特征 (虚线=论文检测, 黑=失效)')
    axes[1].plot(x, hi, 'o-', color='k', lw=1.3)
    for th, c in [(0.25, '#e07b00'), (0.5, '#d62728'), (0.7, '#8b0000')]:
        axes[1].axhline(th, color=c, lw=0.7, ls='--', alpha=0.6)
    axes[1].set_ylabel('HI (局部峰偏离归一)'); axes[1].set_ylim(-0.02, 1.05)
    # 热图: 应变分布随块变化(减基准)
    td, M, blk = None, None, None
    td, _, M = _reload(gid)
    blk = _blk(gid)
    nblk = len(cyc)
    row_blk = _rows(gid, td, blk)
    base_prof = z['base_prof']
    hm = np.full((nblk, M.shape[1]), np.nan)
    for k in range(nblk):
        rows = np.where(row_blk == k)[0]
        if len(rows):
            hm[k] = np.nanmedian(M[rows], axis=0) - base_prof
    im = axes[2].imshow(hm, aspect='auto', origin='lower',
                        extent=[pos[0], pos[-1], 0, nblk], cmap='RdBu_r',
                        vmin=-np.nanpercentile(np.abs(hm), 99), vmax=np.nanpercentile(np.abs(hm), 99))
    axes[2].set_ylabel('块 #'); axes[2].set_xlabel('沿加强筋脚位置 (mm)')
    axes[2].set_title('应变分布 vs 基准 (红=拉应变峰)')
    fig.colorbar(im, ax=axes[2])
    for ax in axes[:2]:
        for lab, c in META[gid]['refs']:
            ax.axvline(c / 1000, color='grey', ls=':', lw=0.8)
        ax.axvline(nf / 1000, color='k', ls='--', lw=1)
    fig.tight_layout()
    out = os.path.join(FIG, f'l1_dfos_{gid}.png')
    fig.savefig(out, dpi=110)
    print('已存:', out)


_cache = {}


def _reload(gid):
    if gid not in _cache:
        _cache[gid] = load_dfos(gid)
    return _cache[gid]


def _blk(gid):
    if ('blk', gid) not in _cache:
        _cache[('blk', gid)] = load_fbg_blocks(gid)
    return _cache[('blk', gid)]


def _rows(gid, td, blk):
    if ('rows', gid) not in _cache:
        _cache[('rows', gid)] = distribute_rows_to_blocks(td, blk)
    return _cache[('rows', gid)]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(META))
    ap.add_argument('--mode', default='hi', choices=['hi', 'plot'])
    ap.add_argument('--base', default='post', choices=['bi', 'post'],
                    help="健康基准: post=冲击后前10段(默认, 疲劳视角, 与历史口径可比) "
                         "/ bi=BI冲击前段(冲击影响评估视角)")
    a = ap.parse_args()
    groups = a.groups.split(',')
    if a.mode == 'hi':
        rows = [analyze(g, a.base) for g in groups]
        pd.DataFrame(rows).to_csv(os.path.join(RES, 'l1_dfos_metrics.csv'),
                                  index=False, encoding='utf-8-sig')
        for g in groups:
            plot(g)
    else:
        for g in groups:
            plot(g)


if __name__ == '__main__':
    main()
