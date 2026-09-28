# -*- coding: utf-8 -*-
"""无监督异常检测（L1 二批 9 组；无标签、组内自校准）。

为什么用组内基线
----------------
RUL 失败的根因是「缺少跨组可比的退化指标」（n_f 跨 17 倍、传感器绝对量不可比、
退化速率个体差异大，见 README §12.6）。无监督检测把基线取自**每组自身的早期段**：

  - 不需要任何损伤标签或人工阈值；
  - 每个试件只与自己的健康态比较 → **天然规避跨组标定问题**。

两种互补的统计量
----------------
1. **马氏距离** ``D_M``：偏离基线均值，且考虑特征间相关性。
   阈值取**基线样本自身 D_M 的 99 分位**（非参数，不假设 χ² 分布）。
2. **PCA 重构误差**：用基线样本估计主成分；重构误差大 = 偏离「正常退化子空间」，
   可区分「沿正常方向的缓慢演进」与「出现新机制 / 突变 / 传感器异常」。

特征（统一到 100 格寿命网格）
------------------------------
    ae_rate     AE 事件率 [hits/s]
    ae_energy   AE 能量中位 [aJ]
    ae_amp      AE 幅值中位 [dB]
    dfos_local  DFOS 空间重分布量（§12.5 中唯一有效的 DFOS 特征）
    raf_shear   RA–AF 剪切型占比（§13）

寿命坐标
--------
二批无 FBG，只能用组内有效加载时间近似：``life = ae_time_s / (eff_h * 3600)``
（``eff_h`` 取自 results/l1_time_align.csv，已排除停机）。
⚠️ AE-cycle-rate 组间差 3.4 倍 → **跨组比较"异常起始寿命"时必须带上此误差**。

用法
----
    python l1/unsup_anomaly.py                     # 全 9 组
    python l1/unsup_anomaly.py --groups L1-49 --plot
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')

GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54',
          'L1-55', 'L1-56', 'L1-59', 'L1-60']

FEATS = ['ae_rate', 'ae_energy', 'ae_amp', 'dfos_local', 'raf_shear']
N_BIN = 100          # 寿命网格格数
BASE_FRAC = 0.15     # 基线 = 前 15% 寿命
AE_BIN_S = 60.0      # AE 预聚合窗口 [s]
PCA_KEEP = 3         # 保留的主成分数（特征 5 维）
SMOOTH_W = 5         # 移动中位宽度（抑制 AE 的 bursty 单格波动）

# ⚠️ 基线内部稳定性：基线段内 D_M 的「99 分位 / 中位」。
# 设计初衷是「基线自己不稳 → 该组不可信」，**但 2026-09-17 实测无效**：
# 9 组全部落在 1.2~1.8（L1-56 也只有 1.8），毫无区分度 ——
# 因为 maha 已按基线 MAD 标准化，baseline 内部的相对离散被归一化掉了。
# 保留该字段仅作记录；真正起作用的是下面的 OUTLIER_K 组级复核。
BASE_INSTAB_MAX = 6.0

# 组级复核：末段偏离倍数（末段 D_M / 阈值）超过其他组中位的多少倍 → 标记"疑采集异常"。
# 实测该倍数：L1-56 = 45.2（其余 2.8~10.5，中位 ≈8.9）→ 5.1 倍，取 3 倍可干净地只标出它。
OUTLIER_K = 3.0


# --------------------------------------------------------------------------
def _align_meta():
    fp = os.path.join(RES, 'l1_time_align.csv')
    if not os.path.exists(fp):
        return {}
    d = pd.read_csv(fp)
    return {r['gid']: dict(n_f=float(r['n_f']), eff_h=float(r['eff_h']))
            for _, r in d.iterrows()}


def _ae_series(gid, eff_h, n_bin=N_BIN, bin_s=AE_BIN_S):
    """AE -> (life_grid, rate, energy_med, amp_med)。"""
    cand = glob.glob(os.path.join(HERE, gid, f'*声发射*.csv'))
    cand = [c for c in cand if '（' not in c] or cand
    if not cand:
        return None
    d = pd.read_csv(cand[0])
    d = d.dropna(subset=['time'])
    if d.empty:
        return None
    t = d['time'].to_numpy(float)
    total_s = eff_h * 3600.0
    life = t / total_s
    ok = (life >= 0) & (life <= 1)
    d, life = d[ok], life[ok]
    if d.empty:
        return None
    # 以 bin_s 预聚合（避免秒级噪声），再映射到寿命网格
    grp = (t[ok] // bin_s).astype(int)
    g = d.assign(g=grp).groupby('g')
    gb_s = g['time'].min().to_numpy(float)
    rate = g['n_hits'].sum().to_numpy(float) / bin_s
    eny = g['energy'].median().to_numpy(float)
    amp = g['amplitude'].median().to_numpy(float)
    gb_life = gb_s / total_s
    grid = np.linspace(0, 1, n_bin + 1)[:-1] + 0.5 / n_bin
    interp = lambda v: np.interp(grid, gb_life, v, left=np.nan, right=np.nan)
    # 数据**实际覆盖**的寿命区间 —— 区间外的格是端点外推，不可信
    # （_fill 会把 NaN 补上，不额外记下来的话矩阵里就分不出外推格）
    covered = (grid >= gb_life.min()) & (grid <= gb_life.max())
    return grid, interp(rate), interp(eny), interp(amp), covered


def _dfos_series(gid, n_f, n_bin=N_BIN):
    fp = os.path.join(RES, f'_l1_dfos_hi_{gid}.npz')
    if not os.path.exists(fp) or n_f <= 0:
        return None, None
    z = np.load(fp)
    cyc, loc, rmse = z['cyc'].astype(float), z['local'].astype(float), z['rmse'].astype(float)
    m = (cyc >= 0) & np.isfinite(loc)
    if m.sum() < 5:
        return None, None
    # 首段 local 系统性偏大（L1-49: 863.6 vs 后续 ~350），属解调起始伪影 -> 剔除首段
    if m.sum() > 6:
        idx = np.nonzero(m)[0][1:]
        m = np.zeros_like(m)
        m[idx] = True
    life = cyc[m] / n_f
    grid = np.linspace(0, 1, n_bin + 1)[:-1] + 0.5 / n_bin
    lo = np.interp(grid, life, loc[m], left=np.nan, right=np.nan)
    rm = np.interp(grid, life, rmse[m], left=np.nan, right=np.nan)
    covered = (grid >= life.min()) & (grid <= life.max())
    return lo, rm, covered


def _raf_series(gid, n_bin=N_BIN):
    fp = os.path.join(RES, f'_l1_raf_{gid}.npz')
    if not os.path.exists(fp):
        return None
    z = np.load(fp)
    sf = z['shear_frac'].astype(float)
    if sf.size != n_bin:
        grid = np.linspace(0, 1, n_bin + 1)[:-1] + 0.5 / n_bin
        src = np.linspace(0, 1, sf.size)
        sf = np.interp(grid, src, sf)
    return sf


def _fill(x):
    """线性插值补 NaN（端点用最近值），再兜底 0。"""
    x = np.asarray(x, float)
    if np.isfinite(x).sum() == 0:
        return np.zeros_like(x)
    idx = np.arange(x.size)
    good = np.isfinite(x)
    out = np.interp(idx, idx[good], x[good])
    return out


def _smooth(X, w=SMOOTH_W):
    """对每列做宽度 w 的移动中位（抑制 AE 的 bursty 波动）。"""
    if w <= 1:
        return X
    k = w // 2
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        out[:, j] = [np.median(col[max(0, i - k):i + k + 1]) for i in range(col.size)]
    return out


def build_matrix(gid, meta=None):
    """构建 (life_grid, X, cov) —— X 形状 (n_bin, len(FEATS))。

    cov: dict，给出每格是否有**真实数据支撑**（AE / DFOS）。
         区间外的格是端点外推，物理上不可判定 → 不应报“异常”。
    """
    meta = meta or {}
    nm = meta.get(gid, {})
    n_f, eff_h = nm.get('n_f'), nm.get('eff_h')
    if not eff_h or not n_f:
        return None, None, None

    ae = _ae_series(gid, eff_h)
    dfos = _dfos_series(gid, n_f)
    sf = _raf_series(gid)

    grid = np.linspace(0, 1, N_BIN + 1)[:-1] + 0.5 / N_BIN
    cols = {}
    cov = {}
    if ae is not None:
        _, rate, eny, amp, cov['ae'] = ae
        cols['ae_rate'] = _fill(rate)
        cols['ae_energy'] = _fill(np.log10(np.clip(eny, 1e-6, None)))
        cols['ae_amp'] = _fill(amp)
    if dfos is not None:
        lo, _, cov['dfos'] = dfos
        cols['dfos_local'] = _fill(np.log10(np.clip(lo, 1e-6, None)))
    if sf is not None:
        cols['raf_shear'] = _fill(sf)

    if not cols:
        return None, None, None
    X = np.column_stack([cols.get(f, np.zeros(N_BIN)) for f in FEATS])
    # AE 天生 bursty（60 s 窗内可仅 2~3 个事件，相邻格幅值可差 24 倍）
    # -> 移动中位平滑：保留退化趋势，抑制单格噪声
    X = _smooth(X, SMOOTH_W)
    return grid, X, cov


def detect(X, base_frac=BASE_FRAC, valid=None):
    """马氏距离 + PCA 重构误差（基线 = 前 base_frac）。

    valid: 每格是否有**真实数据支撑**。区间外是端点外推，在那里报「异常」
           没有物理意义 → 不参与「首次超限」的寻找（异常分数本身照常输出）。

    返回 dict(maha, recon, thr_maha, thr_recon, first_maha, first_recon,
              first_maha_cov, first_recon_cov, base_instab, base_reliable, …)
    """
    n = X.shape[0]
    nb = max(5, int(round(n * base_frac)))
    B = X[:nb]
    mu = np.median(B, axis=0)          # 稳健中心（避免起始段伪影拉偏均值）
    # 稳健标准化（用基线 MAD），避免某个特征尺度主导
    mad = np.median(np.abs(B - mu), axis=0)
    mad[mad < 1e-12] = 1.0
    Z = (X - mu) / (1.4826 * mad)
    Zb = (B - mu) / (1.4826 * mad)

    # --- 1) 马氏距离 ---
    S = np.cov(Zb, rowvar=False) + np.eye(Z.shape[1]) * 1e-6
    Si = np.linalg.pinv(S)
    maha = np.sqrt(np.einsum('ij,jk,ik->i', Z, Si, Z))
    thr_m = float(np.percentile(maha[:nb], 99))

    # --- 2) PCA 重构误差 ---
    U, s, Vt = np.linalg.svd(Zb, full_matrices=False)
    k = min(PCA_KEEP, Vt.shape[0])
    P = Vt[:k].T                       # (d, k)
    rec = Z - (Z @ P) @ P.T
    recon = np.linalg.norm(rec, axis=1)
    thr_r = float(np.percentile(recon[:nb], 99))

    # 首次连续 3 格超限（抗单点噪声）
    def first_exceed(v, thr, k_need=3):
        over = v > thr
        run = 0
        for i, o in enumerate(over):
            run = run + 1 if o else 0
            if run >= k_need:
                return i - k_need + 1
        return -1

    # --- 3) 基线可信判据（半监督的那一半：用「已知不可信」修正判定）---
    # 阈值本身取自基线 99 分位；若该值远离基线中位，说明「健康态」自己就不平稳
    # （典型：L1-56 基线 AE 活动仅 3.4，其他组 35.8/44.3 —— 事件太稀、
    # 单格统计量在抖）。此时基线不成立、整组结论应降级。
    mb = maha[:nb]
    base_instab = float(np.percentile(mb, 99) / max(np.median(mb), 1e-6))

    # 可信区内的首次超限：把外推格置 NaN（NaN > thr 恒为 False，自然跳过）
    v = np.ones(n, bool) if valid is None else np.asarray(valid, bool)
    maha_c = np.where(v, maha, np.nan)
    recon_c = np.where(v, recon, np.nan)

    return dict(maha=maha, recon=recon, thr_maha=thr_m, thr_recon=thr_r,
                first_maha=first_exceed(maha, thr_m),
                first_recon=first_exceed(recon, thr_r),
                first_maha_cov=first_exceed(maha_c, thr_m),
                first_recon_cov=first_exceed(recon_c, thr_r),
                base_instab=base_instab,
                base_reliable=bool(base_instab <= BASE_INSTAB_MAX),
                n_base=nb, base_frac=base_frac)


def run(gid, meta=None, verbose=True):
    meta = meta or _align_meta()
    grid, X, cov = build_matrix(gid, meta)
    if grid is None:
        print(f'  [skip] {gid}: 特征不足')
        return None
    n = X.shape[0]
    # 格级可信度：参与判定的模态都应有**真实数据支撑**（区间外为端点外推）
    valid = np.ones(n, bool)
    for c in cov.values():
        valid &= c
    r = detect(X, valid=valid)
    f_m = r['first_maha']
    f_r = r['first_recon']
    fm_c, fr_c = r['first_maha_cov'], r['first_recon_cov']
    if verbose:
        def _first(idx):
            return f'{grid[idx]:.1%}' if idx >= 0 else '未超限'
        flag = '' if r['base_reliable'] else '  ⚠基线不稳'
        print(f'  [{gid}] n={n} 基线={r["n_base"]}格({r["base_frac"]:.0%}) '
              f'可信格={valid.mean():.0%}{flag}')
        print(f'         马氏阈={r["thr_maha"]:.2f} 首次={_first(f_m)}'
              f'（仅可信区 {_first(fm_c)}）   '
              f'PCA阈={r["thr_recon"]:.2f} 首次={_first(f_r)}'
              f'（仅可信区 {_first(fr_c)}）')
    out = os.path.join(RES, f'_l1_anom_{gid}.npz')
    # 新增字段（valid/base_instab/base_reliable）为**追加**，旧字段含义不变
    np.savez_compressed(out, life=grid, X=X, valid=valid, **r)
    return dict(gid=gid, grid=grid, X=X, valid=valid, **r)


def summary(rows):
    rec = []
    for r in rows:
        if not r:
            continue
        g = r['grid']
        rec.append(dict(
            gid=r['gid'],
            cov_pct=round(float(np.mean(r['valid'])), 3),
            first_maha=round(float(g[r['first_maha']]), 4) if r['first_maha'] >= 0 else None,
            first_recon=round(float(g[r['first_recon']]), 4) if r['first_recon'] >= 0 else None,
            thr_maha=round(r['thr_maha'], 2), thr_recon=round(r['thr_recon'], 2),
            maha_late=round(float(np.median(r['maha'][-30:])), 2),
            recon_late=round(float(np.median(r['recon'][-30:])), 2),
        ))
    if not rec:
        return None
    # --- 组级复核标记 ---
    # 末段偏离倍数 = 末段 D_M / 阈值。**这不是跨组归一化**（那需要"可比"、本数据做不到），
    # 而是**离群检测**：单看某组"偏离得是否远超同伴"，用于发现采集异常。
    # 实测 L1-56 = 45.2（其余 2.8~10.5，中位 ≈8.9）→ 5 倍离群，正是 §15.4 所说
    # "基线 AE 活动异常低、末段 ×31 暴涨，更像前期 AE 未工作"的那一组。
    for d in rec:
        d['late_ratio'] = round(d['maha_late'] / max(d['thr_maha'], 1e-6), 1)
    med = float(np.median([d['late_ratio'] for d in rec]))
    for d in rec:
        bad_cov = d['cov_pct'] < 0.90
        d['flag'] = '离群(疑采集异常)' if d['late_ratio'] > OUTLIER_K * med else (
            '覆盖不足' if bad_cov else '')
    df = pd.DataFrame(rec)
    out = os.path.join(RES, 'l1_anomaly_summary.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print('\n=== 汇总 ===')
    print(df.to_string(index=False))
    fm = [d['first_maha'] for d in rec if d['first_maha'] is not None]
    fr = [d['first_recon'] for d in rec if d['first_recon'] is not None]
    print(f'\n马氏：{len(fm)}/{len(rec)} 组检出，首次超限中位 {np.median(fm):.1%}' if fm else '马氏：无检出')
    print(f'PCA ：{len(fr)}/{len(rec)} 组检出，首次超限中位 {np.median(fr):.1%}' if fr else 'PCA ：无检出')
    print(f'已存 {out}')
    return df


def plot(gid):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fp = os.path.join(RES, f'_l1_anom_{gid}.npz')
    if not os.path.exists(fp):
        print(f'  [skip] {gid}: 缺 {fp}')
        return
    z = np.load(fp)
    g = z['life']
    fig, ax = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True)
    ax[0].plot(g, z['maha'], lw=1.4, color='C3')
    ax[0].axhline(z['thr_maha'], ls='--', color='k', lw=1,
                  label=f'thr={z["thr_maha"]:.1f}')
    ax[0].axvspan(0, z['base_frac'], color='0.85', alpha=.7, label='baseline')
    ax[0].set_ylabel('Mahalanobis $D_M$')
    ax[0].legend(fontsize=8, loc='upper left')
    ax[1].plot(g, z['recon'], lw=1.4, color='C0')
    ax[1].axhline(z['thr_recon'], ls='--', color='k', lw=1,
                  label=f'thr={z["thr_recon"]:.1f}')
    ax[1].axvspan(0, z['base_frac'], color='0.85', alpha=.7)
    ax[1].set_ylabel('PCA reconstruction err')
    ax[1].set_xlabel('life fraction')
    ax[1].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'{gid}  unsupervised anomaly score', fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG, f'l1_anom_{gid}.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f'  已存 {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS))
    ap.add_argument('--plot', action='store_true')
    ap.add_argument('--base-frac', type=float, default=BASE_FRAC)
    a = ap.parse_args()
    meta = _align_meta()
    rows = []
    for g in [s.strip() for s in a.groups.split(',')]:
        print(f'--- {g} ---')
        r = run(g, meta)
        rows.append(r)
        if a.plot and r:
            plot(g)
    summary(rows)


if __name__ == '__main__':
    main()
