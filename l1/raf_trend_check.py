# -*- coding: utf-8 -*-
"""RA-AF 趋势量 vs DFOS 趋势量：跨组一致性检验。

为什么不用逐点相关
------------------
`ae_raf.compare_with_dfos()` 的逐点 Pearson 相关要求「AE 事件时间 → 疲劳循环」
的映射是精确的。而 L1 二批没有 FBG 锚点，只能按「寿命均匀分布」近似，
实测 AE-cycle-rate 组间差 3.4 倍（0.38~1.28 Hz），时间轴误差会直接污染逐点相关。

本脚本改用「趋势幅度」：
    Δ = mean(寿命后 20%) − mean(寿命前 20%)
只依赖寿命**排序**，对时间标定的绝对误差稳健。

输出: results/l1_raf_trend_check.csv
"""
import os
import glob

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
FRAC = 0.20


def _trend(x, frac=FRAC):
    """后 frac 寿命段均值 − 前 frac 寿命段均值。"""
    x = np.asarray(x, float)
    k = max(3, int(round(x.size * frac)))
    return float(np.nanmean(x[-k:]) - np.nanmean(x[:k]))


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return np.nan
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    rows = []
    for fp in sorted(glob.glob(os.path.join(RES, '_l1_raf_*.npz'))):
        gid = os.path.basename(fp)[len('_l1_raf_'):-len('.npz')]
        fd = os.path.join(RES, f'_l1_dfos_hi_{gid}.npz')
        if not os.path.exists(fd):
            print(f'  [skip] {gid}: 缺 DFOS npz')
            continue
        z, zd = np.load(fp), np.load(fd)
        cyc = zd['cyc'].astype(float)
        m = cyc >= 0
        if m.sum() < 10:
            print(f'  [skip] {gid}: DFOS 有效点不足 ({m.sum()})')
            continue
        loc = zd['local'][m].astype(float)
        sf = z['shear_frac'].astype(float)
        k = max(3, int(round(sf.size * FRAC)))
        rows.append(dict(
            gid=gid,
            shear_first=float(np.nanmean(sf[:k])),
            shear_last=float(np.nanmean(sf[-k:])),
            d_shear=_trend(sf),
            d_ra=_trend(z['ra_med']),
            d_af=_trend(z['af_med']),
            d_local=_trend(loc),
            bi_shear=float(z['bi_shear']) if np.isfinite(z['bi_shear']) else np.nan,
        ))

    if not rows:
        print('无可用数据')
        return

    df = pd.DataFrame(rows)
    out = os.path.join(RES, 'l1_raf_trend_check.csv')
    df.round(4).to_csv(out, index=False, encoding='utf-8-sig')

    print('\n各组趋势幅度（后 20% − 前 20%）：')
    print(df[['gid', 'shear_first', 'shear_last', 'd_shear',
              'd_ra', 'd_af', 'd_local', 'bi_shear']]
          .round(3).to_string(index=False))

    print(f'\n跨组相关 vs ΔDFOS local（n={len(df)}）：')
    for name, col in [('Δ剪切占比', 'd_shear'), ('ΔRA中位', 'd_ra'),
                      ('ΔAF中位', 'd_af')]:
        x, y = df[col].to_numpy(), df['d_local'].to_numpy()
        mm = np.isfinite(x) & np.isfinite(y)
        if mm.sum() >= 4:
            r = float(np.corrcoef(x[mm], y[mm])[0, 1])
            print(f'  {name:>10s}: Pearson r={r:+.3f}  '
                  f'Spearman ρ={_spearman(x, y):+.3f}  (n={mm.sum()})')

    print('\n组内方向一致性（不依赖 DFOS）：')
    for c, exp in [('d_shear', '↑'), ('d_ra', '↑'), ('d_af', '↓')]:
        v = df[c].to_numpy()
        fin = np.isfinite(v)
        if exp == '↓':
            n_pos = int((v[fin] < 0).sum())
        else:
            n_pos = int((v[fin] > 0).sum())
        print(f'  {c:>8s} 期望{exp}: {n_pos}/{int(fin.sum())} 组符合')

    triad = int(((df['d_shear'] > 0) & (df['d_ra'] > 0) & (df['d_af'] < 0)).sum())
    print(f'  三参数同时指向「剪切型增强」的组数 = {triad}/{len(df)}')
    print(f'\n已存: {out}')


if __name__ == '__main__':
    main()
