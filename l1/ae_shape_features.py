# -*- coding: utf-8 -*-
"""AE 分布形状特征：**b 值** + 幅值/能量分位比（第二批试件专用）

为什么需要单独一脚本
  第二批（L1-49..60）的 AE **有真实时间戳**（.pridb 的 Time + Suspend/Resume markers）
  → 事件级统计特征**可用**（主样本因 AE 无时间戳而构造上不可用）。
  但 `{gid}声发射.csv` 已按 1 秒 bin 聚合（只留 max），**丢失事件分布**
  → b 值 / 分位比必须回到 `.pridb` 重读事件级 Amp/Eny。

特征（均按寿命进度分 N_BIN 箱；cycle 锚用 `l1_time_align.time_to_cycle(method='seg')`）：
  b_val         Gutenberg-Richter 斜率：log10 N(≥A) = a − b·A（A 用 dB）
                （AE 经典损伤量：b 值下降 → 大事件占比升高 → 损伤加剧）
  amp_p50/90/99 幅值(dB) 分位数
  amp_r90/95   幅值尾部分位比 p90/p50、p95/p50
  eny_lp50/90  能量(log10) 分位数
  hit_rate     每箱事件数（速率类特征，主样本不可用）

用法:
  python ae_shape_features.py                       # 全部 9 组
  python ae_shape_features.py --groups L1-55        # 单组
输出: results/_l1_ae_shape_{gid}.npz  （life + 各特征，长度 N_BIN）
"""
import os
import io
import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')
sys.path.insert(0, ROOT)
from l1_meta import load_meta                       # noqa: E402
import l1_time_align as ta                          # noqa: E402
import ae_io                                        # noqa: E402

GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
          'L1-59', 'L1-60']
TIME_BASE = 1e7
N_BIN = 100
AE_CHUNK = 2_000_000


def b_value(amp_db, binw=1.0, min_ev=200):
    """Gutenberg-Richter b 值（幅值 dB 的累积计数对数斜率）。"""
    a = amp_db[np.isfinite(amp_db)]
    if a.size < min_ev:
        return np.nan
    lo, hi = np.percentile(a, [5, 99])
    if hi - lo < 2.0:
        return np.nan
    edges = np.arange(lo, hi + binw, binw)
    if edges.size < 6:
        return np.nan
    cnt, _ = np.histogram(a, bins=edges)
    n_ge = np.cumsum(cnt[::-1])[::-1].astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])          # 与 cnt 同长
    ok = n_ge > 0
    x, y = centers[ok], np.log10(n_ge[ok])
    if x.size < 5:
        return np.nan
    return float(-np.polyfit(x, y, 1)[0])


def compute(gid, n_bin=N_BIN):
    nf = load_meta(gid)['n_f']
    d = os.path.join(ROOT, gid, 'AE')
    files = sorted(f for f in os.listdir(d) if f.endswith('.pridb'))
    if not files:
        print(f'  [{gid}] 无 AE'); return None
    # --- 多段 .pridb 会话缝合（见 ae_io / docs/details.md §17.11）---
    pl = ae_io.plan(gid)
    print(ae_io.describe(gid))
    files = [(s['path'], o) for s, o in zip(pl['parts'], pl['offsets'])]
    # 逐箱累积 amp_db（float32 省内存）与能量
    bins_amp = [[] for _ in range(n_bin)]
    bins_eny = [[] for _ in range(n_bin)]
    n_total = 0
    for fp, off in files:
        con = sqlite3.connect(f'file:{fp}?mode=ro', uri=True)
        sql = 'SELECT Time, Eny, Amp FROM ae_data WHERE SetType=2'
        for chunk in pd.read_sql_query(sql, con, chunksize=AE_CHUNK):
            n_total += len(chunk)
            t = chunk['Time'].to_numpy('float64') / TIME_BASE + off
            amp = chunk['Amp'].to_numpy('float64')
            eny = chunk['Eny'].fillna(0.0).to_numpy('float64')
            cyc = ta.time_to_cycle(gid, t, nf)                 # 段锚（因果不涉及未来）
            li = np.clip((cyc / nf * n_bin).astype('int64'), 0, n_bin - 1)
            with np.errstate(divide='ignore', invalid='ignore'):
                amp_db = 20.0 * np.log10(np.where(amp > 0, amp, np.nan))
            eny_l = np.log10(np.maximum(eny, 1.0))
            order = np.argsort(li, kind='stable')
            li_s, a_s, e_s = li[order], amp_db[order], eny_l[order]
            edges = np.searchsorted(li_s, np.arange(n_bin + 1))
            for b in range(n_bin):
                s, e = edges[b], edges[b + 1]
                if e > s:
                    bins_amp[b].append(a_s[s:e].astype(np.float32))
                    bins_eny[b].append(e_s[s:e].astype(np.float32))
        con.close()
        print(f'    {fn}: 已读')
    print(f'  [{gid}] 总 hits={n_total:,}')

    keys = ['b_val', 'amp_p50', 'amp_p90', 'amp_p99', 'amp_r90', 'amp_r95',
            'eny_lp50', 'eny_lp90', 'hit_rate']
    out = {k: np.full(n_bin, np.nan) for k in keys}
    for b in range(n_bin):
        if not bins_amp[b]:
            continue
        a = np.concatenate(bins_amp[b])
        e = np.concatenate(bins_eny[b])
        out['hit_rate'][b] = a.size
        if a.size < 20:
            continue
        p50, p90, p95, p99 = np.nanpercentile(a, [50, 90, 95, 99])
        out['amp_p50'][b], out['amp_p90'][b] = p50, p90
        out['amp_p99'][b] = p99
        out['amp_r90'][b] = p90 / p50 if p50 else np.nan
        out['amp_r95'][b] = p95 / p50 if p50 else np.nan
        out['eny_lp50'][b] = float(np.nanmedian(e))
        out['eny_lp90'][b] = float(np.nanpercentile(e, 90))
        out['b_val'][b] = b_value(a)
    life = (np.arange(n_bin) + 0.5) / n_bin
    os.makedirs(RES, exist_ok=True)
    fp = os.path.join(RES, f'_l1_ae_shape_{gid}.npz')
    np.savez_compressed(fp, life=life, **out)
    print(f'    → {os.path.basename(fp)}  b_val 中位={np.nanmedian(out["b_val"]):.2f}  '
          f'b: 前20%={np.nanmedian(out["b_val"][:20]):.2f} / 后20%={np.nanmedian(out["b_val"][-20:]):.2f}')
    return dict(gid=gid, n_hits=n_total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS))
    ap.add_argument('--bins', type=int, default=N_BIN)
    a = ap.parse_args()
    for g in a.groups.split(','):
        g = g.strip()
        print(f'--- {g} ---')
        compute(g, a.bins)
        print()


if __name__ == '__main__':
    main()
