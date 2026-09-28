# -*- coding: utf-8 -*-
"""L1 公开数据集：016-020 式"连续损伤度 D(t) + 三级预警"——【基线重定义 + 应变漂移证据】

背景(§5.3): 原方法假设"健康基线→检测损伤起始"; L1 是冲击后疲劳(0 cycle 即 BVID),
早期冲击脱粘 AE 被正确判为损伤型事件 → 默认 D 在 ~20% 寿命即饱和。

两道适配(均自动、不硬编码):
  A. 基线重定义: 以"冲击后稳定态"为 D=0, D 度量之后的损伤进一步扩展。
     c0 = 前 60% 寿命内 AE 计数(3 块平滑)最小值之后、首个"平滑计数>1.5×最小值"的块
          (即活动自稳定态可持续回升 = 扩展起点); 从 c0 起重新起算, 线首(<c0) D=0。
  B. 应变漂移证据(可选 --strain-evidence): 补上"块级 FBG 应变相对冲击后基线的漂移"证据,
     归一化尺度自动取数据范围。与原方法一致: risk = max(e_ae, e_strain)。
     —— 解决 L1-04 这类"AE 只覆盖 61% 寿命 / 早期 AE 误导"的组: D 末期不再因无 AE 而衰减,
        并由(100% 覆盖的)应变漂移驱动。

点网格: 1 点/10 cycle → 500 点 = 5000 cycle = 1 个 FBG 块(与 EXT_BLOCK_PTS 对齐)。
每点 strain = FBG 块应变均值插值; AE = 10-cycle bin 内事件峰值(sqrt(energy))。
AE 事件按 time 去重(Vallen 多通道叠加)。

用法:
  python evaluate_l1_degree.py --baseline
  python evaluate_l1_degree.py --baseline --strain-evidence --params rise=0.05
输出: results/l1_degree.csv + figures/l1_degree_<gid>.png
"""
import os, argparse, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根(含 shm)
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # l1/
from shm.damage_index import OnlineDamageIndex
from shm.config import EXT_BLOCK_PTS

ROOT = os.path.dirname(os.path.abspath(__file__))   # <项目根>\l1
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

_spec = importlib.util.spec_from_file_location('r', os.path.join(ROOT, 'reproduce_broer_l1.py'))
_r = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_r)
META = _r.META
CYCS_PER_BLK = 5000
CYCS_PER_PT = 10                    # 1 点 = 10 cycle → 500 点 = 5000 cycle
C0_RISE_FAC = 1.5                   # c0 拐点: AE 活动回升超过 1.5× 稳定最小值
FBG_COLS = ['fbg1', 'b1', 'b2', 'b3', 'b4', 'b5']
GAP_S = 300.0
PTS_PER_BLK = CYCS_PER_BLK // CYCS_PER_PT   # =500 (应等于 EXT_BLOCK_PTS)


def fbg_block_strain(gid):
    """每 FBG 块(>300s 间隔分隔): cycle=(k+0.5)*5000, 应变=块内 FBG_COLS 均值。"""
    df = pd.read_csv(os.path.join(ROOT, gid, f'{gid}光纤.csv'), encoding='utf-8-sig')
    t = df['timestamp'].to_numpy(float)
    cols = [c for c in FBG_COLS if c in df.columns]
    s = df[cols].mean(axis=1, skipna=True).to_numpy(float)
    gap = np.where(np.diff(t) > GAP_S)[0]
    bounds = np.concatenate([[0], gap + 1, [len(t)]])
    cyc, val = [], []
    k = 0
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 500:
            continue
        cyc.append((k + 0.5) * CYCS_PER_BLK); val.append(float(np.nanmean(s[a:b]))); k += 1
    return np.array(cyc), np.array(val)


def load_ae_dedup(gid):
    """AE 事件: 按 time 去重(Vallen 多通道叠加)取 max energy → (time, energy)。"""
    df = pd.read_csv(os.path.join(ROOT, gid, f'{gid}声发射.csv'),
                     encoding='utf-8-sig', usecols=['time', 'energy'])
    g = df.groupby('time', sort=True)['energy'].max()
    return g.index.to_numpy(float), g.to_numpy(float)


def block_counts(gid, ae_time):
    """每 5000-cycle 块的 AE 事件数。"""
    cyc = _r.time_to_cycle(gid, ae_time)
    nb = int(META[gid]['n_f'] / CYCS_PER_BLK)
    bi = np.clip((cyc / CYCS_PER_BLK).astype(int), 0, nb - 1)
    return np.bincount(bi, minlength=nb)


def shakedown_cycle(gid, ae_time):
    """c0 = 冲击后稳定态终点(拐点, 自动)。"""
    cnt = block_counts(gid, ae_time).astype(float)
    nb = len(cnt)
    sm = np.convolve(cnt, np.ones(3) / 3, mode='same')
    hi = max(2, int(nb * 0.6))
    valid = np.arange(1, hi)
    valid = valid[cnt[valid] > 0]
    if len(valid) == 0:
        return 0.0
    kmin = int(valid[np.argmin(sm[valid])])
    base = sm[kmin]
    after = np.where(sm[kmin:hi] > base * C0_RISE_FAC)[0]
    k = kmin + (int(after[0]) if len(after) else 0)
    return float(k * CYCS_PER_BLK)


def strain_drift_evidence(strain, i0):
    """块级 FBG 应变漂移证据 e_strain∈[0,1]（自动）。
    基线 ref = c0 所在块的应变; drift = |应变块均值 − ref|;
    尺度 = drift 的数据范围(自动 min-max 归一, 同论文式1)。返回每块 e_strain。
    """
    nblk = len(strain) // PTS_PER_BLK
    blk = np.array([np.nanmean(strain[b * PTS_PER_BLK:(b + 1) * PTS_PER_BLK])
                    for b in range(nblk)])
    i0_blk = min(max(0, i0 // PTS_PER_BLK), nblk - 1)
    ref = float(blk[i0_blk])
    drift = np.abs(blk - ref)
    mn, mx = float(np.nanmin(drift)), float(np.nanmax(drift))
    if mx - mn < 1e-12:
        return np.zeros(nblk)
    return (drift - mn) / (mx - mn)


def run_group(gid, params=None, baseline=False, strain_ev=False, fusion='max'):
    nf = META[gid]['n_f']
    nb = int(nf / CYCS_PER_PT) + 1
    cyc_grid = np.arange(nb) * CYCS_PER_PT
    bcyc, bval = fbg_block_strain(gid)
    strain = np.interp(cyc_grid, bcyc, bval, left=bval[0], right=bval[-1])
    tae, eae = load_ae_dedup(gid)
    cyc_ae = _r.time_to_cycle(gid, tae)
    idx = np.clip((cyc_ae / CYCS_PER_PT).astype(int), 0, nb - 1)
    peak = np.zeros(nb)
    np.maximum.at(peak, idx, np.sqrt(np.maximum(eae, 0.0)))
    c0 = shakedown_cycle(gid, tae) if baseline else 0.0
    i0 = int(round(c0 / CYCS_PER_PT))
    di = OnlineDamageIndex(params)
    D = np.zeros(nb)
    RISK = np.zeros(nb)          # 逐点 risk  (供看板导出)
    EAE = np.zeros(nb)           # 逐点 e_ae  (供看板导出)
    EST = np.zeros(nb)           # 逐点 e_strain(供看板导出)
    if not strain_ev:
        for i in range(i0, nb):
            pk = float(peak[i]) if peak[i] > 0 else None
            D[i] = di.update(float(strain[i]), pk)
            RISK[i] = di.risk; EAE[i] = di._last_e_ae; EST[i] = di._last_e_strain
    else:
        p = dict(params or {})
        rise = float(p.get('rise', 0.05)); fall = float(p.get('fall', 0.008))
        l_en = bool(p.get('latch_enable', True))
        l_low = float(p.get('latch_conf_low', 0.30)); l_drop = float(p.get('latch_conf_drop', 0.15))
        l_blk = int(p.get('latch_conf_blk', 3)); l_fast = float(p.get('latch_rise_fast', 0.5))
        e_st = strain_drift_evidence(strain, i0)
        d = 0.0; b_prev = 0; d_hist = []; latched = False
        risk = 0.0; ae_ev = 0.0; st_ev = 0.0
        for i in range(i0, nb):
            pk = float(peak[i]) if peak[i] > 0 else None
            di.update(float(strain[i]), pk)
            if di._bcount > b_prev:                    # 一个块结算
                b_prev = di._bcount
                gb = min(i // PTS_PER_BLK, len(e_st) - 1)
                ae_ev = float(di._last_e_ae); st_ev = float(e_st[gb])
                if fusion == 'mean':
                    risk = 0.5 * ae_ev + 0.5 * st_ev         # 论文式: 等权融合
                elif fusion == 'min':
                    risk = min(ae_ev, st_ev)                 # 需双证据互证
                else:
                    risk = max(ae_ev, st_ev)                 # 任一证据(原方法式)
                d = d + rise * (risk - d) if risk > d else d - fall * d
                d = min(1.0, max(0.0, d))
                if l_en:
                    d_hist.append(d)
                    if len(d_hist) > l_blk:
                        d_hist.pop(0)
                    if not latched and len(d_hist) >= l_blk and d >= l_low \
                       and min(d_hist) >= l_drop:
                        latched = True
                    if latched and risk > d:
                        d = min(1.0, d + l_fast * (risk - d))
            D[i] = d
            RISK[i] = risk; EAE[i] = ae_ev; EST[i] = st_ev
    return dict(gid=gid, cyc=cyc_grid, D=D, strain=strain, peak=peak, c0=c0,
                risk=RISK, eae=EAE, est=EST)


def metrics(r):
    gid = r['gid']; nf = META[gid]['n_f']; cyc = r['cyc']; D = r['D']

    def fg(th):
        j = np.where(D >= th)[0]
        return float(cyc[j[0]]) if len(j) else float('nan')
    t25, t55, t85 = fg(.25), fg(.55), fg(.85)
    return dict(gid=gid, n_f=nf, c0=round(r['c0']),
                D_end=round(float(D[-1]), 3),
                t25=round(t25) if not np.isnan(t25) else None,
                t55=round(t55) if not np.isnan(t55) else None,
                t85=round(t85) if not np.isnan(t85) else None,
                t85_pct=round(t85 / nf * 100, 1) if not np.isnan(t85) else None,
                lead85=int(nf - t85) if not np.isnan(t85) else None)


def plot(r):
    gid = r['gid']; nf = META[gid]['n_f']; x = r['cyc'] / 1000
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, r['D'], 'k-', lw=1.6, label='D(t) 损伤度')
    for th, c in [(0.25, 'tab:green'), (0.55, 'tab:orange'), (0.85, 'tab:red')]:
        ax.axhline(th, color=c, ls=':', lw=0.8)
    if r['c0'] > 0:
        ax.axvline(r['c0'] / 1000, color='tab:purple', ls='-', lw=1,
                   label=f'基线终点 c0={r["c0"]/1000:.1f}k')
    for lab, c in META[gid]['refs']:
        ax.axvline(c / 1000, color='grey', ls=':', lw=0.7)
    ax.axvline(nf / 1000, color='k', ls='--', lw=1, label='n_f')
    ax.set_xlim(0, nf / 1000 * 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel('cycle (k)'); ax.set_ylabel('D'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title(f'{gid}  D(t) 连续损伤度 (基线重定义)')
    fig.tight_layout()
    out = os.path.join(FIG, f'l1_degree_{gid}.png'); fig.savefig(out, dpi=120)
    print('已存:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(META))
    ap.add_argument('--baseline', action='store_true', help='启用基线重定义')
    ap.add_argument('--strain-evidence', dest='strain_ev', action='store_true',
                    help='加入 FBG 应变漂移证据')
    ap.add_argument('--fusion', default='max', choices=['max', 'mean', 'min'],
                    help='AE 与应变证据融合方式')
    ap.add_argument('--params', default='', help='形如 rise=0.05,fall=0.008')
    a = ap.parse_args()
    params = {}
    for kv in [s for s in a.params.split(',') if '=' in s]:
        k, v = kv.split('=', 1)
        try:
            params[k] = float(v)
        except ValueError:
            params[k] = v
    rows = []
    for g in a.groups.split(','):
        r = run_group(g, params or None, baseline=a.baseline, strain_ev=a.strain_ev,
                      fusion=a.fusion)
        plot(r); rows.append(metrics(r))
        print(f'  [{g}] ' + str(rows[-1]))
    pd.DataFrame(rows).to_csv(os.path.join(RES, 'l1_degree.csv'),
                              index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
