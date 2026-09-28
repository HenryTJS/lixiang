# -*- coding: utf-8 -*-
"""级别层异源分级：AE 决定检测级(L1) + 刚度损失率作为 L2/L3 闸门。

设计:
  risk / D / L1  ← 声发射单源（已实测最优，5/5 精准、误差 3.0%）
  stiff_loss     ← 应变幅值相对基线增长率（载荷控制下 = 刚度损失率）
  L2 判定 = D≥0.55 且 stiff_loss ≥ θ2      （闸门）
  L3 判定 = D≥0.85 且 stiff_loss ≥ θ3      （闸门）

做法: 每组只跑一次（sources=('ae',), strain_mode='stiff', 闸门关），
      记录逐点 [damage, stiff_loss, e_ae, e_strain]，再离线扫 θ2/θ3。
      这样"有效级别时刻" = max(D 达阈时刻, 刚度达阈时刻)，无需反复重跑。

输出:
  main/cache/_grade_cache/<gid>.npy
  main/results/grade_stiff_traj.csv     逐组刚度损失轨迹与越界时刻
  main/results/grade_levels.csv         逐组有效级别时刻(扫 θ2×θ3)
  main/results/grade_summary.csv        阈值组合覆盖率汇总
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
os.chdir(ROOT)

import eval_common as ec                            # noqa: E402
from shm.streaming import StreamSimulator           # noqa: E402
from shm.damage_index import OnlineDamageIndex      # noqa: E402
from shm.config import EXT_BLOCK_PTS, GRADE_GATE   # noqa: E402

CACHE = os.path.join(ROOT, 'cache', '_grade_cache')
RESDIR = os.path.join(ROOT, 'results')

TH_STIFF = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
GRID2 = [0.10, 0.15]
GRID3 = [0.30, 0.50]
SUSTAIN = 3
WARM_BLK = 10          # 从第 10 块起开始找越界(避开开机瞬态)


def run_one(gid, force=False):
    """跑一组(AE 单源 + 刚度损失率)，返回逐点 [damage, stiff_loss, e_ae, e_strain]。"""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, gid + '.npy')
    if os.path.exists(p) and not force:
        return np.load(p)
    di = OnlineDamageIndex({'sources': ('ae',), 'strain_mode': 'stiff'})
    sim = StreamSimulator(gid)
    sim.load_data()
    rows = []
    while sim.has_next():
        q = sim.next_point()
        st = q['strain']
        pk = None
        if q.get('ae_new') and q.get('ae'):
            pk = q['ae'].get('ae_Peak', 0.0) or 0.0
        di.update(st, float(pk) if pk is not None else None, q.get('fo'),
                  di.shape_value(q.get('ae')))
        rows.append((di.damage, di.stiff_loss, di._last_e_ae, di._last_e_strain))
    sim.cleanup()
    arr = np.array(rows, dtype=np.float32)
    np.save(p, arr)
    return arr


def blk_series(arr_col):
    """逐点 → 逐块(取每块结算后的值)。"""
    return arr_col[EXT_BLOCK_PTS - 1::EXT_BLOCK_PTS]


def first_sustained(v, nb, th, warm=WARM_BLK, n=SUSTAIN):
    """首个 life%(连续 n 块 ≥ th)。"""
    for i in range(warm, nb - n + 1):
        seg = v[i:i + n]
        if (seg >= th).all():
            return 100.0 * i / nb
    return None


def first_ge_block(v, nb, th):
    idx = [i for i in range(nb) if np.isfinite(v[i]) and v[i] >= th]
    return 100.0 * idx[0] / nb if idx else None


def run_gated(gid, force=False):
    """按 GRADE_GATE 逐组口径跑(含级别闸门) → 逐点 [damage, level, stiff_loss]。"""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, gid + '_gated.npy')
    if os.path.exists(p) and not force:
        return np.load(p)
    params = dict(GRADE_GATE.get(gid, {}))
    params['sources'] = ('ae',)
    params.setdefault('strain_mode', 'stiff')     # 仍算 stiff_loss 供诊断
    di = OnlineDamageIndex(params)
    sim = StreamSimulator(gid)
    sim.load_data()
    rows = []
    while sim.has_next():
        q = sim.next_point()
        st = q['strain']
        pk = None
        if q.get('ae_new') and q.get('ae'):
            pk = q['ae'].get('ae_Peak', 0.0) or 0.0
        di.update(st, float(pk) if pk is not None else None, q.get('fo'),
                  di.shape_value(q.get('ae')))
        rows.append((di.damage, di.level, di.stiff_loss))
    sim.cleanup()
    arr = np.array(rows, dtype=np.float32)
    np.save(p, arr)
    return arr


def first_level(lv, n, k):
    """级别首次 ≥ k 的 life%。"""
    idx = np.where(lv >= k)[0]
    return float(idx[0]) / n * 100.0 if len(idx) else None


def main():
    os.makedirs(RESDIR, exist_ok=True)
    traj_rows, lvl_rows = [], []
    per_group = {}

    for gid in ec.GROUPS:
        arr = run_one(gid)
        d_pt = arr[:, 0]
        st_pt = arr[:, 1]
        n = len(arr)
        d_b = blk_series(d_pt)
        s_b = blk_series(st_pt)
        nb = len(d_b)

        m = ec.per_group_metrics(gid, d_pt)
        t25, t55, t85 = m['t25'], m['t55'], m['t85']
        t_l1 = m['t_warn']

        # --- 刚度损失轨迹 ---
        e = np.linspace(0, nb, 11).astype(int)
        seg10 = np.array([np.nanmean(s_b[e[k]:e[k + 1]]) for k in range(10)])
        xrow = {'gid': gid, 'nblk': nb, 'stiff_end': float(s_b[-1]),
                'stiff_max': float(np.nanmax(s_b)), 't_L1_ae': t_l1,
                't25': t25, 't55': t55, 't85': t85, 'grade': m['grade'],
                'err': m['err']}
        for th in TH_STIFF:
            x = first_sustained(s_b, nb, th)
            xrow['x_%.2f' % th] = x
        traj_rows.append(xrow)
        per_group[gid] = dict(d_b=d_b, s_b=s_b, nb=nb, t25=t25, t55=t55,
                              t85=t85, t_l1=t_l1, seg10=seg10)

    # ---- 打印轨迹 ----
    print('=' * 100)
    print('A) 刚度损失率轨迹 (10 段均值) 与越界时刻 [life%%], 连续 %d 块' % SUSTAIN)
    for r in traj_rows:
        print('  %s  nblk=%3d  stiff_end=%.2f max=%.2f' %
              (r['gid'], r['nblk'], r['stiff_end'], r['stiff_max']))
        a = ['%+.2f' % v for v in per_group[r['gid']]['seg10']]
        print('       seg10  ' + ' '.join('%6s' % v for v in a))
        print('       x(θ)   ' + ' '.join('%6s' % ('%.2f' % t)
                                          for t in TH_STIFF))
        print('       life%  ' + ' '.join(
            '%6s' % ('--' if r['x_%.2f' % t] is None else '%.1f' % r['x_%.2f' % t])
            for t in TH_STIFF))
        print('       AE 检测 t_L1=%.1f  D: t25=%.1f t55=%.1f t85=%.1f  %s'
              % (r['t_L1_ae'] or np.nan, r['t25'], r['t55'], r['t85'], r['grade']))
        print()

    # ---- 扫 θ2 × θ3 ----
    print('=' * 100)
    print('B) 有效级别时刻 = max(D 达阈, 刚度达阈)  [life%]')
    for th2 in GRID2:
        for th3 in GRID3:
            print('--- θ2=%.2f (L2)  θ3=%.2f (L3) ---' % (th2, th3))
            print('  gid   tL1(AE)   t25    tL2eff   Gap12    t85    tL3eff   Gap23')
            for gid in ec.GROUPS:
                g = per_group[gid]
                nb, s_b = g['nb'], g['s_b']
                x2 = first_sustained(s_b, nb, th2)
                x3 = first_sustained(s_b, nb, th3)
                tL2 = max([t for t in (g['t55'], x2) if t is not None]) \
                    if (g['t55'] is not None or x2 is not None) else None
                if x2 is None:
                    tL2 = None
                tL3 = None
                if x3 is not None and g['t85'] is not None:
                    tL3 = max(g['t85'], x3, tL2 if tL2 is not None else 0.0)
                gap12 = (tL2 - g['t_l1']) if (tL2 is not None and g['t_l1']) else None
                gap23 = (tL3 - tL2) if (tL3 is not None and tL2 is not None) else None
                lvl_rows.append(dict(th2=th2, th3=th3, gid=gid,
                                     t_L1=g['t_l1'], t25=g['t25'], t55=g['t55'],
                                     tL2_eff=tL2, gap12=gap12,
                                     t85=g['t85'], tL3_eff=tL3, gap23=gap23))
                f = lambda v: '  --  ' if v is None else '%6.1f' % v
                print('  %s  %s %s %s %s %s %s %s'
                      % (gid, f(g['t_l1']), f(g['t25']), f(tL2), f(gap12),
                         f(g['t85']), f(tL3), f(gap23)))
            print()

    # ---- D) 最终口径：逐组级别时刻（含 GRADE_GATE 闸门） ----
    print('=' * 100)
    print('D) 最终口径逐组级别时刻 [life%%]  (GRADE_GATE 逐组闸门)')
    print('  gid   闸门          L1检测    L2预警   L1→L2    L3临危   L2→L3')
    fin_rows = []
    for gid in ec.GROUPS:
        a = run_gated(gid)
        n = len(a)
        lv = a[:, 1]
        gate = GRADE_GATE.get(gid, {})
        t1 = first_level(lv, n, 1)
        t2 = first_level(lv, n, 2)
        t3 = first_level(lv, n, 3)
        g12 = (t2 - t1) if (t1 is not None and t2 is not None) else None
        g23 = (t3 - t2) if (t2 is not None and t3 is not None) else None
        tag = ('θ2=%.2f θ3=%.2f' % (gate['lvl2_stiff'], gate['lvl3_stiff'])) \
            if gate else 'off'
        fin_rows.append(dict(gid=gid, gate=tag, t_L1=t1, t_L2=t2, gap12=g12,
                             t_L3=t3, gap23=g23))
        f = lambda v: '  --  ' if v is None else '%6.1f' % v
        print('  %s   %-12s %s %s %s %s %s'
              % (gid, tag, f(t1), f(t2), f(g12), f(t3), f(g23)))
    pd.DataFrame(fin_rows).to_csv(
        os.path.join(RESDIR, 'grade_levels_final.csv'),
        index=False, encoding='utf-8-sig')
    print()

    df = pd.DataFrame(traj_rows)
    df.to_csv(os.path.join(RESDIR, 'grade_stiff_traj.csv'),
              index=False, encoding='utf-8-sig')
    dl = pd.DataFrame(lvl_rows)
    dl.to_csv(os.path.join(RESDIR, 'grade_levels.csv'),
              index=False, encoding='utf-8-sig')

    # ---- 覆盖率汇总 ----
    print('=' * 100)
    print('C) 覆盖率汇总')
    rows = []
    for th2 in GRID2:
        for th3 in GRID3:
            sub = dl[(dl['th2'] == th2) & (dl['th3'] == th3)]
            nL2 = int(sub['tL2_eff'].notna().sum())
            nL3 = int(sub['tL3_eff'].notna().sum())
            g12 = sub['gap12'].dropna()
            g23 = sub['gap23'].dropna()
            rows.append(dict(th2=th2, th3=th3, nL2=nL2, nL3=nL3,
                             gap12_med=float(g12.median()) if len(g12) else np.nan,
                             gap23_med=float(g23.median()) if len(g23) else np.nan,
                             gap12_min=float(g12.min()) if len(g12) else np.nan,
                             gap23_min=float(g23.min()) if len(g23) else np.nan))
    sdf = pd.DataFrame(rows)
    sdf.to_csv(os.path.join(RESDIR, 'grade_summary.csv'),
               index=False, encoding='utf-8-sig')
    print(sdf.to_string(index=False))
    print()
    print('wrote: grade_stiff_traj.csv / grade_levels.csv / grade_summary.csv')
    print('wrote: grade_levels_final.csv')


if __name__ == '__main__':
    main()
