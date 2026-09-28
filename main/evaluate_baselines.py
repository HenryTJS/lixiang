# -*- coding: utf-8 -*-
"""基线对照：AE 单源口径下，本项目方法 vs 经典/朴素基线 + 监督 ML 基线

为什么要做
----------
「我们的方法能用」≠「比现有方法好」。没有对照表，论文的核心主张不成立。
本脚本给出**同口径、同预警判据、同评估指标**的对照，并做两件事避免
"稻草人基线"指责：

  1. **阈值 oracle 扫描**：阈值型基线在网格上扫一遍，报**最佳 |err|**及其阈值
     （即"事后挑最好阈值"），仍不达标才算真不行；
  2. **消除跨试件尺度差**：所有指标按各组**自身校准段**（块 [20,45)，与本项目
     `stiff_cal`/`shape_cal` 同窗口）归一 → 跨试件阈值才有意义；
     ML 基线额外给一个「逐试件因果归一 + log」的特征版本。

对照对象
--------
  ours_off   : OnlineDamageIndex 正式口径（逐点不可逆规则，与交付一致）
  ours_blk   : 同上，但改用与基线**完全一致**的块级不可逆规则
  ours_ae_blk: 同上，但只用声发射（单源口径，公平比）
  B1 累积 AE 能量   h = cumsum(块内 Σpeak²) / 校准段 cumsum 中位
  B2 块能量 EWMA     h = EWMA_α(块内 Σpeak²) / 校准段块能量中位
  B3 累积 AE 事件数  h = cumsum(块内事件数) / 校准段 cumsum 中位
  B4raw  监督 ML  25 列块均值 + log(Σpeak²) + 事件数 → LogisticRegression，LOSO
  B4norm 监督 ML  同上，但特征改为 逐试件校准段中位归一 + log10

⚠️ **不对称须写明**：B4 在训练时**看过 b2 标签**，本项目方法**从不看标签**。
   B4 赢是正常的；B4 赢不了才是强结论。

预警判据（块级，与本项目逐点规则同构）
  「首次不可逆越阈」= h ≥ level，且其后 hold 块内不回落至 level×(1−drop_frac) 以下。
  hold = max(4, 2% 块数)（对应逐点的 max(2000 点, 2% 寿命)），
  drop_frac = DROP/LOW = 0.15/0.30 = 0.5（允许回落一半）。

输出：results/baseline_compare.csv（长表）+ 控制台汇总
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from shm.streaming import StreamSimulator                        # noqa: E402
from shm.damage_index import OnlineDamageIndex                   # noqa: E402
from shm.config import EXT_BLOCK_PTS                             # noqa: E402
from eval_common import ref_map, onset_of, LOW, DROP, HOLD_FRAC  # noqa: E402

GROUPS = ['016', '017', '018', '019', '020']       # 有 b2/b3 标签
NOB2 = ['023', '024', '025', '026']                 # 同台架同协议，**无标签** → 只看是否报警
NOLABEL = ['021', '022'] + NOB2                     # 其余无标签组（含问题组 021）
ALL_GROUPS = GROUPS + NOLABEL
CAL_LO, CAL_HI = 20, 45          # 校准段块号（与本项目 stiff/shape 同窗口）
ALPHA = 0.3                      # EWMA 系数（与 e_dmg 的 0.7/0.3 同构）
DROP_FRAC = DROP / LOW           # 0.5
N_SPLIT = 6                      # ML 的 LOOCV 折数（= 试件数）
EPS = 1e-12
OUT = os.path.join('results', 'baseline_compare.csv')


# ---------------------------------------------------------------- 工具
def _ewma(a, alpha=ALPHA):
    out = np.empty_like(np.asarray(a, dtype=float))
    s = 0.0
    for i, v in enumerate(a):
        s = s + alpha * (v - s)
        out[i] = s
    return out


def _cal_med(a):
    """校准段中位（因果参考：只用校准段）。"""
    a = np.asarray(a, dtype=float)
    w = a[CAL_LO:CAL_HI] if len(a) > CAL_HI else a
    w = w[np.isfinite(w)]
    return float(np.median(w)) if len(w) else 0.0


def warn_blk(h, level, hold):
    """块级「首次不可逆越阈」→ 寿命% 或 None（与 onset_of 同构）。"""
    h = np.asarray(h, dtype=float)
    n = len(h)
    drop = level * (1.0 - DROP_FRAC)
    i = 0
    while i < n:
        if h[i] >= level:
            j = min(n - 1, i + hold)
            seg = h[i:j + 1]
            if seg.min() >= drop:
                return 100.0 * i / n
            below = np.where(seg < drop)[0]
            i = i + (int(below[0]) if len(below) else (j - i + 1))
        else:
            i += 1
    return None


def grade_of(tw, b2):
    if tw is None:
        return 'C'
    if b2 is None:
        return '-'          # 无标签组不计分级
    e = tw - b2
    return 'E' if e < -15 else ('D' if e > 15 else 'A')


# ---------------------------------------------------------------- 逐组特征
def group_features(gid):
    """一次流式 → 块级原始量 + 两组逐点 D（正式口径 / AE 单源口径）。"""
    sim = StreamSimulator(gid)
    n = sim.load_data()
    cols = list(sim.ae_cols)
    di_full = OnlineDamageIndex()                     # 正式（多源默认）
    di_ae = OnlineDamageIndex({'abl': 'no_strain'})   # AE 单源
    ncol = len(cols)
    E_blk, N_blk, F_blk = [], [], []
    Dfull, Dae = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
    bpts, bE, bN = 0, 0.0, 0
    bsum = np.zeros(ncol)
    for idx in range(n):
        p = sim.next_point()
        strain = p['strain']
        pk = None
        sv = None
        if p.get('ae_new') and p.get('ae'):
            pk = float(p['ae'].get('ae_Peak', 0.0) or 0.0)
            sv = di_full.shape_value(p['ae'])
            for c_i, c in enumerate(cols):
                v = p['ae'].get(c)
                if v is not None and np.isfinite(v):
                    bsum[c_i] += float(v)
        Dfull[idx] = di_full.update(strain, pk, None, sv)
        Dae[idx] = di_ae.update(strain, pk, None, di_ae.shape_value(p.get('ae')))
        if pk is not None:
            bE += pk ** 2
            bN += 1
        bpts += 1
        if bpts >= EXT_BLOCK_PTS:
            E_blk.append(bE)
            N_blk.append(bN)
            F_blk.append(bsum / bN if bN else np.zeros(ncol))
            bpts, bE, bN = 0, 0.0, 0
            bsum = np.zeros(ncol)
    sim.cleanup()
    nb = len(E_blk)
    ends = np.minimum((np.arange(1, nb + 1) * EXT_BLOCK_PTS) - 1, n - 1)
    return dict(gid=gid, n=n, nb=nb, cols=cols,
                E=np.asarray(E_blk, float), N=np.asarray(N_blk, float),
                F=np.vstack(F_blk), Dfull=Dfull[ends], Dae=Dae[ends],
                Dfull_pt=Dfull, Dae_pt=Dae)


# ---------------------------------------------------------------- 指标
def rows_for(method, level, g, h, b2, b3, hold):
    tw = warn_blk(h, level, hold)
    err = (tw - b2) if (tw is not None and b2 is not None) else None
    lead = (b3 - tw) if (tw is not None and b3 is not None) else None
    return dict(method=method, level=float(level), gid=g['gid'],
                t_warn=tw, err=err, lead=lead, grade=grade_of(tw, b2))


def summarize(df):
    """只在**有标签组**上聚合（无标签组 err/grade 无意义）。"""
    d = df[df['gid'].isin(GROUPS)]
    out = []
    for (m, lv), s in d.groupby(['method', 'level'], sort=False, dropna=False):
        e = s['err'].dropna()
        out.append(dict(method=m, level=lv, nA=int((s['grade'] == 'A').sum()),
                        nE=int((s['grade'] == 'E').sum()),
                        nD=int((s['grade'] == 'D').sum()),
                        nC=int((s['grade'] == 'C').sum()),
                        mean_abs_err=float(e.abs().mean()) if len(e) else np.nan,
                        mean_lead=float(s['lead'].dropna().mean())
                        if s['lead'].notna().any() else np.nan))
    return pd.DataFrame(out)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    refs = ref_map()
    data = {}
    for gid in ALL_GROUPS:
        data[gid] = group_features(gid)
        print(f'  特征完成 {gid}  块数={data[gid]["nb"]}')

    # ---- 逐方法 × 逐阈值 ----
    all_rows = []
    for gid in ALL_GROUPS:
        g = data[gid]
        nb = g['nb']
        b2, b3 = refs.get(gid, (None, None))
        hold = max(4, int(round(0.02 * nb)))
        cum_E, cum_N = np.cumsum(g['E']), np.cumsum(g['N'])
        rE, rN = max(_cal_med(cum_E), EPS), max(_cal_med(cum_N), EPS)
        rB = max(_cal_med(g['E']), EPS)
        hB1, hB2 = cum_E / rE, _ewma(g['E']) / rB
        hB3 = cum_N / rN
        hOurs, hOursAE = g['Dfull'], g['Dae']

        grids = [('B1_累积能量', hB1, np.geomspace(1.5, 50, 40)),
                 ('B2_块能量EWMA', hB2, np.geomspace(1.5, 200, 40)),
                 ('B3_累积事件数', hB3, np.geomspace(1.5, 50, 40)),
                 ('ours_off_blk', hOurs, np.arange(0.20, 0.96, 0.01)),
                 ('ours_ae_blk', hOursAE, np.arange(0.20, 0.96, 0.01))]
        for name, h, grid in grids:
            for lv in grid:
                all_rows.append(rows_for(name, lv, g, h, b2, b3, hold))
        # ours 正式逐点规则（交付口径，非块级）
        tw = onset_of(g['Dfull_pt'], max(2000, int(nb * EXT_BLOCK_PTS * HOLD_FRAC)))
        all_rows.append(dict(method='ours_off_pt', level=np.nan, gid=gid,
                             t_warn=tw,
                             err=(tw - b2) if (tw is not None and b2 is not None) else None,
                             lead=(b3 - tw) if (tw is not None and b3 is not None) else None,
                             grade=grade_of(tw, b2)))
        tw2 = onset_of(g['Dae_pt'], max(2000, int(nb * EXT_BLOCK_PTS * HOLD_FRAC)))
        all_rows.append(dict(method='ours_ae_pt', level=np.nan, gid=gid,
                             t_warn=tw2,
                             err=(tw2 - b2) if (tw2 is not None and b2 is not None) else None,
                             lead=(b3 - tw2) if (tw2 is not None and b3 is not None) else None,
                             grade=grade_of(tw2, b2)))

    # ---- B4 监督 ML（训练见过 b2 标签；有标签组走 LOOCV，无标签组用全标签训练后预测）----
    F_all = np.vstack([data[g]['F'] for g in ALL_GROUPS])
    norm_blocks = []
    for gid in ALL_GROUPS:
        g = data[gid]
        base = np.array([_cal_med(g['F'][:, j]) for j in range(g['F'].shape[1])])
        base = np.where(np.abs(base) < EPS, np.nan, base)
        r = g['F'] / base
        r = np.where(np.isfinite(r) & (r > 0), np.log10(np.maximum(r, 1e-6)), np.nan)
        norm_blocks.append(r)
    Xnorm = np.hstack([np.nan_to_num(np.vstack(norm_blocks), nan=0.0),
                       np.vstack([np.column_stack([np.log10(1 + data[g]['E']), data[g]['N']])
                                  for g in ALL_GROUPS])])
    Xraw = np.hstack([F_all,
                      np.vstack([np.column_stack([np.log10(1 + data[g]['E']), data[g]['N']])
                                 for g in ALL_GROUPS])])
    gid_vec, y = [], []
    for gid in ALL_GROUPS:
        nb = data[gid]['nb']
        b2 = refs.get(gid, (None, None))[0]
        y += [1 if (b2 is not None and 100.0 * k / nb >= b2) else 0 for k in range(nb)]
        gid_vec += [gid] * nb
    gid_vec, y = np.asarray(gid_vec), np.asarray(y)
    labelled = np.isin(gid_vec, GROUPS)

    for tag, X in (('B4raw', Xraw), ('B4norm', Xnorm)):
        prob = np.zeros(len(y))
        for gid in GROUPS:                      # 有标签组 → LOOCV
            te = gid_vec == gid
            tr = labelled & ~te
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        for gid in NOB2:                        # 无标签组 → 用全标签训练后外推
            te = gid_vec == gid
            sc = StandardScaler().fit(X[labelled])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(
                sc.transform(X[labelled]), y[labelled])
            prob[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        k0 = 0
        for gid in ALL_GROUPS:
            nb = data[gid]['nb']
            h = _ewma(prob[k0:k0 + nb])
            k0 += nb
            b2, b3 = refs.get(gid, (None, None))
            hold = max(4, int(round(0.02 * nb)))
            for lv in np.arange(0.30, 0.99, 0.02):
                all_rows.append(rows_for(tag, lv, data[gid], h, b2, b3, hold))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT, index=False, float_format='%.4f')
    summ = summarize(df)

    # ---- 汇总：每方法 oracle 最佳阈值（优先 nC=0，再 mean|err| 最小）----
    print(f'\n{"=" * 96}\n每方法「oracle 阈值」最佳结果（事后挑最好阈值，给基线最好机会）\n{"=" * 96}')
    print(f'{"方法":<16}{"阈值":>10}{"nA":>4}{"nE":>4}{"nD":>4}{"nC":>4}'
          f'{"mean|err|":>11}{"mean lead":>11}')
    best = []
    for m in sorted(summ['method'].unique()):
        s = summ[summ['method'] == m].copy()
        s['pen'] = s['mean_abs_err'].fillna(999) + 100.0 * s['nC']
        b = s.sort_values(['pen', 'mean_abs_err']).iloc[0]
        best.append(b)
        lv = 'na' if not np.isfinite(b['level']) else f"{b['level']:.3f}"
        print(f'{m:<16}{lv:>10}{int(b["nA"]):>4}{int(b["nE"]):>4}{int(b["nD"]):>4}'
              f'{int(b["nC"]):>4}{b["mean_abs_err"]:>11.2f}{b["mean_lead"]:>11.1f}')
    pd.DataFrame(best).to_csv(os.path.join('results', 'baseline_best.csv'), index=False)

    # ---- 无标签组（023-026）：各方法在「自身最佳阈值」下是否报警 ----
    print(f'\n{"=" * 96}\n无标签同协议组（023-026）泛化对照（各方法用上表选定的最佳阈值）\n{"=" * 96}')
    hdr = f'{"gid":<6}' + ''.join(f'{b["method"]:>16}' for b in best)
    print(hdr)
    for gid in NOB2:
        line = f'{gid:<6}'
        for b in best:
            s = df[(df['method'] == b['method']) & (df['gid'] == gid)]
            if np.isfinite(b['level']):
                s = s[np.isclose(s['level'], b['level'])]
            tw = s['t_warn'].iloc[0] if len(s) else None
            line += f'{("%.1f" % tw) if tw is not None else "未报警":>16}'
        print(line)
    # ---- 表 3：全组·无标签统一协议（用可计算的物理锚 f_est）----
    # f_est = 块能量峰值所在块位(%寿命)。**与 016-020 的 b3(=99.0) 互相校验**：
    #   f_est 为 95.0/100/100/98.7/100 → 与记录末尾一致（016 的余段 5% 已由用户确认为正常）。
    # 判定（展示约定，非调参）：漏报 = 未报警；过早 = lead > 50 个百分点；否则合理。
    print(f'\n{"=" * 104}\n表 3 · 全组·无标签统一协议（f_est = 块能量峰值位置；lead = f_est - t_warn）\n{"=" * 104}')
    fest = {}
    for gid in ALL_GROUPS:
        g = data[gid]
        E, nb = g['E'], g['nb']
        k = int(np.argmax(E))
        fest[gid] = (100.0 * (k + 1) / nb,
                     float(E[int(nb * 0.95):].sum() / max(E.sum(), EPS) * 100.0))
    methods3 = [b['method'] for b in best]
    print(f'{"gid":<5}{"f_est%":>7}{"末5%能量":>9}' +
          ''.join(f'{m[:13]:>15}' for m in methods3))
    for gid in ALL_GROUPS:
        f, tail = fest[gid]
        line = f'{gid:<5}{f:>7.1f}{tail:>9.1f}'
        for m in methods3:
            b = [x for x in best if x['method'] == m][0]
            s = df[(df['method'] == m) & (df['gid'] == gid)]
            if np.isfinite(b['level']):
                s = s[np.isclose(s['level'], b['level'])]
            tw = s['t_warn'].iloc[0] if len(s) else None
            if tw is None:
                line += f'{"漏报":>15}'
            else:
                lead = f - tw
                tag = '过早' if lead > 50 else '合理'
                line += f'{f"{tw:.1f}({lead:+.0f}){tag}":>15}'
        mark = '  ← 问题组' if gid == '021' else ('' if gid in GROUPS else '')
        print(line + mark)
    print('\n注：括号内为 lead（百分点，正=报警早于 f_est）。f_est 对 016-020 与 b3(=99.0) 一致，'
          '故 016-020 的该列可当作标签无关的复核。')
    print('    · f_est 为可计算的物理锚（AE 块能量峰值位置），对无 b2/b3 的组也适用。')

    print(f'\nsaved {os.path.abspath(OUT)}')


if __name__ == '__main__':
    main()
