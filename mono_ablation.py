# -*- coding: utf-8 -*-
"""物理约束层（保序投影）在数据集 A / B / C 上的消融对照

要回答的问题
------------
保序投影（单调不减 + 非负）在 phmdc（数据集 D）上让 **4 个特征臂 + 10 个 DL 配置
全部** RMSE 下降、折内 rho 上升、单调违例归零（`phmdc/README.md` §5.2.2 ③）。

它在 **A 主样本 / B L1 一批 / C L1 二批** 上是否**同样**有效？
D(t) 本身就是"升快降慢"的累积量，单调性天然较强 ⇒ **收益可能接近 0**。
本脚本就是去量这个收益到底有多大（以及方向是好是坏）。

⚠️ 三条必须先说清的口径
------------------------
1. **因果性**：投影必须只用"当前时刻及之前"的信息。故主用
   `causal_mono(D) = clip(running_max(D), 0, 1)` —— 它可在流式管线里逐点执行。
   另跑一遍**离线** PAVA（用整条曲线 → 用了未来信息）作为**对照**，
   用来量化"如果不守在线语义、能多拿多少"。**只有 causal 那一列可对外声称可部署。**
2. **投影不是"白白变好"**：running_max 只会让曲线**抬升**，因此所有阈值穿越时刻
   **只会提前、不会推后**。若原始 D 本来就偏早，投影会把"过早预警"变得更早
   ⇒ 所以本脚本同时报 `err`（与 b2 锚之差）**投影前后**的变化，看方向。
3. **单调 ≠ 正确**：真实损伤若存在卸载/修复，强制单调就是**错的**。
   A/B/C 上没有修复工况，所以这里只讨论"抖动抑制"，不宣称"单调是普适物理"。

数据来源
--------
- A 主样本 016–020：`main/eval_common.run_one`（默认参数 + `SHAPE_DEFAULTS`，读缓存）
  锚：`main/weak_labels/labels_summary.csv` 的 b2 / b3；另算 `onset_of` 的 A-预警时刻。
- B L1 一批 L1-03/04/05/09：`l1/evaluate_l1_degree.run_group(..., baseline=True,
  strain_ev=True, fusion='max', params={'rise':0.05})` —— 与交付口径一致。
  锚：`n_f`（失效循环）。
- C L1 二批 9 组：`l1/evaluate_l1_hi_ae.hi_of_group` 的 `cum_hits` → `unity01`
  （= 交付用的离线复评 HI_AE）。**构造上单调** ⇒ 作为**阴性对照**：
  收益必须恰好为 0，用来证明本脚本的度量不是"总能测出点东西"。

输出
----
  results/mono_ablation_groups.csv    逐组 × 投影前/后/离线 的全部指标
  results/mono_ablation_summary.csv   逐数据集汇总
  results/mono_ablation_report.txt

依赖：numpy / pandas
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

PROJ = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(PROJ, 'main')
L1 = os.path.join(PROJ, 'l1')
RES = os.path.join(PROJ, 'results')
os.makedirs(RES, exist_ok=True)

for p in (PROJ, MAIN, L1):
    if p not in sys.path:
        sys.path.insert(0, p)

_REPORT = []


def say(msg=''):
    print(msg, flush=True)
    _REPORT.append(msg)


# ============================================================
# 投影算子
# ============================================================
def causal_mono(d, lo=0.0, hi=1.0):
    """**因果**保序投影：D_mono[t] = max(D[0..t])，再钳到 [lo, hi]。

    只用当前时刻及之前的信息 ⇒ 可在线逐点执行，零未来信息。
    效果：非单调点被"抬"到历史最大值，因此所有阈值穿越只会**提前或不变**。
    """
    x = np.clip(np.asarray(d, float), lo, hi)
    return np.clip(np.maximum.accumulate(x), lo, hi)


def pava_up(d):
    """**离线**保序回归（PAVA，全曲线），仅作对照 —— 它用了未来信息，不可部署。"""
    v = np.clip(np.asarray(d, float), 0.0, 1.0)
    if len(v) <= 1:
        return v
    blocks = []
    for i in range(len(v)):
        blocks.append([i, i, float(v[i])])
        while len(blocks) > 1:
            a, b = blocks[-2], blocks[-1]
            if a[2] / (a[1] - a[0] + 1) <= b[2] / (b[1] - b[0] + 1):
                break
            blocks.pop()
            blocks.pop()
            blocks.append([a[0], b[1], a[2] + b[2]])
    out = np.empty(len(v))
    for a, b, s in blocks:
        out[a:b + 1] = s / (b - a + 1)
    return out


# ============================================================
# 指标
# ============================================================
def first_ge(life, d, th):
    """D 首次 ≥ th 的寿命% ；未达返回 nan。"""
    idx = np.where(np.asarray(d) >= th)[0]
    if not len(idx):
        return np.nan
    return float(np.asarray(life)[idx[0]])


def metrics(life, d, tag):
    d = np.asarray(d, float)
    out = {'variant': tag, 'n': int(len(d)), 'D_end': float(d[-1]) if len(d) else np.nan}
    if len(d) > 1:
        dd = np.diff(d)
        out['viol_pct'] = float(np.mean(dd < -1e-9)) * 100.0
    else:
        out['viol_pct'] = np.nan
    run = np.maximum.accumulate(d) if len(d) else d
    out['max_drawdown'] = float(np.max(run - d)) if len(d) else np.nan
    # "抖动量"：一阶差分的绝对值和 / 值域 —— 与单调投影的收益直接相关
    if len(d) > 1:
        rng = float(d.max() - d.min())
        out['tv_over_range'] = (float(np.abs(np.diff(d)).sum()) / rng) if rng > 1e-12 else np.nan
    else:
        out['tv_over_range'] = np.nan
    for th in (0.25, 0.55, 0.85):
        out['t%02d' % int(round(th * 100))] = first_ge(life, d, th)
    return out


def warn_onset(d, hold_frac=0.02, low=0.30, drop=0.15):
    """A-预警不可逆 onset（寿命%），口径同 main/eval_common.onset_of。"""
    d = np.asarray(d, float)
    n = len(d)
    low_th, drop_th = low, low - drop
    hold = max(2000, int(n * hold_frac))
    i = 0
    while i < n:
        if d[i] >= low_th:
            j = min(n - 1, i + hold)
            seg = d[i:j + 1]
            if seg.min() >= drop_th:
                return float(i) / n * 100.0
            below = np.where(seg < drop_th)[0]
            i = i + (int(below[0]) if len(below) else (j - i + 1))
        else:
            i += 1
    return np.nan


# ============================================================
# 三个数据集
# ============================================================
def groups_A():
    """主样本 016–020：返回 [(gid, life%, D, 锚 dict)]。"""
    import eval_common as E

    out = []
    for gid in E.GROUPS:
        d = E.run_one((gid, {}))[1].astype(float)
        life = np.arange(len(d)) / max(len(d) - 1, 1) * 100.0
        b2, b3 = E.ref_map().get(gid, (np.nan, np.nan))
        out.append((gid, life, d, {'b2': float(b2), 'b3': float(b3),
                                   'unit': '点(等间隔)'}))
    return out


def groups_B():
    """L1 一批：与交付口径一致（baseline + strain 证据 + max 融合）。"""
    import evaluate_l1_degree as L

    out = []
    for gid in L.META:
        r = L.run_group(gid, params={'rise': 0.05, 'fall': 0.008},
                        baseline=True, strain_ev=True, fusion='max')
        nf = float(L.META[gid]['n_f'])
        life = r['cyc'] / nf * 100.0
        out.append((gid, life, np.asarray(r['D'], float),
                    {'n_f': nf, 'b2': np.nan, 'b3': 100.0,
                     'refs': L.META[gid].get('refs', [])}))
    return out


def groups_C():
    """L1 二批：交付用的离线复评 HI_AE = unity01(累积事件数)，构造上单调（阴性对照）。"""
    import evaluate_l1_hi_ae as H

    out = []
    for gid in H.GROUPS_V2:
        r = H.hi_of_group(gid)
        if r is None:
            continue
        nf = float(r['n_f'])
        life = r['cyc'] / nf * 100.0
        hi = np.asarray(H.unity01(r['cum_hits']), float)
        out.append((gid, life, hi, {'n_f': nf, 'b2': np.nan, 'b3': 100.0,
                                    'refs': []}))
    return out


# ============================================================
# 主流程
# ============================================================
def run_set(name, items):
    rows = []
    say('')
    say('=' * 104)
    say('[%s] %d 组' % (name, len(items)))
    say('=' * 104)
    say('  %-7s %-7s %8s %8s %9s %9s %9s %9s %9s'
        % ('试件', '变体', '违例%', '最大回撤', 't25', 't55', 't85', 'D_end', 'TV/值域'))
    say('  ' + '-' * 96)
    for gid, life, d, anc in items:
        variants = [('原始', d), ('因果投影', causal_mono(d)), ('离线PAVA', pava_up(d))]
        rec = {'set': name, 'gid': gid}
        rec.update({('anchor_' + k): v for k, v in anc.items() if k != 'refs'})
        if name == 'A':
            rec['warn_原始'] = warn_onset(d)
            rec['warn_因果投影'] = warn_onset(causal_mono(d))
            rec['warn_离线PAVA'] = warn_onset(pava_up(d))
        for tag, dd in variants:
            m = metrics(life, dd, tag)
            rec.update({('%s_%s' % (tag, k)): v for k, v in m.items()
                        if k not in ('variant', 'n')})
            say('  %-7s %-7s %8.2f %8.3f %9s %9s %9s %9.4f %9.2f'
                % (gid, tag, m['viol_pct'], m['max_drawdown'],
                   ('%.1f' % m['t25']) if np.isfinite(m['t25']) else '—',
                   ('%.1f' % m['t55']) if np.isfinite(m['t55']) else '—',
                   ('%.1f' % m['t85']) if np.isfinite(m['t85']) else '—',
                   m['D_end'], m['tv_over_range']))
        rows.append(rec)
        say('  ' + '-' * 96)
    return pd.DataFrame(rows)


def summarize(df, name):
    """逐数据集汇总：投影带来的"提前量"与"违例消除"。"""
    if not len(df):
        return None
    r = {'set': name, 'n_groups': len(df)}
    for v in ('因果投影', '离线PAVA'):
        adv = {}
        for th in ('t25', 't55', 't85'):
            a, b = df['原始_%s' % th], df['%s_%s' % (v, th)]
            both = a.notna() & b.notna()
            adv[th] = float((a[both] - b[both]).mean()) if both.any() else np.nan
        r['%s_advance_t25' % v] = adv['t25']
        r['%s_advance_t55' % v] = adv['t55']
        r['%s_advance_t85' % v] = adv['t85']
        r['%s_viol_before' % v] = float(df['原始_viol_pct'].mean())
        r['%s_viol_after' % v] = float(df['%s_viol_pct' % v].mean())
        r['%s_drawdown' % v] = float(df['原始_max_drawdown'].mean())
    return r


def main():
    ap = argparse.ArgumentParser(
        description='物理约束层（保序投影）在 A/B/C 上的消融对照')
    ap.add_argument('--sets', default='A,B,C', help='要跑哪几套，逗号分隔')
    args = ap.parse_args()
    want = [s.strip().upper() for s in args.sets.split(',') if s.strip()]

    say('=' * 104)
    say('物理约束层（保序投影）消融对照 —— 数据集 A / B / C')
    say('生成时间: %s' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    say('=' * 104)
    say('')
    say('[0] 三个变体的区别（决定了哪一列可以对外声称）')
    say('')
    say('  · **原始**      ：现有交付口径，不投影。')
    say('  · **因果投影**  ：D_mono[t] = max(D[0..t])，再钳 [0,1]。')
    say('                    只用当前及过去信息 ⇒ **可在线部署**。这是本对照的主列。')
    say('  · **离线PAVA**  ：用整条曲线做保序回归 ⇒ **用了未来信息，不可部署**，')
    say('                    仅用于量化"不守在线语义能多拿多少"。')
    say('')
    say('[1] 指标定义')
    say('')
    say('  · 违例%     ：ΔD < 0 的步数占比。投影后必须为 0（构造保证）。')
    say('  · 最大回撤  ：max(运行最大 − D)，即最深的一次"掉坑"。')
    say('                **它直接决定因果投影能把穿越时刻提前多少**（提前量 ≤ 回撤深度）。')
    say('  · t25/t55/t85：D 首次达阈的寿命%。投影只会让它**提前或不变**。')
    say('  · TV/值域   ：Σ|ΔD| / 值域。越大说明曲线越"抖"（= 可压制空间越大）。')

    dfs = []
    if 'A' in want:
        dfs.append(run_set('A', groups_A()))
    if 'B' in want:
        dfs.append(run_set('B', groups_B()))
    if 'C' in want:
        dfs.append(run_set('C', groups_C()))

    allg = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if not len(allg):
        raise SystemExit('没有任何数据，请检查 --sets 与数据源')
    allg.to_csv(os.path.join(RES, 'mono_ablation_groups.csv'),
                index=False, encoding='utf-8-sig')

    # ---------- 汇总 ----------
    say('')
    say('=' * 104)
    say('[2] 汇总：投影带来的提前量（正 = 投影后更早）与违例消除')
    say('=' * 104)
    say('  %-4s %4s %12s %12s %12s %10s %10s %10s'
        % ('集', 'n', '因果提前t25', '因果提前t55', '因果提前t85',
           '违例前%', '违例后%', '平均回撤'))
    say('  ' + '-' * 92)
    srows = []
    for name in ('A', 'B', 'C'):
        s = summarize(allg[allg['set'] == name], name)
        if s is None:
            continue
        srows.append(s)
        say('  %-4s %4d %12s %12s %12s %10.2f %10.2f %10.3f'
            % (name, s['n_groups'],
               '%.1f' % s['因果投影_advance_t25'] if np.isfinite(s['因果投影_advance_t25']) else '—',
               '%.1f' % s['因果投影_advance_t55'] if np.isfinite(s['因果投影_advance_t55']) else '—',
               '%.1f' % s['因果投影_advance_t85'] if np.isfinite(s['因果投影_advance_t85']) else '—',
               s['因果投影_viol_before'], s['因果投影_viol_after'],
               s['因果投影_drawdown']))
    pd.DataFrame(srows).to_csv(os.path.join(RES, 'mono_ablation_summary.csv'),
                               index=False, encoding='utf-8-sig')

    say('')
    say('  「提前量」单位 = 寿命百分点（正 = 投影后穿越更早）。')
    say('  ⚠️ 提前量不是"改善"——它只是把曲线抬高。判断好坏必须看锚：')

    # ---------- A 的锚方向 ----------
    dfA = allg[allg['set'] == 'A']
    if len(dfA) and 'warn_原始' in dfA:
        say('')
        say('  [A 的 A-预警时刻 vs b2 锚]（b2 是弱标签，仅作离线参考）')
        say('    %-7s %10s %10s %10s %10s %10s'
            % ('试件', 'b2', '原始', '因果投影', '离线PAVA', '原始err'))
        say('    ' + '-' * 62)

        def _f(v):
            return '%.1f' % v if np.isfinite(v) else '—'

        for r in dfA.itertuples():
            w0 = float(getattr(r, 'warn_原始'))
            e = w0 - r.anchor_b2 if (np.isfinite(w0)
                                     and np.isfinite(r.anchor_b2)) else np.nan
            say('    %-7s %10s %10s %10s %10s %10s'
                % (r.gid, '%.1f' % r.anchor_b2, _f(w0),
                   _f(float(getattr(r, 'warn_因果投影'))),
                   _f(float(getattr(r, 'warn_离线PAVA'))),
                   ('%+.1f' % e) if np.isfinite(e) else '—'))

    # ---------- 结论 ----------
    say('')
    say('=' * 104)
    say('[3] 结论')
    say('=' * 104)
    say('  读法：')
    say('    ① 若某集的「违例前%」本来就接近 0 ⇒ 约束层在该集上**无事可做**；')
    say('    ② 「因果提前量」的量级由「平均最大回撤」封顶 —— 回撤越小，收益越小；')
    say('    ③ C 集（HI_AE = unity01(累积事件数)）构造上单调 ⇒ 收益必须恰好为 0，')
    say('       它是本脚本的**阴性对照**：若 C 也测出非零收益，说明度量本身有问题。')
    say('  对外可声称的只有「因果投影」那一列；「离线PAVA」只用来量化')
    say('  "不守在线语义能多拿多少"，**不得作为结果引用**。')

    with open(os.path.join(RES, 'mono_ablation_report.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(_REPORT) + '\n')
    say('')
    say('[产出] results/mono_ablation_groups.csv · mono_ablation_summary.csv · '
        'mono_ablation_report.txt')


if __name__ == '__main__':
    main()
