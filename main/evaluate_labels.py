# -*- coding: utf-8 -*-
"""标签审计：b2 的来源、标签敏感性、剔除规则事前化、正确的统计口径

为什么必须做
------------
四项检查，对应论文审稿最可能致命攻击的四点：

  1. **b2 是不是独立真值？**（`main/prepare_data.py::run_weaklabel`）
     b2 = 对 **log10(累积 AE 能量) 曲线** 做"台阶/拐点"检测（`find_b2`，要求
     `post−pre >= d_log` 且 `tail−pre >= d_tail`）；若应变发散点更早则**用应变发散覆盖**。
     ⇒ **b2 与本方法用的是同一批信号**（AE 累积能量 + 应变 std 发散），
       而 `e_full` 正是 log 累积能量的短/长窗斜率差。**故 err = t_warn − b2 有循环论证风险。**
     本脚本用可复算的数字把这一点**量化**：若 `t25`（D 首次达 0.25）与 b2 几乎重合，
     说明"精度高"在很大程度上是**自洽**，而非独立验证。

  2. **结论对标签有多敏感？** 把 b2 整体平移 ±2/±5/±10 个百分点，看 5/5 级 A 还剩几组。

  3. **剔除规则能否事前化？** 用**只看数据属性**的事前判据给出纳入/排除决定
     （见 PRE_CRITERIA），而不是"结果不好就剔"。

  4. **统计口径**：N=5 时 bootstrap 置信区间**偏窄**。给出 per-specimen 表 +
     t 分布 95% CI，并与 bootstrap 并列对比。

输出：results/label_audit.csv / results/label_sensitivity.csv /
      results/inclusion_criteria.csv / results/per_specimen_stats.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

GROUPS = ['016', '017', '018', '019', '020']
ALL = ['015'] + GROUPS + ['021', '022', '023', '024', '025', '026', '027']

# ---- 事前判据（**与任何结果无关**，只依赖数据属性）----
# 声明在先：任何不满足以下任一条的试件**不进入"疲劳失效预警"统计集**。
PRE_CRITERIA = {
    'P1_数据完整性': '时间列语义可识别(t_kind != "?") 且 应变行数>0 且 采样间隔>0',
    'P2_AE存量': 'AE 行数>0 且 AE 行数/应变行数 >= 0.05（确有事件流）',
    'P3_试验类型': '循环次数 = 无上限（疲劳至失效）；有上限(循环+静力)属**另一类试验**',
}


def load():
    base = 'results'
    dg = pd.read_csv(os.path.join(base, 'damage_degree_metrics.csv'))
    wn = pd.read_csv(os.path.join(base, 'warning_onset.csv'))
    for df in (dg, wn):
        df['gid'] = df['gid'].astype(str).str.zfill(3)
    cc = pd.read_csv(os.path.join(base, 'candidate_check.csv'))
    cc['gid'] = cc['gid'].astype(str).str.zfill(3)
    lab = pd.read_csv('weak_labels/labels_summary.csv')
    lab['gid'] = lab['gid'].astype(int).map(lambda x: f'{x:03d}')
    return dg, wn, cc, lab


def _int(x, d=0):
    """安全取整（NaN/None/非数值 → d）。"""
    try:
        v = float(x)
        return d if not np.isfinite(v) else int(v)
    except (TypeError, ValueError):
        return d


def protocol():
    """加载协议表：循环次数是否无上限。"""
    rec = pd.read_csv('数据记录.xlsx', sheet_name='Sheet1') if False else None
    df = pd.read_excel('数据记录.xlsx', sheet_name='Sheet1')
    d = {}
    for _, r in df.iterrows():
        gid = f'{int(r["序号"]):03d}'
        cyc = str(r['循环次数']).strip()
        d[gid] = dict(cycles=cyc, unlimited=(cyc == '无上限'),
                      peak_kn=r['峰值应力（KN）'], freq=r['频率'])
    return d


def grade(tw, b2, band=15.0):
    if tw is None or (isinstance(tw, float) and not np.isfinite(tw)):
        return 'C'
    e = tw - b2
    return 'E' if e < -band else ('D' if e > band else 'A')


def stat_summary(v, name):
    """mean / t 分布 95% CI / bootstrap 95% CI 对比。"""
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return None
    m, sd, n = float(v.mean()), float(v.std(ddof=1)), len(v)
    try:
        from scipy import stats as sps
        tcrit = float(sps.t.ppf(0.975, n - 1))
    except Exception:
        tcrit = 2.776 if n == 5 else 2.0
    se = sd / np.sqrt(n)
    rng = np.random.default_rng(0)
    bs = [rng.choice(v, n, replace=True).mean() for _ in range(20000)]
    return dict(metric=name, n=n, mean=m, sd=sd, se=se,
                ci_t_lo=m - tcrit * se, ci_t_hi=m + tcrit * se,
                ci_boot_lo=float(np.percentile(bs, 2.5)),
                ci_boot_hi=float(np.percentile(bs, 97.5)),
                t_crit=tcrit)


def main():
    dg, wn, cc, lab = load()
    proto = protocol()
    b2 = dict(zip(lab['gid'], lab['b2']))
    b3 = dict(zip(lab['gid'], lab['b3']))
    dgi = dg.set_index('gid')
    wni = wn.set_index('gid')

    # ---------- 1. b2 来源与循环性量化 ----------
    print('=' * 100)
    print('1. b2 是不是独立真值？（b2 由 log10(累积 AE 能量) 的台阶检测得到，'
          '与 e_full 同源）')
    print('=' * 100)
    print(f'{"gid":<5}{"b2(%)":>8}{"t25(%)":>8}{"t25-b2":>9}{"t_warn(%)":>11}'
          f'{"tw-b2":>8}{"t85(%)":>8}{"b3":>5}')
    rows = []
    for g in GROUPS:
        r25 = float(dgi.loc[g, 't25'])
        tw = float(wni.loc[g, 't_warn'])
        t85 = float(dgi.loc[g, 't85'])
        rows.append(dict(gid=g, b2=b2[g], t25=r25, t25_minus_b2=r25 - b2[g],
                         t_warn=tw, tw_minus_b2=tw - b2[g], t85=t85, b3=b3[g],
                         D_end=float(dgi.loc[g, 'D_end'])))
        print(f'{g:<5}{b2[g]:>8.1f}{r25:>8.1f}{r25 - b2[g]:>9.1f}{tw:>11.1f}'
              f'{tw - b2[g]:>8.1f}{t85:>8.1f}{b3[g]:>5.1f}')
    aud = pd.DataFrame(rows)
    aud.to_csv('results/label_audit.csv', index=False, float_format='%.3f',
               encoding='utf-8-sig')
    dev = aud['t25_minus_b2'].abs()
    print(f'\n  >>> |t25 - b2| : 均值 {dev.mean():.2f} 个百分点，最大 {dev.max():.2f}')
    print('  >>> 判读：t25 是「D 首次达 0.25」，b2 是「log 累积 AE 能量的台阶位置」——')
    print('      两者在 5/5 组上相差不到 2.5 个百分点，说明**标签与检测器锁在同一条曲线上**。')
    print('      => err = t_warn - b2 的"高精度"(|err| 2.05) 有相当部分是**自洽**，')
    print('         不能单独作为"方法准确"的证据。必须补独立锚（见第 3 节）。')

    # ---------- 2. 标签敏感性 ----------
    print('\n' + '=' * 100)
    print('2. 标签敏感性：把 b2 整体平移 delta 个百分点后，5 组里还有几组是 A 级')
    print('=' * 100)
    bl = pd.read_csv('results/baseline_compare.csv', dtype={'gid': str})
    bl['gid'] = bl['gid'].str.zfill(3)
    b1 = bl[(bl['method'] == 'B1_累积能量') & np.isclose(bl['level'], 1.5)]
    tw_b1 = dict(zip(b1['gid'], b1['t_warn']))
    sens = []
    deltas = [-10, -5, -2, 0, 2, 5, 10]
    print(f'{"delta":>7}' + ''.join(f'{d:>9}' for d in deltas))
    for meth, twmap in (('本项目(交付口径)',
                         {g: float(wni.loc[g, 't_warn']) for g in GROUPS}),
                        ('B1_累积能量(最佳基线)', tw_b1)):
        line = f'{meth:>7}'
        for d in deltas:
            nA = sum(1 for g in GROUPS if grade(twmap.get(g), b2[g] + d) == 'A')
            y = {g: (twmap.get(g), b2[g] + d) for g in GROUPS}
            me = np.mean([abs(t0 - b0) for t0, b0 in y.values()
                          if t0 is not None and np.isfinite(t0)])
            sens.append(dict(method=meth, delta=d, nA=nA, mean_abs_err=me))
            line += f'{nA:d}/{me:.1f}'.rjust(9)
        print(line)
    pd.DataFrame(sens).to_csv('results/label_sensitivity.csv', index=False,
                              float_format='%.3f', encoding='utf-8-sig')
    print('  格式 = A组数/均值|err|；delta 为对 b2 的整体平移（百分点）')

    # ---------- 3. 剔除规则事前化 ----------
    print('\n' + '=' * 100)
    print('3. 纳入/排除决定：**事前判据**（只依赖数据属性，与结果无关）')
    print('=' * 100)
    for k, v in PRE_CRITERIA.items():
        print(f'   {k}: {v}')
    cci = cc.set_index('gid')
    inc = []
    for g in ALL:
        if g not in cci.index:
            inc.append(dict(gid=g, P1=False, P2=False, P3=False, include=False,
                            reason='体检表缺该组'))
            continue
        r = cci.loc[g]
        n_ae, n_str, dt_s = _int(r.get('n_ae')), _int(r.get('n_str')), \
            (float(r.get('dt_s', 0)) if np.isfinite(r.get('dt_s', np.nan)) else 0.0)
        p1 = (str(r.get('t_kind', '?')) != '?') and n_str > 0 and dt_s > 0
        p2 = n_ae > 0 and n_ae / max(n_str, 1) >= 0.05
        p3 = bool(proto.get(g, {}).get('unlimited', False))
        reasons = []
        if not p1:
            reasons.append('时间列/采样不可识别')
        if not p2:
            reasons.append('AE 存量不足')
        if not p3:
            reasons.append('有上限试验(循环+静力)，非疲劳失效类')
        inc.append(dict(gid=g, P1=bool(p1), P2=bool(p2), P3=bool(p3),
                        include=bool(p1 and p2 and p3),
                        n_ae=n_ae, n_str=n_str,
                        t_kind=str(r.get('t_kind', '')),
                        cycles=proto.get(g, {}).get('cycles', ''),
                        reason='；'.join(reasons)))
    ii = pd.DataFrame(inc)
    ii.to_csv('results/inclusion_criteria.csv', index=False, encoding='utf-8-sig')
    print(f'\n{"gid":<5}{"P1":>4}{"P2":>4}{"P3":>4}{"纳入":>6}{"循环次数":>10}'
          f'{"AE行":>8}  排除理由')
    for _, r in ii.iterrows():
        print(f'{r["gid"]:<5}{("Y" if r["P1"] else "n"):>4}'
              f'{("Y" if r["P2"] else "n"):>4}{("Y" if r["P3"] else "n"):>4}'
              f'{("纳入" if r["include"] else "排除"):>6}{str(r["cycles"]):>10}'
              f'{r["n_ae"]:>8}  {r["reason"]}')
    ins = list(ii[ii['include']]['gid'])
    print(f'\n  >>> 事前判据下的纳入集（{len(ins)} 组）: {", ".join(ins)}')
    print('  >>> 注意：023-026 属纳入集但**无 b2 标签**（标签生成只覆盖 016-020）→')
    print('      它们只能在"无标签统一协议"（§3.8 表 3，物理锚 f_est）下评估。')

    # ---------- 4. 统计口径 ----------
    print('\n' + '=' * 100)
    print('4. 统计口径：per-specimen 表 + t 分布 95% CI（N=5 时 bootstrap 偏窄）')
    print('=' * 100)
    per = []
    for g in GROUPS:
        tw = float(wni.loc[g, 't_warn'])
        per.append(dict(gid=g, t25=float(dgi.loc[g, 't25']), t55=float(dgi.loc[g, 't55']),
                        t85=float(dgi.loc[g, 't85']), t_warn=tw, b2=b2[g],
                        err=tw - b2[g], abs_err=abs(tw - b2[g]),
                        lead=b3[g] - tw, D_end=float(dgi.loc[g, 'D_end']),
                        grade=grade(tw, b2[g])))
    ps = pd.DataFrame(per)
    ps.to_csv('results/per_specimen_stats.csv', index=False, float_format='%.3f',
              encoding='utf-8-sig')
    print(ps.to_string(index=False))
    st = [stat_summary(ps['abs_err'], '|err| (本项目, 交付口径)')]
    eb1 = [abs(tw_b1[g] - b2[g]) for g in GROUPS if g in tw_b1]
    st.append(stat_summary(eb1, '|err| (B1 累积能量, 最佳基线)'))
    st = [s for s in st if s]
    sd = pd.DataFrame(st)
    print()
    print(sd[['metric', 'n', 'mean', 'sd', 'ci_t_lo', 'ci_t_hi',
              'ci_boot_lo', 'ci_boot_hi']].to_string(index=False, float_format=lambda x: f'{x:.3f}'))
    print('\n  >>> 判读：N=5 时 **bootstrap CI 明显窄于 t 分布 CI**，'
          '论文里应用后者（或直接报 per-specimen 值）。')
    print('\nsaved results/label_audit.csv, label_sensitivity.csv, '
          'inclusion_criteria.csv, per_specimen_stats.csv')


if __name__ == '__main__':
    main()
