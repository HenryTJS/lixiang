# -*- coding: utf-8 -*-
"""留一试件交叉验证（LOSO）：回答「5/5 精准是不是靠在同一批试件上调参」

为什么必须做
------------
`b2` 锚 + 参数（rise/fall/acc_scale/lift/shape_w/latch_*）都是在**同一批 5 组**上定的。
审稿人必然问：「换一组试件，你的参数还成立吗？」

做法（标准模型选择 + LOSO）
---------------------------
  1. 候选配置网格（**一次一变**，非全因子）：
       5 个既有参数 ±30%  +  形状证据参数（shape_w / shape_gain / shape_q）
       +  latch 速率  →  共 18 个变体 + 默认 = 19 个配置；
  2. **每个留出折**：在**其余 4 组**上按项目自身的聚合目标 `score` 选**单个最优配置**，
     再用它评估**留出的那一组**（该组及其标签从未参与选择）；
  3. 汇总留出性能，并与两个参照对比：
       default : 默认配置在全部 5 组上的成绩（= 论文主结果，样本内）
       oracle  : 事后挑在全部 5 组上最好的配置（乐观上界）
  结论读法：LOSO ≈ default ⇒ 参数不是靠这批试件"喂"出来的；
            LOSO 明显差于 default ⇒ 存在过拟合，必须报告。

⚠️ 局限（须写进论文）：搜索空间是**一次一变**的 19 个配置，不是全因子网格；
   故 LOSO 结果是**乐观侧**的（搜索空间越小越不容易过拟合）。全因子等价于放弃可解释性。

用法：python main/evaluate_loso.py
输出：main/results/loso_cv.csv（逐折）、main/results/loso_summary.csv（汇总）
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from eval_common import (GROUPS, DEFAULT, cfg_id, run_cfg,   # noqa: E402
                         table_for, per_group_metrics)

OUT_FOLD = os.path.join('results', 'loso_cv.csv')
OUT_SUM = os.path.join('results', 'loso_summary.csv')

# --- 待检验的参数与档位（一次一变；±30% 与既有 sens 口径一致）---
VARY = [
    ('rise', [DEFAULT['rise'] * 0.7, DEFAULT['rise'] * 1.3]),
    ('fall', [DEFAULT['fall'] * 0.7, DEFAULT['fall'] * 1.3]),
    ('acc_scale', [DEFAULT['acc_scale'] * 0.7, DEFAULT['acc_scale'] * 1.3]),
    ('estrain_w', [DEFAULT['estrain_w'] * 0.7, DEFAULT['estrain_w'] * 1.3]),
    ('lift', [DEFAULT['lift'] * 0.7, DEFAULT['lift'] * 1.3]),
    # —— 形状证据参数（本轮新增，必须纳入，否则 LOSO 覆盖不到最新引入的旋钮）——
    ('shape_w', [0.3, 1.0]),
    ('shape_gain', [1.0, 4.0]),
    ('shape_q', [50.0, 98.0]),
    ('latch_rise_fast', [0.30, 0.80]),
]


def score(df):
    """聚合目标（与 robustness.py 一致）：A 对齐 + 达级 - 漏报重罚，越大越好。"""
    nA = int((df['grade'] == 'A').sum())
    n55 = int((df['D_end'] >= .55).sum())
    nC = int((df['grade'] == 'C').sum())
    return nA + 0.25 * n55 - 1.5 * nC


def build_grid():
    cfgs = [(dict(DEFAULT), 'default')]
    for par, vals in VARY:
        for v in vals:
            cp = dict(DEFAULT)
            cp[par] = float(v)
            cfgs.append((cp, f'{par}={v:g}'))
    return cfgs


def main():
    grid = build_grid()
    print(f'候选配置 {len(grid)} 个 × {len(GROUPS)} 组 = {len(grid) * len(GROUPS)} 次流式\n',
          flush=True)
    tabs, names = {}, []
    for params, label in grid:
        gid_d = run_cfg(params, workers=8)
        tabs[label] = table_for(cfg_id(params), gid_d)
        names.append(label)
        print(f'  [{label}] A={int((tabs[label]["grade"] == "A").sum())} '
              f'nC={int((tabs[label]["grade"] == "C").sum())} '
              f'med|err|={tabs[label]["err"].abs().mean():.2f}', flush=True)

    # ---- 参照 ----
    d0 = tabs['default']
    ref = dict(default_nA=int((d0['grade'] == 'A').sum()),
               default_abs_err=float(d0['err'].abs().mean()),
               default_nC=int((d0['grade'] == 'C').sum()))
    oracle = max(names, key=lambda k: (score(tabs[k]), -tabs[k]['err'].abs().mean()))
    worst = min(names, key=lambda k: (score(tabs[k]), -tabs[k]['err'].abs().mean()))
    ref['oracle_cfg'] = oracle
    ref['oracle_nA'] = int((tabs[oracle]['grade'] == 'A').sum())
    ref['oracle_abs_err'] = float(tabs[oracle]['err'].abs().mean())

    # ---- LOSO ----
    # ⚠️ 评分粒度：19 个配置**全部**是 5/5 级 A、nC=0 → 单纯按 `score` 排序会**平局**，
    #    选择退化成"总是第一个（default）"。故用 (score, 更小的|err|) 字典序，
    #    让选择有真实区分度；两种口径的结果都报告。
    fold = []
    for g in GROUPS:
        train = [x for x in GROUPS if x != g]
        best_l, best_key = None, None
        for k in names:
            t = tabs[k]
            tr = t[t['gid'].isin(train)]
            key = (score(tr), -float(tr['err'].abs().mean()))
            if best_key is None or key > best_key:
                best_key, best_l = key, k
        row = tabs[best_l][tabs[best_l]['gid'] == g].iloc[0]
        row0 = d0[d0['gid'] == g].iloc[0]
        fold.append(dict(gid=g, chosen=best_l,
                         train_abs_err=-best_key[1],
                         tw_loso=row['t_warn'], err_loso=row['err'],
                         grade_loso=row['grade'], D_end_loso=row['D_end'],
                         tw_def=row0['t_warn'], err_def=row0['err'],
                         grade_def=row0['grade'], D_end_def=row0['D_end']))
        print(f'  留出 {g}: 选到 [{best_l}] (训练 4 组 |err|={-best_key[1]:.2f}) '
              f'→ 留出 t_warn={row["t_warn"]:.1f} grade={row["grade"]} '
              f'|err|={abs(row["err"]):.2f} '
              f'(默认 t_warn={row0["t_warn"]:.1f} grade={row0["grade"]})', flush=True)
    print(f'  选到的不同配置数：{len(set(f["chosen"] for f in fold))} / {len(GROUPS)} 折',
          flush=True)

    fd = pd.DataFrame(fold)
    fd.to_csv(OUT_FOLD, index=False, float_format='%.3f', encoding='utf-8-sig')
    summ = pd.DataFrame([dict(setting='LOSO(留一选参)',
                              nA=int((fd['grade_loso'] == 'A').sum()),
                              nC=int((fd['grade_loso'] == 'C').sum()),
                              nE=int((fd['grade_loso'] == 'E').sum()),
                              nD=int((fd['grade_loso'] == 'D').sum()),
                              mean_abs_err=float(fd['err_loso'].abs().mean()),
                              mean_lead=np.nan,
                              cfg='%d 种/5 折' % len(set(f['chosen'] for f in fold))),
                         dict(setting='default(样本内)', nA=ref['default_nA'],
                              nC=ref['default_nC'],
                              nE=int((d0['grade'] == 'E').sum()),
                              nD=int((d0['grade'] == 'D').sum()),
                              mean_abs_err=ref['default_abs_err'],
                              mean_lead=float(d0['lead'].mean()), cfg='default'),
                         dict(setting='oracle(事后最优)',
                              nA=ref['oracle_nA'],
                              nC=int((tabs[oracle]['grade'] == 'C').sum()),
                              nE=int((tabs[oracle]['grade'] == 'E').sum()),
                              nD=int((tabs[oracle]['grade'] == 'D').sum()),
                              mean_abs_err=ref['oracle_abs_err'],
                              mean_lead=float(tabs[oracle]['lead'].mean()),
                              cfg=oracle),
                         dict(setting='最差配置(网格内)',
                              nA=int((tabs[worst]['grade'] == 'A').sum()),
                              nC=int((tabs[worst]['grade'] == 'C').sum()),
                              nE=int((tabs[worst]['grade'] == 'E').sum()),
                              nD=int((tabs[worst]['grade'] == 'D').sum()),
                              mean_abs_err=float(tabs[worst]['err'].abs().mean()),
                              mean_lead=float(tabs[worst]['lead'].mean()),
                              cfg=worst)])
    summ.to_csv(OUT_SUM, index=False, float_format='%.3f', encoding='utf-8-sig')

    print(f'\n{"=" * 78}\nLOSO 汇总\n{"=" * 78}')
    print(summ[['setting', 'nA', 'nE', 'nD', 'nC', 'mean_abs_err',
                'mean_lead', 'cfg']].to_string(index=False))
    print(f'\n判读：LOSO 级 A 数 = {int((fd["grade_loso"] == "A").sum())}/5 '
          f'（default {ref["default_nA"]}/5, oracle {ref["oracle_nA"]}/5）；'
          f'|err| {fd["err_loso"].abs().mean():.2f} '
          f'（default {ref["default_abs_err"]:.2f}, '
          f'oracle {ref["oracle_abs_err"]:.2f}）')
    print(f'\nsaved {os.path.abspath(OUT_FOLD)} / {os.path.abspath(OUT_SUM)}')


if __name__ == '__main__':
    main()
