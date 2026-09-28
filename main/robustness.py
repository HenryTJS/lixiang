# -*- coding: utf-8 -*-
"""阶段③ 方法稳健性研究（5 组主样本）

子命令（可组合，如 `python robustness.py sens loso ablation` 或 `all`）：
  sens      参数 ±30% 敏感性 → results/parameter_sensitivity.csv（缓存 cache/_t7_cache）
  loso      留一试件交叉验证选参 → results/leave_one_out_cv.csv（需先跑 sens 生成缓存）
  ablation  证据消融(去各源/结构) → results/ablation.csv + figures/ablation.png
  stats     T9 统计显著性(多源 vs 单源配对 + Bootstrap CI) → results/statistical_test.csv
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from concurrent.futures import ProcessPoolExecutor
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # main/
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex
from eval_common import (PARS, DEFAULT, GROUPS, cfg_id, T7CACHE,
                         per_group_metrics, run_cfg, table_for, summary_row)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(ROOT, 'figures')
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)
WORKERS = 8


# ============================================================
# 公共：逐点跑 D（事件门控 AE 能量）
# ============================================================
def _stream_d(gid, di):
    sim = StreamSimulator(gid)
    sim.load_data()
    dlist = []
    while sim.has_next():
        p = sim.next_point()
        strain = p['strain']
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        dlist.append(di.update(strain, float(pk) if pk is not None else None,
                               None, di.shape_value(p.get('ae'))))
    sim.cleanup()
    return np.array(dlist, dtype=np.float32)


def _cached_d(gid, di, cdir):
    """按缓存目录计算/读取逐点 D。"""
    os.makedirs(cdir, exist_ok=True)
    fp = os.path.join(cdir, f'{gid}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    d = _stream_d(gid, di)
    np.save(fp, d)
    return d


# ============================================================
# 子命令 sens —— 参数敏感性
# ============================================================
SENS_OUT = os.path.join(RESULTS, 'parameter_sensitivity.csv')


def build_configs():
    """默认 + 每参数 ±30%。返回 [(cfg_params, label)]"""
    cfgs = [({}, 'default')]
    for p in PARS:
        for m in (0.7, 1.3):
            cp = dict(DEFAULT)
            cp[p] = DEFAULT[p] * m
            cfgs.append((cp, f'{p}x{m}'))
    return cfgs


def cmd_sens():
    cfgs = build_configs()
    rows = []
    print(f'[T7-sens] 共 {len(cfgs)} 配置 × {len(GROUPS)} 组, workers={WORKERS}', flush=True)
    t0 = time.time()
    for params, label in cfgs:
        cid = cfg_id(params)
        ts = time.time()
        gid_d = run_cfg(params, workers=WORKERS)
        df = table_for(cid, gid_d)
        row = summary_row(cid, df)
        row['label'] = label
        rows.append(row)
        print(f'  [{label}] {time.time()-ts:.0f}s  n55={row["n55"]} n85={row["n85"]} '
              f'A={row["nA"]} E={row["nE"]} D={row["nD"]} C={row["nC"]} '
              f'med|err|={row["mean_abs_err"] if row["mean_abs_err"]==row["mean_abs_err"] else "-"}', flush=True)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(SENS_OUT, index=False, encoding='utf-8-sig')
    print(f'总耗时 {time.time()-t0:.0f}s; 结果已存: {SENS_OUT}', flush=True)


# ============================================================
# 子命令 loso —— 留一试件交叉验证
# ============================================================
LOSO_OUT = os.path.join(RESULTS, 'leave_one_out_cv.csv')


def load_table(params):
    """从缓存读 {gid:d} 并转逐组指标表。缺缓存则报错提示先跑 sens。"""
    cid = cfg_id(params)
    cdir = os.path.join(T7CACHE, cid)
    rows = []
    for g in GROUPS:
        fp = os.path.join(cdir, f'{g}.npy')
        if not os.path.exists(fp):
            sys.exit(f'缺缓存 {fp} —— 请先运行 robustness.py sens(或单独补算该配置)')
        d = np.load(fp)
        rows.append(per_group_metrics(g, d))
    return pd.DataFrame(rows)


def cand_cfgs(p):
    """参数 p 的 3 档候选(其余默认)。"""
    out = []
    for m in (0.7, 1.0, 1.3):
        cp = dict(DEFAULT)
        cp[p] = DEFAULT[p] * m
        out.append((m, cp))
    return out


def score(df):
    """聚合目标(越大越好): A 对齐 + 达级 + 漏报重罚。"""
    nA = int((df['grade'] == 'A').sum())
    n55 = int((df['D_end'] >= .55).sum())
    nC = int((df['grade'] == 'C').sum())
    return nA + 0.25 * n55 - 1.5 * nC


def cmd_loso():
    recs = []
    for p in PARS:
        tabs = {m: load_table(cp) for m, cp in cand_cfgs(p)}
        tab0 = tabs[1.0]
        chosen = []
        diff_rows = []
        for g in GROUPS:
            train = [x for x in GROUPS if x != g]
            best_m, best_s = None, -1e18
            for m, cp in cand_cfgs(p):
                s = score(tabs[m][tabs[m]['gid'].isin(train)])
                if s > best_s:
                    best_s, best_m = s, m
            chosen.append(best_m)
            row0 = tab0[tab0['gid'] == g].iloc[0]
            rowc = tabs[best_m][tabs[best_m]['gid'] == g].iloc[0]
            diff_rows.append(dict(gid=g, par=p, chosen_m=best_m,
                                  g0=row0['grade'], gc=rowc['grade'],
                                  err0=row0['err'], errc=rowc['err'],
                                  dend0=row0['D_end'], dendc=rowc['D_end']))
        ch = pd.Series(chosen).value_counts().to_dict()
        diff = pd.DataFrame(diff_rows)
        improved = int(((diff['g0'] == 'E') & (diff['gc'] == 'A')).sum())
        worsened = int(((diff['g0'].isin(['A', 'D'])) & (diff['gc'] == 'E')).sum())
        nA0 = int((tab0['grade'] == 'A').sum())
        recs.append(dict(par=p, desc='', ch_0_7=ch.get(0.7, 0),
                         ch_1_0=ch.get(1.0, 0), ch_1_3=ch.get(1.3, 0),
                         nA_default=nA0, EtoA=improved, AtoE=worsened))
        print(f'[{p}] 选档(0.7/1.0/1.3)={ch.get(0.7,0)}/{ch.get(1.0,0)}/{ch.get(1.3,0)}  '
              f'留出: E→A {improved}, A→E {worsened}  (默认全数据 nA={nA0})', flush=True)
    out = pd.DataFrame(recs)
    out.to_csv(LOSO_OUT, index=False, encoding='utf-8-sig')
    print('结果已存:', LOSO_OUT, flush=True)


# ============================================================
# 子命令 ablation —— 证据消融
# ============================================================
ABL_CACHE = os.path.join(ROOT, 'cache', '_t8_cache')
ABL_MODES = [
    ('full', '完整 v6(基准)'),
    ('no_dmg', '去损伤型事件(仅全能量)'),
    ('no_full', '去全能量加速(仅损伤型)'),
    ('no_strain', '去应变证据(仅 AE)'),
    ('only_strain', '仅应变证据'),
    ('no_accum', '去单调累积(D=risk)'),
]
KEY = ['016', '020', '017', '019']   # 消融展示重点组(均属主样本)


def _abl_run_one(args):
    gid, mode = args
    cdir = os.path.join(ABL_CACHE, mode)
    return gid, _cached_d(gid, OnlineDamageIndex({'abl': mode}), cdir)


def cmd_ablation():
    rows_all = []
    agg = []
    for mode, desc in ABL_MODES:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(_abl_run_one, [(g, mode) for g in GROUPS]))
        gid_d = dict(res)
        per = {r['gid']: r for r in (per_group_metrics(g, d) for g, d in gid_d.items())}
        df = pd.DataFrame([per[g] for g in GROUPS]).sort_values('gid')
        df.insert(0, 'mode', mode)
        rows_all.append(df)
        n55 = int((df['D_end'] >= .55).sum())
        n85 = int((df['D_end'] >= .85).sum())
        cnt = df['grade'].value_counts()
        agg.append(dict(mode=mode, desc=desc, n55=n55, n85=n85,
                        D_end_med=float(df['D_end'].median()),
                        nA=int(cnt.get('A', 0)), nE=int(cnt.get('E', 0)),
                        nD=int(cnt.get('D', 0)), nC=int(cnt.get('C', 0)),
                        tail_mono=float(df['tail_mono'].mean())))
        print(f'[{desc}] n55={n55} n85={n85} A/E/D/C='
              f'{int(cnt.get("A",0))}/{int(cnt.get("E",0))}/{int(cnt.get("D",0))}/{int(cnt.get("C",0))} '
              f'D_end中位={df["D_end"].median():.2f}', flush=True)
    big = pd.concat(rows_all, ignore_index=True)
    big.to_csv(os.path.join(RESULTS, 'ablation.csv'), index=False, encoding='utf-8-sig')
    key_row = big[big['gid'].isin(KEY)].pivot_table(index='gid', columns='mode',
                                                    values='D_end').reindex(KEY)
    print('\n=== 重点组 D_end 对比 ===')
    print(key_row.round(2).to_string())
    a = pd.DataFrame(agg)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(a))
    axes[0].bar(x, a['n55'], color='tab:blue'); axes[0].set_xticks(x)
    axes[0].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[0].set_title('达 0.55 组数')
    axes[1].bar(x, a['nA'], color='tab:green'); axes[1].set_xticks(x)
    axes[1].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[1].set_title('A-预警精准组数')
    axes[2].bar(x, a['nE'], color='tab:red'); axes[2].set_xticks(x)
    axes[2].set_xticklabels(a['mode'], rotation=30, ha='right'); axes[2].set_title('E-过早组数')
    fig.suptitle('T8 消融: 各证据/结构对 D(t) 的影响')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, 'ablation.png'), dpi=110)
    print('结果已存: results/ablation.csv + figures/ablation.png')


# ============================================================
# 子命令 stats —— T9 统计检验
# ============================================================
STATS_CACHE = os.path.join(ROOT, 'cache', '_stats_cache')
STATS_MODES = {'full': '多源融合D(full)', 'no_strain': '仅AE(no_strain)',
               'only_strain': '仅应变(only_strain)'}
NBOOT = 2000


def _stats_run_one(args):
    gid, mode = args
    cdir = os.path.join(STATS_CACHE, mode)
    return gid, _cached_d(gid, OnlineDamageIndex({'abl': mode}), cdir)


def wilcoxon_onesided(a, b):
    """H1: a < b(多源融合误差更小)。返回 p(单侧) 与 n 配对。"""
    diff = np.asarray(a, float) - np.asarray(b, float)
    diff = diff[~np.isnan(diff)]
    n = len(diff)
    if n == 0:
        return np.nan, 0
    try:
        p = sp_stats.wilcoxon(diff, alternative='less').pvalue
        return p, n
    except ValueError:
        return np.nan, n


def bootstrap_ci(x, stat=np.mean, alpha=0.05):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    boot = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(NBOOT)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def cmd_stats():
    modes = list(STATS_MODES.keys())
    jobs = [(g, m) for g in GROUPS for m in modes]
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(_stats_run_one, jobs))
    per = {m: {} for m in modes}
    for (g, m), (gid, d) in zip(jobs, res):
        per[m][gid] = per_group_metrics(gid, d)
    rows = []
    for g in GROUPS:
        base = per['full'][g]
        row = {'gid': g, 'b2': base['b2'], 'b3': base['b3']}
        for m in modes:
            mm = per[m][g]
            row[f'{m}_t_warn'] = mm['t_warn']
            row[f'{m}_err'] = mm['err']
            row[f'{m}_lead'] = mm['lead']
            row[f'{m}_grade'] = mm['grade']
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'statistical_test.csv'), index=False, encoding='utf-8-sig')

    def colerr(m):
        return df[f'{m}_err'].to_numpy(float)

    print('=== 逐组 |预警-扩展| 误差(小=好) ===')
    print(df[['gid', 'full_err', 'no_strain_err', 'only_strain_err']]
          .assign(**{'|full|': df.full_err.abs(), '|AE|': df.no_strain_err.abs(),
                     '|strain|': df.only_strain_err.abs()})[['gid', '|full|', '|AE|', '|strain|']]
          .round(1).to_string(index=False))

    a_full = np.abs(colerr('full'))
    a_ae = np.abs(colerr('no_strain'))
    a_st = np.abs(colerr('only_strain'))
    print('\n=== Wilcoxon 配对检验(H1: 多源融合 |err| 更小) ===')
    p1, n1 = wilcoxon_onesided(a_full, a_ae)
    p2, n2 = wilcoxon_onesided(a_full, a_st)
    print(f'  D vs 仅AE   : p={p1:.3f} (n={n1})  均值|err| D={np.nanmean(a_full):.1f} vs AE={np.nanmean(a_ae):.1f}')
    print(f'  D vs 仅应变 : p={p2:.3f} (n={n2})  均值|err| D={np.nanmean(a_full):.1f} vs 应变={np.nanmean(a_st):.1f}')

    print('\n=== 漏报(C, 无不可逆onset)组数 ===')
    for m in modes:
        print(f'  {STATS_MODES[m]}: {int((df[f"{m}_grade"]=="C").sum())}')

    print('\n=== Bootstrap 95% CI(主样本 5 组, 试件重采样) ===')
    lo, hi = bootstrap_ci(a_full)
    print(f'  D  |err| 均值 CI: [{lo:.1f}, {hi:.1f}]  样本均值 {np.nanmean(a_full):.1f}')
    lead = df['full_lead'].to_numpy(float)
    lo2, hi2 = bootstrap_ci(lead)
    print(f'  D  断裂前提前量 lead 均值 CI: [{lo2:.1f}, {hi2:.1f}]  样本均值 {np.nanmean(lead):.1f}')
    print('  (小样本 n=5，CI 仅示意；结论需谨慎解读)')
    print('结果已存: results/statistical_test.csv')


# ============================================================
# 分发
# ============================================================
TASKS = {'sens': cmd_sens, 'loso': cmd_loso,
         'ablation': cmd_ablation, 'stats': cmd_stats}


def main():
    ap = argparse.ArgumentParser(description='阶段③ 稳健性研究：sens / loso / ablation / stats')
    ap.add_argument('tasks', nargs='+', choices=list(TASKS) + ['all'],
                    help='要执行的任务（可多个，或用 all）')
    ap.add_argument('--workers', type=int, default=8,
                    help='多进程 worker 数(默认 8)')
    a = ap.parse_args()
    global WORKERS
    WORKERS = a.workers
    todo = list(TASKS) if 'all' in a.tasks else a.tasks
    for t in todo:
        print(f'\n########## [{t}] ##########', flush=True)
        TASKS[t]()
        print(f'########## [{t}] 完成 ##########', flush=True)


if __name__ == '__main__':
    main()
