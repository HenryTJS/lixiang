# -*- coding: utf-8 -*-
"""阶段② 主样本评估与出图（5 组 016-020）

子命令（可组合，如 `python evaluate.py degree warning` 或 `all`）：
  degree   评估损伤度 D 达阈/单调 → results/damage_degree_metrics.csv（缓存 cache/_hi_cache）
  warning  A-预警 onset 分级 → results/warning_onset.csv（读 cache/_hi_cache，先跑 degree）
  curves   D(t) 曲线总览 → figures/damage_degree_curves.png（读 cache/_hi_cache）
  paper    论文级四联图 + 方法流程图 → figures/paper_*.png（逐点重算，较慢）
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from concurrent.futures import ProcessPoolExecutor
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # main/
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex
from eval_common import GROUPS, ref_map, onset_of, LOW, DROP, HOLD_FRAC

ROOT = os.path.dirname(os.path.abspath(__file__))
DNCACHE = os.path.join(ROOT, 'cache', '_hi_cache')      # 逐点 D 缓存
RESULTS = os.path.join(ROOT, 'results')
FIGDIR = os.path.join(ROOT, 'figures')
os.makedirs(DNCACHE, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================
# 公共：逐点跑 D（事件门控 AE 能量）
# ============================================================
def _stream_d(gid, di):
    """StreamSimulator 逐点喂 OnlineDamageIndex → 返回逐点 D 数组。"""
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


def _first_ge(d, th):
    idx = np.where(d >= th)[0]
    return float(idx[0]) / len(d) * 100.0 if len(idx) else np.nan


# ============================================================
# 子命令 degree —— 评估损伤度 D（达阈/单调）
# ============================================================
DEG_OUT = os.path.join(RESULTS, 'damage_degree_metrics.csv')
WORKERS = 4


def degree_run_one(gid):
    dpath = os.path.join(DNCACHE, f'{gid}.npy')
    if os.path.exists(dpath):
        d = np.load(dpath)
    else:
        d = _stream_d(gid, OnlineDamageIndex())
        np.save(dpath, d)
    return gid, d


def cmd_degree():
    if WORKERS > 1:
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(degree_run_one, GROUPS))
    else:
        res = [degree_run_one(g) for g in GROUPS]
    rows = []
    for gid, d in res:
        n = len(d)
        t25, t55, t85 = _first_ge(d, .25), _first_ge(d, .55), _first_ge(d, .85)
        tail = d[int(n * .80):]                       # 断裂前 20% 单调性
        mono = float(np.mean(np.diff(tail) >= 0)) if len(tail) > 2 else np.nan
        rows.append(dict(gid=gid, n=n, D_end=round(float(d[-1]), 2),
                         t25=round(t25, 1) if not np.isnan(t25) else np.nan,
                         t55=round(t55, 1) if not np.isnan(t55) else np.nan,
                         t85=round(t85, 1) if not np.isnan(t85) else np.nan,
                         lead85=round(99.0 - t85, 1) if not np.isnan(t85) else np.nan,
                         tail_mono=round(mono * 100, 1)))
    df = pd.DataFrame(rows).sort_values('gid')
    df.to_csv(DEG_OUT, index=False, encoding='utf-8-sig')
    print('=== D(t) 达阈统计 ===')
    print('  D_end>=0.85 组数:', int((df['D_end'] >= .85).sum()), '/', len(df))
    print('  D_end>=0.55 组数:', int((df['D_end'] >= .55).sum()))
    l85 = df['lead85'].dropna()
    print(f'  达0.85组 平均断裂前提前量={l85.mean():.1f}%  (n={len(l85)})')
    print(df.to_string(index=False))
    print('结果已存:', DEG_OUT)


# ============================================================
# 子命令 warning —— A-预警 onset 分级
# ============================================================
WARN_OUT = os.path.join(RESULTS, 'warning_onset.csv')


def cmd_warning():
    rows = []
    for gid in GROUPS:
        d = np.load(os.path.join(DNCACHE, f'{gid}.npy'))
        n = len(d)
        hold = max(2000, int(n * HOLD_FRAC))
        tw = onset_of(d, hold)
        refs = ref_map()
        b2, b3 = refs.get(gid, (np.nan, 99.0))
        err = round(tw - b2, 1) if tw is not None else None
        lead = round(b3 - tw, 1) if tw is not None else None
        if tw is None:
            grade = 'C漏报'
        elif err is not None and err < -15:
            grade = 'E过早'
        elif err is not None and err > 15:
            grade = 'D偏晚'
        else:
            grade = 'A合理'
        rows.append(dict(gid=gid, t_warn=tw, b2=b2, b3=b3,
                         err=err, lead=lead, grade=grade))
        print(f'{gid}: t_warn={tw}  err={err}  lead={lead}  [{grade}]', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(WARN_OUT, index=False, encoding='utf-8-sig')
    cnt = df['grade'].value_counts()
    print('=== 分级计数 ===')
    for k in ['A合理', 'D偏晚', 'E过早', 'C漏报']:
        print(f'  {k}: {int(cnt.get(k, 0))} 组')
    ok = df[df['grade'] == 'A合理']
    if len(ok):
        print(f'A 组平均 lead(断裂前提前)={ok["lead"].mean():.1f}%')
    print('结果已存:', WARN_OUT)


# ============================================================
# 子命令 curves —— D(t) 曲线总览
# ============================================================
def cmd_curves():
    refs = ref_map()
    n = len(GROUPS)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.4 * nrows))
    axes = axes.ravel()
    for ax, g in zip(axes, GROUPS):
        d = np.load(os.path.join(DNCACHE, f'{g}.npy'))
        x = np.linspace(0, 100, len(d))
        ax.plot(x, d, lw=0.8, color='tab:blue')
        for th, c in [(0.25, 'tab:orange'), (0.55, 'tab:red'), (0.85, 'darkred')]:
            ax.axhline(th, color=c, lw=0.6, ls='--', alpha=0.6)
        b2, b3 = refs.get(g, (np.nan, 99.0))
        ax.axvline(b2, color='tab:green', lw=0.8, ls='-.', alpha=0.8)
        ax.axvline(b3, color='k', lw=0.8, ls=':', alpha=0.8)
        ax.set_title(f'{g}  D_end={d[-1]:.2f}', fontsize=10)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 25, 50, 75, 99])
        ax.grid(alpha=0.3)
    for ax in axes[n:]:                       # 多余子图隐藏
        ax.axis('off')
    fig.suptitle('主样本 5 组 D(t) 曲线（默认 v6+latch；分级 0.25/0.55/0.85；绿虚线 b2，黑点线 b3=断裂）',
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = os.path.join(FIGDIR, 'damage_degree_curves.png')
    fig.savefig(out, dpi=110)
    print('已存:', out)


# ============================================================
# 子命令 paper —— 论文级四联图 + 方法流程图
# ============================================================
BLK = 500
LV_COLORS = ['#d9ead3', '#fff2cc', '#fce5cd', '#f4cccc']   # 0正常..3临危


def paper_collect(gid):
    """逐点跑 full，每块记 {pct, strain, ae_en, e_ae, e_strain, D, level}"""
    di = OnlineDamageIndex()
    sim = StreamSimulator(gid)
    sim.load_data()
    pts = 0
    blk_st, blk_en = [], 0.0
    rows = []
    while sim.has_next():
        p = sim.next_point()
        strain = p['strain']
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        if strain is not None and not np.isnan(strain):
            blk_st.append(float(strain))
        if pk is not None:
            blk_en += float(pk) ** 2
        d = di.update(strain, float(pk) if pk is not None else None,
                      None, di.shape_value(p.get('ae')))
        pts += 1
        if pts % BLK == 0:
            rows.append(dict(
                pct=pts / sim.total_points * 100.0,
                strain=float(np.mean(blk_st)) if blk_st else np.nan,
                ae_en=np.log10(blk_en + 1.0),
                e_ae=di._last_e_ae, e_strain=di._last_e_strain,
                D=float(d), level=di.level))
            blk_st, blk_en = [], 0.0
    sim.cleanup()
    return pd.DataFrame(rows)


def paper_draw_one(gid):
    df = paper_collect(gid)
    refs = ref_map()
    b2, b3 = refs.get(gid, (np.nan, 99.0))
    fig = plt.figure(figsize=(12, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.3, 0.6], hspace=0.32,
                          left=0.09, right=0.97, top=0.94, bottom=0.08)
    ax = [fig.add_subplot(gs[i]) for i in range(4)]
    # A 原始信号
    a = ax[0]
    a.plot(df['pct'], df['strain'], color='#1f77b4', lw=0.9, label='应变(块均值)')
    a2 = a.twinx()
    a2.plot(df['pct'], df['ae_en'], color='#d62728', lw=0.8, alpha=0.75, label='AE 块能量 log')
    a2.set_ylabel('log(能量+1)')
    a.set_ylabel('应变')
    a.set_title(f'{gid}  原始多源信号')
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc='upper left', framealpha=0.6)
    # B 证据
    b = ax[1]
    b.plot(df['pct'], df['e_ae'], color='#9467bd', lw=1.0, label='e_ae (AE 证据)')
    b.plot(df['pct'], df['e_strain'], color='#2ca02c', lw=1.0, label='e_strain (应变证据)')
    b.set_ylim(-0.02, 1.02)
    b.set_ylabel('证据')
    b.legend(fontsize=8, loc='upper left', framealpha=0.6)
    b.set_title('各源损伤证据')
    # C D + 分级
    c = ax[2]
    c.plot(df['pct'], df['D'], color='#000000', lw=1.4, label='D(t)')
    for th, col, lab in [(0.25, '#e07b00', '注意'), (0.55, '#d62728', '预警'), (0.85, '#8b0000', '临危')]:
        c.axhline(th, color=col, lw=0.9, ls='--', alpha=0.6)
        c.text(1.5, th + 0.01, f'{th:.2f} {lab}', color=col, fontsize=7, va='bottom')
    c.axvline(b2, color='#2e7d32', lw=1.0, ls='-.', alpha=0.9)
    c.text(b2 + 0.5, 0.95, 'b2 扩展', color='#2e7d32', fontsize=7, rotation=90, va='top')
    c.axvline(b3, color='k', lw=1.0, ls=':', alpha=0.8)
    c.text(b3 - 0.5, 0.95, 'b3 断裂', fontsize=7, rotation=90, va='top', ha='right')
    tw = df[df['D'] >= 0.30]
    if len(tw):
        seg = df['D'].to_numpy()
        idx0 = np.where(seg >= 0.30)[0]
        onset = None
        hold = max(1, int(len(seg) * 0.02))
        for i in idx0:
            if seg[i:min(len(seg), i + hold)].min() >= 0.15:
                onset = df['pct'].iloc[i]
                break
        if onset is not None:
            c.plot(onset, 0.30, 'r*', ms=14, zorder=5)
            c.annotate(f'预警 {onset:.0f}%', (onset, 0.30), textcoords='offset points',
                       xytext=(8, 6), fontsize=8, color='red')
    c.set_ylim(-0.02, 1.02)
    c.set_ylabel('损伤度 D')
    c.legend(fontsize=8, loc='lower right', framealpha=0.6)
    c.set_title('损伤度 D(t) 与分级预警')
    # D 预警级别
    dax = ax[3]
    for lv in range(0, 4):
        m = df['level'] == lv
        dax.fill_between(df['pct'], 0, 1, where=m, color=LV_COLORS[lv], step='post',
                         label=f'级别{lv}({"正常/注意/预警/临危".split("/")[lv]})')
    dax.set_ylim(0, 1)
    dax.set_yticks([])
    dax.set_ylabel('预警级别')
    dax.set_title('分级预警输出(0正常-1注意-2预警-3临危)')
    for a2x in ax:
        a2x.set_xlim(0, 99)
        a2x.grid(alpha=0.3)
    ax[-1].set_xlabel('寿命 (%)')
    out = os.path.join(FIGDIR, f'paper_{gid}.png')
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print('已存:', out)


def paper_flowchart():
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis('off')
    boxes = [
        (0.03, '原始多源信号\n光纤 / 声发射(AE) / 应变', '#eef3fb'),
        (0.24, '证据提取(块级,在线自适应)\ne_dmg: 损伤型AE事件能量\ne_full: 全能量累积加速度\ne_strain: 应变std发散', '#fdf2e9'),
        (0.47, '证据融合\nrisk = max(e_ae, e_strain)\n(e_ae = max(e_dmg, e_full))', '#f3f0fb'),
        (0.70, '单调累积损伤度\nD(t): 升快降慢(损伤记忆)\n确认后 latch 快追(0.85可达)', '#e9f7ef'),
        (0.90, '分级预警\n0.25注意 / 0.55预警 / 0.85临危', '#fdecec'),
    ]
    for x, txt, col in boxes:
        ax.add_patch(FancyBboxPatch((x, 0.35), 0.16, 0.42, boxstyle='round,pad=0.012',
                                    fc=col, ec='0.55', lw=1.1, transform=ax.transAxes))
        ax.text(x + 0.08, 0.56, txt, ha='center', va='center', fontsize=8.5,
                transform=ax.transAxes)
    for x0, x1 in [(0.19, 0.24), (0.40, 0.47), (0.63, 0.70), (0.86, 0.90)]:
        ax.add_patch(FancyArrowPatch((x0 + 0.01, 0.56), (x1 - 0.005, 0.56),
                                     arrowstyle='-|>', mutation_scale=16, lw=1.3,
                                     color='0.3', transform=ax.transAxes))
    ax.text(0.5, 0.92, '多源连续损伤度 D(t) 方法流程（在线因果、零标签）', ha='center',
            fontsize=12, fontweight='bold')
    ax.text(0.5, 0.08, '在线每点更新：证据(块) → risk → D 单调累积 → 分级输出', ha='center',
            fontsize=9, color='0.35')
    out = os.path.join(FIGDIR, 'method_flowchart.png')
    fig.savefig(out, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('已存:', out)


def cmd_paper():
    paper_flowchart()
    for g in GROUPS:
        paper_draw_one(g)


# ============================================================
# 分发
# ============================================================
TASKS = {'degree': cmd_degree, 'warning': cmd_warning,
         'curves': cmd_curves, 'paper': cmd_paper}


def main():
    ap = argparse.ArgumentParser(description='阶段② 主样本评估与出图：degree / warning / curves / paper')
    ap.add_argument('tasks', nargs='+', choices=list(TASKS) + ['all'],
                    help='要执行的任务（可多个，或用 all）')
    ap.add_argument('--workers', type=int, default=4,
                    help='多进程 worker 数(默认 4)')
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
