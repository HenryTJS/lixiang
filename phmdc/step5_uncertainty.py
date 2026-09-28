# -*- coding: utf-8 -*-
"""phmdc Step 5 —— 指标的不确定性量化（t 区间 / bootstrap / 配对检验）

为什么需要
----------
Step 1/2 目前报的都是**折间均值的点估计**（如 `cnn+mono` RMSE 1.692 mm）。
但只有 **8 个试件** ⇒ 任何"谁比谁好"的断言都必须带不确定性，否则不可引用。
主样本（数据集 A）已经在主 `README.md` §3.10.4 做过同样的事，这里补齐 phmdc。

⚠️ 三条必须一起写进口径说明的局限（否则 CI 会被误读成"真实精度范围"）
------------------------------------------------------------------------
1. **LOSO 的折不是独立观测**：每折的训练集共享 7/8 个试件 ⇒ 8 个折误差
   **正相关**。因此下面所有 t-CI / bootstrap-CI 都**低估**了真实不确定性，
   只能当作**乐观下界**。诚实做法是同时给 `均值 ± sd`、t-CI、bootstrap-CI
   与**逐试件明细**，并声明该局限。
2. **bootstrap 在 N=8 时偏窄**：主 README §3.10.4 已量化过（宽度 1.85 vs t 的 2.97）。
   故**以 t 区间为主**，bootstrap 作对照。
3. **单位是试件不是样本**：同一次重复测量（rep1/rep2）不独立，先按试件取
   rep 均值，再在 8 个试件上统计 ⇒ 避免把 n 从 8 虚增到 16/92。

配对检验
--------
折是**配对**的（同一留出试件），所以"臂 A vs 臂 B"应当用**配对**检验
（逐试件差值 → paired t / Wilcoxon），而不是比较两个独立的均值。
这比分别给 CI 更有力，也是本步的主产出。

输出
----
  phmdc/results/step5_uncertainty.csv        逐臂的均值/sd/t-CI/bootstrap-CI
  phmdc/results/step5_per_specimen.csv       逐 (臂, 试件) 明细
  phmdc/results/step5_paired.csv             配对比较（含差值 CI 与 p 值）
  phmdc/results/step5_uncertainty_report.txt

依赖：numpy / pandas / scipy
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')

_REPORT = []


def say(msg=''):
    print(msg, flush=True)
    _REPORT.append(msg)


# ============================================================
# 工具
# ============================================================
def t_ci(x):
    """t 分布 95% CI（双侧）。返回 (lo, hi, half_width, t_crit, df)。"""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, n - 1
    m, sd = float(np.mean(x)), float(np.std(x, ddof=1))
    tc = float(stats.t.ppf(0.975, n - 1))
    hw = tc * sd / np.sqrt(n)
    return m - hw, m + hw, hw, tc, n - 1


def boot_ci(x, n_boot=20000, seed=0):
    """对**试件**做 bootstrap（有放回重抽 n 个），百分位 95% CI。"""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_specimen(df, arm, metric='rmse'):
    """按 (折=留出试件) 先对 rep 取均值 → 每个试件一个值。返回 (specimens, values)。"""
    g = df[df['arm'] == arm]
    if not len(g):
        return np.array([]), np.array([])
    s = g.groupby('fold')[metric].mean().sort_index()
    return s.index.to_numpy(), s.to_numpy(float)


def load_sources():
    """收集所有可用的 (来源标签, DataFrame)。"""
    srcs = []
    f = os.path.join(RES, 'step1_loso.csv')
    if os.path.exists(f):
        srcs.append(('Step1(mean_zero)', pd.read_csv(f)))
    f = os.path.join(RES, 'step1_loso_first.csv')
    if os.path.exists(f):
        srcs.append(('Step1(first)', pd.read_csv(f)))
    for f in sorted(glob.glob(os.path.join(RES, 'step2_loso*.csv'))):
        tag = os.path.basename(f).replace('step2_loso', '').replace('.csv', '') \
            or '_default'
        srcs.append(('Step2' + tag, pd.read_csv(f)))
    return srcs


# ============================================================
# 主流程
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='phmdc Step 5 不确定性量化')
    ap.add_argument('--boot', type=int, default=20000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    say('=' * 100)
    say('phmdc Step 5 —— 指标的不确定性量化（t 区间 / bootstrap / 配对检验）')
    say('生成时间: %s' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    say('=' * 100)

    srcs = load_sources()
    if not srcs:
        raise SystemExit('未找到 step1_loso*.csv / step2_loso*.csv，请先跑 Step 1/2')

    # ---------- [1] 逐臂的不确定性 ----------
    say('')
    say('[1] 逐臂 RMSE：点估计 + 不确定性（单位 = 试件，先按 rep 取均值）')
    say('')
    say('  %-26s %-14s %8s %7s %8s %20s %20s'
        % ('来源', '臂', 'RMSE', 'sd', 'n_spec', 't 95% CI', 'bootstrap 95% CI'))
    say('  ' + '-' * 110)
    rows, per = [], []
    for tag, df in srcs:
        for arm in sorted(df['arm'].unique()):
            sp, v = per_specimen(df, arm)
            if len(v) < 2:
                continue
            lo, hi, hw, tc, dof = t_ci(v)
            blo, bhi = boot_ci(v, args.boot, args.seed)
            rows.append({'source': tag, 'arm': arm, 'n_spec': len(v),
                         'rmse_mean': float(np.mean(v)), 'rmse_sd': float(np.std(v, ddof=1)),
                         'ci_t_lo': lo, 'ci_t_hi': hi, 'ci_t_hw': hw,
                         't_crit': tc, 'dof': dof,
                         'ci_boot_lo': blo, 'ci_boot_hi': bhi,
                         'boot_hw': (bhi - blo) / 2})
            for s, x in zip(sp, v):
                per.append({'source': tag, 'arm': arm, 'specimen': s, 'rmse': x})
            say('  %-26s %-14s %8.3f %7.3f %8d %20s %20s'
                % (tag, arm, np.mean(v), np.std(v, ddof=1), len(v),
                   '[%.3f, %.3f]' % (lo, hi), '[%.3f, %.3f]' % (blo, bhi)))
    und = pd.DataFrame(rows)
    und.to_csv(os.path.join(RES, 'step5_uncertainty.csv'), index=False,
               encoding='utf-8-sig')
    pd.DataFrame(per).to_csv(os.path.join(RES, 'step5_per_specimen.csv'),
                             index=False, encoding='utf-8-sig')

    say('')
    say('  ⚠️ **三条口径局限（必须随表一起报）**：')
    say('     ① LOSO 的 8 个折**不独立**（训练集共享 7/8 试件）⇒ 所有 CI 都**低估**')
    say('        不确定性，只能当**乐观下界**；')
    say('     ② N=8 时 bootstrap 偏窄（主 README §3.10.4 已量化）⇒ **以 t 区间为主**；')
    say('     ③ 单位是**试件**（先按 rep 取均值），不是 16 个"样本"。')

    # ---------- [2] 配对比较 ----------
    say('')
    say('[2] 配对比较（折 = 配对单位；这是"谁比谁好"的正确检验）')
    say('')
    say('  %-24s %-14s %-14s %9s %19s %8s %10s'
        % ('来源', '臂 A', '臂 B', 'Δ均值', 'Δ 的 t 95% CI', 'p(t)', 'p(Wilcoxon)'))
    say('  ' + '-' * 108)
    PAIRS = [
        ('Step2_wave_first_c2', 'cnn', 'cnn+mono'),        # 约束层是否逐试件有效
        ('Step1(mean_zero)', 'rf', 'rf+mono'),             # 同上（特征侧）
        ('Step1(mean_zero)', 'linear', 'linear+mono'),
        ('Step2_wave_first_c2', 'cnn+mono', None),         # 占位：跨来源比
    ]
    prows = []
    by_tag = {t: d for t, d in srcs}

    def paired(tagA, aA, tagB, aB, label=None):
        dA, dB = by_tag.get(tagA), by_tag.get(tagB)
        if dA is None or dB is None:
            return
        _, va = per_specimen(dA, aA)
        _, vb = per_specimen(dB, aB)
        if len(va) != len(vb) or len(va) < 3:
            say('  %-24s %-14s %-14s  （试件集合不一致或 n<3，跳过）'
                % (tagB, aA, aB))
            return
        d = vb - va
        lo, hi, hw, tc, dof = t_ci(d)
        pt = float(stats.ttest_rel(vb, va).pvalue)
        try:
            pw = float(stats.wilcoxon(vb, va).pvalue)
        except Exception:
            pw = np.nan
        prows.append({'tagA': tagA, 'armA': aA, 'tagB': tagB, 'armB': aB,
                      'delta_mean': float(np.mean(d)), 'ci_lo': lo, 'ci_hi': hi,
                      'n_spec': len(d), 'p_ttest': pt, 'p_wilcoxon': pw})
        say('  %-24s %-14s %-14s %9.3f %19s %8.4f %10s'
            % (tagB, aA, aB, np.mean(d), '[%.3f, %.3f]' % (lo, hi), pt,
               ('%.4f' % pw) if np.isfinite(pw) else '—'))

    # 同来源内的配对（约束层效果）
    for tag, a, b in [(t, 'cnn', 'cnn+mono') for t in by_tag if t.startswith('Step2')]:
        paired(tag, a, tag, b)
    for tag in ('Step1(mean_zero)', 'Step1(first)'):
        for a, b in (('linear', 'linear+mono'), ('rf', 'rf+mono'),
                     ('svr', 'svr+mono'), ('gpr', 'gpr+mono')):
            paired(tag, a, tag, b)

    say('')
    say('  —— 跨来源：DL 最好臂 vs 特征工程最好臂（同 `first` 口径才可比）——')
    paired('Step1(first)', 'rf+mono', 'Step2_wave_first_c2', 'cnn+mono')
    paired('Step1(first)', 'linear+mono', 'Step2_wave_first_c2', 'cnn+mono')
    paired('Step2_wave_first_c2', 'cnn+mono', 'Step2_wave_first_c2_anch',
           'cnn+mono+anchor')
    paired('Step2_wave_zero_c2', 'cnn+mono', 'Step2_wave_first_c2', 'cnn+mono')

    # —— 第二通道消融（Step 6）：真实差值 vs "不含信息"的对照 ——
    #    若 shuffle/rand/dup/zero 都明显更差且回落单通道水平 ⇒ 增益来自**信息**。
    say('')
    say('  —— 第二通道消融（§5.6）：真实差值 vs 不含信息的对照 ——')
    say('     （对照组文件名带 `_ch2<variant>`；未跑过的会自动跳过）')
    paired('Step2_wave_first_c2', 'cnn+mono', 'Step2_wave_first_c2_ch2shuffle',
           'cnn+mono')
    paired('Step2_wave_first_c2', 'cnn+mono', 'Step2_wave_first_c2_ch2rand',
           'cnn+mono')
    paired('Step2_wave_first_c2', 'cnn+mono', 'Step2_wave_first_c2_ch2dup',
           'cnn+mono')
    paired('Step2_wave_first_c2', 'cnn+mono', 'Step2_wave_first_c2_ch2zero',
           'cnn+mono')
    say('     —— 与单通道对比（1 通道 = `Step2_default`）——')
    paired('Step2_default', 'cnn+mono', 'Step2_wave_first_c2', 'cnn+mono')
    paired('Step2_default', 'cnn+mono', 'Step2_wave_first_c2_ch2shuffle',
           'cnn+mono')

    if prows:
        pd.DataFrame(prows).to_csv(os.path.join(RES, 'step5_paired.csv'),
                                   index=False, encoding='utf-8-sig')

    say('')
    say('  读法：`Δ均值 = RMSE(B) − RMSE(A)`，**负值 = B 更好**。')
    say('  `p(t)` = 配对 t 检验（df = 试件数−1）；`p(Wilcoxon)` = 配对符号秩（不假设正态）。')
    say('  ⚠️ 同一局限仍然成立：8 个折不独立 ⇒ p 值同样**偏乐观**。')
    say('     N=8 时配对检验的功效有限，**不显著 ≠ 等效**；应结合 §5.4 的置换检验一起读。')

    # ---------- [3] 外推（点数太少，只给逐点） ----------
    f = os.path.join(RES, 'step3_extrap.csv')
    if os.path.exists(f):
        ex = pd.read_csv(f)
        say('')
        say('[3] 外推（T7/T8）：目标点只有 4/5 个 ⇒ **不给 CI**，只报逐点误差')
        say('')
        b = ex[ex['phase'] == 'B_trajectory']
        say('  %-6s %-12s %6s %8s %9s %9s' % ('试件', '律', 'n', 'RMSE', 'MAE', '最大误差'))
        say('  ' + '-' * 56)
        for (sp, law), g in b.groupby(['specimen', 'law']):
            e = g['err'].dropna()
            if not len(e):
                continue
            say('  %-6s %-12s %6d %8.3f %9.3f %9.3f'
                % (sp, law, len(e), float(np.sqrt(np.mean(e ** 2))),
                   float(np.abs(e).mean()), float(np.abs(e).max())))
        say('')
        say('  ⇒ 4~5 个点的均值不足以支撑区间估计；报均值时必须同时给逐点明细。')

    # ---------- 结论 ----------
    say('')
    say('=' * 100)
    say('[4] 结论与对外表述口径')
    say('=' * 100)
    say('  1. **所有指标都应带不确定性报出**，而不是只报折间均值；')
    say('  2. **"谁比谁好"用配对检验**（同一折配对），而不是比较两个独立均值；')
    say('  3. 🔴 **必须同时声明局限**：8 折不独立 ⇒ CI 与 p 都是**乐观**的；')
    say('     N=8 时"不显著"不能读成"等效"。')
    say('  4. 真正能把上述乐观性去掉的只有**外部真值锚**（目前没有）。')

    with open(os.path.join(RES, 'step5_uncertainty_report.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(_REPORT) + '\n')
    say('')
    say('[产出] results/step5_uncertainty.csv · step5_per_specimen.csv · '
        'step5_paired.csv · step5_uncertainty_report.txt')


if __name__ == '__main__':
    main()
