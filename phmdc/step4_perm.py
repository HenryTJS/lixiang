# -*- coding: utf-8 -*-
"""PHM2019 铝搭接件（数据集 D） Step 4 —— 标签置换检验（非平凡性 / 显著性）

要回答的问题
------------
Step 1/2 报出来的 RMSE（1.501 / 1.692 mm）与折内 rho（0.93~0.97）**是不是过拟合**？
本项目**没有外部真值锚**（真实断裂时刻 / NDT），所以无法做"绝对精度验证"。
退一步能做的、且审稿必问的，是**非平凡性检验**：

    零假设 H0：波形/特征里**不含**裂纹长度信息。
    在 H0 下，把标签随机置换后重跑同一套 LOSO，应当得到**同样好**的指标。

若真实指标远优于置换分布（经验 p 值很小），则说明模型学到的是**真信号**，
不是噪声拟合。⚠️ 这**不能**替代外部真值验证，只能把结论从
"我们算出了这些数"升级为"这些数不可能来自噪声"。

置换方案（两个，都做，互为敏感性）
----------------------------------
- `within`（**主用**）：在**每个试件内**打乱 `crack_mm`。保持了
  ① 每个试件的裂纹取值集合；② 试件划分；只破坏"波形 ↔ 裂纹"的对应。
  这是分组数据下的标准零假设。若改为跨试件打乱，会把"试件间裂纹量级差异"
  这一**合法**信息也毁掉，导致检验过于容易通过。
- `global`（次用）：全部样本一起打乱。作为"更粗的零假设"报告，
  用于说明结论对零假设选择不敏感。

为什么能做 500 次（特征级）
--------------------------
用的是 `step1_features_first.csv` —— `--baseline first` 口径，
**特征本身不含标签**（参考 = 该试件最早一次测量），
所以置换标签**不需要重抽特征**，只需重拟合 ⇒ 每次几秒。

⚠️ `--baseline zero` 口径的特征用到了"哪些时刻裂纹 = 0"，置换标签就必须
连同特征一起重抽（贵 ~20 倍）。本脚本**只做 `first` 口径**，
`zero` 的置换检验记入"未做"清单。

输出
----
  phmdc/results/step4_perm_feat.csv    特征级：每次置换 × 每臂的指标
  phmdc/results/step4_perm_cnn.csv     CNN 级（--cnn-perm > 0 时）
  phmdc/results/step4_perm_report.txt

依赖：numpy / pandas / scikit-learn（特征级）；torch（CNN 级，可选）

⚠️ 已知坑（实测踩过，2026-09-23 更正）
-----------------------------------
`RandomForestRegressor(n_jobs=-1)` 在本环境下会**反复抛**
`sklearn.utils.parallel.delayed` 的 UserWarning，量级惊人：
**一次跑批实测写出 1.54 GB 日志**（约 500 万条）。

❌ **错误做法（无效）**：`warnings.filterwarnings('ignore')`、
   `python -W ignore`、`set PYTHONWARNINGS=ignore` **全都压不住** ——
   告警由 **joblib/loky 工作进程** 抛出，子进程会重建自己的警告过滤器，
   环境变量不起作用。（曾经根据"单次耗时变短"误以为已解决，那是假象。）

✅ **正确做法（二选一）**：
   1. **把 stderr 引开**（推荐，不影响并行速度）：
      `python step4_perm.py ... > run.log 2> nul`
      或 `2> warn.log` 后删掉——进度在 stdout，告警在 stderr，两者本来就可分。
   2. `--n-jobs 1`：彻底不生成子进程，日志干净，但 RF 变单线程、
      每次置换耗时约变成 2~3 倍。
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd

import prep
import step1_model as S1

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = prep.ROOT
RES = os.path.join(ROOT, 'results')

_REPORT = []


def say(msg=''):
    # flush=True 是必需的：重定向到文件时默认块缓冲（8 KB），
    # 会让"跑到哪了"完全不可见。
    print(msg, flush=True)
    _REPORT.append(msg)


# ============================================================
# 置换
# ============================================================
def permute_within(df, rng, col='crack_mm'):
    """在每个 (specimen) 内打乱 col。返回新的一列（不修改原 df）。"""
    out = df[col].to_numpy(float).copy()
    for sp, idx in df.groupby('specimen').groups.items():
        pos = df.index.get_indexer(idx)
        out[pos] = rng.permutation(out[pos])
    return out


def permute_global(df, rng, col='crack_mm'):
    """全部样本一起打乱。"""
    return rng.permutation(df[col].to_numpy(float))


# ============================================================
# 特征级 LOSO（轻量版：只留读得快的臂）
# ============================================================
# 说明：不直接用 S1.run_loso 是因为它含 GPR / SVR / 4 个单特征臂，
# 每次约 30~40 s，做 500 次要几小时。这里保留**结论最相关的 4 个臂**：
#   linear / rf  + 各自的物理约束版。指标函数仍复用 S1.metrics（口径一致）。
PERM_ARMS = ['linear', 'linear+mono', 'rf', 'rf+mono']
N_JOBS = -1          # 由 --n-jobs 覆盖；见文件头"已知坑"第 2 条


def loso_once(feat, cols, seed=0):
    """一次完整 LOSO，返回 {臂: 指标 dict} 与真实标签下的逐点预测。"""
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    reps = sorted(feat['rep'].unique())
    rows, n_used = {a: [] for a in PERM_ARMS}, 0
    for rp in reps:
        data = feat[feat['rep'] == rp]
        for sp in prep.SPECIMENS:
            tr = data[data['specimen'] != sp]
            te = data[data['specimen'] == sp]
            if not len(tr) or not len(te):
                continue
            sc = StandardScaler().fit(tr[cols].to_numpy(float))
            Xtr = sc.transform(tr[cols].to_numpy(float))
            Xte = sc.transform(te[cols].to_numpy(float))
            ytr = tr['crack_mm'].to_numpy(float)
            yte = te['crack_mm'].to_numpy(float)
            preds = {
                'linear': Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte),
                'rf': RandomForestRegressor(n_estimators=400,
                                            min_samples_leaf=2,
                                            random_state=seed,
                                            n_jobs=N_JOBS).fit(Xtr, ytr).predict(Xte),
            }
            base = te[['specimen', 'rep', 'cycle']].copy()
            for nm in ('linear', 'rf'):
                preds[nm + '+mono'] = S1.apply_mono(
                    base.assign(_p=preds[nm]), '_p')
            for nm, yp in preds.items():
                m = S1.metrics(yte, yp, base, nm)
                m['fold'] = sp
                m['rep'] = rp
                rows[nm].append(m)
            n_used += len(te)
    # 折间均值（与 phmdc/README.md §5.1.2 / §5.2.1 完全同一口径）
    out = {}
    for nm, lst in rows.items():
        d = pd.DataFrame(lst)
        out[nm] = {'rmse': float(d['rmse'].mean()),
                   'mae': float(d['mae'].mean()),
                   'rho': float(d['rho_within'].mean()),
                   'n': int(d['n'].sum())}
    return out, n_used


def run_feat_perm(n_perm, scheme, seed, feat_path):
    feat0 = pd.read_csv(feat_path)
    cols = S1.feature_cols(feat0)
    say('    特征表: %s（%d 行 × %d 维可用特征）'
        % (os.path.basename(feat_path), len(feat0), len(cols)))
    say('    臂: %s' % '、'.join(PERM_ARMS))
    say('')

    # ---- 真实标签 ----
    t0 = time.time()
    real, n_used = loso_once(feat0, cols, seed=seed)
    dt_real = time.time() - t0
    say('  [真实标签] %d 个样本；耗时 %.1f s' % (n_used, dt_real))
    say('    %-14s %8s %8s %9s' % ('臂', 'RMSE', 'MAE', '折内rho'))
    for a in PERM_ARMS:
        say('    %-14s %8.3f %8.3f %9.3f'
            % (a, real[a]['rmse'], real[a]['mae'], real[a]['rho']))

    # ---- 置换 ----
    say('')
    say('  [置换 %d 次，方案 = %s]' % (n_perm, scheme))
    rng = np.random.default_rng(seed)
    recs = []
    t0 = time.time()
    for k in range(n_perm):
        f = feat0.copy()
        f['crack_mm'] = (permute_within(feat0, rng) if scheme == 'within'
                         else permute_global(feat0, rng))
        st, _ = loso_once(f, cols, seed=seed)
        for a in PERM_ARMS:
            recs.append({'arm': a, 'perm': k, 'rmse': st[a]['rmse'],
                         'mae': st[a]['mae'], 'rho': st[a]['rho']})
        if (k + 1) % 25 == 0 or k == 0:
            el = time.time() - t0
            say('    ... %3d/%d 次，已用 %.0f s，预计总 %.0f s'
                % (k + 1, n_perm, el, el / (k + 1) * n_perm))
    null = pd.DataFrame(recs)

    # ---- p 值与分布 ----
    say('')
    say('  %-14s %8s %12s %12s %10s %9s %9s'
        % ('臂', '真实', '零分布中位', '零分布最小', '零/真实', 'z', 'p 值'))
    say('  ' + '-' * 82)
    rows = []
    for a in PERM_ARMS:
        g = null[null['arm'] == a]['rmse'].to_numpy(float)
        r = real[a]['rmse']
        # 单侧（越小越好）：越小越"好"，故 p = P(T_null <= T_real)
        p = (1 + int((g <= r).sum())) / (1 + len(g))
        z = (float(np.mean(g)) - r) / (float(np.std(g, ddof=1)) + 1e-12)
        rows.append({'arm': a, 'real_rmse': r, 'null_median': float(np.median(g)),
                     'null_min': float(g.min()), 'null_max': float(g.max()),
                     'ratio': float(np.median(g) / r), 'z': z, 'p': p,
                     'n_perm': len(g),
                     'rho_real': real[a]['rho'],
                     'rho_null_median': float(
                         null[null['arm'] == a]['rho'].median())})
        say('  %-14s %8.3f %12.3f %12.3f %10.2f %9.1f %9s'
            % (a, r, np.median(g), g.min(), np.median(g) / r, z,
               ('%.4f' % p) if p > 1e-4 else '<1e-4'))
    say('')
    say('    `零/真实` = 置换分布中位 / 真实值：> 1 表示真实**更好**。')
    say('    `p` = (1 + #{T_perm ≤ T_real}) / (1 + N) —— 单侧经验 p 值（越小越好）。')
    say('    ⚠️ p 的分辨率下界 = 1/(N+1) = %.4f；若 p 刚好等于它，说明'
        % (1.0 / (n_perm + 1)))
    say('       真实值位于**整个**零分布之外（只能报"p < 该值"）。')
    return real, null, pd.DataFrame(rows)


# ============================================================
# CNN 级置换（可选，贵）
# ============================================================
def run_cnn_perm(n_perm, scheme, seeds, epochs, chans, device):
    import torch
    import step2 as S2

    S2.lock_determinism()
    waves, refs, meta = S2.build_dataset(norm='first')
    X = S2.to_input(waves, meta, refs, 'wave', chans=chans)
    say('    样本 %d 个；输入形状 %s；通道数 %d'
        % (len(meta), X.shape, chans))
    say('    配置：seeds=%d epochs=%d device=%s（**为控成本而降配**，'
        % (seeds, epochs, device))
    say('        故本节的"真实值"与 §5.2.1 的 1.692（seeds=5）不可直接比）')
    say('')

    def one(labels):
        m = meta.copy().reset_index(drop=True)
        m['crack_mm'] = labels
        yp_all = np.full(len(m), np.nan)
        for sp in prep.SPECIMENS:
            tr = (m['specimen'] != sp).to_numpy()
            if not tr.any() or not (~tr).any():
                continue
            yp, _ = S2.train_fold(X[tr], m.loc[tr, 'crack_mm'].to_numpy(float),
                                  X[~tr], 'wave', seeds, epochs, device,
                                  chans=chans)
            yp_all[~tr] = yp
        m['pred'] = yp_all
        m['cnn'] = yp_all
        m['cnn+mono'] = S2.apply_mono(m, 'pred')
        # ⚠️ 指标必须按 **(留出试件, 重复)** 分折算，再折间均值 ——
        #    与 step2.py 的 [4] 总体指标完全同一口径；若只按 rep 分组，
        #    会把 8 个试件混在一起，得到的是另一个量。
        out = {}
        for nm in ('cnn', 'cnn+mono'):
            ms = [S2.metrics(g['crack_mm'].to_numpy(float),
                             g[nm].to_numpy(float),
                             g[['specimen', 'rep', 'cycle']], nm)
                  for _, g in m.groupby(['specimen', 'rep'])]
            d = pd.DataFrame(ms)
            out[nm] = {'rmse': float(d['rmse'].mean()),
                       'mae': float(d['mae'].mean()),
                       'rho': float(d['rho_within'].mean()),
                       'fa': (float(d['false_alarm'].mean())
                              if d['false_alarm'].notna().any() else np.nan)}
        return out

    t0 = time.time()
    real = one(meta['crack_mm'].to_numpy(float))
    dt = time.time() - t0
    say('  [真实标签] 单次耗时 %.1f s' % dt)
    for nm, v in real.items():
        say('    %-12s RMSE %.3f  MAE %.3f  折内rho %.3f  零裂纹误报 %.0f%%'
            % (nm, v['rmse'], v['mae'], v['rho'], 100 * v['fa']))

    say('')
    say('  [置换 %d 次，方案 = %s，预计 %.0f min]'
        % (n_perm, scheme, n_perm * dt / 60))
    rng = np.random.default_rng(0)
    recs = []
    for k in range(n_perm):
        lab = (permute_within(meta, rng) if scheme == 'within'
               else permute_global(meta, rng))
        st = one(lab)
        for nm, v in st.items():
            recs.append({'arm': nm, 'perm': k, 'rmse': v['rmse'],
                         'mae': v['mae'], 'rho': v['rho'], 'fa': v['fa']})
        say('    ... %2d/%d' % (k + 1, n_perm))
    null = pd.DataFrame(recs)

    say('')
    say('  %-12s %8s %12s %12s %10s %9s %9s'
        % ('臂', '真实', '零分布中位', '零分布最小', '零/真实', 'z', 'p 值'))
    say('  ' + '-' * 78)
    rows = []
    for nm in ('cnn', 'cnn+mono'):
        g = null[null['arm'] == nm]['rmse'].to_numpy(float)
        r = real[nm]['rmse']
        p = (1 + int((g <= r).sum())) / (1 + len(g))
        z = (float(np.mean(g)) - r) / (float(np.std(g, ddof=1)) + 1e-12)
        rows.append({'arm': nm, 'real_rmse': r, 'null_median': float(np.median(g)),
                     'null_min': float(g.min()), 'ratio': float(np.median(g) / r),
                     'z': z, 'p': p, 'n_perm': len(g),
                     'rho_real': real[nm]['rho'],
                     'rho_null_median': float(null[null['arm'] == nm]['rho'].median())})
        say('  %-12s %8.3f %12.3f %12.3f %10.2f %9.1f %9s'
            % (nm, r, np.median(g), g.min(), np.median(g) / r, z,
               ('%.4f' % p) if p > 1e-4 else '<1e-4'))
    return real, null, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description='phmdc Step 4 标签置换检验（非平凡性）')
    ap.add_argument('--n-perm', type=int, default=500,
                    help='within 方案的置换次数（默认 500）')
    ap.add_argument('--n-perm-global', type=int, default=0,
                    help='global 方案的置换次数（0 = 与 --n-perm 相同）；'
                         'global 只是敏感性检查，通常小于 within 即可')
    ap.add_argument('--scheme', choices=['within', 'global', 'both'],
                    default='both', help='置换方案')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-jobs', type=int, default=-1,
                    help='RF 的并行度；-1 = 用满核（但 joblib 子进程会刷屏告警，'
                         '见文件头"已知坑"）。要干净日志就用 1。')
    ap.add_argument('--feat', default='results/step1_features_first.csv',
                    help='**必须用 first 口径**（特征不含标签）')
    ap.add_argument('--cnn-perm', type=int, default=0,
                    help='CNN 级置换次数（0 = 跳过；每次约 1 min）')
    ap.add_argument('--cnn-seeds', type=int, default=1)
    ap.add_argument('--cnn-epochs', type=int, default=200)
    ap.add_argument('--chans', type=int, default=2)
    ap.add_argument('--device', default='cpu',
                    help='CPU 才逐位确定；GPU 亦可（已锁 cudnn 确定性）')
    args = ap.parse_args()

    global N_JOBS
    N_JOBS = int(args.n_jobs)

    feat_path = args.feat if os.path.isabs(args.feat) else os.path.join(ROOT, args.feat)

    say('=' * 96)
    say('PHM2019 铝搭接件（数据集 D） Step 4 —— 标签置换检验')
    say('生成时间: %s' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    say('=' * 96)
    say('')
    say('[0] 这个检验在回答什么')
    say('')
    say('    本项目**没有外部真值锚**（真实断裂时刻 / NDT）⇒ 无法做绝对精度验证。')
    say('    退一步能做的：**非平凡性检验** —— 证明指标不是噪声拟合。')
    say('    H0：波形/特征里不含裂纹信息 ⇒ 置换标签后重跑同一套 LOSO，指标应同样好。')
    say('    ⚠️ 通过本检验 **不等于** 通过外部真值验证，对外表述必须区分二者。')
    say('')
    say('    统计量 = 折间均值 RMSE（与 §5.1.2 / §5.2.1 同口径）')
    say('    p 值   = (1 + #{T_perm ≤ T_real}) / (1 + N)（单侧，越小越好）')

    schemes = ['within', 'global'] if args.scheme == 'both' else [args.scheme]
    n_global = args.n_perm_global if args.n_perm_global > 0 else args.n_perm
    all_sum = []
    for sc in schemes:
        n_this = n_global if sc == 'global' else args.n_perm
        say('')
        say('=' * 96)
        say('[1] 特征级置换（口径 first；方案 %s；N = %d）' % (sc, n_this))
        say('=' * 96)
        if sc == 'within':
            say('    ⚠️ 主用方案：**试件内**打乱。保持每个试件的裂纹取值集合与试件划分，')
            say('       只破坏"波形 ↔ 裂纹"对应；若跨试件打乱，会把"试件间裂纹量级差异"')
            say('       这一**合法**信息也毁掉 ⇒ 检验会过于容易通过。')
        else:
            say('    ⚠️ 次用方案：全部样本一起打乱（更粗的零假设，用于敏感性）。')
        say('')
        real, null, summ = run_feat_perm(n_this, sc, args.seed, feat_path)
        summ['scheme'] = sc
        all_sum.append(summ)
        null.to_csv(os.path.join(RES, 'step4_perm_feat_%s.csv' % sc),
                    index=False, encoding='utf-8-sig')
        summ.to_csv(os.path.join(RES, 'step4_perm_feat_%s_summary.csv' % sc),
                    index=False, encoding='utf-8-sig')

    if args.cnn_perm > 0:
        say('')
        say('=' * 96)
        say('[2] CNN 级置换（方案 within；`--cnn-perm %d`）' % args.cnn_perm)
        say('=' * 96)
        _, null_c, summ_c = run_cnn_perm(args.cnn_perm, 'within', args.cnn_seeds,
                                         args.cnn_epochs, args.chans,
                                         args.device)
        null_c.to_csv(os.path.join(RES, 'step4_perm_cnn.csv'),
                      index=False, encoding='utf-8-sig')
        summ_c.to_csv(os.path.join(RES, 'step4_perm_cnn_summary.csv'),
                      index=False, encoding='utf-8-sig')
    else:
        say('')
        say('[2] CNN 级置换：跳过（`--cnn-perm 30` 开启；每次约 0.5~1 min）')

    # ---------- 结论 ----------
    say('')
    say('=' * 96)
    say('[3] 结论与口径')
    say('=' * 96)
    for s in all_sum:
        w = s[s['arm'].str.endswith('+mono')]
        for r in w.itertuples():
            say('    %-6s %-14s 真实 RMSE %.3f vs 零分布中位 %.3f（比值 %.2f）；'
                % (s['scheme'].iloc[0], r.arm, r.real_rmse, r.null_median, r.ratio))
            say('    %-6s %-14s 折内 rho：真实 %.3f vs 零分布中位 %.3f；'
                % ('', '', r.rho_real, r.rho_null_median))
            say('    %-6s %-14s p %s' % ('', '', ('%.4f' % r.p)
                                        if r.p > 1e-4 else '<1e-4'))
    say('')
    say('    ⇒ 若 p 极小且折内 rho 从 ~0 抬到 ~0.9，则：**特征确实携带裂纹信息**，')
    say('      Step 1/2 的指标不是过拟合。')
    say('    ⇒ 但必须同时写明：**这不构成对外部真值的验证**。本项目能对外声称的是')
    say('      「方法在有真值的平台上被**标定**、在无真值数据上被**统计验证**」，')
    say('      而不是「精度已被验证」。')

    with open(os.path.join(RES, 'step4_perm_report.txt'), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(_REPORT) + '\n')
    say('')
    say('[产出] results/step4_perm_feat_*.csv · step4_perm_cnn*.csv · '
        'step4_perm_report.txt')


if __name__ == '__main__':
    main()
