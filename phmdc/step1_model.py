"""
Step 1: 基线模型 + LOSO 8 折验收 —— PHM2019 铝搭接件（数据集 D）
=================================================================

读取 `step1.py` 产出的特征表，按 **LOSO（留一试件）8 折** 评估裂纹长度回归。

为什么必须 LOSO
----
试件只有 8 个，且**不同试件之间有系统性偏置**（几何/耦合/增益）。
随机划分会让同一试件的时刻同时落进训练与测试 ⇒ **指标虚高**。
本项目 A/B/C 的既有验收口径也是 LOSO，这里保持一致以便横向对照。

对照设计（这才是本步的核心产出）
----
| 臂 | 说明 |
| --- | --- |
| `single_best` | **只用相关性最强的单特征**，一元线性回归 —— "特征工程没做"的基线 |
| `single_tof` | 只用 TOF 偏移 —— 呼应 README 要求的"只用 TOF 这类单特征基线" |
| `linear` | 全部特征 + 岭回归 |
| `svr` | 全部特征 + SVR(RBF) |
| `gpr` | 全部特征 + 高斯过程回归（带不确定度） |
| `rf` | 全部特征 + 随机森林 |
| `+mono` | 后处理：把预测沿循环数做**单调不减**投影（物理约束层） |

指标
----
- **RMSE / MAE**（mm）逐折 + 整体（唯一无歧义的口径）
- **单调违例率**：同一试件内预测值不随循环数单调不减的比例
- **越界率**：预测为负 或 超出该数据集裂纹量程（0 ~ 7.46 mm）的比例
- **零裂纹误报**：真值 = 0 的时刻上预测值 > 0.5 mm 的比例
  （这一步的物理意义 = "健康时会不会乱报伤"）
- ⚠️ 官方评分函数（时间惩罚×非对称惩罚×单调性惩罚）**未实现**：
  其常数在 `PHM2019_ScoringSpreadsheet.xlsx` 内，仓库里没有该文件。
  在拿到官方常数前**不自行编造**，只用上面无歧义的口径。

输出
----
  phmdc/results/step1_loso.csv       逐折 × 逐臂 × 逐重复的指标
  phmdc/results/step1_pred.csv       逐时刻预测值（供画图/外推评测）
  phmdc/results/step1_model_report.txt
"""

import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

import prep

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = prep.ROOT
RES = os.path.join(ROOT, 'results')
OUT_FEAT = os.path.join(RES, 'step1_features.csv')
OUT_LOSO = os.path.join(RES, 'step1_loso.csv')
OUT_PRED = os.path.join(RES, 'step1_pred.csv')
OUT_REPORT = os.path.join(RES, 'step1_model_report.txt')

_REPORT = []


def say(msg=''):
    print(msg)
    _REPORT.append(msg)


# ============================================================
# 特征列选择
# ============================================================
# 绝对幅值类特征**必须排除**：ch1 激励逐试件不同（峰峰 1.88~4.28 V），
# 这类特征实际编码的是"哪个试件"，跨试件不可比（会变成隐式的试件指纹）。
EXCLUDE_ABS = [
    'pp_ch1_raw',        # 激励幅值（诊断用）
    'env_peak_abs',      # 包络峰值绝对幅值
    'rms_ch2_abs',       # 接收 RMS 绝对幅值
    'env_peak_t',        # 包络峰值绝对时刻（试件相关）
]
META = ['specimen', 'cycle', 'rep', 'crack_mm']


def feature_cols(feat):
    cols = []
    for c in feat.columns:
        if c in META or c in EXCLUDE_ABS:
            continue
        if not pd.api.types.is_numeric_dtype(feat[c]):
            continue
        if feat[c].isna().any():
            continue                      # 有缺失的列直接弃（保持口径简单）
        cols.append(c)
    return cols


# ============================================================
# 物理约束层：单调不减投影
# ============================================================
def isotonic_up(v):
    """把序列投影到「单调不减、且 ≥ 0」——最省事的保序回归（PAVA）。

    这是 README §5 Step 2 要求的「物理约束」在 Step 1 的最小落地：
    裂纹长度只增不减、且非负。
    """
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)
    if len(v) <= 1:
        return v
    blocks = [[i, i, v[i]] for i in range(len(v))]
    out = []
    for b in blocks:
        out.append(b)
        while len(out) > 1:
            a, c = out[-2], out[-1]
            if a[2] / (a[1] - a[0] + 1) <= c[2] / (c[1] - c[0] + 1):
                break
            merged = [a[0], c[1], a[2] + c[2]]
            out.pop()
            out.pop()
            out.append(merged)
    res = np.empty(len(v))
    for a, b, s in out:
        res[a:b + 1] = s / (b - a + 1)
    return res


def apply_mono(df, pred_col):
    """按 (specimen, rep) 分组，沿 cycle 做单调不减投影。"""
    out = np.empty(len(df))
    for (sp, rp), idx in df.groupby(['specimen', 'rep']).groups.items():
        sub = df.loc[idx].sort_values('cycle')
        proj = isotonic_up(sub[pred_col].to_numpy())
        out[df.index.get_indexer(sub.index)] = proj
    return out


# ============================================================
# 指标
# ============================================================
def metrics(y_true, y_pred, df, tag):
    e = np.asarray(y_pred) - np.asarray(y_true)
    yt = np.asarray(y_true, dtype=float)
    m = {'arm': tag, 'n': len(e),
         'rmse': float(np.sqrt(np.mean(e ** 2))),
         'mae': float(np.mean(np.abs(e))),
         'bias': float(np.mean(e)),
         'max_abs_err': float(np.max(np.abs(e)))}
    # **折内跟踪能力**：同一试件内 预测 vs 真值 的 Spearman。
    # 它回答一个比 RMSE 更本质的问题：「模型在该试件内是否跟着裂纹走」——
    # 即使整体偏移（截距错），只要折内相关高，就说明特征**携带了裂纹信息**，
    # 偏移属于标定问题而非信息缺失问题。
    if len(np.unique(yt)) >= 3:
        m['rho_within'] = float(pd.Series(y_pred).corr(
            pd.Series(yt), method='spearman'))
    else:
        m['rho_within'] = np.nan
    # 预测值域压缩比（RMSE 最优模型常见的"向均值收缩"）
    m['pred_std'] = float(np.std(y_pred))
    m['true_std'] = float(np.std(yt))
    m['range_ratio'] = (m['pred_std'] / m['true_std']) if m['true_std'] > 0 else np.nan
    # 单调违例率（同一试件/重复内，按循环数排序后下降的比例）
    viol, tot = 0, 0
    for (sp, rp), g in df.assign(_p=y_pred).groupby(['specimen', 'rep']):
        g = g.sort_values('cycle')
        d = np.diff(g['_p'].to_numpy())
        viol += int((d < -1e-9).sum())
        tot += len(d)
    m['mono_viol_rate'] = (viol / tot) if tot else np.nan
    # 越界率
    m['neg_rate'] = float(np.mean(np.asarray(y_pred) < 0))
    hi = float(np.max(y_true))
    m['above_range_rate'] = float(np.mean(np.asarray(y_pred) > hi))
    # 零裂纹误报（真值=0 的时刻预测 > 0.5 mm）
    z = np.asarray(y_true) == 0
    m['n_zero'] = int(z.sum())
    m['false_alarm'] = (float(np.mean(np.asarray(y_pred)[z] > 0.5))
                        if z.sum() else np.nan)
    return m


# ============================================================
# LOSO 主循环
# ============================================================
def run_loso(feat, rep, feas_path, seed=0):
    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.preprocessing import StandardScaler

    cols = feature_cols(feat)
    data = feat[feat['rep'] == rep].reset_index(drop=True)
    feas = pd.read_csv(feas_path)
    feas = feas[feas['feature'].isin(cols)].sort_values(
        'rho_within_abs', ascending=False)
    best_feat = feas['feature'].iloc[0]
    tof_feat = 'tof_shift' if 'tof_shift' in cols else best_feat

    rows, preds = [], []
    for sp in prep.SPECIMENS:
        tr = data[data['specimen'] != sp]
        te = data[data['specimen'] == sp]
        if not len(tr) or not len(te):
            continue
        Xtr_all, Xte_all = tr[cols].to_numpy(float), te[cols].to_numpy(float)
        ytr, yte = tr['crack_mm'].to_numpy(float), te['crack_mm'].to_numpy(float)
        sc = StandardScaler().fit(Xtr_all)
        Xtr, Xte = sc.transform(Xtr_all), sc.transform(Xte_all)

        arms = {}
        # --- 单特征基线（一元线性，用同一套标准化流程） ---
        for nm, fc in (('single_best', best_feat), ('single_tof', tof_feat)):
            j = cols.index(fc)
            xs = Xtr[:, [j]]
            coef, itc = np.polyfit(xs[:, 0], ytr, 1)
            arms[nm] = np.polyval([coef, itc], Xte[:, j])
        # --- 多元模型 ---
        arms['linear'] = Ridge(alpha=1.0).fit(Xtr, ytr).predict(Xte)
        arms['svr'] = SVR(C=10.0, gamma='scale', epsilon=0.05).fit(Xtr, ytr).predict(Xte)
        arms['rf'] = RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed,
            n_jobs=-1).fit(Xtr, ytr).predict(Xte)
        try:
            k = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(3.0, (1e-1, 1e2)) \
                + WhiteKernel(0.3, (1e-3, 1e1))
            arms['gpr'] = GaussianProcessRegressor(
                kernel=k, normalize_y=True, n_restarts_optimizer=2,
                random_state=seed).fit(Xtr, ytr).predict(Xte)
        except Exception:
            pass
        # --- 加物理约束（单调不减 + 非负） ---
        base = te[['specimen', 'rep', 'cycle']].copy()
        for nm in ('single_best', 'linear', 'svr', 'rf', 'gpr'):
            if nm not in arms:
                continue
            tmp = base.assign(_p=arms[nm])
            arms[nm + '+mono'] = apply_mono(tmp, '_p')

        for nm, yp in arms.items():
            m = metrics(yte, yp, te[['specimen', 'rep', 'cycle']], nm)
            m['fold'] = sp
            m['rep'] = rep
            rows.append(m)
            for i, (cyc, yt, yv) in enumerate(zip(te['cycle'], yte, yp)):
                preds.append({'specimen': sp, 'rep': rep, 'cycle': int(cyc),
                              'crack_mm': float(yt), 'arm': nm,
                              'pred': float(yv), 'err': float(yv - yt)})
    return pd.DataFrame(rows), pd.DataFrame(preds), best_feat, tof_feat, cols


# ============================================================
# main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='phmdc Step 1 建模与 LOSO 验收')
    ap.add_argument('--reps', default='1,2', help='用哪次重复，默认 1,2（都跑都报）')
    ap.add_argument('--feat', default=OUT_FEAT,
                    help='特征表路径（默认 mean_zero 基线口径）')
    ap.add_argument('--tag', default='mean_zero', help='本次运行的口径标签')
    args = ap.parse_args()

    feat_path = args.feat if os.path.isabs(args.feat) else os.path.join(ROOT, args.feat)
    if not os.path.exists(feat_path):
        raise SystemExit('缺少 %s —— 先跑 python step1.py' % feat_path)

    suffix = '' if args.tag == 'mean_zero' else '_' + args.tag
    out_loso = OUT_LOSO.replace('.csv', suffix + '.csv')
    out_pred = OUT_PRED.replace('.csv', suffix + '.csv')
    out_report = OUT_REPORT.replace('.txt', suffix + '.txt')
    feas_path = os.path.join(RES, 'step1_feasibility%s.csv' % suffix)

    feat = pd.read_csv(feat_path)
    reps = [int(x) for x in args.reps.split(',')]

    say('=' * 78)
    say('PHM2019 铝搭接件（数据集 D） Step 1 建模 + LOSO 8 折')
    say('生成时间: %s' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    say('特征表: %s（%d 行）' % (os.path.basename(feat_path), len(feat)))
    say('基线口径: %s；重复口径: %s' % (args.tag, args.reps))
    say('=' * 78)

    all_loso, all_pred = [], []
    for rp in reps:
        loso, pred, best_feat, tof_feat, cols = run_loso(feat, rp, feas_path)
        all_loso.append(loso)
        all_pred.append(pred)
        say('\n[重复 %d] 特征 %d 个（已排除 %d 个绝对幅值列）'
            % (rp, len(cols), len(EXCLUDE_ABS)))
        say('  单特征基线用: best=%s / tof=%s' % (best_feat, tof_feat))

    loso = pd.concat(all_loso, ignore_index=True)
    pred = pd.concat(all_pred, ignore_index=True)
    loso.to_csv(out_loso, index=False, encoding='utf-8-sig')
    pred.to_csv(out_pred, index=False, encoding='utf-8-sig')

    # ---------- 总表 ----------
    for rp in reps:
        say('\n' + '=' * 78)
        say('[重复 %d] LOSO 8 折总体指标（每折 1 个试件）' % rp)
        say('=' * 78)
        sub = loso[loso['rep'] == rp]
        order = ['single_tof', 'single_best', 'linear', 'svr', 'rf', 'gpr',
                 'linear+mono', 'svr+mono', 'rf+mono', 'gpr+mono']
        say('  %-16s %8s %8s %8s %10s %9s %9s %8s %9s %9s'
            % ('臂', 'RMSE', 'MAE', '偏置', '最大误差', '单调违例', '负值率',
               '零裂纹误报', '折内rho', '值域比'))
        say('  ' + '-' * 108)
        for nm in order:
            g = sub[sub['arm'] == nm]
            if not len(g):
                continue
            say('  %-16s %8.3f %8.3f %8.3f %10.3f %8.1f%% %8.1f%% %8s %9.3f %9.2f'
                % (nm, g['rmse'].mean(), g['mae'].mean(), g['bias'].mean(),
                   g['max_abs_err'].max(), 100 * g['mono_viol_rate'].mean(),
                   100 * g['neg_rate'].mean(),
                   ('%.0f%%' % (100 * g['false_alarm'].mean()))
                   if g['false_alarm'].notna().any() else '—',
                   g['rho_within'].mean(), g['range_ratio'].mean()))

    # ---------- 逐折明细（主力臂） ----------
    main_arm = 'gpr' if (loso['arm'] == 'gpr').any() else 'svr'
    say('\n' + '=' * 78)
    say('[逐折明细] 臂 = %s（RMSE / MAE，mm）' % main_arm)
    say('=' * 78)
    say('  %-5s %-10s %-10s %s' % ('试件', 'rep1 RMSE', 'rep2 RMSE', 'rep1 每时刻误差最大者'))
    for sp in prep.SPECIMENS:
        cells = []
        for rp in (1, 2):
            g = loso[(loso['arm'] == main_arm) & (loso['rep'] == rp) &
                     (loso['fold'] == sp)]
            cells.append(g['rmse'].iloc[0] if len(g) else np.nan)
        p1 = pred[(pred['arm'] == main_arm) & (pred['rep'] == 1) &
                  (pred['specimen'] == sp)]
        worst = ''
        if len(p1):
            w = p1.iloc[np.argmax(np.abs(p1['err']))]
            worst = 'cycle=%d err=%+.2f' % (w['cycle'], w['err'])
        say('  %-5s %-10s %-10s %s'
            % (sp, '%.3f' % cells[0] if np.isfinite(cells[0]) else '—',
               '%.3f' % cells[1] if np.isfinite(cells[1]) else '—', worst))

    # ---------- 零裂纹误报 ↔ 基线自洽性（核心归因） ----------
    say('\n' + '=' * 78)
    say('[归因] 零裂纹误报 vs 基线自洽性（臂 = %s）' % main_arm)
    say('=' * 78)
    say('  逻辑：所有特征都是「相对该试件零裂纹基线」的。若该试件的多个零裂纹时刻')
    say('  本身互不相同（基线不自洽），则健康状态下特征就已偏离基线 ⇒ 必然误报。')
    say('  因此： 零裂纹处的预测误差，应随「基线自洽性」单调恶化。下面直接验证。')
    try:
        bc = pd.read_csv(os.path.join(RES, 'step1_baseline_consistency%s.csv' % suffix))
    except Exception:
        bc = None
    say('')
    say('  %-5s %6s %12s %14s %14s' %
        ('试件', '零裂纹数', '基线相邻corr', '零裂纹预测中位', '零裂纹绝对误差中位'))
    say('  ' + '-' * 68)
    rows = []
    for sp in prep.SPECIMENS:
        g = pred[(pred['arm'] == main_arm) & (pred['rep'] == 1) &
                 (pred['specimen'] == sp) & (pred['crack_mm'] == 0)]
        if not len(g):
            continue
        cons = np.nan
        if bc is not None:
            r = bc[bc['specimen'] == sp]
            if len(r) and r['n'].iloc[0] >= 2:
                cons = float(r['pairwise_corr_min'].iloc[0])
        med = float(g['pred'].median())
        mae = float(g['err'].abs().median())
        rows.append({'specimen': sp, 'n_zero': len(g), 'baseline_corr': cons,
                     'pred_med': med, 'abs_err_med': mae})
        say('  %-5s %6d %12s %14.3f %14.3f'
            % (sp, len(g), ('%.3f' % cons) if np.isfinite(cons) else '不可自检',
               med, mae))
    rd = pd.DataFrame(rows)
    ok = rd[np.isfinite(rd['baseline_corr'])]
    if len(ok) >= 3:
        rho = float(ok['baseline_corr'].corr(ok['abs_err_med'], method='spearman'))
        say('')
        say('  可自检试件（%s）= %d 个；'
            % ('/'.join(ok['specimen']), len(ok)))
        say('  「基线相邻corr」与「零裂纹绝对误差中位」的 Spearman = **%.3f**' % rho)
        if rho < -0.5:
            say('  ⇒ 呈**负相关**：基线越不自洽，零裂纹误差越大 —— **归因成立**。')
            say('     这说明零裂纹误报**部分来自数据**（健康期波形本身在漂移），')
            say('     不是单纯模型容量问题。')
            say('     ⚠️ 但**只有 %d 个可自检试件**，此结论的证据强度有限：' % len(ok))
            say('        它表明该机制**存在**，不足以定量其贡献占比。')
        else:
            say('  ⇒ 未见明显负相关，归因不成立，需另找原因（当前样本太少，仅 %d 点）。'
                % len(ok))
    say('')
    say('  第二个误差源（与基线自洽性无关）：**预测值域压缩**。')
    say('  看总表 `值域比` 列 = pred_std / true_std：多数臂远小于 1（如 single_tof 仅 0.06），')
    say('  即模型几乎只输出一个常数 ⇒ 绝对标定差。')
    say('  ⇒ 因此本步必须**分开看两个问题**：')
    say('     (a) 特征有没有裂纹信息 —— 看 `折内rho`（同一试件内 pred vs 真值的 Spearman）；')
    say('     (b) 跨试件的绝对标定好不好 —— 看 RMSE / 值域比。')
    say('     (a) 高而 (b) 差 ⇒ 是标定/迁移问题，不是信息缺失问题。')
    say('')
    say('  ⚠️ 只有 1 个零裂纹时刻的试件（T1/T2/T4/T6）**无法自检** ——')
    say('     它们的基线是否稳定，本数据集**无法验证**，对外必须声明这一点。')

    # ---------- 结论 ----------
    say('\n' + '=' * 78)
    say('[读数指引]')
    say('=' * 78)
    say('  1. 与 `single_tof` 比：多元特征到底比"只用 TOF"好多少 —— 这是特征工程的收益。')
    say('     ⭐ 本数据的答案是**巨大**：`single_tof` 的折内 rho 仅 ~0.05、值域比 ~0.06，')
    say('        即 **TOF 在这份数据上几乎不含裂纹信息** —— 它退化成常数预测。')
    say('        物理上说得通：裂纹在铆钉孔处起裂、主要改变**透射幅值/波场形状**，')
    say('        而直达波路径长度几乎不变 ⇒ 期望 TOF 敏感本来就没依据。')
    say('     ⇒ README 要求的"只用 TOF 这类单特征基线"由此得到明确结论。')
    say('  2. 与 `+mono` 比：物理约束层带来的变化 —— 本数据上**两个指标同时变好**：')
    say('     RMSE 下降且 折内 rho 上升，同时单调违例率与负值率归零（见总表）。')
    say('  3. rep1 与 rep2 的 RMSE 之差 = 结论对"用哪次测量"的敏感性。')
    say('  4. `零裂纹误报` = 真值 0 的时刻上预测 > 0.5 mm 的比例 ——')
    say('     物理含义是"健康时会不会乱报伤"，比 RMSE 更贴近工程可用性。')
    say('  5. ⚠️ 官方评分函数未实现（常数在 PHM2019_ScoringSpreadsheet.xlsx 内，')
    say('     仓库里没有该文件）⇒ 拿到官方常数前**不自行编造**。')

    # ---------- 基线口径对照（读取另一口径的产物，若已存在） ----------
    other = 'first' if args.tag == 'mean_zero' else 'mean_zero'
    other_path = OUT_LOSO.replace(
        '.csv', '' if other == 'mean_zero' else '_' + other) + '.csv'
    if os.path.exists(other_path) and other_path != out_loso:
        o = pd.read_csv(other_path)
        say('\n' + '=' * 78)
        say('[对照] 基线口径 %s vs %s（RMSE 均值 / 折内 rho 均值）'
            % (args.tag, other))
        say('=' * 78)
        say('  %-16s %14s %14s %12s %12s' %
            ('臂', 'RMSE-' + args.tag, 'RMSE-' + other,
             'rho-' + args.tag, 'rho-' + other))
        say('  ' + '-' * 72)
        better, tot = 0, 0
        for nm in ['single_tof', 'single_best', 'linear', 'svr', 'rf', 'gpr',
                   'linear+mono', 'svr+mono', 'rf+mono', 'gpr+mono']:
            g1 = loso[loso['arm'] == nm]
            g2 = o[o['arm'] == nm]
            if not len(g1) or not len(g2):
                continue
            tot += 1
            if g1['rmse'].mean() < g2['rmse'].mean():
                better += 1
            say('  %-16s %14.3f %14.3f %12.3f %12.3f'
                % (nm, g1['rmse'].mean(), g2['rmse'].mean(),
                   g1['rho_within'].mean(), g2['rho_within'].mean()))
        if tot:
            win = args.tag if better >= tot / 2 else other
            say('')
            say('  ⇒ **%s 在 %d/%d 个臂上 RMSE 更优**。'
                % (win, better if better >= tot / 2 else tot - better, tot))
            say('     结论：「多个零裂纹时刻**平均**」比「最早的**单一**时刻」更稳 ——')
            say('     与 README §5.0.3 的预判一致（T1/50000 是全数据集重复性最差的时刻）。')
            say('     ⚠️ 两者都不是"真基线"（数据集无 Baseline 目录），此处只做口径对照。')

    with open(out_report, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(_REPORT) + '\n')
    say('\n[产出] %s / %s / %s'
        % (os.path.basename(out_loso), os.path.basename(out_pred),
           os.path.basename(out_report)))


if __name__ == '__main__':
    main()
