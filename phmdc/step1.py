"""
Step 1: 特征提取 —— PHM2019 铝搭接件（数据集 D）
==================================================

基于 `prep.py` 的预处理层（重建时间轴 + 带通 100~300 kHz + 包络）提取特征。

两条设计原则（均由 Step 0 的实测逼出来）
----
1. **一律用相对量**，不用绝对幅值。
   理由：ch1 激励幅值逐试件不同（峰峰 1.88~4.28 V），ch2 被量化
   （步长 8e-4~2e-3 V，每试件仅 108~244 个取值）。
   ⇒ 所有幅值类特征都写成「相对该试件健康基线」的比值或差值。

2. **健康基线用「零裂纹时刻的平均」**，不用「最早的单一时刻」。
   理由：Step 0 §5.0.3 —— T1/50000（最早、未起裂）是全数据集重复性最差的时刻
   （带通后 corr 仍只有 0.9046），拿它单独当基线最不稳。
   同时保留 `--baseline first` 开关以便对照。

输出
----
  phmdc/results/step1_features.csv    逐 (specimen, cycle, rep) 的特征表
  phmdc/results/step1_feasibility.csv 每特征 vs 裂纹长度的相关性与单调性
  phmdc/results/step1_report.txt      可读报告

依赖：numpy / pandas / scipy / pywt(可选，缺则跳过小波包族)
"""

import os
import sys
import argparse
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd

import prep

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = prep.ROOT
OUT_FEAT = os.path.join(ROOT, 'results', 'step1_features.csv')
OUT_FEAS = os.path.join(ROOT, 'results', 'step1_feasibility.csv')
OUT_BC = os.path.join(ROOT, 'results', 'step1_baseline_consistency.csv')
OUT_REPORT = os.path.join(ROOT, 'results', 'step1_report.txt')

_REPORT = []


def say(msg=''):
    print(msg)
    _REPORT.append(msg)


# ============================================================
# 特征定义（全部为「相对健康基线」的量）
# ============================================================
# 时间窗（µs）——切在带通后的 ch2 上。激励 burst 长约 5 周期 / 200 kHz ≈ 25 µs，
# 因此 [0,40] 主要含直达波，[40,100] 含板端/铆钉列的多次反射。
WIN_DIRECT = (0.0, 40e-6)
WIN_LATE = (40e-6, 100e-6)
WIN_TAIL = (100e-6, 200e-6)


def _win_energy(x, t, lo, hi):
    m = (t >= lo) & (t < hi)
    return float(np.sum(x[m] ** 2))


def _first_arrival(t, env, frac=0.15):
    """包络首达时刻：包络首次升到 `frac × 全局峰值` 的时刻 [s]。"""
    peak = float(env.max())
    if peak <= 0:
        return np.nan
    idx = np.argmax(env >= frac * peak)
    return float(t[idx]) if idx > 0 or env[0] >= frac * peak else np.nan


def extract_one(specimen, cycle, rep, sp_baseline=None):
    """提取一个 (试件, 时刻, 重复) 的特征。

    返回 (features_dict, state_dict)；`state_dict` 供调用方拼装基线。
    """
    d = prep.prep(specimen, cycle, rep)
    t, ch2, env = d['t'], d['ch2'], d['env2']
    ch1 = d['ch1']
    dt = 1.0 / d['fs']          # 降采样后的**实际**采样间隔（勿用全局 DT）

    f = OrderedDict()
    f['specimen'] = specimen
    f['cycle'] = cycle
    f['rep'] = rep

    # ---------- 绝对量（仅作诊断，不进模型） ----------
    f['pp_ch1_raw'] = d['pp_ch1_raw']

    # ---------- 时域：绝对量（用于拼基线） ----------
    env_peak = float(env.max())
    env_peak_t = float(t[int(np.argmax(env))])
    tof_first = _first_arrival(t, env)
    e_dir = _win_energy(env, t, *WIN_DIRECT)
    e_late = _win_energy(env, t, *WIN_LATE)
    e_tail = _win_energy(env, t, *WIN_TAIL)
    e_tot = e_dir + e_late + e_tail
    rms_ch2 = float(np.sqrt(np.mean(ch2 ** 2)))

    f['env_peak_abs'] = env_peak
    f['env_peak_t'] = env_peak_t
    f['tof_first'] = tof_first
    f['rms_ch2_abs'] = rms_ch2

    # ---------- 时域：相对量（相对健康基线；由调用方回填） ----------
    f['_win'] = (e_dir, e_late, e_tail, e_tot)

    # ---------- 频域：传递函数 H(f) ----------
    # H(f)=FFT(ch2)/FFT(ch1) 消除激励差异（README §5 Step 1 的核心做法）
    fh, H = prep.transfer_function(ch1, ch2, dt=dt)
    band = (fh >= prep.BAND_HZ[0]) & (fh <= prep.BAND_HZ[1])
    Hb = np.abs(H[band])
    fb = fh[band]
    Hn = Hb / (Hb.sum() + 1e-30)           # 归一谱（形状量，与幅值无关）
    f['h_centroid'] = float(np.sum(fb * Hn))
    f['h_peak_f'] = float(fb[int(np.argmax(Hb))])
    f['h_peak_mag_rel'] = float(Hb.max() / (Hb.mean() + 1e-30))
    f['h_bw90'] = float(np.sum(Hn[fb <= f['h_centroid']]))   # 谱在下侧累积比例
    f['h_flatness'] = float(np.exp(np.mean(np.log(Hb + 1e-30))) / (Hb.mean() + 1e-30))
    f['_H_mag_band'] = Hb

    # ---------- 频域：ch2 自身谱（形状量） ----------
    C = np.abs(np.fft.rfft(ch2 - ch2.mean()))
    fc = np.fft.rfftfreq(len(ch2), 1.0 / d['fs'])
    mb = (fc >= prep.BAND_HZ[0]) & (fc <= prep.BAND_HZ[1])
    Cn = C[mb] / (C[mb].sum() + 1e-30)
    f['c_centroid'] = float(np.sum(fc[mb] * Cn))
    f['c_entropy'] = float(-np.sum(Cn * np.log(Cn + 1e-30)))

    # ---------- 时频：小波包（3 层 db4，8 节点能量比） ----------
    try:
        import pywt
        wp = pywt.WaveletPacket(data=ch2, wavelet='db4', maxlevel=3)
        e = np.array([np.sum(np.asarray(n.data) ** 2) for n in wp.get_level(3)],
                     dtype=float)
        e = e / (e.sum() + 1e-30)
        for i, v in enumerate(e):
            f['wpt_e%d' % i] = float(v)
    except Exception:
        pass

    # 保存两段带通波形（供基线相似度用；不进 CSV）
    state = {
        'ch2': ch2, 'env': env, 't': t,
        'env_peak': env_peak, 'tof_first': tof_first,
        'e_dir': e_dir, 'e_late': e_late, 'e_tail': e_tail, 'e_tot': e_tot,
        'H_mag_band': Hb,
    }
    return f, state


def build_baselines(baseline_mode='mean_zero'):
    """为每个试件构造「健康基线」（零裂纹时刻）。

    baseline_mode:
      'mean_zero' —— 该试件**所有裂纹 = 0 的时刻**的平均（默认，更稳）
      'first'     —— 该试件**最早的单一时刻**（README §5.0.3 要求对照的口径）
    返回 {specimen: {'state': 平均后的 state, 'cycles': [用到的时刻]}
    """
    lab = prep.labels()
    out = {}
    for sp in prep.SPECIMENS:
        zero = sorted(int(r.cycle) for r in lab[(lab['specimen'] == sp) &
                                                (lab['crack_mm'] == 0)].itertuples())
        if not zero:
            out[sp] = None
            continue
        reps = [1, 2]
        states, used_cycles = [], []
        for cyc in zero:
            got = 0
            for rp in reps:
                try:
                    _, st = extract_one(sp, cyc, rp)
                    states.append(st)
                    got += 1
                except Exception:
                    pass
            if got:
                used_cycles.append(cyc)
        if not states:
            out[sp] = None
            continue
        if baseline_mode == 'first':
            first = used_cycles[0]
            keep = [st for c, st in zip(sum([[c] * 2 for c in used_cycles], []),
                                        states) if c == first]
            states = keep if keep else states[:2]
            used_cycles = [first]
        agg = {}
        # 标量取平均
        for k in ('env_peak', 'e_dir', 'e_late', 'e_tail', 'e_tot'):
            agg[k] = float(np.mean([s[k] for s in states]))
        # 向量取平均（长度一致才行）
        for k in ('ch2', 'env', 'H_mag_band'):
            L = min(len(s[k]) for s in states)
            agg[k] = np.mean([np.asarray(s[k])[:L] for s in states], axis=0)
        # tof 用中位（受首达阈值抖动影响大）
        agg['tof_first'] = float(np.median([s['tof_first'] for s in states
                                            if np.isfinite(s['tof_first'])]))
        out[sp] = {'state': agg, 'cycles': used_cycles,
                   'n_used': len(states)}
    return out


def add_relative(f, state, base):
    """把绝对量转成相对健康基线的特征（核心：跨试件可比）。"""
    b = base['state']
    # --- 幅值比 ---
    f['env_peak_ratio'] = state['env_peak'] / (b['env_peak'] + 1e-30)
    f['env_peak_drop'] = 1.0 - f['env_peak_ratio']
    f['rms_ratio'] = f['rms_ch2_abs'] / (
        np.sqrt(np.mean(b['ch2'] ** 2)) + 1e-30)
    # --- TOF 偏移 ---
    f['tof_shift'] = state['tof_first'] - b['tof_first']
    f['env_peak_t_shift'] = f['env_peak_t'] - float(
        np.argmax(b['env'])) * (state['t'][1] - state['t'][0])
    # --- 窗能量比（相对总量，再看与基线的比） ---
    for name, key in (('direct', 'e_dir'), ('late', 'e_late'), ('tail', 'e_tail')):
        f['e_%s_frac' % name] = state[key] / (state['e_tot'] + 1e-30)
        f['e_%s_ratio' % name] = state[key] / (b[key] + 1e-30)
    # --- 与基线的相似度（损伤指数类） ---
    L = min(len(state['ch2']), len(b['ch2']))
    x, y = state['ch2'][:L], b['ch2'][:L]
    f['corr_baseline'] = prep.corr(x, y)
    f['di_relres'] = (float(np.sqrt(np.mean((x - y) ** 2))) /
                      (np.sqrt(np.mean(y ** 2)) + 1e-30))
    Le = min(len(state['env']), len(b['env']))
    xe, ye = state['env'][:Le], b['env'][:Le]
    xd, yd = xe - xe.mean(), ye - ye.mean()
    den = np.sqrt(np.dot(xd, xd) * np.dot(yd, yd))
    f['corr_env'] = float(np.dot(xd, yd) / den) if den > 0 else np.nan
    f['env_relres'] = (float(np.sqrt(np.mean((xe - ye) ** 2))) /
                       (np.sqrt(np.mean(ye ** 2)) + 1e-30))
    # --- 传递函数幅值差的相对量 ---
    Lh = min(len(state['H_mag_band']), len(b['H_mag_band']))
    a, c = state['H_mag_band'][:Lh], b['H_mag_band'][:Lh]
    f['H_l1_rel'] = float(np.mean(np.abs(a - c)) / (np.mean(np.abs(c)) + 1e-30))
    return f


# ============================================================
# 主流程
# ============================================================
def baseline_consistency():
    """量化「零裂纹时刻之间是否彼此一致」——基线的自洽性。

    为什么必须查：所有特征都是**相对该试件零裂纹基线**的。
    若某试件的多个零裂纹时刻本身互不相同，则基线对谁都不合适，
    特征会出现**与裂纹无关的漂移** ⇒ 直接表现为"健康时误报损伤"。
    （Step 1 的第一版正是踩到这里：T5/T8 的零裂纹时刻互相不一致。）
    """
    lab, idx = prep.labels(), prep.index()
    have = {(r.specimen, r.cycle) for r in idx.itertuples()}
    rows = []
    for sp in prep.SPECIMENS:
        z = sorted(int(c) for c in
                   lab[(lab['specimen'] == sp) & (lab['crack_mm'] == 0)]['cycle']
                   if (sp, int(c)) in have)
        if not z:
            rows.append({'specimen': sp, 'zero_cycles': [], 'n': 0,
                         'pairwise_corr_min': np.nan, 'pairwise_corr_med': np.nan,
                         'pairwise_relrms_max': np.nan})
            continue
        ws = {c: prep.prep(sp, c, 1)['ch2'] for c in z}
        cs, rs, cs_al, lags = [], [], [], []
        for a, b in zip(z, z[1:]):
            n = min(len(ws[a]), len(ws[b]))
            x, y = ws[a][:n], ws[b][:n]
            cs.append(prep.corr(x, y))
            rs.append(prep.rel_rms_diff(x, y))
            # 整体时移是否解释得了差异？（±50 点 = ±0.5 ms，远超 TOF 量级）
            best_r, best_k = -2.0, 0
            for k in range(-50, 51):
                yk = np.roll(y, -k)
                m = slice(0, n - k) if k >= 0 else slice(-k, n)
                r = prep.corr(x[m], yk[m])
                if np.isfinite(r) and r > best_r:
                    best_r, best_k = r, k
            cs_al.append(best_r)
            lags.append(best_k)
        rows.append({'specimen': sp, 'zero_cycles': z, 'n': len(z),
                     'pairwise_corr_min': min(cs) if cs else np.nan,
                     'pairwise_corr_med': float(np.median(cs)) if cs else np.nan,
                     'pairwise_relrms_max': max(rs) if rs else np.nan,
                     'pairwise_corr_aligned_min': min(cs_al) if cs_al else np.nan,
                     'lag_max_pts': max((abs(v) for v in lags), default=np.nan)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description='phmdc Step 1 特征提取')
    ap.add_argument('--baseline', choices=['mean_zero', 'first'],
                    default='mean_zero', help='健康基线口径（默认 mean_zero）')
    args = ap.parse_args()

    # 基线口径不同 -> 产物分开命名，便于 2×2 对照（README §5.0.3 要求）
    suffix = '' if args.baseline == 'mean_zero' else '_' + args.baseline
    out_feat = OUT_FEAT.replace('.csv', suffix + '.csv')
    out_feas = OUT_FEAS.replace('.csv', suffix + '.csv')
    out_bc = OUT_BC.replace('.csv', suffix + '.csv')
    out_report = OUT_REPORT.replace('.txt', suffix + '.txt')

    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)

    say('=' * 78)
    say('PHM2019 铝搭接件（数据集 D） Step 1 特征提取')
    say('生成时间: %s' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    say('基线口径: %s' % args.baseline)
    say('=' * 78)

    # ---------- 基线 ----------
    say('\n[1] 健康基线（该试件的零裂纹时刻）')
    bases = build_baselines(args.baseline)
    for sp in prep.SPECIMENS:
        b = bases.get(sp)
        if b is None:
            say('    %-3s 无零裂纹时刻 ⇒ 无法构造基线' % sp)
        else:
            say('    %-3s 零裂纹时刻 %-28s 用 %d 次测量平均'
                % (sp, str(b['cycles']), b['n_used']))

    # ---------- 逐时刻提特征 ----------
    say('\n[2] 逐 (试件, 时刻, 重复) 提特征')
    rows, skipped = [], []
    for sp in prep.SPECIMENS:
        if bases.get(sp) is None:
            continue
        lab = prep.labels()
        cycs = sorted(int(r.cycle) for r in lab[lab['specimen'] == sp].itertuples())
        have = {(r.specimen, r.cycle) for r in prep.index().itertuples()}
        for cyc in cycs:
            if (sp, cyc) not in have:
                skipped.append((sp, cyc, 'no_waveform'))
                continue
            for rp in (1, 2):
                try:
                    f, st = extract_one(sp, cyc, rp)
                except Exception as e:
                    skipped.append((sp, cyc, 'rep%d: %s' % (rp, e)))
                    continue
                f = add_relative(f, st, bases[sp])
                rows.append(f)
    feat = pd.DataFrame(rows)
    say('    提取 %d 行（%d 个时刻 × 2 次重复）' % (len(feat), len(feat) // 2))
    if skipped:
        say('    跳过 %d 条: %s' % (len(skipped), skipped[:5]))

    meta_cols = ['specimen', 'cycle', 'rep', 'pp_ch1_raw']
    drop = [c for c in feat.columns if c.startswith('_')]
    feat_clean = feat.drop(columns=drop)
    feat_clean = feat_clean.merge(
        prep.labels()[['specimen', 'cycle', 'crack_mm']],
        on=['specimen', 'cycle'], how='left')
    feat_clean.to_csv(out_feat, index=False, encoding='utf-8-sig')

    # ---------- 可行性：每个特征 vs 裂纹长度 ----------
    say('\n[3] 特征可行性（相对量 vs 裂纹长度）')
    fcols = [c for c in feat_clean.columns
             if c not in meta_cols + ['crack_mm']
             and pd.api.types.is_numeric_dtype(feat_clean[c])]
    feas = []
    for c in fcols:
        sub = feat_clean[['specimen', 'cycle', 'crack_mm', c]].dropna()
        if len(sub) < 10 or sub[c].nunique() < 2:
            continue
        # (a) 全体 Spearman（跨试件，含试件间偏置）
        rho_all = float(sub[c].corr(sub['crack_mm'], method='spearman'))
        # (b) **试件内** Spearman 的中位（去掉试件间偏置 —— 更接近真实可用性）
        per = []
        for sp, g in sub.groupby('specimen'):
            if len(g) >= 3 and g[c].nunique() >= 2:
                per.append(float(g[c].corr(g['crack_mm'], method='spearman')))
        rho_within = float(np.median(per)) if per else np.nan
        # (c) 试件内单调走势一致率：取「不减」与「不增」中较大者，
        #     并记录方向（损伤指标升或降都可能单调，关键是**方向一致**）
        up_list, down_list = [], []
        for sp, g in sub.groupby('specimen'):
            g = g.sort_values('cycle')
            if len(g) < 3:
                continue
            v = g.groupby('cycle')[c].median().to_numpy()
            d = np.diff(v)
            up_list.append(float(np.mean(d >= 0)))
            down_list.append(float(np.mean(d <= 0)))
        up = float(np.mean(up_list)) if up_list else np.nan
        down = float(np.mean(down_list)) if down_list else np.nan
        direction = ('↑随裂纹增' if np.nanmean([up, down]) >= 0 and up >= down
                     else '↓随裂纹减')
        feas.append({'feature': c, 'n': len(sub),
                     'rho_all': rho_all, 'rho_within_med': rho_within,
                     'rho_within_abs': abs(rho_within) if np.isfinite(rho_within) else np.nan,
                     'mono_up_rate': up, 'mono_down_rate': down,
                     'mono_consistency': max(up, down) if up_list else np.nan,
                     'direction': direction,
                     'n_specimen': len(per)})
    feas = pd.DataFrame(feas).sort_values('rho_within_abs', ascending=False)
    feas.to_csv(out_feas, index=False, encoding='utf-8-sig')

    say('    %-22s %6s %10s %12s %10s %10s  %s' %
        ('特征', 'n', 'rho(全体)', 'rho(试件内)', '单调一致率', '走势', '方向'))
    say('    ' + '-' * 84)
    for r in feas.head(14).itertuples():
        arrow = '↑' if r.mono_up_rate >= r.mono_down_rate else '↓'
        say('    %-22s %6d %10.3f %12.3f %9.1f%% %10s  %s'
            % (r.feature, r.n, r.rho_all, r.rho_within_med,
               100 * r.mono_consistency, arrow, r.direction))
    say('')
    say('    读法：`rho(试件内)` 是**留一试件之外真正可用的相关性** ——')
    say('          它把「试件间偏置」去掉了，因此比 `rho(全体)` 可信。')
    say('          `单调一致率` = 各试件内特征走势方向一致的比例（100% = 8 组同向）。')
    say('          `方向` 只是观测到的走势，**不是**要求的物理方向；')
    say('          物理上「随裂纹增」还是「减」取决于该特征的定义。')

    strong = feas[feas['rho_within_abs'] >= 0.6]
    say('\n    |rho(试件内)| ≥ 0.6 的特征：%d 个'
        % len(strong))
    if len(strong):
        say('      ' + ', '.join(strong['feature'].head(12)))
    else:
        say('      ⚠️ 没有 —— 说明当前特征族不足以支撑跨试件回归，需换特征再试。')

    # ---------- 基线自洽性（决定"零裂纹期会不会误报"） ----------
    say('\n' + '=' * 78)
    say('[4] 基线自洽性 —— 零裂纹时刻之间是否彼此一致')
    say('=' * 78)
    say('    必要性：所有特征都是「相对该试件零裂纹基线」的。')
    say('    若同一试件的多个零裂纹时刻本身互不相同，基线对谁都不合适，')
    say('    特征就会出现**与裂纹无关的漂移** ⇒ 直接表现成"健康时误报损伤"。')
    say('')
    bc = baseline_consistency()
    say('    %-5s %5s %-22s %10s %10s %12s %8s' %
        ('试件', '零裂纹', '零裂纹时刻', '相邻corr', 'relRMS最大',
         '对齐后corr', 'lag(点)'))
    say('    ' + '-' * 80)
    for r in bc.itertuples():
        if r.n == 0:
            say('    %-5s %5d %-22s %10s %10s %12s %8s'
                % (r.specimen, 0, '(无)', '—', '—', '—', '—'))
            continue
        if r.n < 2:
            say('    %-5s %5d %-22s %10s %10s %12s %8s'
                % (r.specimen, r.n, str(r.zero_cycles)[:22], '—', '—', '—', '—'))
            continue
        flag = ''
        if r.pairwise_corr_aligned_min >= 0.9 > r.pairwise_corr_min:
            flag = '  <== 差异可由时移解释'
        elif r.pairwise_corr_min < 0.9:
            flag = '  <== 基线不自洽（时移也解释不了）'
        say('    %-5s %5d %-22s %10.3f %9.0f%% %12.3f %8d%s'
            % (r.specimen, r.n, str(r.zero_cycles)[:22],
               r.pairwise_corr_min, 100 * r.pairwise_relrms_max,
               r.pairwise_corr_aligned_min, r.lag_max_pts, flag))
    bad_bc = bc[(bc['n'] >= 2) & (bc['pairwise_corr_min'] < 0.9)]
    bc.to_csv(out_bc, index=False, encoding='utf-8-sig')
    if len(bad_bc):
        say('')
        say('    ⚠️ **%s 的零裂纹时刻彼此不一致** ⇒ 这些试件的基线特征'
            % '/'.join(bad_bc['specimen']))
        say('       天然带一个"与裂纹无关"的偏移，模型在它们身上必然出现零裂纹误报。')
        say('       ⇒ 这不是模型缺陷，是**数据里"波形先变、裂纹后测到"**的证据，')
        say('         属 Step 1 的核心发现（见 step1_model_report.txt 的"零裂纹误报"一列）。')
        say('       ⇒ 诊断价值：把"时移可解释"与"时移解释不了"分开 ——')
        say('         前者可用对齐修掉，后者只能靠"同一次装夹内取基线"或标注不可用。')
    say('')
    say('    只能给 1 个零裂纹时刻的试件（%s）无法自检 ——'
        % '/'.join(bc[bc['n'] == 1]['specimen']))
    say('    必须记住它们的基线**未经验证**。')

    say('\n' + '=' * 78)
    say('[产出]')
    say('  %s' % os.path.relpath(out_feat, ROOT))
    say('  %s' % os.path.relpath(out_feas, ROOT))
    say('  %s' % os.path.relpath(out_bc, ROOT))
    say('=' * 78)

    with open(out_report, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(_REPORT) + '\n')
    return feat_clean, feas


if __name__ == '__main__':
    main()
