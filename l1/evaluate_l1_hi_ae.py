# -*- coding: utf-8 -*-
"""L1 的 AE 健康指标（HI_AE）—— 按文献口径重建，用于替代／对照自研 D(t)

【为什么换指标（2026-09-16 结论）】
自研 `D(t)` 在 **主样本 016-020** 上可行，因为那里 D 的触发（risk 抬头）恰好
与真损伤起始同步；但在 **L1（冲击后疲劳）** 上不同步：0 cycle 即带 BVID，
`e_ae` 自冲击后即活跃 → `risk = max(e_ae, e_strain)` 一抬头 D 就在 ≈8.3 块
（rise=0.12 的追赶时间常数）后冲上 0.85，与真实寿命无关。
实测证据：块长按寿命比例统一后 t85 反而更早（L1-49 68.6%→25.1%）——
因为「块内越界」是 max 统计量，块越细抽样越密、越早触发。
⇒ 问题不在时间尺度，而在**指标语义**：D 是「损伤活动度的即时探测器」，
而三级预警语义是「剩余寿命裕度」。

【文献口径（本数据集原产地论文）】
A. Broer et al. 2021, Struct Health Monit 20(2) —— Level 4 严重度
   `HI_F = 0.5·HI_AE + 0.5·HI_OF`，其中 **HI_AE = 每 500 cycles 累积 AE 能量**
   （全局累积 → 天然单调）；论文自述其优点是 "inherent monotonic behavior
   with the number of fatigue cycles"。
   已在 `reproduce_broer_l1.py` 复现：第一批 4 组 t85 = **79~95%** 寿命（不早报）。
B. Galanopoulos et al. 2021, Sensors 21, 5701（同一批试件的 HI 开发论文）
   - 选 500 cycles 作窗口的理由：趋势平滑，且能与 FBG 共享同一测量区间以便融合；
   - Eq.10 `HI_AE(t) = Σ_{i=t-T}^{t} F(i)`，T = 500 cycles，F = hits 或 RA
     —— **窗口累积**（本文按同一步长分箱，即"每 500 cycle 的事件量"）；
   - 结论：AE 类 HI 的**单调性低于应变类，但 prognosability 更高**
     （"failure values have less scatter, making it easier to set a failure
     threshold"），且与加载工况无关；
   - 论文明确的下一步：**AE 与应变 HI 的融合**（= Broer 的 HI_F 已做的事）。

【与自研 D 的关键差别】
本脚本全部指标都是**积分/窗口累积量**（对"早期活跃"不敏感 → 不早报，
且天然平滑无阶梯），而 D 用的是**块内是否越界的即时判据**。

【验收口径 —— 用论文的两把尺子，而不是"曲线好不好看"】
  - 单调性 Mon（Coble & Hines，即论文引用 [37]）：
        Mon = |(正增量个数 − 负增量个数)| / (n−1)      ∈ [0,1]
  - prognosability Prog（跨试件）：
        Prog = exp( − std(HI_fail) / mean(|HI_fail − HI_init|) )   ∈ (0,1]
        1 = 各试件失效时的 HI 值完全一致（可直接设统一阈值）。
  - 以及最实用的一条：**先用各试件失效值的中位作阈值，再反算各组的达阈寿命百分比** ——
    这直接回答"会不会过早临危"。
  - 另对照 Broer Level-4 的**自归一化 + 固定分数**（各组自身归一化后取 0.50/0.85）→
    其 t85 可与第一批已复现的 79~95% 直接比较（见表 t85_*）。

用法:
  python evaluate_l1_hi_ae.py                      # 第二批 9 组（默认）
  python evaluate_l1_hi_ae.py --batch both         # 并入第一批 4 组对照
  python evaluate_l1_hi_ae.py --groups L1-49,L1-60
输出:
  results/l1_hi_ae.csv           逐组 × 逐指标
  results/l1_hi_ae_summary.csv   逐指标 × 跨组汇总（Mon / Prog / 达阈寿命%）
  figures/l1_hi_ae_<指标>.png
"""
import os
import sys
import argparse
import importlib.util

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, os.path.dirname(ROOT))
sys.path.insert(0, ROOT)
from l1_meta import load_meta                       # noqa: E402
import l1_time_align as ta                         # noqa: E402

GROUPS_V2 = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
             'L1-59', 'L1-60']
GROUPS_V1 = ['L1-03', 'L1-04', 'L1-05', 'L1-09']

CYCS_BIN = 500          # 论文的窗口/分箱步长（且与 FBG/DFOS 测量区间一致）
ROLL_WINS = (10,)       # 额外窗口（10 箱 = 5000 cycle），平滑版速率
# 参与"共同阈值"标定的最低 AE 覆盖度（AE 在该组寿命内的覆盖比例）
COV_MIN = 0.90
# 方案B —— 因果归一锚点（寿命比例）。u(t) = HI(t) / HI(锚点)：
#   锚点在 t_anchor ≤ t 处已知 → **纯因果**，不需要全寿命最大值（对比 unity01 的离线归一）。
#   若各组"失效时的比值 u_fail"离散足够小，则可定统一的比值阈值。
ANCHOR_PCTS = (0.05, 0.10, 0.20, 0.30, 0.50)

VARIANTS = ['cum_hits', 'cum_energy', 'win1_hits', 'win1_energy',
            'win10_hits', 'win10_energy']
PLOT_VARIANTS = ['cum_hits', 'cum_energy', 'win1_hits', 'win1_energy']
VLABEL = {
    'cum_hits': '$HI_{AE}$ 累积 hits（Broer 口径）',
    'cum_energy': '$HI_{AE}$ 累积能量（Broer 口径）',
    'win1_hits': '$HI_{AE}$ 窗口 hits（论文式10, T=500）',
    'win1_energy': '$HI_{AE}$ 窗口能量（论文式10, T=500）',
    'win10_hits': '$HI_{AE}$ 窗口 hits（T=5000）',
    'win10_energy': '$HI_{AE}$ 窗口能量（T=5000）',
}

_RB = None


# ------------------------------------------------------------------ cycle 锚
def _cycle_of(gid, t, nf):
    """时间 → cycle。

    有 `{gid}dfos_anchor.csv` 的组（第二批，无 FBG）用 DFOS 段锚
    （`clip_out_of_life=True`：剔除冲击前 BI 事件，它们不属于疲劳寿命）；
    否则（第一批）回退 FBG 块锚 —— 按文件路径加载，避免与本模块同目录互相遮蔽。
    """
    if os.path.exists(os.path.join(ROOT, gid, f'{gid}dfos_anchor.csv')):
        return ta.time_to_cycle(gid, t, nf, method='seg', clip_out_of_life=True)
    global _RB
    if _RB is None:
        spec = importlib.util.spec_from_file_location(
            '_broer_anchor', os.path.join(ROOT, 'reproduce_broer_l1.py'))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _RB = mod
    return _RB.time_to_cycle(gid, t)


# ------------------------------------------------------------------ 特征
def _roll_sum(x, w):
    """长度 w 的滑动窗累积（前 w−1 点按已有部分，即"累积起点"语义）。"""
    x = np.asarray(x, float)
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    return c[idx + 1] - c[np.maximum(0, idx + 1 - w)]


def hi_of_group(gid):
    """按 500-cycle 分箱，返回 dict(cyc, cov, hit_bin, en_bin, <各指标>) 或 None。

    论文式10 `HI_AE(t)=Σ_{i=t-T}^{t}F(i)`：窗口 T = 500 cycles 与分箱步长相同
    → 窗口恰含 1 箱，故 "win1_*" = 该箱内的事件量（hits/500cyc、能量/500cyc），
    即**速率型**指标（可升可降 → 单调性天然较低，与论文结论一致）。
    "cum_*" = 全局累积（Broer 的 HI_AE 口径）→ 构造上单调。
    """
    nf = load_meta(gid)['n_f']
    if not nf:
        print(f'  [{gid}] 无 n_f → 跳过')
        return None
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    if not os.path.exists(fp):
        print(f'  [{gid}] 缺 AE → 跳过')
        return None
    ae = pd.read_csv(fp, encoding='utf-8-sig')
    if not len(ae):
        print(f'  [{gid}] AE 空 → 跳过')
        return None
    t = ae['time'].to_numpy(float)
    # 第二批 = 1 s bin 带 n_hits；第一批 = 命中级（每行 1 个事件）→ 与导出器同约定
    n_hits = (ae['n_hits'].to_numpy(float) if 'n_hits' in ae.columns
              else np.ones(len(ae), float))
    en = np.maximum(ae['energy'].to_numpy(float), 0.0)

    ac = np.asarray(_cycle_of(gid, t, nf), float)
    ok = np.isfinite(ac)
    ac, n_hits, en = ac[ok], n_hits[ok], en[ok]
    if ac.size == 0:
        print(f'  [{gid}] AE 全部落在寿命外 → 跳过')
        return None

    # 只用**完整箱**（floor），并丢弃末段不足 1 箱的事件。
    # 否则最后一个残缺箱（如 L1-52 末箱仅 58 cycle）会把速率型指标的
    # 失效值打成 0，产生假低值。丢弃量 ≤ 500 cycle（≤ 1.1% 寿命）。
    nbin = max(2, int(np.floor(nf / CYCS_BIN)))
    cyc_c = (np.arange(nbin) + 0.5) * CYCS_BIN
    keep = ac < nbin * CYCS_BIN
    ac, n_hits, en = ac[keep], n_hits[keep], en[keep]
    if ac.size == 0:
        print(f'  [{gid}] 完整箱内无 AE → 跳过')
        return None
    bi = (ac / CYCS_BIN).astype(int)
    hit_bin = np.bincount(bi, weights=n_hits, minlength=nbin)[:nbin]
    en_bin = np.bincount(bi, weights=en, minlength=nbin)[:nbin]

    # **覆盖率截断**（2026-09-16 修正）：AE 可能在 n_f 之前结束（cov<1），
    # 此时末端的 10 箱窗大部分为空 → 会把"失效值"系统性压低成伪低值。
    # 实测 L1-60 的 cov=0.972 → win10_hits 失效值仅 1.0（AE 恰在 97.2% 处停），
    # 并非其真实活动度。故把所有序列截断到**最后一个完整落在 AE 覆盖内的箱**，
    # 损失 ≤ 1 箱（0.5% 寿命）。注意 cov 字段仍报告真实覆盖率，供读者判断。
    cov_cycle = float(ac.max())
    i_end = int(np.clip(int(np.floor(min(cov_cycle, nf) / CYCS_BIN)) - 1,
                        1, nbin - 1))

    out = {'cyc': cyc_c[:i_end + 1], 'n_f': float(nf),
           'cov': float(min(cov_cycle / nf, 1.0)),
           'hit_bin': hit_bin[:i_end + 1], 'en_bin': en_bin[:i_end + 1]}
    out['cum_hits'] = np.cumsum(out['hit_bin'])
    out['cum_energy'] = np.cumsum(out['en_bin'])
    out['win1_hits'] = out['hit_bin'].copy()
    out['win1_energy'] = out['en_bin'].copy()
    for w in ROLL_WINS:
        out[f'win{w}_hits'] = _roll_sum(out['hit_bin'], w)
        out[f'win{w}_energy'] = _roll_sum(out['en_bin'], w)
    return out


# ------------------------------------------------------------------ 度量
def monotonicity(x):
    """Coble & Hines 单调性（论文引用 [37]）: |正增量−负增量|/(n−1)。"""
    d = np.diff(np.asarray(x, float))
    n = d.size
    if n == 0:
        return np.nan
    return abs(int((d > 0).sum()) - int((d < 0).sum())) / n


def unity01(x):
    x = np.asarray(x, float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if mx - mn < 1e-12 else (x - mn) / (mx - mn)


def first_ge(cyc, x, th):
    """首个 x ≥ th 的 cycle；未达返回 nan。"""
    j = np.where(np.asarray(x, float) >= th)[0]
    return float(cyc[j[0]]) if len(j) else np.nan


def analyse(series, variant):
    """单组的 (Mon, HI_init, HI@50%寿命, HI_fail)。

    HI_fail = 该组 AE 覆盖末端的**原始值**（= 失效时的指标值，跨组比较的基础）。
    """
    x = series[variant]
    i_half = int(np.argmin(np.abs(series['cyc'] / series['n_f'] - 0.5)))
    return dict(mon=monotonicity(x),
                hi_init=float(x[0]),
                hi_half=float(x[i_half]),
                hi_fail=float(x[-1]))


def anchor_ratio(series, variant, pct):
    """方案B：因果归一 u(t) = HI(t) / HI(t_anchor)，t_anchor = pct × 寿命。

    返回 (u 序列, u_fail, 锚点 cycle) 或 None（锚点值 ≤ 0 无法作分母）。
    锚点处的 u 恒为 1；失效时的 u_fail = HI_fail/HI_anchor 即"相对早期已增长多少倍"。
    """
    cyc, nf = series['cyc'], series['n_f']
    i_a = int(np.argmin(np.abs(cyc / nf - pct)))
    a = float(series[variant][i_a])
    if not np.isfinite(a) or a <= 0:
        return None
    u = np.asarray(series[variant], float) / a
    return u, float(u[-1]), float(cyc[i_a])


def loso_threshold(series, variant):
    """留一法（LOSO）：模拟"用历史试件标定阈值 → 对**未见**试件在线判断"的真实工况。

    对每个组 g：阈值 = **其余组**失效值的中位（不含 g）；再回算 g 的达阈寿命%。
    与样本内（阈值包含 g 自己）的结果对比，差异即"外推代价"。
    返回 {gid: 达阈寿命% 或 nan(未达)}。
    """
    gs = list(series)
    out = {}
    for g in gs:
        vf = [series[x][variant][-1] for x in gs
              if x != g and series[x]['cov'] >= COV_MIN]
        if len(vf) < 3:
            out[g] = np.nan
            continue
        c = first_ge(series[g]['cyc'], series[g][variant], float(np.median(vf)))
        out[g] = (c / series[g]['n_f'] * 100.0) if np.isfinite(c) else np.nan
    return out


def anchor_sweep(series, variants=VARIANTS):
    """扫多个锚点，逐 (指标, 锚点) 汇总 u_fail 离散度与达阈时刻。"""
    rows = []
    for v in variants:
        for pct in ANCHOR_PCTS:
            uf, ua, valid = [], [], {}
            for g, s in series.items():
                r = anchor_ratio(s, v, pct)
                if r is None:
                    continue
                valid[g] = r[0]
                uf.append(r[1])
                ua.append(s['cov'])
            uf = np.asarray(uf, float)
            ua = np.asarray(ua, float)
            mc = uf[ua >= COV_MIN]
            if len(mc) < 2:
                continue
            cv = float(np.std(mc) / np.mean(mc))
            den = float(np.mean(np.abs(mc - 1.0)))       # 锚点处 u≡1
            prog = float(np.exp(-np.std(mc) / den)) if den > 1e-12 else np.nan
            thr = float(np.median(mc))
            pcts = []
            for g, u in valid.items():
                c = first_ge(series[g]['cyc'], u, thr)
                pcts.append(c / series[g]['n_f'] * 100.0 if np.isfinite(c) else np.nan)
            pv = np.asarray(pcts, float)
            fin = pv[np.isfinite(pv)]
            rows.append(dict(
                variant=v, anchor_pct=int(pct * 100), n_cal=int((ua >= COV_MIN).sum()),
                n_group=len(pv), fail_cv=round(cv, 3),
                prog=round(prog, 3) if np.isfinite(prog) else None,
                u_thr=round(thr, 3),
                t_thr_min=round(float(fin.min()), 1) if fin.size else None,
                t_thr_med=round(float(np.median(fin)), 1) if fin.size else None,
                t_thr_max=round(float(fin.max()), 1) if fin.size else None,
                n_reach=int(fin.size)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default='2', choices=['1', '2', 'both'])
    ap.add_argument('--groups', default=None)
    a = ap.parse_args()
    groups = GROUPS_V2 if a.batch == '2' else (
        GROUPS_V1 if a.batch == '1' else GROUPS_V2 + GROUPS_V1)
    if a.groups:
        groups = [g.strip() for g in a.groups.split(',') if g.strip()]

    print(f'=== L1 HI_AE（文献口径：500 cycle 窗口/分箱）— {len(groups)} 组 ===')
    series = {}
    for g in groups:
        s = hi_of_group(g)
        if s is not None:
            series[g] = s

    # ---------------- 跨组阈值标定（先算阈值，再逐组反算达阈时刻）----------------
    # 论文口径: prognosability 高 = 各试件**失效时的 HI 值**离散小 → 可设统一阈值。
    # 故阈值取「覆盖足够的组」的失效值中位；再用它反算各组的达阈寿命百分比。
    cal, thr_of = {}, {}
    for v in VARIANTS:
        vf = np.array([s[v][-1] for s in series.values()], float)
        vi = np.array([s[v][0] for s in series.values()], float)
        cv = np.array([s['cov'] for s in series.values()], float)
        m = cv >= COV_MIN
        vf_c, vi_c = vf[m], vi[m]
        den = float(np.mean(np.abs(vf_c - vi_c))) if vf_c.size else 0.0
        if vf_c.size == 0 or den < 1e-12:
            thr_of[v] = np.nan
        else:
            thr_of[v] = float(np.median(vf_c))
        cal[v] = (vf, vi, cv, den)

    # ---------------- 逐组 × 逐指标 ----------------
    # 两套阈值口径并列，因为它们是**不同的问题**：
    #  ① t_thr_pct：用各试件**失效时**的原始值中位作“物理阈值”→ 直接检验
    #     prognosability（跨试件可比性）。这是本数据集论文 B 的原始诉求。
    #  ② t50/t85_pct：各组**自身归一化**后取 0.50/0.85 固定分数 → 即 Broer 2021
    #     Level-4 实际用的方案（离线，靠各 HI 单调性使归一化后 起0终1），
    #     可与第一批已复现的 t85 = 79~95% 直接对比。
    rows = []
    for g, s in series.items():
        for v in VARIANTS:
            m = analyse(s, v)
            th = thr_of[v]
            c = first_ge(s['cyc'], s[v], th) if np.isfinite(th) else np.nan
            u = unity01(s[v])
            c50 = first_ge(s['cyc'], u, 0.50)
            c85 = first_ge(s['cyc'], u, 0.85)
            rows.append(dict(gid=g, variant=v, n_bin=len(s['cyc']),
                             n_f=round(s['n_f']), cov=round(s['cov'], 3),
                             mon=round(m['mon'], 3),
                             hi_init=m['hi_init'], hi_half=m['hi_half'],
                             hi_fail=m['hi_fail'],
                             t_thr_pct=(round(c / s['n_f'] * 100.0, 1)
                                        if np.isfinite(c) else None),
                             t50_pct=(round(c50 / s['n_f'] * 100.0, 1)
                                      if np.isfinite(c50) else None),
                             t85_pct=(round(c85 / s['n_f'] * 100.0, 1)
                                      if np.isfinite(c85) else None)))
    per = pd.DataFrame(rows)
    fp1 = os.path.join(RES, 'l1_hi_ae.csv')
    per.to_csv(fp1, index=False, encoding='utf-8-sig')

    # ---------------- 跨组汇总 ----------------
    loso_of = {v: loso_threshold(series, v) for v in VARIANTS}
    srows = []
    for v in VARIANTS:
        sub = per[per['variant'] == v]
        vf, vi, cv, den = cal[v]
        n_cal = int((cv >= COV_MIN).sum())
        mc = vf[cv >= COV_MIN]
        prog = float(np.exp(-np.std(mc) / den)) if den > 1e-12 else np.nan
        pv = sub['t_thr_pct'].to_numpy(float)
        fin = pv[np.isfinite(pv)]
        f85 = sub['t85_pct'].to_numpy(float)
        f85 = f85[np.isfinite(f85)]
        # LOSO（外推）：阈值 = 其余组的失效值中位
        lv = loso_of[v]
        lfin = np.array([x for x in lv.values() if np.isfinite(x)], float)
        srows.append(dict(
            variant=v, n_cal=n_cal,
            mon_mean=round(float(sub['mon'].mean()), 3),
            mon_min=round(float(sub['mon'].min()), 3),
            prog=round(prog, 3) if np.isfinite(prog) else None,
            fail_cv=round(float(np.std(mc) / np.mean(mc)), 3)
            if len(mc) and np.mean(mc) else None,
            thr=round(thr_of[v], 4) if np.isfinite(thr_of[v]) else None,
            t_thr_min=round(float(fin.min()), 1) if fin.size else None,
            t_thr_med=round(float(np.median(fin)), 1) if fin.size else None,
            t_thr_max=round(float(fin.max()), 1) if fin.size else None,
            n_reach=int(fin.size), n_group=len(pv),
            loso_min=round(float(lfin.min()), 1) if lfin.size else None,
            loso_med=round(float(np.median(lfin)), 1) if lfin.size else None,
            loso_max=round(float(lfin.max()), 1) if lfin.size else None,
            loso_reach=int(lfin.size),
            t85_min=round(float(f85.min()), 1) if f85.size else None,
            t85_med=round(float(np.median(f85)), 1) if f85.size else None,
            t85_max=round(float(f85.max()), 1) if f85.size else None))
    sm = pd.DataFrame(srows)
    fp2 = os.path.join(RES, 'l1_hi_ae_summary.csv')
    sm.to_csv(fp2, index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 200)
    pd.set_option('display.unicode.east_asian_width', True)
    print('\n--- 逐组（原始值；cov=AE 覆盖寿命比例）---')
    print(per.to_string(index=False))
    print('\n--- 跨组汇总（Mon/Prog 见模块 docstring；t_thr* = 达共同阈值时的寿命%）---')
    print(sm.to_string(index=False))

    # LOSO 明细: 直接回答"外推到一个新试件会怎样"
    print('\n--- LOSO 外推明细（阈值 = 其余 8 组的失效值中位）---')
    lo = pd.DataFrame({v: loso_of[v] for v in VARIANTS})
    lo.index.name = 'gid'
    print(lo.round(1).to_string())
    print('\n（None=未达阈，即漏报）')

    # ---------------- 方案B: 因果锚点归一 ----------------
    # 目标: 在**不依赖未来信息**的前提下，把"失效时的指标值"跨组拢到一起。
    # fail_cv ↓ + prog ↑ + t_thr 集中在后段 → 该锚点可用作因果阈值。
    anc = anchor_sweep(series)
    fp3 = os.path.join(RES, 'l1_hi_ae_anchor.csv')
    anc.to_csv(fp3, index=False, encoding='utf-8-sig')
    print('\n--- 方案B: 因果锚点归一 u(t)=HI(t)/HI(锚点) ---')
    print(anc.to_string(index=False))
    best = anc[anc['variant'] == 'cum_hits'].sort_values('fail_cv')
    if len(best):
        b = best.iloc[0]
        print(f"\n[cum_hits] 最优锚点 {b['anchor_pct']}% 寿命: "
              f"失效值 CV={b['fail_cv']}（原始口径 0.803）→ "
              f"t_thr={b['t_thr_min']}~{b['t_thr_max']}%（中位 {b['t_thr_med']}%，"
              f"{b['n_reach']}/{b['n_group']} 可达）")

    # ---------------- 图 ----------------
    made = []
    for v in PLOT_VARIANTS:
        if v not in VARIANTS:
            continue
        sub = per[per['variant'] == v]
        th = sm.loc[sm['variant'] == v, 'thr'].iloc[0]
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
        for g, s in series.items():
            x = s['cyc'] / s['n_f'] * 100.0
            axes[0].plot(x, unity01(s[v]), lw=1.1, label=g)
            axes[1].plot(s['cyc'] / 1000.0, s[v], lw=1.1, label=g)
        axes[0].set_xlabel('寿命 %'); axes[0].set_ylabel('HI（各组自身归一化）')
        axes[0].set_title('归一化：曲线是否重合（prognosability 直观）')
        axes[1].set_xlabel('cycle (k)'); axes[1].set_ylabel('HI（原始值）')
        axes[1].set_title('原始值：失效值离散度（阈值由失效值反推）')
        if th is not None and np.isfinite(th):
            axes[1].axhline(th, color='k', ls='--', lw=1.0,
                            label=f'共同阈值 = 失效值中位 {th:.4g}')
        for ax in axes:
            ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        mono_m = sub['mon'].mean()
        fig.suptitle(f'{VLABEL[v]}   |   平均单调性 Mon={mono_m:.3f}', fontsize=11)
        fig.tight_layout()
        fp = os.path.join(FIG, f'l1_hi_ae_{v}.png')
        fig.savefig(fp, dpi=115); plt.close(fig)
        made.append(fp)

    # 方案B 扫描图: 失效值离散度与达阈时刻 vs 锚点位置
    if len(anc):
        fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
        for v in VARIANTS:
            sub = anc[anc['variant'] == v].sort_values('anchor_pct')
            if not len(sub):
                continue
            axes[0].plot(sub['anchor_pct'], sub['fail_cv'], 'o-', lw=1.2,
                         ms=4, label=v)
            axes[1].plot(sub['anchor_pct'], sub['t_thr_med'], 'o-', lw=1.2,
                         ms=4, label=v)
        axes[0].axhline(0.30, color='k', ls=':', lw=1.0)
        axes[0].text(5, 0.32, 'CV=0.30（可接受上限参考）', fontsize=8)
        axes[0].set_xlabel('锚点位置 / 寿命 %')
        axes[0].set_ylabel('失效时相对值的 CV')
        axes[0].set_title('方案B：因果归一后，失效值还差多少？')
        axes[1].axhline(85.0, color='k', ls=':', lw=1.0)
        axes[1].text(5, 86.0, '85% 寿命参考', fontsize=8)
        axes[1].set_xlabel('锚点位置 / 寿命 %')
        axes[1].set_ylabel('达阈中位时刻 / 寿命 %')
        axes[1].set_title('达阈时刻（越大越晚、越安全）')
        for ax in axes:
            ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        fig.suptitle('方案B 扫描：u(t) = HI(t) / HI(早期锚点)  （纯因果，无未来信息）',
                     fontsize=11)
        fig.tight_layout()
        fp = os.path.join(FIG, 'l1_hi_ae_anchor.png')
        fig.savefig(fp, dpi=115); plt.close(fig)
        made.append(fp)

    print('\n图:', *made, sep='\n  ')
    print(f'\n产物: {fp1}\n      {fp2}\n      {fp3}')


if __name__ == '__main__':
    main()
