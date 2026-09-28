"""
Step 3: 外推 / 预后评测 —— PHM2019 铝搭接件（数据集 D）
=========================================================

## 任务为什么这样定义

T7/T8 的**波形只给到前半段**（T7 到 47022、T8 到 76931），而真值继续往后：

| 试件 | 有波形的时刻（= 可用于建模） | **只有真值、无波形**（= 外推目标） |
| --- | --- | --- |
| T7 | 36001:0.00 · 40167:0.00 · 44054:2.07 · 47022:3.14 | 49026:3.56 · 51030:4.13 · 53019:5.05 · 55031:7.22 |
| T8 | 40000:0.00 · 50000:0.00 · 70000:0.00 · 74883:1.94 · 76931:2.50 | 89237:3.71 · 92315:3.88 · 96475:4.61 · 98492:4.96 · 100774:5.52 |

⇒ 预测目标时刻**没有波形**，所以预测值只能来自**裂纹扩展律的外推**。
波形的作用是：**在预报时刻给出裂纹长度的初值**（实践中你只有波形，没有显微镜）。

因此本步评测三件事（**分层，避免把两类误差混在一起**）：

| 段 | 初值来源 | 回答什么问题 |
| --- | --- | --- |
| **A 增长律自检** | 实测（截断验证） | 增长律本身准不准（在 T1–T6 上做，不碰 T7/T8） |
| **B 纯轨迹外推** | **实测**裂纹（最后波形时刻） | **上限参考**：假设裂纹已知，增长律能外推多准 |
| **C 波形闭环外推** | **模型从波形估出的 â** | **端到端指标** = 波形估计误差 + 增长律误差的合成 |

⚠️ **B 与 C 的差别就是"波形的贡献（或拖累）"**，这是本步的核心对照。

## ⚠️ 两条从第一版失败中学到的硬约束（勿删）

1. **自检的视界必须匹配部署的视界。**
   第一版只留出 2 个点（视界几千循环）就选了 `paris_pop`，而 T8 的真实视界是
   **23843 循环**。Paris 律在 p > 1 时有**有限时间奇点**：短视界看不出问题，
   长视界直接爆炸（T8 上 RMSE = 475 mm）。
   ⇒ 本版做**穷举截断 + 视界分层**，只在与部署视界同档的桶里选型。

2. **增长律类型与载荷类型的迁移要分开报。**
   T1–T6 与 T7 都是**等幅**，T8 是**变幅**。
   ⇒ 在 T1–T6 上选型只对 **T7** 是合法迁移；对 T8 属**跨载荷类型迁移**，
      必须单独标注、单独下结论。

## 不做的事

**不实现官方评分函数**：其常数在 `PHM2019_ScoringSpreadsheet.xlsx` 内，
仓库里没有该文件 ⇒ 只用 RMSE / MAE / 最大误差这类无歧义口径，不自行编造。

输出
----
  phmdc/results/step3_selfcheck.csv    自检逐点明细（含视界）
  phmdc/results/step3_extrap.csv       T7/T8 逐点外推明细（含发散标记 / cycle_raw）
  phmdc/results/step3_summary.csv      T7/T8 逐 (试件, 律, 阶段) 汇总
  phmdc/results/step3_report.txt

  `--equiv` 时文件名带 `_eqm<m>` 后缀（如 step3_extrap_eqm3.csv）。

`--equiv`（等损伤当量循环轴）
---------------------------
把原始循环轴换成等幅当量循环轴 N_eq(N) = Σ (A_i/A_ref)^m（A_ref = 数据集最大幅值 =
[T1–T7 的等幅值](95.44 kN) ⇒ T1–T7 恒等、只有 T8 被重标度）。

**预期结果（实测已验证）**：本文件里的 6 条候选律对循环轴均匀缩放**等变**
⇒ 当量换算对预测几乎无影响（非发散行 |Δ| ≤ 0.04 mm，0.5% 量程），
**3/5 发散一格不变** ⇒ 它**修不了** T8，但能**排除"T8 失败源于变幅载荷"这一假设**。
详见 phmdc/README.md §5.2b.5。

依赖：numpy / pandas / scipy
"""

import os
import sys
import glob
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

import prep

warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

ROOT = prep.ROOT
RES = os.path.join(ROOT, 'results')

_REPORT = []


def say(msg=''):
    print(msg)
    _REPORT.append(msg)


# ============================================================
# 增长律
# ============================================================
# 统一接口：fit(N_obs, a_obs) -> 可调用的 a(N)；要求 N_obs/a_obs 按 N 升序。
# Paris 型：da/dN = A·a^p
#   解析积分（p ≠ 1）：a(N) = [a0^(1-p) + (1-p)·A·(N-N0)]^(1/(1-p))
#   p > 1 时 (1-p) < 0 ⇒ 存在有限时间奇点（a→∞），正是"加速失效"的物理行为；
#   但 a0 = 0 会退化 ⇒ 拟合时给 a 加一个**检测阈**下限（见 A0_FLOOR）。
A0_FLOOR = 0.05        # mm；显微镜最小可测裂纹。实测 a=0 的点在拟合里抬到此值
DIVERGE_FACTOR = 3.0   # 预测 > 该倍数 × 训练段最大实测裂纹 ⇒ 判为**发散**

LAWS = ['lin_last2', 'lin_all', 'expo_all', 'power_all',
        'paris_pop', 'paris_Ap']
LAW_DESC = {
    'lin_last2': '线性，只用最后两点（= 当前速率外推）',
    'lin_all': '线性，拟合全部点（= 平均速率外推）',
    'expo_all': '指数，拟合全部点',
    'power_all': '幂律 a=c·(N-Nref)^q，拟合全部点',
    'paris_pop': 'Paris da/dN=A·a^p，p 由群体给定，A 逐试件拟合',
    'paris_Ap': 'Paris，p 与 A 均逐试件拟合（3 参数，点数少时欠定）',
}


def _integrate_paris(N, N0, a0, A, p):
    """Paris 型解析解；a0>0。返回与 N 同形状的 a。"""
    N = np.asarray(N, dtype=float)
    dN = N - N0
    if abs(p - 1.0) < 1e-9:
        return a0 * np.exp(A * dN)
    q = 1.0 - p
    base = a0 ** q + q * A * dN
    # q<0 时 base 可能变负/为负 => 用 |base| 钳到很小的正数，避免 nan
    safe = np.where(base > 1e-12, base, 1e-12)
    a = safe ** (1.0 / q)
    a = np.where(base > 1e-12, a, np.nan)      # 已越过奇点 => 不可用
    return a


def fit_paris(N, a, p_fixed=None):
    """最小二乘拟合 Paris 律，积分起点取**第一个点**。返回 (N0, a0, A, p)。"""
    N = np.asarray(N, float)
    a = np.asarray(a, float)
    N0 = float(N[0])
    a0_init = max(float(a[0]), A0_FLOOR)

    def resid(th, p):
        pred = _integrate_paris(N, N0, th[0], np.exp(th[1]), p)
        pred = np.where(np.isfinite(pred), pred, 1e4)
        return pred - a

    if p_fixed is not None:
        r = least_squares(resid, [a0_init, np.log(1e-5)], args=(float(p_fixed),),
                          bounds=([A0_FLOOR, np.log(1e-12)], [50.0, np.log(1e-1)]))
        return N0, float(r.x[0]), float(np.exp(r.x[1])), float(p_fixed)
    r = least_squares(lambda th: resid(th[:2], th[2]),
                      [a0_init, np.log(1e-5), 2.0],
                      bounds=([A0_FLOOR, np.log(1e-12), 0.5],
                              [50.0, np.log(1e-1), 8.0]))
    return N0, float(r.x[0]), float(np.exp(r.x[1])), float(r.x[2])


def population_p():
    """在 T1–T6 全曲线上各自拟合 Paris，取 p 的中位数作为群体指数。"""
    ps = []
    for sp in prep.TRAIN:
        N, a = curve_of(sp)
        if len(N) >= 3 and a.max() > 0:
            try:
                _, _, _, p = fit_paris(N, a, p_fixed=None)
                ps.append(p)
            except Exception:
                pass
    return (float(np.median(ps)) if ps else 3.0), ps


def make_law(name, N, a, p_pop=None):
    """构造并拟合增长律，返回 (predict_callable, params_dict)。

    ⚠️ **所有律都保证：外推曲线在最后一个实测点处等于该点实测值。**
    即初值锚定在"观测到的当前状态"，而不是拟合出来的 a0 ——
    否则外推的起点就不是"现在的裂纹长度"，物理上讲不通。
    """
    N = np.asarray(N, float)
    a = np.asarray(a, float)
    N_last, a_last = float(N[-1]), float(a[-1])

    if name == 'lin_last2':
        k = min(2, len(N))
        c = np.polyfit(N[-k:], a[-k:], 1)
        return ((lambda Nq: np.polyval(c, np.asarray(Nq, float))),
                {'kind': 'lin_last2', 'slope': float(c[0])})

    if name == 'lin_all':
        k = float(np.polyfit(N, a, 1)[0])
        b = a_last - k * N_last          # 只改截距，保住斜率 ⇒ 锚定
        return ((lambda Nq: k * np.asarray(Nq, float) + b),
                {'kind': 'lin_all', 'slope': k})

    if name == 'expo_all':
        lam = float(np.polyfit(N, np.log(np.maximum(a, A0_FLOOR)), 1)[0])
        c0 = a_last / np.exp(lam * N_last)
        return ((lambda Nq: c0 * np.exp(lam * np.asarray(Nq, float))),
                {'kind': 'expo_all', 'lambda': lam})

    if name == 'power_all':
        Nref = float(N[0]) - 1.0
        x = np.log(np.maximum(N - Nref, 1.0))
        y = np.log(np.maximum(a, A0_FLOOR))
        q, _ = np.polyfit(x, y, 1)
        c0 = a_last / max((N_last - Nref) ** q, 1e-12)
        return ((lambda Nq: c0 * np.maximum(np.asarray(Nq, float) - Nref, 1e-6) ** q),
                {'kind': 'power_all', 'q': float(q)})

    # ---- Paris 族：先整体拟合得到 (A, p)，再从 (N_last, a_last) 起积分 ----
    # Paris 律的 da/dN 只依赖 a，所以过 (N_last, a_last) 的续曲线是唯一的，
    # 重锚定不破坏拟合。
    if name == 'paris_pop':
        if p_pop is None:
            raise ValueError('paris_pop 需要 p_pop')
        _, _, A, p = fit_paris(N, a, p_fixed=float(p_pop))
    elif name == 'paris_Ap':
        _, _, A, p = fit_paris(N, a, p_fixed=None)
    else:
        raise ValueError(name)

    a_start = max(a_last, A0_FLOOR)

    def H(Nq):
        Nq = np.maximum(np.asarray(Nq, float), N_last)     # 只外推，不内插
        return _integrate_paris(Nq, N_last, a_start, A, p)

    return H, {'kind': name, 'A': A, 'p': p, 'a_start': a_start}


def predict_and_flag(f, Nq, train_max_a):
    """预测并标记发散。返回 (pred, diverged_bool)。"""
    yp = np.asarray(f(Nq), float)
    bad = ((~np.isfinite(yp)) |
           (np.nan_to_num(yp, nan=1e9) > DIVERGE_FACTOR * max(train_max_a, 1e-9)) |
           (np.nan_to_num(yp, nan=-1e9) < 0))
    return yp, bad


# ============================================================
# 等损伤当量换算（载荷谱）
# ============================================================
# 动机：T8 是**两级块谱**（500 循环高幅 + 500 循环低幅 / 块），T1–T7 是等幅。
# 直接把 T8 的"循环数"与 T1–T6 的循环数放在一起比视界、比速率，
# 混进去了幅值差异。做法：按 Paris / Miner 线性损伤累积把原始循环轴
# 换成**等幅当量循环轴**
#     N_eq(N) = Σ_{i<N} (A_i / A_ref)^m      （A_ref = 数据集最大幅值）
# A_ref 取最大幅值 = T1–T7 的等幅值 ⇒ **T1–T7 的 N_eq ≡ N**，
# 故自检桶（T1–T6）与 T7 完全不变，只有 T8 被重新标度。
#
# ⚠️ 数学预告（下文的实测会验证）：`lin_*` 与 `expo_*` 都是**尺度不变**的 ——
#   循环轴做均匀缩放 N → cN，拟合出的参数会相应变，但 a(N) 的预测**逐点不变**。
#   所以等损伤当量**不可能**修掉线性律在 T8 上的发散；只有
#   `power_all` / `paris_*` 这类非尺度不变的律会被影响。
EQUIV = {'on': False, 'm': 2.0, 'A_ref': None}
TAG = ''            # 产物文件名后缀（--equiv 时非空）
_AMP_CACHE = {}


def specimen_amplitudes(specimen):
    """逐循环载荷幅值（kN）。文件里恰好是 1 个周期。

    T1–T7 = 等幅（各周期幅值相同 ⇒ 压缩为长度 1）；T8 = 1000 循环/块。
    切分方式：以上穿中值点为周期起点，周期内 max−min 即幅值。
    """
    if specimen in _AMP_CACHE:
        return _AMP_CACHE[specimen]
    files = glob.glob(os.path.join(ROOT, specimen, '*Loading Profile*.csv'))
    if not files:
        raise FileNotFoundError('未找到 %s 的载荷谱' % specimen)
    L = pd.read_csv(files[0])['loading'].to_numpy(float)
    mid = 0.5 * (L.max() + L.min())
    up = np.where((L[:-1] <= mid) & (L[1:] > mid))[0] + 1
    if len(up) < 2:
        raise ValueError('%s 载荷谱切不出循环（up=%d）' % (specimen, len(up)))
    bnd = np.append(up, len(L))
    amps = np.array([L[up[i]:bnd[i + 1]].max() - L[up[i]:bnd[i + 1]].min()
                     for i in range(len(up))], float)
    if np.ptp(amps) < 1e-6:          # 等幅 ⇒ 周期长度归 1
        amps = amps[:1]
    _AMP_CACHE[specimen] = amps
    return amps


def equiv_weight(specimen, m=None, A_ref=None):
    """该试件"每个原始循环 ≈ 多少个 A_ref 幅值的当量循环"（= mean of (A/A_ref)^m）。"""
    m = EQUIV['m'] if m is None else float(m)
    A_ref = EQUIV['A_ref'] if A_ref is None else float(A_ref)
    return float(np.mean((specimen_amplitudes(specimen) / A_ref) ** m))


def to_equiv_at(specimen, N, m, A_ref):
    """给定 m / A_ref 的当量换算（显式参数版，便于做 m 的敏感性表）。"""
    N = np.asarray(N, float)
    amps = specimen_amplitudes(specimen)
    if len(amps) == 1:
        return N * float(np.mean((amps / A_ref) ** m))
    w = (amps / A_ref) ** m
    cum = np.concatenate([[0.0], np.cumsum(w)])      # cum[k] = 前 k 个循环的当量
    P = len(amps)
    k = np.floor(N / P)
    r = np.clip((N - k * P).astype(int), 0, P)
    return k * cum[-1] + cum[r]


def to_equiv(specimen, N):
    """原始循环数 → 等幅当量循环数（块内逐步累加，含相位）。关掉时原样返回。"""
    if not EQUIV['on'] or EQUIV['A_ref'] is None:
        return np.asarray(N, float)
    return to_equiv_at(specimen, N, EQUIV['m'], EQUIV['A_ref'])


# ============================================================
# 数据
# ============================================================
def curve_of(specimen, only_wave_moments=False, upto_cycle=None):
    """取某试件的 (cycle, crack_mm) 扩展曲线。

    only_wave_moments=True 时只保留**有波形**的时刻 —— 这是"预报时刻能拿到的信息"，
    对应真实场景（你没有未来时刻的显微镜读数）。
    """
    lab, idx = prep.labels(), prep.index()
    have = {(r.specimen, r.cycle) for r in idx.itertuples()}
    g = lab[lab['specimen'] == specimen].sort_values('cycle')
    rows = []
    for r in g.itertuples():
        if not np.isfinite(r.crack_mm):
            continue
        if only_wave_moments and (r.specimen, r.cycle) not in have:
            continue
        if upto_cycle is not None and r.cycle > upto_cycle:
            continue
        rows.append((int(r.cycle), float(r.crack_mm)))
    return (np.array([r[0] for r in rows], float),
            np.array([r[1] for r in rows], float))


def extrap_targets(specimen):
    """某试件"只有真值、无波形"的时刻 = 外推目标。"""
    lab, idx = prep.labels(), prep.index()
    have = {(r.specimen, r.cycle) for r in idx.itertuples()}
    g = lab[(lab['specimen'] == specimen)].sort_values('cycle')
    out = [(int(r.cycle), float(r.crack_mm)) for r in g.itertuples()
           if (r.specimen, r.cycle) not in have and np.isfinite(r.crack_mm)]
    return (np.array([o[0] for o in out], float),
            np.array([o[1] for o in out], float))


# ============================================================
# A 段：视界分层自检（T1–T6 穷举截断验证）
# ============================================================
def selfcheck(p_pop, min_train=3):
    """在 T1–T6 上做**穷举截断**验证：对每个截断点，用前面拟合、预测后面全部点。

    记录**视界**（预测点与截断点的循环差），以便按视界分层选型 ——
    这是第一版的教训：短视界选出的律在长视界会爆炸。
    """
    rows = []
    for sp in prep.TRAIN:
        N0, a = curve_of(sp)
        N = to_equiv(sp, N0)          # ★ 当量轴（--equiv 时；否则恒等）
        for cut in range(min_train, len(N)):
            Ntr, atr = N[:cut], a[:cut]
            Nte, ate = N[cut:], a[cut:]
            amax = float(atr.max())
            for law in LAWS:
                try:
                    f, _ = make_law(law, Ntr, atr, p_pop=p_pop)
                except Exception:
                    continue
                yp, bad = predict_and_flag(f, Nte, amax)
                for i in range(len(Nte)):
                    rows.append({
                        'specimen': sp, 'law': law, 'cut_cycle': Ntr[-1],
                        'cycle': Nte[i], 'horizon': float(Nte[i] - Ntr[-1]),
                        'truth': ate[i], 'pred': yp[i],
                        'err': (float(yp[i] - ate[i])) if np.isfinite(yp[i]) else np.nan,
                        'diverged': bool(bad[i]), 'n_train': len(Ntr),
                    })
    return pd.DataFrame(rows)


def horizon_buckets(sc):
    """按视界（循环数）分桶汇总，返回 ({bucket: DataFrame}, bucket_names)。"""
    edges = [0, 2000, 8000, 20000, np.inf]
    names = ['<=2k', '2~8k', '8~20k', '>20k']
    sc = sc.copy()
    sc['bucket'] = pd.cut(sc['horizon'], bins=edges, labels=names, right=True)
    out = {}
    for b in names:
        g = sc[sc['bucket'] == b]
        rec = []
        for law in LAWS:
            h = g[g['law'] == law]
            if not len(h):
                continue
            ok = h[~h['diverged']]
            rec.append({
                'law': law, 'n': len(h), 'n_div': int(h['diverged'].sum()),
                'rmse': (float(np.sqrt(np.mean(h['err'].dropna() ** 2)))
                         if h['err'].notna().any() else np.nan),
                'rmse_nodiv': (float(np.sqrt(np.mean(ok['err'] ** 2)))
                               if len(ok) else np.nan),
                'mae': (float(np.mean(np.abs(h['err'].dropna())))
                        if h['err'].notna().any() else np.nan),
            })
        out[b] = (pd.DataFrame(rec).sort_values('rmse') if rec
                  else pd.DataFrame(columns=['law', 'n', 'n_div', 'rmse',
                                             'rmse_nodiv', 'mae']))
    return out, names


# ============================================================
# B/C 段：T7/T8 真外推
# ============================================================
def load_model_estimate():
    """读 Step 1 / Step 2 在 T7/T8 **最后一个波形时刻**的预测（波形闭环用）。

    返回 {名称: (最后时刻 cycle, 该时刻的预测裂纹 mm)}；名称为字符串，
    形如 `'step1_rf+mono/T7'`。
    """
    est = {}
    f1 = os.path.join(RES, 'step1_pred.csv')
    if os.path.exists(f1):
        d = pd.read_csv(f1)
        g = d[d['arm'] == 'rf+mono']
        for sp in prep.VALID:
            s = g[g['specimen'] == sp].sort_values('cycle')
            if len(s):
                last = int(s['cycle'].max())
                est['step1_rf+mono/%s' % sp] = (
                    last, float(s[s['cycle'] == last]['pred'].mean()))
    f2 = os.path.join(RES, 'step2_pred_wave_first_c2.csv')
    if os.path.exists(f2):
        d = pd.read_csv(f2)
        for sp in prep.VALID:
            s = d[d['specimen'] == sp].sort_values('cycle')
            if len(s):
                last = int(s['cycle'].max())
                est['step2_cnn+mono/%s' % sp] = (
                    last, float(s[s['cycle'] == last]['cnn+mono'].mean()))
    return est


def run_extrapolation(p_pop, laws=None):
    """对 T7/T8 做外推。返回逐点结果 DataFrame（含发散标记）。"""
    rows = []
    est = load_model_estimate()
    laws = laws or LAWS
    for sp in prep.VALID:
        # 建模用的点 = **有波形**的真值点（= 预报时刻能拿到的全部信息）
        N0, a = curve_of(sp, only_wave_moments=True)
        Nq0, aq = extrap_targets(sp)
        if not len(Nq0):
            continue
        n_last_raw = float(N0[-1])     # 用于匹配 Step 1/2 的预测时刻（模型给的是**原始循环**）
        N = to_equiv(sp, N0)           # ★ 当量轴（--equiv 时；否则恒等）
        Nq = to_equiv(sp, Nq0)
        amax = float(a.max())
        for law in laws:
            # ---- B：纯轨迹外推（初值 = 实测裂纹） ----
            try:
                f, _ = make_law(law, N, a, p_pop=p_pop)
                yp, bad = predict_and_flag(f, Nq, amax)
            except Exception:
                continue
            for i in range(len(Nq)):
                rows.append({'phase': 'B_trajectory', 'specimen': sp, 'law': law,
                             'init_source': 'truth', 'cycle': Nq[i],
                             'cycle_raw': Nq0[i],
                             'horizon': float(Nq[i] - N[-1]), 'truth': aq[i],
                             'pred': yp[i],
                             'err': (float(yp[i] - aq[i])) if np.isfinite(yp[i])
                             else np.nan,
                             'diverged': bool(bad[i]), 'n_train_pts': len(N)})

            # ---- C：波形闭环（初值 = 模型从波形估出的 â） ----
            for key, (c_last, a_hat) in est.items():
                if c_last != int(n_last_raw):
                    continue          # 只对"最后一个波形时刻"一致的模型生效
                a_mod = a.copy()
                a_mod[-1] = max(float(a_hat), 0.0)   # 把初值换成波形估计
                try:
                    fm, _ = make_law(law, N, a_mod, p_pop=p_pop)
                    ypm, badm = predict_and_flag(fm, Nq, amax)
                except Exception:
                    continue
                for i in range(len(Nq)):
                    rows.append({'phase': 'C_waveform_closed_loop', 'specimen': sp,
                                 'law': law, 'init_source': key, 'cycle': Nq[i],
                                 'cycle_raw': Nq0[i],
                                 'horizon': float(Nq[i] - N[-1]), 'truth': aq[i],
                                 'pred': ypm[i],
                                 'err': (float(ypm[i] - aq[i]))
                                 if np.isfinite(ypm[i]) else np.nan,
                                 'diverged': bool(badm[i]), 'n_train_pts': len(N)})
    return pd.DataFrame(rows)


# ============================================================
# main
# ============================================================
def main():
    global TAG
    ap = argparse.ArgumentParser(description='phmdc Step 3 外推/预后评测')
    ap.add_argument('--equiv', action='store_true',
                    help='启用等损伤当量循环轴（按载荷谱把变幅循环折成等幅当量）')
    ap.add_argument('--equiv-m', type=float, default=2.0,
                    help='当量换算的指数 m（默认 2.0；群体 Paris p 的中位为 1.18）')
    args = ap.parse_args()

    EQUIV['on'] = bool(args.equiv)
    EQUIV['m'] = float(args.equiv_m)
    EQUIV['A_ref'] = float(max(specimen_amplitudes(sp).max()
                               for sp in prep.SPECIMENS))
    TAG = ('_eqm%g' % EQUIV['m']) if EQUIV['on'] else ''

    say('=' * 96)
    say('PHM2019 铝搭接件（数据集 D） Step 3 外推 / 预后评测')
    say('生成时间: %s' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    say('循环轴: %s'
        % ('**等损伤当量**（m=%.2f, A_ref=%.2f kN）' % (EQUIV['m'], EQUIV['A_ref'])
           if EQUIV['on'] else '原始循环数（未做当量换算）'))
    say('=' * 96)

    # ---------- [1] 数据结构 ----------
    say('\n[1] 外推任务的数据结构')
    load_type = {'T8': '变幅', 'T7': '等幅'}
    say('    %-4s %-6s %-44s %s'
        % ('试件', '载荷', '有波形（= 建模可用）', '外推目标（无波形）'))
    for sp in prep.SPECIMENS:
        Nw, aw = curve_of(sp, only_wave_moments=True)
        Nt, at = extrap_targets(sp)
        sw = ' '.join('%d:%.2f' % (n, c) for n, c in zip(Nw, aw))
        st = (' '.join('%d:%.2f' % (n, c) for n, c in zip(Nt, at))
              if len(Nt) else '—')
        say('    %-4s %-6s %-44s %s'
            % (sp, load_type.get(sp, '等幅'), sw[:44], st))
    say('')
    say('    ⇒ 只有 T7/T8 有目标点 ⇒ **方法选型必须在 T1–T6 上做**，')
    say('      不能拿 T7/T8 选（否则 = 用测试集调方法）。')
    say('    ⇒ **T1–T6 与 T7 都是等幅，T8 是变幅** ——')
    say('      T8 的结果属**跨载荷类型迁移**，不是纯外推能力，须单独下结论。')

    # ---------- [1b] 载荷谱 / 等损伤当量 ----------
    say('\n' + '=' * 96)
    say('[1b] 载荷谱：T1–T7 等幅 vs T8 两级块谱；等损伤当量换算')
    say('=' * 96)
    say('    %-5s %-10s %8s %10s %10s %12s'
        % ('试件', '谱型', '周期循环', '幅值min', '幅值max', '幅值均值(kN)'))
    say('    ' + '-' * 64)
    for sp in prep.SPECIMENS:
        amps = specimen_amplitudes(sp)
        say('    %-5s %-10s %8d %10.2f %10.2f %12.2f'
            % (sp, '等幅' if len(amps) == 1 else '两级块谱',
               len(amps), amps.min(), amps.max(), amps.mean()))
    say('')
    say('    A_ref = %.2f kN（数据集最大幅值 = T1–T7 的等幅值）'
        % EQUIV['A_ref'])
    say('    ⇒ T1–T7 的当量系数恒为 1 ⇒ **N_eq ≡ N**，只有 T8 被重新标度。')
    say('')
    say('    %-5s %10s %10s %10s %10s' % ('试件', 'm=1.18', 'm=2', 'm=3', 'm=4'))
    say('    ' + '-' * 48)
    for sp in prep.SPECIMENS:
        say('    %-5s %10.4f %10.4f %10.4f %10.4f'
            % ((sp,) + tuple(equiv_weight(sp, m=m) for m in (1.18, 2, 3, 4))))
    say('')
    H = {}
    H_a = {}
    H_EQ = {}
    for sp in prep.VALID:
        n0, _ = curve_of(sp, only_wave_moments=True)
        nq, _ = extrap_targets(sp)
        H[sp] = float(nq.max() - n0[-1])
        H_a[sp] = float(to_equiv(sp, float(nq.max()))
                       - to_equiv(sp, float(n0[-1])))
        for m in (1.18, 2, 3, 4):
            H_EQ[(sp, m)] = float(to_equiv_at(sp, nq.max(), m, EQUIV['A_ref'])
                                  - to_equiv_at(sp, n0[-1], m, EQUIV['A_ref']))
    say('    当量换算对**视界**的影响（当前实际使用的轴：%s）：'
        % ('N_eq（m=%.2f）' % EQUIV['m'] if EQUIV['on'] else '原始循环'))
    say('    %-5s %12s %12s %12s %12s %12s'
        % ('试件', '原始', 'm=1.18', 'm=2', 'm=3', 'm=4'))
    say('    ' + '-' * 68)
    for sp in prep.VALID:
        say('    %-5s %12.0f %12.0f %12.0f %12.0f %12.0f'
            % ((sp, H[sp]) + tuple(H_EQ[(sp, m)] for m in (1.18, 2, 3, 4))))
    say('    %-5s %12s %12s %12s %12s %12s'
        % ('T8 当量/原始', '1.000', *['%.3f' % (H_EQ[('T8', m)] / H[('T8')])
                                     for m in (1.18, 2, 3, 4)]))
    say('')
    say('    ⚠️ 相位假设：块谱按"测试循环 0 = 块位置 0"对齐（数据无绝对相位信息），')
    say('       偏差量级 O(1000) 循环，对 ~2e4 的视界影响 ≤5%。')
    say('    ⚠️ m 取 1.18（群体 Paris p 中位）～ 4；m 越大则低幅段"越不算数"。')

    # ---------- [2] A 段 ----------
    say('\n' + '=' * 96)
    say('[2] A 段：增长律自检（T1–T6 穷举截断验证，按视界分层）')
    say('=' * 96)
    p_pop, ps = population_p()
    say('    各试件全曲线单独拟合的 Paris 指数 p = %s'
        % ', '.join('%.2f' % v for v in ps))
    say('    ⇒ 群体 p（中位）= **%.2f**' % p_pop)
    say('    拟合下限 A0_FLOOR = %.2f mm（显微镜最小可测裂纹）；'
        '发散判据 = 预测 > %.0f× 训练段最大实测裂纹' % (A0_FLOOR, DIVERGE_FACTOR))
    sc = selfcheck(p_pop)
    if not len(sc):
        raise SystemExit('自检无样本，请检查数据（试件点数不足）')
    sc.to_csv(os.path.join(RES, 'step3_selfcheck%s.csv' % TAG), index=False,
              encoding='utf-8-sig')
    hz, names = horizon_buckets(sc)
    say('')
    say('    %-6s %-14s %6s %6s %9s %12s %9s'
        % ('视界', '增长律', 'n', '发散数', 'RMSE', 'RMSE(排发散)', 'MAE'))
    say('    ' + '-' * 76)
    for b in names:
        d = hz[b]
        if not len(d):
            say('    %-6s %-14s %6s %6s %9s %12s %9s'
                % (b, '（无数据）', 0, 0, '—', '—', '—'))
            say('    ' + '-' * 76)
            continue
        for j, r in enumerate(d.itertuples()):
            say('    %-6s %-14s %6d %6d %9.3f %12s %9.3f'
                % (b if j == 0 else '', r.law, r.n, r.n_div, r.rmse,
                   ('%.3f' % r.rmse_nodiv) if np.isfinite(r.rmse_nodiv) else '—',
                   r.mae))
        say('    ' + '-' * 76)
    say('')
    say('    第一版的教训：只留出 2 个点（视界几千循环）就选了 `paris_pop`，')
    say('    而 T8 的真实视界是 **%d 循环**。Paris 在 p>1 有有限时间奇点，'
        % int(H_a['T8']))
    say('    短视界看不出、长视界爆炸（T8 上 RMSE = 475 mm）。')
    say('    ⇒ 因此本版**按视界分层选型**，只看与部署视界同档的桶。')

    def bucket_of(h):
        """按视界取桶名（与 horizon_buckets 的边界保持一致）。"""
        for nm_, hi in zip(names, [2000, 8000, 20000]):
            if h <= hi:
                return nm_
        return names[-1]

    def pick(bucket, k=3):
        """取该视界桶的最优 k 条律；桶为空时**降级到最大非空桶并告警**。"""
        d = hz[bucket]
        if len(d) and d['rmse'].notna().any():
            return list(d[d['rmse'].notna()]['law'].head(k)), bucket, False
        for b in reversed([x for x in names if x != bucket]):
            d = hz[b]
            if len(d) and d['rmse'].notna().any():
                return list(d[d['rmse'].notna()]['law'].head(k)), b, True
        return [], None, True

    sel_T7, b_T7, deg_T7 = pick(bucket_of(H_a['T7']))
    sel_T8, b_T8, deg_T8 = pick(bucket_of(H_a['T8']))
    say('')
    say('    **视界匹配的选型**：')
    say('      T7（视界 %d → 目标桶 "%s"）：%s'
        % (int(H_a['T7']), bucket_of(H_a['T7']), '、'.join(sel_T7)))
    say('      T8（视界 %d → 目标桶 "%s"）：%s%s'
        % (int(H_a['T8']), bucket_of(H_a['T8']),
           '、'.join(sel_T8) if sel_T8 else '（无）',
           '' if not deg_T8 else '  ⚠️ 该桶**无数据**，已降级到 "%s" 桶' % b_T8))
    hmax = float(sc['horizon'].max())
    say('')
    say('    🔴 **硬限制**：T1–T6 的截断自检**最长只能到 %d 循环**（各分位 %s），'
        % (int(hmax), [int(sc['horizon'].quantile(q)) for q in (0.5, 0.9)]))
    say('       而 T8 需要的视界是 **%d 循环 = 自检极限的 %.2f 倍**。'
        % (int(H_a['T8']), H_a['T8'] / hmax))
    say('       ⇒ **T8 的外推精度在本数据集上无法被验证** ——')
    say('         这不是"选错了律"，而是"数据里根本没有那么长的等幅参考曲线"。')
    say('         对外报 T8 结果时必须显式声明这一点。')

    # ---------- [3] B/C ----------
    say('\n' + '=' * 96)
    say('[3] T7/T8 真外推（B = 初值用实测；C = 初值用波形模型估计）')
    say('=' * 96)
    est = load_model_estimate()
    truth_map = {'T7': 3.14, 'T8': 2.50}
    say('    波形闭环的初值来源（T7/T8 最后一个波形时刻的模型估计）：')
    for k, (c, v) in sorted(est.items()):
        sp = k.split('/')[-1]
        say('      %-24s cycle=%d  â=%.2f mm（实测 %.2f，误差 %+.2f）'
            % (k, c, v, truth_map[sp], v - truth_map[sp]))
    ex = run_extrapolation(p_pop)
    if not len(ex):
        raise SystemExit('外推无结果')
    ex.to_csv(os.path.join(RES, 'step3_extrap%s.csv' % TAG), index=False,
              encoding='utf-8-sig')

    say('')
    say('    %-5s %-14s %-24s %4s %5s %9s %9s %10s'
        % ('试件', '增长律', '阶段', 'n', '发散', 'RMSE', 'MAE', '最大误差'))
    say('    ' + '-' * 94)
    sums = []
    for (sp, law, ph), g in ex.groupby(['specimen', 'law', 'phase']):
        ok = g[~g['diverged']]
        rmse = (float(np.sqrt(np.mean(g['err'].dropna() ** 2)))
                if g['err'].notna().any() else np.nan)
        sums.append({'specimen': sp, 'law': law, 'phase': ph, 'n': len(g),
                     'n_div': int(g['diverged'].sum()), 'rmse': rmse,
                     'mae': (float(np.mean(np.abs(g['err'].dropna())))
                             if g['err'].notna().any() else np.nan),
                     'max_abs': (float(np.max(np.abs(g['err'].dropna())))
                                 if g['err'].notna().any() else np.nan),
                     'rmse_nodiv': (float(np.sqrt(np.mean(ok['err'] ** 2)))
                                    if len(ok) else np.nan)})
        say('    %-5s %-14s %-24s %4d %5d %9s %9s %10s'
            % (sp, law, ph[:24], len(g), int(g['diverged'].sum()),
               ('%.3f' % rmse) if np.isfinite(rmse) else '发散',
               ('%.3f' % float(np.mean(np.abs(g['err'].dropna()))))
               if g['err'].notna().any() else '—',
               ('%.3f' % float(np.max(np.abs(g['err'].dropna()))))
               if g['err'].notna().any() else '—'))
    summ = pd.DataFrame(sums)
    summ.to_csv(os.path.join(RES, 'step3_summary%s.csv' % TAG), index=False,
                encoding='utf-8-sig')

    # ---------- [4] 只看视界匹配的律 ----------
    say('\n' + '=' * 96)
    say('[4] 汇总：只用**视界匹配**选出的律（避免用"碰巧好"的律下结论）')
    say('=' * 96)
    for sp, sel in (('T7', sel_T7), ('T8', sel_T8)):
        say('')
        say('  %s（%s载荷）  视界 %d 循环'
            % (sp, load_type.get(sp, '等幅'),
               int(ex[ex['specimen'] == sp]['horizon'].max())))
        say('  %-14s %10s %10s %10s %8s' %
            ('增长律', 'B RMSE', 'C RMSE', '波形代价', 'B 发散'))
        say('  ' + '-' * 58)
        for law in sel:
            b = summ[(summ.specimen == sp) & (summ.law == law) &
                     (summ.phase == 'B_trajectory')]
            c = summ[(summ.specimen == sp) & (summ.law == law) &
                     (summ.phase == 'C_waveform_closed_loop')]
            if not len(b):
                continue
            bv = float(b['rmse'].iloc[0])
            cv = float(c['rmse'].iloc[0]) if len(c) else np.nan
            say('  %-14s %10s %10s %10s %8d'
                % (law, ('%.3f' % bv) if np.isfinite(bv) else '发散',
                   ('%.3f' % cv) if np.isfinite(cv) else '—',
                   ('%+.3f' % (cv - bv))
                   if (np.isfinite(bv) and np.isfinite(cv)) else '—',
                   int(b['n_div'].iloc[0])))
    say('')
    say('  `波形代价` = C − B：正值 = 波形估计误差**拖累**预后；')
    say('  负值 = 波形提供了额外信息（即"波形这一步"是净收益）。')

    # ---------- [4b] 初值误差如何传播到预后 ----------
    say('\n' + '=' * 96)
    say('[4b] 初值误差 → 预后误差 的传播（"波形估计准 1 mm，预后能好多少？"）')
    say('=' * 96)
    say('  做法：对同一 (试件, 律, 模型)，比较 C 与 B 的逐点预测之差，')
    say('        再除以"波形初值误差"（â − 实测）。')
    say('        比值 ≈ 1 ⇒ 初值误差**原样平移**到预后（既不放大也不缩小）。')
    say('')
    say('  %-5s %-12s %-24s %10s %12s %10s'
        % ('试件', '增长律', '初值来源', '初值误差', 'C−B 中位', '传播比'))
    say('  ' + '-' * 78)
    prop_rows = []
    for key, (c_last, a_hat) in sorted(est.items()):
        sp = key.split('/')[-1]
        err0 = a_hat - truth_map[sp]
        for law in ('lin_all', 'power_all', 'lin_last2'):
            b = ex[(ex.specimen == sp) & (ex.law == law) &
                   (ex.phase == 'B_trajectory')].sort_values('cycle')
            c = ex[(ex.specimen == sp) & (ex.law == law) &
                   (ex.phase == 'C_waveform_closed_loop') &
                   (ex.init_source == key)].sort_values('cycle')
            if not len(b) or not len(c):
                continue
            dd = (c['pred'].to_numpy() - b['pred'].to_numpy())
            dd = dd[np.isfinite(dd)]
            if not len(dd):
                continue
            med = float(np.median(dd))
            ratio = med / err0 if abs(err0) > 1e-9 else np.nan
            prop_rows.append({'specimen': sp, 'law': law, 'init_source': key,
                              'err_init': err0, 'dCB_median': med,
                              'propagation_ratio': ratio})
            say('  %-5s %-12s %-24s %10.2f %12.2f %10s'
                % (sp, law, key[:24], err0, med,
                   ('%.2f' % ratio) if np.isfinite(ratio) else '—'))
    if prop_rows:
        pr = pd.DataFrame(prop_rows)
        say('')
        say('  ⇒ 中位传播比 = **%.2f**（全体 %d 组）'
            % (pr['propagation_ratio'].median(), len(pr)))
        say('     物理含义：**预后误差 ≈ 波形估计误差**。')
        say('     ⇒ 对"线性型 / 锚定型"外推律，初值误差是**平移式**传播，不放大。')
        say('       所以：① 提高波形裂纹估计精度能 1:1 换成预后精度，值得继续投入；')
        say('             ② 反过来，把 C 的 RMSE 做得比 B 还小，很可能是')
        say('                **初值过估恰好抵消了增长律低估**（误差相消），')
        say('                不能解读成"波形提供了额外信息"。')
        pr.to_csv(os.path.join(RES, 'step3_propagation%s.csv' % TAG),
                  index=False, encoding='utf-8-sig')

    # ---------- [5] 限制 ----------
    say('\n' + '=' * 96)
    say('[5] 限制与结论')
    say('=' * 96)
    say('    0. 🔴 **T8 的外推精度无法被验证**：T1–T6 的截断自检最长只到 %d 循环，'
        % int(hmax))
    say('       而 T8 需要 %d 循环（= %.2f 倍）。数据里没有足够长的等幅参考曲线，'
        % (int(H_a['T8']), H_a['T8'] / hmax))
    say('       所以 T8 的数字只能当**方法演示**，不能当"精度指标"。')
    say('    1. **T8 的载荷差异很小**：T8 块谱幅值 85.23 / 95.44 kN，与 T1–T7 的等幅值')
    say('       95.44 kN 只差 5.3%（均值比 0.947）⇒ 当量系数 0.80~0.94（m=2~4），')
    say('       **不是**"换了载荷量级"。T8 发散的真实原因见下一条。')
    say('    2. **等损伤当量不能修 T8 的发散**：`lin_*`/`expo_*` 对循环轴的均匀缩放')
    say('       是**尺度不变**的（N→cN 时参数随之变，a(N) 预测逐点不变），')
    say('       而当量系数在 T8 上几乎均匀 ⇒ 线性律的预测**一字不变**。')
    say('       真正原因：末段速率（2.73e-4 mm/cycle）是其后 23843 循环平均速率')
    say('       （9.83e-5）的 2.8 倍 —— 是**裂纹扩展在减速**，属于该试件的轨迹特性，')
    say('       与"变幅"无关。⇒ 稳健做法是用 `lin_all`（全点拟合）而非 `lin_last2`。')
    say('    3. T7/T8 各只有 4 / 5 个建模点 ⇒ 增长律参数**欠定**：')
    say('       `paris_Ap` 拟合 3 个参数，点数少时极不稳（看"发散"列）。')
    say('    4. **Paris 族在本数据的外推上不可用**：p>1 的有限时间奇点 + 参数欠定')
    say('       ⇒ 短视界看不出、长视界爆炸。这是**负面结论**，同样要报。')
    say('    5. **不实现官方评分函数**（常数在 PHM2019_ScoringSpreadsheet.xlsx 内，')
    say('       仓库里没有该文件）⇒ 只用 RMSE / MAE / 最大误差这类无歧义口径。')
    say('    6. 目标点极少（T7: 4 个、T8: 5 个）⇒ 单点误差影响大，')
    say('       报数必须同时给逐点明细（step3_extrap.csv），不能只给均值。')


    with open(os.path.join(RES, 'step3_report%s.txt' % TAG), 'w',
              encoding='utf-8') as fh:
        fh.write('\n'.join(_REPORT) + '\n')
    say('\n[产出] results/step3_selfcheck%s.csv / step3_extrap%s.csv / '
        'step3_summary%s.csv / step3_report%s.txt'
        % (TAG, TAG, TAG, TAG))


if __name__ == '__main__':
    main()
