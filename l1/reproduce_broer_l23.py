# -*- coding: utf-8 -*-
"""复现 Broer et al. 2021 (Struct Health Monit 20(2)) **Level 2 + Level 3**。

`reproduce_broer_l1.py` 只做了 Level 1（检测）与 Level 4（严重度）。本模块补齐：

Level 3 —— 损伤类型识别（论文 §"Level 3: damage type identification"）
  3a 刚度退化 (Stiffness degradation)
     论文: "its effect is most apparent at the compressed stiffener foot.
            Consequently, only the strain measurements from the more compressed
            stiffener foot are employed"
           "windowing approach ... window size of five measurements"
           "|m2 - m1| > 2s, 其中 s = 无偏窗方差的算术平均的平方根"
           "an additional constraint: identified for five consecutive measurement intervals"
     实现: 逐 500cycle 测量点算沿脚均值 mu → 滑窗(5) → 相邻窗均值差 > 2s → 需连续 5 次
     产出: 向量 n_SD

  3b 脱粘识别 (Disbond identification)
     论文: "the change in strain value in time is evaluated for each measurement
            location along the stiffener foot" … "A change is detected if the
            absolute change in strain value is larger than 2s" …
            "for the compressed stiffener, an increase in absolute value is required,
             while for the tensed stiffener both a decrease and an increase in
             absolute value are needed"（后者即 valley-peak-valley, Fig 5(d)）
            "five consecutive windows need to be flagged (i.e. a period of 2000 cycles)"
            "at least five neighboring measurement locations need to show this behavior"
     实现: 逐空间位置的应变序列 → 滑窗(5) → |Δ| > 2s_x + 符号约束
           → 空间需 ≥N_NEIGH 个相邻位置同时满足 → 时间需连续 N_CONSEC 窗
     产出: 向量 n_DB（分左右脚各一条）

Level 2 —— 损伤定位（论文 §"Level 2: damage localization"）
  应变支路: "we can estimate the location of the local peak/valley in the strain
             distribution for these cycles"（即在 n_DB 的 cycle 上找局部极值位置）
     实现: 对 n_DB 的 cycle 取 Δε = ε[k,:] − ε[基线]，减去平滑趋势得局部特征 δ，
           取 max|δ| 的位置作为脱粘沿脚位置估计
  AE 支路:  4 传感器平行四边形 + Geiger 平面定位（见 ae_locate.py，后续接入）

数据事实（本工作区实测，2026-09-17）
  - DFOS 行严格 `谷(卸载) → 峰(最大载荷)` 交替，每块 18 行 = 9 对测量，
    块跨 ~3100 s ≈ 5000 cycles → **测量间隔 ≈ 555 cycles**（即论文的 measuring
    cycle interval；论文 Level3b 称"5 窗 = 2000 cycles"⇒ 其间隔 400 cycles）
  - 故本模块**不做 500cycle 分箱**（Level1/4 因需与 AE 融合才分箱），直接以
    "测量序列" 为时间轴，避免空箱前值填充污染方差估计
  - 坏行（|min| ≈ 16000 的伪帧）由 `min(脚L,脚R) < -1000 且 max(...) < -200` 判据排除

验收靶子（论文 Results 段，已收录于 l1_meta.REFS）
  L1-03: n_SD 69,000 起 ; n_DB 143,000(右脚)/146,000(左脚)
  L1-04: n_SD 151,000-194,500 与 229,500+（前 30,000 为论文自认误报）
         n_DB 239,500(右脚)/272,000(左脚)
  L1-05: n_SD 68,000-82,000 与 98,000-123,000
         n_DB 66,500-69,500(左脚上段)/110,000(右脚)/129,000

用法
  python l1/reproduce_broer_l23.py --mode l3              # Level3 应变支路
  python l1/reproduce_broer_l23.py --mode l2              # Level2 应变定位
  python l1/reproduce_broer_l23.py --mode plot            # 复现论文 Figure 12
  python l1/reproduce_broer_l23.py --mode all             # 全部 + 图
  python l1/reproduce_broer_l23.py --sweep                # 判据敏感度扫描
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:                                    # Windows 控制台 GBK → µε/℃ 会崩
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                                  errors='replace', line_buffering=True)
except Exception:                       # noqa: BLE001
    pass

from l1_meta import load_meta                                   # noqa: E402
from reproduce_broer_l1 import (load_dfos, time_to_cycle,       # noqa: E402
                                compressed_is_R, META)

RES = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09']

# --- 论文判据默认参数 -------------------------------------------------------
W_MEAS = 5          # 窗长 = 5 个测量点（论文: "window size of five measurements"）
K_SIGMA = 2.0       # 阈值倍数（论文: 2s）
N_CONSEC = 5        # 需连续多少个窗/区间被标记（论文: 5）
N_NEIGH = 5         # 脱粘需多少个相邻测量位置同时满足（论文: 5）
RES_MM = 5.0        # 空间重采样步长(mm)

# --- 论文报告的实测结果（Results 段；用于对照，非参与计算）---------------------
PAPER = {
    'L1-03': dict(n_SD=[69000], n_SD_txt='69,000 起',
                  n_DB={'右脚': [143000], '左脚': [146000]},
                  loc_txt='左脚上半段; 右脚下半段靠中心'),
    'L1-04': dict(n_SD=[151000, 194500, 229500], n_SD_txt='151k-194.5k, 229.5k+',
                  n_DB={'右脚': [239500], '左脚': [272000]},
                  loc_txt='右脚(论文自述应变定位不可靠)'),
    'L1-05': dict(n_SD=[68000, 82000, 98000, 123000],
                  n_SD_txt='68k-82k, 98k-123k',
                  n_DB={'左脚': [66500, 69500], '右脚': [110000, 129000]},
                  loc_txt='左脚上段; 右脚下移'),
}


# ============================================================ 数据
def _block_id(t, gap_s=300.0):
    """按时间间隔切块（块 = 一个准静态测量段）。返回每行的块号。"""
    dt = np.diff(t)
    bnd = np.concatenate([[0], np.where(dt > gap_s)[0] + 1])
    return np.searchsorted(bnd, np.arange(len(t)), side='right') - 1


def _foot_block_mask(s, span_min=400.0, min_rows=3):
    """单脚在块内的最大载荷掩码（返回该块的布尔数组）。

    块内物理事实：最大载荷行形成"最受压平台"，最小载荷行（谷）浅得多，
    中间载荷行（准静态斜坡）介于两者之间。故按块取相对阈值：
        thr = (p10 + p90) / 2
    分位数用于抗伪帧（伪帧可能使 min/max 失真）。若块内跨度 < span_min
    （整块同一载荷水平，无加载斜坡），退化为绝对判据 s < -200。
    """
    n = len(s)
    if n < min_rows:
        return np.zeros(n, bool)
    lo, hi = np.percentile(s, 10), np.percentile(s, 90)
    if hi - lo <= span_min:
        return s < -200.0
    thr = 0.5 * (lo + hi)
    return (s < thr) & (s < -200.0)


def _peak_rows(rL, rR, t, gap_s=300.0):
    """挑出"最大载荷行"——**逐脚、按块（准静态测量段）相对判定**。

    ⚠️ 不能用绝对阈值（如 min<-1000 且 max<-200）：准静态段是一条**加载斜坡**，
    中间载荷行（如左脚 -1468 / 右脚 -685）同样满足绝对阈值 → 会混入序列并制造
    假跳变（L1-03 实测在 5.7k 处出现 ±1900 µε 假跳）→ `s`（窗方差均值）被污染
    虚高数十倍 → 论文的 2s 判据永不触发。

    两脚**分别**判据后取与：任意一脚掉线（如 L1-09 右脚整帧 −91 而左脚正常）
    会被另一脚挡下。
    """
    bid = _block_id(t, gap_s)
    keep = np.zeros(len(t), bool)
    for b in np.unique(bid):
        sel = np.where(bid == b)[0]
        a, z = sel[0], sel[-1] + 1
        keep[a:z] = (_foot_block_mask(rL[a:z]) & _foot_block_mask(rR[a:z]))
    return keep


def _median_filter3(P):
    """沿空间 x 做 3 点中值滤波，去掉孤立的空间坏点。

    为什么安全：重采样到 5 mm 后，真实特征（论文 Fig 5(d) 的 valley-peak-valley
    "grows in both size and width"）宽度 ≥ 25 mm = ≥5 个格点；而实测存在跨 8 个
    原始位置（≈5 mm）的孤立空间坏点，会产生 |δ| ~ 10⁴ µε 的伪特征。3 点中值
    对 ≥5 格的真实特征无损，对 1~2 格伪点必除。
    """
    if P.shape[1] < 3:
        return P
    Q = np.empty_like(P)
    Q[:, 0] = P[:, 0]
    Q[:, -1] = P[:, -1]
    Q[:, 1:-1] = np.median(np.stack([P[:, :-2], P[:, 1:-1], P[:, 2:]]), axis=0)
    return Q


def _despike(P, k=4.0, floor=50.0, half=3):
    """滚动去尖峰：剔除"单帧偏离、下一帧立刻恢复"的孤立坏帧。

    论文: "The mean values are stationary if no damage occurs" —— 最大载荷状态在
    相邻测量之间应连续。实测坏帧表现为孤立跳变（L1-04 ±757 µε、L1-05 ±2166 µε、
    L1-09 右脚整帧掉到 −91 而左脚正常=单脚掉线），且**绝对阈值与块内中位数法都抓不住**
    （块内只剩 1–2 行时中位数法失效）。

    做法(在按 cycle 排序后的序列上): 以邻域 ±half 行的**中位剖面**为参考，
    行级稳健偏差 dev = median_x|P_i − ref|；阈值 max(k·1.4826·MAD(dev), floor)。
    真实损伤演化在邻域内连续 → 中位剖面跟随 → 不会被误杀；孤立坏帧必被剔除。
    """
    n = P.shape[0]
    if n < 2 * half + 1:
        return np.ones(n, bool)
    dev = np.empty(n)
    for i in range(n):
        a, b = max(0, i - half), min(n, i + half + 1)
        ref = np.median(P[a:b], axis=0)
        dev[i] = float(np.median(np.abs(P[i] - ref)))
    thr = max(k * 1.4826 * float(np.median(dev)), floor)
    return dev <= thr


def resample_space(pos, M, res_mm=RES_MM, min_valid=0.2):
    """把 (n_row, n_pos) 按 ~res_mm 空间重采样，并处理 NaN。

    为什么必须重采样: 原始位置步长 0.653 mm，若直接按"≥5 个相邻位置"判定，
    5 个点仅 3.3 mm —— 噪声即可满足，判据失效。论文的 SMARTape 空间分辨率
    远粗于此。重采样到 5 mm 后，"5 个相邻位置" = 25 mm，与论文
    "bump grows in both size and width" 的空间尺度同量级。

    NaN 处理: 实测部分位置列**整列无效**（光纤脱粘/量程外），`nanmean` 对
    全 NaN 切片返回 NaN 并告警 → 会污染 `s`。故先剔除有效率 < min_valid 的
    重采样列，再逐行沿 x 线性插值补缺。
    """
    dp = float(np.median(np.diff(pos)))
    if not np.isfinite(dp) or dp <= 0:
        raise ValueError(f'空间步长异常: {dp}')
    step = max(1, int(round(res_mm / dp)))
    nb = M.shape[1] // step
    if nb < 2 * N_NEIGH:
        raise ValueError(f'空间点太少: {M.shape[1]} 列 step {step} -> {nb}')
    xc = np.empty(nb)
    P = np.empty((M.shape[0], nb))
    for j in range(nb):
        sl = slice(j * step, (j + 1) * step)
        blk = M[:, sl]
        xc[j] = np.nanmean(pos[sl])
        with np.errstate(invalid='ignore'):
            cnt = np.isfinite(blk).sum(axis=1)
            tot = np.where(cnt > 0, cnt, 1)
            P[:, j] = np.nansum(np.where(np.isfinite(blk), blk, 0.0),
                                axis=1) / tot
    # 剔除整列基本无效者
    good = np.isfinite(P).mean(axis=0) >= min_valid
    if good.sum() < 2 * N_NEIGH:
        raise ValueError(f'空间列几乎全 NaN: {good.sum()}')
    xc, P = xc[good], P[:, good]
    # 逐行沿 x 插值补缺
    for i in range(P.shape[0]):
        r = P[i]
        bad = ~np.isfinite(r)
        if bad.any() and (~bad).sum() >= 2:
            r[bad] = np.interp(xc[bad], xc[~bad], r[~bad])
    return xc, P


def measure_sequence(gid, res_mm=RES_MM):
    """取出"准静态测量序列"——每个测量段一行（最大载荷行），沿左右脚重采样后的剖面。

    返回 dict(cyc, xL, PL, xR, PR, c_is_R, n_meas)
      cyc  : (n_meas,) 每个测量点的疲劳循环数（FBG 锚校准，同 Level1/4 口径）
      PL   : (n_meas, nL) 左脚剖面（µε，压缩为负）
    """
    m = META[gid]
    t, pos, M = load_dfos(gid)
    fL, fR = m['foot_L'], m['foot_R']
    mL = (pos >= fL[0]) & (pos <= fL[1])
    mR = (pos >= fR[0]) & (pos <= fR[1])
    rL = np.nanmean(M[:, mL], axis=1)
    rR = np.nanmean(M[:, mR], axis=1)
    keep = _peak_rows(rL, rR, t)
    if keep.sum() < 20 * W_MEAS:
        raise RuntimeError(f'{gid}: 有效最大载荷行仅 {keep.sum()} 个')
    idx = np.where(keep)[0]
    xL, PL = resample_space(pos[mL], M[np.ix_(idx, np.where(mL)[0])], res_mm)
    xR, PR = resample_space(pos[mR], M[np.ix_(idx, np.where(mR)[0])], res_mm)
    # 双脚剖面均须有效，行才保留
    ok = np.isfinite(PL).all(axis=1) & np.isfinite(PR).all(axis=1)
    idx, PL, PR = idx[ok], PL[ok], PR[ok]
    # 按 cycle 排序后去尖峰（孤立坏帧/单脚掉线）
    cyc = time_to_cycle(gid, t[idx])
    o = np.argsort(cyc, kind='stable')
    idx, cyc, PL, PR = idx[o], cyc[o], PL[o], PR[o]
    ok = _despike(PL) & _despike(PR)
    if ok.sum() < 20 * W_MEAS:
        raise RuntimeError(f'{gid}: 去尖峰后仅 {ok.sum()} 行')
    idx, cyc, PL, PR = idx[ok], cyc[ok], PL[ok], PR[ok]
    # 空间 3 点中值滤波（去孤立空间坏点，对 ≥5 格的真实特征无损）
    PL, PR = _median_filter3(PL), _median_filter3(PR)
    cR = compressed_is_R(PL.mean(axis=1), PR.mean(axis=1))
    return dict(cyc=cyc, xL=xL, PL=PL, xR=xR, PR=PR, c_is_R=cR,
                n_meas=len(cyc), t0=float(cyc[0]), t1=float(cyc[-1]))



# ============================================================ 判据内核
def window_stats(seq, W=W_MEAS):
    """滑窗统计 → (d, s)。

    seq : (n_time, n_loc) 应变序列
    d   : (n_win-1, n_loc) 相邻窗均值差  m2 - m1
    s   : (n_loc,)        sqrt(算术平均的无偏窗方差)  —— 论文的 s

    统一用于 3a（n_loc=1，对沿脚均值）与 3b（n_loc=沿脚各位置）。
    """
    n = seq.shape[0]
    nw = n - W + 1
    if nw < 2:
        raise RuntimeError(f'序列太短: n={n}, W={W}')
    cs = np.concatenate([np.zeros((1, seq.shape[1])),
                         np.cumsum(seq, axis=0)], axis=0)
    cs2 = np.concatenate([np.zeros((1, seq.shape[1])),
                          np.cumsum(seq ** 2, axis=0)], axis=0)
    wsum = cs[W:] - cs[:-W]                     # (nw, n_loc)
    wss = cs2[W:] - cs2[:-W]
    m = wsum / W
    # 无偏方差: (Σx² − n·m̄²)/(n−1)
    var = (wss - W * m ** 2) / (W - 1)
    var = np.maximum(var, 0.0)
    s = np.sqrt(np.mean(var, axis=0))           # (n_loc,)
    d = m[1:] - m[:-1]                          # (nw-1, n_loc)
    return d, s


def _runs(mask):
    """布尔数组的连续 True 段 → [(start, length), ...]"""
    out = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j - i + 1))
            i = j + 1
        else:
            i += 1
    return out


def _consec_cycles(cyc, flag, n_consec, offset):
    """返回满足"连续 n_consec 个 True"的 cycle 列表（取每段的首个 cycle）。"""
    hits = []
    for st, ln in _runs(flag):
        if ln >= n_consec:
            for k in range(st, st + ln):
                hits.append(float(cyc[min(k + offset, len(cyc) - 1)]))
    return hits


def detect_stiffness(cyc, mu, W=W_MEAS, k=K_SIGMA, n_consec=N_CONSEC):
    """Level 3a 刚度退化：论文式 |m2 − m1| > 2s + 连续 n_consec 个区间。

    mu : (n_time,) 更受压脚的沿脚均值应变（µε，压缩为负）
    返回 (n_SD_cycles, info)
    """
    d, s = window_stats(mu.reshape(-1, 1), W)
    d = d[:, 0]
    thr = k * s[0]
    flag = np.abs(d) > thr
    # d[i] 比较窗 i 与 i+1，边界落在测量点 i+W
    n_SD = _consec_cycles(cyc, flag, n_consec, offset=W)
    return n_SD, dict(thr=float(thr), s=float(s[0]),
                      frac_flagged=float(flag.mean()))


def detect_disbond(cyc, P, x, rule, W=W_MEAS, k=K_SIGMA,
                   n_consec=N_CONSEC, n_neigh=N_NEIGH):
    """Level 3b 脱粘识别：逐空间位置 |Δ| > 2s_x + 符号约束 + 空间邻域约束。

    P    : (n_time, n_loc) 该脚沿脚剖面序列（压缩为负）
    rule : 符号约束口径
      'neg'     —— 论文"压缩脚"规则: "an increase in absolute value is required"
                   ⇒ |ε| 增大 ⇒ 应变更负 ⇒ d < 0
      'pos_neg' —— 论文"受拉脚"规则: "both a decrease and an increase ... are
                   needed" ⇒ 邻域内既有 d<0 又有 d>0（谷-峰-谷, Fig 5(d)）
      'any'     —— 只用幅度判据，不做符号约束（诊断用）
    返回 (n_DB_cycles, loc_per_cycle, info)
    """
    d, s = window_stats(P, W)                   # (npair, n_loc), (n_loc,)
    thr = k * s
    flag = np.abs(d) > thr[None, :]
    npair, nloc = d.shape
    loc_ok = np.zeros(npair, bool)
    loc_at = {}
    for i in range(npair):
        run_best = None
        for st, ln in _runs(flag[i]):
            if ln < n_neigh:
                continue
            seg = d[i, st:st + ln]
            if rule == 'neg':
                if not np.all(seg < 0):
                    continue
            elif rule == 'pos_neg':
                if not (seg < 0).any() or not (seg > 0).any():
                    continue
            # 取 |Δ| 最大者作为该窗对的定位
            j = st + int(np.argmax(np.abs(seg)))
            run_best = (float(x[j]), float(d[i, j]), ln)
            break
        if run_best is not None:
            loc_ok[i] = True
            loc_at[i] = run_best
    n_DB = _consec_cycles(cyc, loc_ok, n_consec, offset=W)
    # 与 n_DB 对应的定位（去重：同一连续段取首个）
    locs = []
    for st, ln in _runs(loc_ok):
        if ln >= n_consec:
            locs.append(loc_at[st])
    return n_DB, locs, dict(thr_med=float(np.median(thr)),
                            frac_loc_ok=float(loc_ok.mean()))


def _rule_of(side, side_mode):
    """把"脚的角色"映射到符号判据口径。"""
    if side_mode == 'paper':
        return 'neg' if side == 'compressed' else 'pos_neg'
    if side_mode == 'swap':
        return 'pos_neg' if side == 'compressed' else 'neg'
    return 'any'


# ============================================================ Level 2 定位
LOC_SMOOTH_FRAC = 0.15      # 平滑窗 = 沿脚长度的比例（用于分离局部特征与整体重构形）


def local_feature_series(P, x, smooth_frac=LOC_SMOOTH_FRAC):
    """逐 cycle 的局部形状特征与定位。

    论文（Level 2 应变支路）: "we can estimate the location of the local peak/valley
    in the strain distribution for these cycles"（"these cycles" = n_DB）。

    ⚠️ 本工作区实测 n_DB 为空（见模块 docstring 与 --sweep），故本函数**不被 n_DB
    驱动**，而是对全寿命逐 cycle 给出定位，供诊断与后续使用。这是论文
    "replace 定位"思路的实现，**不是论文原判据**。

    做法: Δε = ε[k] − ε[首测]（论文 Fig12 口径: 相对 500cycle 曲线的变化）
          → 减去滑窗平滑趋势（刚度退化引起的整体重构形）得局部特征 δ
          → 取 max|δ| 的位置为脱粘沿脚位置
    返回 (dE, delta, loc_mm, amp)
    """
    dE = P - P[0]
    w = max(3, (int(round(len(x) * smooth_frac)) | 1))
    pad = w // 2
    ker = np.ones(w) / w
    trend = np.empty_like(dE)
    for i in range(dE.shape[0]):
        trend[i] = np.convolve(np.pad(dE[i], pad, mode='edge'), ker,
                               mode='valid')
    delta = dE - trend
    # 边缘效应：滑窗在两端用边缘填充 → 趋势有偏 → δ 在首/末 pad 个点不可信，置 0
    if pad > 0:
        delta[:pad] = 0.0
        delta[-pad:] = 0.0
    j = np.argmax(np.abs(delta), axis=1)
    return dE, delta, x[j], np.abs(delta[np.arange(len(j)), j])


# ============================================================ 主流程
def run(gids, W=W_MEAS, k=K_SIGMA, n_consec=N_CONSEC, n_neigh=N_NEIGH,
        res_mm=RES_MM, side_mode='paper', verbose=True):
    rows, store = [], {}
    for gid in gids:
        try:
            D = measure_sequence(gid, res_mm)
        except Exception as e:                                  # noqa: BLE001
            print(f'[跳过] {gid}: {e}')
            continue
        cyc = D['cyc']
        # 更受压脚 = 论文的 stiffener 1（论文: 刚度退化只在它上面明显）
        if D['c_is_R']:
            xc, Pc, xt, Pt, cname, tname = (D['xR'], D['PR'], D['xL'], D['PL'],
                                            '右脚', '左脚')
        else:
            xc, Pc, xt, Pt, cname, tname = (D['xL'], D['PL'], D['xR'], D['PR'],
                                            '左脚', '右脚')
        mu = Pc.mean(axis=1)
        n_SD, sd_info = detect_stiffness(cyc, mu, W, k, n_consec)
        per = {}
        for nm, PP, xx, side in ((cname, Pc, xc, 'compressed'),
                                 (tname, Pt, xt, 'tensed')):
            n_DB, locs, info = detect_disbond(cyc, PP, xx,
                                              _rule_of(side, side_mode),
                                              W, k, n_consec, n_neigh)
            # Level 2 应变定位（全寿命逐 cycle；不被 n_DB 驱动，见 docstring）
            _, _dl, loc_mm, amp = local_feature_series(PP, xx)
            thr_loc = 3.0 * float(np.median(amp[:max(5, len(amp) // 20)]))
            per[nm] = dict(n_DB=n_DB, locs=locs, info=info,
                           loc_mm=loc_mm, amp=amp, thr_loc=thr_loc)
        store[gid] = dict(D=D, cyc=cyc, cname=cname, tname=tname,
                          xc=xc, Pc=Pc, xt=xt, Pt=Pt, mu=mu,
                          n_SD=n_SD, sd_info=sd_info, per=per)
        if verbose:
            _report(gid, store[gid])
    return store


def _fmt(cs, n=6):
    if not cs:
        return '—'
    cs = sorted(cs)
    if len(cs) <= n:
        return ' '.join(f'{c/1000:.1f}k' for c in cs)
    return f'{cs[0]/1000:.1f}k … {cs[-1]/1000:.1f}k ({len(cs)})'


def _report(gid, S):
    paper = PAPER.get(gid)
    print('=' * 76)
    print(f'{gid}  测量点 {S["D"]["n_meas"]} 个, cycle {S["D"]["t0"]/1000:.1f}k'
          f'→{S["D"]["t1"]/1000:.1f}k  更受压脚 = {S["cname"]}')
    print(f'  剖面: 左脚 {S["D"]["PL"].shape[1]} 点 @{np.median(np.diff(S["D"]["xL"])):.1f}mm'
          f' / 右脚 {S["D"]["PR"].shape[1]} 点')
    print(f'  [L3a 刚度退化] 阈值 2s = {S["sd_info"]["thr"]:.1f} µε, '
          f'标记窗占比 {S["sd_info"]["frac_flagged"]:.3f}')
    print(f'      n_SD = {_fmt(S["n_SD"])}')
    if paper:
        print(f'      论文: {paper["n_SD_txt"]}')
    for nm in (S['cname'], S['tname']):
        p = S['per'][nm]
        side = '受压' if nm == S['cname'] else '受拉'
        print(f'  [L3b 脱粘/{nm}({side})] 满足空间约束的窗对占比 '
              f'{p["info"]["frac_loc_ok"]:.3f}, 阈值中位 {p["info"]["thr_med"]:.0f} µε')
        print(f'      n_DB = {_fmt(p["n_DB"])}')
        for pos, dlt, ln in p['locs'][:4]:
            print(f'        定位 {pos:.0f} mm (Δ={dlt:+.0f} µε, 邻域宽 {ln})')
        # Level 2 定位（替代判据）
        sel = p['amp'] > p['thr_loc']
        if sel.any():
            first = int(np.argmax(sel))
            cyc = S['cyc']
            print(f'  [L2 定位/{nm}] 替代判据 |δ|>{p["thr_loc"]:.0f} µε: '
                  f'{int(sel.sum())}/{len(sel)} 个 cycle; '
                  f'首次 {cyc[first]/1000:.1f}k 定位于 {p["loc_mm"][first]:.0f} mm; '
                  f'末态 {p["loc_mm"][-1]:.0f} mm'
                  f' (|δ|max={p["amp"].max():.0f} µε)')
        else:
            print(f'  [L2 定位/{nm}] 无 (|δ|max={p["amp"].max():.0f} '
                  f'< 阈 {p["thr_loc"]:.0f} µε)')
        if paper and nm in paper['n_DB']:
            print(f'      论文: {_fmt(paper["n_DB"][nm])}')
            print(f'      论文位置: {paper["loc_txt"]}')


def plot(store, W=W_MEAS, n_consec=N_CONSEC, n_neigh=N_NEIGH):
    """复现论文 Figure 12（Δε 色图 + n_SD/n_DB）并附局部形状特征/定位曲线。

    论文 Fig12 原文: "Colorplots indicating the change in strain with respect to the
    strain curve measured at 500 cycles for each SSC. The x-axis indicates the number
    of cycles, y-axis indicates the location along the stiffener foot, and the color
    indicates the change in strain. In addition, the identified location of the
    disbond is shown by the black scatter points. Below the colorplots, the cycle
    numbers at which stiffness degradation or a disbond is identified are presented
    (n_SD and n_DB)."
    """
    for gid, S in store.items():
        cyc = S['cyc'] / 1000.0
        fig, axs = plt.subplots(4, 2, figsize=(14.5, 12.5),
                                gridspec_kw=dict(height_ratios=[3, 2, 2, 0.9]))
        order = [(S['cname'], S['xc'], S['Pc'], 'compressed'),
                 (S['tname'], S['xt'], S['Pt'], 'tensed')]
        for j, (nm, x, P, side) in enumerate(order):
            dE, delta, loc_mm, amp = local_feature_series(P, x)
            v = float(np.nanpercentile(np.abs(dE), 99)) or 1.0
            a = axs[0, j]
            m = a.pcolormesh(cyc, x, dE.T, cmap='RdBu_r', vmin=-v, vmax=v,
                             shading='nearest')
            plt.colorbar(m, ax=a, label='Δε (µε)')
            lab = '压缩脚(stiffener1)' if side == 'compressed' else '拉伸脚(stiffener2)'
            a.set_title(f'{gid} {nm}（{lab}）Δε 沿脚分布  [论文 Fig.12 口径]')
            a.set_ylabel('沿脚位置 (mm)')
            # 论文报告的 n_SD / n_DB（绿虚线）
            if gid in PAPER:
                for c in PAPER[gid]['n_DB'].get(nm, []):
                    a.axvline(c / 1000, color='lime', ls='--', lw=1.2,
                              label='论文 n_DB')
                for c in PAPER[gid]['n_SD']:
                    a.axvline(c / 1000, color='cyan', ls=':', lw=1.0)
            # 我们检出的 n_DB / n_SD（黑点线）
            for c in S['per'][nm]['n_DB']:
                a.axvline(c / 1000, color='k', ls=':', lw=0.8)
            for c in S['n_SD']:
                a.axvline(c / 1000, color='tab:orange', ls='-', lw=0.8, alpha=.7)
            hand = a.get_legend_handles_labels()
            if hand[0]:
                a.legend(fontsize=7, loc='upper left')

            b = axs[1, j]
            b.plot(cyc, amp, 'k-', lw=1.2)
            b.axhline(0, color='grey', lw=0.5)
            thr_loc = 3.0 * float(np.median(amp[:max(5, len(amp) // 20)]))
            b.axhline(thr_loc, color='r', ls='--', lw=0.9,
                      label=f'3×基线中位 = {thr_loc:.0f} µε')
            b.set_ylabel('|δ| 局部形状变化 (µε)')
            b.set_title(f'{nm} 局部特征幅值（替代判据，非论文原文）')
            b.grid(alpha=.3)
            b.legend(fontsize=7)

            c = axs[2, j]
            sel = amp > thr_loc
            c.scatter(cyc[sel], loc_mm[sel], s=5, c='k')
            if gid in PAPER:
                for cc in PAPER[gid]['n_DB'].get(nm, []):
                    c.axvline(cc / 1000, color='lime', ls='--', lw=1.2)
                for cc in PAPER[gid]['n_SD']:
                    c.axvline(cc / 1000, color='cyan', ls=':', lw=1.0)
            c.set_ylim(float(np.min(x)), float(np.max(x)))
            c.set_ylabel('定位 (mm)')
            c.set_xlabel('cycle (k)')
            c.set_title(f'{nm} 脱粘定位轨迹（|δ|>{thr_loc:.0f} 时, '
                        f'{int(sel.sum())} 点）')
            c.grid(alpha=.3)

            d = axs[3, j]
            d.plot(cyc, np.full_like(cyc, 0.62), 'w.')
            if S['n_SD']:
                cs = np.array(S['n_SD'])
                d.plot(cs / 1000, np.full(len(cs), 0.62), 's',
                       color='tab:orange', ms=4, label='n_SD (刚度退化)')
            nd = S['per'][nm]['n_DB']
            if nd:
                cs = np.array(nd)
                d.plot(cs / 1000, np.full(len(cs), 0.18), 's',
                       color='tab:purple', ms=4, label='n_DB (脱粘)')
            if gid in PAPER:
                cs = PAPER[gid]['n_SD']
                d.plot(np.array(cs) / 1000, np.full(len(cs), 0.90), 'v',
                       color='cyan', ms=5, label='论文 n_SD')
                cs = PAPER[gid]['n_DB'].get(nm, [])
                if cs:
                    d.plot(np.array(cs) / 1000, np.full(len(cs), 0.18), '^',
                           color='lime', ms=5, label='论文 n_DB')
            d.set_ylim(-0.05, 1.05)
            d.set_yticks([])
            d.set_xlabel('cycle (k)')
            for vv in (0.18, 0.62, 0.90):
                d.axhline(vv, color='grey', lw=0.4, alpha=.5)
            hand = d.get_legend_handles_labels()
            if hand[0]:
                d.legend(fontsize=7, ncol=2, loc='center left')
            d.grid(axis='x', alpha=0.3)
        fig.suptitle(f'{gid}  Broer Level2/3 复现（论文 Fig.12 口径）  '
                     f'2s 判据, 窗长 {W}, 连续 {n_consec}, 邻域 {n_neigh}')
        fig.tight_layout()
        out = os.path.join(FIG, f'l1_broer_l23_{gid}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print('已存:', out)


def save_csv(store):
    rows = []
    for gid, S in store.items():
        for nm in (S['cname'], S['tname']):
            p = S['per'][nm]
            rows.append(dict(
                gid=gid, foot=nm,
                side='压缩脚(stiffener1)' if nm == S['cname'] else '拉伸脚(stiffener2)',
                n_meas=S['D']['n_meas'],
                n_SD=len(S['n_SD']),
                n_SD_first=round(min(S['n_SD'])) if S['n_SD'] else None,
                n_DB=len(p['n_DB']),
                n_DB_first=round(min(p['n_DB'])) if p['n_DB'] else None,
                n_DB_cycles=' '.join(str(round(c)) for c in p['n_DB'][:20]),
                loc_first_mm=round(p['locs'][0][0], 1) if p['locs'] else None,
                loc_first_dE=round(p['locs'][0][1], 0) if p['locs'] else None,
                loc_thr_uE=round(p['thr_loc'], 1),
                loc_amp_max=round(float(p['amp'].max()), 0),
                loc_n_flagged=int((p['amp'] > p['thr_loc']).sum()),
                loc_first_cycle=(round(float(S['cyc'][int(np.argmax(p['amp'] > p['thr_loc']))]))
                                     if (p['amp'] > p['thr_loc']).any() else None),
                loc_end_mm=round(float(p['loc_mm'][-1]), 1),
            ))
    out = os.path.join(RES, 'l1_broer_l23.csv')
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print('已存:', out)
    return out


def sweep(gids, ks=(1.0, 1.25, 1.5, 2.0, 2.5), n_neighs=(5, 3)):
    """判据敏感度扫描：阈值倍数 k 与空间邻域要求如何影响检出。

    为什么必须做：论文 3b 判据比较的是**窗口均值差**，而 s 由**窗口方差**定 ——
    `d` 的 std ≈ 0.63·s（均值抵消了测量点间高频噪声），故 `|d| > 2s` 实际要求
    ~3.2σ，理论概率 ~0.1%。所以 k 的取值直接决定"能否检出"，必须显式给出。
    """
    print('=' * 96)
    print('L3b 判据敏感度扫描（consec=%d 固定）' % N_CONSEC)
    print('%-6s %-4s %-5s | %-28s | %-28s'
          % ('gid', 'k', '邻域', '受压脚 n_DB(首/数)', '受拉脚 n_DB(首/数)'))
    for k in ks:
        for nn in n_neighs:
            store = run(gids, k=k, n_consec=N_CONSEC, n_neigh=nn, verbose=False)
            for gid in gids:
                if gid not in store:
                    continue
                S = store[gid]
                cells = []
                for nm in (S['cname'], S['tname']):
                    cs = S['per'][nm]['n_DB']
                    cells.append('—' if not cs
                                 else f'{min(cs)/1000:.1f}k / {len(cs)}')
                print('%-6s %-4s %-5d | %-28s | %-28s'
                      % (gid, k, nn, cells[0], cells[1]))
        print('-' * 96)
    # 逐位置 window 均值差 / 窗口方差 的统计（解释为何难触发）
    print('统计解释（window 均值差 d 与 2s 阈值）：')
    for gid in gids:
        D = measure_sequence(gid)
        cname = '右脚' if D['c_is_R'] else '左脚'
        Pc = D['PR'] if D['c_is_R'] else D['PL']
        d, s = window_stats(Pc)
        print(f'  {gid} {cname}(受压): |d| 中位 {np.median(np.abs(d)):.1f}, '
              f'std(d) {d.std():.1f}, s 中位 {np.median(s):.1f}, '
              f'std(d)/s = {d.std()/np.median(s):.2f}'
              f'  ⇒ |d|>2s 需 {2/(d.std()/np.median(s)):.1f}σ')



# ============================================================ AE 支路
# 论文 L2 AE 定位: "four AE sensors were placed on the skin of the SSC to form a
#   parallelogram" + Geiger 平面定位 → 复用 ae_locate.py 的 TDOA + 各向异性走时
# 论文 L3 AE 聚类: "five features ... amplitude A, rise time R, duration D,
#   energy E, and counts/duration CNTS/D" → PCA(2) → k-means(2)
#
# 单位（`.pridb` 的 `ae_fieldinfo` 表，L1-03 实查）:
#   Time [s] factor 1e-7 · Amp [µV] 原始 · RiseT [µs] ×0.1 · Dur [µs] ×0.1
#   Eny [eu] 原始 · Counts 无量纲
#   ⇒ Amp 是 **µV 不是 dB**: dB = 20·log10(µV)。论文的 "amplitude threshold 60 dB"
#     = 1000 µV。（注: 5 特征做 z-score 后 PCA 与单位无关，故此换算只影响可读性。）
# 注: 数据集 B 的声速 PDF 原文标注为失效后测量、"might not be useful" ⇒ 用缺省值；
#     冲击点真值不可用（§16.1）⇒ **只能用组内重复性(σ)，不得报"距冲击点偏差"**。
AE_FEATS = ('amp_db', 'rise_us', 'dur_us', 'eny', 'cnts_dur')
AE_FEAT_LABEL = {'amp_db': '幅值 A [dB]', 'rise_us': '上升时间 R [µs]',
                 'dur_us': '持续时间 D [µs]', 'eny': '能量 E [eu]',
                 'cnts_dur': '计数/时长 CNTS/D [1/µs]'}
AE_WIN_US = 20.0            # 事件聚类窗（继承 ae_locate 实测标定值）
# ⚠️ 必须 4 通道（= 论文「four AE sensors」）。min_ch=3 时 TDOA 在 2D 下是
# **恰定方程**（2 方程 2 未知）⇒ 残差 rms 恒 ≈0（实测 0.17 µs），既无冗余度
# 也无定位质量度量；4 通道才有 1 个自由度可算 rms。
AE_MIN_CH = 4
AE_ENRICH_HALF = 10000      # 时段关联的半宽 [cycles]


def read_ae_hits(gid, min_amp_db=None, chunk=500_000, multifile='auto'):
    """读 `.pridb` SetType=2 逐 hit 表 → (Time[s], Chan, Amp[µV], RiseT[µs],
    Dur[µs], Eny[eu], Counts)。

    多段 `.pridb` 的**会话缝合**统一由 `ae_io` 处理（判据与全部实测证据见
    `ae_io` 模块 docstring 与 docs/details.md §17.11）；本函数只做**单位换算**
    （`ae_io` 返回 `.pridb` 原值，此处只换算 RiseT/Dur ×0.1；Time 已由 ae_io
    换算为秒）。

    multifile: 'auto'(默认，按 ae_io.plan 判据) | 'stitch' | 'time' | 'largest'
    """
    import ae_io
    a, _pl = ae_io.read_hits(gid, cols=ae_io.DEFAULT_COLS,
                             multifile=multifile, chunk=chunk)
    fac = np.array([1.0, 1.0, 1.0, 0.1, 0.1, 1.0, 1.0])   # 见文件头单位说明
    a = a * fac[None, :]
    if min_amp_db is not None:
        a = a[20.0 * np.log10(np.maximum(a[:, 2], 1e-3)) >= min_amp_db]
    return a


def ae_events(gid, win_us=AE_WIN_US, min_ch=AE_MIN_CH, min_amp_db=None,
              max_events=None, seed=0, res_mm=2.0, multifile='auto'):
    """事件表（仅 ≥min_ch 通道的簇，与 `ae_locate.run` 同口径）。

    - 簇内每通道取**最早** hit（同 ae_locate）
    - 定位用 `ae_locate.locate`（TDOA 最小二乘 + 各向异性走时表）
    - 事件级 5 特征取该簇内**幅值最大 hit**：Vallen 惯例，幅值在耦合最好/离源最近的
      传感器上最可靠，也是 `.pridb` 事件级参数的常规来源
    - cycle 用 FBG 块锚（`time_to_cycle`，与 Level1/4 同一口径）
    """
    import ae_locate as al
    a = read_ae_hits(gid, min_amp_db, multifile=multifile)
    t_s = a[:, 0]
    chan = a[:, 1].astype(np.int32)
    gx = np.arange(0.0, al.LX + 1e-9, res_mm)
    gy = np.arange(0.0, al.LY + 1e-9, res_mm)
    tmap = al.make_tmap(gx, gy, *al.VEL.get(gid, (al.VX_DEF, al.VY_DEF)))

    starts, ends = al.cluster_runs(t_s, win_us * 1e-6)
    cid = np.repeat(np.arange(starts.size), ends - starts)
    order = np.lexsort((t_s, chan, cid))
    c_s, ch_s = cid[order], chan[order]
    first = np.ones(order.size, bool)
    first[1:] = (c_s[1:] != c_s[:-1]) | (ch_s[1:] != ch_s[:-1])
    sel = order[first]
    nch_all = np.bincount(cid[sel], minlength=starts.size)
    ok = np.nonzero(nch_all >= min_ch)[0]
    if max_events and ok.size > max_events:
        ok = np.sort(np.random.default_rng(seed).choice(ok, max_events,
                                                        replace=False))
    sel_c, sel_ch = cid[sel], chan[sel]
    pos = np.searchsorted(ok, sel_c)
    m = (pos < ok.size) & (ok[np.clip(pos, 0, ok.size - 1)] == sel_c)
    sel2, grp_c, grp_ch = sel[m], sel_c[m], sel_ch[m]
    bnd = np.nonzero(np.diff(grp_c))[0] + 1
    st = np.concatenate(([0], bnd))
    en = np.concatenate((bnd, [grp_c.size]))

    xs, ys, rms, nch, t0, feats = [], [], [], [], [], []
    for i0, i1 in zip(st, en):
        idx = sel2[i0:i1]
        tt = t_s[idx]
        cc = grp_ch[i0:i1]
        o = np.argsort(tt, kind='stable')
        x, y, r = al.locate(tt[o] * 1e6, cc[o], tmap, gx, gy)
        j = idx[int(np.argmax(a[idx, 2]))]      # 幅值最大 hit
        amp, rise, dur, eny, cnt = a[j, 2], a[j, 3], a[j, 4], a[j, 5], a[j, 6]
        xs.append(x); ys.append(y); rms.append(r); nch.append(len(cc))
        t0.append(float(tt[o][0]))
        feats.append((20.0 * np.log10(max(amp, 1e-3)), rise, dur, eny,
                      cnt / dur if dur > 0 else 0.0))
    ev = dict(t_s=np.asarray(t0), x=np.asarray(xs), y=np.asarray(ys),
              rms_us=np.asarray(rms), nch=np.asarray(nch),
              F=np.asarray(feats))
    ev['cyc'] = time_to_cycle(gid, ev['t_s'])
    nf = float(META[gid]['n_f'])
    cov = float(ev['cyc'].max()) / nf if nf else float('nan')
    print(f'  [AE] {gid}: hit {a.shape[0]:,}  簇 {starts.size:,}  '
          f'≥{min_ch}通道事件 {ok.size:,}  '
          f'rms 中位 {np.median(ev["rms_us"]):.2f} µs  '
          f'cycle {ev["cyc"].min()/1000:.1f}k→{ev["cyc"].max()/1000:.1f}k '
          f'(n_f={nf/1000:.1f}k, 覆盖 {cov*100:.1f}%)')
    return ev


def ae_pca_kmeans(ev, n_clusters=2, seed=0, standardize=True, rms_max=None):
    """论文 L3 AE 支路: 5 特征 → PCA(2) → k-means(2)。

    standardize=True 用 z-score（论文 5 个特征量纲差 6 个数量级：Amp dB ~60、
    Eny ~1e10，不标准化则 PCI 完全被 Eny 主导；论文称"前 2 主成分保留约 70% 方差"
    也与标准化后的典型结果一致）。
    rms_max: 只保留可信定位的事件（论文："only localized events are considered"）。
    返回 dict(F, X, Z, evr, labels, km, used_mask)
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    used = np.ones(ev['F'].shape[0], bool)
    if rms_max is not None:
        used &= ev['rms_us'] <= rms_max
    X = ev['F'][used]
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd if standardize else X
    pca = PCA(n_components=min(5, Xs.shape[1]), random_state=seed)
    Z = pca.fit_transform(Xs)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    lab = km.fit_predict(Z[:, :2])
    # 标签确定化：k-means 的簇号与输入顺序/初始化相关（同一数据换读取路径就可能
    # 把 0/1 互换）。按**事件数降序**重排，使「簇 0 = 更大的簇」在多次运行间稳定。
    if n_clusters == 2 and int((lab == 1).sum()) > int((lab == 0).sum()):
        lab = 1 - lab
    return dict(F=X, X=Xs, Z=Z, evr=pca.explained_variance_ratio_,
                labels=lab, km=km, used=used, mu=mu, sd=sd)


def _ae_cluster_stats(ev, lab, m_used, k):
    """单簇统计: 规模 / 特征质心 / 空间集中度 / 时间速率。"""
    m = m_used & (lab == k)
    n = int(m.sum())
    if n == 0:
        return dict(n=0)
    x, y, cy = ev['x'][m], ev['y'][m], ev['cyc'][m]
    # 空间集中度: 2 mm 网格直方图峰值占比（分层/脱粘应比摩擦等更集中）
    h, _, _ = np.histogram2d(x, y, bins=[np.arange(0, 167, 4.0),
                                         np.arange(0, 245, 4.0)])
    return dict(n=n, frac=float(n / m_used.sum()),
                xc=float(np.median(x)), yc=float(np.median(y)),
                xs_=float(np.std(x)), ys_=float(np.std(y)),
                peak_density=float(h.max() / n),
                rms=float(np.median(ev['rms_us'][m])),
                cyc_q=np.percentile(cy, [10, 50, 90]))


def ae_enrichment(cyc, ref_cycles, life, half=AE_ENRICH_HALF):
    """时段关联: 该簇有多少比例的事件落在论文报告的脱粘 cycle 附近。

    论文: "if the AE events in a cluster have similar timestamps [as n_DB], this
    provides strong indications that the data in that cluster is related to the
    disbond." 这里 n_DB 为空（§17.5），故改用**论文报告的脱粘 cycle** 作外部参考。
    返回 (观测占比, 均匀分布期望占比, 富集倍数)。

    ⚠️ 必须对**事件**求均值（早期写成对 ref_cycles 求均值，且用 np.min 取全体
    事件的最小距离 ⇒ 恒为 True ⇒ 恒 100%）。
    """
    if len(cyc) == 0 or not ref_cycles:
        return (float('nan'), float('nan'), float('nan'))
    cyc = np.asarray(cyc, float)
    refs = np.asarray(ref_cycles, float)
    d = np.min(np.abs(cyc[:, None] - refs[None, :]), axis=1)   # 每个事件到最近参考点
    exp = min(1.0, len(refs) * 2 * half / max(life, 1.0))
    obs = float(np.mean(d <= half))
    return (obs, exp, obs / exp if exp > 0 else float('nan'))


def run_ae(gids, res_mm=2.0, min_ch=AE_MIN_CH, win_us=AE_WIN_US,
           min_amp_db=None, max_events=None, rms_max=None, multifile='auto'):
    """跑 AE 支路并出图 + 汇总。"""
    store = {}
    for gid in gids:
        try:
            ev = ae_events(gid, win_us=win_us, min_ch=min_ch,
                           min_amp_db=min_amp_db, max_events=max_events,
                           res_mm=res_mm, multifile=multifile)
        except Exception as e:                                  # noqa: BLE001
            print(f'[跳过 AE] {gid}: {e}')
            continue
        cl = ae_pca_kmeans(ev, rms_max=rms_max)
        evr2 = float(cl['evr'][:2].sum())
        m_used = cl['used']
        stats = {k: _ae_cluster_stats(ev, cl['labels'], m_used, k)
                 for k in range(2)}
        ref = PAPER.get(gid, {}).get('n_DB', {})
        refc = sorted({c for v in ref.values() for c in v})
        life = float(META[gid]['n_f'])
        enr = {k: ae_enrichment(ev['cyc'][m_used & (cl['labels'] == k)],
                                refc, life) for k in range(2)}
        # 两个识别判据（论文对同一件事的两处描述）:
        #  A 时间关联 —— 论文互补融合原话: "if the AE events in a cluster have similar
        #    timestamps [as n_DB] ... related to the disbond"
        #  B 空间集中 —— 论文对 cluster1 的描述: "located near the impact or centered
        #    in areas at the stiffener"
        # 注: A 用**论文报告的 n_DB** 作外部参考（本工作区自产 n_DB 为空，§17.5），
        #     故 A 只能作为**外部校验**，不是自洽的自动标注。
        by_t = max(range(2), key=lambda k: (enr[k][2] if np.isfinite(enr[k][2])
                                            else -1.0))
        by_d = max(range(2), key=lambda k: stats[k].get('peak_density', 0.0))
        store[gid] = dict(ev=ev, cl=cl, stats=stats, enr=enr, evr2=evr2,
                          refc=refc, by_t=by_t, by_d=by_d, m_used=m_used)

        print(f'  [L3-AE] {gid}: 定位事件 {int(m_used.sum()):,}  '
              f'PCA 前2主成分解释 {evr2*100:.1f}%  （论文约 70%）')
        for k in range(2):
            s = stats[k]
            o, e, r = enr[k]
            print(f'     簇{k}: n={s["n"]:,} ({s["frac"]*100:.1f}%)  '
                  f'空间中位 ({s["xc"]:.0f},{s["yc"]:.0f}) mm  '
                  f'σ=({s["xs_"]:.0f},{s["ys_"]:.0f})  '
                  f'峰值密度 {s["peak_density"]:.2e}  rms {s["rms"]:.2f} µs')
            print(f'        时段关联(±{AE_ENRICH_HALF} cyc 于论文 n_DB {refc}): '
                  f'观测 {o*100:.1f}% vs 均匀期望 {e*100:.1f}% → 富集 {r:.2f}×')
        print(f'     ⇒ 判据A(时间关联,论文融合规则) 选 簇{by_t}; '
              f'判据B(空间集中,论文 cluster1 描述) 选 簇{by_d}; '
              f'{"一致" if by_t == by_d else "**不一致**"}')
    if store:
        _save_ae_csv(store)
        _plot_ae(store)
    return store


def _save_ae_csv(store):
    rows = []
    for gid, S in store.items():
        ev = S['ev']
        for k in range(2):
            s = S['stats'][k]
            if s['n'] == 0:
                continue
            o, e, r = S['enr'][k]
            rows.append(dict(
                gid=gid, cluster=k, n=s['n'], frac=round(s['frac'], 4),
                pc2_var=round(S['evr2'], 4),
                xc_mm=round(s['xc'], 1), yc_mm=round(s['yc'], 1),
                x_std=round(s['xs_'], 1), y_std=round(s['ys_'], 1),
                peak_density=s['peak_density'], rms_us=round(s['rms'], 2),
                cyc_p10=round(float(s['cyc_q'][0])), cyc_p50=round(float(s['cyc_q'][1])),
                cyc_p90=round(float(s['cyc_q'][2])),
                enrich_obs=round(o, 4), enrich_exp=round(e, 4),
                enrich_ratio=round(r, 2),
                pick_by_temporal=int(k == S['by_t']),
                pick_by_density=int(k == S['by_d']),
                n_events_total=int(ev['F'].shape[0]),
            ))
    out = os.path.join(RES, 'l1_broer_l23_ae.csv')
    pd.DataFrame(rows).to_csv(out, index=False, encoding='utf-8-sig')
    print('已存:', out)


def _plot_ae(store):
    import ae_locate as al
    cm = ['tab:blue', 'tab:red']
    for gid, S in store.items():
        ev, cl = S['ev'], S['cl']
        fig, axs = plt.subplots(3, 2, figsize=(13.5, 12))
        # (0,0) scree / 累计方差
        a = axs[0, 0]
        cols = np.arange(1, len(cl['evr']) + 1)
        a.bar(cols, cl['evr'] * 100, color='0.7', label='各主成分')
        a.plot(cols, np.cumsum(cl['evr']) * 100, 'o-', color='tab:red',
               label='累计')
        a.axhline(70, color='k', ls='--', lw=0.8, label='论文约 70%')
        a.set_xlabel('主成分'); a.set_ylabel('解释方差 [%]')
        a.set_title(f'{gid} AE 特征 scree plot（论文 Fig.8 口径）')
        a.legend(fontsize=8); a.grid(alpha=.3)
        # (0,1) PCA 散点
        a = axs[0, 1]
        for k in range(2):
            m = cl['labels'] == k
            a.plot(cl['Z'][m, 0], cl['Z'][m, 1], '.', ms=1.5,
                   color=cm[k], label=f'簇{k} n={int(m.sum()):,}')
        a.set_xlabel('PC1'); a.set_ylabel('PC2')
        a.set_title(f'{gid} PCA(2) + k-means(2)')
        a.legend(fontsize=8, markerscale=4); a.grid(alpha=.3)
        # (1,*) 各簇空间密度（左=判据A 选中的簇）
        for j, k in enumerate((S['by_t'], 1 - S['by_t'])):
            a = axs[1, j]
            m = cl['used'] & (cl['labels'] == k)
            a.hist2d(ev['x'][m], ev['y'][m], bins=[22, 26],
                     range=[[0, al.LX], [0, al.LY]], cmap='viridis', cmin=1)
            for chch, (sx, sy) in al.SENSORS.items():
                a.plot(sx, sy, 'r^', ms=10, mec='k', zorder=5)
                a.text(sx + 4, sy + 4, f'S{chch}', color='r', fontsize=8,
                       weight='bold', zorder=6)
            s = S['stats'][k]
            o, e, r = S['enr'][k]
            a.plot(s['xc'], s['yc'], 'w+', ms=14, mew=2, zorder=6)
            a.set_xlim(0, al.LX); a.set_ylim(0, al.LY)
            a.set_aspect('equal')
            a.set_xlabel('X [mm] 横跨加筋条'); a.set_ylabel('Y [mm] 沿加筋条')
            tag = '疑分层/脱粘(判据A:时间关联)' if k == S['by_t'] else '其余'
            a.set_title(f'簇{k} {tag}  n={s["n"]:,}  '
                        f'σ=({s["xs_"]:.0f},{s["ys_"]:.0f}) mm\n'
                        f'峰值密度 {s["peak_density"]:.2e}  '
                        f'时段富集 {r:.2f}×  rms {s["rms"]:.2f} µs')
        # (2,0) 事件率 vs cycle
        a = axs[2, 0]
        edges = np.linspace(0, float(META[gid]['n_f']), 25)
        for k in range(2):
            m = cl['used'] & (cl['labels'] == k)
            h, _ = np.histogram(ev['cyc'][m], bins=edges)
            a.plot(edges[:-1] / 1000, h / np.maximum(np.diff(edges), 1) * 1000,
                   '-o', ms=3, color=cm[k], label=f'簇{k}')
        for c in S['refc']:
            a.axvline(c / 1000, color='lime', ls='--', lw=1.1)
        for c in (S.get('n_SD') or []):
            a.axvline(c / 1000, color='tab:orange', ls=':', lw=0.9)
        a.set_xlabel('cycle (k)'); a.set_ylabel('事件率 [1/kcyc]')
        a.set_title(f'{gid} 各簇 AE 事件率（绿虚线=论文 n_DB）')
        a.legend(fontsize=8); a.grid(alpha=.3)
        # (2,1) 各簇 cycle 直方图
        a = axs[2, 1]
        for k in range(2):
            m = cl['used'] & (cl['labels'] == k)
            a.hist(ev['cyc'][m] / 1000, bins=25, color=cm[k], alpha=.6,
                   label=f'簇{k}')
        for c in S['refc']:
            a.axvline(c / 1000, color='lime', ls='--', lw=1.1)
        a.set_xlabel('cycle (k)'); a.set_ylabel('事件数')
        a.set_title(f'{gid} 各簇时间分布'); a.legend(fontsize=8)
        a.grid(alpha=.3)
        fig.suptitle(f'{gid}  Broer Level2/3 AE 支路（论文 Fig.8/13-15 口径）'
                     f'  窗 {AE_WIN_US:.0f} µs, ≥{AE_MIN_CH} 通道')
        fig.tight_layout()
        out = os.path.join(FIG, f'l1_broer_l23_ae_{gid}.png')
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print('已存:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS))
    ap.add_argument('--mode', default='all',
                    choices=['l3', 'l2', 'ae', 'plot', 'all'])
    ap.add_argument('--window', type=int, default=W_MEAS)
    ap.add_argument('--k-sigma', type=float, default=K_SIGMA)
    ap.add_argument('--consec', type=int, default=N_CONSEC)
    ap.add_argument('--neigh', type=int, default=N_NEIGH)
    ap.add_argument('--res-mm', type=float, default=RES_MM)
    ap.add_argument('--side-mode', default='paper',
                    choices=['paper', 'swap', 'any'],
                    help='符号约束口径: paper=论文原式, swap=互换受压/受拉, any=无符号约束')
    ap.add_argument('--ae-res-mm', type=float, default=2.0, help='AE 定位网格步长 [mm]')
    ap.add_argument('--ae-win', type=float, default=AE_WIN_US, help='AE 事件聚类窗 [µs]')
    ap.add_argument('--ae-min-ch', type=int, default=AE_MIN_CH)
    ap.add_argument('--ae-min-amp-db', type=float, default=None,
                    help='AE 幅值门限 [dB]（论文设 60 dB）')
    ap.add_argument('--ae-rms-max', type=float, default=None,
                    help='只用 rms ≤ 该值的事件（论文"only localized events"）')
    ap.add_argument('--ae-max-events', type=int, default=None)
    ap.add_argument('--ae-multifile', default='auto',
                    choices=['auto', 'stitch', 'time', 'largest'],
                    help='多 .pridb 时: auto=按 ae_io 判据(默认), '
                         'stitch=强制会话缝合, time=按 Time 合并, '
                         'largest=只用最大者')
    ap.add_argument('--sweep', action='store_true')
    a = ap.parse_args()
    gids = [s.strip() for s in a.groups.split(',') if s.strip()]

    if a.sweep:
        sweep(gids)
        return
    store = {}
    if a.mode in ('l3', 'l2', 'all'):
        store = run(gids, W=a.window, k=a.k_sigma, n_consec=a.consec,
                    n_neigh=a.neigh, res_mm=a.res_mm, side_mode=a.side_mode)
        if store:
            save_csv(store)
            if a.mode in ('plot', 'all'):
                plot(store, W=a.window, n_consec=a.consec, n_neigh=a.neigh)
    if a.mode in ('ae', 'all'):
        run_ae(gids, res_mm=a.ae_res_mm, min_ch=a.ae_min_ch, win_us=a.ae_win,
               min_amp_db=a.ae_min_amp_db, max_events=a.ae_max_events,
               rms_max=a.ae_rms_max, multifile=a.ae_multifile)


if __name__ == '__main__':
    main()
