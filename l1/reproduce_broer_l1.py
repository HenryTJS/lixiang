# -*- coding: utf-8 -*-
"""复现 Broer et al. 2022 (L1-03/04/05) 方法 —— 论文式融合 AE + DFOS
参考论文 Methodology:
  Level 1 损伤检测: y_F = y'_AE + y'_OFc − y'_OFt (各 unity 归一化[0,1]+减初值,
                    缺失区间前值填充) → 滑动窗(5×500cycle)变点检测(相邻窗均值差>2σ)
  Level 4 损伤严重度: HI_AE = 每500cycle累积AE能量; HI_OFc/t = 每脚应变均值相对首测变化;
                    HI_OF = HI_OFc − HI_OFt; 归一化[0,1];
                    HI_F = 0.5·HI_AE + 0.5·HI_OF → 初~0 终~1

数据:
  - DFOS(分布式应变): {gid}分布式应变.csv 每500cycle在 valley(min)/peak(max) load 各测一行
    左右脚空间段: ODiSi-B location 见 L1-xx.pdf (L1-03: 左4530-4660 右1510-1690 等)
  - AE: {gid}声发射.csv 事件级(energy), 连续记录

用法: python reproduce_broer_l1.py [--groups L1-03,L1-05] [--mode level1|level4|plot|all]
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.abspath(__file__))   # <项目根>\l1 (本脚本目录)
sys.path.insert(0, ROOT)                            # 使 l1_meta 可导入
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# 元信息自适应: n_f 与 DFOS 左右脚空间段由各组 L1-xx.pdf 自动解析(适配不同光纤布设);
# refs(论文损伤检测点)属标签, 仅 03/04/05 有, 静态维护在 l1_meta。
from l1_meta import load_meta                       # noqa: E402
GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09']
META = {g: load_meta(g) for g in GROUPS}
CYCLES_BIN = 500   # 论文 y/HI 的 cycle 步长
# L1-05 起点修正开关: 对 HI_OF 负值截断(见 level4 注释)。设为 False 即按论文原式(min/max)
CLIP_HI_OF_NEG = True
# 每脚应变均值的小窗平滑(方案A): 窗 w=5(≈2500cycle, 同论文变点窗长); 0/1=关闭
SMOOTH_WIN = 5


def smooth_mean(x, w):
    """nan-aware 滑动均值(窗 w, 边缘按可用点数平均)。用于抑制 DFOS 每脚均值逐bin抖动。"""
    x = np.asarray(x, float)
    n = len(x)
    if w <= 1 or n == 0:
        return x.copy()
    k = w // 2
    out = np.empty(n)
    for i in range(n):
        a = max(0, i - k)
        b = min(n, i + k + 1)
        out[i] = np.nanmean(x[a:b])
    return out


def load_dfos(gid):
    df = pd.read_csv(rf'{ROOT}\{gid}\{gid}分布式应变.csv', encoding='utf-8-sig')
    t = df['timestamp'].to_numpy(float)
    pos = np.array([float(c[:-2]) for c in df.columns[1:]])
    M = df.iloc[:, 1:].to_numpy(float)
    return t, pos, M


def load_ae(gid):
    df = pd.read_csv(rf'{ROOT}\{gid}\{gid}声发射.csv', encoding='utf-8-sig')
    return df[['time', 'energy']].to_numpy()


def fbg_cycle_anchor(gid):
    """用 FBG 测量块作 cycle 锚: 块 k(>300s gap 分隔) ↔ cycle ≈ (k+0.5)*5000。
    返回 (块中心时间数组, 对应 cycle 数组) 供 np.interp 校准 DFOS/AE 行的 cycle。
    """
    fb = pd.read_csv(rf'{ROOT}\{gid}\{gid}光纤.csv', encoding='utf-8-sig')
    tf = fb['timestamp'].to_numpy(float)
    gap = np.where(np.diff(tf) > 300.0)[0]
    bounds = np.concatenate([[0], gap + 1, [len(tf)]])
    mid_t, cyc = [], []
    k = 0
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < 500:
            continue
        mid_t.append(0.5 * (tf[a] + tf[b - 1]))
        cyc.append((k + 0.5) * 5000)
        k += 1
    return np.array(mid_t), np.array(cyc)


def time_to_cycle(gid, t):
    """任意时间(与 FBG 同墙钟) → cycle, 用 FBG 块锚线性插值(外推用首末块延伸)。"""
    nf = META[gid]['n_f']
    mt, mc = fbg_cycle_anchor(gid)
    # 首块前 → cycle 按 0~首块中心比例; 末块后 → 线性延伸到 nf
    cyc = np.interp(t, mt, mc, left=0.0, right=float(mc[-1]))
    # 末块后到 nf 用斜率外推
    tail = t > mt[-1]
    if tail.any() and len(mt) > 1:
        slope = (mc[-1] - mc[-2]) / (mt[-1] - mt[-2])
        cyc[tail] = mc[-1] + (t[tail] - mt[-1]) * slope
    return np.clip(cyc, 0, nf)


def dfos_binned_mean(gid, nf, smooth=0):
    """把 DFOS 行按 cycle 分箱(500cycle, FBG 锚校准), 每箱取 peak(max load) 行
    的左右脚应变均值。
    smooth: 每脚均值的小窗滑动均值窗长(方案A, 仅 Level4 使用; 0=不平滑)。
    DFOS 行结构: valley(min load, 应变~0) 与 peak(max load, 大压缩平台) 交替;
    用"左脚应变最负"判 peak 行更稳(此前用 RMS 最大会误选到 valley 行)。
    但个别 bin 因行→bin 归属抖动只含 valley 行, argmin 会选到非加载杂散行
    (左应变 ~ -100..-800), 造成 -3300↔-200 交替跳变。故要求所选行两脚至少一脚
    深度受压(应变 < -1000, 三组真实 max-load 压缩脚均 < -2000)才接受, 否则置 NaN
    由前后填充替代(论文同样对缺失区间做前值填充)。
    输出 cycle 网格 + 左右脚应变均值(缺失前值填充, 前导 NaN 用首个有效值回填)。
    """
    t, pos, M = load_dfos(gid)
    foot_L = META[gid]['foot_L']
    foot_R = META[gid]['foot_R']
    mL = (pos >= foot_L[0]) & (pos <= foot_L[1])
    mR = (pos >= foot_R[0]) & (pos <= foot_R[1])
    rawL = np.nanmean(M[:, mL], axis=1)   # 每行左脚应变均值
    rawR = np.nanmean(M[:, mR], axis=1)   # 每行右脚应变均值
    # cycle 网格 (FBG 锚校准)
    nb = int(nf / CYCLES_BIN)
    cyc_grid = np.arange(nb + 1) * CYCLES_BIN
    cyc_row = time_to_cycle(gid, t)
    bin_idx = np.clip((cyc_row / CYCLES_BIN).astype(int), 0, nb)
    Lm = np.full(nb + 1, np.nan)
    Rm = np.full(nb + 1, np.nan)
    for b in range(nb + 1):
        sel = np.where(bin_idx == b)[0]
        if len(sel) == 0:
            continue
        # peak 行 = 左脚应变最小(最负, 最大压缩)
        pk = sel[int(np.nanargmin(rawL[sel]))]
        # 真实 max-load 行: 两脚同时受压(轴向压缩下两脚均在压缩域)。
        # 条件1 min<-1000 排除 valley/杂散行; 条件2 max<-200 排除"左脚受压但右脚
        # 卸荷"的伪行(如准静态段或测量伪影, 曾致 L1-05 ~82.5k 处单点 -1403 异常)。
        if min(rawL[pk], rawR[pk]) < -1000 and max(rawL[pk], rawR[pk]) < -200:
            Lm[b] = np.nanmean(M[pk, mL])
            Rm[b] = np.nanmean(M[pk, mR])
    # 缺失填充: 前值填充 + 前导 NaN 用首个有效值回填
    for arr in (Lm, Rm):
        for i in range(1, len(arr)):
            if np.isnan(arr[i]):
                arr[i] = arr[i - 1]
        if np.isnan(arr[0]):
            first = np.where(~np.isnan(arr))[0]
            if len(first):
                arr[:first[0]] = arr[first[0]]
    # 方案A: 每脚应变均值小窗平滑(抑制逐bin抖动/离群; 仅 Level4 使用)
    if smooth and smooth > 1:
        Lm = smooth_mean(Lm, smooth)
        Rm = smooth_mean(Rm, smooth)
    return cyc_grid, Lm, Rm


def ae_binned_energy(gid, nf):
    """每 500cycle 累积 AE 能量 (FBG 锚校准 AE time→cycle)。"""
    ae = load_ae(gid)
    t, e = ae[:, 0], ae[:, 1]
    nb = int(nf / CYCLES_BIN)
    cyc_row = time_to_cycle(gid, t)
    bin_idx = np.clip((cyc_row / CYCLES_BIN).astype(int), 0, nb)
    acc = np.bincount(bin_idx, weights=np.maximum(e, 0), minlength=nb + 1)
    return acc


def compressed_is_R(yL, yR):
    """压缩脚 = 受载时更负(更受压)的一侧 stiffener 1(论文 Fig5b: stiffener1 更大压缩)。
    用两脚都真实受压(应变<-500)的 bin 比较: 均值更负者=压缩脚。
    返回 True 表示压缩脚为右脚。
    """
    yL, yR = np.asarray(yL, float), np.asarray(yR, float)
    both = (yL < -500) & (yR < -500)
    if not both.any():
        return bool(np.nanmean(yL) > np.nanmean(yR))
    return bool(np.nanmean(yL[both]) > np.nanmean(yR[both]))


def ae_binned_count(gid, nf):
    """每 500cycle 的 AE 事件数 (论文 Level1 y_AE 用事件数)。"""
    ae = load_ae(gid)
    t = ae[:, 0]
    nb = int(nf / CYCLES_BIN)
    cyc_row = time_to_cycle(gid, t)
    bin_idx = np.clip((cyc_row / CYCLES_BIN).astype(int), 0, nb)
    return np.bincount(bin_idx, minlength=nb + 1)


def unity01(x):
    """unity 归一化到 [0,1]: (x-min)/(max-min)。"""
    x = np.asarray(x, float)
    if np.all(np.isnan(x)):
        return x
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def level1(gid):
    """复现 Level 1: y_F = y'_AE + y'_OFc − y'_OFt (论文式)。
    步骤: ① y_AE = 每500cycle累积 AE 事件数; ② 压缩/拉伸脚应变均值曲线
    y_OFc/y_OFt; ③ 各 unity 归一化[0,1](式1) 再减初始值(起点0); ④ 相加。
    """
    nf = META[gid]['n_f']
    cyc, yL, yR = dfos_binned_mean(gid, nf)
    aeC = ae_binned_count(gid, nf)
    # y_AE = 累积 AE 事件数
    y_AE = np.cumsum(aeC)
    # 压缩/拉伸脚(压缩=更负)
    cR = compressed_is_R(yL, yR)
    y_OFc = yR if cR else yL
    y_OFt = yL if cR else yR
    # unity 归一化[0,1] 再减初值 → 各起点 0
    y_AE0 = unity01(y_AE) - unity01(y_AE)[0]
    y_OFc0 = unity01(y_OFc) - unity01(y_OFc)[0]
    y_OFt0 = unity01(y_OFt) - unity01(y_OFt)[0]
    # 融合: 压缩脚 +, 拉伸脚 − (反向行为)
    y_F = y_AE0 + y_OFc0 - y_OFt0
    # ---- 论文式变点检测: 滑动窗长5(2000cycle), 相邻窗均值差>2σ ----
    # 论文式4: σ 为全程窗方差算术均值的平方根, 视为恒定; 用局部 σ 会在平坦段
    # 阈值过小导致过度检测。
    W = 5
    vars_all = [float(np.var(y_F[i:i + W], ddof=1)) for i in range(len(y_F) - W)]
    sigma = float(np.sqrt(np.mean(vars_all))) if vars_all else 0.0
    n_D = []
    for i in range(W, len(y_F) - W):
        mu1 = np.mean(y_F[i - W:i])
        mu2 = np.mean(y_F[i:i + W])
        if abs(mu2 - mu1) > 2 * sigma:
            n_D.append(float(cyc[i]))
    # 去重(相邻变点合并: 论文收集所有变化 cycle; 此处保留首个+后续间隔>2000)
    n_D_sorted = []
    for c in n_D:
        if not n_D_sorted or c - n_D_sorted[-1] > 2000:
            n_D_sorted.append(c)
    return dict(gid=gid, cyc=cyc, y_F=y_F, y_AE=y_AE, y_OF_L=yL, y_OF_R=yR,
                n_D=n_D_sorted)


def level4(gid):
    """复现 Level 4: HI_F = 0.5·HI_AE + 0.5·HI_OF, 各[0,1]。
    HI_OFc/t = 压缩/拉伸脚应变均值相对首测(带符号)变化; HI_OF = HI_OFc − HI_OFt
    (拉伸脚损伤行为反向故减去, 论文式6); HI_AE = 累积AE能量/500cycle;
    unity 归一化(式1)后各 1/2 相加(式7)。
    """
    nf = META[gid]['n_f']
    cyc, yL, yR = dfos_binned_mean(gid, nf, smooth=SMOOTH_WIN)
    aeE = ae_binned_energy(gid, nf)
    # HI_AE = 每500cycle AE 累积能量
    HI_AE = np.cumsum(aeE)
    cR = compressed_is_R(yL, yR)
    y_OFc = yR if cR else yL      # 压缩脚(更负)
    y_OFt = yL if cR else yR      # 拉伸脚
    # 带符号的相对首测均值变化
    dL = yL - yL[0]
    dR = yR - yR[0]
    HI_OFc = y_OFc - y_OFc[0]
    HI_OFt = y_OFt - y_OFt[0]
    HI_OF = HI_OFc - HI_OFt       # 式6: 减拉伸脚(反向)
    # L1-05 起点修正: 拉伸脚边沿伪松弛(论文 Fig11(c) 亲述: 边沿应变下降、中心恒定)
    # 使 HI_OF 前段为负; 负值对"损伤严重度"非物理(0=无损伤) → 截断到 0, 令归一化
    # 起点归零。仅影响 L1-05: L1-03/04 的 HI_OF 最小值本就在起点, 截断后完全不变。
    if CLIP_HI_OF_NEG:
        HI_OF = np.maximum(HI_OF, 0.0)
    # 归一化 [0,1] + 等权融合 (式1, 式7; 不额外减起点——论文靠各 HI 单调性达成 起0终1)
    HI_AE_n = unity01(HI_AE)
    HI_OF_n = unity01(HI_OF)
    HI_F = 0.5 * HI_AE_n + 0.5 * HI_OF_n
    return dict(gid=gid, cyc=cyc, HI_F=HI_F, HI_AE=HI_AE_n, HI_OF=HI_OF_n,
                c_is_R=cR, dL=dL, dR=dR)


def plot_level4(r):
    gid = r['gid']
    nf = META[gid]['n_f']
    x = r['cyc'] / 1000
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, r['HI_F'], 'k-', lw=1.5, label='HI_F (融合)')
    ax.plot(x, r['HI_AE'], '--', color='tab:blue', lw=1, label='HI_AE (AE能量)')
    ax.plot(x, r['HI_OF'], '--', color='tab:green', lw=1, label='HI_OF (DFOS)')
    for lab, c in META[gid]['refs']:
        ax.axvline(c / 1000, color='grey', ls=':', lw=0.8)
    ax.axvline(nf / 1000, color='k', ls='--', lw=1, label='n_f')
    ax.set_xlim(0, nf / 1000 * 1.05)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('cycle (k)'); ax.set_ylabel('HI')
    ax.set_title(f'{gid}  Level4 融合损伤严重度 HI_F = 0.5·HI_AE+0.5·HI_OF '
                 f'(c_is_R={r["c_is_R"]})')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, f'l1_repro_{gid}.png')
    fig.savefig(out, dpi=120)
    print('已存:', out)


def plot_level1(r):
    gid = r['gid']
    nf = META[gid]['n_f']
    x = r['cyc'] / 1000
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, r['y_F'], 'k-', lw=1.3, label='y_F (AE+DFOS 融合)')
    ax.plot(x, unity01(r['y_OF_L']) - unity01(r['y_OF_L'])[0], '--', color='tab:green', label='脚L(相对)')
    ax.plot(x, unity01(r['y_OF_R']) - unity01(r['y_OF_R'])[0], '--', color='tab:red', label='脚R(相对)')
    for nd in r['n_D']:
        ax.axvline(nd / 1000, color='r', lw=1.0, ls='-')
    for lab, c in META[gid]['refs']:
        ax.axvline(c / 1000, color='grey', ls=':', lw=0.8)
    ax.axvline(nf / 1000, color='k', ls='--', lw=1)
    ax.set_xlabel('cycle (k)'); ax.set_ylabel('y_F')
    ax.set_title(f'{gid}  Level1 y_F + 变点检测 n_D (红实线=检测, 灰点线=论文参考)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG, f'l1_repro_l1_{gid}.png')
    fig.savefig(out, dpi=120)
    # 控制台对照
    print(f'  [Level1 {gid}] 检测 n_D (cycle): {[round(v) for v in r["n_D"]]}')
    print(f'    论文参考: ' + ', '.join(f'{lab}@{c}' for lab, c in META[gid]['refs']))
    print('    已存:', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(META))
    ap.add_argument('--mode', default='all', choices=['level1', 'level4', 'plot', 'all'])
    a = ap.parse_args()
    groups = a.groups.split(',')
    if a.mode in ('level1', 'all'):
        for g in groups:
            r = level1(g)
            plot_level1(r)
    if a.mode in ('level4', 'all'):
        rows = []
        for g in groups:
            r = level4(g)
            plot_level4(r)
            cyc, HF = r['cyc'], r['HI_F']
            HO, HA = r['HI_OF'], r['HI_AE']

            def fr(y, thr):
                j = np.where(y >= thr)[0]
                return round(float(cyc[j[0]])) if len(j) else float('nan')
            rows.append(dict(gid=g, c_is_R=r['c_is_R'],
                             HI_F_start=round(float(HF[0]), 3),
                             HI_F_end=round(float(HF[-1]), 3),
                             HI_OF_start=round(float(HO[0]), 3),
                             HI_OF_end=round(float(HO[-1]), 3),
                             HI_AE_end=round(float(HA[-1]), 3),
                             F_at_05=fr(HF, 0.5), F_at_085=fr(HF, 0.85),
                             OF_at_05=fr(HO, 0.5)))
        pd.DataFrame(rows).to_csv(os.path.join(RES, 'l1_broer_level4.csv'),
                                  index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
