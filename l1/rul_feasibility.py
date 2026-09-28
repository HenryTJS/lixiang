# -*- coding: utf-8 -*-
"""L1 新数据集（11 组）RUL 可行性验证

背景：主样本 016-020 做 RUL 已判不可行（README §11.1，D 阶跃饱和、N=5）。
L1 第二批（L1-49..L1-56）改变了条件：
  - 同工况（10 J 冲击 → -6.5/-65 kN, 2 Hz 压-压疲劳）全寿命试件增至 **11 组**（含第一批 4 组）
  - AE **有真实时间戳**（.pridb 的 Time + markers）→ 事件率/能量率类经典 RUL 特征**可用**
    （主样本 AE 无时间戳，这类特征构造上不可用）
  - DFOS 提供**空间分布式应变**（连续退化量候选）
  - 失效标签 = PDF 实测 n_f（非"数据末端"近似）

本脚本做三件事（全部 LOSO = 留一试件）：
  1. **单调性检验**：特征随寿命进度是否单调（Spearman ρ 与显著组数）
  2. **可分性检验**：早/晚寿命段的特征差异（vs 组内噪声）
  3. **RUL 回归**：轨迹相似性匹配（相似度 = 归一化特征轨迹的欧氏距离），
     报告 MAE（寿命%），并区分"早期预测"（20%/40% 寿命处查询）与非早期。
     —— 早期误差大 = RUL 不可用；晚期误差小但早期大 = 只能做"临近剩余寿命"。

用法:
  python rul_feasibility.py                 # 全部 11 组
  python rul_feasibility.py --groups L1-49,L1-50 --grid 100
输出: results/l1_rul_feasibility.csv + figures/l1_rul_*.png
"""
import os
import sys
import argparse
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
sys.path.insert(0, ROOT)
from l1_meta import load_meta                          # noqa: E402
import l1_time_align as ta                             # noqa: E402

GROUPS_V1 = ['L1-03', 'L1-04', 'L1-05', 'L1-09']
GROUPS_V2 = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
             'L1-59', 'L1-60']
GROUPS = GROUPS_V1 + GROUPS_V2
GRID = 100                     # 寿命归一化网格点数


def _resample(x_life, x_val, grid):
    """把 (寿命进度, 值) 重采样到统一 grid（0~1）。返回 nan-aware 插值。"""
    ok = np.isfinite(x_val) & np.isfinite(x_life)
    if ok.sum() < 3:
        return np.full(len(grid), np.nan)
    return np.interp(grid, x_life[ok], x_val[ok], left=np.nan, right=np.nan)


def dfos_features(gid, grid):
    """DFOS 段级特征轨迹（长度=grid）。列: local(局部峰偏离) / rmse / hi。"""
    fp = os.path.join(RES, f'_l1_dfos_hi_{gid}.npz')
    if not os.path.exists(fp):
        return None
    z = np.load(fp)
    nf = float(z['nf'])
    life = z['cyc'] / nf
    m = life >= 0                                     # 排除 BI 块（cyc = −1）
    out = {}
    for k in ('local', 'rmse', 'hi'):
        if k in z:
            out[k] = _resample(life[m], z[k][m], grid)
    return out


def dfos_peak_feature(gid, grid, nf):
    """段内**峰值载荷行**的脚部应变**相对基线漂移**轨迹。

    优于中位口径的假设：中位把 valley(卸载)/peak(最大压缩) 行平均抹平，
    peak 行才对应第一批 FBG 的「块级循环幅值」语义。
    特征取负号 → 压缩幅值增大时特征上升。
    """
    fp = os.path.join(ROOT, gid, f'{gid}分布式应变_peak.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, encoding='utf-8-sig')
    pos = np.array([float(c[:-2]) for c in df.columns[1:]])
    M = df.iloc[:, 1:].to_numpy(float)
    m = load_meta(gid)
    fm = np.zeros(pos.size, bool)
    for seg in (m.get('foot_L'), m.get('foot_R')):
        if seg:
            fm |= (pos >= seg[0]) & (pos <= seg[1])
    if not fm.any():
        fm[:] = True
    s = np.nanmean(M[:, fm], axis=1)
    anc = ta.dfos_anchor(gid, n_f=nf)
    if anc is None or len(anc) != len(s):
        return None
    life = anc['cyc_seg'].to_numpy(float) / nf
    m = life >= 0                                     # 排除 BI（冲击前, cyc=−1）
    base = float(np.nanmedian(s[:10])) if len(s) >= 10 else float(np.nanmedian(s))
    d = -(s - base) / (abs(base) + 1e-9)
    return _resample(life[m], d[m], grid)


def dfos_amp_feature(gid, grid, nf):
    """段内**循环幅值**(p90−p10) 的脚部均值相对基线变化。

    载荷控制下 幅值 ∝ 1/刚度 → 幅值增长 = 刚度损失，
    与第一批 FBG 的「块内 std」同语义。优于 `dfos_peak`（单行，实测被相位偶然主导）。
    """
    fp = os.path.join(ROOT, gid, f'{gid}分布式应变_amp.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, encoding='utf-8-sig')
    pos = np.array([float(c[:-2]) for c in df.columns[1:]])
    M = df.iloc[:, 1:].to_numpy(float)
    m = load_meta(gid)
    fm = np.zeros(pos.size, bool)
    for seg in (m.get('foot_L'), m.get('foot_R')):
        if seg:
            fm |= (pos >= seg[0]) & (pos <= seg[1])
    if not fm.any():
        fm[:] = True
    s = np.nanmean(M[:, fm], axis=1)
    anc = ta.dfos_anchor(gid, n_f=nf)
    if anc is None or len(anc) != len(s):
        return None
    life = anc['cyc_seg'].to_numpy(float) / nf
    m = life >= 0                                     # 排除 BI（冲击前, cyc=−1）
    base = float(np.nanmedian(s[:10])) if len(s) >= 10 else float(np.nanmedian(s))
    return _resample(life[m], s[m] / (abs(base) + 1e-9) - 1.0, grid)


def dfos_peakpos_feature(gid, grid, nf):
    """|段分布 − 基线| 峰值位置的**归一化**位置轨迹（0~1）。

    物理假设（幅值类特征已证伪，故改试位置类）：冲击后 BVID → 脱粘前沿推进
    → 应变重分布峰位置**迁移**。用归一化位置（pos/pos_max）消除各组光纤长度差异。
    """
    fp = os.path.join(ROOT, gid, f'{gid}分布式应变.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, encoding='utf-8-sig')
    pos = np.array([float(c[:-2]) for c in df.columns[1:]])
    M = df.iloc[:, 1:].to_numpy(float)
    base = np.nanmedian(M[:10], axis=0)
    dev = np.abs(M - base)
    n_seg = len(M)
    p = np.full(n_seg, np.nan)
    for i in range(n_seg):
        d = dev[i]
        if np.isfinite(d).any():
            p[i] = pos[int(np.nanargmax(d))] / pos[-1]
    anc = ta.dfos_anchor(gid, n_f=nf)
    if anc is None or len(anc) != n_seg:
        return None
    life = anc['cyc_seg'].to_numpy(float) / nf
    m = life >= 0                                     # 排除 BI（冲击前, cyc=−1）
    return _resample(life[m], p[m], grid)


def ae_shape_features(gid, grid, nf, cache=None):
    """AE 分布形状特征（b 值 / 幅值分位比）轨迹。

    数据来源：`results/l1_ae_shape_{gid}.npz`（由 `l1/ae_shape_features.py` 预计算，
    因需重读 .pridb 事件级参数，代价高）。不存在时返回 None。
    """
    fp = cache or os.path.join(RES, f'_l1_ae_shape_{gid}.npz')
    if not os.path.exists(fp):
        return None
    z = np.load(fp)
    life = z['life']
    out = {}
    for k in z.files:
        if k != 'life':
            out[k] = _resample(life, z[k].astype(float), grid)
    return out


def _cycle_of(gid, t, nf):
    """时间(s) → cycle 的统一入口（按试件批次分派锚）。

    V2（L1-49..L1-60，无 FBG）: `l1_time_align`（markers 墙钟 + DFOS 测量段）
    V1（L1-03/04/05/09，有 FBG）: `reproduce_broer_l1` 的 FBG 块锚
                                 （块 k ↔ (k+0.5)×5000，线性插值）
    判据: 存在 `{gid}dfos_anchor.csv` ⇒ V2。
    """
    if os.path.exists(os.path.join(ROOT, gid, f'{gid}dfos_anchor.csv')):
        return ta.time_to_cycle(gid, t, nf)
    import reproduce_broer_l1 as _rb
    return _rb.time_to_cycle(gid, np.atleast_1d(np.asarray(t, float)))


def ae_features(gid, grid, nf):
    """AE → 寿命进度 → 网格特征: hits/s, log-energy 峰值。

    兼容两种 AE CSV 格式:
      V2 (`step0_v2.py`): 已是 1 s bin 聚合, 列 = time, n_hits, energy, amplitude
      V1 (`step0.py`)   : hit 级原始记录, 列 = time, channel, amplitude, energy, ...
                          → 此处现场按 1 s bin 聚合成同样的 n_hits / energy
    """
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, encoding='utf-8-sig')
    if df.empty:
        return None
    try:
        cyc = _cycle_of(gid, df['time'].to_numpy(float), nf)
    except Exception as e:                                       # noqa: BLE001
        print(f'  [{gid}] AE 时间对齐失败: {e}')
        return None
    e = df['energy'].to_numpy(float)
    if 'n_hits' in df.columns:                                   # V2: 已是 1 s bin
        nh = df['n_hits'].to_numpy(float)
        esum, cbin = e, cyc
    else:                                                        # V1: hit 级 → 1 s bin
        t = df['time'].to_numpy(float)
        _b, inv = np.unique(np.floor(t).astype(np.int64), return_inverse=True)
        nh = np.bincount(inv).astype(float)                      # hits/s
        esum = np.bincount(inv, weights=e)                       # 每秒能量和
        cbin = np.bincount(inv, weights=cyc) / np.maximum(nh, 1.0)
        print(f'  [{gid}] AE hit 级 {len(t)} 行 → 1 s bin {len(nh)} 行')
    life = cbin / nf
    out = {}
    out['ae_env'] = _resample(life, np.log10(esum + 1.0), grid)
    out['ae_hits'] = _resample(life, nh, grid)
    return out


def norm_traj(v):
    """组内 min-max 归一（消除组间尺度差，只保留形状）。"""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 5:
        return np.full_like(v, np.nan)
    lo, hi = np.nanmin(v[ok]), np.nanmax(v[ok])
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


def build(groups, grid):
    """返回 {gid: dict(life_grid, feats{name: traj}), nf}。"""
    data = {}
    for g in groups:
        nf = load_meta(g)['n_f']
        if not nf:
            print(f'  [{g}] n_f 解析失败，跳过')
            continue
        f = dfos_features(g, grid) or {}
        pk = dfos_peak_feature(g, grid, nf)
        if pk is not None:
            f['dfos_peak'] = pk
        am = dfos_amp_feature(g, grid, nf)
        if am is not None:
            f['dfos_amp'] = am
        pp = dfos_peakpos_feature(g, grid, nf)
        if pp is not None:
            f['dfos_peakpos'] = pp
        ash = ae_shape_features(g, grid, nf)
        if ash:
            f.update(ash)
        a = ae_features(g, grid, nf) or {}
        f.update(a)
        if not f:
            print(f'  [{g}] 无可用特征，跳过')
            continue
        data[g] = dict(nf=nf, feats=f)
        print(f'  [{g}] n_f={nf} 特征={list(f.keys())}')
    return data


def spearman(x, y):
    """无 scipy 的 Spearman（秩相关）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        return np.nan
    rx = pd.Series(x[m]).rank().to_numpy()
    ry = pd.Series(y[m]).rank().to_numpy()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else np.nan


def rul_similarity(train, test, j, feat_names, grid):
    """相似性 RUL：用测试组前 j 格轨迹匹配训练组，预测剩余寿命（%）。

    距离 = 前 j 格归一化特征轨迹的欧氏距离（多特征取平均）；
    预测 RUL = 训练组在该匹配点的剩余寿命，按 1/dist 加权。
    """
    q = np.zeros(j)
    wsum = 0.0
    preds, ws = [], []
    for name in feat_names:
        tv = test['feats'].get(name)
        if tv is None:
            continue
        qv = norm_traj(tv)[:j]
        if not np.isfinite(qv).any():
            continue
        for gid, tr in train.items():
            rv = tr['feats'].get(name)
            if rv is None:
                continue
            rn = norm_traj(rv)
            # 训练组所有可能的截断点 k（k>=j，留出预测空间）
            for k in range(j, grid):
                seg = rn[k - j:k]
                if not np.isfinite(seg).all() or not np.isfinite(qv).all():
                    continue
                d = float(np.mean((seg - qv) ** 2)) ** 0.5
                w = 1.0 / (d + 1e-3)
                preds.append((1.0 - (k + 0.5) / grid) * 100.0)   # 剩余寿命(%)
                ws.append(w)
    if not preds:
        return np.nan
    preds, ws = np.asarray(preds), np.asarray(ws)
    return float((preds * ws).sum() / ws.sum())


def causal_norm(v, grid, frac=0.10):
    """**因果**归一化：只用寿命前 frac 段的统计量（在线可得，不引入未来信息）。

    跨试件绝对尺度不可比（应变基线/贴片位置不同）→ 必须组内标准化；
    用全寿命 min-max 是离线口径（RUL 场景不可用），故取前 frac 的 median/MAD。
    """
    v = np.asarray(v, float)
    m = max(3, int(len(v) * frac))
    seg = v[:m][np.isfinite(v[:m])]
    if len(seg) < 3:
        return np.full_like(v, np.nan)
    med = float(np.median(seg))
    mad = float(np.median(np.abs(seg - med))) * 1.4826
    if not (mad > 0):
        mad = float(np.nanstd(seg)) or 1.0
    return (v - med) / mad


def rul_direct(train, test, j, names, grid, k=5):
    """直接映射 RUL：特征(因果归一) → life 的 kNN 回归 → RUL(%)。

    训练集 = 其余试件的全部 (特征值, 寿命进度) 对（离线可得）；
    测试 = 该试件前 j 格（只向外提供已观测到的部分，因果）。
    """
    X, Y = [], []
    for name in names:
        for gid, tr in train.items():
            v = tr['feats'].get(name)
            if v is None:
                continue
            vn = causal_norm(v, grid)
            for i in range(grid):
                if np.isfinite(vn[i]):
                    X.append(vn[i]); Y.append((i + 0.5) / grid)
    if not X:
        return np.nan
    X, Y = np.asarray(X), np.asarray(Y)
    preds = []
    for name in names:
        v = test['feats'].get(name)
        if v is None:
            continue
        vn = causal_norm(v, grid)
        seg = vn[max(0, j - 5):j]
        seg = seg[np.isfinite(seg)]
        if len(seg) == 0:
            continue
        q = float(np.mean(seg))                     # 当前时刻的特征水平
        idx = np.argsort(np.abs(X - q))[:k]
        preds.append((1.0 - float(Y[idx].mean())) * 100.0)
    return float(np.mean(preds)) if preds else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS))
    ap.add_argument('--grid', type=int, default=GRID)
    ap.add_argument('--queries', default='20,40,60,80',
                    help='查询点(寿命%)')
    a = ap.parse_args()
    grid = a.grid
    gid_list = [g.strip() for g in a.groups.split(',')]
    grid_arr = np.linspace(0, 1, grid)

    print('=== 构建特征（DFOS 段级 + AE 事件率/能量）===')
    data = build(gid_list, grid_arr)
    if not data:
        print('无可用数据'); return
    names = sorted({k for d in data.values() for k in d['feats']})
    print(f'可用特征: {names}\n')

    # ---- 1) 单调性 ----
    print('=== 1) 单调性检验（特征 vs 寿命进度, Spearman ρ）===')
    mono = []
    for gid, d in data.items():
        row = dict(gid=gid)
        for n in names:
            v = d['feats'].get(n)
            row[n] = round(spearman(grid_arr, v), 3) if v is not None else None
        mono.append(row)
    mdf = pd.DataFrame(mono)
    print(mdf.to_string(index=False))

    # ---- 2) 可分性（早 30% vs 晚 30%）----
    print('\n=== 2) 可分性（life<30% vs >70% 的组内标准化差异）===')
    seps = []
    for gid, d in data.items():
        row = dict(gid=gid)
        for n in names:
            v = np.asarray(d['feats'].get(n, np.full(grid, np.nan)), float)
            e, l = v[:int(grid * .3)], v[int(grid * .7):]
            e, l = e[np.isfinite(e)], l[np.isfinite(l)]
            if len(e) < 3 or len(l) < 3:
                row[n] = None; continue
            sd = np.nanstd(np.concatenate([e, l]))
            row[n] = round(float((np.nanmean(l) - np.nanmean(e)) / sd), 2) if sd > 0 else None
        seps.append(row)
    sdf = pd.DataFrame(seps)
    print(sdf.to_string(index=False))

    # ---- 3) LOSO RUL：轨迹匹配 vs 直接映射 ----
    print('\n=== 3) LOSO RUL（MAE, 寿命%）===')
    qs = [int(x) for x in a.queries.split(',')]
    rows = []
    for gid in data:
        train = {k: v for k, v in data.items() if k != gid}
        row = dict(gid=gid, n_f=data[gid]['nf'])
        for q in qs:
            j = max(3, int(grid * q / 100))
            truth = 100.0 - q
            p1 = rul_similarity(train, data[gid], j, names, grid)
            p2 = rul_direct(train, data[gid], j, names, grid)
            row[f'sim@{q}%'] = round(p1, 1) if np.isfinite(p1) else None
            row[f'dir@{q}%'] = round(p2, 1) if np.isfinite(p2) else None
            row[f'err_sim@{q}%'] = round(abs(p1 - truth), 1) if np.isfinite(p1) else None
            row[f'err_dir@{q}%'] = round(abs(p2 - truth), 1) if np.isfinite(p2) else None
        rows.append(row)
    rdf = pd.DataFrame(rows)
    print(rdf[[c for c in rdf.columns if not c.startswith(('sim@', 'dir@'))]]
          .to_string(index=False))
    print('\n各查询点 MAE（真值 RUL = 100 − 查询点）：')
    for q in qs:
        for m, lab in (('err_sim', '轨迹匹配'), ('err_dir', '直接映射')):
            v = pd.to_numeric(rdf[f'{m}@{q}%'], errors='coerce')
            print(f'  [{lab}] @{q}% 寿命: MAE={v.mean():5.1f}  '
                  f'median={v.median():5.1f}  n={v.notna().sum()}')

    out = os.path.join(RES, 'l1_rul_feasibility.csv')
    mdf.to_csv(out.replace('.csv', '_monotonic.csv'), index=False, encoding='utf-8-sig')
    sdf.to_csv(out.replace('.csv', '_separability.csv'), index=False, encoding='utf-8-sig')
    rdf.to_csv(out, index=False, encoding='utf-8-sig')
    print('\n已存:', out)

    # ---- 图：特征轨迹总览 ----
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for gid, d in data.items():
        axes[0].plot(grid_arr * 100, norm_traj(d['feats'].get('local', np.full(grid, np.nan))),
                     lw=1.0, alpha=.8, label=gid)
        axes[1].plot(grid_arr * 100, norm_traj(d['feats'].get('ae_env', np.full(grid, np.nan))),
                     lw=1.0, alpha=.8, label=gid)
    axes[0].set_ylabel('DFOS local (归一)'); axes[0].set_title('DFOS 局部峰偏离 vs 寿命进度')
    axes[1].set_ylabel('AE log-energy (归一)'); axes[1].set_xlabel('寿命进度 (%)')
    axes[1].set_title('AE 能量包络 vs 寿命进度')
    for ax in axes:
        ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=6)
    fig.tight_layout()
    fp = os.path.join(FIG, 'l1_rul_features.png')
    fig.savefig(fp, dpi=110)
    print('已存:', fp)


if __name__ == '__main__':
    main()
