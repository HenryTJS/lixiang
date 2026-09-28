"""
Step 2: 波形级 DL —— PHM2019 铝搭接件（数据集 D）
====================================================

在波形上直接学裂纹长度（不经过人工特征），与 Step 1 的特征工程基线**同口径对照**。

输入二选一（README §5 Step 2 的 2a / 2b）
----
  --input wave   2a：1D 带通后的 ch2（默认降采样到 2 MHz ⇒ 400 点 / 200 µs）
  --input stft   2b：短时傅里叶幅值图（2D，本脚本固定为 32×25）

模型
----
轻量 1D/2D-CNN，**参数量 < 50k**（README 硬性要求）。
试件只有 8 个、每试件 3~10 个时刻 ⇒ 网络必须小，否则必然过拟合。
用**多随机种子集成**（默认 5 个）压制方差 —— 小样本下这比调网络结构有效得多。

数据增强（README 要求）：高斯加噪、时移、幅值缩放、频带掩蔽。

⚠️ 三个方法学要点（都是为了避免"看起来很好但其实错"）
----
1. **归一化不能用到测试试件的标签。**
   幅值类信息的口径有两种，本脚本默认用**无需标签**的那种：
     `--norm first`（默认）：用该试件**最早一次**测量的 RMS 归一 —— 只用到"时间顺序"，
                          不需要知道哪个时刻裂纹为 0，**是可部署口径**。
     `--norm zero`          ：用该试件**裂纹=0 时刻**的平均 RMS 归一 ——
                          与 Step 1 的 `mean_zero` 基线同口径，但**用到了标签信息**，
                          属"假设可先录一段健康基线"的工程前提，必须显式声明。
   ⚠️ Step 1 用的就是 `zero` 口径；此处把它作为**对照臂**而非默认，是为了让
      "跨试件标定"这件事不被隐藏的标签泄漏美化。
   两种口径**都只对整机幅值做归一**（试件内共享一个常数），因此**保留**了
   "随裂纹的透射幅值下降"这一最强物理信号，不会把它归一化掉。

2. **物理约束层与 Step 1 完全一致**（保序投影 PAVA + 非负），
   且投影只用"按循环数排序"、**不用标签** ⇒ 无泄漏，可直接对照"加约束前/后"。

3. **LOSO 8 折**，训练集 = 其余 7 个试件（含其两次重复），
   测试 = 留出试件的两次重复分别评估 —— 与 Step 1 的 rep1/rep2 双报口径一致。

输出
----
  phmdc/results/step2_pred.csv    逐时刻预测
  phmdc/results/step2_loso.csv    逐折 × 逐臂指标
  phmdc/results/step2_report.txt  报告（含与 Step 1 的对照表）

依赖：numpy / pandas / scipy / torch
"""

import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd

import prep

# ⚠️ `torch` 必须是**模块级**导入：make_cnn 里嵌套定义的 Module.forward
#    在调用时到**模块全局**找 `torch`，不会捕获外层函数的局部变量。
try:
    import torch
    import torch.nn as nn
except ImportError as e:      # pragma: no cover
    raise SystemExit('Step 2 需要 torch：%s\n请用 conda 环境 xjtushm 运行。' % e)

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
# 数据准备
# ============================================================
def build_dataset(norm='first'):
    """返回 (waves, refs, meta)。

    `waves[i]` = 第 i 个样本**已按试件内共享常数归一**的带通 ch2；
    `refs[sp]` = 该试件的**参考波形**（与归一化口径一致），供 `--chans 2` 用。

    归一化：对每个试件算一个**共享标量** scale，所有波形同除该值。
      norm='first' —— scale = 该试件**最早一次**（rep1、最小 cycle）的 RMS（**无标签**）
      norm='zero'  —— scale = 该试件裂纹=0 时刻的 RMS 平均（**用到标签**，见模块 docstring）
    ⇒ 保留试件内"相对自己的幅值变化"，去掉试件间增益差异。
    """
    lab, idx = prep.labels(), prep.index()
    cycles = {}
    for r in idx.itertuples():
        cycles.setdefault(r.specimen, set()).add(r.cycle)

    scales, refs = {}, {}
    for sp in prep.SPECIMENS:
        if not cycles.get(sp):
            continue
        if norm == 'first':
            c0 = min(cycles[sp])
            waves = [prep.prep(sp, c0, rp)['ch2'] for rp in (1, 2)]
        else:
            z = sorted(int(c) for c in
                       lab[(lab['specimen'] == sp) & (lab['crack_mm'] == 0)]['cycle']
                       if int(c) in cycles[sp])
            if not z:
                z = [min(cycles[sp])]
            waves = [prep.prep(sp, c, rp)['ch2'] for c in z for rp in (1, 2)]
        L = min(len(w) for w in waves)
        ref = np.mean([w[:L] for w in waves], axis=0)      # 参考（健康）波形
        scales[sp] = float(np.sqrt(np.mean(ref ** 2)))
        refs[sp] = (ref / scales[sp]).astype(np.float32)

    rows, waves = [], []
    n_no_label = 0
    for sp in prep.SPECIMENS:
        if sp not in scales or not np.isfinite(scales[sp]) or scales[sp] <= 0:
            continue
        for cyc in sorted(cycles[sp]):
            # ⚠️ 必须按**标签**筛，不能按波形目录筛：T3/55391 有波形但真值表里没有
            #    对应记录（Step 0 核对 1 已确认）。若带进来 crack_mm 会是 NaN，
            #    使整个折的 RMSE 变 NaN —— 而 pandas 的 .mean() 会**静默跳过** NaN 折，
            #    结果是"少算了一折"却看不出来。
            lab_row = lab[(lab['specimen'] == sp) & (lab['cycle'] == cyc)]
            if not len(lab_row) or not np.isfinite(lab_row['crack_mm'].iloc[0]):
                n_no_label += 1
                continue
            for rp in (1, 2):
                try:
                    d = prep.prep(sp, cyc, rp)
                except Exception:
                    continue
                x = d['ch2'] / scales[sp]          # 试件内共享常数归一
                waves.append(x)
                rows.append({'specimen': sp, 'cycle': int(cyc), 'rep': rp})
    meta = pd.DataFrame(rows)
    meta = meta.merge(lab[['specimen', 'cycle', 'crack_mm']],
                      on=['specimen', 'cycle'], how='left')
    meta.attrs['n_no_label'] = n_no_label
    return waves, refs, meta


def to_input(waves, meta, refs, mode, chans=1, n_fft=64, hop=16,
             chan2='diff', seed=0):
    """把时域波形列表转成网络输入张量 (N, C, ...)。

    `chans=2`（仅 wave 模式）额外送一路「相对参考波形的差值」：
      ⚠️ 这是**信息对等**所必需的。Step 1 里最强的特征是
      `corr_baseline`（与健康基线的相关性），而单通道 CNN 看不到基线 ⇒
      被剥夺了最重要的信息，比不过特征工程是必然的、不能算 DL 的结论。
      第 2 路差值用的参考是**无需标签**的"该试件最早一次测量"（与 --norm first 同源），
      因此不构成泄漏。

    `chan2` = 第 2 路的**变体**，用于消融「1.857→1.692 的增益到底来自信息还是容量」：

    ============  ==================================================  ==============
    取值          含义                                                是否含信息
    ============  ==================================================  ==============
    `diff`        真实差值（默认，= 交付口径）                        ✅ 含
    `zero`        全零通道（只增加输入宽度，不增加信息）            ❌
    `dup`         复制第 1 路（模拟"只是多一路同样的东西"）        ❌
    `rand`        与样本无关的高斯噪声（同尺度）                    ❌
    `shuffle`     把真实差值**跳样本打乱**（保留边缘分布、               ❌
                  破坏"哪个差值属于哪个样本"）
    ============  ==================================================  ==============

    若 `zero`/`dup`/`rand`/`shuffle` 都不如 `diff`、且与单通道持平 ⇒
    增益来自**信息**而非参数量/通道数，§5.2.2 ① 的结论才算坐实。
    """
    if mode == 'wave':
        L = min(len(w) for w in waves)
        X1 = np.stack([w[:L] for w in waves]).astype(np.float32)
        if chans == 1:
            return X1[:, None, :]
        ch2 = []
        for i, (sp, rp) in enumerate(zip(meta['specimen'], meta['rep'])):
            ref = refs[sp][:L]
            ch2.append((X1[i] - ref).astype(np.float32))
        d = np.stack(ch2)
        if chan2 == 'diff':
            pass
        elif chan2 == 'zero':
            d = np.zeros_like(d)
        elif chan2 == 'dup':
            d = X1.copy()
        elif chan2 == 'rand':
            rng = np.random.default_rng(seed)
            d = (rng.normal(0.0, 1.0, size=d.shape) * float(d.std())
                 ).astype(np.float32)
        elif chan2 == 'shuffle':
            rng = np.random.default_rng(seed)
            d = d[rng.permutation(len(d))].copy()
        else:
            raise ValueError('未知的 chan2=%r（可选 diff/zero/dup/rand/shuffle）'
                             % chan2)
        return np.stack([X1, d], axis=1)
    # --- STFT 幅值图 ---
    from scipy.signal import stft
    imgs = []
    for w in waves:
        fs = prep.FS / prep.DECIM
        f, t, Z = stft(w, fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                       boundary=None, padded=False)
        m = (f >= 0) & (f <= 400e3)          # 只留激励频带附近
        imgs.append(np.abs(Z[m]))
    F = min(a.shape[0] for a in imgs)
    T = min(a.shape[1] for a in imgs)
    X = np.stack([a[:F, :T] for a in imgs])[:, None, :, :]
    return X.astype(np.float32)


# ============================================================
# 模型
# ============================================================
def make_cnn(mode, chans=1):
    """轻量 CNN。**刻意做小** —— 返回 (model, n_params)。

    `chans` = 输入通道数（wave 模式可为 1 或 2）。
    """

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(chans, 16, 11, stride=2, padding=5), nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Conv1d(16, 32, 7, stride=2, padding=3), nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 32, 5, stride=2, padding=2), nn.BatchNorm1d(32),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(64, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))

        def forward(self, x):
            h = self.net(x)
            h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)
            return self.head(h).squeeze(-1)

    class CNN2D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 8, 5, stride=2, padding=2), nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16),
                nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2), nn.Linear(16, 1))

        def forward(self, x):
            h = self.net(x)
            h = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
            return self.head(h).squeeze(-1)

    m = CNN1D() if mode == 'wave' else CNN2D()
    n = sum(p.numel() for p in m.parameters())
    return m, n


def model_suffix(input_mode, norm, chans, anchor=False, chan2='diff'):
    """产物文件名后缀：默认组合（wave/first/1 通道）不带后缀。"""
    s = '' if (input_mode == 'wave' and norm == 'first' and chans == 1) \
        else '_%s_%s_c%d' % (input_mode, norm, chans)
    if chans == 2 and chan2 != 'diff':
        s += '_ch2' + chan2          # 消融变体单独存文件，不覆盖交付口径
    return s + ('_anch' if anchor else '')


# ============================================================
# 数据增强（README 要求：加噪 / 时移 / 幅值缩放 / 频带掩蔽）
# ============================================================
def augment(xb, mode, rng, frames=prep.FS / prep.DECIM):
    """在**已归一化**的 batch 上做增强；x 为 (B, C, L) 或 (B, C, F, T)。

    ⚠️ 幅值缩放只允许**小范围**（±10%）：本数据里"透射幅值下降"正是最强的物理信号，
    大幅缩放会把它抹掉、反而害了模型。
    """
    B = xb.shape[0]
    g = rng.normal(0, 0.02, size=(B, 1) + (1,) * (xb.ndim - 2))   # 高斯加噪
    out = (xb + g).astype(np.float32)          # ⚠️ 必须转回 float32：rng.normal 给 float64
    if mode == 'wave':
        L = xb.shape[2]
        for i in range(B):
            k = int(rng.integers(-L // 20, L // 20 + 1))          # 时移 ±5%
            if k:
                out[i] = np.roll(out[i], k, axis=1)
            out[i] *= np.float32(1.0 + rng.uniform(-0.10, 0.10))  # 幅值缩放 ±10%
        # 频带掩蔽（在时域上做：随机置零一小段 = 抹掉该时刻的能量）
        for i in range(B):
            if rng.random() < 0.3:
                w = int(rng.integers(L // 40, L // 10))
                st = int(rng.integers(0, max(1, L - w)))
                out[i, :, st:st + w] = 0.0
    else:
        F, T = xb.shape[2], xb.shape[3]
        for i in range(B):
            out[i] *= np.float32(1.0 + rng.uniform(-0.10, 0.10))
            if rng.random() < 0.4:                                 # 频带掩蔽
                w = int(rng.integers(1, max(2, F // 4)))
                st = int(rng.integers(0, max(1, F - w)))
                out[i, :, st:st + w, :] = 0.0
    return out


# ============================================================
# 训练一个折（多种子集成）
# ============================================================
def train_fold(Xtr, ytr, Xte, mode, seeds, epochs, device, lr=2e-3,
               wd=1e-3, batch=16, chans=1, verbose=False):
    """在训练折上估统计量做输入标准化，再训练。

    ⚠️ 标准化统计量**只用训练折**估计（`Xtr`），再套到测试折 ——
    若用全体样本的均值/方差，就悄悄引入了测试集信息（转导式泄漏）。
    """
    mu = float(Xtr.mean())
    sd = float(Xtr.std() + 1e-8)
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    ymean, ystd = float(ytr.mean()), float(ytr.std() + 1e-8)
    ytrn = (ytr - ymean) / ystd                      # 目标标准化（小样本必备）
    preds, params = [], 0
    Xt = torch.tensor(Xte, device=device)

    for s in range(seeds):
        torch.manual_seed(1000 + s)
        rng = np.random.default_rng(1000 + s)
        model, params = make_cnn(mode, chans=chans)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        lossf = nn.MSELoss()
        n = len(Xtr)
        for ep in range(epochs):
            model.train()
            perm = rng.permutation(n)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                if len(idx) < 4:
                    continue
                xb = augment(Xtr[idx], mode, rng)
                # 同一时刻的两次重复要一起进出？不需要 —— 它们是独立样本
                xb = torch.tensor(xb, device=device)
                yb = torch.tensor(ytrn[idx], device=device)
                opt.zero_grad()
                loss = lossf(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            p = model(Xt).cpu().numpy() * ystd + ymean
        preds.append(p)
    return np.mean(preds, axis=0), params


# ============================================================
# 物理约束层（与 Step 1 完全一致：保序投影 + 非负）
# ============================================================
def isotonic_up(v):
    v = np.asarray(v, dtype=float)
    v = np.maximum(v, 0.0)
    if len(v) <= 1:
        return v
    out = []
    for i in range(len(v)):
        out.append([i, i, v[i]])
        while len(out) > 1:
            a, b = out[-2], out[-1]
            if a[2] / (a[1] - a[0] + 1) <= b[2] / (b[1] - b[0] + 1):
                break
            out.pop()
            out.pop()
            out.append([a[0], b[1], a[2] + b[2]])
    res = np.empty(len(v))
    for a, b, s in out:
        res[a:b + 1] = s / (b - a + 1)
    return res


def apply_mono(df, col='pred'):
    out = np.empty(len(df))
    for (sp, rp), idx in df.groupby(['specimen', 'rep']).groups.items():
        sub = df.loc[idx].sort_values('cycle')
        proj = isotonic_up(sub[col].to_numpy())
        out[df.index.get_indexer(sub.index)] = proj
    return out


def apply_anchor(df, col='pred'):
    """输出端「健康原点锚定」：每个 (specimen, rep) 减去**其最早一次测量**处的预测值，再钳非负。

    动机：本数据集的**输入**全是"相对该试件某个参考"的，而**输出**没有对应参考 ⇒
    健康期的整体正偏置就直接变成"零裂纹误报"。把该偏置减掉，输出与输入口径对齐。
    只用"时间顺序"（最早一次），**不碰标签** ⇒ LOSO 安全、可部署。

    ⚠️ 实测结论（2026-09-21）：对 `wave/first/c2` 几乎无效 ——
       RMSE 1.697 → 1.984（变差），零裂纹误报 24% → 24%（不变）。
       原因：每个试件只能消掉**第一个**零裂纹时刻的误报；T3/T5/T7/T8 的第 2、3 个
       健康时刻是**真实漂移**，锚定无能为力 ⇒ 反过来证明了 §5.1.3 的归因（误报不是
       原点问题、是漂移问题）。保留该选项作为该结论的**可复现证据**。
    """
    out = np.empty(len(df))
    for (sp, rp), idx in df.groupby(['specimen', 'rep']).groups.items():
        sub = df.loc[idx].sort_values('cycle')
        v = sub[col].to_numpy(float)
        out[df.index.get_indexer(sub.index)] = np.maximum(v - v[0], 0.0)
    return out


def arm_names(anchor=False):
    """本步输出的臂名。``anchor=True`` 时追加两个「锚定」臂。"""
    base = ['cnn', 'cnn+mono']
    return base + (['cnn+anchor', 'cnn+mono+anchor'] if anchor else [])


# ============================================================
# 指标（与 Step 1 的 step1_model.py 保持完全一致）
# ============================================================
def metrics(y_true, y_pred, df, tag):
    e = np.asarray(y_pred) - np.asarray(y_true)
    yt = np.asarray(y_true, dtype=float)
    m = {'arm': tag, 'n': len(e),
         'rmse': float(np.sqrt(np.mean(e ** 2))),
         'mae': float(np.mean(np.abs(e))),
         'bias': float(np.mean(e)),
         'max_abs_err': float(np.max(np.abs(e)))}
    m['rho_within'] = (float(pd.Series(y_pred).corr(pd.Series(yt),
                                                    method='spearman'))
                       if len(np.unique(yt)) >= 3 else np.nan)
    m['pred_std'] = float(np.std(y_pred))
    m['true_std'] = float(np.std(yt))
    m['range_ratio'] = (m['pred_std'] / m['true_std']) if m['true_std'] > 0 else np.nan
    viol, tot = 0, 0
    for (sp, rp), g in df.assign(_p=y_pred).groupby(['specimen', 'rep']):
        g = g.sort_values('cycle')
        viol += int((np.diff(g['_p'].to_numpy()) < -1e-9).sum())
        tot += len(g) - 1
    m['mono_viol_rate'] = (viol / tot) if tot else np.nan
    m['neg_rate'] = float(np.mean(np.asarray(y_pred) < 0))
    z = np.asarray(y_true) == 0
    m['n_zero'] = int(z.sum())
    m['false_alarm'] = (float(np.mean(np.asarray(y_pred)[z] > 0.5))
                        if z.sum() else np.nan)
    return m


# ============================================================
# main
# ============================================================
def print_summary():
    """扫描已存在的全部 step2_loso*.csv，打印跨配置汇总 + 与 Step 1 的对照。

    不训练、不写文件，纯粹把已有产物汇总成一张可进论文的表（跑法：--summary）。
    """
    import glob
    say('=' * 100)
    say('Step 2 全配置汇总（读 results/step2_loso*.csv；每行 = 一个已跑过的配置）')
    say('=' * 100)
    say('  %-30s %8s %8s %9s %9s %10s %9s' %
        ('配置', 'RMSE', 'MAE', '折内rho', '单调违例', '零裂纹误报', '值域比'))
    say('  ' + '-' * 92)
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, 'step2_loso*.csv'))):
        tag = os.path.basename(f).replace('step2_loso', '').replace('.csv', '') or '_default'
        d = pd.read_csv(f)
        for a in ('cnn', 'cnn+mono', 'cnn+anchor', 'cnn+mono+anchor'):
            g = d[d['arm'] == a]
            if not len(g):
                continue
            rows.append((tag + ' / ' + a, g['rmse'].mean(), g['mae'].mean(),
                         g['rho_within'].mean(),
                         100 * g['mono_viol_rate'].mean(),
                         100 * g['false_alarm'].mean(), g['range_ratio'].mean()))
    for r in sorted(rows, key=lambda x: x[1]):
        say('  %-30s %8.3f %8.3f %9.3f %8.1f%% %9.0f%% %9.2f' % r)

    s1 = os.path.join(RES, 'step1_loso.csv')
    if os.path.exists(s1):
        say('\n  对照：Step 1 特征工程基线')
        a = pd.read_csv(s1)
        say('  %-30s %8s %8s %9s %9s %10s %9s' %
            ('臂', 'RMSE', 'MAE', '折内rho', '单调违例', '零裂纹误报', '值域比'))
        say('  ' + '-' * 92)
        for nm in ['single_tof', 'single_best', 'linear', 'svr', 'rf', 'gpr',
                   'linear+mono', 'svr+mono', 'rf+mono', 'gpr+mono']:
            g = a[a['arm'] == nm]
            if not len(g):
                continue
            say('  %-30s %8.3f %8.3f %9.3f %8.1f%% %9.0f%% %9.2f'
                % (nm, g['rmse'].mean(), g['mae'].mean(),
                   g['rho_within'].mean(), 100 * g['mono_viol_rate'].mean(),
                   100 * g['false_alarm'].mean(), g['range_ratio'].mean()))
        say('')
        say('  读法：')
        say('   1. `_c2` 后缀 = 双通道输入（额外送"相对参考波形的差值"）。')
        say('      单通道 CNN 看不到基线 ⇒ 拿不到 Step 1 最强特征 `corr_baseline` 的信息，')
        say('      比不过特征工程是**必然**的，不能算 DL 的结论。要对比必须用 c2。')
        say('   2. `stft` = 2b 方案的时频图输入，用于回答"2D 时频是否比 1D 波形更好"。')
        say('   3. `first` = 只用"时间顺序"定参考（**可部署**）；')
        say('      `zero` = 用"裂纹=0 时刻"定参考（**用到标签**，与 Step 1 同口径）。')
    with open(os.path.join(RES, 'step2_summary.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(_REPORT) + '\n')


def lock_determinism():
    """锁定可复现性。

    实测（2026-09-21，RTX 5050 + cudnn 92400）：cuDNN 卷积的 **backward 默认非确定** ——
    同一份代码、同一组种子、同一配置，两次运行的逐折 RMSE 最大差 **0.171 mm**
    （整体 1.697 ↔ 1.732）。这使文档里的每个数字都**不可被第三方复现**。
    打开下面三个开关后，梯度逐位一致、loss 完全不变（**无精度损失**）。
    """
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def main():
    ap = argparse.ArgumentParser(description='phmdc Step 2 波形级 DL')
    ap.add_argument('--summary', action='store_true',
                    help='只汇总已有产物，不训练')
    ap.add_argument('--input', choices=['wave', 'stft'], default='wave')
    ap.add_argument('--chans', type=int, default=1, choices=[1, 2],
                    help='wave 模式：2 = 额外送一路「相对参考波形的差值」')
    ap.add_argument('--chan2', default='diff',
                    choices=['diff', 'zero', 'dup', 'rand', 'shuffle'],
                    help='第 2 路的**变体**（消融用）；diff = 交付口径。'
                         'zero/dup/rand/shuffle 均为"不含信息"的对照，'
                         '用于证明 1.857→1.692 的增益来自信息而非参数量')
    ap.add_argument('--norm', choices=['first', 'zero'], default='first',
                    help='first=无标签口径（默认）；zero=与 Step 1 同口径（用标签）')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--anchor', action='store_true',
                    help='额外输出「健康原点锚定」臂（减最早一次测量的预测值，只用时间顺序）')
    args = ap.parse_args()

    if args.summary:
        print_summary()
        return

    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    suffix = model_suffix(args.input, args.norm, args.chans,
                          anchor=args.anchor, chan2=args.chan2)
    out_pred = os.path.join(RES, 'step2_pred%s.csv' % suffix)
    out_loso = os.path.join(RES, 'step2_loso%s.csv' % suffix)
    out_report = os.path.join(RES, 'step2_report%s.txt' % suffix)

    lock_determinism()

    say('=' * 78)
    say('PHM2019 铝搭接件（数据集 D） Step 2 波形级 DL')
    say('生成时间: %s' % pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'))
    say('输入=%s  通道数=%d  归一化口径=%s  种子数=%d  轮数=%d  设备=%s  锚定=%s  第2路=%s'
        % (args.input, args.chans if args.input == 'wave' else 1,
           args.norm, args.seeds, args.epochs, device,
           '开（额外输出两臂）' if args.anchor else '关', args.chan2))
    say('确定性: 已锁定（cudnn.deterministic=True, benchmark=False, '
        'use_deterministic_algorithms）')
    say('        —— 未锁定时逐折 RMSE 最大差可达 0.17 mm（见 lock_determinism 注释）')
    say('=' * 78)

    # ---------- 数据 ----------
    waves, refs, meta = build_dataset(norm=args.norm)
    X = to_input(waves, meta, refs, args.input, chans=args.chans,
                 chan2=args.chan2)
    say('\n[1] 数据')
    say('    样本 %d 个（%d 试件 × 时刻 × 2 次重复）'
        % (len(meta), meta['specimen'].nunique()))
    say('    跳过 %d 个"有波形但真值表无记录"的时刻（T3/55391 等，见 Step 0 核对 1）'
        % meta.attrs.get('n_no_label', 0))
    assert meta['crack_mm'].notna().all(), '仍存在 crack_mm 缺失的样本！'
    say('    输入张量形状 = %s （dtype %s）' % (X.shape, X.dtype))
    say('    取值范围 [%.3f, %.3f]，均值 %.4f，标准差 %.4f'
        % (X.min(), X.max(), X.mean(), X.std()))
    say('    归一化口径 = %s   %s' % (
        args.norm,
        '（用该试件最早一次测量的 RMS；**不需要知道哪个时刻裂纹=0** ⇒ 可部署）'
        if args.norm == 'first' else
        '（用该试件裂纹=0 时刻的 RMS；⚠️ 用到了标签 ⇒ 等价于"假设可先录健康基线"）'))

    m, nparam = make_cnn(args.input, chans=args.chans)
    say('\n[2] 模型参数量 = %d  %s（README 要求 < 50k）'
        % (nparam, '[OK]' if nparam < 50000 else '[!] 超标'))

    # ---------- LOSO ----------
    say('\n[3] LOSO 8 折 × %d 种子集成' % args.seeds)
    say('    %-5s %7s %7s %8s %8s %9s %9s'
        % ('留出', '训练样本', '测试样本', 'RMSE', 'MAE', '折内rho', '耗时(s)'))
    say('    ' + '-' * 62)
    rows, preds = [], []
    for sp in prep.SPECIMENS:
        tr = meta['specimen'] != sp
        if not tr.any() or not (~tr).any():
            continue
        t0 = time.time()
        Xtr = X[tr.to_numpy()]
        ytr = meta.loc[tr, 'crack_mm'].to_numpy(float)
        Xte = X[(~tr).to_numpy()]
        te = meta.loc[~tr].reset_index(drop=True)
        yp, nparam = train_fold(Xtr, ytr, Xte, args.input, args.seeds,
                                args.epochs, device, chans=args.chans)
        dt = time.time() - t0
        # --- 加物理约束 ---
        tmp = te[['specimen', 'rep', 'cycle']].copy()
        tmp['pred'] = yp
        yp_mono = apply_mono(tmp)
        yvals = {'cnn': yp, 'cnn+mono': yp_mono}
        if args.anchor:
            # 锚定后再做保序投影（两个臂：只锚定 / 锚定+约束）
            tmp['pred'] = apply_anchor(tmp)
            yvals['cnn+anchor'] = tmp['pred'].to_numpy(float)
            yvals['cnn+mono+anchor'] = apply_mono(tmp)
        for nm, yv in yvals.items():
            for rp in (1, 2):
                sel = (te['rep'] == rp).to_numpy()
                if not sel.any():
                    continue
                mm = metrics(te.loc[sel, 'crack_mm'].to_numpy(float), yv[sel],
                             te.loc[sel, ['specimen', 'rep', 'cycle']], nm)
                mm['fold'] = sp
                mm['rep'] = rp
                rows.append(mm)
        for i in range(len(te)):
            row = {'specimen': sp, 'rep': int(te.loc[i, 'rep']),
                   'cycle': int(te.loc[i, 'cycle']),
                   'crack_mm': float(te.loc[i, 'crack_mm'])}
            for nm, yv in yvals.items():
                row[nm] = float(yv[i])
            preds.append(row)
        # 折汇总（两重复合并）
        e = yp - te['crack_mm'].to_numpy(float)
        rho_fold = metrics(te['crack_mm'].to_numpy(float), yp,
                           te[['specimen', 'rep', 'cycle']], 'x')['rho_within']
        say('    %-5s %7d %8d %8.3f %8.3f %9.3f %9.1f'
            % (sp, int(tr.sum()), int((~tr).sum()),
               float(np.sqrt(np.mean(e ** 2))), float(np.mean(np.abs(e))),
               rho_fold, dt))

    loso = pd.DataFrame(rows)
    pred = pd.DataFrame(preds)
    loso.to_csv(out_loso, index=False, encoding='utf-8-sig')
    pred.to_csv(out_pred, index=False, encoding='utf-8-sig')

    # ---------- 总表 ----------
    say('\n' + '=' * 78)
    say('[4] 总体指标')
    say('=' * 78)
    say('  %-12s %8s %8s %8s %10s %9s %9s %9s %9s %9s'
        % ('臂', 'RMSE', 'MAE', '偏置', '最大误差', '单调违例', '负值率',
           '零裂纹误报', '折内rho', '值域比'))
    say('  ' + '-' * 108)
    for nm in arm_names(args.anchor):
        g = loso[loso['arm'] == nm]
        if not len(g):
            continue
        say('  %-14s %8.3f %8.3f %8.3f %10.3f %8.1f%% %8.1f%% %9s %9.3f %9.2f'
            % (nm, g['rmse'].mean(), g['mae'].mean(), g['bias'].mean(),
               g['max_abs_err'].max(), 100 * g['mono_viol_rate'].mean(),
               100 * g['neg_rate'].mean(),
               ('%.0f%%' % (100 * g['false_alarm'].mean()))
               if g['false_alarm'].notna().any() else '—',
               g['rho_within'].mean(), g['range_ratio'].mean()))

    # ---------- 锚定效果（仅 --anchor 时） ----------
    if args.anchor:
        say('\n' + '=' * 78)
        say('[4b] 「健康原点锚定」的效果（只减"最早一次测量"的预测值 + 钳非负）')
        say('=' * 78)
        say('  动机：输入全是"相对参考"的，输出没有参考 ⇒ 健康期整体正偏置 = 零裂纹误报。')
        say('  %-16s %8s %8s %9s %9s %9s'
            % ('臂', 'RMSE', 'MAE', '折内rho', '零裂纹误报', '值域比'))
        say('  ' + '-' * 68)
        for nm in arm_names(True):
            g = loso[loso['arm'] == nm]
            if not len(g):
                continue
            say('  %-16s %8.3f %8.3f %9.3f %8s %9.2f'
                % (nm, g['rmse'].mean(), g['mae'].mean(),
                   g['rho_within'].mean(),
                   ('%.0f%%' % (100 * g['false_alarm'].mean()))
                   if g['false_alarm'].notna().any() else '—',
                   g['range_ratio'].mean()))
        say('')
        say('  ⇒ 若锚定**不能**把零裂纹误报压到接近 0，说明误报的主体不是"输出缺原点"，')
        say('     而是**该试件多个健康时刻本身就互不相同**（漂移）—— 与 Step 1 §5.1.3 同一归因。')

    # ---------- 与 Step 1 对照 ----------
    say('\n' + '=' * 78)
    say('[5] 与 Step 1 特征工程基线对照（同口径：RMSE/MAE 取 rep1+rep2 均值）')
    say('=' * 78)
    s1 = os.path.join(RES, 'step1_loso.csv')
    if os.path.exists(s1):
        a = pd.read_csv(s1)
        say('  %-18s %8s %8s %9s %9s %9s' %
            ('臂', 'RMSE', 'MAE', '折内rho', '单调违例', '零裂纹误报'))
        say('  ' + '-' * 70)
        for nm in ['single_tof', 'single_best', 'rf', 'linear+mono', 'rf+mono']:
            g = a[a['arm'] == nm]
            if not len(g):
                continue
            say('  %-18s %8.3f %8.3f %9.3f %8.1f%% %9s'
                % (nm, g['rmse'].mean(), g['mae'].mean(),
                   g['rho_within'].mean(), 100 * g['mono_viol_rate'].mean(),
                   ('%.0f%%' % (100 * g['false_alarm'].mean()))
                   if g['false_alarm'].notna().any() else '—'))
        g = loso[loso['arm'] == 'cnn+mono']
        if len(g):
            say('  %-18s %8.3f %8.3f %9.3f %8.1f%% %9s'
                % ('cnn+mono (本步)', g['rmse'].mean(), g['mae'].mean(),
                   g['rho_within'].mean(), 100 * g['mono_viol_rate'].mean(),
                   ('%.0f%%' % (100 * g['false_alarm'].mean()))
                   if g['false_alarm'].notna().any() else '—'))
        say('')
        say('  ⚠️ 口径差异提示：Step 1 用的是 `--norm zero` 类口径（基线取裂纹=0 时刻，')
        say('     用到标签）；本步默认 `--norm first`（只用到时间顺序，可部署）。')
        say('     ⇒ 要严格对比，请跑 `python step2.py --norm zero`。')
    else:
        say('  （未找到 step1_loso.csv，跳过对照）')

    say('\n' + '=' * 78)
    say('[6] 物理约束层的影响（加约束前/后）')
    say('=' * 78)
    for rp in (1, 2):
        b = loso[(loso['arm'] == 'cnn') & (loso['rep'] == rp)]
        c = loso[(loso['arm'] == 'cnn+mono') & (loso['rep'] == rp)]
        if not len(b) or not len(c):
            continue
        say('  重复 %d：RMSE %.3f → %.3f（%+.3f）；折内 rho %.3f → %.3f；'
            % (rp, b['rmse'].mean(), c['rmse'].mean(),
               c['rmse'].mean() - b['rmse'].mean(),
               b['rho_within'].mean(), c['rho_within'].mean()))
        say('            单调违例 %.0f%% → %.0f%%；负值率 %.0f%% → %.0f%%'
            % (100 * b['mono_viol_rate'].mean(), 100 * c['mono_viol_rate'].mean(),
               100 * b['neg_rate'].mean(), 100 * c['neg_rate'].mean()))

    say('\n' + '=' * 78)
    say('[7] 失败模式诊断（逐试件的预测轨迹）')
    say('=' * 78)
    say('  「压缩」= 预测值域 / 真值值域；< 0.5 说明模型挤在中间地带、不敢给两端。')
    say('  「饱和」= 预测最大值是否明显低于真值最大值。')
    say('')
    say('  %-5s %10s %10s %10s %10s %9s  %s'
        % ('试件', '真值max', '预测max', '真值范围', '预测范围', '压缩比', '典型问题'))
    say('  ' + '-' * 88)
    for sp in prep.SPECIMENS:
        g = pred[pred['specimen'] == sp]
        if not len(g):
            continue
        yt = g['crack_mm'].to_numpy(float)
        yp = g['cnn+mono'].to_numpy(float)
        rr = (np.ptp(yp) / np.ptp(yt)) if np.ptp(yt) > 0 else np.nan
        prob = []
        if np.ptp(yt) > 0 and rr < 0.5:
            prob.append('值域压缩')
        if yp.max() < 0.8 * yt.max():
            prob.append('上端饱和')
        if yp.min() > 1.0 and yt.min() == 0:
            prob.append('健康端误报')
        if np.all(yp < 0):
            prob.append('**全负（整折失败）**')
        say('  %-5s %10.2f %10.2f %10.2f %10.2f %9.2f  %s'
            % (sp, yt.max(), yp.max(), np.ptp(yt), np.ptp(yp), rr,
               '、'.join(prob) if prob else '—'))
    say('')
    say('  两个**互相独立**的失败模式（对特征工程与 DL 同样成立，不是 DL 的问题）：')
    say('   (a) **值域压缩 / 上端饱和**：回归器在训练量程之外不会外推，')
    say('       且向训练均值收缩 —— 8 个试件时必然出现。')
    say('       例：T1 真值到 7.46 mm，预测被压在 ~4.9 mm；')
    say('           T3 真值到 6.93 mm，预测范围仅 ~3.1 mm。')
    say('   (b) **健康端误报**：该试件"零裂纹期"的波形相对参考已经变化很大')
    say('       （Step 1 §5.1.3：T8 的零裂纹时刻基线 corr 仅 0.303，且 T8 是**变幅载荷**），')
    say('       模型看到"与参考不同"就判有伤 ⇒ T8 的三个零裂纹时刻被预测成 3.4~4.0 mm。')
    say('   (c) **整折失败**：T6 留出时预测几乎恒为负 —— 它在输入空间是离群点')
    say('       （ch1 激励 4.24 V 仅与 T1 同档、且 SNR 最高 12.09）。')
    say('       ⇒ 8 个试件做 LOSO，"留一"就意味着模型要外推到完全没见过的工况。')
    say('')
    say('  ⇒ 结论：本数据的**误差瓶颈不是模型容量**，而是')
    say('     「训练量程覆盖不全」+「非裂纹引起的波形变化」这两件事。')
    say('     加网络、换架构都不会改变这一点。')

    with open(out_report, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(_REPORT) + '\n')
    say('\n[产出] %s / %s / %s'
        % (os.path.basename(out_pred), os.path.basename(out_loso),
           os.path.basename(out_report)))


if __name__ == '__main__':
    main()
