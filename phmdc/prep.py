"""
phmdc 共享预处理层 —— PHM2019 铝搭接件（数据集 D）
====================================================

`step0.py` / `step1.py` / 后续 Step 2 全部从这里取信号，**不各写一遍**
（依据：主项目教训「同一物理量在两个脚本里各写一遍，是这类项目最容易出的错」）。

三个关键决定（均由 Step 0 实测支撑，见 README §5.0）
----
1. **时间轴一律重建** `t[i] = i × 5.0e-8 s`
   文件里的 `time` 列被 Excel 量化为科学计数法且逐试件精度不同
   （T3 是 15 位有效数字，T6/68091 写作 `1E-07` 即相邻两点同值）
   ⇒ 92/94 个文件与重建轴偏差 > 半个采样间隔，不可用于 TOF / 时频分析。

2. **必须先带通** 到 `BAND_HZ = (100 kHz, 300 kHz)`
   激励是 **200 kHz 窄带 burst**（实测 ch1 峰频 200~210 kHz），但原始 ch2 记录
   被**带外噪声主导**：0~300 kHz 的能量占比 T6 为 91%、T1 为 63%，
   而 **T5 仅 9~40%**。逐带扫描（`step0_report.txt` 核对 8，判据 = 两次重复的相关性，
   **无真值指标**）结果：

   | 频带 | corr 中位 | corr 最小 | <0.9 的时刻 | relRMS 中位 | T5 |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | 原始（不滤波） | 0.9545 | 0.1620 | 10 | 33.5% | 0.298 |
   | **100~300 kHz** | **0.9985** | **0.9046** | **0** | **5.6%** | **0.983** |
   | 150~350 kHz | 0.9984 | 0.8356 | 1 | 5.9% | 0.982 |
   | 100~400 kHz | 0.9984 | 0.8069 | 1 | 5.8% | 0.980 |

   ⇒ README §5.0.2 曾把 T5 判为"噪声主导、不可用"，**该结论已被推翻**
     （不是试件坏，是没做带通）。**T5 保留**。
   ⇒ **噪声下界由 33.5% 降到 5.6%** —— 报"检出损伤"时以带通后的下界为准。

3. **幅值一律用相对量**
   ch2 被量化（每试件仅 108~244 个不同取值，步长 8e-4 ~ 2e-3 V），
   且 ch1 激励幅值**逐试件不同**（峰峰 1.88 ~ 4.28 V）
   ⇒ 绝对幅值不可跨试件比较，须走传递函数 H(f)=FFT(ch2)/FFT(ch1) 或"与自身基线之比"。

依赖：numpy / pandas / scipy
"""

import os
import functools

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, hilbert, windows

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------- 采集参数（实测，README §1.1） ----------
DT = 5.0e-8                 # 采样间隔 [s]
FS = 1.0 / DT               # 20 MHz
N_POINTS = 4000             # 每文件点数
T_MAX = (N_POINTS - 1) * DT  # 1.9995e-4 s

# ---------- 处理参数（Step 0 核对 8 逐带实测选定） ----------
BAND_HZ = (100e3, 300e3)    # 带通频带（围绕 200 kHz 激励；见 step0_report.txt 核对 8）
BP_ORDER = 4                # Butterworth 阶数
DECIM = 10                  # 降采样倍数（20 MHz -> 2 MHz，够用且省内存）

SPECIMENS = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8']
TRAIN = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
VALID = ['T7', 'T8']

INDEX_CSV = os.path.join(ROOT, 'index.csv')
LABELS_CSV = os.path.join(ROOT, 'labels.csv')


# ============================================================
# 基础 IO
# ============================================================
def path_of(specimen, cycle, rep):
    return os.path.join(ROOT, specimen, str(cycle), 'signal_%d.csv' % rep)


@functools.lru_cache(maxsize=256)
def load_raw(specimen, cycle, rep):
    """读原始波形 -> (t, ch1, ch2)。

    时间轴**重建**（不用文件里的 time 列，理由见模块 docstring ①）。
    列**按名取**（T6 有 9 个文件带尾部空列，按位置取会踩坑）。
    """
    df = pd.read_csv(path_of(specimen, cycle, rep))
    n = len(df)
    t = np.arange(n, dtype=np.float64) * DT
    return t, df['ch1'].to_numpy(dtype=np.float64), df['ch2'].to_numpy(dtype=np.float64)


# ============================================================
# 滤波 / 变换
# ============================================================
def bandpass(x, band=BAND_HZ, order=BP_ORDER):
    """零相位 Butterworth 带通（filtfilt）。

    零相位很重要：TOF 类特征要看**到达时刻**，一阶相位失真就会把 TOF 带偏。
    """
    lo, hi = band
    sos = butter(order, [lo / (FS / 2), hi / (FS / 2)], btype='band', output='sos')
    return sosfiltfilt(sos, x)


def envelope(x):
    """Hilbert 包络（解析信号幅值），也即瞬时幅度。"""
    return np.abs(hilbert(x))


def prep(specimen, cycle, rep, decim=DECIM):
    """一步到位：读原始 -> 带通 -> 降采样。

    返回 dict：
      t_pre   处理后时间轴 [s]（降采样后）
      ch1     带通后的激励
      ch2     带通后的接收
      env2    ch2 的包络
      pp_ch1  原始 ch1 峰峰值（用于诊断，不进模型）
    """
    t, ch1_raw, ch2_raw = load_raw(specimen, cycle, rep)
    ch1 = bandpass(ch1_raw)
    ch2 = bandpass(ch2_raw)
    env2 = envelope(ch2)
    out = {
        't_full': t, 'ch1': ch1, 'ch2': ch2, 'env2': env2,
        'ch1_raw': ch1_raw, 'ch2_raw': ch2_raw,
        'pp_ch1_raw': float(np.ptp(ch1_raw)),
        'pp_ch2_raw': float(np.ptp(ch2_raw)),
    }
    if decim and decim > 1:
        for k in ('ch1', 'ch2', 'env2'):
            out[k + '_full'] = out[k]
            out[k] = out[k][::decim]
        out['t'] = t[::decim]
        out['fs'] = FS / decim
    else:
        out['t'] = t
        out['fs'] = FS
    return out


def transfer_function(ch1, ch2, dt=DT, reg=1e-6):
    """传递函数 H(f) = FFT(ch2) / FFT(ch1)（README §5 Step 1 的核心做法）。

    目的：ch1 激励幅值逐试件不同（峰峰 1.88~4.28 V）⇒ 绝对幅值不可跨试件比较。
    正则化 `reg` 防止 ch1 谱谷附近的除零放大。

    ⚠️ `dt` 必须传**实际**采样间隔：本模块的 `prep()` 会降采样 `DECIM` 倍，
    若仍用全局 `DT` 会使频率轴错 `DECIM` 倍，频带选择随之全部落错
    （曾因此在 Step 1 的第一版里把 100~300 kHz 选成了 10~30 kHz）。
    """
    n = len(ch1)
    A = np.fft.rfft(ch1 - ch1.mean(), n)
    B = np.fft.rfft(ch2 - ch2.mean(), n)
    return np.fft.rfftfreq(n, dt), B / (A + reg * np.max(np.abs(A)))


# ============================================================
# 清单表
# ============================================================
@functools.lru_cache(maxsize=1)
def labels():
    """真值表：specimen, cycle, cycle_raw, crack_mm, source, note。"""
    return pd.read_csv(LABELS_CSV)


@functools.lru_cache(maxsize=1)
def index():
    """波形索引：specimen, cycle, rep, path, ..."""
    return pd.read_csv(INDEX_CSV)


def labeled_moments(include_no_wave=False):
    """返回「既有真值又有波形」的 (specimen, cycle, crack_mm) 表。

    include_no_wave=False 时剔除 T6/cycle=0（有标签无波形）与 T7/T8 的外推段标签。
    """
    lab = labels()
    idx = index()
    have = {(r.specimen, r.cycle) for r in idx.itertuples()}
    rows = []
    for r in lab.itertuples():
        if (r.specimen, r.cycle) in have:
            rows.append({'specimen': r.specimen, 'cycle': r.cycle,
                         'crack_mm': r.crack_mm, 'note': r.note})
        elif include_no_wave:
            rows.append({'specimen': r.specimen, 'cycle': r.cycle,
                         'crack_mm': r.crack_mm, 'note': 'no_waveform'})
    return pd.DataFrame(rows).sort_values(['specimen', 'cycle']).reset_index(drop=True)


def extrapolation_truth():
    """T7/T8 天然外推评测集：波形只给前半段、真值给到后半段的那些标签。"""
    lab, idx = labels(), index()
    have = {(r.specimen, r.cycle) for r in idx.itertuples()}
    out = []
    for sp in VALID:
        w = sorted(c for s, c in have if s == sp)
        for r in lab[lab['specimen'] == sp].itertuples():
            out.append({'specimen': sp, 'cycle': r.cycle, 'crack_mm': r.crack_mm,
                        'has_waveform': r.cycle in have,
                        'beyond_waveform_max': r.cycle > max(w)})
    return pd.DataFrame(out).sort_values(['specimen', 'cycle']).reset_index(drop=True)


# ============================================================
# 小工具
# ============================================================
def corr(x, y):
    """去均值皮尔逊相关。"""
    dx, dy = x - x.mean(), y - y.mean()
    den = np.sqrt(np.dot(dx, dx) * np.dot(dy, dy))
    return float(np.dot(dx, dy) / den) if den > 0 else np.nan


def rel_rms_diff(x, y):
    """相对 RMS 差 = ||x-y|| / ||x-mean||（以 x 的交流幅度为基准）。"""
    ac = np.sqrt(np.mean((x - x.mean()) ** 2))
    return float(np.sqrt(np.mean((x - y) ** 2)) / ac) if ac > 0 else np.nan
