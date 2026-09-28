# -*- coding: utf-8 -*-
"""候选试件可用性体检（main/ 中 016-020 之外的数据）。

背景
----
`main/001..027` 名义上都是全寿命，但实测只有一部分可用。本脚本**不改动正式配置**
（`shm/config.py` / `main/eval_common.py` 的 GROUPS），只对候选组做三项体检，
用于判断"能否纳入正式范围"：

  1. **时间列语义**：采样间隔 dt、总时长 —— 判断是否纯加载时间（含停机则不可用）
  2. **AE 口径**：AE 行数 / 时长 = 等效频率 —— 判断是否与应变同源同频
     （主样本 AE 无时间戳，靠 `np.linspace` 均匀映射，口径不一致则时间轴失真）
  3. **应变退化信号**：逐块循环幅值（`_AmpChannel` 自动判别解调状态）
     → 刚度损失率 `L = A / A_cal - 1`（`A_cal` = 校准段 [20,45) 块的 p30）
     → `L` 首达 5/10/30/50% 的寿命位置

⚠️ **刚度口径必须与 `main/grade_compare.py` 逐项对齐**（2026-09-15 修正，曾不一致）：

| 环节 | 官方口径（本脚本现已采用） | 曾经的错误做法 |
| --- | --- | --- |
| 校准段 | **固定块号 [20,45)**（与 `stiff_cal_lo/hi` 默认值一致） | 写成 `[20%,45%) × nblk` → 随 nb 而变 |
| L 取值 | `max(0, ·)`（与 `OnlineDamageIndex.stiff_loss` 一致） | 允许负值 |
| L 有效起点 | **第 45 块之后**（校准段未走完无基线可言，之前恒为 0） | 全序列 |
| 阈值判据 | **连续 `SUSTAIN` 块 ≥ 阈值，且从 `WARM_BLK` 块起扫** | 单块首次跳阈 → 在块 0 误触发 |

后果：016（nblk=100）因分子分母巧合相等而两者一致，掩盖了问题；
018/019/020 的阈值时刻曾分别报 1.4% / 0.0% / 1.7%，与官方
95.7% / 57.0% / 未达 **相差极大**。现版本对 016-020 会**自校验**并打印比对结果。

用法
----
    python main/candidate_check.py                       # 默认 015-027
    python main/candidate_check.py --groups 022,023,024,025
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from shm.damage_index import _AmpChannel      # noqa: E402

BLK = 500                 # 与主样本一致（EXT_BLOCK_PTS=500）
# 校准段用**固定块号**，与 OnlineDamageIndex 的 stiff_cal_lo/hi 默认值一致。
# 不能用“寿命比例 × nblk”：那会让校准窗口随试件块数漂移（曾经出错的原因）。
CAL_LO, CAL_HI = 20, 45
BASE_PCT = 30             # 低分位（与 stiff_base_pct 一致）
SUSTAIN = 3               # 连续 N 块 ≥ 阈值（与 grade_compare.SUSTAIN 一致）
WARM_BLK = 10             # 从第 N 块起扫（避开开机瞬态，与 grade_compare 一致）
TH = [0.05, 0.10, 0.30, 0.50]
SR_HZ = 10.0              # **采样率恒为 10 Hz**（数据方口径，全组一致）
OUT = os.path.join(HERE, 'results', 'candidate_check.csv')


def timebase(t):
    """识别第一列的时间语义 → (语义标签, 每点代表的秒数)。

    ⚠️ **不同导出工具写的第一列语义不同**（2026-09-20 实测）：

    | 表头 | 列名 | 实际语义 |
    | ---- | ---- | -------- |
    | 中文导出（016/023/024/025）  | `time` / `时间(s)` | **秒**，步长 0.1 |
    | 英文导出（018/019/020/026）  | `Time` / `时间`   | **整数计数器**，每行 +1（**不是秒**） |

    若把计数器当秒读 → 时长 **放大 10 倍**、AE 频率 **缩小 10 倍**（曾因此误报
    “018/026 是 1 Hz”，实际采样率一直恒为 10 Hz）。
    """
    if t.size < 11:
        return '?', np.nan
    seg = t[:5000]
    med = float(np.median(np.diff(seg)))
    ints = bool(np.all(np.isclose(seg, np.round(seg))))
    if ints and med >= 1.0:
        return '计数器(每点+1)', 1.0 / SR_HZ
    return '秒', med


def read_any(fp):
    """兼容多种编码读 CSV（主样本部分文件不是 UTF-8）。"""
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
        try:
            return pd.read_csv(fp, encoding=enc)
        except Exception:
            continue
    return None


def pick_amp_col(df):
    num = df.select_dtypes(include=[np.number])
    cand = [c for c in num.columns if '幅值' in str(c)]
    return (cand[0] if cand else num.columns[-1]), num


def strain_blocks(fp, blk=BLK):
    """逐块循环幅值 + 解调判别结果。"""
    df = read_any(fp)
    if df is None:
        return None, None
    col, num = pick_amp_col(df)
    v = num[col].to_numpy(float)
    ch = _AmpChannel(5000, 2.0)          # 与 OnlineDamageIndex 默认 demod 参数一致
    amps = []
    for k in range(len(v) // blk):
        for x in v[k * blk:(k + 1) * blk]:
            ch.add(float(x))
        amps.append(ch.block_amp())
    return np.asarray(amps, float), ch.mode()


def stiff_series(A):
    """块幅值 → 刚度损失率序列（严格复现 `OnlineDamageIndex.stiff_loss`）。

    两处必须照搬官方实现，否则阈值时刻会偏：
    1. **因果性**：校准段 [CAL_LO, CAL_HI) 未走完之前拿不到基线 → L 恒为 0；
    2. **单向**：只计正向增长（`max(0, ·)`），幅值下降不产生“负损伤”。
    """
    L = np.full(A.size, np.nan)
    base = np.nanpercentile(A[CAL_LO:CAL_HI], BASE_PCT) if A.size > CAL_HI else np.nan
    if not np.isfinite(base) or base <= 1e-9:
        return L, np.nan
    raw = A / float(base) - 1.0
    L[CAL_HI:] = np.maximum(raw[CAL_HI:], 0.0)      # 之前的块保持 NaN（等价于 0）
    return L, float(base)


def first_sustained(v, th, warm=WARM_BLK, n=SUSTAIN):
    """首个 life%（连续 n 块 ≥ th）—— 与 `grade_compare.first_sustained` 同义。"""
    v = np.where(np.isfinite(v), v, -np.inf)
    nb = v.size
    for i in range(warm, nb - n + 1):
        if (v[i:i + n] >= th).all():
            return 100.0 * i / nb
    return None


def analyse(gid):
    d = os.path.join(HERE, gid)
    r = {'gid': gid}
    sf = sorted(glob.glob(os.path.join(d, '*应变*.csv')))
    af = sorted(glob.glob(os.path.join(d, '*声发射*.csv')))
    t = None
    if sf:
        df = read_any(sf[0])
        t = df.iloc[:, 0].to_numpy(float)
        dt = np.diff(t)
        # 时长按**已知 10 Hz 采样率**算（列语义可能为计数器，不能拿列值当秒）
        r['t_kind'], r['dt_s'] = timebase(t)
        r['n_str'] = len(t)
        r['t_max_h'] = len(t) * r['dt_s'] / 3600.0 if np.isfinite(r['dt_s']) else np.nan
        r['dt_med'] = float(np.median(dt))        # 原始列步长（仅作语义诊断）
        r['dt_max'] = float(dt.max())
    if af:
        a = read_any(af[0])
        r['n_ae'] = len(a)
        if t is not None and np.isfinite(r.get('dt_s', np.nan)):
            r['ae_hz'] = len(a) / (len(t) * r['dt_s'])
    if sf:
        A, mode = strain_blocks(sf[0])
        if A is not None and A.size > 10:
            nb = A.size
            r['nblk'] = nb
            r['mode'] = mode
            L, base = stiff_series(A)
            raw = A / base - 1.0 if np.isfinite(base) and base > 1e-9 else np.full(nb, np.nan)
            r['amp_cal'] = base
            r['L_end'] = float(np.nanmax(L[-5:])) if np.isfinite(L[-5:]).any() else np.nan
            r['L_max'] = float(np.nanmax(L)) if np.isfinite(L).any() else np.nan
            r['L_raw_end'] = float(raw[-1])      # 未钳位末值：保留“幅值下降”这类异常信号
            for th in TH:
                t = first_sustained(L, th)
                r['t%.2f' % th] = round(t, 1) if t is not None else None
    return r


def self_check(rows):
    """与官方 `results/grade_stiff_traj.csv` 逐项对账（仅 016-020）。

    存在的意义：这两个表一旦又漂开，答辩时会被直接质疑。
    """
    fp = os.path.join(HERE, 'results', 'grade_stiff_traj.csv')
    if not os.path.exists(fp):
        return []
    g = read_any(fp)
    # ⚠️ CSV 里的 '016' 会被 pandas 读成整数 16 → 必须零填充，否则与 r['gid'] 比不中
    g['gid'] = g['gid'].astype(str).str.strip().str.zfill(3)
    g = g.set_index('gid')
    msgs = []
    for r in rows:
        gid = str(r['gid']).strip()
        if gid not in g.index or 'nblk' not in r:
            continue
        a, b = g.loc[gid], r
        bad = []
        for mine, theirs in (('L_max', 'stiff_max'), ('t0.05', 'x_0.05'),
                             ('t0.10', 'x_0.10'), ('t0.30', 'x_0.30'),
                             ('t0.50', 'x_0.50')):
            x, y = b.get(mine), a.get(theirs)
            x = None if x is None or (isinstance(x, float) and not np.isfinite(x)) else float(x)
            y = None if y is None or (isinstance(y, float) and not np.isfinite(y)) else float(y)
            if (x is None) != (y is None) or (x is not None and abs(x - y) > 0.15):
                bad.append('%s=%s vs %s=%s' % (mine, x, theirs, y))
        if bad:
            msgs.append('  [XX] %s 与官方口径不符：%s' % (gid, '；'.join(bad)))
        else:
            msgs.append('  [OK] %s 与官方口径一致（%d 块）' % (gid, b['nblk']))
    return msgs


def fmt(r):
    return ('%-5s 时长=%6.2fh 时间列=%-12s AE=%-8s AE频率=%-6s 块=%-4s 模式=%-6s '
            'L_end=%7.2f L_raw末值=%7.2f | L达阈(连续%d块): %s'
            % (r['gid'], r.get('t_max_h', -1), r.get('t_kind', '?'),
               r.get('n_ae', 'N/A'),
               ('%.2f' % r['ae_hz']) if 'ae_hz' in r else 'N/A',
               r.get('nblk', 'N/A'), r.get('mode', 'N/A'), r.get('L_end', np.nan),
               r.get('L_raw_end', np.nan), SUSTAIN,
               '  '.join('%.0f%%@%s' % (t * 100, r.get('t%.2f' % t, '-')) for t in TH)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups',
                    default='015,016,017,018,019,020,021,022,023,024,025,026,027')
    ap.add_argument('--no-check', action='store_true', help='跳过与官方刚度的对账')
    a = ap.parse_args()
    rows = []
    for g in [s.strip() for s in a.groups.split(',')]:
        if not os.path.isdir(os.path.join(HERE, g)):
            print('%-5s 目录不存在' % g)
            continue
        r = analyse(g)
        rows.append(r)
        print(fmt(r))
    if rows and not a.no_check:
        print('\n-- 与官方口径对账（result/grade_stiff_traj.csv）--')
        for m in self_check(rows):
            print(m)
    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        df.to_csv(OUT, index=False, encoding='utf-8-sig')
        print('\n已存', OUT)


if __name__ == '__main__':
    main()
