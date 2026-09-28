# -*- coding: utf-8 -*-
"""AE (`.pridb`) 统一读取 —— 含**多段会话缝合**修复。

## 问题（详见 docs/details.md §17.11）

`.pridb` 的 `Time` 是**各次采集自己的起算秒数**（采集中断后重开会从 ~0 重算），
不是墙钟。若一个试件有多个 `.pridb`，**直接按 `Time` 拼接会把两个会话交错**：

- `l1/step0.py` 旧行为：`concat` → `drop_duplicates(subset='time')` → `sort_values('time')`
  （注释写「可能时间有重叠」⇒ 作者以为有重复，但实测两段**零重合**，去重无效）
- `l1/step0_v2.py` / `ae_locate.py` / `ae_raf.py` / `ae_shape_features.py`：
  `sorted(glob(...))` + 按 `Time` 排序（同样交错）

## 实测两组有多段

| 组 | 段 | 命中数 | `Time` 范围 | 事件率走势 |
| -- | -- | -----: | ----------- | ---------- |
| **L1-04** | `L1-04.pridb` | 159,477 | 0 → 99,355 s (27.5 h) | **递减** ⇒ 早期 |
| | `L1-04-2.pridb` | 3,987,205 | 292 → 141,627 s (39.3 h) | **递增** ⇒ 后期 |
| **L1-59** | `L159.pridb` | 3,067 | 112 → 949 s (0.2 h) | 碎片 |
| | `L159-2.pridb` | 5,613,405 | 152 → 153,333 s (42.6 h) | 主记录 |

L1-04 两段事件**零重合** ⇒ 不是冗余导出，是**两段独立会话**。
按 `Time` 拼接的后果：AE 只覆盖 cycle **61.3%**，且后期高活跃段被映射到
cycle 110k–161k，**与论文 "little AE activity in the first 240,000 cycles" 直接矛盾**。

**按会话顺序缝合**（第 i 段平移前 i−1 段的总跨度）后：
覆盖 **99.9%**；后期高活跃段落到 **cycle 240k–280k**，与论文 *"Between 260,000
cycles and final failure, the activity of cluster 1 accumulates"* **精确吻合**；
相对论文 n_DB 的时段富集 **0.00× → 3.74×**。

## 判据（`plan()`）

- **A** 各段 `Time` 范围**互不重叠** ⇒ 同一/延续时钟 ⇒ 直接按 `Time` 合并
- **B** 有重叠，但存在**碎片段**（时长 < 最长段的 20%）⇒ 碎片不构成会话
  ⇒ 按 `Time` 合并（并告警）。用**时长**而非事件数判定：L1-04 的第 1 段只占
  总事件数的 3.8%，用事件数阈值会把它误判为碎片。
- **C** 有重叠且各段都是实质录制 ⇒ **按会话顺序缝合**

L1-59 命中判据 B（`L159.pridb` 0.2 h / 42.6 h = 0.5%）⇒ 按 `Time` 合并；
其 3,067 个事件占 0.05%，对任何结论无影响。

## 会话序号

不能按文件名排序：`'L1-04-2.pridb' < 'L1-04.pridb'`（`'-'`=45 < `'.'`=46）。
故按去掉 `gid` 前缀后的尾部数字排序（无数字 = 第 1 段）。
另注 L1-59 的文件名去掉了短横（`L159.pridb`），两种写法都要支持。

## 单位

`.pridb` 的 `ae_fieldinfo` 表给出换算式（L1-03 实查）:
`Time[s]×1e-7`、`RiseT[µs]×0.1`、`Dur[µs]×0.1`、`Amp[µV]` 原始（**是 µV 不是 dB**，
dB = 20·log₁₀µV）、`Eny[eu]` 原始、`Counts` 无量纲、`RMS[µV]×0.0065536`。
**本模块默认只把 `Time` 换成秒**（缝合必须在秒域做），其余列**保持 `.pridb` 原值**
以免改变下游脚本（`ae_raf` 的 RA/AF 阈值语义等）的既有行为。
"""
import glob
import os
import re
import sqlite3

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETTYPE_HIT = 2
TIME_BASE = 1e7             # .pridb Time 单位: 100 ns
DUR_FRAC = 0.20             # 判据 B: 时长短于最长段该比例者视为碎片
DEFAULT_COLS = ('Time', 'Chan', 'Amp', 'RiseT', 'Dur', 'Eny', 'Counts')


def _part_no(fp, gid):
    """由文件名推断会话序号（无后缀 = 1）。支持 `L1-04-2` 与 `L159-2` 两种写法。"""
    base = os.path.splitext(os.path.basename(fp))[0]
    for pre in (gid, gid.replace('-', '')):
        if base.startswith(pre):
            rest = base[len(pre):]
            if rest == '':
                return 1
            m = re.match(r'^-?(\d+)$', rest)
            if m:
                return int(m.group(1))
    return 1


def pridb_parts(gid, root=None):
    """返回该组的 `.pridb` 列表，**按会话序号排序**。"""
    d = os.path.join(root or HERE, gid, 'AE')
    fs = sorted(glob.glob(os.path.join(d, '*.pridb')))
    return sorted(fs, key=lambda f: _part_no(f, gid))


def part_stats(gid, root=None, cols=None):
    """逐段的 (name, n_hits, t0_s, t1_s)。"""
    out = []
    for fp in pridb_parts(gid, root):
        con = sqlite3.connect(fp)
        n = con.execute(f'SELECT COUNT(*) FROM ae_data WHERE SetType={SETTYPE_HIT}'
                        ).fetchone()[0]
        mi, ma = con.execute(
            f'SELECT MIN(Time), MAX(Time) FROM ae_data '
            f'WHERE SetType={SETTYPE_HIT}').fetchone()
        con.close()
        out.append(dict(path=fp, name=os.path.basename(fp), n=int(n),
                        t0=float(mi or 0) / TIME_BASE,
                        t1=float(ma or 0) / TIME_BASE,
                        part=_part_no(fp, gid)))
    return out


def plan(gid, multifile='auto', root=None, verbose=True):
    """决定多段处理方式。返回 dict(mode, parts, offsets, reason)。

    mode: 'single'（只有一段）| 'time'（按 Time 合并）| 'stitch'（按会话顺序缝合）
    offsets: 每段要加的时间平移量 [s]（'stitch' 时非零）
    """
    st = part_stats(gid, root)
    if len(st) == 1:
        return dict(mode='single', parts=st, offsets=[0.0], reason='仅一段')
    if multifile == 'largest':
        best = max(st, key=lambda s: s['n'])
        return dict(mode='single', parts=[best], offsets=[0.0],
                    reason=f'用户指定 largest（丢弃 {sum(s["n"] for s in st) - best["n"]:,} 个事件）')
    if multifile == 'stitch':
        off, cur = [], 0.0
        for s in st:
            off.append(cur)
            cur = s['t1'] + cur
        return dict(mode='stitch', parts=st, offsets=off, reason='用户指定 stitch')
    if multifile == 'time':
        return dict(mode='time', parts=st, offsets=[0.0] * len(st),
                    reason='用户指定 time')

    # --- auto ---
    ov = any(max(a['t0'], b['t0']) < min(a['t1'], b['t1'])
             for i, a in enumerate(st) for b in st[i + 1:])
    if not ov:
        return dict(mode='time', parts=st, offsets=[0.0] * len(st),
                    reason='各段 Time 范围互不重叠 ⇒ 同一/延续时钟')
    dmax = max(s['t1'] - s['t0'] for s in st)
    frags = [s for s in st if (s['t1'] - s['t0']) < DUR_FRAC * dmax]
    if frags:
        return dict(mode='time', parts=st, offsets=[0.0] * len(st),
                    reason=('存在碎片段 '
                            + ', '.join(f'{s["name"]}({s["t1"]-s["t0"]:.0f}s, '
                                        f'{s["n"]:,}命中)' for s in frags)
                            + f' —— 时长 < 最长段的 {DUR_FRAC:.0%} ⇒ 不构成会话，'
                              '按 Time 合并'))
    off, cur = [], 0.0
    for s in st:
        off.append(cur)
        cur = s['t1'] + cur
    return dict(mode='stitch', parts=st, offsets=off,
                reason=(f'{len(st)} 段 Time 范围重叠且均为实质录制 '
                        f'⇒ 按会话顺序缝合（平移 ['
                        f'{", ".join(f"{o:.0f}" for o in off)}] s）'))


def describe(gid, multifile='auto', root=None):
    """人类可读的多段诊断（供各脚本打印）。"""
    p = plan(gid, multifile, root, verbose=False)
    lines = [f'[AE-IO] {gid}: {len(p["parts"])} 段 → mode={p["mode"]}；'
             f'{p["reason"]}']
    if len(p['parts']) > 1 or p['mode'] == 'single':
        for s, o in zip(p['parts'], p['offsets']):
            lines.append(f'    {s["name"]}: {s["n"]:>9,} 命中  '
                         f't {s["t0"]:.0f}→{s["t1"]:.0f} s  '
                         f'平移 {o:.0f} s  ⇒ 绝对 {s["t0"]+o:.0f}→{s["t1"]+o:.0f} s')
    return '\n'.join(lines)


def read_hits(gid, cols=DEFAULT_COLS, multifile='auto', chunk=500_000,
              root=None, verbose=True):
    """读逐 hit 表（SetType=2）→ ndarray，**第 0 列 = 缝合后的 Time [s]**，
    其余列 = `.pridb` **原值**（不做单位换算，见模块 docstring）。

    cols 必须包含 'Time'；返回列顺序与 cols 一致（Time 换成秒）。
    """
    cols = tuple(cols)
    if 'Time' not in cols:
        raise ValueError("cols 必须包含 'Time'")
    p = plan(gid, multifile, root, verbose=False)
    if verbose and (len(p['parts']) > 1 or p['mode'] != 'single'):
        print(describe(gid, multifile, root))
    files = [s['path'] for s in p['parts']]
    offs = p['offsets']
    sel = ', '.join(cols)
    parts = []
    for fp, off in zip(files, offs):
        con = sqlite3.connect(fp)
        cur = con.execute(f'SELECT {sel} FROM ae_data '
                          f'WHERE SetType={SETTYPE_HIT} ORDER BY Time')
        while True:
            rows = cur.fetchmany(chunk)
            if not rows:
                break
            parts.append(np.asarray(rows, np.float64))
        con.close()
        if off:
            parts[-1][:, cols.index('Time')] += off * TIME_BASE
    a = np.vstack(parts)
    if p['mode'] != 'stitch':          # 缝合模式下已按会话顺序拼接，不可再排
        a = a[np.argsort(a[:, cols.index('Time')], kind='stable')]
    ti = cols.index('Time')
    a[:, ti] = a[:, ti] / TIME_BASE
    return a, p


def read_hits_vallenae(gid, multifile='auto', root=None, verbose=True):
    """用 `vallenae` 读（保留其列名与单位），并按 `plan` 处理多段。

    供 `step0.py` 使用（它依赖 vallenae 的列名/单位来生成 `{gid}声发射.csv`，
    换成原始 sqlite 会改变下游 CSV 结构）。
    返回 (DataFrame, plan_info)；DataFrame 的 `time` 列已是**缝合后的秒数**。
    """
    import pandas as pd
    import vallenae as vae
    p = plan(gid, multifile, root, verbose=False)
    if verbose and (len(p['parts']) > 1 or p['mode'] != 'single'):
        print(describe(gid, multifile, root))
    frames = []
    for s, off in zip(p['parts'], p['offsets']):
        db = vae.io.PriDatabase(s['path'])
        df = db.read_hits()
        db.close()
        if df is None or len(df) == 0:
            continue
        if off:
            df = df.copy()
            df['time'] = df['time'] + off
        frames.append(df)
    if not frames:
        return None, p
    if len(frames) == 1:
        return frames[0], p
    out = pd.concat(frames, ignore_index=True)
    if p['mode'] != 'stitch':
        out = (out.drop_duplicates(subset='time').sort_values('time')
               .reset_index(drop=True))
    else:
        out = out.sort_values('time', kind='stable').reset_index(drop=True)
    return out, p
