# -*- coding: utf-8 -*-
"""L1 第二批试件（无 FBG）的**时间对齐**：AE ↔ DFOS ↔ cycle

问题（2026-09-14 实测）：
  - 旧组（L1-03/04/05/09）用 **FBG 块**（每 5000 cycles 测一次）作 cycle 锚：
    块 k ↔ cycle=(k+0.5)×5000，`reproduce_broer_l1.fbg_cycle_anchor()` 据此插值。
  - 第二批试件**没有 FBG** → 必须重建锚。

本模块的两条锚：
  A. **DFOS 测段锚**：ODiSi-B 每 500 cycles 测一次（PDF「Measurement Interval 500」），
     段 k ↔ cycle≈(k+0.5)×500。实测段结构：每段 ≈120 行 @1 Hz（≈119 s），段间 ≈373 s。
  B. **有效加载时间锚**（更稳健）：AE markers 记录了每次 Suspend/Resume 的**绝对墙钟**，
     → 可算出"扣除停机后的有效加载时长"；cycle = 有效时长 × f，
     f 由 n_f 校准（f = n_f / 总有效时长）。此法不依赖"每段恰好 500 cycles"。

实测校验（`--report`）会打印 f 的组间一致性：若各组 f 接近，则有效时间锚自洽。

用法:
  python l1_time_align.py --report                 # 打印各组对齐统计
  from l1_time_align import time_to_cycle, dfos_cycles
"""
import os
import re
import sqlite3
import pickle
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

GROUPS_V2 = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
             'L1-59', 'L1-60']
TIME_BASE = 1e7
_ABS_RE = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
_CYC_PER_SEG = 500.0          # PDF: ODiSi-B Measurement Interval = 500 cycles


# ---------------------------------------------------------------- markers
def _pridb(gid):
    d = os.path.join(ROOT, gid, 'AE')
    fl = sorted(f for f in os.listdir(d) if f.endswith('.pridb'))
    return os.path.join(d, fl[0]) if fl else None


def _marker_text(data):
    if isinstance(data, bytes):
        try:
            v = pickle.loads(data)
            return v if isinstance(v, str) else str(v)
        except Exception:                                        # noqa: BLE001
            return data.decode('latin1', 'replace')
    return '' if data is None else str(data)


def read_markers(gid):
    """返回 DataFrame(seg, kind, wall_s, ae_s)。

    Vallen **只在 Resume 后写绝对时间戳**，Suspend 无绝对时间（其后的 Data 为空）。
    故 Suspend 的墙钟由「上一个 Resume 的 (ae_s, wall_s) 锚 + 段内 1:1」推算。
    实测校验（L1-49）：Suspend@ae=3814.1 → 09:18:21 + 3814.1 s = 10:21:55，
    与 marker 文本 '10:21 Suspend' 吻合；末尾 Suspend@ae=80996.4 → 次日 09:21:10，
    与 '09:21 Suspend' 吻合。
    """
    fp = _pridb(gid)
    if fp is None:
        return pd.DataFrame(columns=['seg', 'kind', 'wall_s', 'ae_s'])
    con = sqlite3.connect(f'file:{fp}?mode=ro', uri=True)
    rows = con.execute('SELECT SetID, Data FROM ae_markers').fetchall()
    ev = []
    for sid, data in rows:
        txt = _marker_text(data)
        if not txt:
            continue
        m = _ABS_RE.search(txt)
        if m:
            if ev and ev[-1]['wall_text'] is None:
                ev[-1]['wall_text'] = m.group(1)
            continue
        if 'Resume' in txt:
            ev.append(dict(kind='Resume', sid=sid, wall_text=None))
        elif 'Suspend' in txt:
            ev.append(dict(kind='Suspend', sid=sid, wall_text=None))
    for e in ev:
        t = con.execute('SELECT MIN(Time) FROM ae_data WHERE SetID=?',
                        (e['sid'],)).fetchone()[0]
        e['ae_s'] = (t / TIME_BASE) if t is not None else np.nan
    con.close()
    anchor = None
    for e in ev:
        if e['wall_text']:
            e['wall_s'] = pd.Timestamp(e['wall_text']).value / 1e9
            anchor = (e['ae_s'], e['wall_s'])
        elif anchor is not None and np.isfinite(e['ae_s']):
            e['wall_s'] = anchor[1] + (e['ae_s'] - anchor[0])
        else:
            e['wall_s'] = np.nan
    df = pd.DataFrame(ev)
    if len(df):
        df = df[['kind', 'wall_s', 'ae_s']].reset_index(drop=True)
        df['seg'] = np.arange(len(df))
    return df


# ---------------------------------------------------------------- 有效加载时间
def eff_time_curve(gid):
    """返回 (wall_s[], eff_s[]) 单调递增的"有效加载时间"查表。

    有效时间 = Σ 每个 [Resume, Suspend) 区间的墙钟时长；
    区间外（Suspend→Resume 停机）有效时间冻结。
    末尾若无 Suspend（数据在试验中被截断），用最后一个 Resume 作端点。
    """
    mk = read_markers(gid)
    if mk.empty:
        return np.array([0.0]), np.array([0.0])
    res = mk[mk['kind'] == 'Resume']
    sus = mk[mk['kind'] == 'Suspend']
    pairs = []
    for _, r in res.iterrows():
        cand = sus[sus['wall_s'] > r['wall_s']]
        if len(cand):
            pairs.append((r['wall_s'], cand['wall_s'].iloc[0]))
    if not pairs:
        return np.array([0.0]), np.array([0.0])
    w0 = pairs[0][0]
    xs = [w0 - 1.0]
    ys = [0.0]
    acc = 0.0
    for a, b in pairs:
        acc += (b - a)
        xs.extend([a, b])
        ys.extend([acc, acc])
    xs.append(xs[-1] + 1.0)
    ys.append(acc)
    return np.asarray(xs, float), np.asarray(ys, float)


def effective_rate(gid, n_f):
    """f = n_f / 总有效加载时长 (Hz)。"""
    _, ys = eff_time_curve(gid)
    tot = float(ys[-1])
    return (n_f / tot if tot > 0 else np.nan), tot


def time_to_cycle(gid, ae_time_s, n_f, method='seg', clip_out_of_life=False):
    """AE 相对时间(s) → cycle。

    method='eff': cycle = 有效加载时间(该 AE 时刻) × f   （推荐）
    method='seg': cycle = 该时刻所属 DFOS 段的 (k+0.5)/n_ai × n_f

    clip_out_of_life=True 时，把**早于首个 AI（冲击后）测量段**的 AE 时刻返回 NaN。
    这些事件发生在冲击前（BI 期，如 L1-56 的 BI 跳 5 天），不属于疲劳寿命；
    若强行映射到 cycle≈0，会以大量噪声污染 OnlineDamageIndex 的早期背景估计
    → 真损伤事件反而被当背景丢掉（L1-56 的 e_ae 就是这样归零的）。
    默认 False = 保持历史行为，不影响既有脚本。
    """
    ae_time_s = np.atleast_1d(np.asarray(ae_time_s, float))
    if method == 'eff':
        mk = read_markers(gid)
        if mk.empty:
            return np.zeros_like(ae_time_s)
        # AE time → 墙钟（AE 在停机期间不累计，故段内 1:1）
        wall = np.full_like(ae_time_s, np.nan)
        res = mk[mk['kind'] == 'Resume'].sort_values('ae_s')
        for i, r in res.iterrows():
            nxt = res[res['ae_s'] > r['ae_s']]
            a_from = r['ae_s']
            a_to = nxt['ae_s'].iloc[0] if len(nxt) else np.inf
            m = (ae_time_s >= a_from) & (ae_time_s < a_to)
            if m.any():
                wall[m] = r['wall_s'] + (ae_time_s[m] - a_from)
        # 墙钟 → 有效时间 → cycle
        xs, ys = eff_time_curve(gid)
        eff = np.interp(wall, xs, ys, left=0.0, right=ys[-1])
        f, _ = effective_rate(gid, n_f)
        return np.clip(eff * f, 0, n_f)
    # --- 'seg': DFOS 段锚（默认，实测更可靠）---
    anc = dfos_anchor(gid, n_f=n_f, mode='uniform')
    if anc is None or anc.empty:
        return np.zeros_like(ae_time_s)
    mk = read_markers(gid)
    res = mk[mk['kind'] == 'Resume'].sort_values('ae_s')
    wall = np.full_like(ae_time_s, np.nan)
    for _, r in res.iterrows():
        nxt = res[res['ae_s'] > r['ae_s']]
        a_to = nxt['ae_s'].iloc[0] if len(nxt) else np.inf
        m = (ae_time_s >= r['ae_s']) & (ae_time_s < a_to)
        wall[m] = r['wall_s'] + (ae_time_s[m] - r['ae_s'])
    cyc = np.interp(wall, anc['wall_s'].to_numpy(float),
                    anc['cyc_seg'].to_numpy(float),
                    left=0.0, right=anc['cyc_seg'].to_numpy(float)[-1])
    if clip_out_of_life and 'phase' in anc.columns:
        ph = anc['phase'].astype(str).str.upper().to_numpy()
        is_ai = (ph == 'AI') | (ph == 'MAIN')
        if is_ai.any() and not is_ai.all():          # 确有 BI 段时才需要剔除
            ai_wall0 = float(anc['wall_s'].to_numpy(float)[is_ai][0])
            cyc = np.where(np.isfinite(wall) & (wall < ai_wall0), np.nan, cyc)
    return cyc


# ---------------------------------------------------------------- DFOS 段锚
def dfos_anchor(gid, n_f=None, mode='uniform'):
    """DFOS 段表。列: seg, phase, wall_s, rel_s, n_rows, cyc_seg。

    cyc_seg（2026-09-14 修正，含 BI 段）：
      - **AI（After Impact，冲击后）**：(k_ai+0.5)/n_ai × n_f（k_ai 仅在 AI 内计数）；
      - **BI（Before Impact，冲击前）**：**−1**（不属于寿命，仅供健康基线）；
      - 无 phase 列（旧产物）：全段均匀校准。
    mode='fixed500' 时用 (k+0.5)×500（实测不可靠，仅作对照）。
    """
    fp = os.path.join(ROOT, gid, f'{gid}dfos_anchor.csv')
    if not os.path.exists(fp):
        return None
    df = pd.read_csv(fp, encoding='utf-8-sig')
    # 带微秒的字符串会被 pandas 解析为 datetime64[us]，必须显式转 ns 再取秒
    df['wall_s'] = (pd.to_datetime(df['wall_iso'])
                    .to_numpy(dtype='datetime64[ns]').astype('int64') / 1e9)
    if mode == 'fixed500' or not n_f:
        df['cyc_seg'] = (df['seg'] + 0.5) * _CYC_PER_SEG
        return df
    if 'phase' in df.columns:
        p = df['phase'].astype(str).str.upper().to_numpy()
        # 'main' = 无 BI/AI 标记的组（如 L1-52/L1-54 只有单文件）→ 视为冲击后，全程计入寿命
        is_ai = (p == 'AI') | (p == 'MAIN')
    else:
        is_ai = np.ones(len(df), bool)
    cyc = np.full(len(df), -1.0)
    n_ai = int(is_ai.sum())
    if n_ai:
        cyc[is_ai] = (np.arange(n_ai) + 0.5) / n_ai * float(n_f)
    df['cyc_seg'] = cyc
    return df


def dfos_cycles(gid, n_f, min_rows=30, mode='uniform', include_bi=False):
    """DFOS 每段的 cycle（供 D(t) / RUL 用）。返回 (cyc, keep_mask)。

    默认只返回 **AI（冲击后）** 段；`include_bi=True` 时也含 BI（cyc = −1）。
    """
    anc = dfos_anchor(gid, n_f=n_f, mode=mode)
    if anc is None:
        return np.array([]), np.array([], bool)
    keep = (anc['n_rows'] >= min_rows).to_numpy()
    if not include_bi:
        keep &= anc['cyc_seg'].to_numpy(float) >= 0
    return anc['cyc_seg'].to_numpy(float), keep


# ---------------------------------------------------------------- report
def report(groups=None):
    from l1_meta import load_meta
    groups = groups or GROUPS_V2
    rows = []
    for g in groups:
        nf = load_meta(g)['n_f']
        mk = read_markers(g)
        f, tot = effective_rate(g, nf)
        anc = dfos_anchor(g, n_f=nf, mode='uniform')
        nseg = 0 if anc is None else len(anc)
        nseg_ok = 0 if anc is None else int((anc['n_rows'] >= 30).sum())
        n_bi = 0 if anc is None else int(
            (anc['cyc_seg'].to_numpy(float) < 0).sum())
        n_ai = nseg - n_bi
        ratio = (n_ai * _CYC_PER_SEG / nf) if (nf and n_ai) else np.nan
        # DFOS 覆盖度 = (DFOS 末段墙钟 − 试验起始墙钟) / 试验总墙钟
        cover = np.nan
        ws = mk['wall_s'].dropna() if len(mk) else pd.Series(dtype=float)
        if anc is not None and len(ws) >= 2 and ws.iloc[-1] > ws.iloc[0]:
            cover = (anc['wall_s'].iloc[-1] - ws.iloc[0]) / (ws.iloc[-1] - ws.iloc[0])
        # 段间隔规整度：CV 小 → "每段等值 cycle"的均匀锚可靠
        gap_cv = np.nan
        early_late = np.nan
        if anc is not None and len(anc) > 8:
            g_ = np.diff(anc['wall_s'].to_numpy(float))
            g_ = g_[g_ > 0]
            if len(g_) > 5:
                gap_cv = float(np.std(g_) / np.mean(g_))
                t3 = len(g_) // 3
                early_late = float(np.median(g_[-t3:]) / np.median(g_[:t3]))
        rows.append(dict(
            gid=g, n_f=nf, n_marker=len(mk),
            n_suspend=int((mk['kind'] == 'Suspend').sum()) if len(mk) else 0,
            eff_h=round(tot / 3600, 2) if np.isfinite(tot) else None,
            ae_cycle_rate=round(f, 3) if np.isfinite(f) else None,
            dfos_seg=nseg, seg_bi=n_bi, seg_ai=n_ai, seg_ge30=nseg_ok,
            seg500_over_nf=round(ratio, 3) if np.isfinite(ratio) else None,
            dfos_cover=round(cover, 3) if np.isfinite(cover) else None,
            gap_cv=round(gap_cv, 3) if np.isfinite(gap_cv) else None,
            late_over_early=round(early_late, 2) if np.isfinite(early_late) else None))
        print(f'[{g}] n_f={nf}  有效时长={rows[-1]["eff_h"]}h  '
              f'DFOS段={nseg}(BI={n_bi}/AI={n_ai}, ≥30行:{nseg_ok})  '
              f'AI×500/n_f={rows[-1]["seg500_over_nf"]}  覆盖={rows[-1]["dfos_cover"]}  '
              f'段间隔CV={rows[-1]["gap_cv"]}  后/前={rows[-1]["late_over_early"]}')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, 'l1_time_align.csv'), index=False, encoding='utf-8-sig')
    print('\n已存:', os.path.join(RES, 'l1_time_align.csv'))
    return df


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS_V2))
    ap.add_argument('--report', action='store_true')
    a = ap.parse_args()
    report(a.groups.split(','))
