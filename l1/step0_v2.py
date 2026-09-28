# -*- coding: utf-8 -*-
"""L1 第二批试件（L1-49..L1-56）预处理：AE + DFOS → 紧凑 CSV

与 step0.py 的差异（第二批试件**没有 FBG**、数据量大两个数量级）：
  1. LUNA 以**测量段**为单位聚合：相邻行 dt>GAP_S 视为新段；每段输出
     **段内逐位置中位数**（1 行 × n_pos），而非逐行落盘。
     实测数据事实（2026-09-14）：
       - 每段 ≈ 120 行 @1 Hz（≈119 s），段间间隔 ≈ 373 s
       - 空间网格 AI/BI 一致（同一光纤两端解调）
       - 用 pandas 分块流式读，避免 5.6 GB 级文件爆内存
  2. 空间降采样到 N_SPACE 点（默认 1500），列名保持 `{pos:.2f}mm`，
     与 evaluate_l1_dfos / reproduce_broer_l1 的 `load_dfos` 接口一致。
  3. AE 只读 SetType=2（实测这才是 hit 表；SetType=1 是时间戳索引、
     SetType=3 是参数集），按 **1 秒 bin** 聚合为
     `time / n_hits / energy(max) / amplitude(max)`，避免 4800 万行级 CSV。
     列名 `time` / `energy` 与旧流程 `usecols=['time','energy']` 兼容。
  4. 额外输出 `{gid}dfos_anchor.csv`（段序 / 段起始墙钟 / 段内行数），
     供 l1_time_align.py 建立 AE↔DFOS↔cycle 映射（新组无 FBG 锚）。

用法:
  python step0_v2.py                        # 全部新组
  python step0_v2.py --groups L1-49         # 单组
  python step0_v2.py --space 1500 --gap 60
"""
import os
import io
import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.abspath(__file__))

# 第二批试件（2026-09-14 新增）：无 FBG，AE + DFOS(ODiSi-B)
GROUPS_V2 = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
             'L1-59', 'L1-60']

GAP_S = 60.0          # 段间最小间隔（s）；实测段间 ≈373 s、段内 ≈1 s
N_SPACE = 1500        # 空间降采样点数
AE_BIN_S = 1.0        # AE 聚合 bin（s）
AE_CHUNK = 2_000_000  # sqlite 分批读取行数
TIME_BASE = 1e7       # .pridb Time 单位: 100 ns ticks
TS_SKIPROWS = 4       # LUNA 文件头 4 行（第 5 行为位置行）


# ============================================================
# LUNA / DFOS
# ============================================================
def luna_positions(fp):
    """读第 5 行（位置, mm）。返回 np.ndarray。"""
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        for _ in range(TS_SKIPROWS):
            f.readline()
        raw = f.readline().rstrip('\n')
    cells = raw.split('\t')[1:]
    vals = []
    for c in cells:
        try:
            vals.append(float(c))
        except ValueError:
            pass
    return np.asarray(vals, float)


def pick_cols(n_pos, n_keep):
    """等距抽取 n_keep 个空间索引（保持覆盖全长度）。"""
    if n_pos <= n_keep:
        return np.arange(n_pos)
    return np.unique(np.linspace(0, n_pos - 1, n_keep).round().astype(int))


def foot_mask_of(gid, pos_keep):
    """保留列中的「加强筋脚」掩码（用 l1_meta 从 PDF 解析的空间段）。

    用于判定段内**压缩峰值载荷行**：DFOS 行结构为 valley(卸载)/peak(最大压缩) 交替，
    取两脚应变均值**最负**的那一行（同 reproduce_broer_l1.dfos_binned_mean 口径）。
    两脚均缺时回退 None（改用全体位置均值）。
    """
    try:
        from l1_meta import load_meta
        m = load_meta(gid)
    except Exception as e:                                       # noqa: BLE001
        print(f'    [警告] l1_meta 不可用({e})，peak 行改用全体位置均值')
        return None
    fm = np.zeros(pos_keep.size, bool)
    for seg in (m.get('foot_L'), m.get('foot_R')):
        if seg:
            fm |= (pos_keep >= seg[0]) & (pos_keep <= seg[1])
    return fm if fm.any() else None


def _seg_stats(arr, fmask):
    """段内统计 → (中位分布, 压缩峰值行分布, 循环幅值分布)。

    amp = 逐位置 **(p90 − p10)**：ODiSi-B 采样 1 Hz vs 加载 2 Hz → **欠采样**，
    段内 120 行采到不同相位；取“最负单行”会被相位偶然主导
    （实测：peak 口径的寿命相关性 ρ = −0.40~0.33，比中位的 0.46~1.00 差很多）。
    故用**段内分布**的极差作为循环幅值 —— 与第一批 FBG 的「块内 std」同语义
    （载荷控制下 幅值 ∝ 1/刚度）。
    """
    med = np.nanmedian(arr, axis=0)
    score = (np.nanmean(arr[:, fmask], axis=1) if fmask is not None
             else np.nanmean(arr, axis=1))
    peak = (med.copy() if not np.isfinite(score).any()
            else arr[int(np.nanargmin(score))].copy())
    hi = np.nanpercentile(arr, 90, axis=0)
    lo = np.nanpercentile(arr, 10, axis=0)
    return med, peak, (hi - lo).astype(np.float32)


def luna_segments(fp, keep_idx, gap_s=GAP_S, fmask=None):
    """流式分段：返回 (t_start, t_end, prof_med, prof_peak, prof_amp, nrow)。

    prof_med  = 段内逐位置**中位数**（抗掉点）→ 空间分布 / DFOS HI 分析
    prof_peak = 段内**压缩峰值载荷行**分布 → 参考（单行口径，实测不如 amp）
    prof_amp  = 段内逐位置 **(p90−p10)** 循环幅值 → D(t)/RUL 应变证据首选
    用 pandas 分块读（C 解析器），避免 Python 逐行 float 解析。
    """
    n_pos = luna_positions(fp).size
    usecols = [0] + [i + 1 for i in keep_idx]
    t_start, t_end, nrow = [], [], []
    pmed, ppeak, pamp = [], [], []
    buf, t0, t1 = [], None, None

    def flush():
        if not buf:
            return
        arr = np.vstack(buf)
        m, p, a = _seg_stats(arr, fmask)
        t_start.append(t0); t_end.append(t1); nrow.append(len(buf))
        pmed.append(m); ppeak.append(p); pamp.append(a)

    reader = pd.read_csv(fp, sep='\t', skiprows=TS_SKIPROWS, header=0,
                         usecols=usecols, chunksize=4000, engine='c',
                         on_bad_lines='skip')
    for chunk in reader:
        ts_col = chunk.columns[0]
        tv = pd.to_datetime(chunk[ts_col], errors='coerce')
        tv = tv.to_numpy(dtype='datetime64[ns]').astype('int64') / 1e9
        vals = chunk.iloc[:, 1:].to_numpy(np.float32)
        for i in range(len(tv)):
            t = tv[i]
            if not np.isfinite(t):
                continue
            if t1 is not None and (t - t1) > gap_s:
                flush()
                buf = []
                t0 = t
            if t0 is None:
                t0 = t
            buf.append(vals[i])
            t1 = t
    flush()
    emp = np.zeros((0, len(keep_idx)), np.float32)
    return (np.asarray(t_start, float), np.asarray(t_end, float),
            np.asarray(pmed, np.float32) if pmed else emp,
            np.asarray(ppeak, np.float32) if ppeak else emp,
            np.asarray(pamp, np.float32) if pamp else emp,
            np.asarray(nrow, int))


def luna_files(gid):
    """返回 {阶段: [文件路径,...]}；阶段 ∈ {'AI','BI','main'}。

    **已核实（2026-09-14，据数据集官方描述 + 实测）**：
      AI = **A**fter **I**mpact（冲击后）、BI = **B**efore **I**mpact（冲击前）
      —— 而不是“同一光纤两端解调”（早期推断，已废止）。
    旁证：L1-49 的 BI 仅 65 min /1,293 行（对应 5000 cycles 预疲劳 + 测量暂停，
    其结束时刻即冲击时刻），AI 则长 21.5 h /11,449 行。
    故两组都缺一不可：BI 提供**真正的健康基线**，AI 提供损伤演化段。
    """
    d = os.path.join(ROOT, gid, 'LUNA')
    if not os.path.isdir(d):
        return {}
    out = {}
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith('.txt') or fn == 'LUNASetup.txt':
            continue
        up = fn.upper()
        if '_AI' in up:
            out.setdefault('AI', []).append(os.path.join(d, fn))
        elif '_BI' in up:
            out.setdefault('BI', []).append(os.path.join(d, fn))
        else:
            out.setdefault('main', []).append(os.path.join(d, fn))
    return out


def process_luna(gid, n_space=N_SPACE, gap_s=GAP_S):
    files = luna_files(gid)
    if not files:
        print(f'  [{gid}] 无 LUNA 数据，跳过')
        return None
    # 主阶段 = AI（冲击后）；**BI（冲击前）也纳入** → 恢复完整时间轴与健康基线
    #   旧版只取 AI 并跳过 BI，丢掉了冲击前基线（2026-09-14 修正）。
    key = 'AI' if 'AI' in files else ('main' if 'main' in files else 'BI')
    tasks = [(key, f) for f in files[key]]
    for k, v in files.items():
        if k != key:
            tasks += [(k, f) for f in v]
    pos = luna_positions(tasks[0][1])
    keep = pick_cols(pos.size, n_space)
    fmask = foot_mask_of(gid, pos[keep])
    phases = '/'.join(sorted({p for p, _ in tasks}))
    print(f'  [{gid}] LUNA 阶段={phases} 文件数={len(tasks)} '
          f'空间 {pos.size}→{keep.size} 点  脚位掩码点={0 if fmask is None else int(fmask.sum())}')

    T, TEND, PM, PP, PA, N, PH = [], [], [], [], [], [], []
    for phase, fp in tasks:
        p2 = luna_positions(fp)
        if p2.size != pos.size or not np.allclose(p2, pos, atol=1e-6):
            print(f'    [警告] {os.path.basename(fp)} 空间网格不一致，跳过')
            continue
        ts, te, pm, pp, pa, nr = luna_segments(fp, keep, gap_s, fmask)
        print(f'    [{phase}] {os.path.basename(fp)}: {nr.size} 段 '
              f'(行数中位={int(np.median(nr)) if nr.size else 0})')
        T.append(ts); TEND.append(te)
        PM.append(pm); PP.append(pp); PA.append(pa); N.append(nr)
        PH.append(np.full(nr.size, phase, dtype=object))
    if not T:
        return None
    ts = np.concatenate(T); te = np.concatenate(TEND); nr = np.concatenate(N)
    pm = np.vstack(PM); pp = np.vstack(PP); pa = np.vstack(PA)
    ph = np.concatenate(PH)
    order = np.argsort(ts, kind='stable')
    ts, te, pm, pp, pa, nr, ph = (ts[order], te[order], pm[order], pp[order],
                                  pa[order], nr[order], ph[order])
    uniq = np.concatenate([[True], np.diff(ts) > 1e-6])      # BI/AI 边界去重叠
    ts, te, pm, pp, pa, nr, ph = (ts[uniq], te[uniq], pm[uniq], pp[uniq],
                                  pa[uniq], nr[uniq], ph[uniq])

    # 输出 3 个口径（列名 = {pos:.2f}mm，与 load_dfos 接口一致；timestamp = 相对首段秒数）:
    #   {gid}分布式应变.csv        段内中位        → 空间分布 / DFOS HI
    #   {gid}分布式应变_peak.csv   段内峰值载荷行  → 参考（单行，实测不如 amp）
    #   {gid}分布式应变_amp.csv    段内(p90−p10)   → 循环幅值（与第一批 FBG 块内 std 同语义）
    cols = [f'{p:.2f}mm' for p in pos[keep]]
    rel = ts - ts[0]
    for tag, mat in (('', pm), ('_peak', pp), ('_amp', pa)):
        df = pd.DataFrame(mat, columns=cols)
        df.insert(0, 'timestamp', rel)
        out = os.path.join(ROOT, gid, f'{gid}分布式应变{tag}.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f'    → {os.path.basename(out)} ({os.path.getsize(out)/1e6:.2f} MB)')
    # 输出: {gid}dfos_anchor.csv （段序/阶段/起始墙钟/相对秒/段时长/行数）
    anc = pd.DataFrame({
        'seg': np.arange(len(ts)),
        'phase': ph,
        'wall_iso': pd.to_datetime(ts, unit='s').strftime('%Y-%m-%dT%H:%M:%S.%f'),
        'rel_s': rel, 'seg_span_s': te - ts, 'n_rows': nr,
    })
    out2 = os.path.join(ROOT, gid, f'{gid}dfos_anchor.csv')
    anc.to_csv(out2, index=False, encoding='utf-8-sig')
    n_bi = int((ph == 'BI').sum()); n_ai = int((ph == 'AI').sum())
    print(f'    段数={len(ts)} (BI={n_bi} / AI={n_ai})  跨度={(ts[-1]-ts[0])/3600:.2f} h  '
          f'有效段(≥30行)={int((nr >= 30).sum())}')
    if n_bi and n_ai:
        i0 = int(np.argmax(ph == 'AI'))
        print(f'    冲击分界(BI→AI) = 段 {i0} @ '
              f'{pd.to_datetime(ts[i0], unit="s").strftime("%m-%d %H:%M:%S")}')
    return dict(gid=gid, seg=len(ts), rows=int(nr.sum()), n_bi=n_bi, n_ai=n_ai)


# ============================================================
# AE (.pridb)
# ============================================================
def process_ae(gid, bin_s=AE_BIN_S):
    d = os.path.join(ROOT, gid, 'AE')
    if not os.path.isdir(d):
        print(f'  [{gid}] 无 AE 目录，跳过')
        return None
    fl = sorted(f for f in os.listdir(d) if f.endswith('.pridb'))
    if not fl:
        return None
    # --- 多段 .pridb 会话缝合（见 ae_io / docs/details.md §17.11）---
    import ae_io
    pl = ae_io.plan(gid)
    print(ae_io.describe(gid))
    segs = [(s['path'], o) for s, o in zip(pl['parts'], pl['offsets'])]
    frames = []
    for fp, off in segs:
        con = sqlite3.connect(f'file:{fp}?mode=ro', uri=True)
        n = 0
        parts = []
        sql = 'SELECT Time, Chan, Eny, Amp FROM ae_data WHERE SetType=2'
        for chunk in pd.read_sql_query(sql, con, chunksize=AE_CHUNK):
            n += len(chunk)
            t = chunk['Time'].to_numpy('float64') / TIME_BASE + off
            e = chunk['Eny'].fillna(0.0).to_numpy('float64')
            a = chunk['Amp'].fillna(0.0).to_numpy('float64')
            bi = np.floor(t / bin_s).astype('int64')
            g = pd.DataFrame({'bi': bi, 'energy': e, 'amplitude': a, 'one': 1}) \
                .groupby('bi', sort=True).agg(
                    energy=('energy', 'max'), amplitude=('amplitude', 'max'),
                    n_hits=('one', 'sum'))
            parts.append(g)
        con.close()
        print(f'    {os.path.basename(fp)}: {n:,} hits (offset {off:.0f}s)')
        if not parts:
            continue
        g = pd.concat(parts).groupby(level=0).agg(
            energy=('energy', 'max'), amplitude=('amplitude', 'max'),
            n_hits=('n_hits', 'sum')).reset_index()
        g = g.rename(columns={'bi': 'time'})
        g['time'] = g['time'] * bin_s
        frames.append(g[['time', 'n_hits', 'energy', 'amplitude']])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values('time').reset_index(drop=True)
    out = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    df.to_csv(out, index=False, encoding='utf-8-sig')
    print(f'    hits 总数={int(df["n_hits"].sum()):,}  bin 数={len(df)}  '
          f'time={df["time"].min():.0f}~{df["time"].max():.0f}s '
          f'→ {os.path.basename(out)} ({os.path.getsize(out)/1e6:.2f} MB)')
    return dict(gid=gid, n_bin=len(df), t_max=float(df['time'].max()))


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS_V2))
    ap.add_argument('--space', type=int, default=N_SPACE)
    ap.add_argument('--gap', type=float, default=GAP_S)
    ap.add_argument('--skip-ae', action='store_true')
    ap.add_argument('--skip-dfos', action='store_true')
    a = ap.parse_args()

    rows = []
    for g in a.groups.split(','):
        g = g.strip()
        print(f'--- {g} ---')
        if not a.skip_dfos:
            print('  [DFOS]')
            process_luna(g, n_space=a.space, gap_s=a.gap)
        if not a.skip_ae:
            print('  [AE]')
            process_ae(g)
        print()
    print('完成')


if __name__ == '__main__':
    main()
