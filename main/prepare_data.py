# -*- coding: utf-8 -*-
"""阶段① 数据准备：多源对齐 + 弱标签（5 组主样本 016-020）

子命令（可组合，如 `python prepare_data.py align weaklabels` 或 `all`）：
  align       多源数据对齐(AE 网格) → aligned/
  weaklabels  弱标签生成 → weak_labels/
"""
import os, sys, argparse, warnings, traceback
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根(含 shm)
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # main/

from shm.data_loader import DataLoader   # weaklabels 用

ROOT = os.path.dirname(os.path.abspath(__file__))   # <项目根>\main
GROUPS = ['016', '017', '018', '019', '020']   # 主样本 5 组

# ============================================================
# 子命令 align —— 多源对齐（AE 网格模式）
# ============================================================
OUT = os.path.join(ROOT, "aligned")   # 对齐输出：根目录 aligned\
OUT_D = OUT
os.makedirs(OUT_D, exist_ok=True)


def ts_to_seconds(t_series):
    """时间戳统一换算为秒（等长返回，保留 NaN）。

    采样率 10Hz：光纤/部分应变 dt≈1.0（0.1s 计数单位）需 ×0.1；应变 dt≈0.1 已是秒。
    按中位间隔自动判断（>=0.2s 视为 0.1s 计数单位）。
    """
    t = pd.to_numeric(t_series, errors='coerce').values.astype(float)
    valid = t[~np.isnan(t)]
    if len(valid) < 2:
        return t
    dt = float(np.median(np.diff(valid)))
    if dt >= 0.2:
        return t * 0.1
    return t


def detect_gaps(time_s, mult=1.5, min_gap_s=0.5):
    """检测时间轴缺失段。返回 [(start_s, end_s, lost_s), ...]。"""
    time_s = np.asarray(time_s, dtype=float)
    if len(time_s) < 3:
        return []
    d = np.diff(time_s)
    med = float(np.median(d))
    if med <= 0:
        return []
    gaps = []
    for i, g in enumerate(d):
        if g > mult * med and g >= min_gap_s:
            gaps.append((float(time_s[i]), float(time_s[i + 1]), float(g - med)))
    return gaps


def read_fiber(sid, sdir):
    """读光纤 → DataFrame[time_s, fiber_1..] 或 (None, err)。"""
    files = [f for f in os.listdir(sdir) if "光纤" in f]
    if not files:
        return None, "无光纤文件"
    fp = os.path.join(sdir, files[0])
    skiprows = 0
    if fp.endswith(".xlsx"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            return None, f"xlsx读取失败: {e}"
    else:
        df = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]:
            try:
                df = pd.read_csv(fp, encoding=enc, skiprows=skiprows)
                break
            except Exception:
                continue
        if df is None:
            return None, "CSV编码无法识别"
    cols = list(df.columns)
    if len(cols) < 2:
        return None, f"列数不足: {cols}"
    time_s = ts_to_seconds(df.iloc[:, 0])
    channels = {}
    for i, c in enumerate(cols[1:], 1):
        vals = pd.to_numeric(df[c], errors='coerce').values
        if np.sum(~np.isnan(vals)) > 10:
            channels[f"fiber_{i}"] = vals
    if not channels:
        return None, "无有效通道"
    result = pd.DataFrame({"time_s": time_s})
    for k, v in channels.items():
        result[k] = v
    result = result.dropna(subset=["time_s"])
    ch_cols = list(channels.keys())
    result = result.dropna(subset=ch_cols, how='all')
    result = result.sort_values("time_s").reset_index(drop=True)
    return result, None


def read_strain(sid, sdir):
    """读应变 → DataFrame[time_s, strain] 或 (None, err)。"""
    files = [f for f in os.listdir(sdir) if "应变" in f]
    if not files:
        return None, "无应变文件"
    fp = os.path.join(sdir, files[0])
    if fp.endswith(".xlsx"):
        try:
            df = pd.read_excel(fp)
        except Exception as e:
            return None, f"xlsx读取失败: {e}"
    else:
        df = None
        for enc in ["utf-8-sig", "utf-8", "gbk", "gb2312", "latin1"]:
            try:
                df = pd.read_csv(fp, encoding=enc)
                break
            except Exception:
                continue
        if df is None:
            return None, "CSV编码无法识别"
    if df.shape[1] < 2:
        return None, f"列数不足: {df.shape}"
    time_s = ts_to_seconds(df.iloc[:, 0])
    v = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
    result = pd.DataFrame({"time_s": time_s, "strain": v})
    result = result.dropna()
    result = result.sort_values("time_s").reset_index(drop=True)
    return result, None


def read_ae(sid, sdir):
    """读声发射原始 DataFrame 或 (None, err)。"""
    files = [f for f in os.listdir(sdir) if "声发射" in f]
    if not files:
        return None, "无声发射文件"
    fp = os.path.join(sdir, files[0])
    for enc in ["utf-8-sig", "utf-8", "gbk", "latin1"]:
        try:
            df = pd.read_csv(fp, encoding=enc)
            return df, None
        except Exception:
            continue
    return None, "CSV编码无法识别"


def detect_oscillation(strain_df):
    """应变是否循环振荡：前 5000 点零交叉率 >30%。"""
    v = strain_df['strain'].values
    if len(v) < 100:
        return False
    sample = v[:min(len(v), 5000)]
    signs = np.sign(sample)
    signs[signs == 0] = 1
    zero_crossings = np.sum(signs[1:] != signs[:-1])
    return (zero_crossings / len(signs)) > 0.3


def align_sources_to_ae(ae_df, fb_df, ys_df, t_start, T):
    """以声发射事件为网格对齐多源。AE 无时间戳：行号均匀映射 [0,T]。
    光纤/应变线性插值到 AE 事件时刻，仅在有效覆盖范围内（范围外 NaN，不外推）。"""
    if ae_df is None or len(ae_df) < 2:
        return None
    N = len(ae_df)
    ae_norm = np.linspace(0, 1, N)
    ae_t = np.linspace(t_start, t_start + T, N)
    out = pd.DataFrame({"ae_event": np.arange(N), "ae_time_s": ae_t, "ae_norm": ae_norm})
    for c in ae_df.columns:                       # AE 原始特征 100% 保留
        out[f"ae_{c}"] = pd.to_numeric(ae_df[c], errors='coerce').values
    if fb_df is not None and len(fb_df) > 1:      # 光纤插值
        fb_t = fb_df['time_s'].values.astype(float)
        fb_norm = (fb_t - t_start) / T
        for c in [c for c in fb_df.columns if c.startswith("fiber_")]:
            v = pd.to_numeric(fb_df[c], errors='coerce').values.astype(float)
            valid = ~np.isnan(v) & ~np.isnan(fb_norm)
            if valid.sum() > 1:
                interp = np.interp(ae_norm, fb_norm[valid], v[valid])
                lo, hi = float(np.min(fb_norm[valid])), float(np.max(fb_norm[valid]))
                interp[(ae_norm < lo) | (ae_norm > hi)] = np.nan
                out[c] = interp
        fb_cols = [c for c in out.columns if c.startswith("fiber_") and c != "fiber_mean"]
        if fb_cols:
            out["fiber_mean"] = out[fb_cols].mean(axis=1)
    if ys_df is not None and len(ys_df) > 1:      # 应变插值
        ys_t = ys_df['time_s'].values.astype(float)
        ys_v = ys_df['strain'].values.astype(float)
        ys_norm = (ys_t - t_start) / T
        valid = ~np.isnan(ys_v) & ~np.isnan(ys_norm)
        if valid.sum() > 1:
            interp = np.interp(ae_norm, ys_norm[valid], ys_v[valid])
            lo, hi = float(np.min(ys_norm[valid])), float(np.max(ys_norm[valid]))
            interp[(ae_norm < lo) | (ae_norm > hi)] = np.nan
            out["strain"] = interp
    return out


def process_specimen(sid):
    """处理单个试件 → (ae_grid, meta) 或 (None, error)。"""
    sdir = os.path.join(ROOT, sid)
    meta = {"试件": sid, "问题": []}
    fb_df, fb_err = read_fiber(sid, sdir)
    ys_df, ys_err = read_strain(sid, sdir)
    ae_df, ae_err = read_ae(sid, sdir)
    if fb_err: meta["问题"].append(f"光纤: {fb_err}")
    if ys_err: meta["问题"].append(f"应变: {ys_err}")
    if ae_err: meta["问题"].append(f"声发射: {ae_err}")
    if fb_df is None and ys_df is None:
        return None, "光纤和应变均缺失，无法确定时间轴"
    # 以应变为主时间基准（连续、覆盖完整实验）；无应变时用光纤
    if ys_df is not None and len(ys_df) > 0:
        t_start, t_end, time_ref = float(ys_df['time_s'].iloc[0]), float(ys_df['time_s'].iloc[-1]), "应变"
    elif fb_df is not None and len(fb_df) > 0:
        t_start, t_end, time_ref = float(fb_df['time_s'].iloc[0]), float(fb_df['time_s'].iloc[-1]), "光纤"
    else:
        return None, "光纤和应变均缺失，无法确定时间轴"
    T = t_end - t_start
    meta["实验时长_s"], meta["时间起点_s"], meta["时间基准"] = round(T, 1), round(t_start, 1), time_ref
    for tag, df in (("应变", ys_df), ("光纤", fb_df)):    # 丢数据检测
        if df is not None and len(df) > 1:
            gs = detect_gaps(df['time_s'].values)
            if gs:
                meta[f"{tag}缺失段"] = [f"{a:.1f}-{b:.1f}s(缺{c:.1f}s)" for a, b, c in gs]
                meta[f"{tag}缺失总时长_s"] = round(float(sum(c for _, _, c in gs)), 1)
    if fb_df is not None and len(fb_df) > 0 and ys_df is not None and len(ys_df) > 0:
        fb_dur = float(fb_df['time_s'].iloc[-1] - fb_df['time_s'].iloc[0])
        if fb_dur > T * 1.1:
            meta["问题"].append(f"光纤时长{fb_dur:.0f}s 明显长于应变{T:.0f}s，应变可能提前结束")
        elif fb_dur < T * 0.9:
            meta["问题"].append(f"光纤仅覆盖{fb_dur/T*100:.0f}%实验时长，可能中途中断")
    meta["光纤通道数"] = len([c for c in fb_df.columns if c.startswith("fiber_")]) if fb_df is not None else 0
    if ys_df is not None and len(ys_df) > 0:
        meta["应变振荡"] = detect_oscillation(ys_df)
    meta["声发射行数"] = len(ae_df) if ae_df is not None else 0
    ae_grid = align_sources_to_ae(ae_df, fb_df, ys_df, t_start, T)
    if ae_grid is None:
        meta["问题"].append("声发射数据不足，无法生成AE网格对齐输出")
        return ae_grid, meta
    meta["输出列"] = list(ae_grid.columns)
    meta["数据完整率"] = round(ae_grid.notna().mean().mean(), 3)
    return ae_grid, meta


def cmd_align():
    all_meta, failed = [], []
    for sid in GROUPS:
        try:
            ae_grid, meta = process_specimen(sid)
            if ae_grid is None:
                print(f"[跳过] {sid}: {meta.get('问题', ['未知'])}")
                failed.append((sid, str(meta.get("问题", []))))
                all_meta.append(meta)
                continue
            ae_path = os.path.join(OUT_D, f"{sid}.csv")
            ae_grid.to_csv(ae_path, index=False, encoding="utf-8-sig", float_format="%.6f")
            all_meta.append(meta)
            n_issues = len(meta.get("问题", []))
            miss = meta.get("应变缺失总时长_s", 0) + meta.get("光纤缺失总时长_s", 0)
            print(f"[完成] {sid}: T={meta['实验时长_s']:.0f}s, AE={meta['声发射行数']}行, "
                  f"光纤{meta['光纤通道数']}ch, 问题={n_issues}, 缺失={miss}s")
        except Exception as e:
            traceback.print_exc()
            failed.append((sid, str(e)))
            print(f"[异常] {sid}: {e}")
    meta_df = pd.DataFrame(all_meta)
    meta_df.to_csv(os.path.join(OUT, "对齐元信息.csv"), index=False, encoding="utf-8-sig")
    print(f"\n===== 对齐完成（AE 网格模式） =====")
    print(f"成功: {len(all_meta) - len(failed)}/{len(GROUPS)}")
    print(f"失败/跳过: {len(failed)}")
    for sid, err in failed:
        print(f"  {sid}: {err}")
    print(f"输出目录: {OUT_D}")


# ============================================================
# 子命令 weaklabels —— 弱标签生成
# ============================================================
# 协议：b3=数据末端断裂(≈99%)；b2=损伤扩展(AE 累积能量最大加速或应变发散更早)；
#       b1=损伤萌生(累积 log 能量水平持久抬升；平台型无独立萌生段返回 None)。
def smooth(x, w):
    k = np.ones(w) / w
    return np.convolve(x, k, mode='same')


def find_b2(logc, grid):
    """能量曲线最大正加速点(主要能量台阶)，限 30%~97% → life%"""
    d1 = np.gradient(logc, grid)
    d2 = np.gradient(d1, grid)
    mask = (grid > 0.30) & (grid < 0.97)
    if not mask.any():
        return None
    j = int(np.argmax(d2[mask]))
    return float(grid[mask][j] * 100.0)


def find_b1(logc, grid, b2):
    """b1(萌生)=累积能量水平相对前平台的【持久】抬升最早点。
    条件：post−pre≥0.4(≈2.5×能量) 且 tail−pre≥0.2；平台型无此抬升→None。"""
    if b2 is None:
        return None
    lo, hi = 0.12, (b2 - 1.0) / 100.0
    if hi <= lo:
        return None
    m = (grid >= lo) & (grid <= hi)
    idx = np.where(m)[0]
    d_log, d_tail = 0.4, 0.2
    for ik in idx:
        gk = float(grid[ik])
        i0 = int(np.searchsorted(grid, max(lo, gk - 0.05)))
        i1 = max(ik - 1, i0)
        pre = float(np.median(logc[i0:i1 + 1])) if i1 > i0 else float(logc[ik])
        j1 = int(np.searchsorted(grid, min(hi, gk + 0.05)))
        post = float(np.median(logc[ik:j1 + 1])) if j1 > ik else float(logc[ik])
        tail = float(np.percentile(logc[ik:], 50))
        if post - pre >= d_log and tail - pre >= d_tail:
            return gk * 100.0
    return None


def run_weaklabel(gid):
    """单组弱标签阶梯 (life% → ref_stage)。"""
    dl = DataLoader(gid).load_all()
    peak = dl.ae['Peak'].values if dl.ae is not None and 'Peak' in dl.ae.columns else None
    strain = dl.strain['strain'].values if dl.strain is not None else None
    n_ae = len(peak) if peak is not None else 0
    if n_ae == 0:
        return None
    grid = np.linspace(0, 1, 2001)
    e2 = np.maximum(peak, 0.0) ** 2
    cum = np.cumsum(e2)
    logc = np.log10(np.maximum(np.interp(grid, np.linspace(0, 1, n_ae), cum), 1e-12))
    logc = smooth(logc, 21)                       # ~1% 平滑
    b2 = find_b2(logc, grid)
    b1 = find_b1(logc, grid, b2) if b2 is not None else None
    b3 = 99.0                                     # 失效锚点 = 末端
    # 应变末期发散点（b2 取应变发散更早者 → 应变主导）
    strain_diverge = None
    if strain is not None and len(strain) > 200:
        n = len(strain)
        nb = 200
        edges = np.linspace(0, n, nb + 1).astype(int)
        blk_std = np.array([float(np.std(strain[edges[i]:edges[i+1]])) if edges[i+1] > edges[i] else 0.0 for i in range(nb)])
        bpct = np.linspace(0, 100, nb)
        mid = (bpct > 25) & (bpct < 60)
        base = float(np.median(blk_std[mid])) if mid.any() else 0.0
        tail = bpct > 50
        if base > 1e-9:
            over = np.where(tail & (blk_std > 1.8 * base))[0]
            if len(over):
                strain_diverge = float(bpct[over[0]])
    if strain_diverge is not None and 0 < strain_diverge < 95:
        if b2 is None or strain_diverge < b2:
            b2 = strain_diverge
    n_grid = 1000                                  # 阶梯组装 0.1% 步长
    life = np.arange(n_grid) * 0.1
    stage = np.zeros(n_grid, dtype=int)
    b1v = b1 if b1 is not None else (b2 if b2 is not None else 60.0)
    b2v = b2 if b2 is not None else (b3 * 0.8)
    stage[life >= b1v] = 1
    stage[life >= b2v] = 2
    stage[life >= b3] = 3
    return dict(gid=gid, b1=round(float(b1v), 1), b2=round(float(b2v), 1), b3=b3,
                n_ae=n_ae, strain_diverge=strain_diverge,
                life=life, stage=stage)


def cmd_weaklabels():
    all_rows = []
    for gid in GROUPS:
        res = run_weaklabel(gid)
        if res is None:
            print(f'{gid}: 无 AE，跳过')
            continue
        all_rows.append(res)
        df = pd.DataFrame({'life_pct': res['life'], 'ref_stage': res['stage']})
        df.loc[df['life_pct'] >= res['b3'], 'note'] = 'failure(末端断裂锚点)'
        df.loc[(df['life_pct'] >= res['b2']) & (df['life_pct'] < res['b3']), 'note'] = 'phase2(扩展)'
        df.loc[(df['life_pct'] >= res['b1']) & (df['life_pct'] < res['b2']), 'note'] = 'phase1(微损伤)'
        df.loc[df['life_pct'] < res['b1'], 'note'] = 'phase0(健康/加载)'
        sel = df.iloc[::2].copy()                 # 抽样 0.5% 存标签
        sel.to_csv(rf'{ROOT}\weak_labels\{gid}_label.csv', index=False, encoding='utf-8-sig')
        print(f"{gid}: b1={res['b1']:6.1f}  b2={res['b2']:6.1f}  b3={res['b3']}  "
              f"strain_diverge={res['strain_diverge']}")
    sumdf = pd.DataFrame([{k: r[k] for k in ('gid', 'b1', 'b2', 'b3', 'n_ae', 'strain_diverge')} for r in all_rows])
    sumdf.to_csv(rf'{ROOT}\weak_labels\labels_summary.csv', index=False, encoding='utf-8-sig')
    print(f'\n共 {len(all_rows)} 组。汇总: weak_labels/labels_summary.csv')


# ============================================================
# 分发
# ============================================================
TASKS = {'align': cmd_align, 'weaklabels': cmd_weaklabels}


def main():
    ap = argparse.ArgumentParser(description='阶段① 数据准备：align / weaklabels')
    ap.add_argument('tasks', nargs='+', choices=list(TASKS) + ['all'],
                    help='要执行的任务（可多个，或用 all）')
    a = ap.parse_args()
    todo = list(TASKS) if 'all' in a.tasks else a.tasks
    for t in todo:
        print(f'\n########## [{t}] ##########', flush=True)
        TASKS[t]()
        print(f'########## [{t}] 完成 ##########', flush=True)


if __name__ == '__main__':
    main()
