# -*- coding: utf-8 -*-
"""RA–AF 损伤机制识别（第二批 L1-49..L1-60）

原理
----
复合材料 AE 的经典机制判据：用波形两个形状参数把事件分为
  - **拉伸型（tensile）**：高 AF、低 RA → 基体开裂 / 纤维断裂（裂纹张开，脆性快释放）
  - **剪切型（shear）**  ：低 AF、高 RA → **分层 / 脱粘**（界面剪切滑移+摩擦，释放慢）
其中
  RA = RiseT [µs] / Amp [dB]           （波形上升陡峭度）
  AF = Counts / Dur [µs] × 1000 [kHz]  （平均频率）
Amp[dB] = 20·log10(Amp[µV])。实测 9 组采集阈值一致 = 657 µV = 56.35 dB
→ Counts/Dur 口径一致，具备**跨组可比性**（这是 RUL 失败的根因，RA–AF 恰好绕开）。

判据口径（本实现；避免宣称"绝对分类正确"）
----------------------------------------
RA/AF 绝对值跨试件不可比（受增益/几何影响），故：
  1. 以**文件首个数据块**（试验早期）为基准做稳健标准化（因果，不引入未来信息）：
     ra_n = (RA − med₀)/IQR₀,  af_n = (AF − med₀)/IQR₀
  2. 定义**拉伸–剪切倾向分数** score = af_n − ra_n；score<0 记为剪切型
  3. 按寿命进度分 N_BIN 箱统计 shear_frac / mean_score / RA、AF 中位数
  4. **BI（冲击前）段单独汇总**（不混入寿命进度）

交叉验证（关键）
--------------
剪切型占比的上升时段应与 DFOS「空间重分布量」（evaluate_l1_dfos 的 local/rmse）一致 ——
二者是**独立测量通道**（AE 波形 vs 光纤应变场）。见 `--compare`。

用法:
  python ae_raf.py --groups L1-49            # 单组验证
  python ae_raf.py                           # 全部 9 组
  python ae_raf.py --compare                 # 与 DFOS local 做相关性对照
输出: results/_l1_raf_{gid}.npz + figures/l1_raf_{gid}.png
"""
import os
import io
import sys
import argparse
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, 'results')
FIG = os.path.join(ROOT, 'figures')
os.makedirs(RES, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
sys.path.insert(0, ROOT)
from l1_meta import load_meta                          # noqa: E402
import l1_time_align as ta                             # noqa: E402
import ae_io                                            # noqa: E402

GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54', 'L1-55', 'L1-56',
          'L1-59', 'L1-60']
TIME_BASE = 1e7
N_BIN = 100
CHUNK = 2_000_000
SUB_PER_BIN = 50          # 每箱抽样点数（用于中位数近似）


def compute(gid, n_bin=N_BIN, verbose=True):
    nf = load_meta(gid)['n_f']
    d = os.path.join(ROOT, gid, 'AE')
    files = sorted(f for f in os.listdir(d) if f.endswith('.pridb'))
    if not files:
        print(f'  [{gid}] 无 AE')
        return None
    # --- 多段 .pridb 会话缝合（见 ae_io / docs/details.md §17.11）---
    pl = ae_io.plan(gid)
    print(ae_io.describe(gid))
    files = [(s['path'], o) for s, o in zip(pl['parts'], pl['offsets'])]
    acc_n = np.zeros(n_bin); acc_s = np.zeros(n_bin); acc_sh = np.zeros(n_bin)
    med_ra, med_af = [], []
    sub_ra = [[] for _ in range(n_bin)]
    sub_af = [[] for _ in range(n_bin)]
    bi_n = 0; bi_s = 0.0; bi_sh = 0
    init = None
    n_tot = 0; n_ok = 0
    for fp, off in files:
        con = sqlite3.connect(f'file:{fp}?mode=ro', uri=True)
        for chunk in pd.read_sql_query(
                'SELECT Time, RiseT, Dur, Counts, Amp FROM ae_data WHERE SetType=2',
                con, chunksize=CHUNK):
            n_tot += len(chunk)
            amp = chunk['Amp'].to_numpy('float64')
            rt = chunk['RiseT'].to_numpy('float64')
            dur = chunk['Dur'].to_numpy('float64')
            cnt = chunk['Counts'].to_numpy('float64')
            ok = (dur > 0) & (rt > 0) & (amp > 1.0) & (cnt > 0)
            if not ok.any():
                continue
            n_ok += int(ok.sum())
            t = chunk['Time'].to_numpy('float64')[ok] / TIME_BASE + off
            ra = rt[ok] / (20.0 * np.log10(amp[ok]))
            af = cnt[ok] / dur[ok] * 1000.0
            cyc = ta.time_to_cycle(gid, t, nf)
            if init is None:
                init = (float(np.median(ra)),
                        float(np.percentile(ra, 75) - np.percentile(ra, 25)),
                        float(np.median(af)),
                        float(np.percentile(af, 75) - np.percentile(af, 25)))
                if verbose:
                    print(f'    归一化基准(首个数据块): RA_med={init[0]:.3f} '
                          f'IQR={init[1]:.3f} | AF_med={init[2]:.2f} IQR={init[3]:.2f}')
            iqr = init[1] if init[1] > 0 else 1.0
            iqf = init[3] if init[3] > 0 else 1.0
            score = (af - init[2]) / iqf - (ra - init[0]) / iqr
            is_bi = cyc < 0
            if is_bi.any():
                bi_n += int(is_bi.sum())
                bi_s += float(score[is_bi].sum())
                bi_sh += int((score[is_bi] < 0).sum())
            m = ~is_bi
            if not m.any():
                continue
            li = np.clip((cyc[m] / nf * n_bin).astype('int64'), 0, n_bin - 1)
            dd = pd.DataFrame({'b': li, 'ra': ra[m], 'af': af[m],
                               's': score[m], 'neg': (score[m] < 0).astype('int8')})
            g = dd.groupby('b', sort=True).agg(n=('s', 'size'), ss=('s', 'sum'),
                                               ng=('neg', 'sum'))
            idx = g.index.to_numpy()
            acc_n[idx] += g['n'].to_numpy()
            acc_s[idx] += g['ss'].to_numpy()
            acc_sh[idx] += g['ng'].to_numpy()
            k = max(1, len(dd) // (n_bin * SUB_PER_BIN))
            ds = dd.iloc[::k]
            gs = ds.groupby('b', sort=True).agg(rm=('ra', 'median'), fm=('af', 'median'))
            med_ra.append(gs['rm']); med_af.append(gs['fm'])
            for b, grp in ds.groupby('b', sort=True):
                sub_ra[b].append(grp['ra'].to_numpy('float32'))
                sub_af[b].append(grp['af'].to_numpy('float32'))
        con.close()
    if n_ok == 0 or init is None:
        print(f'  [{gid}] 无可算事件')
        return None
    ra_med = (pd.concat(med_ra, axis=1).median(axis=1)
              .reindex(np.arange(n_bin)).to_numpy(float))
    af_med = (pd.concat(med_af, axis=1).median(axis=1)
              .reindex(np.arange(n_bin)).to_numpy(float))
    nz = acc_n > 0
    shear = np.where(nz, acc_sh / np.maximum(acc_n, 1), np.nan)
    mscore = np.where(nz, acc_s / np.maximum(acc_n, 1), np.nan)
    life = (np.arange(n_bin) + 0.5) / n_bin
    S_RA = np.full(n_bin, np.nan)
    S_AF = np.full(n_bin, np.nan)
    for b in range(n_bin):
        if sub_ra[b]:
            S_RA[b] = float(np.median(np.concatenate(sub_ra[b])))
        if sub_af[b]:
            S_AF[b] = float(np.median(np.concatenate(sub_af[b])))
    out = os.path.join(RES, f'_l1_raf_{gid}.npz')
    np.savez_compressed(out, life=life, n_hits=acc_n, shear_frac=shear,
                        mean_score=mscore, ra_med=ra_med, af_med=af_med,
                        samp_ra=S_RA, samp_af=S_AF, n_f=nf, norm=np.array(init),
                        bi_n=bi_n,
                        bi_shear=(bi_sh / bi_n if bi_n else np.nan),
                        bi_score=(bi_s / bi_n if bi_n else np.nan))
    print(f'  [{gid}] hits={n_tot:,} 可算={n_ok:,}({n_ok/max(n_tot,1)*100:.1f}%)  '
          f'BI={bi_n:,}')
    print(f'    剪切型占比: 前20%寿命={np.nanmean(shear[:20]):.3f}  '
          f'后20%寿命={np.nanmean(shear[-20:]):.3f}  '
          f'整体={acc_sh.sum()/max(acc_n.sum(),1):.3f}  '
          f'BI(冲击前)={bi_sh/bi_n if bi_n else float("nan"):.3f}')
    return out


def plot(gid):
    z = np.load(os.path.join(RES, f'_l1_raf_{gid}.npz'))
    life = z['life'] * 100
    sf = z['shear_frac']; ms = z['mean_score']
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(life, sf, 'o-', ms=3, color='tab:red', lw=1)
    ax[0].axhline(0.5, ls=':', color='grey')
    bi = z['bi_shear']
    if np.isfinite(bi):
        ax[0].axhline(bi, ls='--', color='tab:green', lw=1,
                      label=f'BI(冲击前)={bi:.2f}')
        ax[0].legend(fontsize=8)
    ax[0].set_ylabel('剪切型占比'); ax[0].set_ylim(0, 1); ax[0].grid(alpha=.3)
    ax[0].set_title(f'{gid}  RA–AF 机制演化（剪切型↑ = 分层/脱粘成分增多）')
    ax[1].plot(life, ms, 'o-', ms=3, color='tab:blue', lw=1)
    ax[1].axhline(0, ls=':', color='grey')
    ax[1].set_ylabel('拉伸–剪切倾向\nscore = af_n − ra_n')
    ax[1].set_xlabel('寿命进度 (%)'); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fp = os.path.join(FIG, f'l1_raf_{gid}.png')
    fig.savefig(fp, dpi=110)
    plt.close(fig)
    return fp


def _smooth(x, w=5):
    x = np.asarray(x, float)
    if w <= 1:
        return x
    k = w // 2
    return np.array([np.nanmean(x[max(0, i - k):i + k + 1]) for i in range(len(x))])


def compare_with_dfos(groups, smooth=5):
    """交叉验证：剪切型占比 vs DFOS 空间重分布量 local。

    两者都反映界面损伤，但**时间响应不同**（AE 是事件级瞬时、DFOS 是测段累积），
    故同时给出原始与平滑后的相关。
    """
    rows = []
    for g in groups:
        fp = os.path.join(RES, f'_l1_raf_{g}.npz')
        fd = os.path.join(RES, f'_l1_dfos_hi_{g}.npz')
        if not (os.path.exists(fp) and os.path.exists(fd)):
            continue
        z = np.load(fp); zd = np.load(fd)
        nf = float(z['n_f'])
        sf = z['shear_frac']
        cyc = zd['cyc'].astype(float)
        m = cyc >= 0
        if m.sum() < 5:
            continue
        loc = zd['local'][m]
        life_d = cyc[m] / nf
        gl = np.linspace(0, 1, z['life'].size)
        loc_r = np.interp(gl, life_d, loc)
        ok = np.isfinite(sf) & np.isfinite(loc_r) & (gl <= life_d.max())
        if ok.sum() < 10:
            continue
        a, b = sf[ok], loc_r[ok]
        r_raw = float(np.corrcoef(a, b)[0, 1])
        a_s, b_s = _smooth(a, smooth), _smooth(b, smooth)
        mo = np.isfinite(a_s) & np.isfinite(b_s)
        r_sm = float(np.corrcoef(a_s[mo], b_s[mo])[0, 1]) if mo.sum() > 5 else np.nan
        rows.append(dict(gid=g, n=int(ok.sum()),
                         r_raw=round(r_raw, 3),
                         r_smooth=round(r_sm, 3) if np.isfinite(r_sm) else None,
                         shear_first20=round(float(np.nanmean(a[:20])), 3),
                         shear_last20=round(float(np.nanmean(a[-20:])), 3)))
        print(f'  [{g}] 剪切占比 vs DFOS local: r_raw={r_raw:+.3f}  '
              f'r_smooth={r_sm:+.3f}  (n={ok.sum()})')
    if rows:
        df = pd.DataFrame(rows)
        out = os.path.join(RES, 'l1_raf_dfos_compare.csv')
        df.to_csv(out, index=False, encoding='utf-8-sig')
        rs = pd.to_numeric(df['r_smooth'], errors='coerce')
        print(f'\n  平均 r_raw={df["r_raw"].mean():+.3f}  '
              f'平均 r_smooth={rs.mean():+.3f}')
        print(f'  正相关组数(r_smooth) = {(rs > 0).sum()}/{len(df)}')
        print(f'  已存: {out}')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', default=','.join(GROUPS))
    ap.add_argument('--bins', type=int, default=N_BIN)
    ap.add_argument('--compare', action='store_true', help='与 DFOS local 做交叉验证')
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()
    groups = [g.strip() for g in a.groups.split(',')]
    for g in groups:
        print(f'--- {g} ---')
        out = compute(g, a.bins)
        if out and not a.no_plot:
            plot(g)
        print()
    if a.compare:
        print('=== 交叉验证：RA–AF 剪切占比 vs DFOS 空间重分布 (local) ===')
        compare_with_dfos(groups)


if __name__ == '__main__':
    main()
