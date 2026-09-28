# -*- coding: utf-8 -*-
"""L1 公开集 → 在线监测看板数据包
=================================================================

把 L1-03/04/05/09 的**逐点在线**结果打包成与主样本看板**同格式**的 JS 数据文件，
供 `dashboard/index.html` 直接加载（试件名不冲突: 主样本 016… / L1 为 L1-03…）。

严格复用正式口径（与 `l1/results/l1_degree.csv` 完全一致）:
  D(t) = evaluate_l1_degree.run_group(gid, {'rise':0.05},
                                      baseline=True, strain_ev=True, fusion='max')

与主样本的差异（通过数据包自带字段自适应，不改动主样本导出器）:
  unit      = 'cycle'          ← 横轴为循环数(非秒); 前端据此切换时间格式
  frameDt   = 50               ← 每个回放帧代表 50 cycle (STEP=5 × 10 cycle/点)
  chans     = [...]            ← 通道健康表由数据包驱动(含 DFOS 行)
  dfos      = {...}            ← 分布式应变块级序列(局部峰/rmse/HI)
  meta.b2   = None, b3 = 100   ← L1 无弱标签 b2; 失效锚 n_f = 100%
  meta.c0Pct                   ← 基线重定义终点 c0 的寿命百分比
  meta.refs = [{pct,label}]    ← 论文检测点(灰色参考锚)
  ax.ael    = [lo,hi]          ← AE log 峰值纵轴范围(自动)

数值编码(与主样本一致, 前端解码):
  D / risk / eae / est / dfosHi : ×1000      st / fo / dfosLocal : ×100
  ael : log10(peak)×1000, 无事件 = -99999    aen : 帧内 AE 事件数

用法:
  python l1/export_dashboard_l1.py                 # 全部 4 组
  python l1/export_dashboard_l1.py L1-03 L1-05     # 指定组
  python l1/export_dashboard_l1.py --out <dir>     # 自定义输出目录
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))          # <项目根>\l1
PROJ = os.path.dirname(ROOT)                               # 项目根
sys.path.insert(0, PROJ)                                   # shm
sys.path.insert(0, ROOT)                                   # l1_meta 等同级模块
os.chdir(ROOT)

import evaluate_l1_degree as _deg                          # noqa: E402
import evaluate_l1_dfos as _dfos                           # noqa: E402
import reproduce_broer_l1 as _l4                           # noqa: E402

GAP_S = 300.0                 # FBG 测量块间隔阈值(s)
MIN_ROWS = 500                # 一个有效块的最少采样行
CYCS_PER_BLK = _deg.CYCS_PER_BLK
CYCS_PER_PT = _deg.CYCS_PER_PT
PTS_PER_BLK = _deg.PTS_PER_BLK
STEP = 5                      # 降采样步长(点): 1 帧 = 5 点 = 50 cycle
LEVELS = [0.25, 0.55, 0.85]
AE_EMPTY = -99999
PARAMS = {'rise': 0.05}       # = run.py 的 l1/degree 推荐配置
FUSION = 'max'


# ------------------------------------------------------------------
# 基础工具
# ------------------------------------------------------------------
def _i100(x):
    return int(round(float(x) * 100.0))


def _i1000(x):
    return int(round(float(x) * 1000.0))


def fbg_strain_cols(df):
    """全部 FBG 应变通道(表头自适应); 排除 FBG_A*/FBG_B* 波长列。"""
    return [c for c in df.columns
            if (c.startswith('fbg') and c[3:].isdigit())
            or (c.startswith('b') and c[1:].isdigit())]


def fill_nan(v):
    """线性插值填补 NaN; 全 NaN 返回 None。"""
    v = np.asarray(v, float).copy()
    m = np.isfinite(v)
    if m.sum() == 0:
        return None
    if m.sum() < v.size:
        idx = np.arange(v.size)
        v = np.interp(idx, idx[m], v[m])
    return v


def fbg_block_channels(gid):
    """每个 FBG 测量块的**逐通道**应变均值 → (块 cycle, 通道名, {通道: 值数组})。

    抛掉全 NaN 或有效块不足半数的通道(如 L1 里常年无数据的 fbg2-5 / 波长列)。
    """
    fp = os.path.join(ROOT, gid, f'{gid}光纤.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig')
    t = df['timestamp'].to_numpy(float)
    cols = fbg_strain_cols(df)
    gap = np.where(np.diff(t) > GAP_S)[0]
    bounds = np.concatenate([[0], gap + 1, [len(t)]])
    cyc, vals = [], {c: [] for c in cols}
    k = 0
    for i in range(len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a < MIN_ROWS:
            continue
        cyc.append((k + 0.5) * CYCS_PER_BLK)
        for c in cols:
            seg = df[c].to_numpy(float)[a:b]
            vals[c].append(float(np.nanmean(seg)) if np.isfinite(seg).any() else np.nan)
        k += 1
    cyc = np.asarray(cyc, float)
    keep, out = [], {}
    for c in cols:
        v = fill_nan(vals[c])
        if v is None or np.isfinite(v).sum() < max(2, 0.5 * len(cyc)):
            continue                                   # 无效通道 → 不展示
        keep.append(c)
        out[c] = v
    return cyc, keep, out


def ae_series(gid):
    """AE 事件(按 time 去重取 max energy) → (time, energy, 通道数)。"""
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    df = pd.read_csv(fp, encoding='utf-8-sig', usecols=['time', 'energy', 'channel'])
    n_ch = int(df['channel'].nunique())
    g = df.groupby('time', sort=True)['energy'].max()
    return g.index.to_numpy(float), g.to_numpy(float), n_ch


def dfos_block_series(gid):
    """DFOS 块级序列(局部峰 / rmse / HI / 空间点数) —— 复用 evaluate_l1_dfos 的缓存。"""
    cache = os.path.join(_deg.RES, f'_l1_dfos_hi_{gid}.npz')
    if not os.path.exists(cache):
        _dfos.analyze(gid)
    z = np.load(cache, allow_pickle=True)
    pos = np.asarray(z['pos'], float)
    return (np.asarray(z['cyc'], float), np.asarray(z['local'], float),
            np.asarray(z['rmse'], float), np.asarray(z['hi'], float), int(pos.size))


def dfos_profiles(gid, max_pos=420):
    """DFOS **逐块空间分布**(块内逐点中位曲线) → (位置, prof[nblk,npos], 块 cycle, 修复点数)。

    步骤: 全分辨率建块分布 → 沿块轴补 NaN → **共用 `evaluate_l1_dfos.despike_row` 去尖峰**
    (与 §4.2.4 的 DFOS HI 指标同一判据) → 空间降采样至 ~max_pos 点(控制包体积)。
    """
    td, pos, M = _dfos.load_dfos(gid)
    blk = _dfos.load_fbg_blocks(gid)
    nblk = len(blk)
    row_blk = _dfos.distribute_rows_to_blocks(td, blk)
    idx = np.arange(nblk)
    prof = np.full((nblk, pos.size), np.nan)
    n_fix = 0
    for k in range(nblk):
        pr = _dfos.block_profile(M, row_blk == k)
        if pr is None:
            continue
        pr, nfx = _dfos.despike_row(pr)          # 先去尖峰(与 HI 指标同判据/同顺序)
        n_fix += nfx
        prof[k] = pr
    # 再沿块轴补 NaN —— 仅为显示连续; 必须在去尖峰之后(否则插值会造出假尖峰)
    for j in range(prof.shape[1]):
        col = prof[:, j]
        m = np.isfinite(col)
        prof[:, j] = np.interp(idx, idx[m], col[m]) if m.any() else 0.0
    step = max(1, int(np.ceil(pos.size / float(max_pos))))
    sel = np.arange(0, pos.size, step)
    return pos[sel], prof[:, sel], (idx + 0.5) * CYCS_PER_BLK, n_fix


def levels_from(D):
    """D → 单调分级(在线语义: 级别只升不降)。"""
    lv = np.zeros(len(D), dtype=np.int8)
    cur = 0
    for i, v in enumerate(D):
        while cur < len(LEVELS) and v >= LEVELS[cur]:
            cur += 1
        lv[i] = cur
    return lv


def to_block_frames(arr, cyc_frame):
    """块级数组 → 逐帧数组(按 cycle 就近取块); NaN → 0。"""
    if len(arr) == 0:
        return np.zeros(len(cyc_frame))
    a = np.nan_to_num(np.asarray(arr, float), nan=0.0)
    b = np.clip((cyc_frame / CYCS_PER_BLK).astype(int), 0, len(a) - 1)
    return a[b]


# ------------------------------------------------------------------
# 打包一组
# ------------------------------------------------------------------
# 每块(EXT_BLOCK_PTS=500 点) 对应的帧数 = 500 / STEP
BLK_FRAMES = 500 // STEP


def de_step(arr, blk=BLK_FRAMES):
    """块级阶跃 → 块内线性插值（消除 D(t)/risk 的阶梯观感）。

    OnlineDamageIndex 每 EXT_BLOCK_PTS=500 点才结算一次块级统计, 故 D/risk/e_ae/e_st
    在时间轴上呈"每块一个平台"的阶梯。此处把每块内部线性铺开, 使曲线连续:
      - 块末值保持不变（只把"块末那次跳变"摊回整块内）;
      - 阈值跨越时刻最多提前不到一个块。
    仅用于展示; 分级 lv 由插值后的 D 重算, 保持内部一致。
    """
    a = np.asarray(arr, dtype=float)
    n = len(a)
    if n <= blk:
        return a
    out = a.copy()
    for s in range(blk, n, blk):
        e = min(s + blk, n)
        out[s:e] = np.linspace(a[s - 1], a[s], e - s + 1)[1:]
    return out


def pack(gid):
    print(f'\n=== 导出 {gid} ===')
    nf = _deg.META[gid]['n_f']

    # --- 正式口径 D(t)（基线重定义 + 应变漂移证据 + rise=0.05）---
    r = _deg.run_group(gid, dict(PARAMS), baseline=True, strain_ev=True, fusion=FUSION)
    cyc, D = r['cyc'], r['D']
    nb = len(cyc)
    nfr = int(np.ceil(nb / STEP))

    # --- FBG 逐通道(块级 → 逐点插值) + AE 事件级 ---
    bcyc, fo_cols, fo_blk = fbg_block_channels(gid)
    strain = np.nan_to_num(np.asarray(r['strain'], float), nan=0.0)
    fo_pt = {c: np.interp(cyc, bcyc, v, left=v[0], right=v[-1]) for c, v in fo_blk.items()}
    tae, eae, n_ae_ch = ae_series(gid)
    dfos_cyc, dfos_local, dfos_rmse, dfos_hi, n_pos = dfos_block_series(gid)
    dpos, dprof, dcyc, n_spike = dfos_profiles(gid)

    # --- 逐点 → 逐帧(取每帧最后一个点, 与主样本导出器同口径) ---
    j = np.minimum((np.arange(nfr) + 1) * STEP - 1, nb - 1)
    cyc_f = cyc[j]

    D_f = D[j]
    risk_f = r['risk'][j]
    eae_f = r['eae'][j]
    est_f = r['est'][j]
    st_f = strain[j]
    lv_f = levels_from(D)[j]

    peak_pt = np.asarray(r['peak'], float)
    peak_f = peak_pt[j]
    aen = np.zeros(nfr, dtype=np.int32)
    ael = np.full(nfr, AE_EMPTY, dtype=np.int32)
    for f in range(nfr):
        lo = f * STEP
        hi = min(lo + STEP, nb)
        seg = peak_pt[lo:hi]
        m = seg > 0
        if m.any():
            aen[f] = int(m.sum())
            ael[f] = _i1000(np.log10(float(seg[m].max())))
    fo_f = {c: v[j] for c, v in fo_pt.items()}
    dl_f = to_block_frames(dfos_local, cyc_f)
    dr_f = to_block_frames(dfos_rmse, cyc_f)
    dh_f = to_block_frames(dfos_hi, cyc_f)
    nblk_p = dprof.shape[0]
    blk_of = np.clip((cyc_f / CYCS_PER_BLK).astype(int), 0, nblk_p - 1)

    # --- 摘要 ---
    def first_ge_pct(th):
        idx = np.where(D >= th)[0]
        return round(float(cyc[idx[0]]) / nf * 100.0, 1) if len(idx) else None

    # --- 论文 HI_F（离线参考曲线；方案A：与在线 D(t) **并列**展示，不替换）---
    # HI_F = 0.5·HI_AE + 0.5·HI_OF（Broer 2021 式 7，两项各 unity01 归一化到 [0,1]）。
    # ⚠️ 归一化需全寿命最大值 ⇒ **非因果/离线**，仅作参考基线。
    # 叠加的意义：既保留在线因果 D(t) 作为主指标，又把「与论文口径的差异」量化且可见。
    hiF_f, hi85_pct = None, None
    try:
        r4 = _l4.level4(gid)
        c4 = np.asarray(r4['cyc'], float)
        h4 = np.asarray(r4['HI_F'], float)
        o = np.argsort(c4)
        hiF_f = np.interp(cyc_f, c4[o], h4[o])
        ix = np.where(hiF_f >= 0.85)[0]
        hi85_pct = round(float(cyc_f[ix[0]]) / nf * 100.0, 1) if len(ix) else None
    except Exception as e:                       # noqa: BLE001
        print(f'  [warn] {gid} HI_F 参考曲线不可用: {e}')
    d85_pct = first_ge_pct(0.85)
    hi_lead = (round(hi85_pct - d85_pct, 1)      # 正 = D(t) 早于论文 HI_F
               if (d85_pct is not None and hi85_pct is not None) else None)

    meta = {
        'D_end': round(float(D[-1]), 3),
        't25': first_ge_pct(0.25), 't55': first_ge_pct(0.55), 't85': d85_pct,
        'b2': None,                                  # L1 无弱标签 b2(前端自动隐藏)
        'b3': 100.0,                                 # 失效锚 = n_f
        'c0Pct': round(float(r['c0']) / nf * 100.0, 1) if r['c0'] > 0 else None,
        'refs': [{'pct': round(c / nf * 100.0, 1), 'label': lab}
                 for lab, c in _deg.META[gid]['refs']],
        'aeEvents': int(aen.sum()),
        'nFo': len(fo_cols),
        'nDfos': n_pos,
        # --- 论文 HI_F 离线参考（方案A）---
        'hiF85': hi85_pct,        # HI_F 达 0.85 的寿命% (None = 未达)
        'hiLeadPt': hi_lead,      # D 相对 HI_F 的提前量(百分点, 正 = D 更早)
    }

    # --- 块级阶跃 → 块内线性插值(展示连续化; 块级口径不变) ---
    D_f = de_step(D_f)
    risk_f = de_step(risk_f)
    eae_f = de_step(eae_f)
    est_f = de_step(est_f)
    lv_f = levels_from(D_f)

    # --- 分级升级事件(逐帧, 在线语义) ---
    warn = []
    last = 0
    for f in range(nfr):
        lv = int(lv_f[f])
        if lv > last:
            for g in range(last + 1, lv + 1):
                warn.append({'f': f, 't': round(float(cyc_f[f]), 1), 'lv': g})
            last = lv

    # --- AE log 峰值纵轴范围(自动) ---
    fin = ael[ael != AE_EMPTY]
    if fin.size:
        ax_ael = [round(float(fin.min()) / 1000.0 - 0.3, 2),
                  round(float(fin.max()) / 1000.0 + 0.3, 2)]
    else:
        ax_ael = [-1.0, 1.0]
    rate_hi = max(2.0, float(aen.max()) * 1.2)

    chans = [
        {'key': 'ae', 'name': '声发射 AE', 'mode': '事件流', 'n': str(n_ae_ch), 'unit': '通道'},
        {'key': 'fo', 'name': '光纤光栅 FBG', 'mode': '5000 cyc/块', 'n': str(len(fo_cols)), 'unit': '通道'},
        {'key': 'dfos', 'name': '分布式应变 DFOS', 'mode': '5000 cyc/块', 'n': str(n_pos), 'unit': '点'},
        {'key': 'engine', 'name': '损伤度引擎', 'mode': '500 点/块', 'n': None, 'unit': ''},
    ]

    pkg = {
        'gid': gid, 'ds': 'l1', 'unit': 'cycle',
        'xLabel': '寿命 / cycle',
        'rig': 'L1 压缩-压缩疲劳 (BVID 后)',
        'mode': '多源同步 · 5000 cycle/块',
        'n': int(nb), 'nfr': int(nfr), 'step': STEP,
        'frameDt': float(STEP * CYCS_PER_PT), 'dt': float(STEP * CYCS_PER_PT),
        'dur': float(nf), 'c0': float(r['c0']),
        'foCols': list(fo_cols), 'chans': chans,
        'meta': meta, 'warn': warn,
        # --- 波形数组(长度均 = nfr) ---
        'D': [_i1000(x) for x in D_f],
        'hiF': ([_i1000(x) for x in hiF_f] if hiF_f is not None else None),
        'risk': [_i1000(x) for x in risk_f],
        'eae': [_i1000(x) for x in eae_f],
        'est': [_i1000(x) for x in est_f],
        'lv': [int(x) for x in lv_f],
        'st': [_i100(x) for x in st_f],
        'ael': [int(x) for x in ael],
        'aen': [int(x) for x in aen],
        't': [round(float(x), 1) for x in cyc_f],
        'fo': {c: [_i100(x) for x in v] for c, v in fo_f.items()},
        'dfos': {
            'local': [_i100(x) for x in dl_f],
            'rmse': [_i100(x) for x in dr_f],
            'hi': [_i1000(x) for x in dh_f],
            'nPos': n_pos,
            # --- 空间分布(逐块中位曲线) ---
            'nblk': int(nblk_p), 'npos': int(dprof.shape[1]),
            'pos': [round(float(x), 1) for x in dpos],
            'cyc': [round(float(x), 1) for x in dcyc],
            'base': [_i100(x) for x in dprof[0]],
            'prof': [_i100(x) for x in dprof.reshape(-1)],
            'blkOf': [int(x) for x in blk_of],
            'cycPerBlk': CYCS_PER_BLK,
            'spikeN': int(n_spike),          # 被空间去尖峰修复的采样点数
        },
        'ax': {'ael': ax_ael, 'rate': round(rate_hi, 1)},
    }
    return pkg


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def write_js(path, var_js):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(var_js)
    return os.path.getsize(path) / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='*', default=None, help='默认全部 4 组')
    ap.add_argument('--out', default=os.path.join(PROJ, 'dashboard', 'data'),
                    help='输出目录(默认主样本看板的 data/)')
    a = ap.parse_args()

    groups = a.groups or list(_deg.META.keys())
    outdir = os.path.abspath(a.out)
    os.makedirs(outdir, exist_ok=True)
    print(f'输出目录: {outdir}')

    index = []
    for gid in groups:
        pkg = pack(gid)
        js = ('window.SHM_DATA=window.SHM_DATA||{};'
              f'window.SHM_DATA["{gid}"]=' +
              json.dumps(pkg, ensure_ascii=False, separators=(',', ':')) + ';\n')
        size = write_js(os.path.join(outdir, f'{gid}.js'), js)
        print(f'  写出 {gid}.js  ({size:.0f} KB)  帧数={pkg["nfr"]}  '
              f'D_end={pkg["meta"]["D_end"]}  t85={pkg["meta"]["t85"]}%  '
              f'c0={pkg["meta"]["c0Pct"]}%  AE事件={pkg["meta"]["aeEvents"]}  '
              f'FBG={pkg["meta"]["nFo"]}ch  DFOS={pkg["meta"]["nDfos"]}pt  '
              f'空间块={pkg["dfos"]["nblk"]}×{pkg["dfos"]["npos"]}  去尖峰={pkg["dfos"]["spikeN"]}点  '
              f'ael轴={pkg["ax"]["ael"]}')
        index.append({'gid': gid, 'nfr': pkg['nfr'], 'dur': pkg['dur'], 'n': pkg['n'],
                      'meta': pkg['meta'], 'foCols': pkg['foCols'], 'warn': pkg['warn'],
                      'unit': 'cycle', 'frameDt': pkg['frameDt']})

    ds = {'id': 'l1', 'name': 'ReMAP / TU-Delft L1', 'unit': 'cycle',
          'path': 'data/', 'groups': index}
    js = ('window.SHM_DATASETS=window.SHM_DATASETS||{};'
          'window.SHM_DATASETS["l1"]=' +
          json.dumps(ds, ensure_ascii=False, separators=(',', ':')) + ';\n')
    ipath = os.path.join(outdir, 'index_l1.js')
    write_js(ipath, js)
    print(f'\n写出清单 {ipath}')
    print('提示: dashboard/index.html 已内置 <script src="data/index_l1.js">，刷新即见。')


if __name__ == '__main__':
    main()
