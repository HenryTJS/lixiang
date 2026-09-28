# -*- coding: utf-8 -*-
"""L1 第二批（L1-49..L1-60，**无 FBG**）→ 在线监测看板数据包
=================================================================

把第二批试件的**逐点在线**结果打包成与第一批（`export_dashboard_l1.py`）
**同格式**的 JS 数据文件，落在同一 `dashboard/data/` 目录。

前端**零改动**：`dashboard.js` 完全按数据包自带字段自适应
（无 FBG → FBG 面板自动显示"未接入"；有 `dfos` → 自动启用空间分布面板；
`chans` 驱动通道健康表；`ax` 驱动 AE 纵轴）。

与第一批的差异
--------------
| 项           | 第一批 (L1-03/04/05/09)            | 第二批 (L1-49..L1-60)                        |
| ------------ | ---------------------------------- | -------------------------------------------- |
| FBG          | 10 通道块级应变                    | **无** → `foCols=[]`                       |
| 应变证据     | FBG 块级循环幅值                   | **DFOS 左脚/右脚段级均值**（段内压缩峰值行） |
| 时间锚       | FBG 块锚：块 k ↔ (k+0.5)×5000     | DFOS 测量段锚（markers 墙钟，10-cycle 网格） |
| DFOS 空间分布 | FBG 块划分 + 块内逐点中位          | **段 = 块**，直接用 step0_v2 的段级中位矩阵  |
| 参考锚       | 论文检测点 + `b2`/`b3`         | 无论文检测点；`b2=None`、`b3=n_f`        |

D(t) 口径 = `evaluate_l1_degree_v2.group_series()`（AE 段内 √能量峰值 + DFOS 脚部应变，
`OnlineDamageIndex` 默认参数）—— 与本批 `results/l1_degree_v2.csv` 完全一致。

⚠️ **2026-09-16 变更：主指标改用「离线复评 HI_AE」**
----------------------------------------------------------------
本批经**四条独立路径**检验后确认**不支持在线预警**（1. AE 单源；2. DFOS 四口径；
3. AE×DFOS 融合；4. 严重度分工），推导链见 `docs/details.md` §12.8~§12.13。
唯一成立的交付是**离线损伤复评**，故看板主曲线改为：

    HI_AE = unity01(AE 累积事件数)      （文献口径：Broer 2021 / Galanopoulos 2021）

其自归一化后 t85（达 0.85 的寿命百分比）**中位 92.1%**，与第一批 **92.6%** 一致。

**该口径是非因果的**（unity01 需全寿命最大值）—— 这正是它不能在线用的原因。
看板仍保留逐帧回放以便观察趋势，但**不声称在线语义**：
`ds.name` / `pkg.mode` / `chans[engine].mode` 三处均已标注「离线复评」。

两路**因果**指标作为证据通道同时展示（供观察，不作阀值报警）：
  - `eae` = unity01(最近 5000 cycle 的 AE 事件数)     ← AE 活动度
  - `est` = unity01(DFOS 段级 `local`)                 ← 空间脱粘特征
输出
----
  dashboard/data/L1-49.js ... L1-60.js
  dashboard/data/index_l1v2.js     (window.SHM_DATASETS['l1v2'])

用法
----
  python l1/export_dashboard_l1_v2.py                       # 全部 9 组
  python l1/export_dashboard_l1_v2.py L1-49 L1-55           # 指定组
  python l1/export_dashboard_l1_v2.py --out <dir>           # 自定义输出目录

作者: Roo   日期: 2026-09-15
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
sys.path.insert(0, ROOT)                                   # l1_meta / evaluate_* 等同级模块
os.chdir(ROOT)

import evaluate_l1_dfos as _dfos                           # noqa: E402
import evaluate_l1_degree_v2 as _v2                        # noqa: E402
import evaluate_l1_hi_ae as _hiae                          # noqa: E402
import l1_time_align as ta                                 # noqa: E402
from l1_meta import load_meta                              # noqa: E402

RES = os.path.join(ROOT, 'results')
os.makedirs(RES, exist_ok=True)

GROUPS = list(_v2.GROUPS)
STEP = 5                      # 降采样: 5 点(10-cycle 网格) = 50 cycle/帧 —— 与第一批一致
CYCS_PER_PT = 10              # v2 网格步长(cycle/点), 见 evaluate_l1_degree_v2.group_series
FRAME_CYC = STEP * CYCS_PER_PT     # 每帧代表的 cycle 数 = 50
LEVELS = [0.25, 0.55, 0.85]
AE_EMPTY = -99999             # ael 空值标记(与前端约定一致)
MAX_POS = 420                 # DFOS 空间降采样点数(与第一批一致, 控制包体积)
MAX_SEG = 200                 # DFOS 段轴降采样上限(第二批每 500 cycle 一段, 可上千段)
BLK_FRAMES = 500 // STEP      # 每块(EXT_BLOCK_PTS=500 点) 对应的帧数


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


# ------------------------------------------------------------------
# 基础工具（与 export_dashboard_l1.py 同口径）
# ------------------------------------------------------------------
def _i100(x):
    return int(round(float(x) * 100.0))


def _i1000(x):
    return int(round(float(x) * 1000.0))


def unity01(x):
    """unity 归一化到 [0,1]（文献口径式1）。⚠️ 用全局 min/max → **非因果**。"""
    x = np.asarray(x, float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if mx - mn < 1e-12 else (x - mn) / (mx - mn)


def hi_ae_series(gid, cyc_grid):
    """离线复评主指标 HI_AE，插值到给定 cycle 网格；缺 AE 返回 None。

    HI_AE = unity01(AE 累积事件数)（500-cycle 箱）。详见模块 docstring 与 docs §12.8。
    """
    s = _hiae.hi_of_group(gid)
    if s is None:
        return None
    return np.interp(cyc_grid, s['cyc'], unity01(s['cum_hits']), left=0.0, right=1.0)


def ae_rate_series(gid, cyc_grid):
    """证据通道 eae：unity01(最近 5000 cycle 的 AE 事件数)（**因果**）。"""
    s = _hiae.hi_of_group(gid)
    if s is None:
        return None
    v = unity01(_hiae._roll_sum(s['hit_bin'], 10))
    return np.interp(cyc_grid, s['cyc'], v, left=0.0, right=1.0)


def levels_from(D):
    """D → 单调分级(在线语义: 级别只升不降)。"""
    lv = np.zeros(len(D), dtype=np.int8)
    cur = 0
    for i, v in enumerate(D):
        while cur < len(LEVELS) and v >= LEVELS[cur]:
            cur += 1
        lv[i] = cur
    return lv


def to_seg_frames(arr, seg_cyc, cyc_frame):
    """段级数组 → 逐帧数组（取"cycle ≤ 当前帧"的最近一段）；NaN → 0。"""
    if arr is None or len(arr) == 0:
        return np.zeros(len(cyc_frame))
    a = np.nan_to_num(np.asarray(arr, float), nan=0.0)
    seg = np.asarray(seg_cyc, float)
    b = np.clip(np.searchsorted(seg, cyc_frame, side='right') - 1, 0, len(a) - 1)
    return a[b]


# ------------------------------------------------------------------
# 数据提取
# ------------------------------------------------------------------
def dfos_seg_series(gid):
    """段级 local / rmse / HI（复用 evaluate_l1_dfos 的缓存, 缺则现算）。"""
    cache = os.path.join(RES, f'_l1_dfos_hi_{gid}.npz')
    if not os.path.exists(cache):
        _dfos.analyze(gid)
    z = np.load(cache, allow_pickle=True)
    pos = np.asarray(z['pos'], float)
    return (np.asarray(z['cyc'], float), np.asarray(z['local'], float),
            np.asarray(z['rmse'], float), np.asarray(z['hi'], float), int(pos.size))


def dfos_profiles_v2(gid, max_pos=MAX_POS, max_seg=MAX_SEG):
    """段级空间分布 → (位置, prof[nseg,npos], 段 cycle, 去尖峰点数)。

    第二批**段 = 块**：`step0_v2.py` 已把每段压成"段内逐点中位数"一行，
    故直接取该矩阵（只保留 AI 段, 排除冲击前 BI 段），
    逐段用与 §4.2.4 同一判据去尖峰，再沿段轴补 NaN（仅为显示连续），
    最后**段轴**降采样到 max_seg、**空间轴**降采样到 max_pos 点。
    """
    _t, pos, M = _dfos.load_dfos(gid)
    nf = load_meta(gid)['n_f']
    cyc_all, keep = ta.dfos_cycles(gid, nf, include_bi=True)
    if cyc_all.size != M.shape[0]:
        print(f'  [{gid}] 段数({cyc_all.size}) ≠ DFOS 行数({M.shape[0]}) → 跳过空间分布')
        return None
    M = M[keep]
    cyc_all = np.asarray(cyc_all, float)[keep]
    ai = cyc_all >= 0                                  # 只用冲击后(AI)段
    M, cyc_ai = M[ai], cyc_all[ai]
    if M.shape[0] < 2:
        return None
    nseg = M.shape[0]
    prof = np.full(M.shape, np.nan)
    n_fix = 0
    for k in range(nseg):
        prof[k], nfx = _dfos.despike_row(M[k])
        n_fix += nfx
    idx = np.arange(nseg)
    for jj in range(prof.shape[1]):
        col = prof[:, jj]
        m = np.isfinite(col)
        prof[:, jj] = np.interp(idx, idx[m], col[m]) if m.any() else 0.0
    # 段轴降采样 —— 第二批段数可达上千（每 500 cycle 一段），全存既臃肿、热图也无法分辨
    if nseg > max_seg:
        ksel = np.unique(np.linspace(0, nseg - 1, max_seg).astype(int))
        prof = prof[ksel]
        cyc_ai = cyc_ai[ksel]
    # 空间轴降采样
    pstep = max(1, int(np.ceil(pos.size / float(max_pos))))
    psel = np.arange(0, pos.size, pstep)
    return pos[psel], prof[:, psel], cyc_ai, int(n_fix)


# ------------------------------------------------------------------
# 打包一组
# ------------------------------------------------------------------
def pack(gid):
    print(f'\n=== 导出 {gid} ===')
    nf = load_meta(gid)['n_f']

    # --- 主指标：离线复评 HI_AE（docs §12.8）---
    #   ⚠️ 非因果（unity01 需全寿命最大值）→ 本数据集定位为「离线复评」，界面已标注。
    #   `group_series` 仍用于取 10-cycle 网格与 DFOS 脚部应变（st 通道）。
    r = _v2.group_series(gid)
    if r is None:
        print(f'  [{gid}] 缺数据 → 跳过')
        return None
    cyc, strain = r[0], r[1]
    nb = len(cyc)
    nfr = int(np.ceil(nb / STEP))

    D = hi_ae_series(gid, cyc)                  # 主曲线 = 离线复评 HI_AE
    if D is None:
        print(f'  [{gid}] 缺 AE → 跳过')
        return None
    risk = D.copy()                             # 前端分级/发光逻辑沿用 risk
    eae = ae_rate_series(gid, cyc)              # 因果证据：AE 活动度（5000 cycle 窗）
    if eae is None:
        eae = np.zeros(nb)
    # 因果证据：DFOS 空间脱粘特征（段级，下面 dfos_seg_series 取到后填）
    est = None

    # --- AE（1 s bin, 经 markers 墙钟映射到 cycle）---
    ae_cyc = np.zeros(0)
    ae_e = np.zeros(0)
    ae_n = np.zeros(0)
    fp = os.path.join(ROOT, gid, f'{gid}声发射.csv')
    if os.path.exists(fp):
        ae = pd.read_csv(fp, encoding='utf-8-sig')
        if len(ae):
            ae_cyc = np.asarray(ta.time_to_cycle(gid, ae['time'].to_numpy(float), nf,
                                                 clip_out_of_life=True), float)
            ae_e = np.sqrt(np.maximum(ae['energy'].to_numpy(float), 0.0))
            ae_n = ae['n_hits'].to_numpy(float) if 'n_hits' in ae.columns \
                else np.ones(len(ae), float)
            ok = np.isfinite(ae_cyc)          # 剔除冲击前(BI)事件, 与 D(t) 同口径
            ae_cyc, ae_e, ae_n = ae_cyc[ok], ae_e[ok], ae_n[ok]

    # --- 段级 DFOS 指标 + 空间分布 ---
    dfos_cyc, dfos_local, dfos_rmse, dfos_hi, n_pos = dfos_seg_series(gid)
    profs = dfos_profiles_v2(gid)
    if profs is None:
        dpos = np.zeros(0)
        dprof = np.zeros((1, 1))
        dcyc = np.zeros(1)
        n_spike = 0
    else:
        dpos, dprof, dcyc, n_spike = profs

    # --- 逐点 → 逐帧（取每帧末点, 与第一批同口径）---
    j = np.minimum((np.arange(nfr) + 1) * STEP - 1, nb - 1)
    cyc_f = cyc[j]
    D_f, risk_f = D[j], risk[j]
    eae_f = eae[j]
    est_f = to_seg_frames(unity01(dfos_local) if dfos_local is not None else None,
                          dfos_cyc, cyc_f)
    st_f = strain[j]
    lv_f = levels_from(D)[j]

    # --- 帧内 AE 聚合: 事件数求和 + log 峰值取最大 ---
    aen = np.zeros(nfr, dtype=np.int32)
    ael = np.full(nfr, AE_EMPTY, dtype=np.int32)
    if ae_cyc.size:
        order = np.argsort(ae_cyc)
        ac, ae_s, an_s = ae_cyc[order], ae_e[order], ae_n[order]
        lo = np.searchsorted(ac, cyc_f - FRAME_CYC / 2.0, side='left')
        hi = np.searchsorted(ac, cyc_f + FRAME_CYC / 2.0, side='right')
        for f in range(nfr):
            if hi[f] > lo[f]:
                aen[f] = int(round(float(an_s[lo[f]:hi[f]].sum())))
                pk = float(ae_s[lo[f]:hi[f]].max())
                if pk > 0:
                    ael[f] = _i1000(np.log10(pk))

    dl_f = to_seg_frames(dfos_local, dfos_cyc, cyc_f)
    dr_f = to_seg_frames(dfos_rmse, dfos_cyc, cyc_f)
    dh_f = to_seg_frames(dfos_hi, dfos_cyc, cyc_f)
    nblk_p = dprof.shape[0]
    blk_of = np.clip(np.searchsorted(np.asarray(dcyc, float), cyc_f, side='right') - 1,
                     0, nblk_p - 1) if nblk_p > 1 else np.zeros(nfr, int)

    # --- 摘要 ---
    def first_ge_pct(th):
        idx = np.where(D >= th)[0]
        return round(float(cyc[idx[0]]) / nf * 100.0, 1) if len(idx) else None

    meta = {
        'D_end': round(float(D[-1]), 3),
        't25': first_ge_pct(0.25), 't55': first_ge_pct(0.55), 't85': first_ge_pct(0.85),
        'b2': None,                       # 第二批无弱标签 b2 → 前端自动隐藏
        'b3': 100.0,                      # 失效锚 = n_f
        'c0Pct': None,                    # 第二批不做基线重定义 → 无 c0 锚
        'refs': [],                       # 无论文检测点(仅 03/04/05 收录于 Broer 2021)
        'aeEvents': int(aen.sum()),
        'nFo': 0,
        'nDfos': int(n_pos),
    }

    # --- 块级阶跃 → 块内线性插值(展示连续化) ---
    # 注: 主曲线 HI_AE 已由 np.interp 插值到 10-cycle 网格（无块级台阶），无需 de_step。
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

    chans = [                        # 前端按此渲染通道健康表(无 FBG 行)
        {'key': 'ae', 'name': '声发射 AE', 'mode': '事件流(1 s bin)', 'n': '4', 'unit': '通道'},
        {'key': 'dfos', 'name': '分布式应变 DFOS', 'mode': '500 cycle/段',
         'n': str(n_pos), 'unit': '点'},
        {'key': 'engine', 'name': '离线复评 HI_AE', 'mode': 'AE 累积·需全寿命归一',
         'n': None, 'unit': ''},
    ]

    pkg = {
        'gid': gid, 'ds': 'l1v2', 'unit': 'cycle',
        'xLabel': '寿命 / cycle',
        'rig': 'L1 压缩-压缩疲劳 (BVID 后, 无 FBG)',
        'mode': 'AE + DFOS · 离线复评（非在线）',
        # 主指标名称（覆盖前端默认的“损伤度 D(t)”）
        'indexName': '离线复评 HI_AE（非在线）',
        'trendName': 'HI_AE 趋势 / 阈值 0.25 · 0.55 · 0.85',
        'labels': {'unit': 'HI_AE (0~1)', 'legend': 'HI_AE',
                   'margin': '1 − HI_AE'},
        'n': int(nb), 'nfr': int(nfr), 'step': STEP,
        'frameDt': float(FRAME_CYC), 'dt': float(FRAME_CYC),
        'dur': float(nf), 'c0': 0.0,
        'foCols': [],                 # 无 FBG → 前端 FBG 面板显示"未接入"
        'chans': chans,
        'meta': meta, 'warn': warn,
        # --- 波形数组(长度均 = nfr) ---
        'D': [_i1000(x) for x in D_f],
        'risk': [_i1000(x) for x in risk_f],
        'eae': [_i1000(x) for x in eae_f],
        'est': [_i1000(x) for x in est_f],
        'lv': [int(x) for x in lv_f],
        'st': [_i100(x) for x in st_f],
        'ael': [int(x) for x in ael],
        'aen': [int(x) for x in aen],
        't': [round(float(x), 1) for x in cyc_f],
        'fo': {},
        'dfos': {
            'local': [_i100(x) for x in dl_f],
            'rmse': [_i100(x) for x in dr_f],
            'hi': [_i1000(x) for x in dh_f],
            'nPos': int(n_pos),
            # --- 空间分布(逐段中位曲线) ---
            'nblk': int(nblk_p), 'npos': int(dprof.shape[1]),
            'pos': [round(float(x), 1) for x in dpos],
            'cyc': [round(float(x), 1) for x in dcyc],
            'base': [_i100(x) for x in dprof[0]],
            'prof': [_i100(x) for x in dprof.reshape(-1)],
            'blkOf': [int(x) for x in blk_of],
            'cycPerBlk': float(np.median(np.diff(np.asarray(dcyc, float))))
            if len(dcyc) > 1 else float(FRAME_CYC),
            'spikeN': int(n_spike),
        },
        'ax': {'ael': ax_ael, 'rate': round(rate_hi, 1)},
    }
    return pkg


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def write_js(path, text):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    return os.path.getsize(path) / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('groups', nargs='*', default=None, help=f'默认全部 {len(GROUPS)} 组')
    ap.add_argument('--out', default=os.path.join(PROJ, 'dashboard', 'data'),
                    help='输出目录(默认 dashboard/data/)')
    a = ap.parse_args()

    groups = a.groups or list(GROUPS)
    outdir = os.path.abspath(a.out)
    os.makedirs(outdir, exist_ok=True)
    print(f'输出目录: {outdir}')

    index = []
    for gid in groups:
        pkg = pack(gid)
        if pkg is None:
            continue
        js = ('window.SHM_DATA=window.SHM_DATA||{};'
              f'window.SHM_DATA["{gid}"]=' +
              json.dumps(pkg, ensure_ascii=False, separators=(',', ':')) + ';\n')
        size = write_js(os.path.join(outdir, f'{gid}.js'), js)
        print(f'  写出 {gid}.js  ({size:.0f} KB)  帧数={pkg["nfr"]}  n_f={pkg["dur"]:.0f}  '
              f'D_end={pkg["meta"]["D_end"]}  t85={pkg["meta"]["t85"]}%  '
              f'AE事件={pkg["meta"]["aeEvents"]}  DFOS={pkg["meta"]["nDfos"]}pt  '
              f'空间段={pkg["dfos"]["nblk"]}×{pkg["dfos"]["npos"]}  去尖峰={pkg["dfos"]["spikeN"]}点  '
              f'ael轴={pkg["ax"]["ael"]}')
        index.append({'gid': gid, 'nfr': pkg['nfr'], 'dur': pkg['dur'], 'n': pkg['n'],
                      'meta': pkg['meta'], 'foCols': pkg['foCols'], 'warn': pkg['warn'],
                      'unit': 'cycle', 'frameDt': pkg['frameDt']})

    ds = {'id': 'l1v2', 'name': 'L1 第二批 · 离线复评(非在线)', 'unit': 'cycle',
          'path': 'data/', 'groups': index}
    ipath = os.path.join(outdir, 'index_l1v2.js')
    write_js(ipath, 'window.SHM_DATASETS=window.SHM_DATASETS||{};'
             'window.SHM_DATASETS["l1v2"]=' +
             json.dumps(ds, ensure_ascii=False, separators=(',', ':')) + ';\n')
    print(f'\n写出清单 {ipath}')
    print('提示: dashboard/index.html 已内置 <script src="data/index_l1v2.js">，刷新即见。')


if __name__ == '__main__':
    main()
