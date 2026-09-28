# -*- coding: utf-8 -*-
"""导出疲劳机多源在线监测看板数据
=====================================

把**疲劳机试件**的**逐点在线流式**结果
(连续损伤度 D(t)、证据层 e_ae/e_strain/risk、分级 level、各源原始信号)
降采样打包为前端直接可用的 JS 数据文件。

纳入试件（`DASH_GROUPS`，共 11 组）：**016-020 + 022-027**。
  - 015（数据列异常）与 021（5% 就早报，见 README §3.7）列为**问题组，不纳入**；
  - 023/024/025 **无光纤文件** → `foCols=[]`，看板 FBG 面板显示「未接入」（正常）；
  - 022-027 **无 b2/b3 标签** → `meta.b2/b3 = null`（不给假锚）。

导出字段已包含**应变 `st`** 与**光纤 `fo`**（前端 FBG / STRAIN 面板直接消费）。

严格复用正式口径(见 README):
  - 流式入口 : shm.streaming.StreamSimulator     (逐点, 零未来信息)
  - 损伤模型 : shm.damage_index.OnlineDamageIndex (默认参数 = 正式方法)

输出:
  <项目根>/dashboard/data/{gid}.js   每组一个数据包 (window.SHM_DATA[gid] = {...})
  <项目根>/dashboard/data/index.js   组清单与摘要 (window.SHM_INDEX = [...])

数值编码(压缩体积, 前端解码):
  D / risk / e_ae / e_strain : ×1000 整数
  strain / fo                : ×100 整数
  aeLog                      : log10(peak)×1000 整数, 无事件 = -99999
  aeN                        : 帧内 AE 事件数
  level                      : 0/1/2/3 整数

作者: Roo   日期: 2026-09-10（2026-09-20 扩展至 11 组）
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))   # <项目根>\main
PROJ = os.path.dirname(ROOT)                        # 项目根
sys.path.insert(0, PROJ)                     # 项目根(含 shm)
sys.path.insert(0, ROOT)                     # main/(eval_common)
os.chdir(ROOT)

from eval_common import ref_map                      # noqa: E402
from shm.config import EXT_BLOCK_PTS, GRADE_GATE     # noqa: E402
from shm.damage_index import OnlineDamageIndex       # noqa: E402
from shm.streaming import StreamSimulator            # noqa: E402

OUTDIR = os.path.join(PROJ, 'dashboard', 'data')   # 看板在项目根(与 main/l1 同级)
STEP = 5                 # 降采样步长(点): 10Hz → 2Hz 帧
AE_EMPTY = -99999        # aeLog 空值标记
LEVEL_NAMES = ['正常', '注意', '预警', '临危']

# --- 看板纳入的试件（2026-09-20 扩展）---
# 016-020 = 正式主样本；022-027 = 同台架的其他试件。
# ⚠️ 015（数据列异常）与 **021（5% 就早报，见 README §3.7）列为问题组，不纳入**。
DASH_GROUPS = ['016', '017', '018', '019', '020',
               '022', '023', '024', '025', '026', '027']

# 逐组展示元信息 (类型, 峰值载荷, 备注) —— 下拉与日志展示用。
# 依据 `main/数据记录.xlsx` 的加载协议表。
SPEC_META = {
    '016': ('失效型', '10 kN', ''),
    '017': ('失效型', '10 kN', ''),
    '018': ('失效型', '10 kN', ''),
    '019': ('失效型', '10 kN', ''),
    '020': ('失效型', '8 kN', ''),
    '022': ('循环+静力', '8 kN', '300 cycle 后静力加载'),
    '023': ('失效型', '10 kN', ''),
    '024': ('失效型', '8 kN', ''),
    '025': ('失效型', '8 kN', '断裂前 AE 沉寂，靠形状证据检出'),
    '026': ('失效型', '10 kN', '断裂后余段较长'),
    '027': ('循环+静力', '8 kN', '1000 cycle 后静力加载'),
}


def _i100(x):
    return int(round(float(x) * 100.0))


def _i1000(x):
    return int(round(float(x) * 1000.0))


# 每块(EXT_BLOCK_PTS=500 点) 对应的帧数 = 500 / STEP
BLK_FRAMES = EXT_BLOCK_PTS // STEP


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


def _pack(gid):
    """流式跑一组 → 数据包(dict)。"""
    _m = SPEC_META.get(gid, ('', '', ''))
    sim = StreamSimulator(gid)
    n = sim.load_data()
    di = OnlineDamageIndex(GRADE_GATE.get(gid, {}))   # 逐组级别闸门(见 shm/config.py)
    nfr = (n + STEP - 1) // STEP

    T = np.zeros(nfr, dtype=np.float64)
    Df = np.zeros(nfr, dtype=np.int32)
    RISK = np.zeros(nfr, dtype=np.int32)
    EAE = np.zeros(nfr, dtype=np.int32)
    EST = np.zeros(nfr, dtype=np.int32)
    LV = np.zeros(nfr, dtype=np.int16)
    ST = np.zeros(nfr, dtype=np.int32)
    AEL = np.full(nfr, AE_EMPTY, dtype=np.int32)
    AEN = np.zeros(nfr, dtype=np.int32)
    FO = {c: np.zeros(nfr, dtype=np.int32) for c in sim.fo_cols}
    Dfull = np.zeros(n, dtype=np.float32)

    i = 0
    while sim.has_next():
        p = sim.next_point()
        f = i // STEP
        peak = None
        if p.get('ae_new') and p.get('ae'):
            try:
                peak = float(p['ae'].get('ae_Peak', 0.0) or 0.0)
            except (TypeError, ValueError):
                peak = 0.0
        d = di.update(p['strain'], peak, None, di.shape_value(p.get('ae')))
        Dfull[i] = d

        T[f] = float(p['time'])
        Df[f] = _i1000(d)
        RISK[f] = _i1000(di.risk)
        EAE[f] = _i1000(di._last_e_ae)
        EST[f] = _i1000(di._last_e_strain)
        LV[f] = int(di.level)
        if p['strain'] is not None:
            ST[f] = _i100(p['strain'])
        fo = p.get('fo') or {}
        for c, arr in FO.items():
            v = fo.get(c)
            if v is not None:
                arr[f] = _i100(v)
        if peak is not None and peak > 0:
            lg = _i1000(np.log10(peak))
            AEL[f] = lg if AEL[f] == AE_EMPTY else max(AEL[f], lg)
            AEN[f] += 1
        i += 1

    sim.cleanup()

    # --- 摘要统计(全分辨率 D) ---
    def first_ge(th):
        idx = np.where(Dfull >= th)[0]
        return round(float(idx[0]) / n * 100.0, 1) if len(idx) else None

    refs = ref_map().get(gid, (None, None))    # 022-027 无 b2/b3 → 不给假锚
    t25, t55, t85 = first_ge(0.25), first_ge(0.55), first_ge(0.85)
    dur = float(T[nfr - 1])

    # --- 块级阶跃 → 块内线性插值(展示连续化; 块级口径不变) ---
    Df_s = np.rint(de_step(np.asarray(Df, float) / 1000.0) * 1000).astype(np.int32)
    RISK_s = np.rint(de_step(np.asarray(RISK, float) / 1000.0) * 1000).astype(np.int32)
    EAE_s = np.rint(de_step(np.asarray(EAE, float) / 1000.0) * 1000).astype(np.int32)
    EST_s = np.rint(de_step(np.asarray(EST, float) / 1000.0) * 1000).astype(np.int32)
    LV_s = np.zeros(nfr, dtype=np.int16)
    _c = 0
    for _k in range(nfr):
        while _c < 3 and Df_s[_k] >= int(round([0.25, 0.55, 0.85][_c] * 1000)):
            _c += 1
        LV_s[_k] = _c

    # --- 预警事件(level 首次升级点) ---
    warn = []
    last = 0
    for k in range(nfr):
        lv = int(LV_s[k])
        if lv > last:
            for g in range(last + 1, lv + 1):
                warn.append({'f': k, 't': round(float(T[k]), 1), 'lv': g})
            last = lv

    pkg = {
        'gid': gid,
        'n': int(n),
        'nfr': int(nfr),
        'step': STEP,
        'dt': round(dur / max(nfr - 1, 1), 4),
        'dur': round(dur, 1),
        'dtReal': round(dur / max(n - 1, 1), 4),
        'foCols': list(FO.keys()),
        'spec': {'kind': _m[0], 'load': _m[1], 'note': _m[2]},
        'meta': {
            'D_end': round(float(Dfull[-1]), 3),
            't25': t25, 't55': t55, 't85': t85,
            'b2': refs[0] if refs[0] is not None else None,
            'b3': refs[1] if refs[1] is not None else None,
            'aeEvents': int(AEN.sum()),
            'nFo': len(FO),
        },
        'warn': warn,
        # --- 波形数组 ---
        'D': Df_s.tolist(),
        'risk': RISK_s.tolist(),
        'eae': EAE_s.tolist(),
        'est': EST_s.tolist(),
        'lv': LV_s.tolist(),
        'st': ST.tolist(),
        'ael': AEL.tolist(),
        'aen': AEN.tolist(),
        't': [round(float(x), 1) for x in T.tolist()],
        'fo': {c: a.tolist() for c, a in FO.items()},
    }
    return pkg


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    groups = sys.argv[1:] or list(DASH_GROUPS)
    index = []
    for gid in groups:
        gid = gid.zfill(3)          # PowerShell 会把 021 当数字、吃掉前导零
        print(f'\n=== 导出 {gid} ===')
        pkg = _pack(gid)
        path = os.path.join(OUTDIR, f'{gid}.js')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('window.SHM_DATA=window.SHM_DATA||{};')
            f.write(f'window.SHM_DATA["{gid}"]=')
            f.write(json.dumps(pkg, ensure_ascii=False, separators=(',', ':')))
            f.write(';\n')
        size = os.path.getsize(path) / 1024.0
        print(f'  写出 {path}  ({size:.0f} KB)')
        print(f'  帧数={pkg["nfr"]} 时长={pkg["dur"]}s D_end={pkg["meta"]["D_end"]} '
              f't25={pkg["meta"]["t25"]} t55={pkg["meta"]["t55"]} t85={pkg["meta"]["t85"]} '
              f'AE事件={pkg["meta"]["aeEvents"]} 光纤通道={pkg["meta"]["nFo"]}')
        index.append({'gid': gid, 'nfr': pkg['nfr'], 'dur': pkg['dur'],
                      'n': pkg['n'], 'meta': pkg['meta'], 'foCols': pkg['foCols'],
                      'spec': pkg['spec'], 'warn': pkg['warn']})
    ipath = os.path.join(OUTDIR, 'index.js')
    with open(ipath, 'w', encoding='utf-8') as f:
        f.write('window.SHM_INDEX=')
        f.write(json.dumps(index, ensure_ascii=False, separators=(',', ':')))
        f.write(';\n')
    print(f'\n写出清单 {ipath}')


if __name__ == '__main__':
    main()
