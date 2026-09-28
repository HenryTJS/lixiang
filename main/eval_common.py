# -*- coding: utf-8 -*-
"""T7 参数系统化——公共库

设计（对齐 T3/T4 已验收口径，避免"手调"）:
  - 参数化 OnlineDamageIndex 的待校准参数, 默认 = v6 配置;
  - 每个参数配置在主样本 5 组上重算逐点 D(t), 按配置指纹隔离缓存(_t7_cache/<cfgid>/);
  - 每配置输出聚合指标: 达0.55/0.85 组数、D_end 中位、A-预警分级计数(A/E/D/C)。

口径说明(呼应"不拿 b1/b2 当裁判"):
  - D 的在线计算(证据/状态机)完全不使用 b1/b2/b3——纯因果流式;
  - b2/b3 仅用于【离线】评估/选参的参考锚(与 T3/T4 分级一致), 不进入方法本身;
  - b1 不可靠, 一律不使用。
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根(含 shm)
os.chdir(os.path.dirname(os.path.abspath(__file__)))                            # main/
from shm.streaming import StreamSimulator
from shm.damage_index import OnlineDamageIndex, SHAPE_DEFAULTS

GROUPS = ['016', '017', '018', '019', '020']   # 主样本 5 组
ROOT = os.path.dirname(os.path.abspath(__file__))
T7CACHE = os.path.join(ROOT, 'cache', '_t7_cache')
REF_CSV = os.path.join(ROOT, 'weak_labels', 'labels_summary.csv')

# ============ 待校准参数 + 默认值(v6) ============
# 每个参数名 -> 默认值。±30% 扰动 = *0.7 / *1.3
PARS = ['rise', 'fall', 'acc_scale', 'estrain_w', 'lift']
DEFAULT = {'rise': 0.12, 'fall': 0.008, 'acc_scale': 0.02,
           'estrain_w': 0.6, 'lift': 2.0}
PARS_DESC = {
    'rise': 'D 上升速率(追 risk)',
    'fall': 'D 回落速率(记忆/遗忘)',
    'acc_scale': '全能量加速度证据标度',
    'estrain_w': '应变发散证据权重',
    'lift': '损伤型事件判据(log 抬升)',
}
# A-预警判据(与 eval_warning 一致)
LOW, DROP, HOLD_FRAC = 0.30, 0.15, 0.02


# ============ 参考标签(b2/b3, 仅离线评估) ============
_REF = None


def ref_map():
    global _REF
    if _REF is None:
        sm = pd.read_csv(REF_CSV, encoding='utf-8-sig')
        sm['gid'] = sm['gid'].astype(int).map(lambda x: f'{x:03d}')
        _REF = {r['gid']: (float(r['b2']), float(r['b3'])) for _, r in sm.iterrows()}
    return _REF


def cfg_id(params):
    """参数 dict -> 可读指纹(缓存目录名)。

    注: 形状证据(e_shape)的参数不属 PARS, 但会改变 D → 一并并入指纹,
    否则不同 shape 配置会命中同一缓存目录而取到错误的 D。
    未显式指定时用 `SHAPE_DEFAULTS`(项目默认口径) → 默认指纹已随
    PeakFactor 升为默认而改变，旧缓存不会被误用。
    """
    p = {k: params.get(k, DEFAULT[k]) for k in PARS}
    tag = '_'.join(f'{k}{p[k]:g}' for k in PARS)
    sc = params.get('shape_col', SHAPE_DEFAULTS['shape_col'])
    if sc:
        tag += '_sh{0}q{1:g}w{2:g}'.format(
            sc, float(params.get('shape_q', SHAPE_DEFAULTS['shape_q'])),
            float(params.get('shape_w', SHAPE_DEFAULTS['shape_w'])))
    h = hashlib.md5(tag.encode()).hexdigest()[:6]
    return f'{tag}_{h}'


def run_one(args):
    gid, params = args
    cid = cfg_id(params)
    cdir = os.path.join(T7CACHE, cid)
    os.makedirs(cdir, exist_ok=True)
    dpath = os.path.join(cdir, f'{gid}.npy')
    if os.path.exists(dpath):
        return gid, np.load(dpath)
    di = OnlineDamageIndex(params)
    sim = StreamSimulator(gid)
    sim.load_data()
    dlist = []
    while sim.has_next():
        p = sim.next_point()
        strain = p['strain']
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        dlist.append(di.update(strain, float(pk) if pk is not None else None,
                               None, di.shape_value(p.get('ae'))))
    sim.cleanup()
    d = np.array(dlist, dtype=np.float32)
    np.save(dpath, d)
    return gid, d


def run_cfg(params, workers=8):
    """给定参数, 返回 {gid: d}。缓存命中则直接读。"""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(run_one, [(g, params) for g in GROUPS]))
    return dict(res)


# ============ 指标(逐组, 从逐点 D + b2/b3 参考) ============
def _first_ge(d, th):
    idx = np.where(d >= th)[0]
    return float(idx[0]) / len(d) * 100.0 if len(idx) else np.nan


def onset_of(d, hold):
    """A-预警不可逆 onset(life%), 或 None(同 eval_warning)。"""
    low, drop = LOW, LOW - DROP
    n = len(d)
    i = 0
    while i < n:
        if d[i] >= low:
            j_end = min(n - 1, i + hold)
            seg = d[i:j_end + 1]
            if seg.min() >= drop:
                return float(i) / n * 100.0
            below = np.where(seg < drop)[0]
            i = i + (below[0] if len(below) else (j_end - i + 1))
        else:
            i += 1
    return None


def per_group_metrics(gid, d):
    """逐组指标 dict(含 A 预警分级)。b2/b3 仅作离线评估参考。"""
    n = len(d)
    refs = ref_map()
    b2, b3 = refs.get(gid, (np.nan, 99.0))
    t25 = _first_ge(d, .25)
    t55 = _first_ge(d, .55)
    t85 = _first_ge(d, .85)
    tail = d[int(n * .80):]
    mono = float(np.mean(np.diff(tail) >= 0)) * 100.0 if len(tail) > 2 else np.nan
    hold = max(2000, int(n * HOLD_FRAC))
    tw = onset_of(d, hold)
    err = (tw - b2) if (tw is not None and not np.isnan(b2)) else None
    lead = (b3 - tw) if tw is not None else None
    if tw is None:
        grade = 'C'
    elif err is not None and err < -15:
        grade = 'E'
    elif err is not None and err > 15:
        grade = 'D'
    else:
        grade = 'A'
    return dict(gid=gid, n=n, D_end=float(d[-1]),
                t25=t25, t55=t55, t85=t85,
                tail_mono=mono, t_warn=tw, b2=b2, b3=b3,
                err=err, lead=lead, grade=grade)


def table_for(cfg_id_name, gid_d):
    """把 {gid:d} 变成逐组指标 DataFrame。"""
    rows = [per_group_metrics(g, d) for g, d in gid_d.items()]
    return pd.DataFrame(rows).sort_values('gid')


def summary_row(cfg_id_name, df):
    """配置级聚合(一行 summary)。"""
    n = len(df)
    n55 = int((df['D_end'] >= .55).sum())
    n85 = int((df['D_end'] >= .85).sum())
    cnt = df['grade'].value_counts()
    d = dict(cfg=cfg_id_name, n55=n55, n85=n85,
             D_end_med=float(df['D_end'].median()),
             t25_med=float(df['t25'].median()),
             nA=int(cnt.get('A', 0)), nE=int(cnt.get('E', 0)),
             nD=int(cnt.get('D', 0)), nC=int(cnt.get('C', 0)))
    ok = df[df['grade'] == 'A']
    d['A_lead_med'] = float(ok['lead'].median()) if len(ok) else np.nan
    ae = df['err'].dropna()
    d['mean_abs_err'] = float(np.abs(ae).mean()) if len(ae) else np.nan
    return d
