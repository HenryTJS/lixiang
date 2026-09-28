# -*- coding: utf-8 -*-
"""源融合对照：声发射单源(必做基线) vs 二源 / 三源 —— 逐组比较最终效果。

口径:
  - 主源 = 声发射('ae')，**每组必做单源基线**（设计要求）;
  - 辅助源 = 应变('strain')、光纤('fo')，按"数据有无"参与二源 / 三源融合;
  - 融合方式 = 证据层取 max（risk = max(启用源证据)），逐块结算，不引入未来信息;
  - 评估锚 b2/b3 仅作离线参考（与 eval_common 口径一致）。

输出:
  main/cache/_fusion_cache/<cfg>_<hash>/<gid>.npy   逐点 [D, risk, e_ae, e_strain, e_fo]
  main/results/fusion_compare.csv                   逐组 × 逐配置明细
  main/results/fusion_summary.csv                   配置级聚合
"""
import os
import sys
import hashlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(ROOT))          # 项目根(含 shm)
os.chdir(ROOT)

import eval_common as ec                            # noqa: E402
from shm.streaming import StreamSimulator           # noqa: E402
from shm.damage_index import OnlineDamageIndex      # noqa: E402
from shm.config import FO_PARAMS                    # noqa: E402

CACHE = os.path.join(ROOT, 'cache', '_fusion_cache')
RESDIR = os.path.join(ROOT, 'results')

# 声发射单源必做；其余按可用性叠加。融合算子: max / mean / min
CONFIGS = [
    ('ae',                 ('ae',),               'max'),
    ('ae+strain',          ('ae', 'strain'),      'max'),
    ('ae+fo',              ('ae', 'fo'),          'max'),
    ('ae+strain+fo',       ('ae', 'strain', 'fo'), 'max'),
    ('ae+strain|mean',     ('ae', 'strain'),      'mean'),
    ('ae+fo|mean',         ('ae', 'fo'),          'mean'),
    ('ae+strain+fo|mean',  ('ae', 'strain', 'fo'), 'mean'),
    ('ae+strain|min',      ('ae', 'strain'),      'min'),
    ('ae+strain+fo|min',   ('ae', 'strain', 'fo'), 'min'),
]


def cfg_tag(name):
    """配置指纹（含光纤证据参数），参数变了自动换缓存目录。"""
    h = hashlib.md5((name + repr(sorted(FO_PARAMS.items()))).encode()).hexdigest()[:6]
    return '%s_%s' % (name.replace('|', '_'), h)


def run_one(gid, name, sources, fusion, force=False):
    """跑一组 × 一配置 → (逐点数组, 光纤通道数, 是否有应变数据)。"""
    cdir = os.path.join(CACHE, cfg_tag(name))
    os.makedirs(cdir, exist_ok=True)
    dpath = os.path.join(cdir, gid + '.npy')
    mpath = os.path.join(cdir, gid + '_meta.npy')
    if os.path.exists(dpath) and os.path.exists(mpath) and not force:
        return np.load(dpath), np.load(mpath)

    params = dict(FO_PARAMS)
    params['sources'] = tuple(sources)
    params['fusion'] = fusion
    di = OnlineDamageIndex(params)
    sim = StreamSimulator(gid)
    sim.load_data()
    rows = []
    has_st = False
    while sim.has_next():
        p = sim.next_point()
        st = p['strain']
        if st is not None and not np.isnan(st):
            has_st = True
        pk = None
        if p.get('ae_new') and p.get('ae'):
            pk = p['ae'].get('ae_Peak', 0.0) or 0.0
        di.update(st, float(pk) if pk is not None else None, p.get('fo'),
                  di.shape_value(p.get('ae')))
        rows.append((di.damage, di.risk, di._last_e_ae,
                     di._last_e_strain, di._last_e_fo))
    nfo = len(sim.fo_cols)
    sim.cleanup()
    arr = np.array(rows, dtype=np.float32)
    meta = np.array([nfo, 1 if has_st else 0], dtype=np.int32)
    np.save(dpath, arr)
    np.save(mpath, meta)
    return arr, meta


def main():
    os.makedirs(RESDIR, exist_ok=True)
    rows = []
    for gid in ec.GROUPS:
        print('=== %s ===' % gid)
        for name, sources, fusion in CONFIGS:
            arr, meta = run_one(gid, name, sources, fusion)
            d = arr[:, 0]
            m = ec.per_group_metrics(gid, d)
            nfo, has_st = int(meta[0]), int(meta[1])
            row = dict(cfg=name, gid=gid, sources='+'.join(sources), fusion=fusion,
                       n_fo=nfo, has_strain=has_st,
                       D_end=m['D_end'], t25=m['t25'], t55=m['t55'], t85=m['t85'],
                       tail_mono=m['tail_mono'], t_warn=m['t_warn'],
                       b2=m['b2'], b3=m['b3'], err=m['err'], lead=m['lead'],
                       grade=m['grade'],
                       e_ae_max=float(arr[:, 2].max()),
                       e_st_max=float(arr[:, 3].max()),
                       e_fo_max=float(arr[:, 4].max()))
            rows.append(row)
            print('  %-13s D_end=%.3f t25=%-6s t55=%-6s t85=%-6s err=%-6s '
                  'lead=%-6s %s  (e_ae=%.2f e_st=%.2f e_fo=%.2f)'
                  % (name, m['D_end'],
                     '%.1f' % m['t25'] if np.isfinite(m['t25']) else '--',
                     '%.1f' % m['t55'] if np.isfinite(m['t55']) else '--',
                     '%.1f' % m['t85'] if np.isfinite(m['t85']) else '--',
                     '%.1f' % m['err'] if m['err'] is not None else '--',
                     '%.1f' % m['lead'] if m['lead'] is not None else '--',
                     m['grade'],
                     arr[:, 2].max(), arr[:, 3].max(), arr[:, 4].max()))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESDIR, 'fusion_compare.csv'),
              index=False, encoding='utf-8-sig')

    sums = []
    for name, _, _ in CONFIGS:
        sub = df[df['cfg'] == name].copy()
        s = ec.summary_row(name, sub)
        s['cfg'] = name
        sums.append(s)
    sdf = pd.DataFrame(sums)[
        ['cfg', 'n55', 'n85', 'nA', 'nE', 'nD', 'nC',
         'D_end_med', 't25_med', 'A_lead_med', 'mean_abs_err']]
    sdf.to_csv(os.path.join(RESDIR, 'fusion_summary.csv'),
               index=False, encoding='utf-8-sig')

    print()
    print('=== SUMMARY ===')
    print(sdf.to_string(index=False))
    print()
    print('wrote:', os.path.join(RESDIR, 'fusion_compare.csv'))
    print('wrote:', os.path.join(RESDIR, 'fusion_summary.csv'))


if __name__ == '__main__':
    main()
