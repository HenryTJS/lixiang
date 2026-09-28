# -*- coding: utf-8 -*-
"""Agent 工具集：把项目已有的分析产物封装成可调用接口。

设计原则
--------
1. **只读** —— 工具只读 `results/` 与 `cache/` 中已有的产物，不触发重算。
2. **可追溯** —— 每个返回值都带 `evidence`（数据来源文件）与 `notes`（已知限制），
   供上层做物理一致性校验与自然语言解释。
3. **不猜** —— 数据缺失时返回 `ok=False` 并说明原因，绝不编造。

统一返回结构::

    {
        'ok': bool,               # 是否成功
        'tool': str,              # 工具名
        'gid': str | None,        # 涉及的试件
        'data': dict,             # 结构化结果（供语言层/校验层使用）
        'evidence': [str],        # 证据链：数据来源文件
        'notes': [str],           # 已知限制/警示
        'error': str | None,      # ok=False 时的原因
    }
"""
import os
import re

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_RES = os.path.join(ROOT, 'main', 'results')
L1_RES = os.path.join(ROOT, 'l1', 'results')
MAIN_CACHE = os.path.join(ROOT, 'main', 'cache')

# 三类数据集（A/B/C 编号沿用旧文档口径）
DS_MAIN = 'A'          # 主样本 016-020
DS_L1A = 'B'           # L1 第一批 L1-03/04/05/09
DS_L1B = 'C'           # L1 第二批 9 组

L1A_GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09']
L1B_GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54',
              'L1-55', 'L1-56', 'L1-59', 'L1-60']
MAIN_GROUPS = ['016', '017', '018', '019', '020']

# 主样本数据限制（硬编码，供 notes 引用；依据见 docs/details.md）
LIMIT_MAIN_AE = '主样本 AE 无时间戳、单通道 → 事件速率与定位不可用'
LIMIT_MAIN_DEMOD = '主样本 26 个应变/光纤通道中仅 016 应变已解调，其余为块内 std 口径'


def _res(ok, tool, data=None, gid=None, evidence=None, notes=None, error=None):
    return dict(ok=bool(ok), tool=tool, gid=gid, data=data or {},
                evidence=evidence or [], notes=notes or [], error=error)


def _read_csv(path):
    if not os.path.exists(path):
        return None
    for enc in ('utf-8-sig', 'utf-8', 'gbk', 'latin-1'):
        try:
            df = pd.read_csv(path, encoding=enc)
            # ⚠️ CSV 中 '016' 会被 pandas 读成整数 16 → 统一零填充回三位字符串，
            # 否则与用户的 '016' 字符串比较必然失败。
            if 'gid' in df.columns:
                df['gid'] = df['gid'].astype(str).str.strip().apply(_norm_gid)
            return df
        except Exception:
            continue
    return None


def _norm_gid(g):
    """归一化试件编号：016 / 16 → '016'；L1-49 / l1-49 / L1 49 → 'L1-49'。"""
    s = str(g).strip()
    m = re.match(r'l1[\s\-_]*?(\d{1,3})', s, re.I)      # 必须跳过 'L1' 本身，否则会取到 1
    if m:
        return 'L1-%02d' % int(m.group(1))
    m = re.fullmatch(r'(\d{1,3})', s)
    return '%03d' % int(m.group(1)) if m else s


def norm_gid(g):
    """对外暴露的 gid 归一化。"""
    return _norm_gid(g)


def dataset_of(gid):
    """由试件编号判断所属数据集。"""
    gid = norm_gid(gid)
    if gid in MAIN_GROUPS:
        return DS_MAIN
    if gid in L1A_GROUPS:
        return DS_L1A
    if gid in L1B_GROUPS:
        return DS_L1B
    return None


# ---------------------------------------------------------------- 工具 1
def list_specimens():
    """列出全部试件、所属数据集、数据可用性与已知问题。"""
    ev, rows = [], []
    ck = _read_csv(os.path.join(MAIN_RES, 'candidate_check.csv'))
    if ck is not None:
        ev.append('main/results/candidate_check.csv')
    for g in MAIN_GROUPS:
        row = {'gid': g, 'dataset': DS_MAIN, 'usable': True, 'note': ''}
        if ck is not None and g in set(ck['gid'].astype(str)):
            r = ck[ck['gid'].astype(str) == g].iloc[0]
            row['sampling_hz'] = round(1.0 / float(r['dt_med']), 1) if r.get('dt_med') else None
        rows.append(row)
    ta = _read_csv(os.path.join(L1_RES, 'l1_time_align.csv'))
    if ta is not None:
        ev.append('l1/results/l1_time_align.csv')
    for g in L1A_GROUPS:
        note = ''
        if g == 'L1-04':
            note = 'AE 仅覆盖 61.3% 寿命'
        rows.append({'gid': g, 'dataset': DS_L1A, 'usable': True, 'note': note})
    for g in L1B_GROUPS:
        note = []
        if ta is not None and g in set(ta['gid']):
            r = ta[ta['gid'] == g].iloc[0]
            if float(r['dfos_cover']) < 0.80:
                note.append('DFOS 覆盖 %.0f%%' % (100 * float(r['dfos_cover'])))
            if float(r['gap_cv']) > 5:
                note.append('段间隔极不规整(CV=%.1f)' % float(r['gap_cv']))
        if g in ('L1-59', 'L1-60'):
            note.append('AE 覆盖不足')
        if g == 'L1-56':
            note.append('基线 AE 活动异常低（疑似采集问题）')
        rows.append({'gid': g, 'dataset': DS_L1B, 'usable': True,
                     'note': '；'.join(note)})
    return _res(True, 'list_specimens', {'specimens': rows}, evidence=ev,
                notes=['数据集 A=主样本(5组) / B=L1第一批(4组,有FBG) / C=L1第二批(9组,无FBG)'])


# ---------------------------------------------------------------- 工具 2
def warning_status(gid):
    """查询某主样本试件的 D(t) 三级预警状态与提前量。"""
    gid = norm_gid(gid)
    w = _read_csv(os.path.join(MAIN_RES, 'warning_onset.csv'))
    if w is None:
        return _res(False, 'warning_status', gid=gid, error='warning_onset.csv 不存在')
    if gid not in set(w['gid'].astype(str)):
        return _res(False, 'warning_status', gid=gid,
                    error='仅数据集 A（%s）有 D(t) 预警结果' % '/'.join(MAIN_GROUPS))
    r = w[w['gid'].astype(str) == gid].iloc[0]
    if r['grade'] == 'no warning':
        return _res(False, 'warning_status', gid=gid, error='该试件未触发预警')
    gl = _read_csv(os.path.join(MAIN_RES, 'grade_levels_final.csv'))
    levels = None
    if gl is not None and gid in set(gl['gid'].astype(str)):
        g2 = gl[gl['gid'].astype(str) == gid].iloc[0]
        levels = {'L1': round(float(g2['t_L1']), 1), 'L2': round(float(g2['t_L2']), 1),
                  'L3': round(float(g2['t_L3']), 1), 'gate': str(g2['gate'])}
    d = {'t_warn_pct': round(float(r['t_warn']), 1),
         'ref_b2_pct': float(r['b2']), 'failure_anchor_b3_pct': float(r['b3']),
         'err_pct': round(float(r['err']), 1),
         'lead_pct': round(float(r['lead']), 1),
         'grade': str(r['grade']), 'levels': levels}
    ev = ['main/results/warning_onset.csv']
    if levels:
        ev.append('main/results/grade_levels_final.csv')
    notes = ['预警判据：D 首次不可逆 ≥0.3（2% 寿命窗内不回落到 0.15）',
             'b2 为扩展标签（非独立物理真值）']
    if levels and levels['gate'] == 'off':
        notes.append('该组分级闸门关闭 → L1/L2/L3 间隔不具物理意义')
    return _res(True, 'warning_status', d, gid=gid, evidence=ev, notes=notes)


# ---------------------------------------------------------------- 工具 3
def stiffness(gid):
    """查询刚度损失率轨迹与阈值时刻（独立于自定义标签的物理参照）。"""
    gid = norm_gid(gid)
    s = _read_csv(os.path.join(MAIN_RES, 'grade_stiff_traj.csv'))
    if s is None or gid not in set(s['gid'].astype(str)):
        return _res(False, 'stiffness', gid=gid,
                    error='刚度轨迹仅覆盖数据集 A（%s）' % '/'.join(MAIN_GROUPS))
    r = s[s['gid'].astype(str) == gid].iloc[0]
    ths = {}
    for k in ('x_0.05', 'x_0.10', 'x_0.30', 'x_0.50'):
        v = r.get(k)
        ths[k.replace('x_', 'L>=')] = None if pd.isna(v) else round(float(v), 1)
    d = {'stiff_end': round(float(r['stiff_end']), 3),
         'stiff_max': round(float(r['stiff_max']), 3),
         'n_blocks': int(r['nblk']), 'thresholds': ths}
    notes = ['口径：循环幅值 A 相对校准段([20%,45%)块)的 p30 → L=A/A_cal-1',
             '阈值时刻为**持续性判据**（连续 n 块 ≥ 阈值且排除暖机段）']
    if float(r['stiff_max']) < 0.05:
        notes.append('该组无显著刚度退化信号 → 不可作为独立参照')
    if str(gid) in ('018', '019', '020'):
        notes.append('该组 AE 为等间隔分窗口径（10 条/s = 采样率本身）；'
                     '应变通道未解调（RAW），幅值须用块内 std 口径')
    return _res(True, 'stiffness', d, gid=gid,
                evidence=['main/results/grade_stiff_traj.csv'], notes=notes)


# ---------------------------------------------------------------- 工具 4
def data_quality(gid):
    """查询某试件的数据采集质量（采样率、连续性、AE 口径、传感器完整性）。"""
    gid = norm_gid(gid)
    ck = _read_csv(os.path.join(MAIN_RES, 'candidate_check.csv'))
    if ck is None or gid not in set(ck['gid'].astype(str)):
        return _res(False, 'data_quality', gid=gid, error='仅数据集 A 有体检记录')
    r = ck[ck['gid'].astype(str) == gid].iloc[0]
    dtm, dtmax = float(r['dt_med']), float(r['dt_max'])
    d = {'duration_h': round(float(r['t_max_h']), 2) if pd.notna(r['t_max_h']) else None,
         'dt_median_s': dtm, 'dt_max_s': dtmax,
         'sampling_hz': round(1.0 / dtm, 1) if dtm > 0 else None,
         'n_ae': int(r['n_ae']) if pd.notna(r.get('n_ae')) else None,
         'ae_effective_hz': round(float(r['ae_hz']), 2) if pd.notna(r.get('ae_hz')) else None,
         'demod_mode': str(r.get('mode')) if pd.notna(r.get('mode')) else None}
    notes = []
    if dtmax > 1.5:
        notes.append('采集不连续（最大间隔 %.1f s）→ 时间轴不可信' % dtmax)
    if d['ae_effective_hz'] and d['sampling_hz'] and \
            abs(d['ae_effective_hz'] - d['sampling_hz']) / d['sampling_hz'] > 0.3:
        notes.append('AE 等效频率(%.2f Hz) 与应变(%.1f Hz) 不匹配 → 事件时间由均匀映射猜测'
                     % (d['ae_effective_hz'], d['sampling_hz']))
    if d['demod_mode'] == 'RAW':
        notes.append('应变通道未解调 → 幅值须用块内 std 口径')
    return _res(True, 'data_quality', d, gid=gid,
                evidence=['main/results/candidate_check.csv'], notes=notes)


# ---------------------------------------------------------------- 工具 5
def compare(gids):
    """多试件横向对比（预警时刻 / 提前量 / 刚度）。"""
    if isinstance(gids, str):
        gids = [s.strip() for s in gids.replace('、', ',').split(',') if s.strip()]
    gids = [norm_gid(g) for g in gids]
    rows = []
    for g in gids:
        item = {'gid': g}
        w = warning_status(g)
        if w['ok']:
            item['t_warn_pct'] = w['data']['t_warn_pct']
            item['lead_pct'] = w['data']['lead_pct']
        s = stiffness(g)
        if s['ok']:
            item['stiff_end'] = s['data']['stiff_end']
            item['L10_pct'] = s['data']['thresholds'].get('L>=0.10')
        rows.append(item)
    ok = any('t_warn_pct' in r for r in rows)
    return _res(ok, 'compare', {'rows': rows}, evidence=['main/results/warning_onset.csv',
                                                        'main/results/grade_stiff_traj.csv'],
                notes=['b2 为扩展标签；刚度阈值为独立物理参照'])


# ---------------------------------------------------------------- 工具 6
def anomaly(gid):
    """查询数据集 C 的无监督异常检测结果。"""
    gid = norm_gid(gid)
    fp = os.path.join(L1_RES, '_l1_anom_%s.npz' % gid)
    if not os.path.exists(fp):
        return _res(False, 'anomaly', gid=gid, error='仅数据集 C（9 组）有无监督检测结果')
    z = np.load(fp)
    g, m, r = z['life'], z['maha'], z['recon']
    i_m, i_r = int(z['first_maha']), int(z['first_recon'])
    d = {'first_exceed_maha_pct': round(100 * float(g[i_m]), 1) if i_m >= 0 else None,
         'first_exceed_pca_pct': round(100 * float(g[i_r]), 1) if i_r >= 0 else None,
         'maha_threshold': round(float(z['thr_maha']), 2),
         'late_ratio': round(float(np.median(m[-30:]) / z['thr_maha']), 2),
         'baseline_frac': float(z['base_frac'])}
    return _res(True, 'anomaly', d, gid=gid,
                evidence=['l1/results/_l1_anom_%s.npz' % gid],
                notes=['基线取组内前 %.0f%% 寿命 → 无需标签、规避跨组标定' %
                       (100 * float(z['base_frac'])),
                       'late_ratio 越大表示末段偏离基线越显著'])


# ---------------------------------------------------------------- 工具 7
def mechanism(gid):
    """查询数据集 C 的 RA–AF 损伤机制识别结果。"""
    gid = norm_gid(gid)
    fp = os.path.join(L1_RES, '_l1_raf_%s.npz' % gid)
    if not os.path.exists(fp):
        return _res(False, 'mechanism', gid=gid, error='仅数据集 C（9 组）有 RA–AF 结果')
    z = np.load(fp)
    sf, life = z['shear_frac'], z['life']
    n = sf.size
    k = max(3, n // 5)
    seg = [round(float(np.nanmean(sf[i * k:(i + 1) * k])), 3) for i in range(5)]
    d = {'shear_frac_by_quintile': seg,
         'first20': seg[0], 'last20': seg[-1],
         'delta': round(seg[-1] - seg[0], 3),
         'bi_shear': round(float(z['bi_shear']), 3) if np.isfinite(z['bi_shear']) else None,
         'ra_median': [round(float(np.nanmedian(z['ra_med'][i * k:(i + 1) * k])), 2)
                       for i in range(5)]}
    notes = ['判据：score=af_n-ra_n<0 记为剪切型；基线=首个数据块（因果标准化）',
             'RA–AF 分界线为经验值 → 只看趋势，不看绝对值']
    if str(gid) in ('L1-51', 'L1-56'):
        notes.append('该组时间标定不可靠（§3.3）→ 趋势结论不纳入统计')
    return _res(True, 'mechanism', d, gid=gid, evidence=['l1/results/_l1_raf_%s.npz' % gid],
                notes=notes)


# ---------------------------------------------------------------- 工具 8
def localization(gid):
    """查询数据集 C 的 AE 事件定位精度（X 可信 / Y 不可信）。"""
    gid = norm_gid(gid)
    fp = os.path.join(L1_RES, '_l1_loc_%s.npz' % gid)
    if not os.path.exists(fp):
        return _res(False, 'localization', gid=gid, error='仅数据集 C（9 组）有定位结果')
    z = np.load(fp)
    x, y, r, nc = z['x'], z['y'], z['rms_us'], z['nch']
    good = r <= 5.0
    if good.sum() < 20:
        return _res(False, 'localization', gid=gid, error='可信定位点不足')
    d = {'n_total': int(x.size), 'n_good': int(good.sum()),
         'rms_median_us': round(float(np.median(r)), 2),
         'centroid_x_mm': round(float(x[good].mean()), 1),
         'centroid_y_mm': round(float(y[good].mean()), 1),
         'vx_mps': float(z['vx']), 'vy_mps': float(z['vy'])}
    return _res(True, 'localization', d, gid=gid,
                evidence=['l1/results/_l1_loc_%s.npz' % gid],
                notes=['几何来自 PDF，13 组一致：S1(145,190) S2(145,20) S3(20,50) S4(20,220)',
                       '走时用各向异性椭圆模型（纵向 vy / 横向 vx，差 60%）',
                       '⚠️ 跨组质心散布：X σ≈%.0f mm（小）、Y σ≈%.0f mm（大）—— '
                       '各组真值不同，故这**不是**准确性指标（§3.3b / §14.4）'
                       % (LOC_SIGMA_X_MM, LOC_SIGMA_Y_MM),
                       '⚠️ 「距冲击点」不可用：13 组 PDF 的冲击位置描述互不一致，'
                       '真值未证实（仅 L1-49 自洽）；勿引用距冲击点偏差',
                       'Y 向差的物理解释：加筋条是波导，波沿 Y 传播非直线 → 椭圆模型在 Y 失效'])


# ---------------------------------------------------------------- 工具 9
def l1_migration(gid):
    """查询数据集 B（L1 第一批）的 D(t) 迁移结果。"""
    gid = norm_gid(gid)
    ds = dataset_of(gid)
    if ds == DS_L1A:
        fp = os.path.join(L1_RES, 'l1_degree.csv')
        d = _read_csv(fp)
        if d is None or gid not in set(d['gid'].astype(str)):
            return _res(False, 'l1_migration', gid=gid,
                        error='数据集 B 的 l1/results/l1_degree.csv 不存在或不含该组',
                        notes=['复现：python l1/evaluate_l1_degree.py --baseline '
                               '--strain-evidence --fusion max --params rise=0.05',
                               '该文件产出的是数据集 B（4 组）的 D(t)，'
                               '勿与数据集 C 的 l1_degree_v2.csv 混用'])
        r = d[d['gid'].astype(str) == gid].iloc[0]
        return _res(True, 'l1_migration', {
            k: (round(float(r[k]), 4) if pd.notna(r[k]) else None) for k in d.columns
            if k != 'gid'}, gid=gid, evidence=['l1/results/l1_degree.csv'],
            notes=['数据集 B 有 FBG 块锚 → cycle 坐标可靠；采用基线重定义 + 应变漂移证据'])
    if ds == DS_L1B:
        fp = os.path.join(L1_RES, 'l1_degree_v2.csv')
        d = _read_csv(fp)
        if d is None or gid not in set(d['gid'].astype(str)):
            return _res(False, 'l1_migration', gid=gid,
                        error='数据集 C 的 l1/results/l1_degree_v2.csv 不存在或不含该组')
        r = d[d['gid'].astype(str) == gid].iloc[0]
        return _res(True, 'l1_migration', {
            k: (round(float(r[k]), 4) if pd.notna(r[k]) else None) for k in d.columns
            if k != 'gid'}, gid=gid, evidence=['l1/results/l1_degree_v2.csv'],
            notes=['数据集 C 无 FBG → 时间锚由 DFOS 段近似，位置结论可靠性低于数据集 B',
                   'D_end ≈ 1 是末端归一化所致 → 用 t25/t55/t85 的寿命占比而非 D_end 定级'])
    return _res(False, 'l1_migration', gid=gid,
                error='仅数据集 B（%s）与 C（9 组）有 D(t) 迁移结果' % '/'.join(L1A_GROUPS))


# ---------------------------------------------------------------- 工具 10
def doc_search(query, k=3):
    """在 README 与 docs 中检索相关说明段落（关键词 + 词频密度排序）。"""
    files = [os.path.join(ROOT, 'README.md'),
             os.path.join(ROOT, 'docs', 'details.md')]
    q = str(query)
    keys = [w for w in re.split(r'[\s，。？、；：,?;:]+', q) if len(w) >= 2]
    if not keys:
        keys = [q]
    hits = []
    for fp in files:
        if not os.path.exists(fp):
            continue
        try:
            txt = open(fp, encoding='utf-8').read()
        except Exception:
            continue
        for para in txt.split('\n\n'):
            p = para.strip()
            if len(p) < 30:                      # 跳过标题/短行，避免命中目录
                continue
            cnt = sum(p.count(kw) for kw in keys)
            if cnt == 0:
                continue
            # 词频密度：命中次数 / 段落长度（惩罚长段落）
            hits.append((cnt / (len(p) ** 0.5), cnt, os.path.relpath(fp, ROOT), p))
    hits.sort(key=lambda t: (-t[0], -t[1]))
    return _res(bool(hits), 'doc_search',
                {'hits': [{'source': s, 'text': t[:400]} for _, _, s, t in hits[:k]]},
                evidence=['README.md', 'docs/details.md'],
                notes=['基于关键词与词频密度的简易检索，非语义检索'])


# ---------------------------------------------------------------- 工具 11
# ── 检修建议的推理规则（确定性、可审计；不由 LLM 生成） ────────────────
URGENCY = {
    'A': ('立即处置', '停止继续加载；安排更换／大修评估'),
    'B': ('计划检修', '纳入下一次计划检修窗口；检修前限制载荷等级'),
    'C': ('加强监测', '加密监测频次、缩短巡检周期'),
    'D': ('常规监测', '维持现有巡检周期'),
}
_URG_RANK = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
IM_MIN_LEAD_PCT = 20.0        # 剩余寿命裕度 < 20% → 紧迫度升一级
LOC_SIGMA_X_MM = 14.0         # AE 定位 X 向**跨组质心散布** σ（§3.3b / §14.4，2026-09-18 复算）
LOC_SIGMA_Y_MM = 46.0         # 同上，Y 向（加筋条波导 → 不可用于分区）
UNRELIABLE_TIME = ['L1-51', 'L1-56']
POOR_COVERAGE = {'L1-59': 'AE 覆盖 62.7%', 'L1-60': 'AE 70.8% / DFOS 67.5%',
                 'L1-55': 'DFOS 覆盖 72.2%'}
# 等间隔分窗 AE 组（AE 行数 ÷ 真实时长 = 采样率本身 10 条/s，与应变 1:1）
# ⚠️ 2026-09-20 修正: 原名为 LOW_RATE 并断言「1 Hz 采样」—— **该断言是错的**，
#    全组采样率恒为 10 Hz（018/019/020 第一列是整数计数器，不是秒）。
WINDOW_AE = ['018', '019', '020']
NO_REPAIR_TECH = ('本工具只给处置级别、检修范围与复检手段；不给出维修工艺或部件选型'
                  '（超出所用证据的能力范围）')


def _num(x):
    """把可能为 NaN/None 的值转成 float 或 None。"""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(f) else f


def maintenance(gid):
    """由已有证据**确定性推导**检修建议（推理规则显式、每步带依据）。

    为什么放在工具层而不交给 LLM：「什么损伤程度对应什么处置」属于**工程规范知识**，
    必须可控、可复现、可追溯；LLM 只负责把结论讲成人话。

    与其它工具的区别：它做**推理**而非仅取数，但每条 actions 都返回 `basis`
    （推理所依据的实测事实），因此结论可回溯到具体产物文件。
    """
    gid = norm_gid(gid)
    ds = dataset_of(gid)
    if ds is None:
        return _res(False, 'maintenance', gid=gid, error='未知试件编号：%s' % gid)
    if ds == DS_L1A:
        return _res(
            False, 'maintenance', gid=gid,
            error='数据集 B（L1 一批）未接入维修定级：本工具的定级口径只在主样本 016-020 上标定',
            notes=['D(t) 数值请改用 l1_migration 工具（产物 l1/results/l1_degree.csv）',
                   '如需为数据集 B 出处置建议，须先标定其「D 级别 × 刚度闸门」口径',
                   NO_REPAIR_TECH])

    facts, ev, notes = {}, [], []
    basis_sev, hard, soft, recheck, acts = [], [], [], [], []
    sev, zone, lead = 'D', None, None

    # ───────────────────────── 数据集 A（主样本 016-020）
    if ds == DS_MAIN:
        w, s, q = warning_status(gid), stiffness(gid), data_quality(gid)
        ev = ['main/results/warning_onset.csv', 'main/results/grade_levels_final.csv',
              'main/results/grade_stiff_traj.csv', 'main/results/candidate_check.csv']
        if not w['ok']:
            return _res(False, 'maintenance', gid=gid,
                        error='无 D(t) 预警结果：%s' % w['error'])
        facts['t_warn_pct'] = w['data']['t_warn_pct']
        facts['lead_pct'] = w['data']['lead_pct']
        facts['failure_anchor_b3_pct'] = w['data']['failure_anchor_b3_pct']
        notes += w['notes']
        lead = _num(facts['lead_pct'])

        # ① D 分级（仅当闸门有效时可用）
        lv = w['data'].get('levels')
        gate_on = bool(lv) and str(lv.get('gate', 'off')).strip().lower() != 'off'
        if gate_on:
            facts['levels'] = lv
            for lvl, key in (('L3', 'A'), ('L2', 'B'), ('L1', 'C')):
                t = _num(lv.get(lvl))
                if t is not None:
                    sev = key
                    basis_sev.append('D 达 %s（%.1f%% 寿命）→ 严重度 %s' % (lvl, t, key))
                    break
        else:
            sev = 'C'
            basis_sev.append('D 首次不可逆 ≥0.3（t_warn=%.1f%% 寿命）→ 至少须加强监测'
                             % facts['t_warn_pct'])
            notes.append('该组分级闸门 gate=off → L1/L2/L3 时刻等间距、不具物理意义；'
                         '本建议仅以 t_warn 定级，更高级别由刚度证据独立给出')

        # ② 刚度（独立物理参照，可提升严重度）
        if s['ok']:
            facts['stiffness'] = s['data']
            notes += s['notes']
            th = s['data']['thresholds']
            for k, key in (('L>=0.50', 'A'), ('L>=0.30', 'B'), ('L>=0.10', 'C')):
                t = _num(th.get(k))
                if t is None:
                    continue
                if _URG_RANK[key] > _URG_RANK[sev]:
                    sev = key
                basis_sev.append('刚度 %s 于 %.1f%% 寿命达成 → 严重度 %s'
                                 % (k.replace('L>=', 'L≥'), t, key))
                break
            if s['data']['stiff_max'] < 0.05:
                hard.append('刚度无显著退化信号（stiff_max=%.3f）→ D 缺独立物理确认'
                            % s['data']['stiff_max'])
        else:
            hard.append('无刚度轨迹 → D 缺独立确认')

        # ③ 采集质量
        if q['ok']:
            facts['data_quality'] = q['data']
            notes += q['notes']
            if (_num(q['data'].get('dt_max_s')) or 0) > 1.5:
                hard.append('采集不连续（dt_max=%.1f s）→ 时间轴不可信'
                            % q['data']['dt_max_s'])
            if q['data'].get('demod_mode') == 'RAW':
                soft.append('应变通道未解调 → 幅值仅块内 std 口径，跨试件绝对值不可比')
        if gid in WINDOW_AE:
            soft.append('该组 AE 为等间隔分窗口径（10 条/s = 采样率本身，与应变 1:1）'
                        '→ 不得把记录密度当作事件速率')
        if facts.get('stiffness', {}).get('stiff_max', 0) == 0:
            recheck.append('该组应变幅值单调下降（应力控制下与刚度退化方向矛盾）'
                           '→ 应变通道疑似失效，不建议据此判断刚度')

        recheck.append('AE 无时间戳、单通道 → 不支持事件速率与定位类复检结论')

    # ───────────────────────── 数据集 C（L1 第二批 9 组）
    else:
        d = _read_csv(os.path.join(L1_RES, 'l1_degree_v2.csv'))
        ev.append('l1/results/l1_degree_v2.csv')
        if d is None or gid not in set(d['gid'].astype(str)):
            return _res(False, 'maintenance', gid=gid,
                        error='数据集 C 的 D(t) 产物缺失或不含该组')
        r = d[d['gid'].astype(str) == gid].iloc[0]
        facts['n_f'] = int(r['n_f'])
        facts['D_end'] = _num(r.get('D_end'))
        for k, lab in (('t25_pct', 'L1'), ('t55_pct', 'L2'), ('t85_pct', 'L3')):
            t = _num(r.get(k))
            if t is not None:
                facts[lab + '_pct'] = t
        # 严重度：t85/t55/t25 须与 D_end 一致，否则以异常检测定级（防早段误触发）
        dend = facts.get('D_end')
        for lab, thr, key in (('L3', 0.85, 'A'), ('L2', 0.55, 'B'), ('L1', 0.25, 'C')):
            if lab + '_pct' in facts and dend is not None and dend >= thr:
                sev = key
                lead = round(100.0 - facts[lab + '_pct'], 1)
                basis_sev.append('D 达 %s 阈值（%.1f%% 寿命）且 D_end=%.3f → 严重度 %s'
                                 % (lab, facts[lab + '_pct'], dend, key))
                break
        if not basis_sev:
            notes.append('D(t) 证据内部不一致（阈值时刻与 D_end 不匹配）→ '
                         '改用无监督异常检测定级')
        facts['lead_pct'] = lead
        an = anomaly(gid)
        if an['ok']:
            facts['anomaly'] = an['data']
            notes += an['notes']
            fe = _num(an['data'].get('first_exceed_maha_pct'))
            if fe is not None and _URG_RANK[sev] < _URG_RANK['C']:
                sev = 'C'
                lead = round(100.0 - fe, 1)
                basis_sev.append('无监督检测首次偏离基线（%.1f%% 寿命）→ 严重度 C' % fe)
        # 机制 → 检修重点
        m = mechanism(gid)
        if m['ok']:
            facts['mechanism'] = m['data']
            notes += m['notes']
            dl = _num(m['data'].get('delta')) or 0.0
            if dl >= 0.10:
                acts.append({'action': '检修重点置于连接区／界面（分层、脱粘主导）',
                             'priority': 2,
                             'basis': ['剪切型占比上升 %.3f（%.3f→%.3f）'
                                       % (dl, m['data']['first20'], m['data']['last20'])]})
            elif dl <= -0.10:
                acts.append({'action': '检修重点置于母材区（基体开裂／纤维断裂主导）',
                             'priority': 2, 'basis': ['剪切型占比下降 %.3f' % dl]})
            else:
                notes.append('损伤机制未显著变化（Δ剪切占比 %.3f）→ 不据此调整检修重点' % dl)
        else:
            notes.append('无 RA–AF 结果 → 损伤机制未知，检修范围只能按位置而非机理确定')
        # 定位 → 检修范围（仅 X 可信）
        lo = localization(gid)
        if lo['ok']:
            facts['localization'] = lo['data']
            notes += lo['notes']
            cx = _num(lo['data'].get('centroid_x_mm'))
            if cx is not None:
                zone = {'x_range_mm': [round(cx - LOC_SIGMA_X_MM, 1),
                                       round(cx + LOC_SIGMA_X_MM, 1)],
                        'centroid_x_mm': cx,
                        'basis': ['AE 事件定位可信点 %d 个、中位残差 %.2f µs'
                                  % (lo['data']['n_good'], lo['data']['rms_median_us']),
                                  'X 向 σ≈%.0f mm 可用于判断「哪一侧」；'
                                  'Y 向 σ≈%.0f mm 不可用于分区'
                                  % (LOC_SIGMA_X_MM, LOC_SIGMA_Y_MM)]}
        else:
            notes.append('无可用定位结果 → 无法给出检修范围，只能整体处置')

        if gid in UNRELIABLE_TIME:
            hard.append('%s 时间标定不可靠 → 其寿命位置结论不参与定级' % gid)
        if gid in POOR_COVERAGE:
            soft.append('%s（%s）→ 尾段结论可信度受限' % (gid, POOR_COVERAGE[gid]))
        hard.append('数据集 C 无 FBG、无刚度轨迹 → 缺独立物理参照')

    # ───────────────────────── 紧迫度：剩余裕度不足则升一级
    if lead is not None and lead < IM_MIN_LEAD_PCT and _URG_RANK[sev] < _URG_RANK['A']:
        old = sev
        sev = {4: 'A', 3: 'A', 2: 'B', 1: 'C'}[_URG_RANK[sev] + 1]
        basis_sev.append('剩余寿命裕度仅 %.1f%%（< %.0f%%）→ 紧迫度由 %s 升为 %s'
                         % (lead, IM_MIN_LEAD_PCT, old, sev))

    # ───────────────────────── 处置动作（按级别）
    if sev == 'A':
        acts = [{'action': '停用评估：在复检确认前不再施加同级或更高级载荷', 'priority': 1},
                {'action': '启动更换／大修评估流程', 'priority': 1},
                {'action': '对关注区域实施加密巡检（目视＋局部应变复测）', 'priority': 2}] + acts
    elif sev == 'B':
        acts = [{'action': '纳入最近一次计划检修窗口', 'priority': 1},
                {'action': '检修前限制载荷等级（不高于当前等级）', 'priority': 1},
                {'action': '巡检频次提高至当前的 2 倍以上', 'priority': 2}] + acts
    elif sev == 'C':
        acts = [{'action': '加密监测：缩短复检周期，跟踪 D 是否持续上升', 'priority': 1},
                {'action': '维持现有载荷等级，旁记每级载荷下的 D 增量', 'priority': 2}] + acts
    else:
        acts = [{'action': '维持现有巡检周期', 'priority': 1}] + acts
    for a in acts:
        a.setdefault('basis', list(basis_sev))

    conf = 'high' if not hard else ('low' if len(hard) >= 2 else 'medium')
    not_rec = [NO_REPAIR_TECH]
    if zone is None:
        not_rec.append('不支持按位置分区检修（无可用定位能力）')
    if conf != 'high':
        not_rec.append('不建议仅凭本结论直接实施维修；'
                       + ('先完成下列复检项' if recheck else '需结合其它手段交叉确认'))
    not_rec += recheck

    return _res(
        True, 'maintenance', {
            'urgency': {'level': sev, 'label': URGENCY[sev][0],
                        'hint': URGENCY[sev][1], 'basis': basis_sev},
            'confidence': conf, 'confidence_basis': hard + soft,
            'facts': facts,
            'actions': acts,
            'inspection_zone': zone,
            'recheck': recheck,
            'monitoring': {
                'escalation_rule': '若 D 再上升 ≥0.10，或刚度损失 L 再上升 10 个百分点 '
                                   '→ 处置级别升一级',
                'basis': ['三级语义阈值 D=0.25/0.55/0.85', '刚度闸门 θ2=0.10 θ3=0.50']},
            'not_recommended': not_rec,
        }, gid=gid, evidence=ev,
        notes=notes + ['紧迫度分级规则：严重度取 D 级别与刚度阈值的较高者；'
                       '剩余裕度 <%.0f%% 时升一级' % IM_MIN_LEAD_PCT,
                       '每条建议的 basis 列出其推理所依据的实测事实，可回溯到产物文件'])


TOOLS = {
    'list_specimens': list_specimens,
    'warning_status': warning_status,
    'stiffness': stiffness,
    'data_quality': data_quality,
    'compare': compare,
    'anomaly': anomaly,
    'mechanism': mechanism,
    'localization': localization,
    'l1_migration': l1_migration,
    'doc_search': doc_search,
    'maintenance': maintenance,
}


def call(name, **kw):
    """按名调用工具。"""
    fn = TOOLS.get(name)
    if fn is None:
        return _res(False, name, error='未知工具 %s（可用：%s）' % (name, ', '.join(TOOLS)))
    try:
        return fn(**kw)
    except TypeError as e:
        return _res(False, name, error='参数错误：%s' % e)
