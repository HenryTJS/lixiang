# -*- coding: utf-8 -*-
"""L1 冲击点真值核对：PDF 原文 × 坐标约定 × 实测定位质心。

约定假设（由 ae_locate.py 的 IMPACT=(50,80) 反推）：
    PDF 的位置描述是 **从加筋条侧看**；skin 侧坐标 = 180°旋转
      "N cm from right edge" → x_skin = 10N
      "N cm from left  edge" → x_skin = 165 - 10N
      "N cm from top"        → y_skin = 10N
      "N cm from bottom"     → y_skin = 243 - 10N
输出：l1/results/l1_impact_truth.csv（机器可读）+ .txt（人读对照表）
"""
import os
import re

import numpy as np
import pdfplumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(ROOT, 'l1', 'results', 'l1_impact_truth.csv')
OUT_TXT = os.path.join(ROOT, 'l1', 'results', 'l1_impact_truth.txt')
LX, LY = 165.0, 243.0
GROUPS = ['L1-03', 'L1-04', 'L1-05', 'L1-09', 'L1-49', 'L1-50', 'L1-51',
          'L1-52', 'L1-54', 'L1-55', 'L1-56', 'L1-59', 'L1-60']

buf = []
rows = []


def pdf_text(g):
    fp = os.path.join(ROOT, 'l1', g, '%s.pdf' % g)
    if not os.path.exists(fp):
        return ''
    with pdfplumber.open(fp) as pdf:
        return re.sub(r'\s+', ' ', '\n'.join((p.extract_text() or '') for p in pdf.pages))


def parse_impact(t):
    """从原文抽出 (side_x, n_x, side_y, n_y) —— 尽量保守，抽不到就给 None。"""
    rx = re.search(r'([\d.]+)\s*cm\s*from\s*(right|left)\s*edge', t, re.I)
    ry = re.search(r'([\d.]+)\s*cm\s*from\s*(top|bottom)', t, re.I)
    if not rx:                       # L1-09 那种 "8.25 from edge x 10 cm from top right"
        rx = re.search(r'([\d.]+)\s*from\s*edge', t, re.I)
    nx = (float(rx.group(1)), rx.group(2).lower()) if rx and rx.lastindex and rx.lastindex >= 2 else \
         ((float(rx.group(1)), '?') if rx else (None, None))
    ny = (float(ry.group(1)), ry.group(2).lower()) if ry else (None, None)
    stiff = None
    m = re.search(r'stiffener\s*\(\s*(left|right|centre|center)', t, re.I)
    if m:
        stiff = m.group(1).lower()
    surf = 'skin' if re.search(r'on\s+skin', t, re.I) else 'stiffener'
    return nx, ny, stiff, surf


def to_skin(nx, ny):
    x = y = None
    if nx[0] is not None:
        if nx[1] == 'right':
            x = 10.0 * nx[0]
        elif nx[1] == 'left':
            x = LX - 10.0 * nx[0]
    if ny[0] is not None:
        y = 10.0 * ny[0] if ny[1] == 'top' else (LY - 10.0 * ny[0])
    return x, y


buf.append('%-7s %-46s %-16s %-14s %s' %
           ('组', 'PDF 原文（冲击点位置）', '解析(边,cm)', 'skin坐标', '实测质心(X,Y)'))
buf.append('-' * 130)

for g in GROUPS:
    t = pdf_text(g)
    if not t:
        buf.append('%-7s 无 PDF' % g)
        continue
    m = re.search(r'Impact\s*10\s*Joules.{0,200}', t, re.I)
    raw = m.group(0)[:150] if m else '(未找到 Impact 10 Joules)'
    nx, ny, stiff, surf = parse_impact(t)
    x, y = to_skin(nx, ny)

    # 实测质心（仅数据集 C 有 npz）
    cx = cy = ng = None
    fp = os.path.join(ROOT, 'l1', 'results', '_l1_loc_%s.npz' % g)
    if os.path.exists(fp):
        z = np.load(fp)
        xs, ys, r = z['x'], z['y'], z['rms_us']
        good = r <= 5.0
        if good.sum():
            cx, cy, ng = float(xs[good].mean()), float(ys[good].mean()), int(good.sum())

    ptxt = '%s %.2f cm / %s %.2f cm' % (
        (nx[1] or '?'), nx[0] if nx[0] is not None else -1,
        (ny[1] or '?'), ny[0] if ny[0] is not None else -1)
    sk = ('(%.0f, %.0f)' % (x, y)) if (x is not None and y is not None) else '解析失败'
    ctxt = ('(%.0f, %.0f) n=%d' % (cx, cy, ng)) if cx is not None else '—'
    if cx is not None and x is not None:
        ctxt += '  d=%.0fmm' % np.hypot(cx - x, cy - y)
    buf.append('%-7s %-46s %-16s %-14s %s' % (g, raw[:44], ptxt, sk, ctxt))
    rows.append({'gid': g, 'raw': raw, 'nx_cm': nx[0], 'nx_side': nx[1] or '',
                 'ny_cm': ny[0], 'ny_side': ny[1] or '', 'stiffener': stiff or '',
                 'surface': surf, 'x_skin_mm': x, 'y_skin_mm': y,
                 'meas_cx_mm': cx, 'meas_cy_mm': cy, 'n_good': ng})

# 声速表核对
buf.append('')
buf.append('声速（PDF 实测）与 ae_locate.py 的 VEL 字典')
buf.append('-' * 130)
VEL = {'L1-49': (4064.0, 6426.0), 'L1-50': (4077.2, 6470.6),
       'L1-51': (4090.0, 6535.0), 'L1-52': (4289.0, 6729.0),
       'L1-54': (4049.0, 6529.0), 'L1-55': (2933.0, 6623.0),
       'L1-56': (4022.0, 6659.0), 'L1-59': (4000.0, 6729.0),
       'L1-60': (4116.0, 6550.0)}
buf.append('%-7s %-16s %-16s %s' % ('组', 'PDF 横向/纵向', 'VEL 字典', '状态'))
for g in GROUPS:
    t = pdf_text(g)
    v = re.search(r'longitudinal[^:]*:\s*([\d.]+)\s*m/s[^:]*:\s*([\d.]+)\s*m/s', t, re.I)
    if not v:
        v = re.search(r'longitudinal[^:]*:\s*([\d.]+)\s*m/s.{0,80}?lateral[^:]*:\s*([\d.]+)\s*m/s',
                      t, re.I)
    if v:
        lon, lat = float(v.group(1)), float(v.group(2))
        pdfv = '%.1f / %.1f' % (lat, lon)
    else:
        pdfv, lat, lon = '解析失败', None, None
    have = VEL.get(g)
    if have:
        st = 'OK'
    elif g in ('L1-03', 'L1-04', 'L1-05', 'L1-09'):
        st = '故意不登记（PDF 自述 after-failure 测量不可信）'
    elif pdfv != '解析失败':
        st = '★缺失，应补'
    else:
        st = '无数据'
    buf.append('%-7s %-16s %-16s %s' % (
        g, pdfv, ('%.1f / %.1f' % have) if have else '（缺省 4064/6659）', st))

buf.append('')
buf.append('结论')
buf.append('=' * 130)
buf.append('1. PDF 的冲击位置描述逐组不同（至少 5 类），ae_locate.py 原用单一常数 (50,80)。')
buf.append('2. 改用任何统一坐标约定（含 180° 翻转）都无法同时自洽；')
buf.append('   且 PDF 的 left/right 标注与实测 X 质心无相关性。')
# --- 实测 X 质心分布：**由 rows 实时计算**（2026-09-18 起）。
# 原先此处硬编码 "46,49,53,61,75,76,78,79,81 mm（双峰）" —— 一旦 npz 重跑
# （如 min_ch 由 3 改为 4）该数字与论据都会失效，正是 §16 反复强调的那类"写死结论"。
# --- 实测 X 质心分布：**由 rows 实时计算**（2026-09-18 起）。
# 原先此处硬编码 "46,49,53,61,75,76,78,79,81 mm（双峰）" —— 一旦 npz 重跑
# （如 min_ch 由 3 改为 4）该数字与论据都会失效，正是 §16 反复强调的那类"写死结论"。
_cxs = sorted(round(r['meas_cx_mm']) for r in rows
              if r.get('meas_cx_mm') is not None)
_gap = max(((b - a, a, b) for a, b in zip(_cxs, _cxs[1:])), default=(0, 0, 0))
buf.append('3. 实测 X 质心 = %s mm（n=%d 组）'
           % (','.join(str(v) for v in _cxs), len(_cxs)))
buf.append('   最大相邻空隙 = %d mm（%d→%d）⇒ 分布%s，'
           % (_gap[0], _gap[1], _gap[2],
              '呈弱双峰' if _gap[0] >= 10 else '**无双峰结构**'))
buf.append('   既非"同一真值下的散射"，也非 PDF 描述所能解释。')
_ok = [r for r in rows
       if r.get('meas_cx_mm') is not None and r.get('x_skin_mm') is not None]
_ds = [(np.hypot(r['meas_cx_mm'] - r['x_skin_mm'],
                 r['meas_cy_mm'] - r['y_skin_mm']), r['gid']) for r in _ok]
_n_le10 = sum(1 for d, _ in _ds if d <= 10)
_d49 = next((d for d, g in _ds if g == 'L1-49'), None)
buf.append('4. → 仅 L1-49 的 (50,80) 与实测质心相差 %s mm'
           % ('%.0f' % _d49 if _d49 is not None else '—'))
buf.append('   （≤10 mm 的组共 %d 个 / 有质心的 %d 组）⇒ 可作为单点验证。'
           % (_n_le10, len(_ds)))
buf.append('   其余组的"距冲击点"**不得作为定位准确性指标**；')
buf.append('   定位结论只能用组内重复性 σ（与真值无关，故不受影响）。')
buf.append('5. 要拿到可信真值，需 NDT 成像（PDF 提到 Phased array / IR Camera /')
buf.append('   NDT before-after impact），但成像数据不在本仓库中。')

with open(OUT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(buf))

import csv
with open(OUT_CSV, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print('written', OUT_TXT)
print('written', OUT_CSV)
