# -*- coding: utf-8 -*-
"""L1 元信息自适应加载：从各组自带的 L1-xx.pdf 自动解析

解析项:
  - n_f    失效循环数 (Applied loads 表中 "-6.5 kN" 所在行的 5~7 位整数)
  - foot_L / foot_R  两条 ODiSi 加强筋脚的空间段 (ODiSi-B location 表:
                     "1 Left foot <start> <end> ..." / "2 Right foot <start> <end> ...")

论文检测点 refs 属"标签"(非数据), 仅 03/04/05 有, 保持静态。

用法:
  from l1_meta import load_meta
  m = load_meta('L1-09')   # -> dict(n_f, foot_L, foot_R, refs)
"""
import os
import re
import functools

ROOT = os.path.dirname(os.path.abspath(__file__))

# 论文检测点(标签; 仅 03/04/05 收录于 Broer et al. 2021)
REFS = {
    'L1-03': [('冲击后扩展', 10000), ('刚度退化', 69000),
              ('AE脱粘', 130000), ('应变脱粘', 143000)],
    'L1-04': [('前段低活动', 5000), ('刚度退化(误报)', 30000),
              ('应变脱粘', 239500), ('AE脱粘', 260000)],
    'L1-05': [('短暂disbond', 66500), ('刚度退化', 68000),
              ('AE脱粘', 100000), ('应变脱粘', 110000)],
}


def _pdf_text(gid):
    import pdfplumber
    fp = os.path.join(ROOT, gid, f'{gid}.pdf')
    with pdfplumber.open(fp) as pdf:
        return '\n'.join((p.extract_text() or '') for p in pdf.pages)


def fbg_foot_cols(columns, ch_foot):
    """按 PDF 通道→脚 映射, 把 CSV 的 FBG **应变**列分配到左右脚。
    应变组 = 表头里前缀不同的两组(如 fbg* 与 b*), 按出现顺序依次对应 Channel 1,2,...
    (排除 FBG_A*/FBG_B* 波长列)。ch_foot: {通道号:'Left'/'Right'}。
    返回 {'Left':[...], 'Right':[...]}。
    """
    order, groups = [], {}
    for c in columns:
        m = re.fullmatch(r'([A-Za-z_]+?)(\d+)', c)
        if not m:
            continue
        pre = m.group(1)
        if pre.upper() in ('FBG_A', 'FBG_B'):            # 波长列, 非应变
            continue
        if pre not in groups:
            groups[pre] = []
            order.append(pre)
        groups[pre].append(c)
    out = {}
    for i, pre in enumerate(order, start=1):
        foot = ch_foot.get(i)
        if foot:
            out.setdefault(foot, []).extend(groups[pre])
    return out


@functools.lru_cache(maxsize=None)
def load_meta(gid):
    """返回 dict(n_f, foot_L, foot_R, fbg_ch_foot, refs)。解析失败项为 None(调用方可退回)。"""
    n_f, foot_L, foot_R, ch_foot = None, None, None, {}
    try:
        txt = _pdf_text(gid)
        # --- n_f: 载荷表中的循环数(其后跟两个 kN 值); 支持千分位逗号与分段加载表,
        #     取最大值(最终段)。如 "438,000 -6 kN -60 kN ..." / "152458 -6.5 kN -65 kN" ---
        cands = []
        for line in txt.split('\n'):
            m = re.search(r'([\d,]{4,10})\s+-?\d+(?:\.\d+)?\s*kN\s+-?\d+(?:\.\d+)?\s*kN', line)
            if m:
                cands.append(int(m.group(1).replace(',', '')))
        if cands:
            n_f = max(cands)
        # --- ODiSi 脚空间段: "1 Left foot 2580 2780 Bottom Top" ---
        for line in txt.split('\n'):
            m = re.match(r'\s*\d\s+(Left|Right)\s+foot\s+(\d+)\s+(\d+)', line)
            if m:
                side = m.group(1)
                a, b = int(m.group(2)), int(m.group(3))
                if side == 'Left':
                    foot_L = (min(a, b), max(a, b))
                else:
                    foot_R = (min(a, b), max(a, b))
        # --- FBG 通道→脚: "Channel 1: Left foot from top" ---
        for line in txt.split('\n'):
            m = re.search(r'Channel\s+(\d+)\s*:\s*(Left|Right)\s+foot', line, re.I)
            if m:
                ch_foot[int(m.group(1))] = m.group(2).capitalize()
    except Exception as e:                                   # noqa: BLE001
        print(f'[l1_meta] {gid} PDF 解析失败: {e}')
    return dict(n_f=n_f, foot_L=foot_L, foot_R=foot_R, fbg_ch_foot=ch_foot,
                refs=REFS.get(gid, []))


if __name__ == '__main__':
    for g in ['L1-03', 'L1-04', 'L1-05', 'L1-09']:
        print(g, load_meta(g))
