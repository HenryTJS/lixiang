# -*- coding: utf-8 -*-
"""PhysGuard 规则回归测试（零依赖，直接 python agent/test_physguard.py 运行）。

覆盖：P13 定位能力契约、P14 工艺建议边界（两级判定）、P15 主样本定位声明、
P16 处置建议须附证据，以及 maintenance 工具的定级一致性。

这些用例同时是**论文里「物理约束有效性」的定性证据**：
每条规则都给出「应拦截」与「应放过」两侧样本，避免规则只会拒答。
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tools                                                       # noqa: E402
from physguard import PhysGuard                                    # noqa: E402

PG = PhysGuard(enable_log=False)
N_PASS = N_FAIL = 0


def case(desc, expect, got):
    """expect/got ∈ {'pass', 'error', 'warning'}"""
    global N_PASS, N_FAIL
    ok = (expect == got)
    N_PASS += ok
    N_FAIL += (not ok)
    print('  %s %-58s expect=%-8s got=%s' % ('[OK]' if ok else '[XX]', desc, expect, got))


def kind(v):
    return 'error' if v.errors else ('warning' if v.warnings else 'pass')


def ans(text, question='016 怎么处理'):
    return PG.check('final_answer', {'question': question},
                    {'data': {'answer': text}})


def main():
    print('== P13 维修建议的定位能力契约 ==')
    case('主样本返回 inspection_zone → 拦截',
         'error', kind(PG.check('maintenance', {'gid': '016'}, {
             'ok': True, 'gid': '016',
             'data': {'inspection_zone': {'x_range_mm': [40, 70]}}})))
    case('数据集 C 返回 inspection_zone → 放过',
         'pass', kind(PG.check('maintenance', {'gid': 'L1-49'}, {
             'ok': True, 'gid': 'L1-49',
             'data': {'inspection_zone': {'x_range_mm': [40, 70]}},
             'notes': []})))
    case('主样本 inspection_zone 为空 → 放过',
         'pass', kind(PG.check('maintenance', {'gid': '016'}, {
             'ok': True, 'gid': '016', 'data': {'inspection_zone': None}})))

    print('== P14 工艺建议边界（两级判定）==')
    for t in ['016 建议对损伤区打磨后补片胶接，拧紧扭矩取 40 N·m。',
              '该区域宜采用打磨+补片胶接修复工艺。',
              '可对该处进行铆接加强。']:
        case('越界：' + t[:22], 'error', kind(ans(t)))
    case('仅引用提问（引号内）→ 放过',
         'pass', kind(ans('你问的是\u201c打磨后补片胶接还是更换\u201d，'
                          '这超出本数据范围。')))
    case('声明边界 + 无否定语境的工艺词 → 降为 warning',
         'warning', kind(ans('本工具只给处置级别与复检手段，不给出维修工艺或部件选型。'
                             '对损伤部位应先行打磨并做补片处理，具体方案另行评审。')))
    case('否定式表述 → 放过',
         'pass', kind(ans('本数据给不出拧紧扭矩，也不支持给出打磨方案。')))

    print('== P15 主样本定位声明 ==')
    case('主样本给定位结论且未声明 → warning',
         'warning', kind(ans('016 的损伤定位在 X≈53 mm 处。', '016 损伤在哪里')))
    case('主样本声明不可定位 → 放过',
         'pass', kind(ans('016 的 AE 为单通道，无法定位。', '016 损伤在哪里')))
    case('数据集 C 给定位 → 放过',
         'pass', kind(ans('L1-49 的损伤质心在 X≈53 mm。', 'L1-49 损伤在哪里')))

    print('== P16 处置建议须附证据 ==')
    case('空口给处置建议 → warning', 'warning', kind(ans('建议停用检修。')))
    case('附级别依据 → 放过',
         'pass', kind(ans('D 已达 L3（68.6% 寿命），建议停用检修。')))

    print('== maintenance 工具定级一致性 ==')
    for g, label in (('016', '立即处置'), ('017', '立即处置'), ('018', '加强监测'),
                     ('020', '加强监测')):
        r = tools.call('maintenance', gid=g)
        case('%s 定级' % g, label, r['data']['urgency']['label'])
    case('016 置信度', 'high', tools.call('maintenance', gid='016')['data']['confidence'])
    case('020 无刚度信号 → 置信度降级',
         'medium', tools.call('maintenance', gid='020')['data']['confidence'])
    case('主样本不给检修位置',
         True, tools.call('maintenance', gid='016')['data']['inspection_zone'] is None)
    case('数据集 C 给检修位置（X 区间）',
         True, bool(tools.call('maintenance', gid='L1-49')['data']['inspection_zone']))
    r = tools.call('maintenance', gid='L1-03')
    case('数据集 B 未接入定级 → 如实拒答', False, r['ok'])

    print('\n%d 通过 / %d 失败' % (N_PASS, N_FAIL))
    return 1 if N_FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
