# -*- coding: utf-8 -*-
"""Agent 核心：意图路由 → 工具执行 → 物理校验 → 语言生成。

设计取舍
--------
**意图识别用规则，不用 LLM。** 原因：
  1. 查询空间窄（几乎都是"某试件某指标"），关键词+正则足够；
  2. 小模型（1.5B~3B）的 function calling 不稳定，规则路由**离线可用且零延迟**；
  3. 结论可复现、可审计 —— 这对工程场景比"灵活"更重要。

LLM 只承担**语言层**（把结构化结果说成人话），这是它最擅长、最不易出错的部分；
若未配置 LLM，则退化为模板渲染（功能完整，只是措辞机械）。

用法::

    from agent.core import Agent
    a = Agent()
    r = a.ask('016 什么时候预警的')
    print(r['answer'])
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tools                                       # noqa: E402
from physguard import PhysGuard                    # noqa: E402

GID_RE = re.compile(r'\b(0\d{2}|L1[-\s]?\d{2})\b', re.I)

# 意图 → (触发关键词, 需要 gid?, 说明)
INTENTS = [
    ('list_specimens', ['有哪些试件', '试件列表', '哪些数据', '所有试件', '数据清单',
                        '有哪些数据', '可用数据'], False, '列出全部试件与可用性'),
    ('data_quality', ['数据质量', '可靠吗', '能不能用', '采集质量', '数据可靠',
                      '数据怎么样', '能不能信'], True, '数据采集质量体检'),
    ('warning_status', ['预警', '报警', '什么时候坏', '损伤度', 'D(t)', 'D 值',
                        '什么时候', '剩余', '提前量'], True, 'D(t) 三级预警状态'),
    ('stiffness', ['刚度', '退化到', '刚度损失', '变软', '刚度变化'], True, '刚度损失率轨迹'),
    ('compare', ['对比', '比较', '谁先', '哪个先', '谁更', '排序'], False, '多试件横向对比'),
    ('maintenance', ['维修', '检修', '维护', '处置', '怎么处理', '如何处理', '怎么办',
                     '要不要修', '下一步', '处理建议'], True, '检修建议与处置级别'),
    ('mechanism', ['机制', '剪切', '拉伸', 'RA', 'AF', '脱粘', '开裂', '什么模式',
                   '机理'], True, 'RA–AF 损伤机制'),
    ('localization', ['定位', '位置', '在哪', '哪个位置', '坐标'], True, 'AE 事件定位'),
    ('anomaly', ['异常', '偏离', '异常检测', '离群'], True, '无监督异常检测'),
    ('l1_migration', ['L1 迁移', 'remap', 'reMAP', '论文复现', 'broer'], True, 'L1 第一批 D(t)'),
    ('doc_search', ['为什么', '原理', '方法', '怎么做', '解释', '什么是', '依据'], False, '文档检索'),
]


def route(question):
    """规则路由：返回 (intent, kwargs)。"""
    q = str(question)
    ql = q.lower()
    gids = []
    for m in GID_RE.finditer(q):
        g = tools.norm_gid(m.group(1))
        if g not in gids:
            gids.append(g)
    gid = gids[0] if gids else None

    # 多试件 → compare
    if len(gids) >= 2:
        return 'compare', {'gids': gids}

    best, score = None, 0
    for name, kws, need_gid, _ in INTENTS:
        s = sum(1 for k in kws if k.lower() in ql)
        if s > score:
            best, score = name, s
    if best is None:
        return ('warning_status', {'gid': gid}) if gid else ('list_specimens', {})

    if best == 'doc_search':
        return 'doc_search', {'query': q}
    need_gid = dict((n, ng) for n, _, ng, _ in INTENTS)[best]
    if need_gid:
        if not gid:
            return 'ask_gid', {'intent': best}
        return best, {'gid': gid}
    return best, {}


# ---------------------------------------------------------------- 语言层
def render_template(intent, kw, res, verdict):
    """无 LLM 时的模板渲染（保证功能完整）。"""
    if intent == 'ask_gid':
        return ('请指明试件编号（如 016 / L1-49）。当前可用：\n'
                '  数据集 A 主样本：016 017 018 019 020\n'
                '  数据集 B L1 第一批：L1-03 L1-04 L1-05 L1-09\n'
                '  数据集 C L1 第二批：L1-49 50 51 52 54 55 56 59 60')
    if not res.get('ok'):
        return '【无法回答】%s' % res.get('error', '未知原因')
    d, gid = res.get('data', {}), res.get('gid')
    tag = ('%s ' % gid) if gid else ''

    if intent == 'list_specimens':
        lines = ['共三类数据集：']
        for ds, name in [(tools.DS_MAIN, 'A 主样本（5 组）'),
                         (tools.DS_L1A, 'B L1 第一批（4 组，有 FBG）'),
                         (tools.DS_L1B, 'C L1 第二批（9 组，无 FBG）')]:
            gs = [s['gid'] + (('（%s）' % s['note']) if s['note'] else '')
                  for s in d['specimens'] if s['dataset'] == ds]
            lines.append('  %s：%s' % (name, '，'.join(gs)))
        return '\n'.join(lines)

    if intent == 'warning_status':
        lv = d.get('levels')
        s = ('%s预警触发于寿命 **%.1f%%**（相对扩展标签 b2 偏差 %+.1f%%），'
             '距断裂提前 **%.1f%%**。当前级别判定：%s。'
             % (tag, d['t_warn_pct'], d['err_pct'], d['lead_pct'], d['grade']))
        if lv:
            s += ('\n三级时刻：注意 %.1f%% → 预警 %.1f%% → 临危 %.1f%%，闸门 %s。'
                  % (lv['L1'], lv['L2'], lv['L3'], lv['gate']))
        return s

    if intent == 'stiffness':
        th = d['thresholds']
        ts = '、'.join('%s@%s%%' % (k.replace('L>=', 'L≥'), (v if v is not None else '未达'))
                       for k, v in th.items())
        return ('%s刚度最大变化 %.2f，阈值时刻：%s。'
                % (tag, d['stiff_max'], ts))

    if intent == 'data_quality':
        return ('%s采样 %.1f Hz（最大间隔 %.1f s），时长 %.2f h，'
                'AE 等效频率 %s Hz，通道口径 %s。'
                % (tag, d['sampling_hz'] or -1, d['dt_max_s'], d['duration_h'] or -1,
                   d['ae_effective_hz'], d['demod_mode']))

    if intent == 'compare':
        rs = d['rows']
        return '对比：\n' + '\n'.join(
            '  %-6s 预警 %s%%  提前 %s%%  刚度末值 %s' %
            (r['gid'], r.get('t_warn_pct', '-'), r.get('lead_pct', '-'), r.get('stiff_end', '-'))
            for r in rs)

    if intent == 'mechanism':
        seg = d['shear_frac_by_quintile']
        return ('%s剪切型占比随寿命 5 分段：%s，变化 %+.3f。\n'
                '（RA 中位：%s）— 机制由拉伸型向剪切型演化的趋势判断。'
                % (tag, ' → '.join('%.3f' % x for x in seg), d['delta'],
                   ' → '.join('%.2f' % x for x in d['ra_median'])))

    if intent == 'localization':
        return ('%s可信定位 %d/%d 个，rms 中位 %.2f µs，质心 (%.0f, %.0f) mm。'
                % (tag, d['n_good'], d['n_total'], d['rms_median_us'],
                   d['centroid_x_mm'], d['centroid_y_mm']))

    if intent == 'anomaly':
        return ('%s首次超限：马氏 %.1f%%，PCA %.1f%%；末段偏离为基线的 %.1f 倍。'
                % (tag, d['first_exceed_maha_pct'] or -1,
                   d['first_exceed_pca_pct'] or -1, d['late_ratio']))

    if intent == 'maintenance':
        u = d['urgency']
        out = ['%s处置级别：**%s**（%s）  [置信度 %s]'
               % (tag, u['label'], u['hint'], d['confidence']),
               '定级依据：' + '；'.join(u['basis'])]
        if d.get('actions'):
            out.append('建议动作：')
            for a in sorted(d['actions'], key=lambda x: x['priority']):
                out.append('  P%d）%s' % (a['priority'], a['action']))
        z = d.get('inspection_zone')
        out.append('检修范围：' + (
            'X ∈ [%.1f, %.1f] mm（质心 %.1f mm；%.1f mm 为 X 向 σ）'
            % (z['x_range_mm'][0], z['x_range_mm'][1], z['centroid_x_mm'],
               15.0)
            if z else '无定位能力 → 只能整体处置，不支持分区检修'))
        if d.get('recheck'):
            out.append('复检项：' + '；'.join(d['recheck']))
        rest = [x for x in d['not_recommended'] if x not in d.get('recheck', [])]
        if rest:
            out.append('不建议：' + '；'.join(rest))
        return '\n'.join(out)

    if intent == 'l1_migration':
        ks = [k for k in ('D_end', 't25_pct', 't55_pct', 't85_pct') if d.get(k) is not None]
        if not ks:
            return '%sD(t) 迁移结果已返回（字段：%s）。' % (tag, '、'.join(list(d)[:8]))
        return '%sD(t) 迁移：%s。' % (tag, '  '.join('%s=%s' % (k, d[k]) for k in ks))

    if intent == 'doc_search':
        hits = d.get('hits', [])
        if not hits:
            return '未在文档中找到相关说明。'
        return '相关说明（%s）：\n%s' % (hits[0]['source'], hits[0]['text'])

    return str(d)


def render_llm(question, intent, res, verdict, model_fn):
    """有 LLM 时的渲染：把结构化结果交给模型措辞。"""
    import json
    facts = json.dumps({'intent': intent, 'result': res.get('data'),
                        'evidence': res.get('evidence'),
                        'known_limits': res.get('notes')},
                       ensure_ascii=False, indent=1)
    warns = '；'.join(m for _, m in verdict.warnings) or '无'
    prompt = (
        '你是结构健康监测（SHM）分析助手。请**仅依据**下面的事实回答用户问题，'
        '不得引入事实之外的数据、试件名称或数值。若事实不足，明确说明不足。\n'
        '已知的物理/数据限制（必须在相关时主动提示）：%s\n'
        '事实：\n%s\n\n用户问题：%s\n回答：' % (warns, facts, question))
    return model_fn(prompt)


# ---------------------------------------------------------------- 主循环
class Agent:
    def __init__(self, llm=None, verbose=True):
        """llm: 可选的可调用对象 fn(prompt)->str；为 None 时用模板渲染。"""
        self.llm = llm
        self.pg = PhysGuard()
        self.verbose = verbose
        self.history = []

    def ask(self, question):
        intent, kw = route(question)
        res = tools.call(intent, **kw) if intent != 'ask_gid' else tools._res(
            False, 'ask_gid', error='缺少试件编号')
        verdict = self.pg.check(intent, kw, res)
        if self.llm is not None and res.get('ok'):
            try:
                answer = render_llm(question, intent, res, verdict, self.llm)
            except Exception as e:
                answer = render_template(intent, kw, res, verdict)
                verdict.warn('LLM', '语言层调用失败，已回退模板：%s' % e)
        else:
            answer = render_template(intent, kw, res, verdict)

        rec = {'question': question, 'intent': intent, 'kwargs': kw,
               'ok': res.get('ok'), 'answer': answer,
               'evidence': res.get('evidence', []),
               'notes': res.get('notes', []),
               'blocked': not verdict.ok,
               'warnings': [m for _, m in verdict.warnings],
               'errors': [m for _, m in verdict.errors]}
        self.history.append(rec)
        return rec

    def stats(self):
        return self.pg.stats()


if __name__ == '__main__':
    a = Agent()
    for q in ['有哪些数据', '016 什么时候预警的', '022 的数据质量怎么样',
              'L1-49 的机制是什么', '对比 016 和 017']:
        print('Q:', q)
        r = a.ask(q)
        print('A:', r['answer'])
        if r['blocked']:
            print('   ❌ 拦截:', r['errors'])
        if r['warnings']:
            print('   ⚠️ ', '；'.join(r['warnings'][:2]))
        print()
    print('统计:', a.stats())
