# -*- coding: utf-8 -*-
"""物理约束验证层（PhysGuard）。

作用
----
LLM/Agent 在工程任务上会**幻觉**：给不存在的试件编造结论、把"无信号"说成"健康"、
在数据不连续时仍给出定量判断。本层对**每一条结论**做物理一致性检查，
违规即拦截或降级，并把**每次判定记入日志**。

为什么需要（也是研究价值所在）
------------------------------
本项目积累了完整的**物理先验**（三级语义、刚度闸门、证据层，以及大量
"什么条件下什么方法必然失效"的实测边界，见 README §3.5 / §5）。
这些先验原本是给人看的文档，本层把它们变成**可执行的校验规则**。

规则分级
--------
- ``error``   : 结论在物理上不可能 → 必须拦截
- ``warning`` : 结论可能误导 → 需在回答中显式标注
- 每条规则的**触发次数会被统计**，可用于量化"物理约束带来的可靠性提升"。

用法::

    from agent.physguard import PhysGuard
    pg = PhysGuard()
    v = pg.check('warning_status', {'gid': '016'}, result)
    print(v.ok, v.errors, v.warnings)
"""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(ROOT, 'agent', 'log', 'physguard_log.jsonl')

# 已知口径边界（与 agent/tools.py 保持一致）
MAIN_GROUPS = ['016', '017', '018', '019', '020']
L1B_GROUPS = ['L1-49', 'L1-50', 'L1-51', 'L1-52', 'L1-54',
              'L1-55', 'L1-56', 'L1-59', 'L1-60']
# 时间标定不可靠的组（§3.3 / §5）
UNRELIABLE_TIME = ['L1-51', 'L1-56']
# 覆盖度不足的组（AE / DFOS）
POOR_COVERAGE = {'L1-59': 'AE 覆盖 62.7%', 'L1-60': 'AE 70.8% / DFOS 67.5%',
                 'L1-55': 'DFOS 覆盖 72.2%'}
# 等间隔分窗 AE 组（AE 行数 ÷ 真实时长 = 采样率本身 10.00 条/s，与应变严格 1:1）
# ⚠️ 2026-09-20 修正: 原先此处名为 LOW_RATE 并断言「1 Hz 采样」—— **该断言是错的**。
#    全组采样率恒为 10 Hz；018/019/020 第一列是**整数计数器**（英文导出工具），
#    不是秒。把计数器当秒读会把时长放大 10 倍。真正与其余组的差异是 **AE 记录口径**
#    （等间隔分窗 vs 016/017 的事件驱动），与采样率无关。
WINDOW_AE = ['018', '019', '020']
# 阈值语义（README §2.2）
D_NOTICE, D_WARN, D_CRIT = 0.25, 0.55, 0.85

# ── 维修建议相关（P13~P16）────────────────────────────────────────
# 具体维修工艺类措辞：应变/AE 证据无法支撑，不得作为建议输出
REPAIR_PROC_TERMS = ['打磨', '补片', '胶接', '铆接', '拧紧', '扭矩', '焊接',
                     '喷丸', '注胶', '铺层', '固化', '灌注', '更换螺栓']
# 处置动作措辞（用于检查建议是否附了证据）
ACTION_TERMS = ['停用', '检修', '更换', '限载', '降载', '大修', '报废']
# 能力边界声明：出现任一则说明回答已声明「不给工艺方案」
BOUNDARY_TERMS = ['不给出维修工艺', '超出所用证据', '超出证据', '能力范围', '不能给',
                  '给不出', '无法给', '不予给出', '不支持给出', '不属 SHM', '不属结构维修']
# 证据词：出现任一即认为处置建议附了依据
EVIDENCE_TERMS = ['D(', 'D(t)', 't_warn', '刚度', 'L1', 'L2', 'L3',
                  '0.25', '0.55', '0.85']
# 否定/免责语境标记（避免把"不给出拧紧扭矩"误判为工艺建议）
_NEG_CHARS = '不无未别避免勿非超'
# 引号对：引号内通常是「引用用户提问」，不是 Agent 自己的建议
_QUOTE_PAIRS = [('“', '”'), ('「', '」'), ('"', '"'), ('‘', '’'), ('《', '》')]


def _answer_text(res):
    """取最终回答文本（仅 final_answer 意图有效）。"""
    return str((res.get('data') or {}).get('answer') or '')


def _negated(text, i, n=0, win=20):
    """判断 text[i:i+n] 是否处于否定/免责语境（前后窗口都看）。"""
    seg = text[max(0, i - win):i] + text[i + n:i + n + win]
    return any(c in seg for c in _NEG_CHARS)


def _strip_quoted(text):
    """去掉引号内的内容（引用用户提问、引用规范原文等）。"""
    out = text
    for a, b in _QUOTE_PAIRS:
        while True:
            i = out.find(a)
            if i < 0:
                break
            j = out.find(b, i + 1)
            if j < 0:
                break
            out = out[:i] + ' ' + out[j + 1:]
    return out


def _hits_unnegated(text, terms):
    """返回在非否定语境下命中的词表。"""
    out = []
    for t in terms:
        i = text.find(t)
        while i >= 0:
            if not _negated(text, i, len(t)):
                out.append(t)
                break
            i = text.find(t, i + 1)
    return out


class Verdict:
    def __init__(self):
        self.ok = True
        self.errors = []      # [(rule, msg)]
        self.warnings = []    # [(rule, msg)]
        self.checked = []     # 通过检查的规则 id

    def err(self, rid, msg):
        self.ok = False
        self.errors.append((rid, msg))

    def warn(self, rid, msg):
        self.warnings.append((rid, msg))

    def to_dict(self):
        return {'ok': self.ok,
                'errors': [{'rule': r, 'msg': m} for r, m in self.errors],
                'warnings': [{'rule': r, 'msg': m} for r, m in self.warnings],
                'n_checked': len(self.checked) + len(self.errors) + len(self.warnings)}


# ---------------------------------------------------------------- 规则
def _p1_range(intent, kw, res, v):
    """P1: 损伤度必须落在 [0,1]。"""
    d = res.get('data') or {}
    for key in ('D_end', 'damage', 'd_end'):
        if key in d:
            x = float(d[key])
            if not (0.0 <= x <= 1.0):
                v.err('P1', '损伤度 %.3f 超出物理范围 [0,1]' % x)
                return
    v.checked.append('P1')


def _p2_chronology(intent, kw, res, v):
    """P2: 预警时刻必须早于失效锚（否则计时序矛盾）。"""
    d = res.get('data') or {}
    tw, b3 = d.get('t_warn_pct'), d.get('failure_anchor_b3_pct')
    if tw is not None and b3 is not None and float(tw) >= float(b3):
        v.err('P2', '预警时刻(%.1f%%) 不早于失效锚(%.1f%%)' % (tw, b3))
        return
    v.checked.append('P2')


def _p3_healthy(intent, kw, res, v):
    """P3: 声称"健康/正常"时 D 必须低于注意级阈值。"""
    d = res.get('data') or {}
    claim = str(d.get('assessment', '') or kw.get('claim', ''))
    if any(s in claim for s in ('健康', '正常', '无损伤')):
        dd = d.get('D_end', d.get('damage'))
        if dd is not None and float(dd) >= D_NOTICE:
            v.err('P3', '声称健康但损伤度 %.2f ≥ 注意级阈值 %.2f' % (float(dd), D_NOTICE))
            return
    v.checked.append('P3')


def _p4_degradation_direction(intent, kw, res, v):
    """P4: 声称"退化/损伤"时，D 或刚度至少一个应呈上升。"""
    d = res.get('data') or {}
    claim = str(d.get('assessment', '') or kw.get('claim', ''))
    if any(s in claim for s in ('退化', '损伤', '劣化')):
        dd = d.get('D_end', d.get('damage'))
        se = d.get('stiff_end')
        vals = [float(x) for x in (dd, se) if x is not None]
        if vals and all(x <= 0.01 for x in vals):
            v.warn('P4', '声称退化，但 D 与刚度均无上升迹象 → 结论依据不足')
            return
    v.checked.append('P4')


def _p5_continuity(intent, kw, res, v):
    """P5: 采集不连续时不得给出定量结论。"""
    if intent == 'data_quality':
        d = res.get('data') or {}
        if d.get('dt_max_s') and float(d['dt_max_s']) > 1.5:
            v.warn('P5', '采集不连续（最大间隔 %.1f s）→ 不可用于定量退化判断'
                   % float(d['dt_max_s']))
            return
    v.checked.append('P5')


def _p6_ae_no_timestamp(intent, kw, res, v):
    """P6: 主样本 AE 无时间戳 → 禁止事件速率/定位类结论。"""
    gid = str(res.get('gid') or kw.get('gid') or '')
    if gid in MAIN_GROUPS:
        if intent in ('localization', 'event_rate', 'tdoa'):
            v.err('P6', '主样本 AE 无到达时间且单通道 → %s 不可用' % intent)
            return
    v.checked.append('P6')


def _p7_localization_scope(intent, kw, res, v):
    """P7: 定位结论必须限定在可信维度（X 散布小 / Y 散布大，且勿称准确性）。"""
    if intent == 'localization' and res.get('ok'):
        d = res.get('data') or {}
        if 'centroid_y_mm' in d:
            v.warn('P7', 'Y（沿加筋条）方向跨组质心散布 σ≈46 mm，不可用于跨试件比较；'
                         'X 方向 σ≈14 mm，且这两个 σ 都**只是跨组散布，不是准确性**'
                         '（13 组冲击位置真值未证实）')
        return
    v.checked.append('P7')


def _p8_demod(intent, kw, res, v):
    """P8: 未解调通道不得使用幅值绝对值。"""
    if intent == 'data_quality':
        d = res.get('data') or {}
        if d.get('demod_mode') == 'RAW':
            v.warn('P8', '应变未解调 → 幅值须用块内 std 口径，直接读数值会得到直流电平')
            return
    v.checked.append('P8')


def _p9_stiffness_reference(intent, kw, res, v):
    """P9: 刚度无信号时不得作为独立参照。"""
    if intent == 'stiffness' and res.get('ok'):
        d = res.get('data') or {}
        if float(d.get('stiff_max', 0)) < 0.05:
            v.warn('P9', '刚度最大变化仅 %.3f → 无显著退化信号，不可作为独立参照'
                   % float(d['stiff_max']))
            return
    v.checked.append('P9')


def _p10_window_ae(intent, kw, res, v):
    """P10: 等间隔分窗 AE 组不得把「记录密度」当作 AE 事件速率。

    ⚠️ 2026-09-20 修正: 旧版写的是「1 Hz 采样组不得声称高时间分辨率结论」——
    前提本身错误（采样率恒为 10 Hz，时间分辨率与 016/017 相同）。
    真正成立的约束是 AE 口径：018/019/020 的 AE 是**每采样点一条特征记录**
    （10.00 条/s = 采样率），不是逐次撞击的事件流 → 速率类结论无意义。
    """
    gid = str(res.get('gid') or kw.get('gid') or '')
    if gid in WINDOW_AE:
        if intent in ('stiffness', 'data_quality', 'event_rate'):
            v.warn('P10', '该组 AE 为等间隔分窗口径（10 条/s = 采样率本身，与应变 1:1）'
                           '→ 不得把记录密度当作 AE 事件速率')
            return
    v.checked.append('P10')


def _p11_coverage(intent, kw, res, v):
    """P11: 覆盖度不足的试件必须标注。"""
    gid = str(res.get('gid') or kw.get('gid') or '')
    if gid in POOR_COVERAGE:
        v.warn('P11', '%s（%s）→ 后段结论不可信' % (gid, POOR_COVERAGE[gid]))
        return
    v.checked.append('P11')


def _p12_unreliable_time(intent, kw, res, v):
    """P12: 时间标定不可靠的组不得纳入跨组统计。"""
    gid = str(res.get('gid') or kw.get('gid') or '')
    if isinstance(kw.get('gids'), (list, tuple)) and set(kw['gids']) & set(UNRELIABLE_TIME):
        v.warn('P12', '对比中包含时间标定不可靠的组（%s）→ 跨组结论需排除'
               % '/'.join(sorted(set(kw['gids']) & set(UNRELIABLE_TIME))))
        return
    if gid in UNRELIABLE_TIME and intent in ('mechanism', 'anomaly'):
        v.warn('P12', '%s 时间标定不可靠 → 其趋势结论不纳入统计' % gid)
        return
    v.checked.append('P12')


def _p13_zone_contract(intent, kw, res, v):
    """P13: 无定位能力的数据集不得给出检修位置（工具契约）。"""
    if intent == 'maintenance' and res.get('ok'):
        gid = str(res.get('gid') or kw.get('gid') or '')
        d = res.get('data') or {}
        if d.get('inspection_zone') and gid in MAIN_GROUPS:
            v.err('P13', '主样本 %s 的 AE 为单通道、无到达时间 → 无 TDOA 定位能力，'
                         '不得给出检修位置' % gid)
            return
    v.checked.append('P13')


def _p14_no_repair_technique(intent, kw, res, v):
    """P14: 不得给出具体维修工艺／部件更换方案（超出所用证据的能力范围）。

    两级判定：
    - 出现工艺词且**全文无能力边界声明** → error（确属越界）
    - 出现工艺词但**已声明边界** → warning（可能只是引用或拒答，交人工确认措辞）

    已知局限：规则只能看词形，无法完全区分「建议打磨」与「不能给打磨方案」。
    因此先剥离引号内容（通常为引用用户提问），再用否定语境过滤。
    """
    if intent == 'final_answer':
        a = _strip_quoted(_answer_text(res))
        hit = _hits_unnegated(a, REPAIR_PROC_TERMS)
        if hit:
            if any(t in a for t in BOUNDARY_TERMS):
                # 已声明边界 → 降为 warning，并附上下文供人工复核
                ctx = ''
                for t in hit:
                    i = a.find(t)
                    while i >= 0:
                        if not _negated(a, i, len(t)):
                            ctx = a[max(0, i - 20):i + len(t) + 20].replace('\n', ' ')
                            break
                        i = a.find(t, i + 1)
                    if ctx:
                        break
                v.warn('P14', '回答提到工艺词（%s）但已声明能力边界 → 请确认不是建议语气；'
                              '上下文：「…%s…」' % ('、'.join(hit), ctx))
            else:
                v.err('P14', '回答含具体维修工艺措辞（%s）→ 应变/AE 证据不支持工艺方案，'
                             '应改为「处置级别＋检修范围＋复检手段」' % '、'.join(hit))
            return
    v.checked.append('P14')


def _p15_main_localization(intent, kw, res, v):
    """P15: 涉及主样本时不得给出定位结论，须显式声明不可定位。"""
    if intent == 'final_answer':
        a = _answer_text(res)
        q = str(kw.get('question') or '')
        gids = [g for g in MAIN_GROUPS if g in q or g in a]
        if gids and any(t in a for t in ('定位', '质心', '坐标', '区域位于')):
            if not any(t in a for t in ('不可', '无定位', '不支持', '无法定位')):
                v.warn('P15', '回答涉及主样本 %s 的定位 → 该数据 AE 单通道无 TDOA，'
                              '须显式声明「不可定位」' % '/'.join(gids))
                return
    v.checked.append('P15')


def _p16_advice_evidence(intent, kw, res, v):
    """P16: 处置建议必须附证据（级别/阈值/刚度），不得空口给结论。

    同样先剥离引号内容 —— 否则「你问的是『要不要更换』」会被误判为处置建议。
    """
    if intent == 'final_answer':
        a = _strip_quoted(_answer_text(res))
        if any(t in a for t in ACTION_TERMS) and not any(t in a for t in EVIDENCE_TERMS):
            v.warn('P16', '回答给出了处置建议，但未出现任何损伤级别或阈值依据 → '
                          '建议须附证据链')
            return
    v.checked.append('P16')


RULES = [_p1_range, _p2_chronology, _p3_healthy, _p4_degradation_direction,
         _p5_continuity, _p6_ae_no_timestamp, _p7_localization_scope, _p8_demod,
         _p9_stiffness_reference, _p10_window_ae, _p11_coverage, _p12_unreliable_time,
         _p13_zone_contract, _p14_no_repair_technique, _p15_main_localization,
         _p16_advice_evidence]


class PhysGuard:
    """物理约束校验器（含审计日志）。"""

    def __init__(self, log_path=LOG_PATH, enable_log=True):
        self.log_path = log_path
        self.enable_log = enable_log
        self.history = []

    def check(self, intent, kw, result):
        """对一次工具调用结果做全部规则校验。"""
        v = Verdict()
        for rule in RULES:
            try:
                rule(intent, kw or {}, result or {}, v)
            except Exception as e:                       # 规则本身出错不应影响主流程
                v.warn(rule.__name__, '规则执行异常：%s' % e)
        rec = {'intent': intent, 'kwargs': {k: str(v_) for k, v_ in (kw or {}).items()},
               'ok': v.ok,
               'errors': [r for r, _ in v.errors],
               'warnings': [r for r, _ in v.warnings],
               'n_checked': len(v.checked)}
        self.history.append(rec)
        if self.enable_log:
            try:
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            except Exception:
                pass
        return v

    def stats(self):
        """本次会话的规则触发统计（可直接作为论文表格）。"""
        n = len(self.history)
        cnt_e, cnt_w, cnt_c = {}, {}, {}
        for r in self.history:
            for k in r['errors']:
                cnt_e[k] = cnt_e.get(k, 0) + 1
            for k in r['warnings']:
                cnt_w[k] = cnt_w.get(k, 0) + 1
            cnt_c['total'] = cnt_c.get('total', 0) + r['n_checked']
        return {'n_calls': n,
                'n_blocked': sum(1 for r in self.history if not r['ok']),
                'n_with_warning': sum(1 for r in self.history if r['warnings']),
                'error_hits': cnt_e, 'warning_hits': cnt_w,
                'n_rule_checks': cnt_c.get('total', 0)}
