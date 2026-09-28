# -*- coding: utf-8 -*-
"""真 ReAct Agent：LLM 驱动的多步推理循环。

与 core.py 的区别
-----------------
``core.py``  关键词路由 + 模板填充（单轮、单工具）—— 本质是**查询界面**。
``react.py`` LLM 自主决定：这一步查什么、发现疑点后追加查询、何时收尾 —— 是**Agent**。

循环
----
::

    用户问题
      ↓  ┌─────────────── 最多 max_steps 轮 ───────────────┐
      │  [LLM]  输出 tool_calls（可多个）或最终回答          │
      │    ↓                                                │
      │  [tools]     执行（只读产物，带证据链）              │
      │  [physguard] 校验（每条规则都过，违规即标注）         │
      │    ↓ 结果回灌                                       │
      └─────────────────────────────────────────────────────┘
      ↓ 无 tool_calls
    最终回答（再过一次校验，并附完整推理轨迹）

用法
----
    export DEEPSEEK_API_KEY=sk-xxx        # 或 --api-key
    python agent/react.py "016 现在什么状态"
    python agent/react.py --trace "L1-49 的机制和定位都说说"
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tools                                       # noqa: E402
from physguard import PhysGuard                    # noqa: E402
from envfile import load_env, describe             # noqa: E402

# .env 必须在下面读取 LLM_API_BASE / LLM_MODEL **之前**加载，
# 否则写在 .env 里的模型名/端点不会生效。
# 顺序：<项目根>/.env → <agent>/.env；已存在的环境变量优先，不会被覆盖。
ENV_FILE = load_env()

API_BASE = os.environ.get('LLM_API_BASE', 'https://api.deepseek.com')
DEFAULT_MODEL = os.environ.get('LLM_MODEL', 'deepseek-flash')

SYSTEM_PROMPT = """你是「多源在线损伤度 D(t)」项目的结构健康监测(SHM)分析助手。

你的工作是**主动用工具查证**，而不是凭记忆回答。规则：

1. **先想清楚要查什么**。用户问「状态」时，D(t) 预警只是一个侧面；
   应当进一步查刚度退化（独立物理参照）与数据质量（结论是否可信）。
2. **发现疑点时追加查询**。例如预警时刻与刚度阈值时刻矛盾、数据覆盖不足、
   采样口径不匹配 —— 都要主动查证并在回答中说明。
3. **只依据工具返回的事实**。不要编造试件编号、数值或不存在的能力。
   工具返回 ok=false 时，如实说明"该数据不支持此分析"。
4. **注意工具返回的 notes 与 warnings**：它们是已知的物理/数据限制，
   必须在相关结论中主动提示用户。
5. 信息足够后给出结论，不要再无意义地调用工具。
6. **给出处置/维修建议时**（用户问"怎么处理""要不要修""下一步做什么"时），
   必须调用 `maintenance` 工具 —— 它能给出的只有：处置级别、检修范围、复检手段。
   不要自行发挥维修工艺或部件选型（超出本数据的能力范围）。

项目背景（简）：
- 数据集 A = 主样本 016-020（AE 无时间戳、单通道；仅 016 应变为解调口径）
- 数据集 B = L1 第一批 L1-03/04/05/09（有 FBG，可靠 cycle 锚）
- 数据集 C = L1 第二批 L1-49~L1-60（无 FBG，时间对齐不可靠）
- 三级语义：D≥0.25 检测起始 / ≥0.55 不可逆确认 / ≥0.85 已无裕度
"""

# ⚠️ 消融实验专用：**中性**系统提示（不含任何领域约束）。
#
# 为什么必须有它：生产用的 SYSTEM_PROMPT 里写了「数据集 A 的 AE 无时间戳、单通道」
# 这类物理事实 —— 那等于把约束**提前送给了所有臂**。实测后果：三臂违规率全是 0，
# 消融完全测不出差异（详见 agent/eval/README.md「v1 失败的教训」）。
# 做对照实验时，领域知识**只能**从“工具回灌”进入模型。
SYSTEM_PROMPT_NEUTRAL = """你是结构健康监测(SHM)分析助手，可以调用工具查询本项目的数据与分析结果。

1. 先想清楚需要哪些信息，再调用相应工具；一步可以调多个。
2. 信息足够后给出结论；若信息不足，说明不足在哪里。
3. 用户问“怎么处理/要不要修”时，使用 maintenance 工具。
"""

TOOL_SCHEMAS = [
    {'type': 'function', 'function': {
        'name': 'list_specimens',
        'description': '列出全部试件、所属数据集与已知的数据问题（可用性问题）',
        'parameters': {'type': 'object', 'properties': {}}}},
    {'type': 'function', 'function': {
        'name': 'warning_status',
        'description': '查询数据集 A（主样本 016-020）的 D(t) 三级预警状态：预警时刻、'
                       '相对扩展标签的偏差、距断裂提前量、三级触发时刻',
        'parameters': {'type': 'object', 'properties': {
            'gid': {'type': 'string', 'description': '试件编号，如 016'}},
            'required': ['gid']}}},
    {'type': 'function', 'function': {
        'name': 'stiffness',
        'description': '查询刚度损失率轨迹与阈值时刻（L 达 5%/10%/30%/50% 的寿命位置）。'
                       '这是独立于自定义标签的物理参照，可用于交叉验证预警时刻',
        'parameters': {'type': 'object', 'properties': {
            'gid': {'type': 'string', 'description': '试件编号，如 016'}},
            'required': ['gid']}}},
    {'type': 'function', 'function': {
        'name': 'data_quality',
        'description': '查询数据集 A 的采集质量：采样率、时间连续性、AE 等效频率、'
                       '应变通道解调状态。用于判断结论是否可信',
        'parameters': {'type': 'object', 'properties': {
            'gid': {'type': 'string', 'description': '试件编号，如 016'}},
            'required': ['gid']}}},
    {'type': 'function', 'function': {
        'name': 'compare',
        'description': '多个试件横向对比预警时刻、提前量与刚度末值',
        'parameters': {'type': 'object', 'properties': {
            'gids': {'type': 'array', 'items': {'type': 'string'},
                     'description': '试件编号列表，如 ["016","017"]'}},
            'required': ['gids']}}},
    {'type': 'function', 'function': {
        'name': 'mechanism',
        'description': '查询数据集 C 的 RA–AF 损伤机制：剪切型占比随寿命的演化'
                       '（占比上升表示由基体开裂/纤维断裂主导转向分层/脱粘主导）',
        'parameters': {'type': 'object', 'properties': {
            'gid': {'type': 'string', 'description': '试件编号，如 L1-49'}},
            'required': ['gid']}}},
    {'type': 'function', 'function': {
        'name': 'localization',
        'description': '查询数据集 C 的 AE 事件定位：可信事件数、残差、定位质心。'
                       '注意 X 方向可信、Y 方向不可信',
        'parameters': {'type': 'object', 'properties': {
            'gid': {'type': 'string', 'description': '试件编号，如 L1-49'}},
            'required': ['gid']}}},
    {'type': 'function', 'function': {
        'name': 'anomaly',
        'description': '查询数据集 C 的无监督异常检测：首次偏离基线的寿命位置、'
                       '末段偏离程度（无需标签、组内自校准）',
        'parameters': {'type': 'object', 'properties': {
            'gid': {'type': 'string', 'description': '试件编号，如 L1-49'}},
            'required': ['gid']}}},
    {'type': 'function', 'function': {
        'name': 'doc_search',
        'description': '在项目文档中检索方法与原理说明（非语义检索，关键词匹配）。'
                       '用于回答"为什么这么做""原理是什么"类问题',
        'parameters': {'type': 'object', 'properties': {
            'query': {'type': 'string', 'description': '检索关键词'}},
            'required': ['query']}}},
    {'type': 'function', 'function': {
        'name': 'maintenance',
        'description': '生成检修建议：由已有证据确定性推导处置级别（立即处置/计划检修/'
                       '加强监测/常规监测）、检修范围、复检手段与不建议项。'
                       '回答"怎么处理""要不要修"类问题时必须调用。'
                       '每条建议自带 basis（推理依据），可直接引用',
        'parameters': {'type': 'object', 'properties': {
            'gid': {'type': 'string', 'description': '试件编号，如 016 / L1-49'}},
            'required': ['gid']}}},
]


class LLM:
    """DeepSeek / OpenAI 兼容的 chat 客户端（仅用标准库）。"""

    def __init__(self, api_key, model=DEFAULT_MODEL, base=API_BASE, timeout=90):
        self.api_key = api_key
        self.model = model
        self.base = base.rstrip('/')
        self.timeout = timeout
        self.usage = {'prompt': 0, 'completion': 0, 'calls': 0}

    def chat(self, messages, tools=None, max_tokens=3000, temperature=0.2,
             retries=4):
        """调用 chat/completions。

        ⚠️ deepseek-flash 是推理模型，reasoning token 会计入 completion_tokens；
        max_tokens 给小了会把配额吃光导致 content 为空（实测 900 即会复现）。

        **重试**：免费档模型普遍有严格的**并发/速率**限制（实测 GLM-4.7-Flash
        在 --workers 3 下整批 429）。这里对 429 与 5xx 做指数退避重试，
        其余 4xx（鉴权失败、参数错）**立即抛出** —— 重试也没用，只会白等。
        """
        payload = {'model': self.model, 'messages': messages,
                   'max_tokens': max_tokens, 'temperature': temperature}
        if tools:
            payload['tools'] = tools
            payload['tool_choice'] = 'auto'
        # 智谱 GLM 系列默认"先思考再回答"，reasoning token 计入 completion ——
        # 单条 ReAct 实测 4.8~87 s（波动大就是思考长度的表现）。本任务只需
        # "调工具 + 汇总事实"，不需要深度推理，故显式关闭，可提速数倍。
        # 只在 glm 模型上加这个字段：其它厂商不认，传了会报错。
        # 想临时打开做对照：在 .env 里设 LLM_THINKING=enabled
        if 'glm' in self.model.lower():
            payload['thinking'] = {
                'type': os.environ.get('LLM_THINKING', 'disabled')}
        body = json.dumps(payload).encode('utf-8')
        d = None
        for i in range(retries + 1):
            req = urllib.request.Request(
                self.base + '/chat/completions', data=body,
                headers={'Authorization': 'Bearer ' + self.api_key,
                         'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.loads(r.read().decode('utf-8'))
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500:          # 4xx 非限流 → 不可救
                    raise
                if i >= retries:
                    raise
                wait = 5.0 * (2 ** i)
                ra = e.headers.get('Retry-After') if e.headers else None
                if ra:
                    try:
                        wait = max(wait, float(ra))
                    except ValueError:
                        pass
                time.sleep(min(wait, 60.0))
            except (urllib.error.URLError, TimeoutError, OSError):
                if i >= retries:                            # 网络抖动 → 也退避重试
                    raise
                time.sleep(min(2.0 * (2 ** i), 30.0))
        u = d.get('usage') or {}
        self.usage['prompt'] += u.get('prompt_tokens', 0)
        self.usage['completion'] += u.get('completion_tokens', 0)
        self.usage['calls'] += 1
        return d['choices'][0]['message'], d['choices'][0].get('finish_reason')


# ── 文本形式工具调用的兜底解析 ────────────────────────────────────────
# 部分模型（实测 GLM-4.7-Flash）在给出最终回答的同时，会把工具调用写成
# **XML 风格的正文**，而不是 OpenAI 的结构化 tool_calls 字段：
#
#   …建议继续关注刚度趋势。<tool_call>maintenance
#   <arg_key>gid</arg_key><arg_value>016</arg_value></tool_call>
#
# 本项目只读 msg['tool_calls']，于是该调用被**默默丢弃**，而标签还留在
# 正文里 —— 不仅 maintenance 功能失效，fact/fab 这类基于文本的指标也会被污染。
_TC_RE = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.S)
_TC_ARG_RE = re.compile(
    r'<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>', re.S)


def _parse_text_tool_calls(text):
    """从正文里解析 XML 风格的工具调用，并返回 (calls, 剥离后的正文)。

    calls 为空表示正文里没有这种东西（正常情况）。
    参数值先试 JSON 解析（数组/数字/对象），失败则当字符串 ——
    注意 ``016`` 这类带前导零的编号 JSON 解析会失败，正好回退成字符串。
    """
    calls = []
    for i, m in enumerate(_TC_RE.finditer(text or '')):
        body = m.group(1)
        head = _TC_ARG_RE.split(body, 1)[0].strip()      # 工具名在参数之前
        nm = re.match(r'[A-Za-z_][\w]*', head)
        if not nm:
            continue
        args = {}
        for k, v in _TC_ARG_RE.findall(body):
            k, v = k.strip(), v.strip()
            try:
                args[k] = json.loads(v)
            except Exception:                            # noqa: BLE001
                args[k] = v
        calls.append({'id': 'text-call-%d' % i, 'type': 'function',
                      'function': {'name': nm.group(0),
                                   'arguments': json.dumps(args,
                                                           ensure_ascii=False)}})
    return calls, _TC_RE.sub('', text or '').strip()


def _tool_result_text(result, verdict, arm='full'):
    """把工具结果（+ 按实验臂可选地加上校验结论）压成回灌给模型的一段文本。

    实验臂（消融对照用，见 `agent/eval/README.md`）
    ----------------------------------------------
    ``full``     notes + 约束提示/违规 —— 完整系统
    ``no_guard`` 仅 notes —— 拿掉物理约束层，保留证据链
    ``bare``     两者都不要 —— 裸工具输出（朴素 RAG 基线）
    """
    parts = []
    if result.get('ok'):
        parts.append('结果: ' + json.dumps(result.get('data'), ensure_ascii=False))
    else:
        parts.append('失败: ' + str(result.get('error')))
    if arm != 'bare' and result.get('notes'):
        parts.append('已知限制: ' + '；'.join(result['notes']))
    if result.get('evidence'):
        parts.append('数据来源: ' + '；'.join(result['evidence']))
    if arm == 'full':
        if verdict and verdict.errors:
            parts.append('【物理约束违规·必须纠正】' + '；'.join(m for _, m in verdict.errors))
        if verdict and verdict.warnings:
            parts.append('【物理约束提示·须在回答中说明】' + '；'.join(m for _, m in verdict.warnings))
    return '\n'.join(parts)


class ReActAgent:
    """LLM 驱动的多步推理 Agent。

    ``arm`` 仅用于消融实验，生产使用默认的 ``'full'``。
    """

    def __init__(self, llm, max_steps=6, verbose=True, arm='full',
                 system_prompt=None):
        self.llm = llm
        self.max_steps = max_steps
        self.verbose = verbose
        self.arm = arm
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.pg = PhysGuard()
        self.trace = []

    def run(self, question):
        messages = [{'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': question}]
        self.trace = []
        t0 = time.time()

        for step in range(self.max_steps):
            msg, finish = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            calls = msg.get('tool_calls') or []

            # 兜底：模型可能把工具调用写成正文里的 XML 标签（见上方说明）
            if not calls:
                text_calls, clean = _parse_text_tool_calls(msg.get('content'))
                if text_calls:
                    calls = text_calls
                    msg = dict(msg, content=clean)   # 用剥离标签后的正文

            if not calls:                                   # 模型认为信息够了
                answer = (msg.get('content') or '').strip()
                if not answer:                              # 推理 token 吃满配额 → 重试
                    msg2, _ = self.llm.chat(
                        messages + [{'role': 'user',
                                     'content': '请直接给出结论（不要再调用工具）。'}],
                        tools=None, max_tokens=3000)
                    answer = (msg2.get('content') or '').strip()
                final_verdict = self.pg.check('final_answer',
                                              {'question': question},
                                              {'data': {'answer': answer}})
                return {'ok': True, 'answer': answer, 'trace': self.trace,
                        'steps': step, 'elapsed_s': round(time.time() - t0, 1),
                        'usage': dict(self.llm.usage),
                        'blocked': not final_verdict.ok,
                        'final_errors': [m for _, m in final_verdict.errors],
                        'final_warnings': [m for _, m in final_verdict.warnings],
                        'warnings': [m for _, m in final_verdict.warnings],
                        'physguard': self.pg.stats()}

            # 记录 assistant 的 tool_calls（必须回灌，否则模型丢失上下文）
            messages.append({'role': 'assistant', 'content': msg.get('content') or '',
                             'tool_calls': calls})

            for tc in calls:
                name = tc['function']['name']
                try:
                    args = json.loads(tc['function'].get('arguments') or '{}')
                except Exception:
                    args = {}
                res = tools.call(name, **args)
                verdict = self.pg.check(name, args, res)
                text = _tool_result_text(res, verdict, self.arm)
                messages.append({'role': 'tool', 'tool_call_id': tc['id'],
                                 'content': text})
                self.trace.append({'step': step, 'tool': name, 'args': args,
                                   'ok': res.get('ok'),
                                   'data': res.get('data'),
                                   'evidence': res.get('evidence'),
                                   'notes': res.get('notes'),
                                   'errors': [m for _, m in verdict.errors],
                                   'warnings': [m for _, m in verdict.warnings]})
                if self.verbose:
                    tag = 'OK ' if res.get('ok') else 'FAIL'
                    print('  [step %d] %s %s(%s)' % (step + 1, tag, name,
                                                     json.dumps(args, ensure_ascii=False)))

        # 超出步数：要求模型基于已有信息收尾
        messages.append({'role': 'user',
                         'content': '步数已达上限。请仅依据上面已获得的信息给出结论，'
                                    '并说明哪些方面因信息不足无法判断。'})
        msg, _ = self.llm.chat(messages, tools=None, max_tokens=3000)
        answer = (msg.get('content') or '').strip()
        if not answer:
            msg, _ = self.llm.chat(messages, tools=None, max_tokens=3000)
            answer = (msg.get('content') or '').strip()
        return {'ok': True, 'answer': answer,
                'trace': self.trace, 'steps': self.max_steps,
                'elapsed_s': round(time.time() - t0, 1),
                'usage': dict(self.llm.usage), 'blocked': False, 'warnings': [],
                'physguard': self.pg.stats(), 'note': '达到步数上限后收尾'}


def get_key(explicit=None):
    for k in (explicit, os.environ.get('DEEPSEEK_API_KEY'),
              os.environ.get('LLM_API_KEY'), os.environ.get('LLM_KEY')):
        if k:
            return k
    return None


def get_keys(explicit=None):
    """返回**全部**可用 key —— 供多账号轮询提速。

    平台速率限制是**账户维度**的（智谱错误码 1302 = 该账户在本模型上的并发已达上限），
    所以「N 个账号 = N 倍并发」。`LLM_API_KEYS` 用逗号/分号/空白分隔；
    未设置时退回单 key（行为与从前完全一致）。

    ⚠️ **必须是同一个模型**。混用不同模型或不同服务商虽然也能提并发，
    但会让消融实验的自变量失守 —— 三臂必须跑在同一模型上。
    """
    multi = os.environ.get('LLM_API_KEYS') or ''
    keys = [k.strip() for k in re.split(r'[,;\s]+', multi) if k.strip()]
    if keys:
        return keys
    single = explicit or get_key()
    return [single] if single else []


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace',
                               line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('question', nargs='*')
    ap.add_argument('--api-key', default=None)
    ap.add_argument('--env-file', default=None,
                    help='指定 .env 路径（默认自动查找 <项目根>/.env 与 agent/.env）')
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--max-steps', type=int, default=6)
    ap.add_argument('--trace', action='store_true', help='打印完整推理轨迹')
    ap.add_argument('--out', default=None, help='把结果写入指定文件（便于查看长回答）')
    ap.add_argument('--arm', default='full', choices=['full', 'no_guard', 'bare'],
                    help='消融实验臂（默认 full；见 agent/eval/README.md）')
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()

    lines = []

    def emit(s=''):
        lines.append(s)
        if not a.out:
            print(s)

    if a.env_file:                       # 显式指定的 .env 优先于自动发现的
        load_env([a.env_file], override=True)

    key = get_key(a.api_key)
    if not key:
        print('未找到 API key。三种设法（选一）：\n'
              '  1) 放在 .env 文件里（推荐）：DEEPSEEK_API_KEY=sk-xxx\n'
              "  2) 环境变量：$env:DEEPSEEK_API_KEY='sk-xxx'   (PowerShell)\n"
              '  3) 临时传参：python agent/react.py --api-key sk-xxx "问题"')
        print(describe([a.env_file] if a.env_file else None))
        return 2
    agent = ReActAgent(LLM(key, a.model), max_steps=a.max_steps, arm=a.arm)

    qs = a.question and [' '.join(a.question)] or (
        ['016 现在什么状态？这个预警结果可信吗？'] if a.demo else None)
    if not qs:
        print('用法: python agent/react.py "你的问题"  或  --demo')
        return 1

    for q in qs:
        emit('=' * 68)
        emit('Q: ' + q)
        r = agent.run(q)
        emit('-' * 68)
        emit(r['answer'])
        emit('-' * 68)
        emit('[%d 轮 / %s s / tokens in=%d out=%d]  规则检查=%d 拦截=%d'
             % (r['steps'], r['elapsed_s'], r['usage']['prompt'],
                r['usage']['completion'], r['physguard']['n_rule_checks'],
                r['physguard']['n_blocked']))
        if r.get('warnings'):
            emit('[!] ' + '；'.join(r['warnings']))
        if a.trace:
            emit('\n--- 推理轨迹 ---')
            for t in r['trace']:
                emit('  %s(%s) -> %s' % (t['tool'], json.dumps(t['args'], ensure_ascii=False),
                                         'OK' if t['ok'] else 'FAIL'))
                if t['warnings']:
                    emit('        [!] ' + '；'.join(t['warnings']))
                if t['errors']:
                    emit('        [X] ' + '；'.join(t['errors']))

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print('已写入', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
