# -*- coding: utf-8 -*-
"""消融对照实验：物理约束层到底有没有用？

三臂设计
--------
| 臂 | 工具回灌内容 | 含义 |
| --- | --- | --- |
| `full` | notes + 约束提示/违规 | 完整系统 |
| `no_guard` | 仅 notes | 拿掉物理约束层，保留证据链 |
| `bare` | 两者都不给 | 裸工具输出（朴素 RAG 基线）|

系统提示、工具集、模型、温度**三臂完全相同** —— 唯一变量是"模型能看到什么"。

指标（全部可自动计算、无需人工评分）
------------------------------------
1. **fact**：数值期望命中率。答案中是否出现了足够接近的正确数值（tol 由题集给定）。
2. **disclaim**：限制声明合规率。陷阱题要求答案明确声明"不可/不支持/超出范围"等。
3. **tool**：工具使用合规率。必须调用的工具是否都调了。
4. **guard_err / guard_warn**：**违规率**（核心指标）。
   用完整 PhysGuard 对**最终回答**独立评分 —— 三臂用同一把尺子。
5. **fab**：**编造数值率**。回答里出现了证据（工具 data/notes/题面/期望值）
   中都没有的数值 → 疑似幻觉。是**唯一不与 PhysGuard 同源**的可靠性指标。
   > ⚠️ **局限（必须写进论文）**：该指标仍会捕获**派生量** —— 模型自己
   > 算出的差值/比值（如「刚度 3.758 → 2.574，损失约 31.5%」）。这类数字
   > 合理但不在证据集合里，**无法与真幻觉自动区分**：若允许「任意两数之差」
   > 则候选数达 C(n,2)，指标会直接退化为恒 0。
   > 因此**必须人工抽样核对**（`show_runs.py` 会给每个值附上下文），
   > 并优先使用 **`fab率`**（未取证数值占比）而非 `fab任一`（≥1 个即计 1，
   > 长回答下必然偏高，易被误读为「模型大部分回答都在编造」）。
6. 成本：耗时、token。

> ⚠️ 方法学声明（必须写进论文局限）：指标 4 用的是与实验组同一套规则，
> 存在**同源性偏置**：规则既是"干预手段"又是"评价标准"。因此
> 结论不宣称"PhysGuard 让回答更正确"，只宣称
> "**在给定这组物理约束下，把约束显式提供给模型能降低违反这些约束的比例**"。
> 指标 1/2/3 与之独立，可作为交叉印证。

用法::

    python agent/eval/run_ablation.py --limit 6          # 小样试跑
    python agent/eval/run_ablation.py --arms full bare   # 只跑两臂
    python agent/eval/run_ablation.py --workers 4        # 全量（可续跑）
"""
import argparse
import collections
import io
import itertools
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# line_buffering：不加的话终端里是块缓冲（4~8 KB），
# 而十几条进度总共不到 1 KB —— 长跑时看起来就像卡死了。
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8',
                              errors='replace', line_buffering=True)
HERE = os.path.dirname(os.path.abspath(__file__))
# Python 会把脚本所在目录放在 sys.path[0]。本目录里的 .py 若与标准库同名，
# 会**遮蔽标准库** —— 曾有一个 inspect.py，导致 numpy 2.x 内部 `import inspect`
# 被劫持、并在其顶层 open(sys.argv[1]) 处崩溃。这里主动移除本目录。
while HERE in sys.path:
    sys.path.remove(HERE)
AGENT = os.path.dirname(HERE)
sys.path.insert(0, AGENT)

import react                                                      # noqa: E402

ARMS = ['full', 'no_guard', 'bare']
OUTDIR = os.path.join(HERE, 'results')
RUNS = os.path.join(OUTDIR, 'runs.jsonl')
SUMMARY = os.path.join(OUTDIR, 'summary.csv')
_LOCK = threading.Lock()
# 试件编号 / 数据集代号（016、L1-49、L1–49 …）是标识符、不是测量值：
# 提取前先剔除 —— 注意模型会用 **en dash**（U+2013）写「L1–49」。
_ID_PAT = re.compile(r'0\d{2}|L1[-\u2013\u2014]\d+')
# Markdown 列表序号（"1. "、"**2. **"、"- 3)"）：排版标记，不是测量值。
_LIST_PAT = re.compile(r'(?m)^[\s*]{0,4}\d+[.)]\s')
# 数值：支持千分位逗号 —— 「31,442」是一个数，不能切成 31 和 442。
_NUM_RE = re.compile(r'-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?')


def _clean(text):
    """剔除标识符与列表序号后再提取数值。"""
    return _LIST_PAT.sub(' ', _ID_PAT.sub(' ', text or ''))


def _num_value(s):
    return float(s.replace(',', ''))


# 项目公开判据常量：D 的三级分数（README §2.2）+ 刚度损失阈值档位
# （`stiffness` 工具的 thresholds 键）。它们是**规范值**而非测量值，
# 回答里随时用到（例如提议“下一步可查 L 达 5%/10%/30%/50% 的位置”），
# 不应被算成“编造的数值”。
KNOWN_CONSTANTS = {0.05, 0.10, 0.25, 0.30, 0.50, 0.55, 0.85}


def numbers(text):
    out = []
    for m in _NUM_RE.finditer(_clean(text)):
        try:
            out.append(_num_value(m.group()))
        except ValueError:
            pass
    return out


def evidence_numbers(case, trace):
    """回答里允许出现的数值全集 = 工具数据 ∪ 工具 notes ∪ 题面。

    用它算**编造率**：回答了证据里根本没有的数字 → 幻觉。
    这是本实验**唯一不依赖 PhysGuard** 的核心指标，用于避开“自评自”的循环论证。
    """
    vals = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                # 键里也可能带数值：{"L>=0.10": 59} 的阈值就是键
                walk(k)
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif isinstance(o, bool):
            return
        elif isinstance(o, (int, float)):
            vals.add(float(o))
        elif isinstance(o, str):
            for m in _NUM_RE.finditer(_clean(o)):
                try:
                    vals.add(_num_value(m.group()))
                except ValueError:
                    pass

    walk(case['question'])
    walk(case.get('expect_num'))
    walk(case.get('expect_any'))
    for t in (trace or []):
        walk(t.get('data'))
        walk(t.get('notes'))
    # 规范常量无条件允许（见 KNOWN_CONSTANTS 说明）
    vals |= KNOWN_CONSTANTS
    return vals


def covered(x, vals, rel=0.02, floor=0.05):
    """x 是否被证据里的某个数值覆盖（容差 rel 且至少 floor）。

    除直接匹配外，还接受两种**同一量的不同写法**，否则结构性误判会淹没指标：
      · 百分数 ↔ 小数：证据 0.10，回答写「L ≥ 10%」；
      · 符号：证据 -1（公式常数 A/A_cal−1），回答写「− 1」。
    """
    for v in vals:
        for cand in (v, v * 100.0, abs(v)):
            if abs(x - cand) <= max(rel * abs(cand), floor):
                return True
    return False


def score(case, ans, trace):
    """对一条回答打分（全部规则化、可复现）。"""
    txt = ans or ''
    nums = numbers(txt)
    tools_called = set(t['tool'] for t in (trace or []) if t.get('ok'))

    # ① 数值命中
    hits = 0
    for e in case['expect_num']:
        hits += int(any(abs(x - e['v']) <= e['tol'] for x in nums))
    need_n = len(case['expect_num']) + len(case['expect_any'])
    got = hits
    for grp in case['expect_any']:
        got += int(any(k in txt for k in grp))
    fact = (got / need_n) if need_n else None

    # ② 限制声明
    disclaim = None
    if case['disclaim']:
        disclaim = int(any(k in txt for k in case['disclaim']))

    # ③ 工具使用
    tool_ok = None
    if case['must_call']:
        tool_ok = int(all(t in tools_called for t in case['must_call']))

    # ④ 编造率（独立于 PhysGuard）
    allowed = evidence_numbers(case, trace)
    fab = [x for x in nums if not covered(x, allowed)]

    return {'fact': fact, 'disclaim': disclaim, 'tool': tool_ok,
            'n_tools': len(tools_called), 'answer_chars': len(txt),
            'n_nums': len(nums), 'n_fab': len(fab),
            'fab_rate': (len(fab) / len(nums)) if nums else None,
            'fab_any': int(bool(fab)), 'fab_examples': [round(x, 3) for x in fab[:8]]}


def run_one(llm_factory, case, arm, max_steps, system_prompt):
    llm = llm_factory()
    agent = react.ReActAgent(llm, max_steps=max_steps, verbose=False, arm=arm,
                             system_prompt=system_prompt)
    agent.pg.enable_log = False          # 实验不污染生产日志
    t0 = time.time()
    try:
        r = agent.run(case['question'])
    except Exception as e:                                   # 网络/解析异常不应中断全跑
        return {'error': '%s: %s' % (type(e).__name__, e),
                'elapsed_s': round(time.time() - t0, 1)}
    rec = {'answer': r['answer'], 'elapsed_s': r['elapsed_s'],
           'steps': r['steps'], 'usage': r['usage'],
           'guard_errors': [m for m in r.get('final_errors', [])],
           'guard_warns': [m for m in r.get('final_warnings', [])],
           'trace': [{'tool': t['tool'], 'args': t['args'], 'ok': t['ok'],
                      'data': t.get('data'), 'notes': t.get('notes')}
                     for t in r['trace']]}
    rec.update(score(case, r['answer'], r['trace']))
    return rec


def already_done():
    """已完成 = 有**有效**记录的 (case_id, arm)。

    失败记录（``error``）**不算完成**、会被重跑 —— 否则一次 429 限流或网络抖动
    就把该题永久标成跳过，只能靠 ``--fresh`` 全量重来（而免费档模型限流是常态）。
    """
    done = set()
    if os.path.exists(RUNS):
        with open(RUNS, encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:                                # noqa: BLE001
                    continue
                if 'error' in d:
                    continue
                done.add((d['case_id'], d['arm']))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', default=os.path.join(HERE, 'cases.json'))
    ap.add_argument('--arms', nargs='*', default=ARMS, choices=ARMS)
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 题（试跑用）')
    ap.add_argument('--category', default=None, help='只跑某一类')
    ap.add_argument('--trap-only', action='store_true')
    ap.add_argument('--unique-trap', action='store_true',
                    help='陷阱题每类越界模式只取第一个（去掉 T1b / T2-L1-50 / '
                         'T17-018 这类变体）—— 减量但不损失覆盖面')
    ap.add_argument('--workers', type=int, default=3)
    ap.add_argument('--max-steps', type=int, default=6)
    ap.add_argument('--model', default=react.DEFAULT_MODEL)
    ap.add_argument('--prompt', default='neutral', choices=['neutral', 'project'],
                    help='系统提示：neutral 不含领域约束（消融必须用）、'
                         'project 为生产版（会泄漏约束，仅供对比）')
    ap.add_argument('--prefix', default='v2', help='输出文件前缀，便于版本对比')
    ap.add_argument('--fresh', action='store_true', help='忽略已有结果重跑')
    a = ap.parse_args()

    global RUNS, SUMMARY
    RUNS = os.path.join(OUTDIR, '%s_runs.jsonl' % a.prefix)
    SUMMARY = os.path.join(OUTDIR, '%s_summary.csv' % a.prefix)

    keys = react.get_keys()
    if not keys:
        print('未找到 API key。把 LLM_API_KEY（或 DEEPSEEK_API_KEY）填进 .env，'
              '或设环境变量，或用 --api-key 传入（详见 agent/README.md）')
        return 2

    cases = json.load(open(a.cases, encoding='utf-8'))
    if a.category:
        cases = [c for c in cases if c['category'] == a.category]
    if a.trap_only:
        cases = [c for c in cases if c['trap']]
    if a.unique_trap:
        # 题号形如 T1 / T1b / T2-L1-50 / T17-018 / N3 —— 前缀（T1、T17、N3）
        # 即「越界模式类别」，带后缀的都是同类变体。每类只留第一个，
        # 既减量又不损失覆盖面（--limit 做不到：编号不连续，前 22 个只覆盖到 T12）。
        _seen, _keep = set(), []
        for _c in cases:
            _m = re.match(r'([TN]\d+)', _c['id'])
            _k = _m.group(1) if _m else _c['id']
            if _k in _seen:
                continue
            _seen.add(_k)
            _keep.append(_c)
        cases = _keep
        print('去重后 %d 题（每类越界模式保留一个）' % len(cases))
    if a.limit:
        cases = cases[:a.limit]

    done = set() if a.fresh else already_done()
    todo = [(c, arm) for c in cases for arm in a.arms
            if (c['id'], arm) not in done]

    os.makedirs(OUTDIR, exist_ok=True)
    sp = (react.SYSTEM_PROMPT_NEUTRAL if a.prompt == 'neutral'
          else react.SYSTEM_PROMPT)
    print('题集 %d 题 × %d 臂 = %d 次；已完成 %d 次，待跑 %d 次（提示=%s，前缀=%s）'
          % (len(cases), len(a.arms), len(cases) * len(a.arms), len(done),
             len(todo), a.prompt, a.prefix))
    if not todo:
        print('无待跑项 → 直接汇总')
        return summarize(cases, a.arms)

    # 速率限制是**账户维度**的（智谱 1302 = 该账户在本模型上并发已达上限），
    # 所以「N 个账号 = N 倍并发」：多个 key 按轮询分配给各个任务。
    # 并发数**不该超过 key 数** —— 超出只会换来 429 重试，反而更慢。
    if a.workers > len(keys):
        print('⚠️ workers=%d 超过可用 key 数 %d → 自动降到 %d'
              % (a.workers, len(keys), len(keys)))
        a.workers = len(keys)
    _key_it = itertools.cycle(keys)
    _key_lock = threading.Lock()

    def make_llm():
        with _key_lock:
            return react.LLM(next(_key_it), a.model)

    n_ok = n_err = 0
    t_start = time.time()
    with open(RUNS, 'a', encoding='utf-8') as fout:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(run_one, make_llm, c, arm, a.max_steps, sp): (c, arm)
                    for c, arm in todo}
            for i, fut in enumerate(futs, 1):
                c, arm = futs[fut]
                rec = fut.result()
                rec.update({'case_id': c['id'], 'arm': arm, 'category': c['category'],
                            'trap': c['trap'], 'gid': c['gid'],
                            'question': c['question'], 'ts': time.strftime('%F %T')})
                with _LOCK:
                    fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
                    fout.flush()
                if 'error' in rec:
                    n_err += 1
                    print('  [%d/%d] %s/%-8s ERROR %s'
                          % (i, len(todo), c['id'], arm, rec['error']))
                else:
                    n_ok += 1
                    print('  [%d/%d] %s/%-8s %4.1fs guard_err=%d warn=%d'
                          % (i, len(todo), c['id'], arm, rec['elapsed_s'],
                             len(rec['guard_errors']), len(rec['guard_warns'])))

    print('\n完成 %d 条，失败 %d 条，用时 %.1f min'
          % (n_ok, n_err, (time.time() - t_start) / 60))
    return summarize(cases, a.arms)


def summarize(cases, arms):
    rows = collections.defaultdict(list)
    cost = collections.defaultdict(lambda: {'t': 0.0, 'pin': 0, 'pout': 0, 'n': 0})
    with open(RUNS, encoding='utf-8') as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d['arm'] not in arms or 'error' in d:
                continue
            rows[(d['arm'], d['category'])].append(d)
            rows[(d['arm'], 'ALL')].append(d)
            c = cost[d['arm']]
            c['t'] += d['elapsed_s']
            c['pin'] += d['usage']['prompt']
            c['pout'] += d['usage']['completion']
            c['n'] += 1

    def agg(rs):
        def m(fn, only=None):
            v = [fn(r) for r in rs if (only is None or only(r))]
            v = [x for x in v if x is not None]
            return (sum(v) / len(v)) if v else None
        traps = [r for r in rs if r['trap']]
        return {
            'n': len(rs),
            'fact': m(lambda r: r.get('fact')),
            'disclaim': m(lambda r: r.get('disclaim'), lambda r: r.get('disclaim') is not None),
            'tool': m(lambda r: r.get('tool'), lambda r: r.get('tool') is not None),
            'err_rate': m(lambda r: len(r['guard_errors']) > 0),
            'warn_rate': m(lambda r: len(r['guard_warns']) > 0),
            'n_err': sum(len(r['guard_errors']) for r in rs),
            'fab_rate': m(lambda r: r.get('fab_rate')),
            'fab_any': m(lambda r: r.get('fab_any')),
            'n_fab': sum(r.get('n_fab', 0) for r in rs),
            'trap_n': len(traps),
            'trap_err_rate': (sum(1 for r in traps if r['guard_errors']) / len(traps)
                              if traps else None),
        }

    def pct(x):
        return '' if x is None else '%.3f' % x

    lines = ['arm,category,n,fact,disclaim,tool,guard_err_rate,guard_warn_rate,'
             'n_errors,fab_rate,fab_any_rate,n_fab,trap_n,trap_err_rate,avg_s,in_tok,out_tok']
    print('\n%-9s %-9s %4s %6s %6s %6s %7s %7s %5s %8s %8s %5s' %
          ('arm', 'category', 'n', 'fact', 'discl', 'tool', 'err率', 'warn率',
           'err数', 'fab率', 'fab任一', 'fab数'))
    for arm in arms:
        for cat in ['ALL'] + sorted(set(k[1] for k in rows if k[0] == arm and k[1] != 'ALL')):
            rs = rows.get((arm, cat))
            if not rs:
                continue
            s = agg(rs)
            c = cost[arm]
            print('%-9s %-9s %4d %6s %6s %6s %7s %7s %5d %8s %8s %5d' %
                  (arm, cat, s['n'], pct(s['fact']), pct(s['disclaim']), pct(s['tool']),
                   pct(s['err_rate']), pct(s['warn_rate']), s['n_err'],
                   pct(s['fab_rate']), pct(s['fab_any']), s['n_fab']))
            lines.append('%s,%s,%d,%s,%s,%s,%s,%s,%d,%s,%s,%d,%d,%s,%.1f,%d,%d' % (
                arm, cat, s['n'], pct(s['fact']), pct(s['disclaim']), pct(s['tool']),
                pct(s['err_rate']), pct(s['warn_rate']), s['n_err'],
                pct(s['fab_rate']), pct(s['fab_any']), s['n_fab'], s['trap_n'],
                pct(s['trap_err_rate']),
                c['t'] / max(c['n'], 1), c['pin'], c['pout']))
        print()
    with open(SUMMARY, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(lines) + '\n')
    print('汇总 → %s' % SUMMARY)
    return 0


if __name__ == '__main__':
    sys.exit(main())
