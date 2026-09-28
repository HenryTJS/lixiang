# -*- coding: utf-8 -*-
"""检查实验原始记录：抽样核对指标是否可信。

**用当前打分逻辑重新计算**，而不是读 jsonl 里的旧值 —— 指标口径会演进
（例如 2026-09-16 修了 fab 的四个提取 bug：dict 键丢失、百分数↔小数、
千分位逗号、试件编号），旧记录里的 n_fab 会过期，直接展示会误导核对。

每个被判"编造"的数值都会附上**它在回答中的上下文** —— 核对时不必通读全文，
一眼即可判断是真幻觉还是派生量（例如「刚度 3.758 → 2.574」推出的 31.5%）。

用法::
    python agent/eval/show_runs.py                 # 默认 v2_runs.jsonl
    python agent/eval/show_runs.py smoke_runs.jsonl
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import run_ablation as ra                                       # noqa: E402

# 注意：run_ablation 导入时已包装 sys.stdout，这里不要再包一次
# （重复包装同一 buffer，前一个 wrapper 被 GC 会关掉底层流）。

RUNS = os.path.join(HERE, 'results',
                    sys.argv[1] if len(sys.argv) > 1 else 'v2_runs.jsonl')
CASES = os.path.join(HERE, 'cases.json')

cases = {}
if os.path.exists(CASES):
    cases = {c['id']: c for c in json.load(open(CASES, encoding='utf-8'))}

rows = []
with open(RUNS, encoding='utf-8') as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except Exception:                                        # noqa: BLE001
            pass

print('共 %d 条   (%s)' % (len(rows), os.path.basename(RUNS)))


def rescore(r):
    """用当前逻辑重新打分；缺题集或回答时回退 None。"""
    c = cases.get(r.get('case_id'))
    if c is None or 'answer' not in r:
        return None
    return ra.score(c, r['answer'], r.get('trace'))


def ctx(txt, x, w=48):
    """定位数值 x 在回答中的上下文。"""
    clean = ra._ID_PAT.sub(' ', txt or '')
    for m in ra._NUM_RE.finditer(clean):
        try:
            if abs(ra._num_value(m.group()) - x) < 1e-9:
                s, e = max(0, m.start() - w), min(len(clean), m.end() + w)
                return clean[s:e].replace('\n', ' ')
        except ValueError:
            pass
    return '(未定位)'


print('\n===== 编造数值抽样（前 12 条，按当前逻辑重算）=====')
shown = 0
for r in rows:
    sc = rescore(r)
    if sc and sc['n_fab'] and shown < 12:
        print('\n[%s / %s] n_nums=%d  n_fab=%d'
              % (r['case_id'], r['arm'], sc['n_nums'], sc['n_fab']))
        print('  Q: %s' % str(r.get('question', ''))[:70])
        for x in sc['fab_examples']:
            print('  · %-9s … %s' % (round(x, 3), ctx(r.get('answer'), x)))
        shown += 1

print('\n===== 违规/告警清单 =====')
for r in rows:
    if r.get('guard_errors') or r.get('guard_warns'):
        print('[%s / %-8s] err=%s warn=%s'
              % (r['case_id'], r['arm'],
                 [m[:60] for m in r.get('guard_errors', [])],
                 [m[:60] for m in r.get('guard_warns', [])]))

print('\n===== 按题号看三臂对照（按当前逻辑重算）=====')
by = collections.defaultdict(dict)
for r in rows:
    by[r['case_id']][r['arm']] = r
for cid, arms in sorted(by.items()):
    if 'full' not in arms:
        continue
    line = ['%-12s' % cid]
    for a in ra.ARMS:
        r = arms.get(a)
        if not r:
            line.append('%-18s' % '--')
            continue
        if 'error' in r:
            line.append('%-18s' % 'API 异常')
            continue
        sc = rescore(r)
        line.append('%-18s' % ('?' if sc is None else
                               'fab=%d/%d w=%d' % (sc['n_fab'], sc['n_nums'],
                                                   len(r.get('guard_warns', [])))))
    print('  '.join(line))
