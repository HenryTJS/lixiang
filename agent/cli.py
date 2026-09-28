# -*- coding: utf-8 -*-
"""Agent 命令行入口。

用法
----
    python agent/cli.py                          # 交互模式
    python agent/cli.py "016 什么时候预警的"      # 单问
    python agent/cli.py --demo                   # 跑一组内置示例
    python agent/cli.py --stats                  # 打印物理规则触发统计
    python agent/cli.py --llm qwen2.5:3b         # 接入本地 ollama 模型（可选）

说明
----
语言层默认为**模板渲染**（零依赖）。加 `--llm` 后走本地模型；
若 ollama 不可用会自动回退模板，不影响功能。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Windows 终端默认 GBK，emoji/特殊符号会报 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from core import Agent                            # noqa: E402

DEMO = [
    '有哪些数据',
    '016 什么时候预警的',
    '016 的刚度退化到什么程度了',
    '022 的数据质量怎么样',
    '对比 016 和 017',
    'L1-49 的损伤机制是什么',
    'L1-59 的异常检测结果如何',
    'L1-49 的定位精度怎么样',
    '为什么不做 RUL',
]


def make_llm(spec):
    """按 `name[:model]` 生成语言层函数；不可用则返回 None。"""
    if not spec:
        return None
    try:
        if spec.startswith('ollama'):
            import urllib.request
            model = spec.split(':', 1)[1] if ':' in spec else 'qwen2.5:3b'

            def fn(prompt):
                body = json.dumps({'model': model, 'prompt': prompt,
                                   'stream': False}).encode('utf-8')
                req = urllib.request.Request(
                    'http://localhost:11434/api/generate', data=body,
                    headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return json.loads(r.read().decode('utf-8')).get('response', '').strip()

            fn('ping')                                  # 连通性测试
            print('[LLM] 已接入 ollama:%s' % model)
            return fn
        if spec.startswith('http'):
            import urllib.request
            url = spec

            def fn(prompt):
                body = json.dumps({'prompt': prompt}).encode('utf-8')
                req = urllib.request.Request(url, data=body,
                                             headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=120) as r:
                    d = json.loads(r.read().decode('utf-8'))
                    return d.get('response') or d.get('text') or str(d)

            fn('ping')
            print('[LLM] 已接入 HTTP 端点 %s' % url)
            return fn
    except Exception as e:
        print('[LLM] 接入失败（%s）→ 回退模板渲染' % e)
    return None


def show(r, verbose=True):
    print('Q:', r['question'])
    print('A:', r['answer'])
    if verbose:
        if r['blocked']:
            print('   [X] 物理校验拦截:', '；'.join(r['errors']))
        if r['warnings']:
            print('   [!] 约束提示:', '；'.join(r['warnings']))
        if r['notes']:
            print('   [i] 已知限制:', '；'.join(r['notes'][:2]))
        if r['evidence']:
            print('   [e] 证据链:', '；'.join(r['evidence']))
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('question', nargs='*', help='直接提问（省略则进入交互模式）')
    ap.add_argument('--demo', action='store_true', help='跑内置示例')
    ap.add_argument('--stats', action='store_true', help='打印统计')
    ap.add_argument('--llm', default=None, help='语言层模型，如 qwen2.5:3b 或 http://...')
    ap.add_argument('--quiet', action='store_true', help='只输出答案')
    a = ap.parse_args()

    ag = Agent(llm=make_llm(a.llm), verbose=not a.quiet)

    if a.demo:
        for q in DEMO:
            show(ag.ask(q), verbose=not a.quiet)
    elif a.question:
        show(ag.ask(' '.join(a.question)), verbose=not a.quiet)
    elif a.stats:
        print(json.dumps(ag.stats(), ensure_ascii=False, indent=2))
        return
    else:
        print('SHM 分析 Agent（输入 q 退出，输入 h 看示例）')
        while True:
            try:
                q = input('>>> ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ('q', 'quit', 'exit'):
                break
            if q.lower() in ('h', 'help'):
                print('示例问题：')
                for x in DEMO:
                    print('   ', x)
                continue
            if not q:
                continue
            show(ag.ask(q), verbose=not a.quiet)

    if a.stats or a.demo:
        print('物理规则统计:', json.dumps(ag.stats(), ensure_ascii=False))


if __name__ == '__main__':
    main()
