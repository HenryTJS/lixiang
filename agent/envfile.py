# -*- coding: utf-8 -*-
"""极简 .env 加载器（零依赖）。

为什么自己写而不用 python-dotenv
--------------------------------
本项目约定「自研优先、减少外部依赖」（见 `memory` / README），而 .env 的
常用语法子集（KEY=VALUE / `#` 注释 / 引号 / `export` 前缀）实现不到 40 行。

优先级（从高到低）
------------------
1. 命令行参数  ``--api-key sk-xxx``
2. 进程环境变量 ``$env:DEEPSEEK_API_KEY``
3. ``.env`` 文件

即 **已存在的环境变量不会被 .env 覆盖**（与 python-dotenv 默认行为一致），
这样 CI、临时覆盖、多套 key 切换仍然有效。

查找顺序
--------
``<项目根>/.env`` → ``<agent>/.env``（先找到的先用）

安全约定
--------
- 只读不写；**不打印值**，只在报错时提示"查过哪些路径"
- ``.env`` 已列入 `.gitignore`；仓库里只提交 ``.env.example`` 模板

用法::

    from envfile import load_env
    path = load_env()            # None 表示没找到
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))

#: 查找顺序（先命中先用）
CANDIDATES = (
    os.path.join(ROOT, '.env'),
    os.path.join(HERE, '.env'),
)

#: 本文件至少应当加载成功的键（供报错时提示）
KNOWN_KEYS = ('DEEPSEEK_API_KEY', 'LLM_API_KEY', 'LLM_KEY',
              'LLM_API_BASE', 'LLM_MODEL')


def parse(text):
    """解析 .env 文本 → dict。

    支持：

    ====================  ==========================================
    ``KEY=value``         基本形式
    ``KEY = value``       键值两侧空白会被去掉
    ``export KEY=value``  忽略 export 前缀（方便直接 source 同一文件）
    ``KEY="a b"``         成对引号会被剥掉（单/双引号均可）
    ``KEY=v  # 注释``     无引号时，空格后的 ``#`` 起为行内注释
    ``# 整行注释``        跳过
    ====================  ==========================================

    非法行（没有 ``=``）会被忽略，不抛异常 —— 密钥文件格式错误不应让程序崩掉。
    """
    out = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line.lower().startswith('export '):
            line = line[7:].lstrip()
        if '=' not in line:
            continue
        key, val = line.split('=', 1)
        key, val = key.strip(), val.strip()
        if not key:
            continue
        if len(val) >= 2 and val[0] == val[-1] and val[0] in '"\'':
            val = val[1:-1]
        else:
            for i, ch in enumerate(val):
                if ch == '#' and i > 0 and val[i - 1] in ' \t':
                    val = val[:i].rstrip()
                    break
        out[key] = val
    return out


def load_env(paths=None, override=False):
    """把 .env 读进 ``os.environ``，返回实际读取的文件路径（未找到返回 None）。

    ``override=True`` 时 .env 会覆盖已有环境变量；默认不覆盖。
    """
    for p in (paths or CANDIDATES):
        if not p or not os.path.isfile(p):
            continue
        try:
            with open(p, encoding='utf-8-sig') as f:
                kv = parse(f.read())
        except OSError:
            continue
        for k, v in kv.items():
            if override or k not in os.environ:
                os.environ[k] = v
        return p
    return None


def describe(paths=None):
    """返回给用户看的诊断信息（**不含任何密钥值**）。"""
    tried = list(paths or CANDIDATES)
    found = [p for p in tried if os.path.isfile(p)]
    lines = ['查过：'] + ['  - %s%s' % (p, '  ← 已加载' if p == found[:1] and found else '')
                          for p in tried]
    if not found:
        lines.append('  未找到任何 .env 文件')
    have = [k for k in KNOWN_KEYS if os.environ.get(k)]
    lines.append('当前环境变量中已设置的键：%s' % (', '.join(have) if have else '无'))
    return '\n'.join(lines)
