#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据路径解析 —— 读仓库根的 `paths.json`（全项目唯一的路径入口）。

背景（2026-09-28）：数据集已搬到外接盘，仓库里用 **目录联接（junction）** 指过去，
所以 `main/`、`l1/`、`phmdc/` 下的脚本仍按相对路径工作，**无需改动**。
本模块供"需要知道数据根在哪"的新代码 / 诊断脚本使用，避免各自拼绝对路径。

优先级：环境变量 ``SHM_DATA_ROOT``  >  ``paths.json`` 的 ``data_root``

用法::

    from shm import paths
    paths.data_root()                      # 外接盘数据根，如 'E:\\shm_data'
    paths.items()                          # paths.json 的 items
    paths.link_paths(paths.items()[0])     # 展开通配后，仓库内的目录列表
    paths.status()                         # 逐条状态（仓库侧 / 数据侧）

命令行自检::

    python shm/paths.py
    python shm/paths.py --json
"""
from __future__ import annotations

import glob
import json
import os
import stat as _stat
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(PROJ, 'paths.json')
ENV_KEY = 'SHM_DATA_ROOT'
_REPARSE = getattr(_stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


def load(path=None):
    """读取 paths.json；文件缺失时返回最小结构（不抛异常）。"""
    fp = path or CONFIG
    if not os.path.exists(fp):
        return {'data_root': '', 'items': []}
    with open(fp, encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.setdefault('items', [])
    return cfg


def data_root(path=None):
    """数据根目录。裸盘符会补成 ``E:\\``（否则 join 会得到盘符相对路径 ``E:l1``）。"""
    env = os.environ.get(ENV_KEY)
    root = os.path.abspath(env) if env else (load(path).get('data_root') or '')
    root = root.rstrip('\\/')
    if len(root) == 2 and root[1] == ':':
        root += os.sep
    return root


def items(path=None):
    """paths.json 的 items 列表。"""
    return load(path).get('items', [])


def link_paths(item):
    """把 item['link'] 展开成仓库内的绝对路径列表（支持 * ? [0-9]，只回目录）。"""
    pat = item.get('link') or ''
    if not pat:
        return []
    full = os.path.join(PROJ, pat.replace('/', os.sep))
    if any(ch in pat for ch in '*?['):
        return sorted(p for p in glob.glob(full) if os.path.isdir(p))
    return [full]


def is_link(path):
    """是否为目录联接 / 符号链接（Windows 下看 reparse point 属性）。"""
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return bool(getattr(st, 'st_file_attributes', 0) & _REPARSE)


def link_target(path):
    """读取联接指向的目标；不是联接则返回空串。"""
    try:
        return os.readlink(path)
    except OSError:
        return ''


def target_of(item, link_dir):
    """算出某个仓库目录对应的外接盘目标路径。

    规则：link 含通配符 → target 视为【容器目录】，目标 = data_root/target/<目录名>；
          link 不含通配符 → target 视为【精确目标】，目标 = data_root/target。
    """
    tgt = (item.get('target') or '').replace('/', os.sep)
    root = data_root()
    base = os.path.join(root, tgt) if root else tgt
    if any(ch in (item.get('link') or '') for ch in '*?['):
        base = os.path.join(base, os.path.basename(link_dir.rstrip('\\/')))
    return base


def status(path=None):
    """逐条映射的当前状态（供报告 / 自检）。"""
    out = []
    for it in items(path):
        for lp in link_paths(it):
            tp = target_of(it, lp)
            if is_link(lp):
                state = 'junction'
            elif os.path.isdir(lp):
                state = 'real'
            else:
                state = 'missing'
            out.append({
                'name': it.get('name', ''),
                'link': os.path.relpath(lp, PROJ),
                'target': tp,
                'state': state,
                'target_ok': os.path.isdir(tp),
            })
    return out


def as_dict(path=None):
    """机器可读的整体视图（给 --json 或其它脚本用）。"""
    return {
        'data_root': data_root(path),
        'config': CONFIG,
        'items': status(path),
    }


def report(path=None):
    """打印人读映射表，返回 status()。"""
    rows = status(path)
    print('数据根  : %s' % (data_root() or '(未配置)'))
    print('配置文件: %s' % CONFIG)
    if not rows:
        print('  （paths.json 的 items 为空或未匹配到目录）')
        return rows
    w = max(len(r['link']) for r in rows)
    fmt = '{0:<' + str(w) + '}  {1:<9}  {2:<6}  {3}'
    print()
    print(fmt.format('仓库内路径', '仓库侧', '数据侧', '外接盘目标'))
    mark = {'junction': '已联接', 'real': '实体目录', 'missing': '不存在'}
    for r in rows:
        print(fmt.format(r['link'], mark[r['state']],
                         'OK' if r['target_ok'] else '缺', r['target']))
    n_link = sum(1 for r in rows if r['state'] == 'junction')
    n_real = sum(1 for r in rows if r['state'] == 'real')
    print()
    print('合计 %d 条：已联接 %d · 仍是本地实体目录 %d · 不存在 %d'
          % (len(rows), n_link, n_real, len(rows) - n_link - n_real))
    if n_real:
        print('  → 还有实体目录未搬迁，执行：'
              'powershell -ExecutionPolicy Bypass -File tools\\relink_data.ps1 -Migrate')
    return rows


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if '--json' in sys.argv[1:]:
        print(json.dumps(as_dict(), ensure_ascii=False, indent=2))
    else:
        report()
