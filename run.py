# -*- coding: utf-8 -*-
"""统一入口 —— 在两类数据集间自由切换并执行对应任务

数据集:
  main : 内部疲劳机主样本 016-020 (服役期渐进损伤)
  l1   : 公开集 ReMAP/TU-Delft L1-03/04/05/09 (冲击后疲劳)

用法:
  python run.py --list                      # 查看任务矩阵
  python run.py --dataset main --task degree
  python run.py --dataset l1   --task degree
  python run.py --dataset l1   --task all
  python run.py --dataset l1   --task paper -- --groups L1-03   # '--' 之后透传给底层脚本

说明: 本脚本是"任务调度器", 不重写算法; 每类数据集的任务映射到既有脚本(见下表)。
"""
import os
import sys
import argparse
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# ============ 任务矩阵 ============
# 每个 task -> 若干条底层命令(不带 python 前缀); 'all' 为该数据集全部任务。
L1_DEGREE = ['l1/evaluate_l1_degree.py', '--baseline', '--strain-evidence',
             '--fusion', 'max', '--params', 'rise=0.05']

TASKS = {
    'main': {
        'prepare': [['main/prepare_data.py', 'align']],
        'labels':  [['main/prepare_data.py', 'weaklabels']],
        'degree':  [['main/evaluate.py', 'degree']],
        'warning': [['main/evaluate.py', 'warning']],
        'curves':  [['main/evaluate.py', 'curves']],
        'paper':   [['main/evaluate.py', 'paper']],
        'robust':  [['main/robustness.py', 'sens'],
                    ['main/robustness.py', 'loso'],
                    ['main/robustness.py', 'ablation'],
                    ['main/robustness.py', 'stats']],
        'fusion':  [['main/fusion_compare.py']],              # 源融合对照(单源/二源/三源)
        'grade':   [['main/grade_compare.py']],               # 级别层异源分级(刚度损失闸门)
        'dashboard': [['main/export_dashboard.py']],          # → dashboard/data/
    },
    'l1': {
        'prepare':  [['l1/step0.py']],                       # 原始 .pridb/.txt → CSV
        'degree':   [L1_DEGREE],                             # D(t) + 三级预警
        'curves':   [L1_DEGREE],
        'paper':    [['l1/reproduce_broer_l1.py', '--mode', 'all']],  # 论文 Level1 + Level4
        'paper-l23': [['l1/reproduce_broer_l23.py', '--mode', 'all']],  # 论文 Level2 + Level3
        'dfos':     [['l1/evaluate_l1_dfos.py', '--mode', 'hi']],     # 分布式应变逐块
        'fiber-hi': [['l1/evaluate_l1.py', '--mode', 'hi']],          # 光纤(FBG)块级 HI
        'dashboard': [['l1/export_dashboard_l1.py']],                  # → dashboard/data/
    },
}

DESC = {
    'prepare':  ('数据准备/预处理', 'prepare_data.py align: 多源对齐', 'step0.py: 原始→CSV'),
    'labels':   ('弱标签/失效锚', 'prepare_data.py weaklabels: b2/b3 弱标签', '（无弱标签；以 n_f 为失效锚）'),
    'degree':   ('连续损伤度 D(t)+分级', 'evaluate.py degree: D 达阈/单调', 'evaluate_l1_degree.py: 基线重定义+应变漂移证据'),
    'warning':  ('预警 onset/分级', 'evaluate.py warning: A-预警', '（含在 degree 输出的 results/l1_degree.csv）'),
    'curves':   ('D(t) 曲线出图', 'evaluate.py curves: 5 组曲线', 'evaluate_l1_degree.py: 逐组图'),
    'paper':    ('论文图表', 'evaluate.py paper: 四联图+流程图', 'reproduce_broer_l1.py: 论文 Level1/4 复现'),
    'paper-l23': ('论文 L2/L3 复现', '—', 'reproduce_broer_l23.py: L3a 精确复现; L3b 未复现; AE 支路已接入(PCA2 73-75%)'),
    'dfos':     ('分布式应变分析', '—', 'evaluate_l1_dfos.py: 逐块+热图'),
    'fiber-hi': ('光纤块级 HI', '—', 'evaluate_l1.py: FBG 块级 HI'),
    'robust':   ('稳健性/统计', 'robustness.py: sens/loso/ablation/stats', '— （未做）'),
    'fusion':   ('源融合对照', 'fusion_compare.py: 5 组 × 9 配置', '—'),
    'grade':    ('级别层异源分级', 'grade_compare.py: 刚度损失闸门', '—'),
    'dashboard': ('看板数据导出', 'export_dashboard.py → dashboard/data/', 'export_dashboard_l1.py → dashboard/data/'),
}


def print_matrix():
    print('任务矩阵 (dataset × task):\n')
    hdr = f'{"task":<10}{"说明":<20}{"main (016-020)":<40}{"l1 (L1-03/04/05/09)":<40}'
    print(hdr)
    print('-' * len(hdr))
    for t, (name, m, l) in DESC.items():
        print(f'{t:<10}{name:<20}{m:<40}{l:<40}')


def resolve(dataset, task):
    ds = TASKS.get(dataset)
    if ds is None:
        print(f'[错误] 未知数据集: {dataset} (可选: {", ".join(TASKS)})')
        return None
    if task == 'all':
        cmds, seen = [], set()
        for v in ds.values():
            for c in v:
                key = tuple(c)
                if key not in seen:
                    seen.add(key); cmds.append(c)
        return cmds
    if task not in ds:
        print(f'[提示] 数据集 "{dataset}" 无任务 "{task}"。'
              f' 可用任务: {", ".join(ds)} (或 all)')
        return None
    return ds[task]


def main():
    ap = argparse.ArgumentParser(description='多源损伤度 D(t) —— 数据集统一入口')
    ap.add_argument('--dataset', choices=list(TASKS), help='数据源: main | l1')
    ap.add_argument('--task', help='任务名 (见 --list), 或 all')
    ap.add_argument('--list', action='store_true', help='打印任务矩阵')
    ap.add_argument('--paths', action='store_true', help='打印数据集路径映射（paths.json）并自检')
    ap.add_argument('--dry-run', action='store_true', help='只打印将执行的命令')
    a, extra = ap.parse_known_args()
    # '--' 之后的内容透传
    if '--' in extra:
        extra = extra[extra.index('--') + 1:]
    elif extra and extra[0] == '--':
        extra = extra[1:]

    if a.paths:
        import shm.paths as _paths
        _paths.report()
        return

    if a.list or not a.dataset or not a.task:
        print_matrix()
        if not (a.list or (a.dataset and a.task)):
            ap.print_help()
        return

    cmds = resolve(a.dataset, a.task)
    if cmds is None:
        return
    for c in cmds:
        script = os.path.join(ROOT, c[0])
        full = [PY, script] + c[1:] + extra
        print(f'\n>>> [{a.dataset}/{a.task}] ' + ' '.join(c + extra))
        if a.dry_run:
            continue
        r = subprocess.run(full, cwd=os.path.dirname(script))
        if r.returncode != 0:
            print(f'[警告] 命令返回码 {r.returncode}: {" ".join(c)}')


if __name__ == '__main__':
    main()
