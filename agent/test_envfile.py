# -*- coding: utf-8 -*-
"""`.env` 加载器与 key 优先级的回归测试（零依赖、不联网）。"""
import io
import os
import sys
import tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from envfile import parse, load_env                          # noqa: E402

N_PASS = N_FAIL = 0


def case(desc, expect, got):
    global N_PASS, N_FAIL
    ok = (expect == got)
    N_PASS += ok
    N_FAIL += (not ok)
    print('  %s %-52s expect=%-22r got=%r' % ('[OK]' if ok else '[XX]', desc, expect, got))


def test_parse():
    print('== parse 语法覆盖 ==')
    case('基本形式', {'A': '1'}, parse('A=1'))
    case('键值两侧空白', {'A': '1'}, parse('A  =  1  '))
    case('export 前缀', {'A': '1'}, parse('export A=1'))
    case('双引号', {'A': 'a b'}, parse('A="a b"'))
    case('单引号', {'A': 'a b'}, parse("A='a b'"))
    case('行内注释（空格后 #）', {'A': '1'}, parse('A=1  # 说明'))
    case('井号在值中（无空格）', {'A': 'a#b'}, parse('A=a#b'))
    case('整行注释 / 空行', {}, parse('# c\n\n   \n'))
    case('非法行被忽略', {'A': '1'}, parse('这不是赋值\nA=1'))
    case('值含等号', {'A': 'b=c'}, parse('A=b=c'))
    case('空值', {'A': ''}, parse('A='))
    case('显式覆盖同名键（后写胜）', {'A': '2'}, parse('A=1\nA=2'))


def test_load():
    print('== load_env 行为 ==')
    d = tempfile.mkdtemp()
    p = os.path.join(d, '.env')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('TESTKEY_ENVFILE=A\nTESTKEY_KEEP=B\n')

    os.environ.pop('TESTKEY_ENVFILE', None)
    os.environ['TESTKEY_KEEP'] = '已有值'
    got = load_env([p])
    case('返回命中的路径', p, got)
    case('写入新键', 'A', os.environ.get('TESTKEY_ENVFILE'))
    case('**不覆盖**已存在的环境变量', '已有值', os.environ.get('TESTKEY_KEEP'))
    load_env([p], override=True)
    case('override=True 时可覆盖', 'B', os.environ.get('TESTKEY_KEEP'))
    case('文件不存在 → None', None, load_env([os.path.join(d, 'nope.env')]))

    os.environ.pop('TESTKEY_ENVFILE', None)
    os.environ.pop('TESTKEY_KEEP', None)


def test_priority():
    print('== get_key 优先级：--api-key > 环境变量 > .env ==')
    import react
    d = tempfile.mkdtemp()
    p = os.path.join(d, '.env')
    with open(p, 'w', encoding='utf-8') as f:
        f.write('DEEPSEEK_API_KEY=from_dotenv\n')

    saved = {k: os.environ.get(k) for k in
             ('DEEPSEEK_API_KEY', 'LLM_API_KEY', 'LLM_KEY')}
    try:
        for k in saved:
            os.environ.pop(k, None)
        # ① 只有 .env
        load_env([p], override=True)
        case('仅 .env → 取 .env 的值', 'from_dotenv', react.get_key())
        # ② .env + 环境变量（环境变量优先，因为 load 不覆盖）
        os.environ['DEEPSEEK_API_KEY'] = 'from_env'
        load_env([p])                       # 再加载一次也不该覆盖
        case('环境变量优先于 .env', 'from_env', react.get_key())
        # ③ 显式传参最高
        case('--api-key 最高', 'from_cli', react.get_key('from_cli'))
        # ④ 次要变量名兜底（DEEPSEEK 缺失时）
        os.environ.pop('DEEPSEEK_API_KEY', None)
        os.environ.pop('LLM_API_KEY', None)
        os.environ['LLM_KEY'] = 'from_llmkey'
        case('LLM_KEY 兜底', 'from_llmkey', react.get_key())
        # ⑤ 全空
        for k in ('DEEPSEEK_API_KEY', 'LLM_API_KEY', 'LLM_KEY'):
            os.environ.pop(k, None)
        case('全空 → None', None, react.get_key())
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main():
    test_parse()
    test_load()
    test_priority()
    print('\n%d 通过 / %d 失败' % (N_PASS, N_FAIL))
    return 1 if N_FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
