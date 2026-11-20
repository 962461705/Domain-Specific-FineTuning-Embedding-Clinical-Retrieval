
# =============================================================================
# 配置
# =============================================================================

OUT_DIR = '/home/liver/data/liver_dataset'     

SOURCES = {
    'cmdd': [
        '/home/liver/data/Chinese Medical Dialogue Dataset/liver_CMDD_clean.jsonl',
    ],
    'huatuo26m': [
        '/home/liver/data/HuaTuo/clean_liver_train_datasets.jsonl',
        '/home/liver/data/HuaTuo/clean_liver_validation_datasets.jsonl',
        '/home/liver/data/HuaTuo/clean_liver_test_datasets.jsonl',
        '/home/liver/data/HuaTuo/clean_liver_test.jsonl',
        '/home/liver/data/HuaTuo/kg_liver_train_clean.jsonl',
    ],
    'medchatzh': [
        '/home/liver/data/MedChatZh/clean_liver_MedChatZH_train.jsonl',
    ],
}

REQUIRED_FIELDS = {'questions', 'answers'}

# 不是 UTF-8 的行，尝试按这个编码解码。GB18030 是 GBK 的超集，覆盖常见的中文旧编码。
# 设为 None 则不尝试，所有非 UTF-8 行都丢弃。
ENCODING_FALLBACK = 'gb18030'

# 按 UTF-8 解码时至少出现这么多个坏字符，才认为整行是 GB18030 编码的（见 decode_line）。
MIN_BAD_CHARS_FOR_FALLBACK = 3

# 一个文件里解码失败的行超过这个比例，就提醒检查整个文件的编码。
WARN_UNDECODABLE_RATIO = 0.01

# %%
import datetime, glob, hashlib, json, os, sys
from collections import Counter

REPORT_DIR = f'{OUT_DIR}/reports'
os.makedirs(f'{OUT_DIR}/raw', exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

UTF8_BOM = b'\xef\xbb\xbf'


def resolve(spec):
    """
    通配符 -> 排好序的文件列表；路径列表 -> 原样保留顺序。

    通配符的结果必须排序：glob 的返回顺序取决于文件系统，
    而 01 去重时保留的是先读到的那条，顺序不固定的话两次运行结果可能不同。
    """
    paths = sorted(glob.glob(spec)) if isinstance(spec, str) else list(spec)
    return [os.path.abspath(p) for p in paths]


def read_raw_lines(path):
    """
    按字节逐行读，去掉行尾换行和行首 BOM，产出 (行号, 字节串)。

    按 b'\\n' 切行对 GBK/GB18030 也是安全的：它们的多字节序列里不会出现 0x0A。
    任何一行开头的 BOM 都去掉，包括文件被别人拼接过、BOM 夹在中间的情况。
    """
    with open(path, 'rb') as f:
        for i, raw in enumerate(f):
            raw = raw.rstrip(b'\r\n')
            if raw.startswith(UTF8_BOM):
                raw = raw[len(UTF8_BOM):]
            yield i, raw


def looks_like_record(text):
    """能解析成 JSON 对象，且带 questions/answers 字段"""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and REQUIRED_FIELDS <= obj.keys()


def decode_line(raw):
    """
    解码一行字节，返回 (文本, 编码, 第一个坏字节的位置)。文本为 None 表示无法可靠解码。

    GB18030 几乎能把任何字节串"解码成功"，所以光是解码成功，不能说明这一行真是 GB18030。
    要同时满足两个条件才按 GB18030 转换：
    1. 按 UTF-8 解码时坏字符足够多。整行 GB18030 的中文几乎每个字都按 UTF-8 解不出来；
       而只坏了一两个字节的 UTF-8 行，坏字符只有一两个——这种行按 GB18030 解反而会整行乱码。
    2. 解出来是合法的 JSON 对象，并且带 questions/answers 字段。
    """
    try:
        return raw.decode('utf-8'), 'utf-8', None
    except UnicodeDecodeError as e:
        err_at = e.start

    if ENCODING_FALLBACK:
        try:
            text = raw.decode(ENCODING_FALLBACK)
        except UnicodeDecodeError:
            text = None
        n_bad = raw.decode('utf-8', 'replace').count('\ufffd')
        if text is not None and n_bad >= MIN_BAD_CHARS_FOR_FALLBACK and looks_like_record(text):
            return text, ENCODING_FALLBACK, err_at

    return None, 'undecodable', err_at


def first_keys(path):
    """
    读第一条能解码的非空记录的字段名，用来在拼接前发现格式不对的文件。
    返回 None 表示第一条不是 JSON 对象，返回空集合表示文件里没有能读的内容。
    """
    for _, raw in read_raw_lines(path):
        text, _, _ = decode_line(raw)
        if text is None or not text.strip():
            continue
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return None
        return set(obj) if isinstance(obj, dict) else None
    return set()


def locate(src, out_dir=OUT_DIR):
    """
    把 01 输出里的 src（形如 'huatuo26m:12345'）映射回原始文件和原始行号。
    行号从 0 开始，与 Python enumerate 一致。

    >>> locate('huatuo26m:12345')
    ('/home/liver/data/HuaTuo/clean_liver_test.jsonl', 2345)
    """
    source, line_no = src.rsplit(':', 1)
    line_no = int(line_no)
    with open(f'{out_dir}/raw/{source}_all.manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    for m in manifest['files']:
        if m['start_line'] <= line_no <= m['end_line']:
            return m['file'], line_no - m['start_line']
    raise ValueError(f'{src} is outside every file range in the manifest')


def md_table(headers, rows, numeric=True):
    """Markdown 表格。numeric=True 时第一列左对齐、其余右对齐，适合数字表。"""
    esc = lambda v: str(v).replace('|', '\\|').replace('\n', ' ')
    align = [':---'] + [('---:' if numeric else ':---')] * (len(headers) - 1)
    lines = ['| ' + ' | '.join(map(esc, headers)) + ' |',
             '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(map(esc, r)) + ' |' for r in rows]
    return '\n'.join(lines)


def file_info(path):
    """行数和 md5。md5 用来核对：同样的输入重跑一遍，输出应该完全一致。"""
    h, n = hashlib.md5(), 0
    with open(path, 'rb') as f:
        for line in f:
            h.update(line)
            n += 1
    return {'path': path, 'lines': n, 'md5': h.hexdigest()}


def fmt(v):
    return f'{v:,}' if isinstance(v, int) and not isinstance(v, bool) else str(v)

# %%
# =============================================================================
plan, problems, warnings = {}, [], []

for source, spec in SOURCES.items():
    out_path = os.path.abspath(f'{OUT_DIR}/raw/{source}_all.jsonl')
    files = resolve(spec)
    if not files:
        problems.append(f'[{source}] no file matched: {spec}')

    for p in files:
        if not os.path.isfile(p):
            problems.append(f'[{source}] file not found: {p}')
        elif p == out_path:
            problems.append(f'[{source}] the output file itself matched the pattern: {p}')
        else:
            keys = first_keys(p)
            if keys is None:
                problems.append(f'[{source}] first record is not a JSON object: {p}')
            elif not keys:
                warnings.append(f'[{source}] no readable record: {p}')
            elif not REQUIRED_FIELDS <= keys:
                problems.append(f'[{source}] missing {sorted(REQUIRED_FIELDS - keys)}, '
                                f'found {sorted(keys)}: {p}')

    plan[source] = (files, out_path)

print('=' * 70)
print('Step 00 · Plan')
print('=' * 70)
for source, (files, out_path) in plan.items():
    print(f'\n[{source}]  {len(files)} file(s)  ->  {out_path}')
    for p in files:
        print(f'    {p}')

for w in warnings:
    print(f'\n  warning: {w}')

if problems:
    print('\nProblems found, nothing was written:')
    for p in problems:
        print(f'  - {p}')
    raise RuntimeError('Fix the problems above and re-run step 00.')

# %%
# =============================================================================
# 拼接
# =============================================================================

COUNT_KEYS = ['lines', 'blank', 'bad_json', 'converted', 'undecodable']
summary = []

for source, (files, out_path) in plan.items():
    c = Counter()                     # 这个来源的累计计数
    manifest, issues = [], []
    issues_path = out_path.replace('_all.jsonl', '_encoding_issues.jsonl')

    with open(out_path, 'w', encoding='utf-8') as out:
        for p in files:
            c0 = c.copy()
            for line_in_file, raw in read_raw_lines(p):
                c['lines'] += 1
                text, how, err_at = decode_line(raw)

                if how != 'utf-8':
                    lo = max(0, err_at - 16)
                    issues.append({
                        'file':           p,
                        'line':           line_in_file,
                        'line_in_merged': c['lines'] - 1,
                        'status':         'dropped' if text is None else f'converted_from_{how}',
                        'first_bad_byte': err_at,
                        'bytes_hex':      raw[lo:err_at + 32].hex(' '),
                        'as_utf8':        raw.decode('utf-8', 'replace')[:150],
                        'converted':      text[:150] if text is not None else None,
                    })

                if text is None:
                    # 写一个空行占位：大文件行号减去 start_line 仍然等于原文件行号
                    out.write('\n')
                    c['undecodable'] += 1
                    continue
                if how != 'utf-8':
                    c['converted'] += 1

                # 每行都统一补一个换行：上个文件末行没有换行符时，不会和下个文件首行粘成一行。
                out.write(text + '\n')
                if not text.strip():
                    c['blank'] += 1
                    continue
                try:
                    json.loads(text)
                except json.JSONDecodeError:
                    c['bad_json'] += 1

            d = {k: c[k] - c0[k] for k in COUNT_KEYS}
            manifest.append({
                'file':        p,
                'start_line':  c0['lines'],
                'end_line':    c['lines'] - 1,
                'n_lines':     d['lines'],
                'non_blank':   d['lines'] - d['blank'],
                'blank':       d['blank'],
                'bad_json':    d['bad_json'],
                'converted':   d['converted'],
                'undecodable': d['undecodable'],
            })

    manifest_path = out_path.replace('.jsonl', '.manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump({'source': source, 'output': out_path, 'files': manifest},
                  f, ensure_ascii=False, indent=1)

    with open(issues_path, 'w', encoding='utf-8') as f:
        for r in issues:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary.append({'source': source, 'n_files': len(files),
                    **{k: c[k] for k in COUNT_KEYS},
                    'records': c['lines'] - c['blank'],
                    'out_path': out_path, 'issues_path': issues_path,
                    'issues': issues, 'files': manifest})

# %%
# =============================================================================
# 控制台报告
# =============================================================================

print('=' * 70)
print('Step 00 · Concatenation finished')
print('=' * 70)

print(f'\n{"Source":<14s}{"Files":>7s}{"Lines":>11s}{"Blank":>9s}{"Bad JSON":>10s}'
      f'{"Converted":>11s}{"Dropped":>9s}')
for s in summary:
    print(f'{s["source"]:<14s}{s["n_files"]:>7,}{s["lines"]:>11,}{s["blank"]:>9,}'
          f'{s["bad_json"]:>10,}{s["converted"]:>11,}{s["undecodable"]:>9,}')
print('\n  Converted = lines decoded as GB18030 and re-written as UTF-8 (kept)')
print('  Dropped   = lines that are not valid in any encoding (replaced by an empty line)')

for s in summary:
    print(f'\n[{s["source"]}] {s["out_path"]}')
    for m in s['files']:
        flag = ''
        if m['converted'] or m['undecodable']:
            flag = f'   <- converted {m["converted"]:,}, dropped {m["undecodable"]:,}'
        print(f'    lines {m["start_line"]:>9,} - {m["end_line"]:>9,}   '
              f'{os.path.basename(m["file"])}{flag}')

for s in summary:
    if not s['issues']:
        continue
    print(f'\n[{s["source"]}] encoding issues (full list: {s["issues_path"]})')
    for r in s['issues'][:5]:
        print(f'  {os.path.basename(r["file"])} line {r["line"]:,}: {r["status"]}')
        print(f'      as UTF-8 : {r["as_utf8"][:80]}')
        if r['converted']:
            print(f'      converted: {r["converted"][:80]}')
    for m in s['files']:
        if m['n_lines'] and m['undecodable'] / m['n_lines'] > WARN_UNDECODABLE_RATIO:
            print(f'\n  warning: {m["undecodable"] / m["n_lines"]:.1%} of the lines in '
                  f'{os.path.basename(m["file"])} could not be decoded. '
                  f'Check the encoding of the whole file, e.g. `file -i`.')

print('\nBad JSON lines are kept as-is; step 01 logs them to its dropped file.')

# %%
# =============================================================================
# 汇总报告
# =============================================================================

report = {
    'step':         '00_concat',
    'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'python':       sys.version.split()[0],
    'config':       {'OUT_DIR': OUT_DIR, 'SOURCES': SOURCES,
                     'ENCODING_FALLBACK': ENCODING_FALLBACK,
                     'MIN_BAD_CHARS_FOR_FALLBACK': MIN_BAD_CHARS_FOR_FALLBACK},
    'sources': [
        {'source': s['source'], 'n_files': s['n_files'], 'lines': s['lines'],
         'records': s['records'], 'blank': s['blank'], 'bad_json': s['bad_json'],
         'converted': s['converted'], 'undecodable': s['undecodable'],
         'output': file_info(s['out_path']), 'encoding_issues': s['issues_path'],
         'files': s['files']}
        for s in summary
    ],
}

md = [
    '# Step 00 · Concatenation report',
    '',
    f'Generated {report["generated_at"]} · Python {report["python"]}',
    '',
    '## Summary',
    '',
    md_table(['Source', 'Files', 'Non-blank lines', 'Blank lines', 'Bad JSON',
              f'Converted from {ENCODING_FALLBACK}', 'Undecodable (dropped)'],
             [[s['source'], fmt(s['n_files']), fmt(s['records']), fmt(s['blank']),
               fmt(s['bad_json']), fmt(s['converted']), fmt(s['undecodable'])]
              for s in report['sources']]),
    '',
    '> 数据来源表（每个数据集由哪些原始文件组成、各有多少条）用这张表和下面的文件清单。',
    '> 官方 train/valid/test 合并后重新划分，这里的文件清单就是依据。',
    '> Converted 是从 GB18030 转成 UTF-8 后保留的行；Undecodable 是无法可靠解码、被丢弃的行。',
    '',
    '## Files per source',
]
for s in report['sources']:
    md += ['', f'### {s["source"]}', '',
           md_table(['File', 'Lines in merged file', 'Non-blank', 'Blank', 'Bad JSON',
                     'Converted', 'Dropped'],
                    [[os.path.basename(m['file']),
                      f'{m["start_line"]:,} – {m["end_line"]:,}',
                      fmt(m['non_blank']), fmt(m['blank']), fmt(m['bad_json']),
                      fmt(m['converted']), fmt(m['undecodable'])]
                     for m in s['files']])]

md += ['', '## Encoding issues', '']
any_issue = False
for s in summary:
    if not s['issues']:
        continue
    any_issue = True
    md += [f'### {s["source"]}', '',
           f'Full list: `{s["issues_path"]}`. First 20 shown.', '',
           md_table(['File', 'Line', 'Status', 'Read as UTF-8', 'After conversion'],
                    [[os.path.basename(r['file']), fmt(r['line']), r['status'],
                      r['as_utf8'][:60], (r['converted'] or '')[:60]]
                     for r in s['issues'][:20]], numeric=False),
           '']
if not any_issue:
    md += ['All lines were valid UTF-8.', '']
else:
    md += ['> 请对照 "After conversion" 一列确认转换后的中文是通顺的；如果是乱码，'
           '把 ENCODING_FALLBACK 设为 None 重跑，这些行就会被丢弃。', '']

md += ['## Output files', '',
       md_table(['File', 'Lines', 'md5'],
                [[s['output']['path'], fmt(s['output']['lines']), s['output']['md5']]
                 for s in report['sources']]),
       '',
       '> md5 用来核对：同样的输入重跑一遍，md5 应该完全一样。']

md_path, json_path = f'{REPORT_DIR}/00_concat.md', f'{REPORT_DIR}/00_concat.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(f'\nReport: {md_path}')
print(f'        {json_path}')
print('\nNext: run step 01 once per source, setting SOURCE_NAME to each name above.')
