
OUT_DIR = '/home/liver/data/liver_dataset'     # 00/01/02/03 四个脚本必须一致

# 01 里跑过的 SOURCE_NAME，路径自动拼出来。
#
# ⚠️ 顺序有意义：同一个问答对在多个来源里出现时，**排在前面的来源**的那条被保留。
#    把清洗质量最高、答案最规范的来源放前面，一旦定了就别再改，否则结果不可复现。
SOURCES = [
    'cmdd',
    'huatuo26m',
    'medchatzh',
]

INPUT_PATHS = [f'{OUT_DIR}/converted/{s}.jsonl'    for s in SOURCES]
ONTO_PATHS  = [f'{OUT_DIR}/ontology_src/{s}.jsonl' for s in SOURCES]
OUTPUT_PATH = f'{OUT_DIR}/merged.jsonl'
ONTO_OUTPUT = f'{OUT_DIR}/ontology_merged.jsonl'
REPORT_DIR  = f'{OUT_DIR}/reports'

# %%
import datetime, hashlib, json, os, re, sys, unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from tqdm.auto import tqdm

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

REQUIRED_KEYS = {'anchor', 'positive', 'doc_id', 'source', 'src', 'group_id', 'is_long'}


def normalize(text):
    """与 01 保持一致的归一化，去重键依赖它"""
    text = unicodedata.normalize('NFKC', text or '').lower()
    return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text)


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
# 读入并合并
# =============================================================================

missing_files = [p for p in INPUT_PATHS if not os.path.exists(p)]
if missing_files:
    raise FileNotFoundError(
        'Converted file(s) not found — run step 01 for these sources first:\n  '
        + '\n  '.join(missing_files)
    )

records     = []
seen_pairs  = set()                  # (归一化query, doc_id) 跨来源去重
stats       = Counter()              # 全局计数
per_source  = defaultdict(Counter)   # source -> {read, dup, kept}
docs_by_src = defaultdict(set)       # source -> {doc_id}，用来算来源间重叠

print('=' * 70)
print('Step 02 · Merging converted sources')
print('=' * 70)

for path in INPUT_PATHS:
    n_before = len(records)
    with open(path, encoding='utf-8') as f:
        for line in tqdm(f, desc=os.path.basename(path), leave=False):
            if not line.strip():
                continue
            r = json.loads(line)

            missing = REQUIRED_KEYS - r.keys()
            if missing:
                raise ValueError(
                    f'{path} is missing field(s) {missing}. '
                    f'Re-generate it with the revised step 01.'
                )
            if not isinstance(r['positive'], str):
                raise ValueError(
                    f"{path}: 'positive' must be a string. "
                    f'Re-generate it with the revised step 01.'
                )

            src = r['source']
            per_source[src]['read'] += 1
            docs_by_src[src].add(r['doc_id'])

            key = (normalize(r['anchor']), r['doc_id'])
            if key in seen_pairs:
                per_source[src]['dup'] += 1
                stats['dropped_cross_source_dup'] += 1
                continue
            seen_pairs.add(key)

            records.append(r)
            per_source[src]['kept'] += 1
            stats['kept_pairs'] += 1

    stats['sources_read'] += 1
    print(f'  {os.path.basename(path):34s} +{len(records) - n_before:>9,}')

# %%
# =============================================================================
# 写出
# =============================================================================

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

stats['unique_answers'] = len({r['doc_id'] for r in records})
stats['long_answers']   = sum(bool(r.get('is_long')) for r in records)

# 报告用的标签表，结构与 01 一致：按流水线阶段分组，不用字母序。
REPORT_SECTIONS = [
    ('Input', [
        ('sources_read',             'Sources merged'),
    ]),
    ('Deduplication', [
        ('dropped_cross_source_dup', 'Cross-source duplicate pairs removed'),
    ]),
    ('Output', [
        ('kept_pairs',               'Pairs written'),
        ('unique_answers',           'Unique answers (doc_id)'),
        ('long_answers',             'Flagged is_long (kept, not dropped)'),
    ]),
]
LABELS = {k: label for _, rows in REPORT_SECTIONS for k, label in rows}
W = 40

print('\n' + '=' * 70)
print('Step 02 · Merge finished')
print('=' * 70)

for section, rows in REPORT_SECTIONS:
    print(f'\n{section}')
    for key, label in rows:
        print(f'  {label:<{W}s} {stats[key]:>9,}')

# 兜底：以后新增计数项但忘了登记，也不会被静默吞掉
for key in sorted(set(stats) - set(LABELS)):
    print(f'  {key:<{W}s} {stats[key]:>9,}   <- add to REPORT_SECTIONS')

print(f'\nOutput: {OUTPUT_PATH}')

# %%
# =============================================================================
# 各来源保留情况
# =============================================================================
# "Kept" 受 SOURCES 顺序影响：重复的问答对算在先读到的来源头上。
# 报告里要连同顺序一起写，否则别人复现不出同样的占比。

print('\n' + '=' * 70)
print('Per-source retention')
print('=' * 70)
print(f'\n{"Source":<14s}{"Read":>10s}{"Duplicate":>12s}{"Kept":>10s}'
      f'{"Share":>9s}{"Unique answers":>17s}')

total_kept = max(len(records), 1)
for src in SOURCES:
    c = per_source[src]
    print(f'{src:<14s}{c["read"]:>10,}{c["dup"]:>12,}{c["kept"]:>10,}'
          f'{c["kept"] / total_kept:>8.1%}{len(docs_by_src[src]):>17,}')

print(f'\nSource order used for deduplication: {" > ".join(SOURCES)}')
print('Report this order: a pair found in two sources is credited to the earlier one.')

# %%
# =============================================================================
# 来源间答案重叠（精确匹配层面）
# =============================================================================
# 重叠高说明这些数据集互相转载严重，03 的近重复聚簇会更重要。

overlap_rows = []
for a, b in combinations(SOURCES, 2):
    inter = len(docs_by_src[a] & docs_by_src[b])
    base  = min(len(docs_by_src[a]), len(docs_by_src[b]))
    overlap_rows.append({'a': a, 'b': b, 'shared': inter,
                         'pct_of_smaller': round(inter / max(base, 1), 4)})

print('\n' + '=' * 70)
print('Answer overlap between sources (exact matches)')
print('=' * 70)

if not overlap_rows:
    print('\n  Only one source present, nothing to compare.')
else:
    print(f'\n{"Pair":<30s}{"Shared":>10s}{"% of smaller":>15s}')
    for o in overlap_rows:
        print(f'{o["a"] + " & " + o["b"]:<30s}{o["shared"]:>10,}{o["pct_of_smaller"]:>14.1%}')

# %%
# =============================================================================
# 合并本体池
# =============================================================================

onto, onto_seen = [], set()
onto_stats = Counter()

for p in ONTO_PATHS:
    if not os.path.exists(p):
        print(f'  [skip] ontology pool not found: {p}')
        continue
    with open(p, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            onto_stats['read'] += 1
            key = (normalize(r['question']), normalize(r['answer']))
            if key in onto_seen:
                onto_stats['duplicate'] += 1
                continue
            onto_seen.add(key)
            onto.append(r)

with open(ONTO_OUTPUT, 'w', encoding='utf-8') as f:
    for r in onto:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

usable = sum(bool(r.get('is_atomic') and r.get('informative')) for r in onto)

print('\n' + '=' * 70)
print('Ontology pool merged')
print('=' * 70)
print(f'\n  {"Entries read":<{W}s} {onto_stats["read"]:>9,}')
print(f'  {"Duplicates removed":<{W}s} {onto_stats["duplicate"]:>9,}')
print(f'  {"Entries written":<{W}s} {len(onto):>9,}')
print(f'  {"Atomic and informative":<{W}s} {usable:>9,}')
print(f'\nOutput: {ONTO_OUTPUT}')

# %%
# =============================================================================
# 汇总报告
# =============================================================================

report = {
    'step':         '02_merge',
    'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'python':       sys.version.split()[0],
    'config':       {'OUT_DIR': OUT_DIR, 'SOURCES': SOURCES},
    'stats':        {**{k: stats[k] for k in LABELS}, **stats},
    'per_source': {
        s: {'read': per_source[s]['read'], 'duplicate': per_source[s]['dup'],
            'kept': per_source[s]['kept'], 'unique_answers': len(docs_by_src[s])}
        for s in SOURCES
    },
    'overlap':  overlap_rows,
    'ontology': {'read': onto_stats['read'], 'duplicate': onto_stats['duplicate'],
                 'written': len(onto), 'atomic_informative': usable},
    'outputs':  {'merged': file_info(OUTPUT_PATH), 'ontology_merged': file_info(ONTO_OUTPUT)},
}

md = [
    '# Step 02 · Merge report',
    '',
    f'Generated {report["generated_at"]} · Python {report["python"]}',
    '',
    '## Summary',
    '',
    md_table(['Stage', 'Item', 'Count'],
             [[section, label, fmt(stats[key])]
              for section, rows in REPORT_SECTIONS for key, label in rows]),
    '',
    '## Per-source retention',
    '',
    f'Deduplication order: {" > ".join(SOURCES)}. '
    'A pair found in two sources is kept in the earlier one.',
    '',
    md_table(['Source', 'Read', 'Duplicate', 'Kept', 'Share', 'Unique answers'],
             [[s, fmt(p['read']), fmt(p['duplicate']), fmt(p['kept']),
               f'{p["kept"] / total_kept:.1%}', fmt(p['unique_answers'])]
              for s, p in report['per_source'].items()]),
    '',
    '> 数据集组成表。',
    '',
    '## Answer overlap between sources (exact matches)',
    '',
    md_table(['Source A', 'Source B', 'Shared answers', '% of smaller source'],
             [[o['a'], o['b'], fmt(o['shared']), f'{o["pct_of_smaller"]:.1%}']
              for o in overlap_rows]) if overlap_rows else 'Only one source.',
    '',
    '> 说明公开数据集之间存在重复，这是做近重复去重（03）的理由。',
    '',
    '## Ontology pool',
    '',
    md_table(['Item', 'Count'], [
        ['Entries read',           fmt(onto_stats['read'])],
        ['Duplicates removed',     fmt(onto_stats['duplicate'])],
        ['Entries written',        fmt(len(onto))],
        ['Atomic and informative', fmt(usable)],
    ]),
    '',
    '## Output files',
    '',
    md_table(['File', 'Lines', 'md5'],
             [[i['path'], fmt(i['lines']), i['md5']] for i in report['outputs'].values()]),
    '',
    '> md5 用来核对：同样的输入重跑一遍，md5 应该完全一样。',
]

md_path, json_path = f'{REPORT_DIR}/02_merge.md', f'{REPORT_DIR}/02_merge.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(f'\nReport: {md_path}')
print(f'        {json_path}')
print('\nNext: run step 03 on the merged file.')
