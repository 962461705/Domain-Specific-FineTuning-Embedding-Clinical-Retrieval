# %%
# =============================================================================
# 配置
# =============================================================================

OUT_DIR = '/data/neuro/data/neuro_dataset'     # 和 00–03 一致

INPUT_PATH = f'{OUT_DIR}/deduped.jsonl'         # 03 的输出
ONTO_PATH  = f'{OUT_DIR}/ontology_merged.jsonl' # 02 的输出
SPLIT_DIR  = f'{OUT_DIR}/split'
REPORT_DIR = f'{OUT_DIR}/reports'

SEED   = 42                                     # 定下来就别改
RATIOS = {'train': 0.85, 'dev': 0.05, 'test': 0.10}

STRATIFY_BY_SOURCE = True     # 每个来源分别按比例切，保证三份里各来源的占比一致

# 验证集检索语料的规模：验证集的正确答案 + 从训练集答案里随机抽的干扰项。
# 训练中每次评估都要把整个语料编码一遍，用全部答案太慢，所以抽一部分。
# 验证集只用来选 checkpoint，分数不和测试集比。设成 None 则用全部答案。
DEV_CORPUS_SIZE = 10000

# 某个连通分量占全部数据的比例超过这个值就提醒：太大的分量会让切分比例失控。
WARN_COMPONENT_SHARE = 0.02

# %%
import datetime, hashlib, json, os, random, re, sys, unicodedata
from collections import Counter, defaultdict
from itertools import combinations
from tqdm.auto import tqdm

assert abs(sum(RATIOS.values()) - 1) < 1e-9, 'RATIOS must sum to 1'
SPLITS = list(RATIOS)

os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def normalize(text):
    """与 01/02/03 保持一致的归一化"""
    text = unicodedata.normalize('NFKC', text or '').lower()
    return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text)


def qid_of(nq):
    """问题 id：归一化问题的 md5。同一个问题无论在哪条记录里都得到同一个 id"""
    return hashlib.md5(nq.encode()).hexdigest()[:16]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def md_table(headers, rows, numeric=True):
    """Markdown 表格。numeric=True 时第一列左对齐、其余右对齐，适合数字表。"""
    esc = lambda v: str(v).replace('|', '\\|').replace('\n', ' ')
    align = [':---'] + [('---:' if numeric else ':---')] * (len(headers) - 1)
    lines = ['| ' + ' | '.join(map(esc, headers)) + ' |',
             '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(map(esc, r)) + ' |' for r in rows]
    return '\n'.join(lines)


def file_info(path):
    """行数和 md5。md5 用来核对：同样的输入和种子重跑一遍，输出应该完全一致。"""
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
# 读入
# =============================================================================

REQUIRED = {'anchor', 'positive', 'doc_id', 'source', 'group_id', 'cluster_id'}

records = []
with open(INPUT_PATH, encoding='utf-8') as f:
    for line in tqdm(f, desc='Reading'):
        if line.strip():
            records.append(json.loads(line))

for r in records:
    missing = REQUIRED - r.keys()
    if missing:
        raise ValueError(f'{INPUT_PATH}: record is missing {sorted(missing)}. '
                         f'Re-run step 03.')

norm_q = [normalize(r['anchor']) for r in records]
N = len(records)

print('=' * 70)
print('Step 04 · Train / dev / test split')
print('=' * 70)
print(f'\n  {N:,} pairs read from {INPUT_PATH}')
print(f'  SEED={SEED}  RATIOS={RATIOS}  STRATIFY_BY_SOURCE={STRATIFY_BY_SOURCE}')

# %%
# =============================================================================
# 建图，求连通分量
# =============================================================================
# 节点：('c', cluster_id) 答案簇、('q', 归一化问题)、('g', group_id) 同义问法组。
# 每条记录把这三个节点连起来。连在一起的记录必须分在同一边。

parent = {}


def find(x):
    parent.setdefault(x, x)
    root = x
    while parent[root] != root:
        root = parent[root]
    while parent[x] != root:              # 路径压缩
        parent[x], x = root, parent[x]
    return root


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[rb] = ra


for i, r in enumerate(records):
    c = ('c', r['cluster_id'])
    union(c, ('q', norm_q[i]))
    union(c, ('g', r['group_id']))

by_root = defaultdict(list)
for i, r in enumerate(records):
    by_root[find(('c', r['cluster_id']))].append(i)

# 分量编号按"分量里最早出现的记录"排序，与并查集内部的根节点无关，保证可复现
components = sorted(by_root.values(), key=lambda idxs: idxs[0])
comp_of = [0] * N
for k, idxs in enumerate(components):
    for i in idxs:
        comp_of[i] = k

comp_sizes = sorted((len(c) for c in components), reverse=True)
print(f'\n  {len(components):,} connected components')
print(f'  Largest component: {comp_sizes[0]:,} pairs ({comp_sizes[0] / N:.2%} of all)')
print(f'  Components with one pair: {sum(s == 1 for s in comp_sizes):,}')

# %%
# =============================================================================
# 分配：打乱分量，按来源分层贪心分配
# =============================================================================
# 每个分量按打乱后的顺序，分给"当前离目标比例差得最多"的那一份。
# 分层时，一个分量算在它记录最多的那个来源头上（跨来源的分量很少）。

rng = random.Random(SEED)
order = list(range(len(components)))
rng.shuffle(order)

assigned  = defaultdict(Counter)   # 分层 -> 每份已分到的记录数
processed = Counter()              # 分层 -> 已处理的记录数
split_of_comp = {}

for k in order:
    idxs = components[k]
    n = len(idxs)
    stratum = (Counter(records[i]['source'] for i in idxs).most_common(1)[0][0]
               if STRATIFY_BY_SOURCE else 'all')
    after = processed[stratum] + n
    # 缺口 = 按目标比例应有的记录数 - 已分到的记录数；分给缺口最大的那份
    best = max(SPLITS, key=lambda sp: RATIOS[sp] * after - assigned[stratum][sp])
    split_of_comp[k] = best
    assigned[stratum][best] += n
    processed[stratum] = after

split_idx = {sp: [] for sp in SPLITS}
for i in range(N):
    split_idx[split_of_comp[comp_of[i]]].append(i)

assert sum(len(v) for v in split_idx.values()) == N

# %%
# =============================================================================
# 泄漏检查：任意两份之间，问题 / 答案 / 答案簇 / 问法组都不能有重叠
# =============================================================================

LEAK_KEYS = {
    'normalized question': lambda i: norm_q[i],
    'answer (doc_id)':     lambda i: records[i]['doc_id'],
    'original answer':     lambda i: records[i].get('orig_doc_id', records[i]['doc_id']),
    'answer cluster':      lambda i: records[i]['cluster_id'],
    'paraphrase group':    lambda i: records[i]['group_id'],
}

leak_rows = []
for name, fn in LEAK_KEYS.items():
    sets = {sp: {fn(i) for i in split_idx[sp]} for sp in SPLITS}
    for a, b in combinations(SPLITS, 2):
        leak_rows.append({'key': name, 'a': a, 'b': b, 'overlap': len(sets[a] & sets[b])})

print('\n' + '=' * 70)
print('Leakage check (overlap between splits must be 0)')
print('=' * 70)
for row in leak_rows:
    print(f'  {row["key"]:<22s} {row["a"]:>5s} & {row["b"]:<5s} {row["overlap"]:>7,}')

if any(row['overlap'] for row in leak_rows):
    raise RuntimeError('Leakage detected between splits; nothing was written.')
print('  OK')

# %%
# =============================================================================
# 写出：train / dev / test 问答对
# =============================================================================
# dict(r) 保留原有键的顺序，anchor/positive 仍是前两列；split、component_id 追加在末尾。

for sp in SPLITS:
    rows = []
    for i in split_idx[sp]:
        r = dict(records[i])
        r['split'] = sp
        r['component_id'] = comp_of[i]
        rows.append(r)
    write_jsonl(f'{SPLIT_DIR}/{sp}.jsonl', rows)

# %%
# =============================================================================
# 检索评测文件：queries / qrels / corpus
# =============================================================================
# 一个问题可能有多个正确答案（同一个问题在不同来源里配了不同答案），qrels 全部记下。

doc_info = {}
for i, r in enumerate(records):
    if r['doc_id'] not in doc_info:
        doc_info[r['doc_id']] = {
            'doc_id':     r['doc_id'],
            'text':       r['positive'],
            'source':     r['source'],
            'cluster_id': r['cluster_id'],
            'is_long':    bool(r.get('is_long')),
            'split':      split_of_comp[comp_of[i]],
        }
all_docs   = sorted(doc_info)
train_docs = [d for d in all_docs if doc_info[d]['split'] == 'train']


def build_ir(sp):
    """返回 (queries 行, qrels 行, {qid: 正确答案集合})"""
    text, src, rel = {}, {}, defaultdict(set)
    for i in split_idx[sp]:
        qid = qid_of(norm_q[i])
        text.setdefault(qid, records[i]['anchor'])
        src.setdefault(qid, records[i]['source'])
        rel[qid].add(records[i]['doc_id'])
    qids = sorted(rel)
    q_rows = [{'qid': q, 'text': text[q], 'source': src[q], 'n_positives': len(rel[q])}
              for q in qids]
    r_rows = [{'qid': q, 'doc_id': d, 'score': 1} for q in qids for d in sorted(rel[q])]
    return q_rows, r_rows, rel


ir = {}
for sp in ('dev', 'test'):
    q_rows, r_rows, rel = build_ir(sp)
    write_jsonl(f'{SPLIT_DIR}/{sp}_queries.jsonl', q_rows)
    write_jsonl(f'{SPLIT_DIR}/{sp}_qrels.jsonl', r_rows)
    ir[sp] = {'queries': len(q_rows), 'qrels': len(r_rows),
              'multi_positive': sum(len(v) > 1 for v in rel.values()), 'rel': rel}

# 测试语料：全部答案
test_corpus = all_docs

# 验证语料：验证集正确答案 + 训练集答案里随机抽的干扰项
dev_pos = sorted({d for ds in ir['dev']['rel'].values() for d in ds})
if DEV_CORPUS_SIZE is None:
    dev_corpus = all_docs
else:
    n_extra = max(0, DEV_CORPUS_SIZE - len(dev_pos))
    extra = random.Random(SEED).sample(train_docs, min(n_extra, len(train_docs)))
    dev_corpus = sorted(set(dev_pos) | set(extra))

for name, ids in [('dev', dev_corpus), ('test', test_corpus), ('train', train_docs)]:
    write_jsonl(f'{SPLIT_DIR}/{name}_corpus.jsonl', [doc_info[d] for d in ids])

ir['dev']['corpus'], ir['test']['corpus'] = len(dev_corpus), len(test_corpus)

# %%
# =============================================================================
# 本体池：去掉问题出现在验证集 / 测试集里的条目
# =============================================================================

heldout_q = {norm_q[i] for sp in SPLITS if sp != 'train' for i in split_idx[sp]}
onto = None
if os.path.exists(ONTO_PATH):
    kept, removed = [], []
    with open(ONTO_PATH, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            (removed if normalize(r['question']) in heldout_q else kept).append(r)
    write_jsonl(f'{SPLIT_DIR}/ontology_train.jsonl', kept)
    write_jsonl(f'{SPLIT_DIR}/ontology_removed.jsonl', removed)
    onto = {'read': len(kept) + len(removed), 'kept': len(kept), 'removed': len(removed)}
else:
    print(f'\n  [skip] ontology pool not found: {ONTO_PATH}')

# %%
# =============================================================================
# 统计
# =============================================================================


def split_stats(sp):
    idx = split_idx[sp]
    return {
        'pairs':      len(idx),
        'share':      round(len(idx) / N, 4),
        'questions':  len({norm_q[i] for i in idx}),
        'answers':    len({records[i]['doc_id'] for i in idx}),
        'clusters':   len({records[i]['cluster_id'] for i in idx}),
        'components': len({comp_of[i] for i in idx}),
        'long_pairs': sum(bool(records[i].get('is_long')) for i in idx),
    }


stats      = {sp: split_stats(sp) for sp in SPLITS}
per_source = {sp: Counter(records[i]['source'] for i in split_idx[sp]) for sp in SPLITS}
sources    = sorted({r['source'] for r in records})

big = []
for k in sorted(range(len(components)), key=lambda k: -len(components[k]))[:5]:
    idxs = components[k]
    big.append({
        'component_id': k,
        'pairs':        len(idxs),
        'share':        round(len(idxs) / N, 4),
        'split':        split_of_comp[k],
        'sources':      dict(Counter(records[i]['source'] for i in idxs)),
        'example':      records[idxs[0]]['anchor'][:40],
    })

W = 14
print('\n' + '=' * 70)
print('Split sizes')
print('=' * 70)
print(f'\n{"":<{W}s}' + ''.join(f'{sp:>12s}' for sp in SPLITS))
for key in ['pairs', 'questions', 'answers', 'clusters', 'components', 'long_pairs']:
    print(f'{key:<{W}s}' + ''.join(f'{stats[sp][key]:>12,}' for sp in SPLITS))
print(f'{"share":<{W}s}' + ''.join(f'{stats[sp]["share"]:>12.1%}' for sp in SPLITS))

print('\nPer source (pairs, and share of that source)')
for s in sources:
    tot = sum(per_source[sp][s] for sp in SPLITS)
    cells = ''.join(f'{per_source[sp][s]:>8,} ({per_source[sp][s] / tot:5.1%})' for sp in SPLITS)
    print(f'  {s:<12s}{cells}')

print('\nRetrieval evaluation files')
for sp in ('dev', 'test'):
    e = ir[sp]
    print(f'  {sp:<5s} queries {e["queries"]:>7,}   corpus {e["corpus"]:>7,}   '
          f'qrels {e["qrels"]:>7,}   multi-positive queries {e["multi_positive"]:>6,}')
print(f'  train corpus (for hard-negative mining): {len(train_docs):,} answers')

if onto:
    print(f'\nOntology pool: {onto["read"]:,} read, {onto["removed"]:,} removed '
          f'(question in dev/test), {onto["kept"]:,} kept')

print('\nLargest components')
for b in big:
    print(f'  #{b["component_id"]:<6} {b["pairs"]:>6,} pairs ({b["share"]:.2%})  -> {b["split"]:<5s}  '
          f'{b["sources"]}  e.g. {b["example"]}')
if big and big[0]['share'] > WARN_COMPONENT_SHARE:
    print(f'\n  warning: the largest component holds {big[0]["share"]:.1%} of the data. '
          f'Check that it is not a chain of unrelated answers (see step 03).')

# %%
# =============================================================================
# 汇总报告
# =============================================================================

outputs = {}
for name in ['train', 'dev', 'test',
             'dev_queries', 'dev_qrels', 'dev_corpus',
             'test_queries', 'test_qrels', 'test_corpus', 'train_corpus']:
    outputs[name] = file_info(f'{SPLIT_DIR}/{name}.jsonl')
if onto:
    outputs['ontology_train']   = file_info(f'{SPLIT_DIR}/ontology_train.jsonl')
    outputs['ontology_removed'] = file_info(f'{SPLIT_DIR}/ontology_removed.jsonl')

draft = (
    f'To prevent leakage, we split the data at the level of connected components of a '
    f'graph linking each normalized question, its paraphrase group and the near-duplicate '
    f'cluster of its answer, so that no question, answer or near-duplicate answer appears in '
    f'more than one split. The {len(components):,} components were shuffled with a fixed seed '
    f'({SEED}) and assigned to training, validation and test sets in a '
    f'{round(RATIOS["train"] * 100)}/{round(RATIOS["dev"] * 100)}/{round(RATIOS["test"] * 100)} '
    f'ratio, stratified by source. The resulting sets contain '
    f'{stats["train"]["pairs"]:,}, {stats["dev"]["pairs"]:,} and {stats["test"]["pairs"]:,} '
    f'QA pairs ({stats["train"]["questions"]:,}, {stats["dev"]["questions"]:,} and '
    f'{stats["test"]["questions"]:,} unique questions). Each test question is retrieved '
    f'against the full collection of {len(test_corpus):,} unique answers; '
    f'{ir["test"]["multi_positive"]:,} test questions have more than one relevant answer. '
    f'The validation set, used only for model selection, is evaluated against a corpus of '
    f'{len(dev_corpus):,} answers.'
)
if onto:
    draft += (f' The ontology and all hard negatives are derived from the training split only; '
              f'{onto["removed"]:,} ontology entries whose question occurs in the validation '
              f'or test set were removed.')

report = {
    'step':         '04_split',
    'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'python':       sys.version.split()[0],
    'config': {
        'SEED': SEED, 'RATIOS': RATIOS, 'STRATIFY_BY_SOURCE': STRATIFY_BY_SOURCE,
        'DEV_CORPUS_SIZE': DEV_CORPUS_SIZE,
    },
    'components':      {'count': len(components), 'largest': comp_sizes[0],
                        'singletons': sum(s == 1 for s in comp_sizes)},
    'splits':          stats,
    'per_source':      {sp: dict(per_source[sp]) for sp in SPLITS},
    'leakage':         leak_rows,
    'evaluation':      {sp: {k: v for k, v in ir[sp].items() if k != 'rel'} for sp in ir},
    'train_corpus':    len(train_docs),
    'ontology':        onto,
    'largest_components': big,
    'outputs':         outputs,
    'draft_paragraph': draft,
}

md = [
    '# Step 04 · Split report',
    '',
    f'Generated {report["generated_at"]} · Python {report["python"]}',
    '',
    '## Configuration',
    '',
    md_table(['Parameter', 'Value'], [[k, v] for k, v in report['config'].items()],
             numeric=False),
    '',
    '## Split sizes',
    '',
    md_table(['', *SPLITS],
             [[key, *[fmt(stats[sp][key]) for sp in SPLITS]]
              for key in ['pairs', 'questions', 'answers', 'clusters', 'components', 'long_pairs']]
             + [['share', *[f'{stats[sp]["share"]:.1%}' for sp in SPLITS]]]),
    '',
    '> 用法：数据集划分表。',
    '',
    '## Pairs per source',
    '',
    md_table(['Source', *SPLITS, 'Total'],
             [[s, *[f'{per_source[sp][s]:,} ({per_source[sp][s] / max(sum(per_source[x][s] for x in SPLITS), 1):.1%})'
                    for sp in SPLITS],
               fmt(sum(per_source[sp][s] for sp in SPLITS))] for s in sources]),
    '',
    '## Leakage check',
    '',
    md_table(['Key', 'Split A', 'Split B', 'Overlap'],
             [[x['key'], x['a'], x['b'], fmt(x['overlap'])] for x in leak_rows]),
    '',
    '> 全部为 0 才说明没有泄漏。',
    '',
    '## Retrieval evaluation files',
    '',
    md_table(['Split', 'Queries', 'Corpus', 'Qrels', 'Multi-positive queries'],
             [[sp, fmt(ir[sp]['queries']), fmt(ir[sp]['corpus']), fmt(ir[sp]['qrels']),
               fmt(ir[sp]['multi_positive'])] for sp in ('dev', 'test')]),
    '',
    f'The test corpus contains all {len(test_corpus):,} unique answers. '
    f'The dev corpus contains the dev answers plus training answers sampled as distractors '
    f'(seed {SEED}); dev scores are only for model selection and are not comparable to test scores.',
    '',
    f'Training corpus for hard-negative mining: {len(train_docs):,} answers.',
    '',
]
if onto:
    md += ['## Ontology pool', '',
           md_table(['Item', 'Count'], [['Entries read', fmt(onto['read'])],
                                        ['Removed (question in dev/test)', fmt(onto['removed'])],
                                        ['Kept for ontology construction', fmt(onto['kept'])]]),
           '']
md += ['## Largest components', '',
       md_table(['Component', 'Pairs', 'Share', 'Split', 'Sources', 'Example question'],
                [[b['component_id'], fmt(b['pairs']), f'{b["share"]:.2%}', b['split'],
                  ', '.join(f'{k}: {v}' for k, v in b['sources'].items()), b['example']]
                 for b in big], numeric=False),
       '',
       '## Draft paragraph for the paper', '',
       draft,
       '',
       '',
       '## Output files', '',
       md_table(['File', 'Lines', 'md5'],
                [[i['path'], fmt(i['lines']), i['md5']] for i in outputs.values()]),
       '',
       '> md5 用来核对：同样的输入和种子重跑一遍，md5 应该完全一样。']

md_path, json_path = f'{REPORT_DIR}/04_split.md', f'{REPORT_DIR}/04_split.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(f'\nOutputs: {SPLIT_DIR}/')
print(f'Report : {md_path}')
print(f'         {json_path}')
