
# %%
# =============================================================================
# 配置
# =============================================================================

OUT_DIR = '/data/neuro/data/neuro_dataset'     # 00/01/02/03 四个脚本必须一致

INPUT_PATH   = f'{OUT_DIR}/merged.jsonl'
OUTPUT_PATH  = f'{OUT_DIR}/deduped.jsonl'
MEMBERS_PATH = f'{OUT_DIR}/cluster_members.json'

NUM_PERM  = 128    # MinHash 排列数；降到 64 提速约一倍，精度略降
SHINGLE_K = 4      # 中文用字符 4-gram
THRESHOLD = 0.8    # Jaccard 阈值，越低越激进。务必跑最后一块做人工校准

MERGE_TO_REPRESENTATIVE = True   # 把同簇答案统一成代表文本（建议开）
MAX_MERGE_SIZE          = 20     # 簇大于此值 → 判为误合并链，不折叠文本
MIN_SHINGLE_DOC_LEN     = 40     # 短于此长度的答案不参与聚簇
LONG_ANSWER_CHARS       = 480    # 必须和 01 一致：折叠成代表文本后要重算 is_long

# %%
import datetime, hashlib, json, os, random, re, sys, unicodedata
from collections import Counter, defaultdict
from tqdm.auto import tqdm
from datasketch import MinHash, MinHashLSH

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def normalize(text):
    """与 01/02 保持一致的归一化"""
    text = unicodedata.normalize('NFKC', text or '').lower()
    return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text)


def shingles(text, k=SHINGLE_K):
    if len(text) < k:
        return {text} if text else set()
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def build_minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    sh = shingles(text)
    if sh:
        m.update_batch([s.encode('utf-8') for s in sh])
    return m


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

# %%
# =============================================================================
# 读入，提取唯一答案
# =============================================================================
# 聚簇在**唯一答案**层面做，不在样本层面做。
# 多条样本共享同一个 doc_id，只需为每个唯一答案建一次 MinHash。

records  = []
doc_text = {}       # doc_id -> 答案原文
stats    = Counter()

with open(INPUT_PATH, encoding='utf-8') as f:
    for line in tqdm(f, desc='Reading'):
        if not line.strip():
            continue
        r = json.loads(line)
        records.append(r)
        doc_text.setdefault(r['doc_id'], r['positive'])

doc_list = sorted(doc_text)          # 固定顺序，保证可复现
doc_norm = [normalize(doc_text[d]) for d in doc_list]

stats['input_pairs']    = len(records)
stats['unique_answers'] = len(doc_list)

print('=' * 70)
print('Step 03 · Near-duplicate clustering')
print('=' * 70)
print(f'\n  {len(records):,} pairs, {len(doc_list):,} unique answers')
print(f'  NUM_PERM={NUM_PERM}  SHINGLE_K={SHINGLE_K}  THRESHOLD={THRESHOLD}  '
      f'MAX_MERGE_SIZE={MAX_MERGE_SIZE}')

# %%
# =============================================================================
# MinHash + LSH + 复核 + 并查集
# =============================================================================

print('\nBuilding MinHash signatures (slowest stage)...')
mhs = [build_minhash(t) for t in tqdm(doc_norm, desc='MinHash')]

lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
for i, m in enumerate(tqdm(mhs, desc='Indexing')):
    if len(doc_norm[i]) < MIN_SHINGLE_DOC_LEN:
        continue                      # 太短，shingle 不足，不参与聚簇，自己单独成簇
    lsh.insert(str(i), m)
    stats['answers_clustered'] += 1

stats['answers_too_short'] = len(mhs) - stats['answers_clustered']

uf = UnionFind(len(mhs))
for i, m in enumerate(tqdm(mhs, desc='Querying')):
    if len(doc_norm[i]) < MIN_SHINGLE_DOC_LEN:
        continue
    for key in lsh.query(m):
        j = int(key)
        if j <= i:                    # 每对只处理一次，顺带跳过自己
            continue
        stats['lsh_candidate_pairs'] += 1
        if mhs[i].jaccard(mhs[j]) >= THRESHOLD:
            uf.union(i, j)
            stats['verified_pairs_merged'] += 1

cluster_of_idx = [uf.find(i) for i in range(len(mhs))]
members = defaultdict(list)
for i, c in enumerate(cluster_of_idx):
    members[c].append(i)

stats['clusters'] = len(members)

verify_rate = stats['verified_pairs_merged'] / max(stats['lsh_candidate_pairs'], 1)
print(f'\n  LSH candidate pairs   : {stats["lsh_candidate_pairs"]:>9,}')
print(f'  Verified and merged   : {stats["verified_pairs_merged"]:>9,} '
      f'({verify_rate:.1%})')
print('  A low verification rate quantifies how many spurious merges the')
print('  unverified version would have performed.')
print(f'\n  {len(doc_list):,} unique answers -> {len(members):,} clusters '
      f'({1 - len(members) / max(len(doc_list), 1):.1%} compression)')

size_dist = Counter(len(v) for v in members.values())
print('\nCluster size distribution')
for size in sorted(size_dist)[:10]:
    print(f'  {size:>3} answer(s) per cluster : {size_dist[size]:>9,} clusters')
if size_dist and max(size_dist) > 10:
    big = sum(v for k, v in size_dist.items() if k > 10)
    print(f'  >10 answers per cluster  : {big:>9,} clusters '
          f'(largest: {max(size_dist)})')

# %%
# =============================================================================
# 巨簇体检
# =============================================================================
# 即使复核了 Jaccard，并查集的传递性仍可能把 A~B~C 串起来而 A 与 C 并不像。
# 超过 MAX_MERGE_SIZE 的簇一律不折叠文本——宁可数据里留点近重复，
# 也不能把不同的答案当成同一个删掉。

oversized = {c for c, idxs in members.items() if len(idxs) > MAX_MERGE_SIZE}
stats['oversized_clusters']   = len(oversized)
stats['answers_in_oversized'] = sum(len(members[c]) for c in oversized)

print('\n' + '=' * 70)
print(f'Oversized clusters (> {MAX_MERGE_SIZE} members)')
print('=' * 70)
print(f'\n  Clusters            : {stats["oversized_clusters"]:>9,}')
print(f'  Answers involved    : {stats["answers_in_oversized"]:>9,} '
      f'({stats["answers_in_oversized"] / max(len(doc_list), 1):.1%})')

if oversized:
    print('\n  These retain a cluster_id but their texts are NOT collapsed.')
    print('  Three largest clusters, sampled — check they are really one thing:')
    for c in sorted(oversized, key=lambda x: -len(members[x]))[:3]:
        print(f'\n  -- cluster {c} ({len(members[c])} answers)')
        for i in members[c][:3]:
            print(f'     - {doc_text[doc_list[i]][:90]}...')
    print('\n  If unrelated content has been chained together, raise THRESHOLD '
          'and re-run.')

# %%
# =============================================================================
# 选代表：每簇取最长的答案
# =============================================================================
# 最长通常信息最全。被折叠掉的答案原文不丢，记在 cluster_members.json 里备查。

rep_idx_of_cluster = {c: max(idxs, key=lambda i: len(doc_norm[i]))
                      for c, idxs in members.items()}

# doc_id -> (cluster_id, 代表 doc_id, 代表文本, 是否允许折叠)
doc_map = {}
for i, d in enumerate(doc_list):
    c = cluster_of_idx[i]
    ri = rep_idx_of_cluster[c]
    doc_map[d] = (c, doc_list[ri], doc_text[doc_list[ri]], c not in oversized)

# %%
# =============================================================================
# 回填到样本 + 折叠后再去一次重
# =============================================================================
# 折叠之后，原本指向两条近重复答案的相同问题会变成真正的重复。
# 这对重复在 02 跑的时候还不存在，只有折叠之后才出现，所以必须在这里再查一遍。
#
# new = dict(r) 会保留原有键的顺序，anchor/positive 仍是前两列；
# 新增的 cluster_id、orig_doc_id 追加在末尾。

out        = []
seen_pairs = set()

for r in tqdm(records, desc='Rewriting'):
    cid, rep_did, rep_text, may_merge = doc_map[r['doc_id']]

    new = dict(r)
    new['cluster_id']  = int(cid)
    new['orig_doc_id'] = r['doc_id']
    if MERGE_TO_REPRESENTATIVE and may_merge:
        new['doc_id']    = rep_did
        new['positive']  = rep_text
        new['n_chars_a'] = len(rep_text)
        new['is_long']   = len(rep_text) > LONG_ANSWER_CHARS
        if rep_did != r['doc_id']:
            stats['answers_collapsed'] += 1
    elif not may_merge:
        stats['kept_uncollapsed_oversized'] += 1

    key = (normalize(new['anchor']), new['doc_id'])
    if key in seen_pairs:
        stats['dup_created_by_collapse'] += 1
        continue
    seen_pairs.add(key)

    out.append(new)
    stats['output_pairs'] += 1

stats['output_unique_answers'] = len({r['doc_id'] for r in out})
stats['long_answers'] = sum(bool(r.get('is_long')) for r in out)

# %%
# =============================================================================
# 写出 + 报告
# =============================================================================

with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

# 簇成员留档，方便核对哪些被折叠了
with open(MEMBERS_PATH, 'w', encoding='utf-8') as f:
    json.dump(
        {str(c): [doc_list[i] for i in idxs]
         for c, idxs in members.items() if len(idxs) > 1},
        f, ensure_ascii=False, indent=1
    )


REPORT_SECTIONS = [
    ('Input', [
        ('input_pairs',                 'Pairs read'),
        ('unique_answers',              'Unique answers'),
        ('answers_clustered',           'Answers entering clustering'),
        ('answers_too_short',           f'Excluded, shorter than {MIN_SHINGLE_DOC_LEN}'),
    ]),
    ('Clustering', [
        ('lsh_candidate_pairs',         'LSH candidate pairs'),
        ('verified_pairs_merged',       'Verified above threshold and merged'),
        ('clusters',                    'Clusters formed'),
        ('oversized_clusters',          f'Oversized clusters (> {MAX_MERGE_SIZE})'),
        ('answers_in_oversized',        'Answers in oversized clusters'),
    ]),
    ('Collapsing', [
        ('answers_collapsed',           'Answers replaced by representative'),
        ('kept_uncollapsed_oversized',  'Kept verbatim (oversized cluster)'),
        ('dup_created_by_collapse',     'Duplicates created by collapsing, removed'),
    ]),
    ('Output', [
        ('output_pairs',                'Pairs written'),
        ('output_unique_answers',       'Unique answers remaining'),
        ('long_answers',                'Flagged is_long (kept, not dropped)'),
    ]),
]
LABELS = {k: label for _, rows in REPORT_SECTIONS for k, label in rows}
W = 44

print('\n' + '=' * 70)
print('Step 03 · Deduplication finished')
print('=' * 70)

for section, rows in REPORT_SECTIONS:
    print(f'\n{section}')
    for key, label in rows:
        print(f'  {label:<{W}s} {stats[key]:>9,}')

# 兜底：以后新增计数项但忘了登记，也不会被静默吞掉
for key in sorted(set(stats) - set(LABELS)):
    print(f'  {key:<{W}s} {stats[key]:>9,}   <- add to REPORT_SECTIONS')

print(f'\n  {stats["input_pairs"]:,} -> {stats["output_pairs"]:,} pairs')
print(f'\nOutput         : {OUTPUT_PATH}')
print(f'Cluster members: {MEMBERS_PATH}')

print('\nPer-source counts')
for s, n in Counter(r['source'] for r in out).most_common():
    print(f'  {s:<16s} {n:>9,}')

# %%
# =============================================================================
# 格式检查：确认能直接交给 sentence-transformers
# =============================================================================
# ST 按列的位置把数据交给损失函数，所以 anchor、positive 必须是前两列，
# 且都是非空字符串。不满足就在这里报错，而不是等到训练时才发现。
#
# 另外数一下开头 10 个字里仍有"你好/您好"的答案。这一项只做提示、不报错：
# 它用来确认 01 的寒暄规则是否生效，数字应接近 0。

n_bad_text = sum(
    1 for r in out
    if not (isinstance(r.get('anchor'), str) and r['anchor'].strip()
            and isinstance(r.get('positive'), str) and r['positive'].strip())
)
n_bad_order = sum(1 for r in out if list(r)[:2] != ['anchor', 'positive'])

GREETING_LEFT_RE = re.compile(r'^.{0,8}?(你好|您好)')
greeting_left = Counter(r['source'] for r in out if GREETING_LEFT_RE.match(r['positive']))
greeting_left_examples = {}
for r in out:
    if r['source'] not in greeting_left_examples and GREETING_LEFT_RE.match(r['positive']):
        greeting_left_examples[r['source']] = r['positive'][:40]

print('\n' + '=' * 70)
print('Format check (sentence-transformers)')
print('=' * 70)
print(f'\n  {"Empty or non-string anchor / positive":<{W}s} {n_bad_text:>9,}')
print(f'  {"First two keys not (anchor, positive)":<{W}s} {n_bad_order:>9,}')

print(f'\n  Answers still opening with a greeting (first 10 chars, informational)')
for s in sorted({r['source'] for r in out}):
    print(f'    {s:<16s} {greeting_left.get(s, 0):>9,}')
for s, ex in greeting_left_examples.items():
    print(f'    e.g. [{s}] {ex}')

if n_bad_text or n_bad_order:
    raise ValueError('Format check failed; see the counts above.')

try:
    from datasets import load_dataset
except ImportError:
    load_test = 'skipped: datasets not installed'
    print('\n  `datasets` is not installed; skipped the load test.')
else:
    ds = load_dataset('json', data_files=OUTPUT_PATH, split='train')
    ds = ds.select_columns(['anchor', 'positive'])
    load_test = f'ok: {ds.num_rows:,} rows, columns {ds.column_names}'
    print(f'\n  Loaded with datasets: {ds.num_rows:,} rows, columns {ds.column_names}')
    print('  OK — ready for MultipleNegativesRankingLoss after the 90/10 split.')

# %%
# =============================================================================
# 阈值校准
# =============================================================================

def inspect(k=8, max_chars=110, seed=0):
    """
    随机抽查多元素簇，打印出来并返回抽到的样本。
    """
    rng = random.Random(seed)
    multi_c = [c for c, idxs in members.items() if len(idxs) > 1]
    if not multi_c:
        print('No multi-member clusters; THRESHOLD may be too high.')
        return []

    print('\n' + '=' * 70)
    print(f'Threshold calibration — {len(multi_c):,} multi-member clusters, '
          f'showing {min(k, len(multi_c))}')
    print('=' * 70)

    picked = []
    for c in rng.sample(multi_c, min(k, len(multi_c))):
        idxs  = rng.sample(members[c], min(3, len(members[c])))
        texts = [doc_text[doc_list[i]] for i in idxs]
        state = 'NOT collapsed (oversized)' if c in oversized else 'collapsed'
        print(f'\n-- cluster {c} ({len(members[c])} answers, {state})')
        for t in texts:
            print(f'   - {t[:max_chars]}...')
        picked.append({'cluster_id': int(c), 'size': len(members[c]),
                       'collapsed': c not in oversized,
                       'samples': [t[:200] for t in texts]})
    return picked


calibration_samples = inspect(k=8)

# %%
# =============================================================================
# 汇总报告
# =============================================================================

REPORT_DIR = f'{OUT_DIR}/reports'
os.makedirs(REPORT_DIR, exist_ok=True)


def md_table(headers, rows, numeric=True):
    """Markdown 表格。numeric=True 时第一列左对齐、其余右对齐，适合数字表。"""
    esc = lambda v: str(v).replace('|', '\\|').replace('\n', ' ')
    align = [':---'] + [('---:' if numeric else ':---')] * (len(headers) - 1)
    lines = ['| ' + ' | '.join(map(esc, headers)) + ' |',
             '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(map(esc, r)) + ' |' for r in rows]
    return '\n'.join(lines)


def file_info(path):
    """行数和 md5。md5 用来核对：同样的输入和参数重跑一遍，输出应该完全一致。"""
    h, n = hashlib.md5(), 0
    with open(path, 'rb') as f:
        for line in f:
            h.update(line)
            n += 1
    return {'path': path, 'lines': n, 'md5': h.hexdigest()}


def fmt(v):
    return f'{v:,}' if isinstance(v, int) and not isinstance(v, bool) else str(v)


try:
    from importlib.metadata import version
    datasketch_version = version('datasketch')
except Exception:
    datasketch_version = 'unknown'

report = {
    'step':         '03_dedup',
    'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'python':       sys.version.split()[0],
    'datasketch':   datasketch_version,
    'config': {
        'NUM_PERM':                NUM_PERM,
        'SHINGLE_K':               SHINGLE_K,
        'THRESHOLD':               THRESHOLD,
        'MERGE_TO_REPRESENTATIVE': MERGE_TO_REPRESENTATIVE,
        'MAX_MERGE_SIZE':          MAX_MERGE_SIZE,
        'MIN_SHINGLE_DOC_LEN':     MIN_SHINGLE_DOC_LEN,
        'LONG_ANSWER_CHARS':       LONG_ANSWER_CHARS,
    },
    'stats':                     {**{k: stats[k] for k in LABELS}, **stats},
    'verification_rate':         round(verify_rate, 4),
    'cluster_size_distribution': {str(k): v for k, v in sorted(size_dist.items())},
    'per_source':                dict(Counter(r['source'] for r in out)),
    'format_check': {
        'empty_or_non_string': n_bad_text,
        'wrong_column_order':  n_bad_order,
        'load_test':           load_test,
        'greeting_left':       dict(greeting_left),
    },
    'calibration_samples': calibration_samples,
    'outputs': {'deduped': file_info(OUTPUT_PATH), 'cluster_members': file_info(MEMBERS_PATH)},
}

size_rows = [[str(k), fmt(v)] for k, v in sorted(size_dist.items()) if k <= 10]
n_big = sum(v for k, v in size_dist.items() if k > 10)
if n_big:
    size_rows.append(['> 10', fmt(n_big)])

md = [
    '# Step 03 · Near-duplicate deduplication report',
    '',
    f'Generated {report["generated_at"]} · Python {report["python"]} · '
    f'datasketch {report["datasketch"]}',
    '',
    '## Configuration',
    '',
    md_table(['Parameter', 'Value'],
             [[k, v] for k, v in report['config'].items()], numeric=False),
    '',
    '> 用法：方法部分写清 MinHash 参数（排列数、字符 n-gram 长度、Jaccard 阈值），'
    '以及 LSH 候选对会先用 MinHash 估计的 Jaccard 复核再合并。',
    '',
    '## Processing statistics',
    '',
    md_table(['Stage', 'Item', 'Count'],
             [[section, label, fmt(stats[key])]
              for section, rows in REPORT_SECTIONS for key, label in rows]),
    '',
    f'Verification rate of LSH candidate pairs: {verify_rate:.1%} '
    f'({stats["verified_pairs_merged"]:,} of {stats["lsh_candidate_pairs"]:,}).',
    '',
    '',
    '## Cluster size distribution',
    '',
    md_table(['Answers per cluster', 'Clusters'], size_rows),
    '',
    '## Records per source after deduplication',
    '',
    md_table(['Source', 'Pairs'],
             [[s, fmt(n)] for s, n in sorted(report['per_source'].items(),
                                             key=lambda x: -x[1])]),
    '',
    '## Format check (sentence-transformers)',
    '',
    md_table(['Check', 'Result'], [
        ['Empty or non-string anchor / positive', fmt(n_bad_text)],
        ['First two keys not (anchor, positive)', fmt(n_bad_order)],
        ['Load test with `datasets`',             load_test],
    ], numeric=False),
    '',
    'Answers still opening with a greeting (first 10 characters; informational):',
    '',
    md_table(['Source', 'Answers'],
             [[s, fmt(greeting_left.get(s, 0))] for s in sorted(report['per_source'])]),
    '',
    '',
    '## Threshold calibration samples',
    '',
    'Random multi-member clusters (seed 0); each line is one answer, truncated to 200 characters.',
    '',
]
for s in calibration_samples:
    state = 'collapsed' if s['collapsed'] else 'not collapsed (oversized)'
    md += [f'**Cluster {s["cluster_id"]}** — {s["size"]} answers, {state}', '']
    md += ['- ' + ' '.join(t.split()) for t in s['samples']]
    md += ['']
md += ['> 用法：阈值校准过程放附录——抽查了多少个簇、是否都是真重复、最终阈值是多少。',
       '',
       '## Output files',
       '',
       md_table(['File', 'Lines', 'md5'],
                [[i['path'], fmt(i['lines']), i['md5']] for i in report['outputs'].values()]),
       '',
       '> md5 用来核对：同样的输入和参数重跑一遍，md5 应该完全一样。']

md_path, json_path = f'{REPORT_DIR}/03_dedup.md', f'{REPORT_DIR}/03_dedup.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(f'\nReport: {md_path}')
print(f'        {json_path}')

# %%
# =============================================================================
# 全流程汇总：reports/summary.md + reports/summary.json
# =============================================================================

def load_report(name):
    path = f'{REPORT_DIR}/{name}.json'
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def get(d, *path):
    """按路径取值，任何一层缺失就返回 None"""
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt_opt(v):
    return '—' if v is None else fmt(v)


def total(values):
    return None if any(v is None for v in values) else sum(values)


r00 = load_report('00_concat')
r02 = load_report('02_merge')
r03 = report
sources = list(r02['config']['SOURCES']) if r02 else sorted(r03['per_source'])
r01 = {s: load_report(f'01_convert_{s}') for s in sources}

missing_reports = ([name for name, r in [('00_concat', r00), ('02_merge', r02)] if r is None]
                   + [f'01_convert_{s}' for s, r in r01.items() if r is None])
raw_by_source = {x['source']: x['records'] for x in r00['sources']} if r00 else {}

# ---------- 每一步剩多少 ----------
FUNNEL = [
    ('Raw records (00)',                lambda s: raw_by_source.get(s)),
    ('Question strings (01)',           lambda s: get(r01[s], 'accounting', 'in_input')),
    ('  dropped (01)',                  lambda s: get(r01[s], 'accounting', 'dropped')),
    ('  routed to ontology pool (01)',  lambda s: get(r01[s], 'accounting', 'ontology')),
    ('QA pairs after conversion (01)',  lambda s: get(r01[s], 'accounting', 'finetune')),
    ('After cross-source dedup (02)',   lambda s: get(r02, 'per_source', s, 'kept')),
    ('After near-duplicate dedup (03)', lambda s: r03['per_source'].get(s, 0)),
]
funnel_rows, funnel_json = [], {}
for label, fn in FUNNEL:
    vals = [fn(s) for s in sources]
    funnel_rows.append([label, *map(fmt_opt, vals), fmt_opt(total(vals))])
    funnel_json[label.strip()] = {**dict(zip(sources, vals)), 'total': total(vals)}

# ---------- 文本清洗影响了多少 ----------
# 键名必须与 01 的 CLEANING_RULES 对应（stats 里是 mod_ 前缀）
CLEAN = [('mod_html_entity',     'HTML entities unescaped'),
         ('mod_greeting',        'Greetings stripped'),
         ('mod_template_header', 'Template headers removed'),
         ('mod_question_echo',   'Question echo removed')]
clean_rows = [[label, *[fmt_opt(get(r01[s], 'stats', key)) for s in sources]]
              for key, label in CLEAN]

# ---------- 各原因丢弃了多少（事件数，口径与 01 报告的统计表一致）----------
reasons, reason_label = [], {}
for s in sources:
    for d in get(r01[s], 'dropped_by_reason') or []:
        if d['reason'] not in reason_label:
            reasons.append(d['reason'])
            reason_label[d['reason']] = d['label']
events_by = {s: {d['reason']: d['events'] for d in get(r01[s], 'dropped_by_reason') or []}
             for s in sources}
drop_rows = [[reason_label[r], *[fmt(events_by[s].get(r, 0)) if r01[s] else '—' for s in sources]]
             for r in reasons]

# ---------- 关键参数（01 的参数在各来源之间应该一致，不一致会标出来）----------
cfg01 = {s: get(r01[s], 'config') for s in sources if r01[s]}
param_rows = []
if cfg01:
    all_keys = []
    for c in cfg01.values():
        all_keys += [k for k in c if k not in all_keys]
    for k in all_keys:
        if k == 'INPUT_PATH':
            continue
        vals = {str(c.get(k)) for c in cfg01.values()}
        value = next(iter(vals)) if len(vals) == 1 else \
            'DIFFERS: ' + ', '.join(f'{s}={c.get(k)}' for s, c in cfg01.items())
        param_rows.append(['01', k, value])
param_rows += [['03', k, v] for k, v in r03['config'].items()]

# ---------- 段落草稿 ----------
tot = {label.strip(): total([fn(s) for s in sources]) for label, fn in FUNNEL}
if missing_reports or None in tot.values():
    draft = '(Draft not generated: some step reports are missing.)'
else:
    draft = (
        f'We collected question–answer pairs from {len(sources)} public datasets '
        f'({", ".join(sources)}). After format conversion and text cleaning, '
        f'{tot["QA pairs after conversion (01)"]:,} QA pairs remained; '
        f'{tot["routed to ontology pool (01)"]:,} questions with short answers were set aside '
        f'for ontology construction and {tot["dropped (01)"]:,} were discarded, with every '
        f'input question accounted for. Merging the sources and removing exact cross-source '
        f'duplicates left {tot["After cross-source dedup (02)"]:,} pairs. Near-duplicate answers '
        f'were then clustered with MinHash LSH ({NUM_PERM} permutations, character '
        f'{SHINGLE_K}-gram shingles, Jaccard threshold {THRESHOLD}); every LSH candidate pair '
        f'was verified against its MinHash Jaccard estimate before merging. This grouped '
        f'{stats["unique_answers"]:,} unique answers into {stats["clusters"]:,} clusters. '
        f'The final dataset contains {stats["output_pairs"]:,} QA pairs over '
        f'{stats["output_unique_answers"]:,} unique answers.'
    )

summary = {
    'generated_at':    datetime.datetime.now().isoformat(timespec='seconds'),
    'sources':         sources,
    'missing_reports': missing_reports,
    'funnel':          funnel_json,
    'cleaning':        {label: {s: get(r01[s], 'stats', key) for s in sources}
                        for key, label in CLEAN},
    'dropped_events':  {reason_label[r]: {s: events_by[s].get(r, 0) for s in sources}
                        for r in reasons},
    'parameters':      [{'step': a, 'parameter': b, 'value': c} for a, b, c in param_rows],
    'draft_paragraph': draft,
}

md = ['# Data pipeline summary', '',
      f'Generated {summary["generated_at"]}. Sources in deduplication order: '
      f'{", ".join(sources)}.', '']
if missing_reports:
    md += ['> **Missing reports:** ' + ', '.join(missing_reports)
           + ' — the corresponding cells show "—".', '']
md += [
    '## Records remaining at each step',
    '',
    md_table(['Step', *sources, 'Total'], funnel_rows),
    '',
    '> 用法：数据统计表（dataset statistics）。"dropped" 和 "routed to ontology pool" '
    '两行是从 "Question strings" 里分出去的；02、03 两行是在 01 输出基础上继续去重后的结果。',
    '',
    '## Text cleaning (answers modified, record kept)',
    '',
    md_table(['Rule', *sources], clean_rows),
    '',
    '## Dropped items by reason (events)',
    '',
    md_table(['Reason', *sources], drop_rows) if drop_rows else 'No dropped items.',
    '',
    '> 用法：数据清洗表。每条丢弃记录的原文都在 `dropped/{source}.jsonl` 里，可以随时抽查。',
    '',
    '## Key parameters',
    '',
    md_table(['Step', 'Parameter', 'Value'], param_rows, numeric=False),
    '',
    '> 出现 DIFFERS 说明各来源跑 01 时参数不一致。',
    '',
    '## Draft paragraph for the paper',
    '',
    draft,
    '',
]

md_path, json_path = f'{REPORT_DIR}/summary.md', f'{REPORT_DIR}/summary.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(f'\nPipeline summary: {md_path}')
print(f'                  {json_path}')
if missing_reports:
    print(f'  Missing reports: {", ".join(missing_reports)}')
