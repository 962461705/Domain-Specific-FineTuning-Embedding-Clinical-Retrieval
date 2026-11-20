# %% [markdown]
# # Step 08 (new) · 每种方法一份训练文件
#
# 所有方法的问答对、基础模型、损失和超参数都相同，只是每条数据配的难负例不同。
# 原 08 不动，输出写到 training_new/。
#
# | Method | Hard negative | 是否依赖模型候选 |
# |---|---|---|
# | mnrl          | 无，只用 in-batch 负例 | — |
# | bm25          | BM25 第 1 名 | 是（07） |
# | dense         | Dense 第 1 名 | 是（07） |
# | dense_rank10  | Dense 第 10 名 | 是（07） |
# | dense_rank100 | Dense 第 100 名 | 是（07） |
# | random        | 训练答案库随机取一条（不同连通分量） | 否 |
# | margin        | Dense 前 100 名，去掉得分过高的，在最难前 POOL 个里随机取 | 是（07） |
# | margin_onto   | 候选池与 margin 相同，只多一道本体过滤 | 是（07） |
# | onto_pool_sim      | 本体池里按实体 IDF 重合度排名，在最相关前 POOL 个里随机取；无标签退回 random | 否 |
# | onto_pool_rand| 与 onto_pool_sim 完全相同的候选池，随机取一个（onto_pool_sim 的无分数对照） | 否 |
#
# onto_pool_sim / onto_pool_rand 设计要点：
# - 候选池 = 本体判为"不同病"的训练答案，含 different（跨大类）和 sibling（同大类另一种病，如脑梗死对脑出血），
#   排除 same（同病）、unnamed（正文没点名任何病）、same component；问题无标签时退回 random。
# - 热门答案上限：每个答案作为负例最多出现 MAX_HOT 次，超出后排到次选；确保 distinct 不低于 margin。
# - 泛化长答案排除：实体数 >= GENERIC_ENTITY_THR 且答案点名病种数 >= GENERIC_CONCEPT_THR 的答案排除（如"神经科常见病症状汇总"）。
# - onto_pool_sim 排序：被候选答案提到的、属于"问题这种病资料"的实体，IDF 加权和；同分时随机打破（种子与 qid 相关）。
# - onto_pool_sim 取分数最高的前 POOL 个里随机取一个；onto_pool_rand 在同一池里随机取一个。
# - bge 和 qwen 共用同一份 onto_pool_sim / onto_pool_rand 训练文件（不依赖模型）。
#
# 盲评表（audit）：只从各方法选了不同负例的问题里抽，包含 dense_rank10 / margin / onto_pool_sim / onto_pool_rand / random，
# 各方法混排，有 y / n / ? 三档，用来对比各方法的假负例率。
#
# 用法：
#   python 08_training_sets_new.py                       # 两个模型都生成
#   python 08_training_sets_new.py --models bge-base     # 只生成 bge
#   python 08_training_sets_new.py --pool 5              # 最难前几个里取（默认 5）
# Jupyter 里逐格运行即可。

# %%
import os, sys
import argparse, csv, datetime, hashlib, json, math, random, re, statistics, unicodedata
from collections import Counter, defaultdict

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw):
        total = kw.get('total', None)
        desc  = kw.get('desc', '')
        try:
            n = total or len(it)  # type: ignore
        except TypeError:
            n = '?'
        print(f'{desc} (n={n}) ...', flush=True)
        return it

NL = chr(10)


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.environ.get('NEURO_OUT_DIR', '/data/neuro/data/neuro_dataset'))
    ap.add_argument('--tag', default='new')
    ap.add_argument('--suite', default='main', choices=['main', 'core', 'dose'],
                    help='main=主线方法，输出后缀 new；core=本体打分挑难类方法，输出后缀 new_core')
    ap.add_argument('--models', default='bge-base,qwen3-0.6b')
    ap.add_argument('--pool', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-hot', type=int, default=300,
                    help='每个答案作为 onto_pool_sim/onto_pool_rand 负例的上限次数')
    ap.add_argument('--generic-entity-thr', type=int, default=8,
                    help='实体数 >= 此值 且点名病种数 >= generic-concept-thr 的答案视为泛化答案，从本体池排除')
    ap.add_argument('--generic-concept-thr', type=int, default=3)
    ap.add_argument('--workers', type=int, default=30,
                    help='并行进程数（0=串行）；Linux fork 会继承全局变量，推荐 30~50')
    a, _unknown = ap.parse_known_args()
    return a


ARGS = _args()
OUT_DIR = ARGS.out_dir
TAG = ARGS.tag
SPLIT_DIR = f'{OUT_DIR}/split'
ONTO_DIR = f'{OUT_DIR}/ontology_{TAG}'
CAND_DIR = f'{OUT_DIR}/candidates_{TAG}'
WORKERS = ARGS.workers
SUITE = ARGS.suite
OUT_TAG = TAG if SUITE == 'main' else TAG + '_' + SUITE
TRAIN_DIR = f'{OUT_DIR}/training_{OUT_TAG}'
REPORT_DIR = f'{OUT_DIR}/reports'

MODEL_KEYS = [m for m in ARGS.models.split(',') if m]
POOL = ARGS.pool
SEED = ARGS.seed
MAX_HOT = ARGS.max_hot
GENERIC_ENTITY_THR = ARGS.generic_entity_thr
GENERIC_CONCEPT_THR = ARGS.generic_concept_thr

DEFAULT_MARGIN = 0.95
DEFAULT_MAX_POS = 3
SKIP_RANK = 10
N_EXAMPLES = 8

SINGLE_METHODS = ['mnrl', 'bm25', 'dense', 'dense_rank10', 'dense_rank100',
                  'random', 'margin', 'margin_onto']
if SUITE == 'core':
    SINGLE_METHODS = SINGLE_METHODS + ['bm25_onto', 'dense_onto']
MAIN_MULTI = ['onto_pool_rand', 'onto_pool_strat', 'onto_strat_loose', 'onto_loose_rand']
CORE_MULTI = ['onto_pool_sim', 'onto_pool_hard', 'onto_hard_loose', 'onto_pool_mid', 'onto_nmid_loose', 'onto_hmid_loose']
ALL_MULTI = MAIN_MULTI + CORE_MULTI
DOSE_MULTI = ['onto_dose_p10', 'onto_dose_p30', 'onto_dose_p50']
MULTI_METHODS_ALL = (MAIN_MULTI if SUITE == 'main' else
                     ['onto_pool_rand'] + DOSE_MULTI if SUITE == 'dose' else
                     ['onto_pool_rand'] + CORE_MULTI)
CMP_ALT = {'main': 'onto_loose_rand', 'core': 'onto_pool_sim', 'dose': 'onto_dose_p30'}[SUITE]
# 每次运行只计算本 suite 要写出的多负例方法。以前 _select_one 每次都把全部方法算一遍，
# 没写出的直接丢掉：结果相同，只是慢很多。onto_pool_sim 与 onto_pool_rand 共用热门计数 _hot_count，
# 且先于 onto_pool_rand 计算，所以三个 suite 都照常计算它，保证 onto_pool_rand 选出的负例不变。
MULTI_NEEDED = set(MULTI_METHODS_ALL) | {'onto_pool_sim', 'onto_pool_rand'}
N_AUDIT = 100
AUDIT_METHODS = ['dense_rank10', 'margin', 'onto_pool_rand', CMP_ALT, 'random']
LABEL_COL = 'candidate_also_answers_question (y/n/?)'
REL_KEYS = ['same', 'sibling', 'different', 'unnamed', 'unknown']

# onto_pool_sim / onto_pool_rand 可用的关系层级（sibling = 同大类另一种病，如脑梗死对脑出血）
ONTO_POOL_RELS = {'different', 'sibling'}

METHODS = ['mnrl', 'bm25', 'dense', 'dense_rank10', 'dense_rank100',
           'random', 'margin', 'margin_onto'] + (['bm25_onto', 'dense_onto'] if SUITE == 'core' else [])
HARD_NEG = {
    'bm25_onto':          'BM25 top candidate after removing same-disease and unnamed answers',
    'dense_onto':         'Dense top candidate after removing same-disease and unnamed answers',
    'onto_dose_p10':      'safe pool; 10% of samples draw a same-disease (false) negative instead',
    'onto_dose_p30':      'safe pool; 30% of samples draw a same-disease (false) negative instead',
    'onto_dose_p50':      'safe pool; 50% of samples draw a same-disease (false) negative instead',
    'mnrl':               'none (in-batch only)',
    'bm25':               'BM25 rank 1',
    'dense':              'Dense rank 1',
    'dense_rank10':       f'Dense rank {SKIP_RANK}',
    'dense_rank100':      'Dense rank 100',
    'random':             'random answer, different component',
    'margin':             f'Dense top-100 under score limit; random among hardest {POOL}',
    'margin_onto':        'same as margin, minus same-disease and unnamed answers',
    'onto_pool_sim':            'ontology safe pool; entity cos+aspect ranked, low-band sample',
    'onto_pool_strat': 'ontology safe pool; family-stratified uniform random, no scoring',
    'onto_strat_loose': 'loose pool (excl same only); family-stratified uniform random, no scoring',
    'onto_loose_rand': 'loose pool (excl same only); uniform random, no stratification, no scoring',
    'onto_pool_mid': 'ontology safe pool; family-stratified, 2-gram median band 30-70%, pos-aware cap',
    'onto_nmid_loose': 'loose pool; family-stratified, 2-gram median band 30-70%, pos-aware cap',
    'onto_hmid_loose': 'loose pool; family-stratified, formula median band 30-70%, pos-aware cap',
    'onto_pool_rand':      'ontology safe pool; purely random',
    'onto_pool_hard':      'ontology safe pool; formula=0.75·J_lex+0.10·B_rel+0.15·J_ent, pos-aware cap',
    'onto_hard_loose':     'loose pool; formula=0.75·J_lex+0.10·B_rel+0.15·J_ent, pos-aware cap',
}

os.makedirs(REPORT_DIR, exist_ok=True)

# %%
# =============================================================================
# 基础函数（normalize / qid_of 与 01–04、06、07 完全一致）
# =============================================================================


def normalize(text):
    text = unicodedata.normalize('NFKC', text or '').lower()
    return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text)


def qid_of(nq):
    return hashlib.md5(nq.encode()).hexdigest()[:16]


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + NL)


def md_table(headers, rows, numeric=True):
    esc = lambda v: str(v).replace('|', '/').replace(NL, ' ')
    align = [':---'] + [('---:' if numeric else ':---')] * (len(headers) - 1)
    lines = ['| ' + ' | '.join(map(esc, headers)) + ' |',
             '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(map(esc, r)) + ' |' for r in rows]
    return NL.join(lines)


def fmt(v, spec):
    return '-' if v is None else format(v, spec)

# %%
# =============================================================================
# 读入训练数据和答案库
# =============================================================================

train = read_jsonl(f'{SPLIT_DIR}/train.jsonl')
corpus = {r['doc_id']: r['text'] for r in read_jsonl(f'{SPLIT_DIR}/train_corpus.jsonl')}
doc_ids = sorted(corpus)

# 预计算答案字符 2-gram 集合（onto_hard 难度打分用，不依赖模型）
_doc_2gram = {}
for _d in doc_ids:
    _n = normalize(corpus[_d])
    _doc_2gram[_d] = frozenset(_n[i:i+2] for i in range(len(_n) - 1)) if len(_n) > 1 else frozenset()
print('  Doc 2-gram precomputed          : {:,}'.format(len(_doc_2gram)))
row_qid = [qid_of(normalize(r['anchor'])) for r in train]

qtext, q_comp, doc_comp = {}, {}, {}
positives = defaultdict(set)
for r, q in zip(train, row_qid):
    qtext.setdefault(q, r['anchor'])
    q_comp.setdefault(q, r['component_id'])
    doc_comp.setdefault(r['doc_id'], r['component_id'])
    positives[q].add(r['doc_id'])

# BM25 候选（两个模型共用）
bm25_path = f'{CAND_DIR}/bm25.jsonl'
assert os.path.exists(bm25_path), bm25_path + ' not found — run 07_get_neg_new.py --model bge-base first'
bm25 = {r['qid']: r for r in read_jsonl(bm25_path)}

# 06 本体输出
for name in ('concepts.json', 'question_labels.jsonl', 'doc_labels.jsonl', 'entities.json'):
    assert os.path.exists(f'{ONTO_DIR}/{name}'), f'{ONTO_DIR}/{name} not found — run 06_本体标注_neuro_new.py first'

with open(f'{ONTO_DIR}/concepts.json', encoding='utf-8') as f:
    _onto = json.load(f)
concepts = _onto['concepts']
assert all('family' in c for c in concepts), 'concepts.json missing family — re-run 06'

with open(f'{ONTO_DIR}/entities.json', encoding='utf-8') as f:
    _ents = json.load(f)
idf = {e['id']: e['idf'] for e in _ents['entities']}
entity_by_id = {e['id']: e for e in _ents['entities']}
# profile[concept_id] = set of entity ids (used=True)
_rel_list = _ents['relations']
profile = defaultdict(set)
for r in _rel_list:
    if r['used']:
        profile[r['concept']].add(r['entity'])

# ASPECT_ENTITY_TYPES：从 concepts.json 的 aspect_entity_types 字段加载
ASPECT_ENTITY_TYPES = {k: set(v) for k, v in _onto.get('aspect_entity_types', {}).items()}

parent = {c['id']: c['parent'] for c in concepts}
family = {c['id']: c['family'] for c in concepts}
cname = {c['id']: c['name'] for c in concepts}

# 概念血缘和 rel_set（与 07 完全一致）
def _lineage(c):
    out = {c}
    while parent.get(c):
        c = parent[c]
        out.add(c)
    return out

lin = {c: _lineage(c) for c in parent}
rel_set = {x: {c for c in parent if c in lin[x] or x in lin[c]} for x in parent}

# related_concepts[c]：与 c 相同或互为上下位的概念集合（用来取"问题这种病的资料"）
related_concepts = {c: {x for x in parent if x in lin[c] or c in lin[x]} for c in parent}

q_con = {x['qid']: set(x['concepts']) for x in read_jsonl(f'{ONTO_DIR}/question_labels.jsonl')}
EMPTY = (frozenset(), frozenset(), False, set())
doc_sets = {}
for x in read_jsonl(f'{ONTO_DIR}/doc_labels.jsonl'):
    allc = set(x['mentioned']) | set(x['inherited'])
    doc_ents = set(x.get('entities', {}).keys()) if isinstance(x.get('entities'), dict) else set()
    doc_sets[x['doc_id']] = (frozenset(allc), frozenset(family[c] for c in allc if c in family),
                             bool(x['mentioned']), set(x['mentioned']), doc_ents,
                             len(x['mentioned']))

print('=' * 70)
print('Step 08 (new) · Training sets')
print('=' * 70)
print(NL + '  Training pairs        : {:,}'.format(len(train)))
print('  Unique questions      : {:,}'.format(len(qtext)))
print('  Training answer pool  : {:,}'.format(len(corpus)))

# %%
# =============================================================================
# 疾病关系（与 07 完全一致）
# =============================================================================

def make_context(max_pos):
    q_ctx, src = {}, Counter()
    for qid in qtext:
        cs, how = q_con.get(qid), 'question'
        if not cs:
            ment = set()
            for d in positives[qid]:
                ment |= doc_sets.get(d, EMPTY)[3]
            if 0 < len(ment) <= max_pos:
                cs, how = ment, 'positive'
            else:
                cs, how = set(), 'none'
        src[how] += 1
        if cs:
            q_ctx[qid] = (set().union(*(rel_set[c] for c in cs)),
                          {family[c] for c in cs if c in family}, how)
    return q_ctx, src


def relation(qid, doc, q_ctx):
    ctx = q_ctx.get(qid)
    if ctx is None:
        return 'unknown'
    q_rel, q_fam, _ = ctx
    d_all, d_fam, named, _, _, _ = doc_sets.get(doc, EMPTY)
    if d_all & q_rel:
        return 'same'
    if not named:
        return 'unnamed'
    return 'sibling' if d_fam & q_fam else 'different'


def cap_of(row, margin):
    top = max(row['pos_scores'].values())
    return top - (1 - margin) * abs(top)


def names(ids):
    return '、'.join(sorted(cname[c] for c in ids if c in cname)) or '-'


def load_07_config(model):
    path = f'{REPORT_DIR}/07_candidates_{model}_{TAG}.json'
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            c = json.load(f)['config']
        return c.get('MARGIN', DEFAULT_MARGIN), c.get('MAX_POS_CONCEPTS', DEFAULT_MAX_POS)
    print('  warning: {} not found, using defaults'.format(path))
    return DEFAULT_MARGIN, DEFAULT_MAX_POS

# %%
# =============================================================================
# onto_pool_sim / onto_pool_rand 候选池（不依赖任何模型候选文件）
# =============================================================================

# 泛化长答案：点名多种病 + 大量实体，从本体池排除
generic_docs = {d for d, v in doc_sets.items()
                if len(v[4]) >= GENERIC_ENTITY_THR and v[5] >= GENERIC_CONCEPT_THR}
print('  generic answers excluded from onto pool : {:,}'.format(len(generic_docs)))

# 候选池缓存
_onto_pool_cache = {}


def onto_pool(q_rel, q_fam):
    key = (frozenset(q_rel), frozenset(q_fam))
    if key not in _onto_pool_cache:
        out_diff, out_sib = [], []
        for d in doc_ids:
            if d in generic_docs:
                continue
            d_all, d_fam, named, _, _, _ = doc_sets.get(d, EMPTY)
            if not named or (d_all & q_rel):
                continue
            if d_fam & q_fam:
                out_sib.append(d)
            else:
                out_diff.append(d)
        _onto_pool_cache[key] = (out_diff, out_sib)
    return _onto_pool_cache[key]


# 答案方面：继承配对训练问题的方面（多数派），08 直接用，不依赖 06 的 doc_aspect
_doc_aspect_08 = {}
for _r, _qid in zip(train, row_qid):
    _asp = q_con.get(_qid)   # 用概念而非方面，因为 08 不导入方面函数
    # 这里只需要问题的 aspect 标签，但 08 没有直接访问 q_info，
    # 所以从 question_labels.jsonl 重新读
_q_info_08 = {x['qid']: x for x in read_jsonl(f'{ONTO_DIR}/question_labels.jsonl')}
for _r, _qid in zip(train, row_qid):
    _asp = _q_info_08.get(_qid, {}).get('aspect')
    if _asp:
        _doc_aspect_08.setdefault(_r['doc_id'], Counter())[_asp] += 1
_doc_aspect_08 = {d: c.most_common(1)[0][0] for d, c in _doc_aspect_08.items()}


def onto_cos_score_08(q_qid, q_ctx_row, doc_id):
    '''
    归一化 IDF 余弦 + 方面匹配系数。
    q_vec[e] = idf[e] 若 e 在"问题这种病的资料实体"中
    d_vec[e] = idf[e] 若 e 在答案实体中
    cos = dot / (||q|| * ||d||)
    方面相同 × ASPECT_BOOST，方面不同 / ASPECT_BOOST，无方面信息 × 1
    '''
    ASPECT_BOOST = 2.5
    q_rel, q_fam, _ = q_ctx_row
    exp = set()
    for c in q_rel:
        exp |= profile.get(c, set())
    # 加方面过滤
    q_asp = _q_info_08.get(q_qid, {}).get('aspect')
    if q_asp and q_asp in ASPECT_ENTITY_TYPES:
        filtered = {e for e in exp if entity_by_id[e]['type'] in ASPECT_ENTITY_TYPES[q_asp]}
        if filtered:
            exp = filtered
    d_ent = set(doc_sets.get(doc_id, EMPTY)[4])
    if not exp or not d_ent:
        return 0.0
    dot    = sum(idf.get(e, 0.0) ** 2 for e in exp & d_ent)
    norm_q = math.sqrt(sum(idf.get(e, 0.0) ** 2 for e in exp))
    norm_d = math.sqrt(sum(idf.get(e, 0.0) ** 2 for e in d_ent))
    cos    = dot / (norm_q * norm_d + 1e-9)
    d_asp  = _doc_aspect_08.get(doc_id)
    if q_asp and d_asp:
        cos = min(1.0, cos * ASPECT_BOOST) if q_asp == d_asp else cos / ASPECT_BOOST
    return cos


def difficulty_score(q_qid, q_ctx_row, doc_id, rel_label):
    '''
    无模型难度打分公式：
      difficulty(q, d) = α·J_lex + β·B_rel + γ·J_ent

      J_lex  = |G₂(q) ∩ G₂(d)| / |G₂(q) ∪ G₂(d)|   字符 2-gram Jaccard（词面相似）
      B_rel  = sibling=1.0, unnamed=0.5, different=0.3  疾病关系近度（本体结构）
      J_ent  = |E_q ∩ E_d| / |E_q ∪ E_d|              本体实体 Jaccard（医学术语混淆）
      α=0.75, β=0.10, γ=0.15

    分数越高 = 候选与问题越像 = 负例越难 = 训练信号越强。
    正例感知 cap（与 margin 等价，只换打分器）：
      cap = max_{p∈Pos(q)} difficulty(q, p) × 0.85
    '''
    ALPHA, BETA, GAMMA = 0.75, 0.10, 0.15
    REL_SCORE = {'sibling': 1.0, 'unnamed': 0.5, 'different': 0.5, 'unknown': 0.3, 'same': 0.5}

    # J_lex：字符 2-gram Jaccard
    q_2g = _q_2gram_cache.get(q_qid)
    d_2g = _doc_2gram.get(doc_id, frozenset())
    if q_2g and d_2g:
        inter = len(q_2g & d_2g)
        union = len(q_2g | d_2g)
        j_lex = inter / union if union else 0.0
    else:
        j_lex = 0.0

    # B_rel：疾病关系近度
    b_rel = REL_SCORE.get(rel_label, 0.1)

    # J_ent：本体实体 Jaccard
    q_rel, _, _ = q_ctx_row
    q_ent = set()
    for c in q_rel:
        q_ent |= profile.get(c, set())
    d_ent = set(doc_sets.get(doc_id, EMPTY)[4])
    if q_ent or d_ent:
        union_e = len(q_ent | d_ent)
        j_ent = len(q_ent & d_ent) / union_e if union_e else 0.0
    else:
        j_ent = 0.0

    return ALPHA * j_lex + BETA * b_rel + GAMMA * j_ent


# 问题字符 2-gram 缓存（build 时填充）
_q_2gram_cache = {}


# 热门计数（每次 build 重置）
_hot_count = Counter()

# onto_pool_sim 采样策略（v2）：
#   diff 候选：升序排列（低余弦优先），取前 SAMPLE_HI 分位。
#     本体已保证候选是"不同病"，低余弦 = 实体轮廓与该问题的病种资料重叠最少
#     = 最清晰的错误答案，训练信号干净。跳过余弦 = 0 的（无任何实体，等价随机）。
#   sibling 候选：降序排列，取前 SIBLING_QUOTA*4 个里随机。
#     同家族不同病（如脑梗死 vs 脑出血），取最相似的作为家族内对比，保留对比度。
SAMPLE_HI = 0.55   # diff 升序排列后取前 55%（低相似段）
SIBLING_QUOTA = 1  # 每题 sibling 最多几个


def pick_onto_multi(q, q_ctx, rng_key, use_sim, k):
    '''
    从本体池里取 k 个负例（不重复）。
    use_sim=True  → onto_pool_sim：diff 升序取低分段（清晰负例），sibling 降序取最相似
    use_sim=False → 纯随机（onto_pool_rand）
    无标签 → 取 k 个 random（不重复）
    返回 [(doc_id, rule), ...]
    '''
    if q not in _q_ctx_global:
        results = []
        rng = random.Random(rng_key + '_fb')
        seen = set()
        for _ in range(200 * k):
            d = doc_ids[rng.randrange(len(doc_ids))]
            if doc_comp.get(d) != q_comp[q] and d not in seen:
                seen.add(d); results.append((d, 'fallback: no disease label -> random'))
                if len(results) == k: break
        return results or [(doc_ids[0], 'fallback')]

    ctx = _q_ctx_global[q]
    q_rel, q_fam, _ = ctx
    pool_diff, pool_sib = onto_pool(q_rel, q_fam)

    # 自适应热门上限
    total_pool = len(pool_diff) + len(pool_sib)
    hot_cap = max(10, total_pool // 500) if total_pool else 10

    def eligible(pool):
        return [d for d in pool if doc_comp.get(d) != q_comp[q] and _hot_count[d] < hot_cap]

    elig_diff = eligible(pool_diff)
    elig_sib  = eligible(pool_sib)

    if not elig_diff and not elig_sib:
        # 放开热门限制
        elig_diff = [d for d in pool_diff if doc_comp.get(d) != q_comp[q]]
        elig_sib  = [d for d in pool_sib  if doc_comp.get(d) != q_comp[q]]

    if not elig_diff and not elig_sib:
        rng = random.Random(rng_key + '_empty')
        seen = set()
        results = []
        for _ in range(200 * k):
            d = doc_ids[rng.randrange(len(doc_ids))]
            if doc_comp.get(d) != q_comp[q] and d not in seen:
                seen.add(d); results.append((d, 'fallback: empty onto pool -> random'))
                if len(results) == k: break
        return results

    if use_sim:
        # diff：升序（低余弦优先）；跳过余弦=0的（无实体，等价随机）
        scored_diff_raw = [(onto_cos_score_08(q, ctx, d), d) for d in elig_diff]
        scored_diff = sorted([(s, d) for s, d in scored_diff_raw if s > 0.0])  # ascending
        # 若全部余弦=0则回退到全部候选（随机顺序）
        if not scored_diff:
            scored_diff = [(0.0, d) for d in elig_diff]
        # sibling：降序，取最相似的几个作家族内对比
        scored_sib = sorted([(onto_cos_score_08(q, ctx, d), d) for d in elig_sib], reverse=True)
        # diff 低分段：升序后取前 SAMPLE_HI 分位
        n_diff = len(scored_diff)
        hi_d = max(1, int(n_diff * SAMPLE_HI))
        band_diff = scored_diff[:hi_d]
        # sibling：取前 SIBLING_QUOTA * 4 个里随机
        band_sib = scored_sib[:max(SIBLING_QUOTA * 4, 4)] if scored_sib else []
        rng = random.Random(rng_key)
        chosen, seen_d = [], set()
        # 先加 sibling（最多 SIBLING_QUOTA 个）
        rng.shuffle(band_sib)
        for _, d in band_sib:
            if len([x for x in chosen if x[1].startswith('onto_sim: sibling')]) >= SIBLING_QUOTA:
                break
            if d not in seen_d:
                chosen.append((d, 'onto_sim: sibling, cos+aspect')); seen_d.add(d)
        # 再从低分段补 different
        rng.shuffle(band_diff)
        for _, d in band_diff:
            if len(chosen) >= k: break
            if d not in seen_d:
                chosen.append((d, 'onto_sim: different, cos+aspect, low-band')); seen_d.add(d)
        # 仍不足则从整个 elig_diff 随机补
        all_diff_shuf = list(elig_diff)
        rng.shuffle(all_diff_shuf)
        for d in all_diff_shuf:
            if len(chosen) >= k: break
            if d not in seen_d:
                chosen.append((d, 'onto_sim: different, fallback any')); seen_d.add(d)
    else:
        # onto_pool_rand：随机，同样的 sibling 配额
        rng = random.Random(rng_key + '_pr')
        combined = [(d, 'sib') for d in elig_sib] + [(d, 'dif') for d in elig_diff]
        rng.shuffle(combined)
        chosen, seen_d, n_sib = [], set(), 0
        for d, typ in combined:
            if len(chosen) >= k: break
            if d in seen_d: continue
            if typ == 'sib' and n_sib >= SIBLING_QUOTA: continue
            seen_d.add(d)
            if typ == 'sib': n_sib += 1
            chosen.append((d, 'onto_pool_rand: random from ontology pool'))

    for d, _ in chosen:
        _hot_count[d] += 1

    # 不足 k 个时用 random 补齐
    rng2 = random.Random(rng_key + '_pad')
    seen_d = {d for d, _ in chosen}
    while len(chosen) < k:
        d = doc_ids[rng2.randrange(len(doc_ids))]
        if doc_comp.get(d) != q_comp[q] and d not in seen_d:
            seen_d.add(d); chosen.append((d, 'pad: random'))
    return chosen[:k]


def _scored_pool(q, q_ctx_local, bm25_row, margin, rng_key, scorer, loose):
    '''
    通用打分负例选取函数。
    scorer: 'bm25' | 'ngram' | 'formula'
    loose:  False=安全池(排除same+unnamed+generic), True=宽松池(只排除same)
    返回 [(doc_id, rule)]
    '''
    k = 1
    # fallback：无疾病标签
    if q not in _q_ctx_global:
        rng = random.Random(rng_key + '_fb')
        seen, results = set(), []
        for _ in range(200):
            d = doc_ids[rng.randrange(len(doc_ids))]
            if doc_comp.get(d) != q_comp[q] and d not in seen:
                seen.add(d); results.append((d, 'fallback: no disease label -> random'))
                break
        return results or [(doc_ids[0], 'fallback')]

    ctx = _q_ctx_global[q]
    q_rel, q_fam, _ = ctx

    # 构建候选池
    if not loose:
        pool_diff, pool_sib = onto_pool(q_rel, q_fam)
        pool_with_rel = [(d, 'sibling' if d in set(pool_sib) else 'different')
                         for d in pool_diff + pool_sib if doc_comp.get(d) != q_comp[q]]
    else:
        pool_with_rel = []
        for d in doc_ids:
            d_all, d_fam, named, _, _, _ = doc_sets.get(d, EMPTY)
            if d_all & q_rel: continue      # 排除 same
            if doc_comp.get(d) == q_comp[q]: continue
            if not named:
                rel = 'unnamed'
            elif d_fam & q_fam:
                rel = 'sibling'
            else:
                rel = 'different'
            pool_with_rel.append((d, rel))

    if not pool_with_rel:
        rng = random.Random(rng_key + '_empty')
        for _ in range(200):
            d = doc_ids[rng.randrange(len(doc_ids))]
            if doc_comp.get(d) != q_comp[q]:
                return [(d, 'fallback: empty pool -> random')]
        return []

    pool_tag = 'loose' if loose else 'safe'

    # 打分
    def score_bm25(d, rel):
        if bm25_row:
            idx = next((i for i, x in enumerate(bm25_row['docs']) if x == d), None)
            return bm25_row['scores'][idx] if idx is not None else 0.0
        return 0.0

    def score_ngram(d, rel):
        q_2g = _q_2gram_cache.get(q, frozenset())
        d_2g = _doc_2gram.get(d, frozenset())
        if not q_2g or not d_2g: return 0.0
        union = len(q_2g | d_2g)
        return len(q_2g & d_2g) / union if union else 0.0

    def score_formula(d, rel):
        return difficulty_score(q, ctx, d, rel)

    score_fn = {'bm25': score_bm25, 'ngram': score_ngram, 'formula': score_formula}[scorer]

    # 正例感知 cap
    if scorer == 'bm25' and bm25_row and bm25_row.get('pos_scores'):
        top_pos = max(bm25_row['pos_scores'].values())
        cap = top_pos - (1 - margin) * abs(top_pos)
    elif scorer == 'ngram':
        pos_sc = [score_ngram(p, 'same') for p in positives.get(q, [])]
        cap = max(pos_sc) * margin if pos_sc else None
    elif scorer == 'formula':
        pos_sc = [score_formula(p, 'same') for p in positives.get(q, [])]
        cap = max(pos_sc) * margin if pos_sc else None
    else:
        cap = None

    scored = [(score_fn(d, rel), d, rel) for d, rel in pool_with_rel]
    elig = [(s, d, rel) for s, d, rel in scored if cap is None or s <= cap]
    if not elig:
        elig = scored

    elig.sort(key=lambda x: x[0], reverse=True)
    top_pool = elig[:POOL * 3]
    rng = random.Random(rng_key)
    rng.shuffle(top_pool)
    chosen, seen_d = [], set()
    for s, d, rel in top_pool:
        if d not in seen_d:
            chosen.append((d, f'onto_{scorer}_{pool_tag}: {rel}'))
            seen_d.add(d)
            break

    if not chosen:
        rng2 = random.Random(rng_key + '_pad')
        shuf = list(pool_with_rel); rng2.shuffle(shuf)
        for d, rel in shuf:
            if d not in seen_d:
                chosen.append((d, f'onto_{scorer}_{pool_tag}: fallback any'))
                break

    return chosen[:1]


def _onto_mid(q, margin, rng_key, strat, lo, hi, tag, loose=False, scorer='ngram'):
    '''
    保持随机多样性的本体负例：候选池(安全池或宽松池) -> (可选)打分分位带 [lo, hi) + 正例感知上限
    -> 按疾病大类均匀抽样(strat=True) 或 池内均匀随机(strat=False)。不取最高分段。
    '''
    if q not in _q_ctx_global:
        rng = random.Random(rng_key + '_fb')
        for _ in range(200):
            d = doc_ids[rng.randrange(len(doc_ids))]
            if doc_comp.get(d) != q_comp[q]:
                return [(d, 'fallback: no disease label -> random')]
        return [(doc_ids[0], 'fallback')]
    ctx_q = _q_ctx_global[q]
    q_rel, q_fam, _ = ctx_q
    if loose:
        pool = [d for d in doc_ids if doc_comp.get(d) != q_comp[q] and not (doc_sets.get(d, EMPTY)[0] & q_rel)]
        sib = {d for d in pool if doc_sets.get(d, EMPTY)[1] & q_fam}
    else:
        pool_diff, pool_sib = onto_pool(q_rel, q_fam)
        sib = set(pool_sib)
        pool = [d for d in pool_diff + pool_sib if doc_comp.get(d) != q_comp[q]]
    if not pool:
        rng = random.Random(rng_key + '_empty')
        for _ in range(200):
            d = doc_ids[rng.randrange(len(doc_ids))]
            if doc_comp.get(d) != q_comp[q]:
                return [(d, 'fallback: empty pool -> random')]
        return []
    q_2g = _q_2gram_cache.get(q, frozenset())

    def jac(d):
        if scorer == 'formula':
            _a, d_fam, named, _b, _c, _e = doc_sets.get(d, EMPTY)
            rel = 'unnamed' if not named else ('sibling' if d_fam & q_fam else 'different')
            return difficulty_score(q, ctx_q, d, rel)
        d_2g = _doc_2gram.get(d, frozenset())
        if not q_2g or not d_2g:
            return 0.0
        u = len(q_2g | d_2g)
        return len(q_2g & d_2g) / u if u else 0.0

    if hi - lo < 1.0:
        sc = sorted(((jac(d), d) for d in pool), key=lambda x: x[0])
        pos = [jac(p) for p in positives.get(q, []) if p in _doc_2gram]
        if pos:
            cap = max(pos) * margin
            under = [x for x in sc if x[0] <= cap]
            if under:
                sc = under
        n = len(sc)
        a = int(n * lo)
        b = max(a + 1, int(n * hi))
        band = [d for _, d in sc[a:b]]
    else:
        band = pool
    rng = random.Random(rng_key)
    if strat:
        groups = defaultdict(list)
        for d in band:
            fam = sorted(doc_sets.get(d, EMPTY)[1])
            groups[fam[0] if fam else '-'].append(d)
        d = rng.choice(groups[rng.choice(sorted(groups))])
    else:
        d = rng.choice(band)
    rel = 'sibling' if d in sib else 'different'
    return [(d, 'onto_' + tag + ': ' + rel)]


_same_pool_cache = {}


def _onto_dose(q, rng_key, frac):
    '''
    假负例剂量实验：以概率 frac 从"同病答案"（本体判为 same，排除同连通分量和该问题自己的正例）里取负例，
    否则从本体安全池（different+sibling）里均匀随机取。frac=0 即 onto_pool_rand。
    无疾病标签的问题退回 random（与其他本体方法一致）。
    '''
    if q not in _q_ctx_global:
        rng = random.Random(rng_key + '_fb')
        for _ in range(200):
            d = doc_ids[rng.randrange(len(doc_ids))]
            if doc_comp.get(d) != q_comp[q]:
                return [(d, 'fallback: no disease label -> random')]
        return [(doc_ids[0], 'fallback')]
    q_rel, q_fam, _ = _q_ctx_global[q]
    rng = random.Random(rng_key)
    use_same = rng.random() < frac
    if use_same:
        key = frozenset(q_rel)
        if key not in _same_pool_cache:
            _same_pool_cache[key] = [d for d in doc_ids if doc_sets.get(d, EMPTY)[0] & q_rel]
        pool = [d for d in _same_pool_cache[key]
                if doc_comp.get(d) != q_comp[q] and d not in positives[q]]
        if pool:
            return [(rng.choice(pool), 'onto_dose: same (injected false negative)')]
    pool_diff, pool_sib = onto_pool(q_rel, q_fam)
    pool = [d for d in pool_diff + pool_sib if doc_comp.get(d) != q_comp[q]]
    if not pool:
        for _ in range(200):
            d = doc_ids[rng.randrange(len(doc_ids))]
            if doc_comp.get(d) != q_comp[q]:
                return [(d, 'fallback: empty pool -> random')]
        return []
    return [(rng.choice(pool), 'onto_dose: safe pool')]


def pick_random(q):
    rng = random.Random('{}|rand|{}'.format(SEED, q))
    for _ in range(200):
        d = doc_ids[rng.randrange(len(doc_ids))]
        if doc_comp.get(d) != q_comp[q]:
            return d
    raise RuntimeError('pick_random failed for ' + q)





# =============================================================================
# 顶层工作函数（multiprocessing fork 继承全局变量，不需要传参）
# =============================================================================

def _select_one(args):
    '''每个问题的选负例计算，在 worker 进程里运行。
    args = (q, dense_row, margin, q_ctx_local, random_fallback_d)
    返回 (q, ch_dict, multi_dict)
    注意：热门计数在并行阶段不做全局限制；主进程收集后统计 distinct。
    '''
    q, row, margin, q_ctx_local, fallback_d = args
    docs, scores = row['docs'], row['scores']
    k10  = min(SKIP_RANK, len(docs))
    cap  = row['pos_scores'] and (max(row['pos_scores'].values()) - (1 - margin) *
                                   abs(max(row['pos_scores'].values())))
    # 重算 cap（与 cap_of 等价，不依赖全局 dense）
    top_pos = max(row['pos_scores'].values()) if row.get('pos_scores') else scores[0]
    cap = top_pos - (1 - margin) * abs(top_pos)
    elig = [(ki, d, s) for ki, (d, s) in enumerate(zip(docs, scores), 1) if s <= cap]

    # margin
    if not elig:
        mg = (docs[-1], len(docs), 'fallback: no candidate under score limit', 'dense', scores[-1])
    else:
        k2, d, s = random.Random('{}|{}'.format(SEED, q)).choice(elig[:POOL])
        mg = (d, k2, 'margin', 'dense', s)

    # margin_onto
    if not elig:
        mo = mg
    elif q not in q_ctx_local:
        mo = (mg[0], mg[1], 'margin (no disease label)', 'dense', mg[4])
    else:
        keep = [x for x in elig if relation(q, x[1], q_ctx_local) not in ('same', 'unnamed')]
        if keep:
            k2, d, s = random.Random('{}|onto|{}'.format(SEED, q)).choice(keep[:POOL])
            mo = (d, k2, 'margin + ontology filter', 'dense', s)
        else:
            mo = (mg[0], mg[1], 'margin (filter would leave nothing)', 'dense', mg[4])

    ch = {
        'dense':         (docs[0],     1,    'rank 1',             'dense', scores[0]),
        'dense_rank10':  (docs[k10-1], k10,  f'rank {SKIP_RANK}', 'dense', scores[k10-1]),
        'dense_rank100': (docs[-1],    len(docs), 'rank 100',      'dense', scores[-1]),
        'random':        (fallback_d,  0,    'random answer',      'random', None),
        'margin':        mg,
        'margin_onto':   mo,
    }
    b = bm25.get(q, {'docs': [], 'scores': []})
    if b['docs']:
        ch['bm25'] = (b['docs'][0], 1, 'rank 1', 'bm25', b['scores'][0])
    else:
        ch['bm25'] = (ch['dense_rank10'][0], k10,
                      'fallback: no BM25 candidate, dense rank 10', 'dense', ch['dense_rank10'][4])

    # bm25_onto / dense_onto：与 margin_onto 同口径，跳过 same 和 unnamed 后取排名最靠前的候选
    def _first_onto(cand_docs, cand_scores, src, fallback):
        if q not in q_ctx_local:
            return (fallback[0], fallback[1], src + ' (no disease label)', fallback[3], fallback[4])
        for ki, (d, s) in enumerate(zip(cand_docs, cand_scores), 1):
            if relation(q, d, q_ctx_local) not in ('same', 'unnamed'):
                return (d, ki, src + ' + ontology filter', src, s)
        return (fallback[0], fallback[1], src + ' (filter would leave nothing)', fallback[3], fallback[4])

    ch['bm25_onto']  = (_first_onto(b['docs'], b['scores'], 'bm25', ch['bm25'])
                        if b['docs'] else ch['bm25'])
    ch['dense_onto'] = _first_onto(docs, scores, 'dense', ch['dense'])

    # k=1 的多负例方法。按原来的顺序，只计算 MULTI_NEEDED 里的（见 CMP_ALT 下面的说明）；
    # 每种方法用自己的随机种子，跳过其他方法不改变它选出的负例。
    calls = [
        ('onto_pool_sim',    lambda: pick_onto_multi(q, q_ctx_local, f'{SEED}|sim|{q}|k1',  True,  1)),
        ('onto_pool_rand',   lambda: pick_onto_multi(q, q_ctx_local, f'{SEED}|pr|{q}|k1',   False, 1)),
        ('onto_pool_hard',   lambda: _scored_pool(q, q_ctx_local, bm25.get(q), margin, f'{SEED}|ohs|{q}', 'formula', loose=False)),
        ('onto_hard_loose',  lambda: _scored_pool(q, q_ctx_local, bm25.get(q), margin, f'{SEED}|ohl|{q}', 'formula', loose=True)),
        ('onto_pool_strat',  lambda: _onto_mid(q, margin, f'{SEED}|ost|{q}', True, 0.0, 1.0, 'strat')),
        ('onto_pool_mid',    lambda: _onto_mid(q, margin, f'{SEED}|omd|{q}', True, 0.3, 0.7, 'mid')),
        ('onto_strat_loose', lambda: _onto_mid(q, margin, f'{SEED}|osl|{q}', True, 0.0, 1.0, 'strat_loose', loose=True)),
        ('onto_loose_rand',  lambda: _onto_mid(q, margin, f'{SEED}|olr|{q}', False, 0.0, 1.0, 'loose_rand', loose=True)),
        ('onto_dose_p10',    lambda: _onto_dose(q, f'{SEED}|d10|{q}', 0.10)),
        ('onto_dose_p30',    lambda: _onto_dose(q, f'{SEED}|d30|{q}', 0.30)),
        ('onto_dose_p50',    lambda: _onto_dose(q, f'{SEED}|d50|{q}', 0.50)),
        ('onto_nmid_loose',  lambda: _onto_mid(q, margin, f'{SEED}|onm|{q}', True, 0.3, 0.7, 'nmid_loose', loose=True, scorer='ngram')),
        ('onto_hmid_loose',  lambda: _onto_mid(q, margin, f'{SEED}|ohm|{q}', True, 0.3, 0.7, 'hmid_loose', loose=True, scorer='formula')),
    ]
    mu = {m: f() for m, f in calls if m in MULTI_NEEDED}

    return q, ch, mu

def build(model):
    global _q_ctx_global, _random_fallback
    _hot_count.clear()
    _onto_pool_cache.clear()

    margin, max_pos = load_07_config(model)
    dense = {r['qid']: r for r in read_jsonl(f'{CAND_DIR}/dense_{model}.jsonl')}
    lacking = [q for q in qtext if q not in dense or not dense[q]['docs']]
    assert not lacking, '{} questions missing dense candidates — re-run 07 --model {}'.format(len(lacking), model)

    q_ctx, q_src = make_context(max_pos)
    _q_ctx_global = q_ctx

    # 预计算问题字符 2-gram（onto_hard 难度打分用）
    _q_2gram_cache.clear()
    for q in qtext:
        _n = normalize(qtext[q])
        _q_2gram_cache[q] = frozenset(_n[i:i+2] for i in range(len(_n) - 1)) if len(_n) > 1 else frozenset()

    _random_fallback = {q: pick_random(q) for q in qtext}

    out_dir = f'{TRAIN_DIR}/{model}'
    os.makedirs(out_dir, exist_ok=True)

    # choice[qid][method] = (doc_id, rank, rule, source, score)   单负例方法
    # multi[qid][method]  = [(doc_id, rule), ...]                  多负例方法（k=1/3/5）
    choice = {}
    multi  = {}

    jobs = [(q, dense[q], margin, q_ctx, _random_fallback[q]) for q in qtext]

    if WORKERS > 0:
        import multiprocessing
        with multiprocessing.Pool(processes=WORKERS) as pool:
            for q, ch, mu in tqdm(
                    pool.imap_unordered(_select_one, jobs, chunksize=64),
                    desc=f'[{model}] selecting negatives (workers={WORKERS})',
                    total=len(jobs), unit='q'):
                choice[q] = ch
                multi[q]  = mu
    else:
        for job in tqdm(jobs, desc=f'[{model}] selecting negatives (serial)', unit='q'):
            q, ch, mu = _select_one(job)
            choice[q] = ch
            multi[q]  = mu

    # 写单负例训练文件
    n_bad = Counter()
    for method in tqdm(SINGLE_METHODS, desc=f'[{model}] writing single-neg files', unit='method'):
        rows = []
        for r, q in zip(train, row_qid):
            row = {'anchor': r['anchor'], 'positive': r['positive']}
            if method != 'mnrl':
                neg = corpus[choice[q][method][0]]
                n_bad[method] += neg in (r['positive'], r['anchor'])
                row['negative'] = neg
            rows.append(row)
        write_jsonl(f'{out_dir}/{method}.jsonl', rows)

    # 写多负例训练文件（sentence-transformers 5.x 支持 negative_1/negative_2/...）
    for method in tqdm(MULTI_METHODS_ALL, desc=f'[{model}] writing multi-neg files', unit='method'):
        rows = []
        for r, q in zip(train, row_qid):
            negs = multi[q][method]   # [(doc_id, rule), ...]
            row  = {'anchor': r['anchor'], 'positive': r['positive']}
            for i, (did, _) in enumerate(negs):
                col = 'negative' if i == 0 else f'negative_{i}'
                row[col] = corpus[did]
                n_bad[method] += corpus[did] in (r['positive'], r['anchor'])
            rows.append(row)
        write_jsonl(f'{out_dir}/{method}.jsonl', rows)
    assert not sum(n_bad.values()), 'negative equals positive or anchor: {}'.format(dict(n_bad))

    METHODS_ALL = SINGLE_METHODS + MULTI_METHODS_ALL
    write_jsonl(f'{out_dir}/negatives.jsonl',
                [{'qid': q, 'question': qtext[q],
                  **{m: {'doc_id': choice[q][m][0], 'rule': choice[q][m][2],
                         'relation': relation(q, choice[q][m][0], q_ctx)}
                     for m in SINGLE_METHODS[1:]},
                  **{m: [{'doc_id': d, 'rule': rule,
                          'relation': relation(q, d, q_ctx)} for d, rule in multi[q][m]]
                     for m in MULTI_METHODS_ALL}} for q in qtext])

    # ---- 统计 ----
    nq = len(choice)
    stats = {'mnrl': {'rows': len(train)}}
    for m in SINGLE_METHODS[1:]:
        picked = [choice[q][m] for q in choice]
        rel  = Counter(relation(q, choice[q][m][0], q_ctx) for q in choice)
        freq = Counter(p[0] for p in picked)
        dp   = [(q, choice[q][m]) for q in choice if choice[q][m][3] == 'dense']
        over = sum(1 for q, p in dp if p[4] is not None and p[4] > cap_of(dense[q], margin))
        ranks = [p[1] for p in picked if p[1]]
        stats[m] = {
            'rows': len(train),
            'mean_rank':   round(statistics.mean(ranks), 2) if ranks else None,
            'median_rank': statistics.median(ranks) if ranks else None,
            'over_margin': round(over / max(len(dp), 1), 4),
            **{k: round(rel[k] / nq, 4) for k in REL_KEYS},
            'distinct_negatives': len(freq),
            'top10_negative_share': round(sum(n for _, n in freq.most_common(10)) / nq, 4),
        }
    for m in MULTI_METHODS_ALL:
        all_negs = [d for q in multi for d, _ in multi[q][m]]
        rel  = Counter(relation(q, d, q_ctx) for q in multi for d, _ in multi[q][m])
        freq = Counter(all_negs)
        stats[m] = {
            'rows': len(train),
            'negatives_per_sample': len(multi[list(multi.keys())[0]][m]),
            'mean_rank': None, 'median_rank': None, 'over_margin': 0.0,
            **{k: round(rel[k] / max(len(all_negs), 1), 4) for k in REL_KEYS},
            'distinct_negatives': len(freq),
            'top10_negative_share': round(sum(n for _, n in freq.most_common(10)) / max(len(all_negs), 1), 4),
        }

    lab = [q for q in choice if q in q_ctx]
    onto_stats = {}
    for m in MULTI_METHODS_ALL:
        rules   = Counter(rule for q in multi for _, rule in multi[q][m])
        by_onto = sum(1 for q in multi for _, rule in multi[q][m] if 'fallback' not in rule)
        total   = sum(len(multi[q][m]) for q in multi)
        onto_stats[m] = {'rules': dict(rules), 'by_ontology': by_onto,
                         'share_of_all': round(by_onto / max(total, 1), 4)}

    pool_size_dist = Counter()
    for q in lab:
        ctx = q_ctx[q]
        pd, ps = onto_pool(ctx[0], ctx[1])
        pool_size_dist[min((len(pd)+len(ps)) // 100 * 100, 3000)] += 1

    onto_pool_sim_diff = sum(
        multi[q][CMP_ALT][0][0] != multi[q]['onto_pool_rand'][0][0]
        for q in multi if multi[q][CMP_ALT] and multi[q]['onto_pool_rand'])

    ontology_section = {
        'labelled_questions': len(lab), 'labelled_share': round(len(lab) / nq, 4),
        'disease_source': dict(Counter(q_ctx[q][2] for q in lab)),
        'margin_onto_differs_from_margin': sum(
            choice[q]['margin_onto'][0] != choice[q]['margin'][0] for q in choice),
        'alt_differs_from_pool_rand': onto_pool_sim_diff,
        'generic_docs_excluded': len(generic_docs),
        'sample_hi': SAMPLE_HI,
        'sibling_quota': SIBLING_QUOTA,
        'onto_methods': onto_stats,
        'pool_size_distribution': {str(k): v for k, v in sorted(pool_size_dist.items())},
    }

    # ---- 案例：margin 选到同病但本体换掉了 ----
    rng = random.Random(SEED)
    pool_ex = [q for q in lab if relation(q, choice[q]['margin'][0], q_ctx) == 'same'
               and multi[q][CMP_ALT] and
               relation(q, multi[q][CMP_ALT][0][0], q_ctx) in ('different', 'sibling')]
    examples = []
    for q in rng.sample(pool_ex, min(N_EXAMPLES, len(pool_ex))):
        md_ = choice[q]['margin'][0]
        od_ = multi[q][CMP_ALT][0][0]
        examples.append({'question': qtext[q][:80],
                         'question_diseases': names(q_con.get(q, ())),
                         'margin_neg': corpus[md_][:120],
                         'margin_rel': relation(q, md_, q_ctx),
                         'onto_neg':   corpus[od_][:120],
                         'onto_rel':   relation(q, od_, q_ctx)})

    # ---- 盲评表：包含 dense_rank10 / margin / onto_pool_sim / onto_pool_rand / random ----
    AUDIT_SINGLE = ['dense_rank10', 'margin', 'random']
    cands_q = sorted(q for q in lab if len({
        choice[q][m][0] for m in AUDIT_SINGLE} |
        {multi[q][CMP_ALT][0][0] if multi[q][CMP_ALT] else None} |
        {multi[q]['onto_pool_rand'][0][0] if multi[q]['onto_pool_rand'] else None} -
        {None}) > 1)
    picked_q = random.Random(SEED + 10).sample(cands_q, min(N_AUDIT, len(cands_q)))
    items = []
    for q in picked_q:
        by_doc = defaultdict(list)
        for m in AUDIT_SINGLE:
            by_doc[choice[q][m][0]].append(m)
        for m in (CMP_ALT, 'onto_pool_rand'):
            if multi[q][m]:
                by_doc[multi[q][m][0][0]].append(m)
        items += [(q, d, ms) for d, ms in by_doc.items()]
    random.Random(SEED + 11).shuffle(items)
    sheet = f'{out_dir}/audit_{model}.csv'
    key   = f'{out_dir}/audit_{model}_key.csv'
    with open(sheet, 'w', encoding='utf-8-sig', newline='') as f1, \
         open(key,   'w', encoding='utf-8-sig', newline='') as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(['id', 'question', 'reference_answer', 'candidate_answer', LABEL_COL])
        w2.writerow(['id', 'qid', 'doc_id', 'methods', 'relation', 'positive_score'])
        for i, (q, d, ms) in enumerate(items, 1):
            ref = max(dense[q]['pos_scores'], key=dense[q]['pos_scores'].get)
            w1.writerow([i, qtext[q], corpus[ref][:300], corpus[d][:300], ''])
            w2.writerow([i, q, d, '|'.join(ms), relation(q, d, q_ctx),
                         dense[q]['pos_scores'][ref]])
    ontology_section['audit'] = {'questions': len(picked_q), 'rows': len(items),
                                 'sheet': sheet, 'key': key}

    return {'model': model, 'questions': nq, 'margin': margin,
            'max_pos_concepts': max_pos, 'methods': stats,
            'ontology': ontology_section, 'examples': examples, 'out_dir': out_dir,
            'single_methods': SINGLE_METHODS, 'multi_methods': MULTI_METHODS_ALL}


# %%
# =============================================================================
# 打印 + 报告
# =============================================================================

def print_report(rep):
    q, o   = rep['questions'], rep['ontology']
    smths  = rep.get('single_methods', SINGLE_METHODS)
    mmths  = rep.get('multi_methods',  [])
    all_m  = smths + mmths
    print(NL + '=' * 70)
    print('[{}]  {:,} questions  (MARGIN {}, POOL {}, SAMPLE_HI {:.0%}, SIBLING_QUOTA {})'.format(
        rep['model'], q, rep['margin'], POOL, SAMPLE_HI, SIBLING_QUOTA))
    print('=' * 70)
    hdr = 'method'.ljust(22) + 'k'.rjust(3) + 'meanrank'.rjust(10) + 'median'.rjust(8) \
          + 'over'.rjust(8) + ''.join(k.rjust(10) for k in REL_KEYS) + 'distinct'.rjust(10)
    print(NL + hdr)
    for m in all_m:
        if m not in rep['methods']:
            continue
        s  = rep['methods'][m]
        kk = s.get('negatives_per_sample', 1)
        if m == 'mnrl':
            print(m.ljust(22) + '   (no hard negative)'); continue
        print(m.ljust(22) + str(kk).rjust(3)
              + fmt(s['mean_rank'], '.1f').rjust(10)
              + fmt(s['median_rank'], 'g').rjust(8)
              + format(s['over_margin'], '.1%').rjust(8)
              + ''.join(format(s[k], '.1%').rjust(10) for k in REL_KEYS)
              + format(s['distinct_negatives'], ',').rjust(10))
    print(NL + '  labelled questions            : {:,} ({:.1%})'.format(
        o['labelled_questions'], o['labelled_share']))
    print('  generic docs excluded         : {:,}'.format(o['generic_docs_excluded']))
    print('  SAMPLE_HI                     : {:.0%}'.format(o['sample_hi']))
    print('  SIBLING_QUOTA                 : {}'.format(o['sibling_quota']))
    print('  margin_onto differs margin    : {:,} ({:.1%})'.format(
        o['margin_onto_differs_from_margin'],
        o['margin_onto_differs_from_margin'] / q))
    print(('  ' + CMP_ALT + ' differs onto_pool_rand : {:,} ({:.1%})').format(
        o['alt_differs_from_pool_rand'],
        o['alt_differs_from_pool_rand'] / q))
    for m, om in o['onto_methods'].items():
        print('  {} by ontology: {:,} ({:.1%})'.format(m, om['by_ontology'], om['share_of_all']))
        for rule, n in sorted(om['rules'].items(), key=lambda x: -x[1])[:5]:
            tot = sum(om['rules'].values())
            print('    {:<52s}{:>8,} ({:.1%})'.format(rule, n, n / max(tot, 1)))
    print('  audit : {}  ({} rows)'.format(o['audit']['sheet'], o['audit']['rows']))


for model in MODEL_KEYS:
    if not os.path.exists(f'{CAND_DIR}/dense_{model}.jsonl'):
        print(NL + '[skip] {}: no dense candidates — run 07 --model {}'.format(model, model))
        continue
    rep = build(model)
    rep.update({'step': '08_training_sets_new',
                'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
                'config': {'POOL': POOL, 'SKIP_RANK': SKIP_RANK, 'SEED': SEED,
                           'MAX_HOT': MAX_HOT, 'GENERIC_ENTITY_THR': GENERIC_ENTITY_THR,
                           'GENERIC_CONCEPT_THR': GENERIC_CONCEPT_THR,
                           'hard_negative': HARD_NEG}})
    print_report(rep)
    base = f'{REPORT_DIR}/08_training_sets_{model}_{OUT_TAG}'
    with open(base + '.json', 'w', encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)

    # ---- Markdown 报告 ----
    def _mdt(headers, rows, numeric=True):
        esc = lambda v: str(v).replace('|', '/').replace(NL, ' ')
        align = [':---'] + [('---:' if numeric else ':---')] * (len(headers) - 1)
        lines = ['| ' + ' | '.join(map(esc, headers)) + ' |',
                 '| ' + ' | '.join(align) + ' |']
        lines += ['| ' + ' | '.join(map(esc, r)) + ' |' for r in rows]
        return NL.join(lines)

    q = rep['questions']
    o = rep['ontology']

    # 方法对比表（单负例 + 多负例）
    method_rows = []
    for m in (METHODS + MULTI_METHODS_ALL):
        if m not in rep['methods']:
            continue
        s = rep['methods'][m]
        if m == 'mnrl':
            method_rows.append([m, HARD_NEG.get(m, m), format(s['rows'], ',')]
                               + ['-'] * (3 + len(REL_KEYS)))
        else:
            method_rows.append([
                m, HARD_NEG.get(m, m), format(s['rows'], ','),
                fmt(s['mean_rank'], '.1f'), fmt(s['median_rank'], 'g'),
                format(s['over_margin'], '.1%'),
                *[format(s[k], '.1%') for k in REL_KEYS],
            ])

    # 规则明细
    rule_rows = []
    for m in (['margin', 'margin_onto'] + MULTI_METHODS_ALL):
        for rule, n in sorted(o['onto_methods'].get(m, {}).get('rules', {}).items(),
                              key=lambda x: -x[1]):
            tot = sum(o['onto_methods'][m]['rules'].values())
            rule_rows.append([m, rule, format(n, ','), format(n / max(tot, 1), '.1%')])

    # onto 方法覆盖
    onto_cover_rows = []
    for m in (MULTI_METHODS_ALL):
        om = o['onto_methods'].get(m, {})
        onto_cover_rows.append([m,
                                 format(om.get('by_ontology', 0), ','),
                                 format(om.get('share_of_all', 0), '.1%')])

    # 案例
    ex_rows = [['{} [{}]'.format(e['question'], e['question_diseases']),
                '{} · {}'.format(e['margin_rel'], e['margin_neg']),
                '{} · {}'.format(e['onto_rel'], e['onto_neg'])]
               for e in rep['examples']]

    smths = rep.get('single_methods', [])
    mmths = rep.get('multi_methods', [])
    _md8 = [
        '# Step 08 (new) · Training sets — `{}`'.format(model), '',
        'Generated {}. {:,} unique training questions, {:,} training pairs. '
        'MARGIN {} (from step 07), POOL {}, SAMPLE_HI {:.0%}, SIBLING_QUOTA {}.'.format(
            rep['generated_at'], q, rep['methods']['mnrl']['rows'],
            rep['margin'], POOL, SAMPLE_HI, SIBLING_QUOTA), '',
        '## Hard negative chosen by each method', '',
        _mdt(['Method', 'k', 'Hard negative', 'Rows', 'Mean rank', 'Median rank',
              'Over score limit'] + REL_KEYS, method_rows), '',
        '> k = number of hard negatives per sample. Relation columns: disease relation '
        'between the question and its chosen negative.', '',
        '## Ontology coverage', '',
        _mdt(['Item', 'Value'], [
            ['Questions with a disease label',
             '{:,} ({:.1%})'.format(o['labelled_questions'], o['labelled_share'])],
            ['Disease label source', str(o['disease_source'])],
            ['Generic answers excluded from onto pool', format(o['generic_docs_excluded'], ',')],
            ['SAMPLE_HI', '{:.0%}'.format(o['sample_hi'])],
            ['SIBLING_QUOTA', str(o['sibling_quota'])],
            ['margin_onto differs from margin',
             '{:,} ({:.1%})'.format(o['margin_onto_differs_from_margin'],
                                     o['margin_onto_differs_from_margin'] / q)],
            [CMP_ALT + ' differs from onto_pool_rand',
             '{:,} ({:.1%})'.format(o['alt_differs_from_pool_rand'],
                                     o['alt_differs_from_pool_rand'] / q)],
        ]), '',
        '### Onto-method coverage (questions where ontology selected the negative)', '',
        _mdt(['Method', 'By ontology', 'Share of all'], onto_cover_rows), '',
        '### Rules breakdown', '',
        _mdt(['Method', 'Rule', 'Questions', 'Share'], rule_rows, numeric=False) if rule_rows else '(none)', '',
        '## Examples: margin picked same-disease, ontology filter changed it', '',
        _mdt(['Question', 'margin negative (relation)', 'margin_onto negative (rank · relation)'],
             ex_rows, numeric=False) if ex_rows else 'No examples found.', '',
        '## Blind audit', '',
        '`{}` has {} rows from {} questions (methods: {}). '
        'Fill the last column with y / n / ?, save as `audit_{}_filled.csv` and re-run this file '
        'to get false-negative rates per method.'.format(
            o['audit']['sheet'], o['audit']['rows'], o['audit']['questions'],
            ', '.join(AUDIT_METHODS), model),
    ]

    with open(base + '.md', 'w', encoding='utf-8') as f:
        f.write(NL.join(_md8) + NL)
    print(NL + 'Reports: ' + base + '.json / .md')

# %%
# =============================================================================
# 盲评汇总（判完 audit 表后再运行）
# =============================================================================


def audit_summary(model):
    filled = f'{TRAIN_DIR}/{model}/audit_{model}_filled.csv'
    key_path = f'{TRAIN_DIR}/{model}/audit_{model}_key.csv'
    if not (os.path.exists(filled) and os.path.exists(key_path)):
        print('[{}] no filled audit sheet: {}'.format(model, filled)); return
    with open(key_path, encoding='utf-8-sig', newline='') as f:
        key = {r['id']: r for r in csv.DictReader(f)}
    cnt = defaultdict(Counter)
    with open(filled, encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            v = (r.get(LABEL_COL) or '').strip().lower().replace('？', '?')
            k = key.get(r.get('id'))
            if v in ('y', 'n', '?') and k:
                for m in k['methods'].split('|'):
                    cnt[m][v] += 1
                cnt['relation_' + k['relation']][v] += 1
    print(NL + '[{}] false-negative rate by method:'.format(model))
    for m in AUDIT_METHODS:
        c = cnt[m]; n = sum(c.values())
        if n:
            print('  {:<18s} n={:>4d}  y {:5.1%}  ? {:5.1%}  n {:5.1%}  y+? {:5.1%}'.format(
                m, n, c['y']/n, c['?']/n, c['n']/n, (c['y']+c['?'])/n))
    print('  by relation:')
    for rel in REL_KEYS:
        c = cnt['relation_' + rel]; n = sum(c.values())
        if n >= 5:
            print('  {:<18s} n={:>4d}  y {:5.1%}  y+? {:5.1%}'.format(
                rel, n, c['y']/n, (c['y']+c['?'])/n))


for model in MODEL_KEYS:
    audit_summary(model)

print(NL + 'Next: 09_train_eval_new.py')
