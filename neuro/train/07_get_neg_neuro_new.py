# %% [markdown]
# # Step 07 (new) · 难负例候选挖掘
#
# 为每个训练问题，在训练集答案库里用 BM25 和原始（未微调）模型各检索一份候选负例，
# 08 从这两份候选里挑负例。原 07 的文件不动，输出写到 candidates_new/。
#
# 相对原 07 的改动：
# 1. 候选深度从 30 加到 100（TOP_K），检索深度 300（RETRIEVE_DEPTH）：
#    原来本体规则只在前 30 名里找，约 40% 的问题找不到合适的候选，覆盖率被卡住。
# 2. 候选存成两个平行数组 docs / scores，比 [[doc_id, score], ...] 省文件和内存。
# 3. BM25 得分为 0 的候选丢弃（深度 100 时，短问题会取到毫无词面重合的“候选”）。
# 4. 本体预览改用 06_本体标注_neuro_new.py 的标签，把候选和问题的关系分成五类：
#      same      答案正文点名、或它配对的问题涉及与问题相同/上下位的疾病
#      sibling   不同病，但属于同一个大类（脑梗死 vs 脑出血）
#      different 正文点名了疾病，且大类与问题不同
#      unnamed   正文没点名任何病（泛泛的回答，什么病的问题都能用）
#      unknown   问题（连同它的正确答案正文）都没有疾病标签
#    问题没有疾病词时，用它正确答案正文里点名的疾病（不超过 MAX_POS_CONCEPTS 种）代替。
# 5. 假负例核对表：按 same / sibling / different / unnamed 各抽 N_CHECK 条（只抽分数限制之内的候选），盲评。
#    这直接检验 08 要用的规则，而不是原来的“同病同方面”。
# 6. 参数走命令行（Jupyter 里多余参数会被忽略），服务器和本机都能跑。
#
# 用法（BGE 先跑，同时生成两个模型共用的 BM25；再跑 Qwen）：
#   python 07_get_neg_new.py --model bge-base
#   python 07_get_neg_new.py --model qwen3-0.6b
# 在 Jupyter 里：第一格之前先 os.environ['NEURO_MODEL'] = 'qwen3-0.6b'（默认 bge-base），再逐格运行。

# %%
# =============================================================================
# 环境（必须放在所有 import 之前；改完要重启 kernel）
# =============================================================================
import os, sys
sys.modules['flash_attn'] = None            # 服务器上的 flash-attn 与 PyTorch 版本不匹配，屏蔽它；没有这个问题的机器上无害
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse, csv, datetime, hashlib, json, random, re, statistics, unicodedata
from collections import Counter, defaultdict
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# %%
# =============================================================================
# 配置
# =============================================================================


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.environ.get('NEURO_OUT_DIR', '/data/neuro/data/neuro_dataset'))
    ap.add_argument('--tag', default='new', help='输出目录后缀')
    ap.add_argument('--model', default=os.environ.get('NEURO_MODEL', 'bge-base'),
                    choices=['bge-base', 'qwen3-0.6b'])
    ap.add_argument('--bm25', dest='bm25', action='store_true', default=None,
                    help='同时跑 BM25（默认：bge-base 跑，qwen3-0.6b 不跑，BM25 与模型无关）')
    ap.add_argument('--no-bm25', dest='bm25', action='store_false')
    ap.add_argument('--no-dense', action='store_true')
    ap.add_argument('--top-k', type=int, default=100)
    ap.add_argument('--depth', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    a, _unknown = ap.parse_known_args()
    if a.bm25 is None:
        a.bm25 = (a.model == 'bge-base')
    return a


ARGS = _args()
OUT_DIR = ARGS.out_dir
TAG = ARGS.tag
SPLIT_DIR = f'{OUT_DIR}/split'
ONTO_DIR = f'{OUT_DIR}/ontology_{TAG}'
CAND_DIR = f'{OUT_DIR}/candidates_{TAG}'
REPORT_DIR = f'{OUT_DIR}/reports'

MODEL_KEY = ARGS.model
RUN_BM25 = ARGS.bm25
RUN_DENSE = not ARGS.no_dense
SEED = ARGS.seed

MODEL_PRESETS = {
    'bge-base': {
        'path':         '/data/bge-base-zh-v1.5',
        'query_prompt': '为这个句子生成表示以用于检索相关文章：',   # 训练和评测（09）必须用同一句
        'dtype':        'float16',
    },
    'qwen3-0.6b': {
        'path':         '/data/Qwen3-Embedding-0.6B',
        'query_prompt': 'Instruct: Given a web search query, retrieve relevant passages '
                        'that answer the query\nQuery:',
        'dtype':        'bfloat16',   # Qwen 官方建议 bf16，fp16 可能溢出
    },
}

MAX_SEQ_LEN = 512
TOP_K = ARGS.top_k                         # 每个问题保留的候选数（排除之后）
RETRIEVE_DEPTH = max(ARGS.depth, TOP_K)    # 先取前 N 名，排除同分量的答案后再截到 TOP_K
ENCODE_BATCH = 128
SEARCH_BATCH = 2048
BM25_K1, BM25_B = 1.2, 0.75                # Lucene 默认值

# 以下三项要与 08 保持一致
MARGIN = 0.95              # 分数限制：候选得分不能超过最高正例得分的 95%（NV-Retriever 的 TopK-PercPos 思路）
MAX_POS_CONCEPTS = 3       # 问题没有疾病词时，借用正确答案正文里点名的疾病；超过这个数就不借
CHECK_DEPTH = 30           # 假负例核对表只在前 30 名里抽

N_CHECK = 40               # 核对表里每种关系抽多少条
RANK_BUCKETS = [(1, 10), (11, 30), (31, 100)]

os.makedirs(CAND_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# %%
# =============================================================================
# 基础函数（normalize / qid_of 与 01–04、06、08 完全一致）
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
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# %%
# =============================================================================
# 读入训练问题和训练答案库（只用训练集，不碰验证集和测试集）
# =============================================================================

train = read_jsonl(f'{SPLIT_DIR}/train.jsonl')
corpus = read_jsonl(f'{SPLIT_DIR}/train_corpus.jsonl')

doc_ids = [r['doc_id'] for r in corpus]
doc_text = {r['doc_id']: r['text'] for r in corpus}
doc_idx = {d: i for i, d in enumerate(doc_ids)}

# 答案 -> 连通分量（04 保证每个答案只属于一个分量）
doc_comp = {}
for r in train:
    doc_comp.setdefault(r['doc_id'], r['component_id'])
missing = [d for d in doc_ids if d not in doc_comp]
assert not missing, f'{len(missing)} corpus answers have no component - re-run step 04'

# 去重后的训练问题：同一个归一化问题只检索一次
queries = {}
for r in train:
    qid = qid_of(normalize(r['anchor']))
    q = queries.setdefault(qid, {'qid': qid, 'text': r['anchor'], 'positives': set(),
                                 'component_id': r['component_id'], 'n_pairs': 0})
    assert q['component_id'] == r['component_id'], 'one question in two components'
    q['positives'].add(r['doc_id'])
    q['n_pairs'] += 1

qids = list(queries)                      # 按 train.jsonl 中首次出现的顺序，可复现
q_texts = [queries[q]['text'] for q in qids]
q_comp = [queries[q]['component_id'] for q in qids]
q_pos = [[doc_idx[d] for d in sorted(queries[q]['positives'])] for q in qids]

write_jsonl(f'{CAND_DIR}/train_queries.jsonl',
            [{'qid': q, 'text': queries[q]['text'],
              'positives': sorted(queries[q]['positives']),
              'component_id': queries[q]['component_id'],
              'n_pairs': queries[q]['n_pairs']} for q in qids])

print('=' * 70)
print(f'Step 07 (new) · Candidate mining  (MODEL_KEY={MODEL_KEY}, RUN_BM25={RUN_BM25}, '
      f'TOP_K={TOP_K}, DEPTH={RETRIEVE_DEPTH})')
print('=' * 70)
print(f'\n  Training pairs       : {len(train):,}')
print(f'  Unique questions     : {len(qids):,}')
print(f'  Training answer pool : {len(doc_ids):,}')

# %%
# =============================================================================
# 公共：排除同分量答案，截到 TOP_K
# =============================================================================


def finalize(i, top_ix, top_sc, pos_scores, drop_zero=False):
    '''
    i          问题下标
    top_ix/sc  排除前的前 RETRIEVE_DEPTH 名（答案下标、分数）
    pos_scores {doc_id: 分数}，该问题每个正确答案的分数
    drop_zero  BM25 用：得分 <= 0 的候选没有任何词面重合，丢弃
    '''
    comp = q_comp[i]
    pos_set = set(q_pos[i])
    docs, scores = [], []
    for j, s in zip(top_ix, top_sc):
        j = int(j)
        d = doc_ids[j]
        if j in pos_set or doc_comp[d] == comp:
            continue
        if drop_zero and s <= 0:
            continue
        docs.append(d)
        scores.append(round(float(s), 4))
        if len(docs) == TOP_K:
            break
    raw_hit10 = any(int(j) in pos_set for j in top_ix[:10])
    return {'qid': qids[i], 'pos_scores': pos_scores, 'docs': docs, 'scores': scores,
            'positive_in_raw_top10': raw_hit10}


def summarize(name, rows):
    n = len(rows)
    short = sum(len(r['docs']) < TOP_K for r in rows)
    hit = sum(r['positive_in_raw_top10'] for r in rows)
    print(f'\n[{name}]')
    print(f'  questions with fewer than {TOP_K} candidates : {short:,}')
    print(f'  a positive is in the raw top-10 (sanity)     : {hit / n:.1%}')
    return {'questions': n, 'fewer_than_top_k': short, 'positive_in_raw_top10': round(hit / n, 4)}


results, stats = {}, {}

# %%
# =============================================================================
# BM25（jieba 分词 + 稀疏矩阵，自己实现，避免依赖库版本差异）
# =============================================================================

if RUN_BM25:
    import jieba
    import scipy.sparse as sp

    jieba.setLogLevel(60)
    # 把本体同义词加入 jieba 词典，让“帕金森”“脑梗死”等不被切开。
    # 这只会让 BM25 基线变强，不会偏向本体方法。分词前文本已转小写，所以小写形式也要加。
    concepts_path = f'{ONTO_DIR}/concepts.json'
    if not os.path.exists(concepts_path):
        concepts_path = f'{OUT_DIR}/ontology/concepts.json'
    if os.path.exists(concepts_path):
        with open(concepts_path, encoding='utf-8') as f:
            for c in json.load(f)['concepts']:
                for t in list(c['synonyms']) + list(c['extra']):
                    jieba.add_word(t)
                    jieba.add_word(t.lower())

    TOKEN_RE = re.compile(r'[\u4e00-\u9fa5a-z0-9]')

    def tokenize(text):
        t = unicodedata.normalize('NFKC', text or '').lower()
        return [w for w in jieba.lcut(t) if TOKEN_RE.search(w)]

    doc_toks = [tokenize(doc_text[d]) for d in tqdm(doc_ids, desc='BM25 tokenize corpus')]
    vocab = {}
    for toks in doc_toks:
        for w in toks:
            vocab.setdefault(w, len(vocab))

    # 文档矩阵：每个位置存 BM25 词权重 idf * tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl))
    N = len(doc_ids)
    dl = np.array([len(t) for t in doc_toks], dtype=np.float32)
    avgdl = float(dl.mean())
    ri, ci, tv = [], [], []
    for i, toks in enumerate(doc_toks):
        for w, tf in Counter(toks).items():
            ri.append(i)
            ci.append(vocab[w])
            tv.append(tf)
    ri, ci, tv = np.array(ri), np.array(ci), np.array(tv, dtype=np.float32)
    df = np.bincount(ci, minlength=len(vocab)).astype(np.float32)
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    wt = idf[ci] * tv * (BM25_K1 + 1) / (tv + BM25_K1 * (1 - BM25_B + BM25_B * dl[ri] / avgdl))
    DT = sp.csr_matrix((wt, (ci, ri)), shape=(len(vocab), N))      # 词 x 文档

    # 查询矩阵：词频（未登录词忽略）
    qr, qc, qv = [], [], []
    for i, text in enumerate(tqdm(q_texts, desc='BM25 tokenize queries')):
        for w_, tf in Counter(tokenize(text)).items():
            if w_ in vocab:
                qr.append(i)
                qc.append(vocab[w_])
                qv.append(tf)
    Q = sp.csr_matrix((np.array(qv, dtype=np.float32), (qr, qc)), shape=(len(qids), len(vocab)))

    bm25_rows = []
    depth = min(RETRIEVE_DEPTH, N)
    for s in tqdm(range(0, len(qids), SEARCH_BATCH), desc='BM25 search'):
        S = (Q[s:s + SEARCH_BATCH] @ DT).toarray()
        part = np.argpartition(-S, depth - 1, axis=1)[:, :depth]
        for j in range(S.shape[0]):
            i = s + j
            ix = part[j][np.argsort(-S[j, part[j]], kind='stable')]
            pos_scores = {doc_ids[p]: round(float(S[j, p]), 4) for p in q_pos[i]}
            bm25_rows.append(finalize(i, ix, S[j, ix], pos_scores, drop_zero=True))

    write_jsonl(f'{CAND_DIR}/bm25.jsonl', bm25_rows)
    results['bm25'] = bm25_rows
    stats['bm25'] = summarize('BM25', bm25_rows)
    print(f'  vocabulary {len(vocab):,} terms, avg answer length {avgdl:.0f} tokens')

# %%
# =============================================================================
# Dense（原始模型，不微调）
# =============================================================================

if RUN_DENSE:
    import torch
    from sentence_transformers import SentenceTransformer

    preset = MODEL_PRESETS[MODEL_KEY]
    use_cuda = torch.cuda.is_available()
    dtype = getattr(torch, preset['dtype']) if use_cuda else torch.float32
    model = SentenceTransformer(preset['path'], device='cuda' if use_cuda else 'cpu',
                                local_files_only=True, model_kwargs={'torch_dtype': dtype})
    model.max_seq_length = MAX_SEQ_LEN

    # 答案不加指令，问题加指令（BGE、Qwen3 都是这样用）
    doc_emb = model.encode([doc_text[d] for d in doc_ids], batch_size=ENCODE_BATCH,
                           normalize_embeddings=True, convert_to_tensor=True,
                           show_progress_bar=True)
    q_emb = model.encode(q_texts, prompt=preset['query_prompt'], batch_size=ENCODE_BATCH,
                         normalize_embeddings=True, convert_to_tensor=True,
                         show_progress_bar=True)

    dense_rows = []
    depth = min(RETRIEVE_DEPTH, len(doc_ids))
    with torch.no_grad():
        for s in tqdm(range(0, len(qids), SEARCH_BATCH), desc='Dense search'):
            sims = (q_emb[s:s + SEARCH_BATCH] @ doc_emb.T).float()
            sc, ix = torch.topk(sims, k=depth, dim=1)
            sc, ix = sc.cpu().numpy(), ix.cpu().numpy()
            for j in range(sims.shape[0]):
                i = s + j
                p = q_pos[i]
                ps = sims[j, p].tolist()
                pos_scores = {doc_ids[k]: round(float(v), 4) for k, v in zip(p, ps)}
                dense_rows.append(finalize(i, ix[j], sc[j], pos_scores))

    write_jsonl(f'{CAND_DIR}/dense_{MODEL_KEY}.jsonl', dense_rows)
    results['dense'] = dense_rows
    stats['dense'] = summarize(f'Dense ({MODEL_KEY})', dense_rows)

    del model, doc_emb, q_emb
    if use_cuda:
        torch.cuda.empty_cache()

# %%
# =============================================================================
# 分数统计（不需要标签）：候选离正例有多近
# =============================================================================


def cap_of(r):
    '''分数限制：最高正例得分的 MARGIN 倍（得分为负时按绝对值处理，不会反向）'''
    top = max(r['pos_scores'].values())
    return top - (1 - MARGIN) * abs(top)


def margin_by_bucket(rows):
    '''每个排名区间里，得分超过分数限制（与正例太接近）的候选占多少'''
    out = {}
    for lo, hi in RANK_BUCKETS:
        over = tot = 0
        for r in rows:
            cap = cap_of(r)
            for s in r['scores'][lo - 1:hi]:
                tot += 1
                over += s > cap
        out[f'{lo}-{hi}'] = round(over / max(tot, 1), 4)
    return out


def ratio_by_rank(rows, ranks=(1, 5, 10, 30, 100)):
    '''候选得分 / 最高正例得分 的中位数（只算最高正例得分为正的问题）。用来决定 MARGIN'''
    out = {}
    for k in ranks:
        v = [r['scores'][k - 1] / max(r['pos_scores'].values()) for r in rows
             if len(r['docs']) >= k and max(r['pos_scores'].values()) > 0]
        out[str(k)] = round(statistics.median(v), 4) if v else None
    return out


if 'dense' in results:
    mb = margin_by_bucket(results['dense'])
    rr = ratio_by_rank(results['dense'])
    stats['dense']['over_margin_by_rank'] = mb
    stats['dense']['score_ratio_median_by_rank'] = rr
    print(f'\nDense candidates scoring above {MARGIN:.0%} of the best positive (too close to the answer)')
    for b, v in mb.items():
        print(f'  rank {b:<7s} {v:.1%}')
    print('Median candidate score / best positive score, by rank')
    print('  ' + '   '.join(f'rank {k}: {v}' for k, v in rr.items()))

# %%
# =============================================================================
# 本体预览：候选和问题的疾病关系（需要先跑 06_本体标注_neuro_new.py）
# =============================================================================
# 08 必须用同一条规则（same / sibling / different / unnamed / unknown）。

labels_ok = all(os.path.exists(f'{ONTO_DIR}/{f}')
                for f in ('concepts.json', 'question_labels.jsonl', 'doc_labels.jsonl'))
if labels_ok:
    with open(f'{ONTO_DIR}/concepts.json', encoding='utf-8') as f:
        _concepts = json.load(f)['concepts']
    labels_ok = all('family' in c for c in _concepts)
if not labels_ok:
    print(f'\n  [skip] new ontology labels not found in {ONTO_DIR}: run 06_本体标注_neuro_new.py first')
else:
    parent = {c['id']: c['parent'] for c in _concepts}
    family = {c['id']: c['family'] for c in _concepts}

    def lineage(c):
        out = {c}
        while parent.get(c):
            c = parent[c]
            out.add(c)
        return out

    lin = {c: lineage(c) for c in parent}
    # rel_set[x]：与概念 x 相同、或互为上下位的全部概念（脑卒中 / 脑梗死）
    rel_set = {x: {c for c in parent if c in lin[x] or x in lin[c]} for x in parent}

    q_con = {x['qid']: set(x['concepts']) for x in read_jsonl(f'{ONTO_DIR}/question_labels.jsonl')}
    EMPTY = (frozenset(), frozenset(), False, set())
    doc_sets = {}       # doc_id -> (全部概念, 大类, 正文是否点名, 正文点名的概念)
    for x in read_jsonl(f'{ONTO_DIR}/doc_labels.jsonl'):
        allc = set(x['mentioned']) | set(x['inherited'])
        doc_sets[x['doc_id']] = (frozenset(allc), frozenset(family[c] for c in allc),
                                 bool(x['mentioned']), set(x['mentioned']))

    def q_concept_set(qid):
        '''问题的疾病集合；问题里没有疾病词时，借用正确答案正文点名的疾病'''
        cs = q_con.get(qid)
        if cs:
            return cs, 'question'
        ment = set()
        for d in queries[qid]['positives']:
            ment |= doc_sets.get(d, EMPTY)[3]
        if 0 < len(ment) <= MAX_POS_CONCEPTS:
            return ment, 'positive'
        return set(), 'none'

    q_ctx = {}          # qid -> (与问题相关的概念集合, 问题的大类, 来源)；没有疾病标签的问题不在里面
    q_src = Counter()
    for qid in qids:
        cs, how = q_concept_set(qid)
        q_src[how] += 1
        if cs:
            q_ctx[qid] = (set().union(*(rel_set[c] for c in cs)), {family[c] for c in cs}, how)

    def relation(qid, doc):
        ctx = q_ctx.get(qid)
        if ctx is None:
            return 'unknown'
        q_rel, q_fam, _ = ctx
        d_all, d_fam, named, _ = doc_sets.get(doc, EMPTY)
        if d_all & q_rel:
            return 'same'
        if not named:
            return 'unnamed'
        return 'sibling' if d_fam & q_fam else 'different'

    REL_KEYS = ['same', 'sibling', 'different', 'unnamed']
    TIERS = {'different': {'different'},
             'different+sibling': {'different', 'sibling'},
             'not-same': {'different', 'sibling', 'unnamed'}}

    def rel_by_bucket(rows):
        lab = [r for r in rows if r['qid'] in q_ctx]
        out = {}
        for lo, hi in RANK_BUCKETS:
            c, tot = Counter(), 0
            for r in lab:
                for d in r['docs'][lo - 1:hi]:
                    c[relation(r['qid'], d)] += 1
                    tot += 1
            out[f'{lo}-{hi}'] = {'n': tot, **{k: round(c[k] / max(tot, 1), 4) for k in REL_KEYS}}
        return out

    def eligible_stats(rows, use_margin):
        '''有疾病标签的问题里，在前 30 / 前 TOP_K 名内至少有一个可用负例的比例（08 的本体规则能不能生效）'''
        lab = [r for r in rows if r['qid'] in q_ctx]
        out = {}
        for name, ok in TIERS.items():
            for depth in (30, TOP_K):
                n = 0
                for r in lab:
                    cap = cap_of(r) if use_margin else float('inf')
                    for d, s in zip(r['docs'][:depth], r['scores'][:depth]):
                        if s <= cap and relation(r['qid'], d) in ok:
                            n += 1
                            break
                out[f'{name}@{depth}'] = round(n / max(len(lab), 1), 4)
        return out

    n_q = len(qids)
    print(f'\nQuestion disease source: from the question {q_src["question"]:,} ({q_src["question"] / n_q:.1%}), '
          f'borrowed from its positive answer {q_src["positive"]:,} ({q_src["positive"] / n_q:.1%}), '
          f'none {q_src["none"]:,} ({q_src["none"] / n_q:.1%})')
    stats['question_disease_source'] = dict(q_src)

    for name, rows in results.items():
        bk = rel_by_bucket(rows)
        el = eligible_stats(rows, use_margin=(name == 'dense'))
        stats[name]['relation_by_rank'] = bk
        stats[name]['eligible_questions'] = el
        print(f'\n  [{name}] relation of candidates to the question (questions with a disease label)')
        print('    ' + 'rank'.ljust(9) + 'n'.rjust(11) + ''.join(k.rjust(11) for k in REL_KEYS))
        for b, v in bk.items():
            print('    ' + b.ljust(9) + format(v['n'], ',').rjust(11)
                  + ''.join(format(v[k], '.1%').rjust(11) for k in REL_KEYS))
        print(f'  [{name}] share of labelled questions with at least one usable negative'
              + (f' (score <= {MARGIN:.0%} of best positive)' if name == 'dense' else ' (no score limit)'))
        for k, v in el.items():
            print(f'    {k:<24s} {v:.1%}')

# %%
# =============================================================================
# 假负例核对表（盲评）
# =============================================================================
# 从 Dense 前 CHECK_DEPTH 名、分数限制之内的候选里，按关系 same / sibling / different / unnamed
# 各抽 N_CHECK 条，混在一起打乱。逐条判断：候选答案是否也正确回答了问题（y / n / ?）。
# 判断完再打开 _key 文件，统计四组的 y 比例。
#
# 预期（规则有用的话）：same 组 y 比例最高，different 组最低，unnamed 组介于其间或偏高。
# 如果 different 组 y 比例仍然很高，说明本体标签本身不可靠，要先改标签，不要急着训练。
# 只需要核对 bge-base 这一份。

if labels_ok and 'dense' in results:
    rng = random.Random(SEED)
    pools = defaultdict(list)
    for r in results['dense']:
        if r['qid'] not in q_ctx:
            continue
        cap = cap_of(r)
        for rank, (d, s) in enumerate(zip(r['docs'][:CHECK_DEPTH], r['scores'][:CHECK_DEPTH]), 1):
            if s <= cap:
                pools[relation(r['qid'], d)].append((r, rank, d, s))

    picked = []
    for g in REL_KEYS:
        pool = pools[g]
        picked += [(g, x) for x in rng.sample(pool, min(N_CHECK, len(pool)))]
    rng.shuffle(picked)

    sheet = f'{CAND_DIR}/fn_check_{MODEL_KEY}.csv'
    key = f'{CAND_DIR}/fn_check_{MODEL_KEY}_key.csv'
    with open(sheet, 'w', encoding='utf-8-sig', newline='') as f1, \
         open(key, 'w', encoding='utf-8-sig', newline='') as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(['id', 'question', 'reference_answer', 'candidate_answer',
                     'candidate_also_answers_question (y/n/?)'])
        w2.writerow(['id', 'group', 'rank', 'score', 'positive_score', 'concept_source', 'qid', 'doc_id'])
        for k, (g, (r, rank, d, s)) in enumerate(picked, 1):
            ref = max(r['pos_scores'], key=r['pos_scores'].get)
            w1.writerow([k, queries[r['qid']]['text'], doc_text[ref][:300], doc_text[d][:300], ''])
            w2.writerow([k, g, rank, s, r['pos_scores'][ref], q_ctx[r['qid']][2], r['qid'], d])

    stats['dense']['fn_check'] = {'pool_sizes': {g: len(pools[g]) for g in REL_KEYS},
                                  'sampled_each': {g: min(N_CHECK, len(pools[g])) for g in REL_KEYS}}
    print(f'\nFalse-negative check sheet : {sheet}   ({len(picked)} rows)')
    print(f'Answer key (open after)    : {key}')
    print('  pools: ' + ', '.join(f'{g} {len(pools[g]):,}' for g in REL_KEYS))

# %%
# =============================================================================
# 报告
# =============================================================================

report = {
    'step': '07_candidates_new', 'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'model_key': MODEL_KEY, 'model': MODEL_PRESETS[MODEL_KEY], 'run_bm25': RUN_BM25,
    'config': {'MAX_SEQ_LEN': MAX_SEQ_LEN, 'TOP_K': TOP_K, 'RETRIEVE_DEPTH': RETRIEVE_DEPTH,
               'BM25_K1': BM25_K1, 'BM25_B': BM25_B, 'MARGIN': MARGIN,
               'MAX_POS_CONCEPTS': MAX_POS_CONCEPTS, 'CHECK_DEPTH': CHECK_DEPTH,
               'N_CHECK': N_CHECK, 'SEED': SEED},
    'unique_questions': len(qids), 'answer_pool': len(doc_ids),
    'stats': stats,
}
path = f'{REPORT_DIR}/07_candidates_{MODEL_KEY}_{TAG}.json'
with open(path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

# ---- Markdown 报告 ----
def _md_table_07(headers, rows):
    esc = lambda v: str(v).replace('|', '/').replace('\n', ' ')
    align = [':---'] + ['---:'] * (len(headers) - 1)
    lines = ['| ' + ' | '.join(map(esc, headers)) + ' |',
             '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(map(esc, r)) + ' |' for r in rows]
    return '\n'.join(lines)

_cfg = report['config']
_st  = report['stats']
_md7 = [
    f'# Step 07 (new) · Candidate mining — `{MODEL_KEY}`', '',
    'Generated {}. TOP_K {}, RETRIEVE_DEPTH {}, MARGIN {}.'.format(
        report['generated_at'], _cfg['TOP_K'], _cfg['RETRIEVE_DEPTH'], _cfg['MARGIN']), '',
    '## Overview', '',
    _md_table_07(['Item', 'Value'], [
        ['Unique training questions', '{:,}'.format(report['unique_questions'])],
        ['Training answer pool',      '{:,}'.format(report['answer_pool'])],
        ['Run BM25',                  str(RUN_BM25)],
    ]), '',
]

for src in ([('bm25', 'BM25')] if RUN_BM25 else []) + [('dense', f'Dense ({MODEL_KEY})')]:
    key_s, label = src
    if key_s not in _st:
        continue
    s = _st[key_s]
    _md7 += [f'## {label}', '',
             _md_table_07(['Item', 'Value'], [
                 ['Questions with < TOP_K candidates', '{:,}'.format(s.get('questions_below_topk', 0))],
                 ['Positive in raw top-10 (sanity)',   '{:.1%}'.format(s.get('positive_in_top10', 0))],
             ]), '']
    if key_s == 'bm25':
        _md7 += [_md_table_07(['Item', 'Value'], [
            ['Vocabulary terms', '{:,}'.format(s.get('vocab_size', 0))],
            ['Avg answer length (tokens)', '{:.0f}'.format(s.get('avg_doc_len', 0))],
        ]), '']
    if key_s == 'dense' and 'over_margin_by_rank' in s:
        _md7 += ['### Candidates above score limit ({} × best positive)'.format(MARGIN), '',
                 _md_table_07(['Rank band', 'Share above limit'],
                              [[k, '{:.1%}'.format(v)] for k, v in s['over_margin_by_rank'].items()]), '',
                 '### Median candidate score / best positive score', '',
                 _md_table_07(['Rank', 'Median ratio'],
                              [[k, '{:.4f}'.format(v)] for k, v in s.get('score_ratio_median_by_rank', {}).items()]), '']
    if 'relation_by_rank' in s:
        _md7 += ['### Relation of candidates to the question (labelled questions)', '',
                 _md_table_07(['Rank band', 'n'] + REL_KEYS,
                              [[band,
                                '{:,}'.format(d['n']),
                                *['{:.1%}'.format(d.get(r, 0)) for r in REL_KEYS]]
                               for band, d in s['relation_by_rank'].items()]), '']
    if 'eligible_by_rel' in s:
        _md7 += ['### Questions with ≥1 usable candidate under score limit', '',
                 _md_table_07(['Criterion', 'Share'],
                              [[k, '{:.1%}'.format(v)] for k, v in s['eligible_by_rel'].items()]), '']
    if 'fn_check' in s:
        fc = s['fn_check']
        _md7 += ['### False-negative check sheet', '',
                 _md_table_07(['Pool', 'Size', 'Sampled'],
                              [[g, '{:,}'.format(fc['pool_sizes'].get(g, 0)),
                                fc['sampled_each'].get(g, 0)] for g in REL_KEYS]), '',
                 f'Sheet: `{CAND_DIR}/fn_check_{MODEL_KEY}.csv`  '
                 f'Key: `{CAND_DIR}/fn_check_{MODEL_KEY}_key.csv`', '']

if 'question_disease_source' in _st.get('dense', {}):
    src_d = _st['dense']['question_disease_source']
    _md7 += ['## Question disease label source', '',
             _md_table_07(['Source', 'Questions'],
                          [[k, '{:,}'.format(v)] for k, v in src_d.items()]), '']

_md7 += ['## Output files', '',
         f'`{CAND_DIR}/` — bm25.jsonl (if run), dense_{MODEL_KEY}.jsonl, '
         f'fn_check_{MODEL_KEY}.csv/.key.csv, train_queries.jsonl']

md_path = f'{REPORT_DIR}/07_candidates_{MODEL_KEY}_{TAG}.md'
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(_md7) + '\n')

print(f'\nOutputs: {CAND_DIR}/')
print(f'Reports: {path}  /  {md_path}')
print('Next   : 08_training_sets_new.py')
