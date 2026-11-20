import os, sys
sys.modules['flash_attn'] = None
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import argparse, datetime, json, math, re, statistics, unicodedata
from collections import Counter, defaultdict

try:
    from tqdm.auto import tqdm
except ImportError:
    class tqdm:
        def __init__(self, it=None, **kw):
            self.it = it
        def __iter__(self):
            return iter(self.it)
        def update(self, n=1):
            pass
        def set_postfix_str(self, s):
            pass
        def close(self):
            pass

NL = chr(10)


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',    default=os.environ.get('LIVER_OUT_DIR', '/home/liver/data/liver_dataset'))
    ap.add_argument('--model-root', default=os.environ.get('LIVER_MODEL_ROOT', '/data/finetuned_models'))
    ap.add_argument('--tag',        default='new', help='09 的结果标签，决定读取哪一批已训练模型')
    ap.add_argument('--models',     default='bge-base,qwen3-0.6b')
    ap.add_argument('--methods',    default='mnrl,random,onto_pool_rand,margin')
    ap.add_argument('--seeds',      default='42')
    ap.add_argument('--top-k',      type=int, default=100)
    ap.add_argument('--lambdas',    default='0.01,0.02,0.03,0.05,0.08,0.12')
    ap.add_argument('--bm25-lambdas', default='0.005,0.01,0.02,0.03,0.05')
    ap.add_argument('--report',     action='store_true', help='只根据已有结果出报告')
    ap.add_argument('--overwrite',  action='store_true', help='已有结果也重新计算')
    a, _ = ap.parse_known_args()
    return a


ARGS       = _args()
OUT_DIR    = ARGS.out_dir
TAG        = ARGS.tag
SPLIT_DIR  = f'{OUT_DIR}/split'
ONTO_DIR   = f'{OUT_DIR}/ontology_new'
REPORTS    = f'{OUT_DIR}/reports'
RR_DIR     = f'{OUT_DIR}/rerank_{TAG}'
MODEL_ROOT = ARGS.model_root
DOMAIN     = 'liver'
TOP_K      = ARGS.top_k
LAMBDAS    = [float(x) for x in ARGS.lambdas.split(',') if x]
BM25_LAMBDAS = [float(x) for x in ARGS.bm25_lambdas.split(',') if x]
MODEL_LIST = [m for m in ARGS.models.split(',') if m]
METHOD_LIST = [m for m in ARGS.methods.split(',') if m]
SEEDS      = [int(s) for s in ARGS.seeds.split(',') if s]

MODELS = {   # 与 09 一致
    'bge-base':   {'query_prompt': '为这个句子生成表示以用于检索相关文章：', 'dtype': 'float16', 'batch': 128},
    'qwen3-0.6b': {'query_prompt': 'Instruct: Given a web search query, retrieve relevant passages that answer the query' + NL + 'Query:',
                   'dtype': 'bfloat16', 'batch': 32},
}

RULES = ['none', 'same_up', 'diff_down', 'both', 'hard_down', 'bm25_fuse']
RULE_DESC = {
    'none':      'no reranking (original dense ranking)',
    'same_up':   'same-disease candidates +λ',
    'diff_down': 'different-family candidates −λ',
    'both':      'same +λ and different −λ',
    'hard_down': 'different-family candidates moved below all others',
    'bm25_fuse': 'non-ontology control: + λ · normalised BM25 score',
}
METRICS = ['ndcg@10', 'mrr@10', 'recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@100']
PERM_ITERS, BOOT_ITERS = 10000, 2000

os.makedirs(RR_DIR, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)


def read_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def write_jsonl(p, rows):
    with open(p, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + NL)


# =============================================================================
# 本体标注器：用 06 的 concepts.json 按同一套规则重建
# =============================================================================

try:
    from opencc import OpenCC
    _t2s = OpenCC('t2s').convert
    T2S_BACKEND = 'opencc'
except ImportError:
    _t2s = lambda s: s
    T2S_BACKEND = 'none (opencc not installed; traditional characters are not converted)'
    print('warning: opencc not installed — 06 may have used it, labels can differ slightly. '
          'pip install opencc-python-reimplemented')


def prep(text):
    return _t2s(unicodedata.normalize('NFKC', text or '')).upper()


_cpath = f'{ONTO_DIR}/concepts.json'
assert os.path.exists(_cpath), _cpath + ' not found — run 06_本体标注_new.py first'
CONCEPTS = load_json(_cpath)['concepts']
concept_by_id = {c['id']: c for c in CONCEPTS}
FAMILY = {c['id']: c['family'] for c in CONCEPTS}

term2concepts = defaultdict(set)
for c in CONCEPTS:
    for t in list(c.get('synonyms', [])) + list(c.get('extra', {})):
        term2concepts[prep(t)].add(c['id'])
ascii_terms = {t for t in term2concepts if re.fullmatch(r'[A-Z0-9\-]+', t)}
cjk_terms = sorted((t for t in term2concepts if t not in ascii_terms), key=len, reverse=True)
CJK_RE = re.compile('|'.join(map(re.escape, cjk_terms)))
ASCII_RE = (re.compile(r'(?<![A-Z])(' + '|'.join(map(re.escape, sorted(ascii_terms, key=len, reverse=True)))
                       + r')(?![A-Z])') if ascii_terms else None)

_anc = {}


def ancestors(cid):
    if cid not in _anc:
        out, c = [], cid
        while concept_by_id[c].get('parent'):
            c = concept_by_id[c]['parent']
            out.append(c)
        _anc[cid] = out
    return _anc[cid]


def lineage(c):
    return {c, *ancestors(c)}


def related(a, b):
    return a in lineage(b) or b in lineage(a)


def tag_text(text):
    '''文本点名的概念（最长匹配；同时命中父子概念时只留子概念），与 06 一致'''
    q = prep(text)
    cnt = Counter()
    for m in CJK_RE.finditer(q):
        for c in term2concepts[m.group(0)]:
            cnt[c] += 1
    if ASCII_RE:
        for m in ASCII_RE.finditer(q):
            for c in term2concepts[m.group(1)]:
                cnt[c] += 1
    redundant = {a for c in cnt for a in ancestors(c)}
    return frozenset(c for c in cnt if c not in redundant)


def relation(q_cs, d_cs):
    '''与 08 同一规则：same / sibling / different / unnamed / unknown'''
    if not q_cs:
        return 'unknown'
    if any(related(c, x) for c in d_cs for x in q_cs):
        return 'same'
    if not d_cs:
        return 'unnamed'
    q_fam = {FAMILY[c] for c in q_cs}
    d_fam = {FAMILY[c] for c in d_cs}
    return 'sibling' if q_fam & d_fam else 'different'


# =============================================================================
# 数据：验证/测试的问题、检索库、正确答案
# =============================================================================

def load_ir(split):
    queries = read_jsonl(f'{SPLIT_DIR}/{split}_queries.jsonl')
    corpus  = read_jsonl(f'{SPLIT_DIR}/{split}_corpus.jsonl')
    rel = defaultdict(set)
    for r in read_jsonl(f'{SPLIT_DIR}/{split}_qrels.jsonl'):
        rel[r['qid']].add(r['doc_id'])
    return queries, corpus, rel


def query_metrics(ranked, rel):
    dcg  = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    rr   = next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if d in rel), 0.0)
    m = {'ndcg@10': dcg / idcg if idcg else 0.0, 'mrr@10': rr}
    for k in (1, 5, 10, 20, 100):
        m['recall@{}'.format(k)] = len(rel & set(ranked[:k])) / len(rel)
    return m


_cache = {}


def split_data(split):
    '''问题、检索库、正确答案，以及双方的疾病标签（按 split 缓存）'''
    if split in _cache:
        return _cache[split]
    queries, corpus, rel = load_ir(split)
    q_lab = {q['qid']: tag_text(q['text']) for q in tqdm(queries, desc=f'  [{split}] label queries', leave=False)}
    d_lab = {d['doc_id']: tag_text(d['text']) for d in tqdm(corpus, desc=f'  [{split}] label answers', leave=False)}
    data = {'queries': queries, 'corpus': corpus, 'rel': rel, 'q_lab': q_lab, 'd_lab': d_lab}
    _cache[split] = data
    print('  [{}] {:,} queries ({:.1%} with a disease label), {:,} answers ({:.1%} name a disease)'.format(
        split, len(queries), sum(bool(v) for v in q_lab.values()) / max(len(q_lab), 1),
        len(corpus), sum(bool(v) for v in d_lab.values()) / max(len(d_lab), 1)), flush=True)
    return data


# =============================================================================
# BM25（对照信号，只用于 bm25_fuse）
# =============================================================================

_bm25 = {}


def bm25_scorer(split):
    if split in _bm25:
        return _bm25[split]
    import jieba
    jieba.setLogLevel(60)
    for c in CONCEPTS:
        for t in list(c.get('synonyms', [])) + list(c.get('extra', {})):
            jieba.add_word(t.lower())
    tok_re = re.compile(r'[\u4e00-\u9fa5a-z0-9]')

    def tok(t):
        t = unicodedata.normalize('NFKC', t or '').lower()
        return [w for w in jieba.lcut(t) if tok_re.search(w)]

    corpus = split_data(split)['corpus']
    docs = {d['doc_id']: Counter(tok(d['text'])) for d in tqdm(corpus, desc=f'  [{split}] BM25 index', leave=False)}
    dl = {d: sum(c.values()) for d, c in docs.items()}
    avgdl = sum(dl.values()) / max(len(dl), 1)
    df = Counter(w for c in docs.values() for w in c)
    N = len(docs)
    idf = {w: math.log(1 + (N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
    k1, b = 1.2, 0.75

    def score(qtext, doc_id):
        c = docs[doc_id]
        s = 0.0
        for w in set(tok(qtext)):
            tf = c.get(w, 0)
            if tf:
                s += idf[w] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[doc_id] / avgdl))
        return s

    _bm25[split] = score
    return score


# =============================================================================
# 检索 + 重排
# =============================================================================

def retrieve(model_key, model_path, split):
    '''返回 {qid: [(doc_id, cos), ...]}，按余弦降序，长度 TOP_K'''
    import torch
    from sentence_transformers import SentenceTransformer
    cfg = MODELS[model_key]
    data = split_data(split)
    model = SentenceTransformer(model_path, device='cuda', local_files_only=True,
                                model_kwargs={'torch_dtype': getattr(torch, cfg['dtype'])})
    doc_ids = [d['doc_id'] for d in data['corpus']]
    print('    encoding {} corpus ({:,} answers) and queries ({:,})'.format(
        split, len(data['corpus']), len(data['queries'])), flush=True)
    d_emb = model.encode([d['text'] for d in data['corpus']], batch_size=cfg['batch'],
                         normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
    q_emb = model.encode([q['text'] for q in data['queries']], prompt=cfg['query_prompt'],
                         batch_size=cfg['batch'], normalize_embeddings=True,
                         convert_to_tensor=True, show_progress_bar=True)
    out = {}
    with torch.no_grad():
        for s in range(0, len(data['queries']), 1024):
            sims = (q_emb[s:s + 1024] @ d_emb.T).float()
            val, idx = torch.topk(sims, k=min(TOP_K, len(doc_ids)), dim=1)
            for j, (vrow, irow) in enumerate(zip(val.cpu().tolist(), idx.cpu().tolist())):
                out[data['queries'][s + j]['qid']] = [(doc_ids[i], v) for i, v in zip(irow, vrow)]
    del model
    torch.cuda.empty_cache()
    return out


def rerank(cands, q_cs, d_lab, rule, lam, bm25_fn=None, qtext=None):
    '''返回重排后的 doc_id 列表'''
    if rule == 'none':
        return [d for d, _ in cands]
    if rule == 'bm25_fuse':
        raw = [bm25_fn(qtext, d) for d, _ in cands]
        mx = max(raw) or 1.0
        scored = [(c + lam * r / mx, d) for (d, c), r in zip(cands, raw)]
        return [d for _, d in sorted(scored, key=lambda x: -x[0])]
    if not q_cs:
        return [d for d, _ in cands]
    scored = []
    for rank, (d, c) in enumerate(cands):
        r = relation(q_cs, d_lab.get(d, frozenset()))
        if rule == 'hard_down':
            scored.append(((1 if r == 'different' else 0), -c, rank, d))
            continue
        adj = 0.0
        if r == 'same' and rule in ('same_up', 'both'):
            adj = lam
        elif r == 'different' and rule in ('diff_down', 'both'):
            adj = -lam
        scored.append((0, -(c + adj), rank, d))
    return [x[3] for x in sorted(scored)]


def evaluate(ret, split, rule, lam):
    data = split_data(split)
    bm25_fn = bm25_scorer(split) if rule == 'bm25_fuse' else None
    qtext = {q['qid']: q['text'] for q in data['queries']}
    src = {q['qid']: q.get('source', '') for q in data['queries']}
    per_q = []
    for qid, cands in ret.items():
        q_cs = data['q_lab'].get(qid, frozenset())
        ranked = rerank(cands, q_cs, data['d_lab'], rule, lam, bm25_fn, qtext[qid])
        m = query_metrics(ranked, data['rel'][qid])
        top10 = ranked[:10]
        if q_cs:
            rels = [relation(q_cs, data['d_lab'].get(d, frozenset())) for d in top10]
            m['cross_family@10'] = sum(r == 'different' for r in rels) / max(len(rels), 1)
            m['same@10'] = sum(r == 'same' for r in rels) / max(len(rels), 1)
        per_q.append({'qid': qid, 'source': src[qid], 'labelled': bool(q_cs), **m, 'top10': top10})
    return per_q


def summarise(per_q):
    out = {k: sum(r[k] for r in per_q) / len(per_q) for k in METRICS}
    lab = [r for r in per_q if r['labelled']]
    out['labelled_share'] = len(lab) / len(per_q)
    for k in ('cross_family@10', 'same@10'):
        out[k] = sum(r[k] for r in lab) / len(lab) if lab else None
    out['ndcg@10_labelled'] = sum(r['ndcg@10'] for r in lab) / len(lab) if lab else None
    return out


def model_path(model_key, method, seed):
    return f'{MODEL_ROOT}/{DOMAIN}_{model_key}_{method}_seed{seed}_{TAG}'


def run_one(model_key, method, seed):
    out_dir = f'{RR_DIR}/{model_key}/{method}/seed{seed}'
    if os.path.exists(f'{out_dir}/metrics.json') and not ARGS.overwrite:
        print('  skip (done)  {} {} seed{}'.format(model_key, method, seed), flush=True)
        return
    path = model_path(model_key, method, seed)
    if not os.path.exists(path):
        print('  [missing model] ' + path, flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    print('  {} {} seed{}  <- {}'.format(model_key, method, seed, path), flush=True)

    dev_ret = retrieve(model_key, path, 'dev')
    tst_ret = retrieve(model_key, path, 'test')

    # 在验证集上为每条规则选 λ
    chosen, dev_curve = {}, {}
    grids = {rule: ([0.0] if rule in ('none', 'hard_down') else
                    BM25_LAMBDAS if rule == 'bm25_fuse' else LAMBDAS) for rule in RULES}
    bar = tqdm(total=sum(len(g) for g in grids.values()), desc='    dev: choose λ', unit='run')
    for rule in RULES:
        best = None
        for lam in grids[rule]:
            bar.set_postfix_str(f'{rule} λ={lam:g}')
            s = summarise(evaluate(dev_ret, 'dev', rule, lam))['ndcg@10']
            dev_curve.setdefault(rule, []).append([lam, s])
            if best is None or s > best[1]:
                best = (lam, s)
            bar.update(1)
        chosen[rule] = best[0]
    bar.close()
    print('    λ chosen on dev: ' + ', '.join(f'{r}={chosen[r]:g}' for r in RULES), flush=True)

    # 测试集只用选好的 λ 评一次
    test, rows = {}, []
    for rule in tqdm(RULES, desc='    test: evaluate', unit='rule'):
        per_q = evaluate(tst_ret, 'test', rule, chosen[rule])
        test[rule] = summarise(per_q)
        for r in per_q:
            rows.append({'rule': rule, **{k: v for k, v in r.items() if k != 'top10'},
                         'top10': r['top10']})
    write_jsonl(f'{out_dir}/per_query.jsonl', rows)
    with open(f'{out_dir}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'model': model_key, 'method': method, 'seed': seed, 'model_path': path,
                   'top_k': TOP_K, 'lambda_chosen_on_dev': chosen, 'dev_curve': dev_curve,
                   'test': test, 't2s_backend': T2S_BACKEND,
                   'generated_at': datetime.datetime.now().isoformat(timespec='seconds')},
                  f, ensure_ascii=False, indent=2)


# =============================================================================
# 统计检验与报告
# =============================================================================

def paired_permutation(a, b, iters=PERM_ITERS, seed=0):
    import numpy as np
    d = np.asarray(a) - np.asarray(b)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(0, iters, 1000):
        n = min(1000, iters - s)
        signs = rng.choice([-1., 1.], size=(n, len(d)))
        hits += int((np.abs((signs * d).mean(axis=1)) >= obs).sum())
    return (hits + 1) / (iters + 1)


def paired_bootstrap_ci(a, b, iters=BOOT_ITERS, seed=0):
    import numpy as np
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, len(d), size=(min(200, iters - s), len(d)))].mean(axis=1)
                            for s in range(0, iters, 200)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, mx = [1.] * len(ps), 0.
    for rank, i in enumerate(order):
        mx = max(mx, min(1., (len(ps) - rank) * ps[i]))
        adj[i] = mx
    return adj


def report():
    md = ['# Step 10 (new) · Ontology reranking', '',
          'Generated {}. Reranks the top-{} candidates of models trained in step 09 (tag `{}`). '
          'λ is chosen on the validation set by nDCG@10; the test set is evaluated once. '
          'Disease labels for validation/test questions come from the question text only; '
          'answer labels come from the answer text only (diseases named in it).'.format(
              datetime.datetime.now().isoformat(timespec='seconds'), TOP_K, TAG), '',
          '| Rule | Description |', '| :--- | :--- |']
    md += ['| {} | {} |'.format(r, RULE_DESC[r]) for r in RULES] + ['']
    out = {}
    for model_key in MODEL_LIST:
        for method in METHOD_LIST:
            runs = []
            for s in SEEDS:
                p = f'{RR_DIR}/{model_key}/{method}/seed{s}/metrics.json'
                if os.path.exists(p):
                    runs.append((s, load_json(p)))
            if not runs:
                continue
            key = f'{model_key}/{method}'
            out[key] = {'seeds': [s for s, _ in runs]}
            md += ['## {} · {}'.format(model_key, method), '',
                   'Seeds: {}. Labelled test questions: {:.1%}.'.format(
                       [s for s, _ in runs], runs[0][1]['test']['none']['labelled_share']), '',
                   '| Rule | λ (dev) | nDCG@10 | MRR@10 | R@1 | R@10 | R@100 | nDCG@10 (labelled) | Cross-family in top-10 | Same-disease in top-10 |',
                   '| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
            for rule in RULES:
                vals = {k: [r['test'][rule][k] for _, r in runs if r['test'][rule].get(k) is not None]
                        for k in ('ndcg@10', 'mrr@10', 'recall@1', 'recall@10', 'recall@100',
                                  'ndcg@10_labelled', 'cross_family@10', 'same@10')}

                def cell(k):
                    v = vals[k]
                    if not v:
                        return '-'
                    sd = statistics.stdev(v) * 100 if len(v) > 1 else 0.
                    return '{:.2f} ± {:.2f}'.format(statistics.mean(v) * 100, sd)
                lams = sorted({r['lambda_chosen_on_dev'][rule] for _, r in runs})
                md.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
                    rule, ','.join('{:g}'.format(x) for x in lams), cell('ndcg@10'), cell('mrr@10'),
                    cell('recall@1'), cell('recall@10'), cell('recall@100'), cell('ndcg@10_labelled'),
                    cell('cross_family@10'), cell('same@10')))
                out[key][rule] = {k: v for k, v in vals.items()}
                out[key][rule]['lambda'] = lams
            md.append('')

            # 配对检验：每条规则 vs none（逐题 nDCG@10，跨种子取平均）
            perq = defaultdict(lambda: defaultdict(list))
            for s, _ in runs:
                for r in read_jsonl(f'{RR_DIR}/{model_key}/{method}/seed{s}/per_query.jsonl'):
                    perq[r['rule']][r['qid']].append(r['ndcg@10'])
            base = {q: sum(v) / len(v) for q, v in perq['none'].items()}
            comps = []
            for rule in RULES[1:]:
                cur = {q: sum(v) / len(v) for q, v in perq[rule].items()}
                qs = sorted(set(base) & set(cur))
                a, b = [cur[q] for q in qs], [base[q] for q in qs]
                lo, hi = paired_bootstrap_ci(a, b)
                comps.append({'rule': rule, 'delta': (sum(a) - sum(b)) / len(qs) * 100,
                              'ci': [lo * 100, hi * 100], 'p': paired_permutation(a, b)})
            adj = holm([c['p'] for c in comps])
            md += ['Paired comparison vs `none` on per-query nDCG@10 (averaged over seeds; Holm over {} rules):'.format(len(comps)), '',
                   '| Rule vs none | Δ nDCG@10 | 95% CI | p | p (Holm) |', '| :--- | ---: | ---: | ---: | ---: |']
            for c, pa in zip(comps, adj):
                c['p_holm'] = pa
                md.append('| {} | {:+.2f} | [{:+.2f}, {:+.2f}] | {:.4f} | {:.4f} |'.format(
                    c['rule'], c['delta'], c['ci'][0], c['ci'][1], c['p'], pa))
            md.append('')
            out[key]['comparisons'] = comps

    md += ['> Cross-family in top-10: share of top-10 answers whose named diseases belong to a different '
           'disease family than the question (labelled questions only). '
           'Same-disease answers are not necessarily relevant; see the dose experiment in step 09.', '']
    with open(f'{REPORTS}/10_rerank_{TAG}.md', 'w', encoding='utf-8') as f:
        f.write(NL.join(md) + NL)
    with open(f'{REPORTS}/10_rerank_{TAG}.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(NL.join(md))
    print(NL + 'Report: ' + f'{REPORTS}/10_rerank_{TAG}.md')


def main():
    if not ARGS.report:
        print('Step 10 · ontology reranking  (t2s: {})'.format(T2S_BACKEND))
        jobs = [(m, meth, s) for m in MODEL_LIST for meth in METHOD_LIST for s in SEEDS]
        for i, (model_key, method, seed) in enumerate(jobs, 1):
            print('[{}/{}] {}'.format(i, len(jobs), datetime.datetime.now().strftime('%m-%d %H:%M')), flush=True)
            run_one(model_key, method, seed)
    report()


if __name__ == '__main__':
    main()
