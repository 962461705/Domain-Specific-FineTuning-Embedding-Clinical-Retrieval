# %%
# =============================================================================
# 配置
# =============================================================================

OUT_DIR    = '/home/liver/data/liver_dataset'
REPORT_DIR = f'{OUT_DIR}/reports'
PAPER_DIR  = f'{OUT_DIR}/paper'
TAB_DIR    = f'{PAPER_DIR}/tables'
FIG_DIR    = f'{PAPER_DIR}/figures'
SPLIT_DIR  = f'{OUT_DIR}/split'
ONTO_DIR   = f'{OUT_DIR}/ontology_new'

MODELS     = ['bge-base', 'qwen3-0.6b']
MODEL_NAME = {'bge-base': 'BGE-base-zh', 'qwen3-0.6b': 'Qwen3-Embedding-0.6B'}
SEEDS      = [42, 43, 44]
NEURO_DIR = '/data/neuro/data/neuro_dataset'
DOMAINS   = [('Liver', OUT_DIR), ('Neurology', NEURO_DIR)]

# 主线方法（顺序即表格顺序）
MAIN_METHODS = ['mnrl', 'bm25', 'dense', 'dense_rank10', 'dense_rank100', 'random',
                'margin', 'margin_onto',
                'onto_pool_rand', 'onto_pool_strat', 'onto_loose_rand', 'onto_strat_loose']
CORE_METHODS = ['onto_pool_rand', 'onto_pool_sim', 'onto_pool_hard', 'onto_hard_loose',
                'onto_pool_mid', 'onto_nmid_loose', 'onto_hmid_loose',
                'bm25', 'bm25_onto', 'dense', 'dense_onto', 'margin', 'margin_onto']
DOSE_LEVELS  = [('onto_pool_rand', 0), ('onto_dose_p10', 10), ('onto_dose_p30', 30), ('onto_dose_p50', 50)]

# 方法名
DISPLAY = {
    'mnrl': 'In-batch only (MNRL)', 'bm25': 'BM25 top-1', 'dense': 'Dense top-1',
    'dense_rank10': 'Dense rank 10', 'dense_rank100': 'Dense rank 100',
    'random': 'Random', 'margin': 'TopK-PercPos',
    'margin_onto': 'TopK-PercPos + ontology filter',
    'onto_pool_rand': 'Ontology pool, random', 'onto_pool_strat': 'Ontology pool, family-stratified',
    'onto_loose_rand': 'Loose pool, random', 'onto_strat_loose': 'Loose pool, family-stratified',
    'onto_pool_sim': 'Ontology pool, entity similarity', 'onto_pool_hard': 'Ontology pool, formula top',
    'onto_hard_loose': 'Loose pool, formula top', 'onto_pool_mid': 'Ontology pool, 2-gram mid band',
    'onto_nmid_loose': 'Loose pool, 2-gram mid band', 'onto_hmid_loose': 'Loose pool, formula mid band',
    'bm25_onto': 'BM25 top-1 + ontology filter', 'dense_onto': 'Dense top-1 + ontology filter',
}
# 方法对照表：是否用检索模型挑负例、候选池、选法
METHOD_INFO = {
    'mnrl':            ('No',  'None',                         'In-batch negatives only'),
    'bm25':            ('No',  'Full corpus (BM25 top-100)',   'Highest BM25 score'),
    'dense':           ('Yes', 'Full corpus (dense top-100)',  'Highest dense score'),
    'dense_rank10':    ('Yes', 'Full corpus (dense top-100)',  'Dense rank 10'),
    'dense_rank100':   ('Yes', 'Full corpus (dense top-100)',  'Dense rank 100'),
    'random':          ('No',  'Full corpus',                  'Uniform random'),
    'margin':          ('Yes', 'Full corpus (dense top-100)',  'TopK-PercPos: score < 95% of positive, random among hardest 5; '
                                                              'if no candidate qualifies, dense rank 100'),
    'margin_onto':     ('Yes', 'Dense top-100 minus same/unnamed', 'As above; if filtering leaves nothing, the TopK-PercPos negative'),
    'onto_pool_rand':  ('No',  'Safe pool',                    'Uniform random'),
    'onto_pool_strat': ('No',  'Safe pool',                    'Family-stratified random'),
    'onto_loose_rand': ('No',  'Loose pool',                   'Uniform random'),
    'onto_strat_loose':('No',  'Loose pool',                   'Family-stratified random'),
}
# 附录方法定义：打分挑难、本体过滤检索候选、剂量实验
METHOD_INFO_APPX = {
    'onto_pool_sim':   ('No',  'Safe pool',                    'Entity IDF cosine + aspect; low-score band'),
    'onto_pool_hard':  ('No',  'Safe pool',                    '0.75·char-bigram Jaccard + 0.10·relation + 0.15·entity Jaccard; top scores under 95% of positive'),
    'onto_hard_loose': ('No',  'Loose pool',                   'Same formula; top scores under 95% of positive'),
    'onto_pool_mid':   ('No',  'Safe pool',                    'Char-bigram Jaccard 30–70% band, family-stratified'),
    'onto_nmid_loose': ('No',  'Loose pool',                   'Char-bigram Jaccard 30–70% band, family-stratified'),
    'onto_hmid_loose': ('No',  'Loose pool',                   'Formula 30–70% band, family-stratified'),
    'bm25_onto':       ('No',  'BM25 top-100 minus same/unnamed', 'Highest remaining BM25 score; if none remains, BM25 rank 1'),
    'dense_onto':      ('Yes', 'Dense top-100 minus same/unnamed', 'Highest remaining dense score; if none remains, dense rank 1'),
    'onto_dose_p10':   ('No',  'Safe pool + same-disease answers', '10% of pairs draw a same-disease answer'),
    'onto_dose_p30':   ('No',  'Safe pool + same-disease answers', '30% of pairs draw a same-disease answer'),
    'onto_dose_p50':   ('No',  'Safe pool + same-disease answers', '50% of pairs draw a same-disease answer'),
}
DISPLAY.update({'onto_dose_p10': 'Ontology pool + 10% same-disease', 'onto_dose_p30': 'Ontology pool + 30% same-disease',
                'onto_dose_p50': 'Ontology pool + 50% same-disease'})
# 本体过滤修复检索式难负例：
FILTER_PAIRS = [('bm25', 'bm25_onto'), ('dense', 'dense_onto'), ('margin', 'margin_onto')]
POOL_NOTE = ('Safe pool: answers that name a disease different from the question\'s disease '
             '(same-disease, unnamed and generic answers removed). Loose pool: all answers except '
             'same-disease ones. Ontology methods use no retrieval model.')
# 本体关系的来源
SRC_LABEL = {'wd': 'Wikidata', 'wd_faers': 'Wikidata (FAERS only)', 'msd': 'MSD Manual',
             'zhwiki': 'Chinese Wikipedia', 'medlineplus': 'MedlinePlus'}

FIG_WIDTH_DOUBLE = 7.16   # IEEE 通栏宽度（英寸）
FIG_WIDTH_SINGLE = 3.5    # IEEE 单栏宽度（英寸）
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00', '#F0E442', '#000000']
REL_KEYS  = ['same', 'sibling', 'different', 'unnamed', 'unknown']
REL_COLOR = {'same': '#D55E00', 'sibling': '#E69F00', 'different': '#009E73',
             'unnamed': '#999999', 'unknown': '#CCCCCC'}

# %%
import csv, json, math, os, re, statistics, unicodedata
from collections import defaultdict
N_STEPS = 26
import time
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it=None, **kw):
        return it
_T0, _STEP = time.time(), [0]


def step(msg):
    '''打印进度：第几步 / 共几步、已用时间、当前在做什么'''
    _STEP[0] += 1
    m, s = divmod(int(time.time() - _T0), 60)
    print(f'[{_STEP[0]:>2}/{N_STEPS}] {m:02d}:{s:02d}  {msg}', flush=True)


os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
SKIPPED = []


def load(path):
    if not os.path.exists(path):
        SKIPPED.append(path)
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def name(m):
    return DISPLAY.get(m, m)


def ms(vals, scale=100):
    '''均值 ± 标准差'''
    v = [x * scale for x in vals]
    if not v:
        return '-'
    sd = statistics.stdev(v) if len(v) > 1 else 0.
    return f'{statistics.mean(v):.2f} ± {sd:.2f}'


def sgn(v, nd=2):
    '''带符号的数；负号用数学负号 −（与图一致，也和表里表示"无"的 "-" 区分）；
    四舍五入后为 0 时不带符号，避免出现 "-0.00"'''
    s = f'{v:+.{nd}f}'
    return f'{0:.{nd}f}' if float(s) == 0 else s.replace('-', '−')


def sgn_pct(v):
    '''带符号的百分数（整数）；负号用数学负号 −；四舍五入后为 0 时写 0%'''
    s = f'{v:+.0%}'
    return '0%' if s in ('+0%', '-0%') else s.replace('-', '−')


# main / core / dose 三套实验分别训练，同一方法在不同表里的分数会略有差异，表注里统一说明
SUITE_NOTE = (' The main, ablation and dose experiments were trained as separate runs with the same recipe '
              'and seeds, so the same method can differ slightly between tables.')


def dose_delta(rep, model):
    '''剂量实验相对 0%（本体池）的逐题配对差值与 95% CI，取自 09 dose 报告：
    [(注入比例, Δ, CI 下界, CI 上界)]，0% 处为 (0, 0, 0, 0)；报告里没有时返回 []'''
    comps = {(c['method'], c['baseline']): c for c in ((rep or {}).get(model) or {}).get('comparisons', [])}
    out = [(0, 0., 0., 0.)]
    for m, pct in DOSE_LEVELS[1:]:
        c = comps.get((m, DOSE_LEVELS[0][0]))
        if c:
            out.append((pct, c['delta_ndcg@10'], c['ci95'][0], c['ci95'][1]))
    return out if len(out) > 1 else []


def plot_dose_delta(ax, pts, color, label, marker='o', dx=0.):
    '''画剂量曲线：Δ 与配对 95% CI；0% 是基准（Δ 恒为 0），不画误差线；dx 为横向错开量，避免误差线重叠'''
    xs = [p[0] + dx for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, marker=marker, ms=3.5, lw=1.0, color=color, label=label)
    ax.errorbar(xs[1:], ys[1:], yerr=[[p[1] - p[2] for p in pts[1:]], [p[3] - p[1] for p in pts[1:]]],
                fmt='none', capsize=2.5, lw=1.0, ecolor=color)


# %%
# =============================================================================
# 本体标注器（规则与 06、10 相同），用于测试集统计
# =============================================================================
# 06 只给训练集打标签，这里用 ontology_new/concepts.json 按同一套规则
# （繁转简、全角转半角、最长匹配、同时命中父子概念时只留子概念）给测试集重新打标签。
# 问题只看问题文本，答案只看答案正文，与 10 的口径一致。

try:
    from opencc import OpenCC
    _t2s = OpenCC('t2s').convert
    T2S_BACKEND = 'opencc'
except ImportError:
    _t2s = lambda s: s
    T2S_BACKEND = 'none'


def _prep(t):
    return _t2s(unicodedata.normalize('NFKC', t or '')).upper()


def make_tagger(onto_dir):
    '''按 onto_dir/concepts.json 建标注器（与 06 同一规则），返回 (tag_text, onto_relation)；没有本体文件时返回 None'''
    path = f'{onto_dir}/concepts.json'
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        concepts = json.load(f)['concepts']
    cby = {c['id']: c for c in concepts}
    cfam = {c['id']: c.get('family', '-') for c in concepts}
    t2c = defaultdict(set)
    for c in concepts:
        for t in list(c.get('synonyms', [])) + list(c.get('extra', {})):
            t2c[_prep(t)].add(c['id'])
    ascii_terms = {t for t in t2c if re.fullmatch(r'[A-Z0-9\-]+', t)}
    cjk_terms = sorted((t for t in t2c if t not in ascii_terms), key=len, reverse=True)
    cjk_re = re.compile('|'.join(map(re.escape, cjk_terms))) if cjk_terms else None
    asc_re = (re.compile(r'(?<![A-Z])(' + '|'.join(map(re.escape, sorted(ascii_terms, key=len, reverse=True)))
                         + r')(?![A-Z])') if ascii_terms else None)
    anc = {}

    def ancestors(cid):
        if cid not in anc:
            out, c = [], cid
            while cby.get(c) and cby[c].get('parent'):
                c = cby[c]['parent']
                out.append(c)
            anc[cid] = out
        return anc[cid]

    def tag_text(text):
        '''文本点名的疾病概念（与 06 同一规则）'''
        if not cjk_re and not asc_re:
            return frozenset()
        q = _prep(text)
        cnt = defaultdict(int)
        if cjk_re:
            for m in cjk_re.finditer(q):
                for c in t2c[m.group(0)]:
                    cnt[c] += 1
        if asc_re:
            for m in asc_re.finditer(q):
                for c in t2c[m.group(1)]:
                    cnt[c] += 1
        redundant = {a for c in cnt for a in ancestors(c)}
        return frozenset(c for c in cnt if c not in redundant)

    def onto_relation(q_cs, d_cs):
        '''疾病关系，判定顺序与 08、10 相同：unknown → same → unnamed → sibling / different'''
        if not q_cs:
            return 'unknown'
        q_lin, d_lin = set(), set()
        for c in q_cs:
            q_lin |= {c, *ancestors(c)}
        for c in d_cs:
            d_lin |= {c, *ancestors(c)}
        if any(c in d_lin for c in q_cs) or any(d in q_lin for d in d_cs):
            return 'same'
        if not d_cs:
            return 'unnamed'
        return 'sibling' if {cfam.get(c, '-') for c in q_cs} & {cfam.get(c, '-') for c in d_cs} else 'different'

    return tag_text, onto_relation


def test_ontology_stats(root, desc=''):
    '''一个领域测试集的标注覆盖率（cov）与 (问题, 正确答案) 的疾病关系分布（rel）；缺文件时为 (None, None)'''
    tagger = make_tagger(f'{root}/ontology_new')
    tq, tc, tr = (f'{root}/split/test_queries.jsonl', f'{root}/split/test_corpus.jsonl',
                  f'{root}/split/test_qrels.jsonl')
    if tagger is None or not os.path.exists(tq):
        return None, None
    tag, relation = tagger
    qs = read_jsonl(tq)
    cs = read_jsonl(tc) if os.path.exists(tc) else []
    ql = {r['qid']: tag(r['text']) for r in tqdm(qs, desc=f'  {desc}test queries', leave=False)}
    dl = {r['doc_id']: tag(r['text']) for r in tqdm(cs, desc=f'  {desc}test answers', leave=False)}
    cov = {'queries': len(qs), 'queries_lab': sum(1 for v in ql.values() if v),
           'answers': len(cs), 'answers_lab': sum(1 for v in dl.values() if v)}
    rel = None
    if os.path.exists(tr):
        cnt = defaultdict(int)
        for r in read_jsonl(tr):
            cnt[relation(ql.get(r['qid'], frozenset()), dl.get(r['doc_id'], frozenset()))] += 1
        tot = sum(cnt.values())
        rel = {k: cnt[k] / tot for k in REL_KEYS} if tot else None
    return cov, rel


step('给测试集问题和答案打疾病标签（本体匹配）')
TEST_COV, TEST_REL = None, None
try:
    TEST_COV, TEST_REL = test_ontology_stats(OUT_DIR)
except Exception as _e:
    SKIPPED.append(f'test-set ontology coverage ({_e})')

FONT = 'Times New Roman'
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV',
         'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV']

try:
    from docx import Document
    from docx.shared import Pt, Inches
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.section import WD_ORIENT
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    def _new_doc():
        '''空白 Word 文档：横向页面（宽表多，便于预览），正文 Times New Roman 8 pt'''
        doc = Document()
        sec = doc.sections[0]
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
        sec.left_margin = sec.right_margin = Inches(0.7)
        doc.styles['Normal'].font.name = FONT
        doc.styles['Normal'].font.size = Pt(8)
        return doc

    _DOC = _new_doc()
except ImportError:
    _DOC = None
    _new_doc = lambda: None
    print('WARNING: python-docx not installed — Word tables skipped (pip install python-docx).')


def _font(run, size=8, bold=False, italic=False, caps=False):
    run.font.name, run.font.size = FONT, Pt(size)
    run.bold, run.italic, run.font.small_caps = bold, italic, caps
    rpr = run._element.get_or_add_rPr()
    rf = rpr.find(qn('w:rFonts'))
    if rf is None:
        rf = OxmlElement('w:rFonts')
        rpr.append(rf)
    for k in ('w:ascii', 'w:hAnsi', 'w:cs', 'w:eastAsia'):
        rf.set(qn(k), FONT)


def _line(cell, edge, size):
    '''单元格边线；size 单位 1/8 pt（12 = 1.5 pt 粗线，6 = 0.75 pt 细线）'''
    tcpr = cell._tc.get_or_add_tcPr()
    b = tcpr.find(qn('w:tcBorders'))
    if b is None:
        b = OxmlElement('w:tcBorders')
        tcpr.append(b)
    e = OxmlElement(f'w:{edge}')
    for k, v in (('w:val', 'single'), ('w:sz', str(size)), ('w:space', '0'), ('w:color', '000000')):
        e.set(qn(k), v)
    b.append(e)


def _word_table(caption, headers, rows, note=None, groups=None, align=None, doc=None, n=1):
    '''doc: 写入的 Word 文档；n: 表号（TABLE n，罗马数字）'''
    if doc is None:
        return
    num = ROMAN[n - 1] if n <= len(ROMAN) else str(n)
    A = {'l': WD_ALIGN_PARAGRAPH.LEFT, 'r': WD_ALIGN_PARAGRAPH.RIGHT, 'c': WD_ALIGN_PARAGRAPH.CENTER}
    ncol = len(headers)
    align = (align or ('l' + 'r' * (ncol - 1))).ljust(ncol, 'l')

    for txt, caps, before in ((f'TABLE {num}', False, 12), (caption.rstrip('.'), True, 0)):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(2)
        _font(p.add_run(txt), 8, caps=caps)

    group_at = dict(groups or [])
    body = []
    for i, r in enumerate(rows):
        if i in group_at:
            body.append(('group', group_at[i]))
        body.append(('row', r))
    t = doc.add_table(rows=1 + len(body), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    def fill(cell, text, a, bold=False, italic=False):
        cell.text = ''
        para = cell.paragraphs[0]
        para.alignment = A.get(a, WD_ALIGN_PARAGRAPH.LEFT)
        para.paragraph_format.space_after = Pt(0)
        _font(para.add_run(str(text)), 8, bold=bold, italic=italic)

    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        fill(c, h, align[j], bold=True)
        _line(c, 'top', 12)
        _line(c, 'bottom', 6)
    for i, (kind, val) in enumerate(body, 1):
        cells = t.rows[i].cells
        if kind == 'group':
            m = cells[0].merge(cells[-1])
            fill(m, val, 'l', italic=True)
            if i > 1:
                _line(m, 'top', 4)
        else:
            for j, v in enumerate(val):
                fill(cells[j], v, align[j])
    for c in t.rows[-1].cells:
        _line(c, 'bottom', 12)
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        _font(p.add_run('Note: ' + note), 7)


md_parts = ['# Tables for the paper (steps 06–10)', '',
            'Generated by step 11. Word version: `tables_06_10.docx`; CSV in `tables/`.', '']
table_log = []


def write_table(name_, caption, headers, rows, align=None, note=None, groups=None,
                doc=None, md=None, log=None):
    '''groups: [(行号, 小标题)]，在该行之前插入分组标题行。
    doc / md / log：写入哪个 Word 文档、Markdown 和清单；默认是本领域的 tables_06_10。'''
    doc = _DOC if doc is None else doc
    md_out = md_parts if md is None else md
    log = table_log if log is None else log
    align = align or ('l' + 'r' * (len(headers) - 1))
    with open(f'{TAB_DIR}/{name_}.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    n = sum(1 for nm, _ in log if not nm.endswith('.csv')) + 1
    _word_table(caption, headers, rows, note=note, groups=groups, align=align, doc=doc, n=n)
    group_at = dict(groups or [])
    md_align = [':---' if c == 'l' else '---:' if c == 'r' else ':---:' for c in align]
    md = [f'## {name_}', '', f'*{caption}*', '',
          '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(md_align) + ' |']
    for i, r in enumerate(rows):
        if i in group_at:
            md.append('| **' + group_at[i] + '** |' + ' |' * (len(headers) - 1))
        md.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    if note:
        md += ['', f'Note: {note}']
    md_out.extend(md + [''])
    log.append((name_, caption))


# %%
# =============================================================================
# 读取结果
# =============================================================================

def seed_results(tag, model, method, root=None):
    '''返回 [(seed, metrics.json)]，只取存在的种子；root 为领域输出目录，默认本领域'''
    out = []
    for s in SEEDS:
        p = f'{root or OUT_DIR}/results_{tag}/{model}/{method}/seed{s}/metrics.json'
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                out.append((s, json.load(f)))
    return out


def per_query_mean(tag, model, method, root=None):
    '''逐题 nDCG@10，跨种子取平均：{qid: value}'''
    acc = defaultdict(list)
    for s in SEEDS:
        p = f'{root or OUT_DIR}/results_{tag}/{model}/{method}/seed{s}/per_query.jsonl'
        if os.path.exists(p):
            for r in read_jsonl(p):
                acc[r['qid']].append(r['ndcg@10'])
    return {q: sum(v) / len(v) for q, v in acc.items()}


def paired_ci(a, b, iters=2000, seed=0):
    import numpy as np
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = np.concatenate([d[rng.integers(0, len(d), size=(min(200, iters - s), len(d)))].mean(axis=1)
                            for s in range(0, iters, 200)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


step('读取 06–10 的报告')
r06   = load(f'{REPORT_DIR}/06_ontology_new.json')
r09   = load(f'{REPORT_DIR}/09_results_new.json')
r09c  = load(f'{REPORT_DIR}/09_results_new_core.json')
r09d  = load(f'{REPORT_DIR}/09_results_new_dose.json')
r10   = load(f'{REPORT_DIR}/10_rerank_new.json')
r08   = {s: load(f'{REPORT_DIR}/08_training_sets_bge-base_{t}.json')
         for s, t in (('main', 'new'), ('core', 'new_core'), ('dose', 'new_dose'))}


NUM_WORD = {1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}


def label_note(rep08):
    '''训练时问题文本没有疾病名的，借用正确答案里的疾病（08 的口径），所以训练覆盖率高于 06 的问题文本口径'''
    o = (rep08 or {}).get('ontology') or {}
    pos, share = (o.get('disease_source') or {}).get('positive'), o.get('labelled_share')
    if pos is None or share is None:
        return ''
    k = (rep08 or {}).get('max_pos_concepts', 3)
    k = NUM_WORD.get(k, k)
    return (f'During training, a question whose text names no disease takes the diseases named in its correct '
            f'answer (at most {k}); this labels another {pos:,} questions, so {share:.1%} of training questions '
            f'have a disease label and the remaining {1 - share:.1%} are "Unknown" in the negative tables. '
            f'Test questions are labeled from the question text only, since the correct answer is not known at '
            f'retrieval time.')


LABEL_NOTE = label_note(r08.get('main'))
# 训练问题中有疾病标签（问题文本或正确答案）的比例，用于表注
LABEL_SHARE = ((r08.get('main') or {}).get('ontology') or {}).get('labelled_share')

# %%
# =============================================================================
step('表 1：本体规模与标注覆盖')
# 表 1：本体规模与标注覆盖
# =============================================================================

if r06:
    rl = r06.get('relation_layer', {})
    nq, na = r06['questions'], r06['answers']
    rows = [
        ['Disease concepts',                       f"{r06['n_concepts']:,}"],
        ['Surface terms (synonyms)',               f"{r06['n_terms']:,}"],
        ['Entities (symptom / drug / treatment / exam / cause)', f"{rl.get('entities', 0):,}"],
        ['Disease–entity relations (used)',        f"{rl.get('relations_used', 0):,}"],
        ['Training questions',                     f'{nq:,}'],
        ['  with a disease label (question text)', f"{r06['with_concept']:,} ({r06['with_concept'] / nq:.1%})"],
        ['  with more than one disease',           f"{r06['multi_concept']:,}"],
        ['Training answers',                       f'{na:,}'],
        ['  naming a disease',                     f"{r06['answers_specific']:,} ({r06['answers_specific'] / na:.1%})"],
        ['  naming no disease (unnamed)',          f"{r06['answers_unnamed']:,} ({r06['answers_unnamed'] / na:.1%})"],
    ]
    prior = r06.get('random_pair_relation', {})
    if prior:
        rows.append(['Random question–answer pairs (labeled questions): same / different / unnamed',
                     ' / '.join(f'{prior.get(k, 0):.1%}' for k in ('same', 'different', 'unnamed'))])
    for t, k in sorted((rl.get('relations_by_type') or {}).items(), key=lambda x: -x[1]):
        rows.append([f'  relations of type {t}', f'{k:,}'])
    rel_src = rl.get('relations_by_source') or {}
    for t, k in sorted(rel_src.items(), key=lambda x: -x[1]):
        rows.append([f'  relations from {SRC_LABEL.get(t, t)}', f'{k:,}'])
    if TEST_COV:
        rows += [
            ['Test questions', f"{TEST_COV['queries']:,}"],
            ['  with a disease label',
             f"{TEST_COV['queries_lab']:,} ({TEST_COV['queries_lab'] / max(TEST_COV['queries'], 1):.1%})"],
            ['Test retrieval corpus (answers)', f"{TEST_COV['answers']:,}"],
            ['  naming a disease',
             f"{TEST_COV['answers_lab']:,} ({TEST_COV['answers_lab'] / max(TEST_COV['answers'], 1):.1%})"],
        ]
    write_table('tab_ontology', 'Liver-disease ontology and annotation coverage.',
                ['Item', 'Value'], rows, align='lr',
                note='Concepts follow WHO ICD-10 with Chinese synonyms; entity relations come from '
                     'Wikidata, the MSD Manual (Chinese consumer edition)'
                     + (', MedlinePlus' if 'medlineplus' in rel_src else '') + ' and Chinese Wikipedia. '
                     'Training and test items are labeled with the same rules: a question from its own text '
                     'and an answer from its own text. ' + LABEL_NOTE)

# %%
# =============================================================================
step('表 2：方法对照')
# 表 2：方法对照
# =============================================================================

rows = [[name(m), *METHOD_INFO[m]] for m in MAIN_METHODS if m in METHOD_INFO]
write_table('tab_methods', 'Hard-negative selection methods compared in this study.',
            ['Method', 'Uses a retrieval model', 'Candidate pool', 'Selection'],
            rows, align='llll', note=POOL_NOTE + ' How often each retrieval-based rule could be applied is '
            'reported in the rule-execution table.')
write_table('tab_methods_appx', 'Additional methods used in the ablation and dose experiments.',
            ['Method', 'Uses a retrieval model', 'Candidate pool', 'Selection'],
            [[name(m), *v] for m, v in METHOD_INFO_APPX.items()], align='llll',
            note=POOL_NOTE + ' "Uses a retrieval model" refers to mining negatives, not to the model being fine-tuned.')

# %%
# =============================================================================
step('表 3：主结果')
# 表 3：主结果
# =============================================================================

SHOW = [('ndcg@10', 'nDCG@10'), ('mrr@10', 'MRR@10'), ('recall@1', 'R@1'),
        ('recall@10', 'R@10'), ('recall@100', 'R@100')]
MAIN_HEAD = ['Method', 'Seeds', *[h for _, h in SHOW]]
MAIN_NOTE = ('All methods share the loss, batch size, schedule and model selection; '
             'they differ only in the hard negative attached to each training pair. '
             'TopK-PercPos falls back to dense rank 100 for most questions '
             '(see the rule-execution table), which explains its closeness to "Dense rank 100".')


def main_rows(model, root=None):
    '''tab_main 中一个模型的行：零样本 + 各方法（均值 ± 标准差）；没有任何方法的结果时返回 []'''
    rows = []
    zs = f'{root or OUT_DIR}/results_new/{model}/zeroshot/metrics.json'
    if os.path.exists(zs):
        z = load(zs)
        rows.append(['Zero-shot', '-', *[f"{z['test'][k] * 100:.2f}" for k, _ in SHOW]])
    found = False
    for m in MAIN_METHODS:
        runs = seed_results('new', model, m, root)
        if runs:
            found = True
            rows.append([name(m), str(len(runs)), *[ms([r['test'][k] for _, r in runs]) for k, _ in SHOW]])
    return rows if found else []


all_main, groups = [], []
for model in MODELS:
    rows_m = main_rows(model)
    if rows_m:
        groups.append((len(all_main), MODEL_NAME[model]))
        all_main += rows_m
if all_main:
    write_table('tab_main', 'Retrieval results on the test set (mean ± std over training seeds).',
                MAIN_HEAD, all_main, align='lr' + 'r' * len(SHOW), groups=groups, note=MAIN_NOTE)
else:
    SKIPPED.append('results_new/ (main results)')

# %%
# =============================================================================
step('表 4：关键配对比较')
# 表 4：关键配对比较
# =============================================================================

KEY_PAIRS = [('onto_pool_rand', 'mnrl'), ('onto_loose_rand', 'mnrl'),
             ('onto_pool_rand', 'random'), ('onto_loose_rand', 'random'),
             ('onto_pool_rand', 'margin'), ('margin', 'mnrl'), ('margin', 'random'),
             ('margin_onto', 'margin'), ('onto_pool_strat', 'onto_pool_rand'),
             ('onto_strat_loose', 'onto_loose_rand')]
COMPARE_HEAD = ['Comparison', 'Δ nDCG@10', '95% CI', 'p (Holm)', 'Seeds better', 'Non-inferior']
COMPARE_NOTE = ('Per-query nDCG@10 averaged over seeds; paired bootstrap (2,000 resamples) and '
                'permutation test (10,000). p values are Holm-corrected within each model over all '
                'pre-declared and exploratory comparisons of the main experiment, not only the rows '
                'shown here. Non-inferior: lower CI bound above −0.30 points.')


def compare_rows(rep, model):
    '''tab_compare 中一个模型的行（取自 09 报告）'''
    comps = {(c['method'], c['baseline']): c for c in ((rep or {}).get(model) or {}).get('comparisons', [])}
    rows = []
    for a, b in KEY_PAIRS:
        c = comps.get((a, b))
        if c:
            rows.append([f'{name(a)} vs {name(b)}', sgn(c["delta_ndcg@10"]),
                         f"[{sgn(c['ci95'][0])}, {sgn(c['ci95'][1])}]",
                         f"{c.get('p_holm', c['p']):.4f}",
                         f"{c['seeds_better']}/{c['seeds_compared']}",
                         'yes' if c.get('noninferior') else 'no'])
    return rows


if r09:
    rows, groups = [], []
    for model in MODELS:
        rows_m = compare_rows(r09, model)
        if rows_m:
            groups.append((len(rows), MODEL_NAME[model]))
            rows += rows_m
    if rows:
        write_table('tab_compare', 'Paired comparisons on per-query nDCG@10 (test set).', COMPARE_HEAD,
                    rows, align='lrrrrc', groups=groups, note=COMPARE_NOTE)

# %%
# =============================================================================
step('表 5：负例质量（负例与问题的疾病关系占比，bge 的训练文件；本体方法两个模型共用）')
# 表 5：负例质量（负例与问题的疾病关系占比，bge 的训练文件；本体方法两个模型共用）
# =============================================================================

neg_stats = {}
for suite in ('main', 'core'):
    rep = r08.get(suite)
    if rep:
        for m, s in rep.get('methods', {}).items():
            if m != 'mnrl' and m not in neg_stats:
                neg_stats[m] = s
NEG_ORDER = [m for m in ['random', 'bm25', 'dense', 'dense_rank10', 'dense_rank100', 'margin',
                         'margin_onto', 'bm25_onto', 'dense_onto', 'onto_pool_rand',
                         'onto_pool_strat', 'onto_loose_rand', 'onto_strat_loose', 'onto_pool_sim',
                         'onto_pool_hard', 'onto_hard_loose'] if m in neg_stats]
if NEG_ORDER:
    rows = [[name(m), *[f"{neg_stats[m].get(k, 0):.1%}" for k in REL_KEYS],
             f"{neg_stats[m].get('distinct_negatives', 0):,}"] for m in NEG_ORDER]
    write_table('tab_negatives', 'Disease relation between each training question and its hard negative.',
                ['Method', 'Same disease', 'Same family', 'Different family', 'Unnamed', 'Unknown',
                 'Distinct negatives'], rows, align='lrrrrrr',
                note='Relations are assigned by the ontology. "Unknown": the question has no disease '
                     'label, neither from its text nor from its correct answer (these questions fall back to a '
                     'random negative in ontology methods), so this share is smaller than the share of questions '
                     'whose text names no disease in the ontology table. '
                     'Same-disease answers are potential false negatives.')

# %%
# =============================================================================
step('表 6：消融（core 套件：池内打分挑难 + bm25_onto / dense_onto）')
# 表 6：消融（core 套件：池内打分挑难 + bm25_onto / dense_onto）
# =============================================================================

CORE_NOTE = ('Scoring methods rank the ontology pool by entity similarity, a lexical/ontology '
             'formula or a mid band of it; "+ ontology filter" removes same-disease and unnamed '
             'candidates before taking the top one.' + SUITE_NOTE)


def core_rows(model, root=None):
    '''tab_ablation_core 中一个模型的行'''
    rows = []
    for m in CORE_METHODS:
        runs = seed_results('new_core', model, m, root)
        if runs:
            rows.append([name(m), str(len(runs)), ms([r['test']['ndcg@10'] for _, r in runs]),
                         ms([r['test']['recall@10'] for _, r in runs])])
    return rows


all_core, groups = [], []
for model in MODELS:
    rows_m = core_rows(model)
    if rows_m:
        groups.append((len(all_core), MODEL_NAME[model]))
        all_core += rows_m
if all_core:
    write_table('tab_ablation_core', 'Ablation: ontology-based difficulty scoring and ontology filtering of '
                'BM25 / dense candidates.', ['Method', 'Seeds', 'nDCG@10', 'R@10'], all_core,
                align='lrrr', groups=groups, note=CORE_NOTE)

# %%
# =============================================================================
step('表 7：假负例剂量实验')
# 表 7：假负例剂量实验
# =============================================================================

def dose_points(models, root=None):
    '''{model: ([(注入比例, 均值, 标准差, 种子数)], {参照方法: 均值})}'''
    out = {}
    for model in models:
        pts = []
        for m, pct in DOSE_LEVELS:
            runs = seed_results('new_dose', model, m, root)
            if runs:
                v = [r['test']['ndcg@10'] * 100 for _, r in runs]
                pts.append((pct, statistics.mean(v), statistics.stdev(v) if len(v) > 1 else 0., len(v)))
        refs = {}
        for rm in ('random', 'mnrl'):
            runs = seed_results('new_dose', model, rm, root)
            if runs:
                refs[rm] = statistics.mean(r['test']['ndcg@10'] * 100 for _, r in runs)
        if pts:
            out[model] = (pts, refs)
    return out


dose = dose_points(MODELS)
if dose:
    rows = []
    for pct in [p for _, p in DOSE_LEVELS]:
        row = [f'{pct}%']
        for model in MODELS:
            hit = [x for x in dose.get(model, ([], {}))[0] if x[0] == pct]
            row.append(f'{hit[0][1]:.2f} ± {hit[0][2]:.2f}' if hit else '-')
        rows.append(row)
    for rm in ('random', 'mnrl'):
        rows.append([f'Reference: {name(rm)}'] +
                    [f"{dose[m][1][rm]:.2f}" if m in dose and rm in dose[m][1] else '-' for m in MODELS])
    write_table('tab_dose', 'nDCG@10 when a share of training negatives is replaced by same-disease answers.',
                ['Injected same-disease negatives', *[MODEL_NAME[m] for m in MODELS]], rows,
                align='l' + 'r' * len(MODELS),
                note='Nominal rate: each training pair draws a same-disease answer with this probability, '
                     'otherwise a safe-pool answer. About 14% of questions have no disease label and keep a '
                     'random negative, so the effective rate is about 0.86 × nominal. Random sampling from '
                     'the full corpus yields about 27% same-disease negatives. The reference rows come from the '
                     'dose runs.' + SUITE_NOTE)

# %%
# =============================================================================
step('表 8：本体重排')
# 表 8：本体重排
# =============================================================================

RR_RULES = ['none', 'same_up', 'diff_down', 'both', 'hard_down', 'bm25_fuse']
RR_NAME = {'none': 'No reranking', 'same_up': 'Same disease +λ', 'diff_down': 'Different family −λ',
           'both': 'Same +λ, different −λ', 'hard_down': 'Different family demoted',
           'bm25_fuse': 'Control: + λ·BM25'}
RR_MAIN_METHOD = 'onto_pool_rand'   # 正文只放本体池训练出的模型，其余组合放附录
RR_NOTE = ('λ is selected on the validation set; the test set is evaluated once. '
           'Cross-family in top-10: share of top-10 answers naming a disease of another '
           'disease family (questions with a disease label only). The BM25 control tests '
           'whether gains come from the ontology or from adding any second signal.')


def _rr_cell(v, k):
    xs = v.get(k) or []
    if not xs:
        return '-'
    sd = statistics.stdev(xs) * 100 if len(xs) > 1 else 0.
    return f'{statistics.mean(xs) * 100:.2f} ± {sd:.2f}'


def _rr_mean(v, k):
    xs = (v or {}).get(k) or []
    return statistics.mean(xs) * 100 if xs else None


def _rr_rows(keys, r10_=None, prefix=''):
    '''重排明细表的行；r10_ 为某个领域的 10 报告（默认本领域），prefix 加在分组标题前（如 "Neurology · "）'''
    r10_ = r10 if r10_ is None else r10_
    rows, groups = [], []
    for key in keys:
        val = r10_[key]
        model, method = key.split('/', 1)
        comps = {c['rule']: c for c in val.get('comparisons', [])}
        groups.append((len(rows), f'{prefix}{MODEL_NAME.get(model, model)} · trained with {name(method)}'))
        for rule in RR_RULES:
            v = val.get(rule)
            if not v:
                continue
            c = comps.get(rule)
            rows.append([RR_NAME[rule], ','.join(f'{x:g}' for x in v.get('lambda', [])) or '-',
                         _rr_cell(v, 'ndcg@10'), _rr_cell(v, 'recall@10'), _rr_cell(v, 'cross_family@10'),
                         f"{sgn(c['delta'])} [{sgn(c['ci'][0])}, {sgn(c['ci'][1])}]" if c else '-',
                         f"{c['p_holm']:.4f}" if c else '-'])
    return rows, groups


RR_HEAD = ['Rule', 'λ (dev)', 'nDCG@10', 'R@10', 'Cross-family in top-10', 'Δ vs none [95% CI]', 'p (Holm)']


def rr_keys(r10_, models):
    '''重排结果里的检索器（模型/训练方法），按固定顺序'''
    keys = [f'{m}/{meth}' for m in models for meth in
            ['mnrl', 'random', 'onto_pool_rand', 'margin', 'margin_onto', 'onto_loose_rand']
            if f'{m}/{meth}' in r10_]
    return keys + [k for k in r10_ if k not in keys]


def rerank_summary(r10_, keys):
    '''每条规则在全部检索器上的 Δ 范围、显著组数、跨大类错误的相对变化：{rule: dict}'''
    out = {}
    for rule in RR_RULES[1:]:
        ds, sig_up, sig_down, rel_cf = [], 0, 0, []
        for key in keys:
            c = {x['rule']: x for x in r10_[key].get('comparisons', [])}.get(rule)
            if c:
                ds.append(c['delta'])
                if c['p_holm'] < 0.05:
                    sig_up += c['delta'] > 0
                    sig_down += c['delta'] < 0
            base_cf = _rr_mean(r10_[key].get('none'), 'cross_family@10')
            cur_cf = _rr_mean(r10_[key].get(rule), 'cross_family@10')
            if base_cf and cur_cf is not None:
                rel_cf.append((cur_cf - base_cf) / base_cf)
        if ds:
            out[rule] = {'d_min': min(ds), 'd_max': max(ds), 'up': sig_up, 'down': sig_down,
                         'n': len(ds), 'cf_min': min(rel_cf) if rel_cf else None,
                         'cf_max': max(rel_cf) if rel_cf else None}
    return out


def rerank_summary_rows(summary):
    rows = []
    for rule in RR_RULES[1:]:
        s = summary.get(rule)
        if not s:
            continue
        cf = ('-' if s['cf_min'] is None else
              f"{sgn_pct(s['cf_min'])} to {sgn_pct(s['cf_max'])}" if sgn_pct(s['cf_min']) != sgn_pct(s['cf_max'])
              else sgn_pct(s['cf_min']))
        rows.append([RR_NAME[rule], f"{sgn(s['d_min'])} to {sgn(s['d_max'])}",
                     f"{s['up']} / {s['down']} / {s['n'] - s['up'] - s['down']}", cf])
    return rows


rr_summary = {}   # rule -> dict，供正文汇总表与框架图使用
if r10:
    all_keys = rr_keys(r10, MODELS)
    main_keys = [k for k in all_keys if k.split('/', 1)[1] == RR_MAIN_METHOD] or all_keys[:len(MODELS)]

    rows, groups = _rr_rows(main_keys)
    if rows:
        write_table('tab_rerank', 'Inference-time ontology reranking of the top-100 candidates '
                    '(retrievers fine-tuned with ontology-pool negatives).', RR_HEAD, rows,
                    align='llrrrrr', groups=groups,
                    note=RR_NOTE + ' Results for retrievers trained with the other methods are in the appendix.')
    rows, groups = _rr_rows(all_keys)
    if rows:
        write_table('tab_rerank_appx', 'Ontology reranking for every fine-tuned retriever.', RR_HEAD, rows,
                    align='llrrrrr', groups=groups, note=RR_NOTE)

    rr_summary = rerank_summary(r10, all_keys)
    if rr_summary:
        write_table('tab_rerank_summary', 'Ontology reranking across all fine-tuned retrievers.',
                    ['Rule', 'Δ nDCG@10 vs none (range)', 'Significant better / worse / n.s.',
                     'Relative change in cross-family answers'], rerank_summary_rows(rr_summary), align='lrcr',
                    note='{} fine-tuned retrievers (each model × training method, 3 seeds each). Significance: '
                         'Holm-corrected p < 0.05 within each retriever. Relative change: cross-family share in '
                         'top-10 after reranking divided by the share before, minus one.'.format(len(all_keys)))

# %%
# =============================================================================
step('表 9：本体过滤修复检索式难负例')
# 表 9：本体过滤修复检索式难负例
# =============================================================================

def paired_test(a, b, iters=10000, seed=0):
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


def holm_adj(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, mx = [1.] * len(ps), 0.
    for rank, i in enumerate(order):
        mx = max(mx, min(1., (len(ps) - rank) * ps[i]))
        adj[i] = mx
    return adj


# 选负例规则的执行情况：从 training_new_core/<model>/negatives.jsonl 的 rule 字段统计
# 每个问题一行（不是每个训练对），与 08 的规则名一一对应。
core_stats = (r08.get('core') or {}).get('methods', {})
RULE_METHODS = ['bm25', 'bm25_onto', 'dense', 'dense_onto', 'margin', 'margin_onto']
_neg_cache = {}


def _rule_kind(method, rule):
    '''把 08 的规则字符串归成：applied（按方法规则选）/ fallback_*（退回）/ no_label'''
    r = rule or ''
    if 'no disease label' in r:
        return 'no_label'
    if 'no candidate under score limit' in r:
        return 'fallback_cap'          # TopK-PercPos 找不到低于 95% 上限的候选 → dense rank 100
    if 'filter would leave nothing' in r:
        return 'fallback_filter'       # 本体过滤后没有候选 → 原方法的负例
    if r.startswith('fallback'):
        return 'fallback_other'
    return 'applied'


def neg_records(model, root=None):
    '''读一次 negatives.jsonl，缓存每题每法的 (doc_id, 规则类别)；root 为领域输出目录，默认本领域'''
    root = root or OUT_DIR
    if (root, model) not in _neg_cache:
        path = f'{root}/training_new_core/{model}/negatives.jsonl'
        rec = None
        if os.path.exists(path):
            rec = []
            for r in tqdm(read_jsonl(path), desc=f'  {model} negatives.jsonl', leave=False):
                rec.append({m: (r[m]['doc_id'], _rule_kind(m, r[m].get('rule')))
                            for m in RULE_METHODS if isinstance(r.get(m), dict)})
        else:
            SKIPPED.append(path)
        _neg_cache[(root, model)] = rec
    return _neg_cache[(root, model)]


def rule_shares(method, model, root=None):
    '''{类别: 占全部问题的比例}'''
    rec = neg_records(model, root)
    if not rec:
        return None
    vals = [r[method][1] for r in rec if method in r]
    n = len(vals)
    return {k: sum(v == k for v in vals) / n for k in set(vals)} if n else None


def replaced_share(base, model='bge-base', root=None):
    '''原方法与本体过滤版选中的负例不同的问题比例'''
    filt = dict(FILTER_PAIRS).get(base)
    rec = neg_records(model, root)
    if not filt or not rec:
        return None
    pairs = [(r[base][0], r[filt][0]) for r in rec if base in r and filt in r]
    return sum(a != b for a, b in pairs) / len(pairs) if pairs else None


def filter_repair(models, root=None):
    '''本体过滤前后的逐题配对比较（Holm 在传入的全部比较上做）：{(model, base): dict}'''
    res = {}
    for model in models:
        for base, filt in FILTER_PAIRS:
            rb, rf = seed_results('new_core', model, base, root), seed_results('new_core', model, filt, root)
            if not rb or not rf:
                continue
            pb, pf = per_query_mean('new_core', model, base, root), per_query_mean('new_core', model, filt, root)
            qs = sorted(set(pb) & set(pf))
            if not qs:
                continue
            a, b = [pf[q] for q in qs], [pb[q] for q in qs]
            d, lo, hi = paired_ci(a, b)
            sb = {s: r['test']['ndcg@10'] for s, r in rb}
            sf = {s: r['test']['ndcg@10'] for s, r in rf}
            common = sorted(set(sb) & set(sf))
            mnrl_runs = seed_results('new_core', model, 'mnrl', root)
            res[(model, base)] = {
                'filt': filt, 'base_mean': statistics.mean(x for x in sb.values()) * 100,
                'filt_mean': statistics.mean(x for x in sf.values()) * 100,
                'mnrl_mean': statistics.mean(r['test']['ndcg@10'] for _, r in mnrl_runs) * 100 if mnrl_runs else None,
                'delta': d * 100, 'lo': lo * 100, 'hi': hi * 100, 'p': paired_test(a, b),
                'wins': sum(sf[s] > sb[s] for s in common), 'n': len(common)}
    keys = list(res)
    for k, pa in zip(keys, holm_adj([res[k]['p'] for k in keys])):
        res[k]['p_holm'] = pa
    return res


def filter_repair_row(v, base, model, root=None):
    '''tab_filter_repair 的一行（不含方法名）'''
    rs = replaced_share(base, model, root)
    sh = rule_shares(v['filt'], model, root) or {}
    applied = sh.get('applied')
    fb = sh.get('fallback_filter', 0) + sh.get('fallback_cap', 0) + sh.get('fallback_other', 0)
    gap = (v['mnrl_mean'] - v['base_mean']) if v['mnrl_mean'] is not None else None
    rec = (f"{v['delta'] / gap:.0%}" if gap and gap > 0 and v['delta'] > 0 else '-')
    return [f"{applied:.1%}" if applied is not None else '-',
            f"{fb:.1%}" if sh else '-',
            f"{rs:.1%}" if rs is not None else '-',
            f"{v['base_mean']:.2f}", f"{v['filt_mean']:.2f}",
            f"{sgn(v['delta'])} [{sgn(v['lo'])}, {sgn(v['hi'])}]",
            f"{v['p_holm']:.4f}", f"{v['wins']}/{v['n']}", rec]


KIND_COLS = [('applied', 'Rule applied'), ('fallback_cap', 'No candidate below 95% of positive'),
             ('fallback_filter', 'Filter left no candidate'), ('no_label', 'No disease label'),
             ('fallback_other', 'Other fallback')]
EXEC_HEAD = ['Method'] + [h for _, h in KIND_COLS]
# {} 处填 "No disease label" 一列的说明（本领域为 "below 14%"；跨领域表里各领域不同，写 "lower"）
EXEC_NOTE = ('Share of unique training questions; "-" means the case never occurs. "Rule applied": the '
             'method selected its negative by its own rule. For TopK-PercPos (threshold '
             '95%), questions without any top-100 dense candidate below 95% of the positive score fall '
             'back to dense rank 100; this is why it behaves close to "Dense rank 100". This check comes '
             'before the ontology filter, so for "TopK-PercPos + ontology filter" some questions without '
             'a disease label are counted under the first fallback, and its "No disease label" share is '
             '{}. Without a disease label (neither from the question '
             'text nor from the correct answer) the ontology filter cannot be applied and the original '
             'negative is kept.')


def _pct_small(v):
    if not v:
        return '-'
    return '<0.1%' if v < 0.0005 else f'{v:.1%}'


def exec_rows(model, root=None):
    '''tab_rule_execution 中一个模型的行'''
    rows = []
    for m in RULE_METHODS:
        sh = rule_shares(m, model, root)
        if sh:
            rows.append([name(m)] + [_pct_small(sh.get(k)) for k, _ in KIND_COLS])
    return rows


filter_res = filter_repair(MODELS)
if filter_res:
    keys = list(filter_res)
    rows, groups = [], []
    for model in MODELS:
        ks = [k for k in keys if k[0] == model]
        if not ks:
            continue
        groups.append((len(rows), MODEL_NAME[model]))
        for _, base in ks:
            rows.append([name(base)] + filter_repair_row(filter_res[(model, base)], base, model))
    write_table('tab_filter_repair', 'Ontology filtering of retrieval-mined hard negatives (nDCG@10, test set).',
                ['Mining method', 'Filter applied', 'Fallback', 'Negative changed', 'Original',
                 '+ Ontology filter', 'Δ [95% CI]', 'p (Holm)', 'Seeds better', 'Gap to MNRL recovered'],
                rows, align='lrrrrrrrrr', groups=groups,
                note='The filter removes candidates that the ontology marks as the same disease as the question '
                     'or as naming no disease, then takes the highest-ranked remaining candidate. Shares are over '
                     'unique training questions in the training file of each retriever. "Filter applied": the '
                     'filtered candidate list was non-empty. "Fallback": the original negative was kept, either '
                     'because every top-100 candidate was same-disease or unnamed, or, for TopK-PercPos, '
                     'because no candidate scored below 95% of the positive (then dense rank 100 is used). The '
                     'remaining questions (about 14%) have no disease label and keep the original negative. '
                     '"Negative changed": the selected negative differs from the original method. "Gap to MNRL '
                     'recovered": Δ divided by the distance from the original method to in-batch-only training. '
                     'Paired bootstrap and permutation test on per-query nDCG@10 averaged over seeds; Holm over '
                     'all rows.' + SUITE_NOTE)

    # 选负例规则的执行情况（每个方法：按规则选 / 退回 / 无疾病标签）
    exec_all, exec_groups = [], []
    for model in MODELS:
        rows_m = exec_rows(model)
        if rows_m:
            exec_groups.append((len(exec_all), MODEL_NAME[model]))
            exec_all += rows_m
    if exec_all:
        write_table('tab_rule_execution', 'How often each retrieval-based mining rule could be applied.',
                    EXEC_HEAD, exec_all, align='lrrrrr', groups=exec_groups,
                    note=EXEC_NOTE.format(f'below {round((1 - LABEL_SHARE) * 100)}%' if LABEL_SHARE
                                          else 'lower than for the other filtered methods'))

# %%
# =============================================================================
step('表 10：被选负例的排名')
# 表 10：被选负例的排名
# =============================================================================

RANK_METHODS = ['bm25', 'bm25_onto', 'dense', 'dense_onto', 'dense_rank10', 'dense_rank100',
                'margin', 'margin_onto']


def fmt_or(v):
    return '-' if v is None else f'{v:g}'


RANK_HEAD = ['Method', 'Mean rank', 'Median rank', 'Same disease', 'Unnamed', 'Distinct negatives']


def rank_rows(stats):
    '''tab_negative_rank 的行；stats 为 08 core 报告的 methods'''
    return [[name(m), fmt_or(stats[m].get('mean_rank')), fmt_or(stats[m].get('median_rank')),
             f"{stats[m].get('same', 0):.1%}", f"{stats[m].get('unnamed', 0):.1%}",
             f"{stats[m].get('distinct_negatives', 0):,}"]
            for m in RANK_METHODS if m in stats]


if core_stats:
    rows = rank_rows(core_stats)
    if rows:
        write_table('tab_negative_rank', 'Rank of the selected hard negative in its candidate list (BGE-base-zh).',
                    RANK_HEAD, rows, align='lrrrrr',
                    note='Rank 1 is the most similar candidate (BM25 or dense, top-100). Statistics come from the '
                         'BGE-base-zh training files and include questions where a fallback applies. With the BM25 '
                         'candidates the filter applies to most questions, and the selected negative still sits near '
                         'the top of the list (median rank 3), so the gain is not explained by switching to easy '
                         'negatives. The dense and TopK-PercPos rows are dominated by fallbacks (the original rank-1 '
                         'or rank-100 negative is kept; see the rule-execution table), which explains their rank and '
                         'remaining same-disease share.')

# %%
# =============================================================================
step('表 11：实验设置（训练配方，取自结果文件里的 config）')
# 表 11：实验设置（训练配方，取自结果文件里的 config）
# =============================================================================

SETUP_HEAD = ['Base checkpoint', 'Loss', 'Batch', 'Mini-batch', 'LR', 'Epochs', 'Warmup',
              'Max length', 'Eval every', 'Training pairs']
SETUP_NOTE = ('Linear schedule, bf16, NO_DUPLICATES batch sampler; training rows are shuffled with the '
              'seed before each run (seeds 42, 43, 44). The checkpoint with the best validation '
              'nDCG@10 is evaluated once on the test set. Hard-negative mining: top-100 candidates; '
              'TopK-PercPos keeps candidates scoring below 95% of the positive score.')


def setup_cells(model, root=None):
    '''tab_setup 中一个模型的配置（取自结果文件里的 config）；没有结果时返回 None'''
    runs = seed_results('new', model, 'mnrl', root) or seed_results('new', model, 'random', root)
    if not runs:
        return None
    c = runs[0][1].get('config', {})
    tr = runs[0][1].get('train', {})
    n_rows = tr.get('rows')
    return [os.path.basename(str(c.get('base_model_path', '-'))),
            str(c.get('loss', '-')).replace('CachedMultipleNegativesRankingLoss', 'Cached MNRL'),
            str(c.get('batch_size', '-')), str(c.get('mini_batch', '-')), f"{c.get('lr', 0):g}",
            str(c.get('epochs', '-')), str(c.get('warmup_ratio', '-')), str(c.get('max_seq_len', '-')),
            str(c.get('eval_steps', '-')), f'{n_rows:,}' if isinstance(n_rows, int) else str(n_rows or '-')]


setup_rows = []
for model in MODELS:
    cells = setup_cells(model)
    if cells:
        setup_rows.append([MODEL_NAME[model], *cells])
if setup_rows:
    write_table('tab_setup', 'Fine-tuning configuration shared by all methods.',
                ['Model', *SETUP_HEAD], setup_rows, align='ll' + 'l' + 'r' * 8, note=SETUP_NOTE)

# %%
# =============================================================================
step('表 12：重排诊断——测试集正确答案与问题的疾病关系（本体误判率）')
# 表 12：重排诊断——测试集正确答案与问题的疾病关系（本体误判率）
# =============================================================================
# 使用本文件内的标注器（规则与 06、10 相同），不依赖 10 的脚本，也不重复编码。

if TEST_REL:
    _lab = sum(v for k, v in TEST_REL.items() if k != 'unknown')
    write_table('tab_rerank_diag', 'Disease relation between test questions and their correct answers.',
                ['Relation', 'Share of all (question, correct answer) pairs', 'Share among labeled questions'],
                [[k.capitalize(), f'{TEST_REL[k]:.1%}',
                  f'{TEST_REL[k] / _lab:.1%}' if k != 'unknown' and _lab else '-'] for k in REL_KEYS],
                align='lrr',
                note='Labels use the same ontology and matching rules as negative mining and reranking. A correct answer '
                     'labeled "different" either names another disease (e.g., a complication) or reflects a '
                     'labeling miss; the hard-demotion rule pushes such answers out of the top-10, which is '
                     'consistent with its small nDCG loss. Pairs, not questions, are counted (a question may '
                     'have several correct answers).')
else:
    SKIPPED.append('tab_rerank_diag (split/test_qrels.jsonl or ontology labels unavailable)')

# %%
# =============================================================================
step('表 13：案例（margin 或 BM25 选到同病负例，本体过滤换掉）。')
# 表 13：案例（margin 或 BM25 选到同病负例，本体过滤换掉）。
# =============================================================================

examples_out = []
for suite in ('main', 'core'):
    for e in ((r08.get(suite) or {}).get('examples') or []):
        examples_out.append(['TopK-PercPos vs ' + ('loose pool, random' if suite == 'main' else 'ontology pool, entity similarity'),
                             e.get('question', ''), e.get('question_diseases', ''),
                             f"[{e.get('margin_rel', '')}] {e.get('margin_neg', '')}",
                             f"[{e.get('onto_rel', '')}] {e.get('onto_neg', '')}"])
neg_path = f'{OUT_DIR}/training_new_core/bge-base/negatives.jsonl'
if os.path.exists(neg_path):
    import random as _random
    corpus_txt = {}
    cp = f'{OUT_DIR}/split/train_corpus.jsonl'
    if os.path.exists(cp):
        corpus_txt = {r['doc_id']: r['text'] for r in read_jsonl(cp)}
    pool = []
    for r in tqdm(read_jsonl(neg_path), desc='  BM25 examples', leave=False):
        b, f_ = r.get('bm25'), r.get('bm25_onto')
        if b and f_ and b['relation'] == 'same' and f_['relation'] in ('different', 'sibling') and len(r['question']) <= 40:
            pool.append((r['question'], b, f_))
    for qtxt, b, f_ in _random.Random(42).sample(pool, min(6, len(pool))):
        examples_out.append(['BM25 top-1 vs + ontology filter', qtxt, '',
                             f"[{b['relation']}] {corpus_txt.get(b['doc_id'], '')[:120]}",
                             f"[{f_['relation']}] {corpus_txt.get(f_['doc_id'], '')[:120]}"])
if examples_out:
    with open(f'{PAPER_DIR}/examples_negatives.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['comparison', 'question', 'question_diseases', 'original_negative (relation)',
                    'ontology_negative (relation)', 'question_en (fill in)', 'notes'])
        w.writerows([r + ['', ''] for r in examples_out])
    table_log.append(('examples_negatives.csv', 'Hard-negative case studies (Chinese, pick and translate)'))

# %%
# =============================================================================
step('表 14：训练收敛（验证集最佳 nDCG@10、步数、耗时）')
# 表 14：训练收敛（验证集最佳 nDCG@10、步数、耗时）
# =============================================================================

CONV_METHODS = ['mnrl', 'random', 'margin', 'onto_pool_rand', 'onto_loose_rand']
CONV_HEAD = ['Model', 'Method', 'Seeds', 'Best dev nDCG@10', 'Selected checkpoint', 'Steps', 'Minutes']


def conv_rows(model, root=None):
    '''tab_training 中一个模型的行'''
    rows = []
    for m in CONV_METHODS:
        runs = seed_results('new', model, m, root)
        tr = [r.get('train', {}) for _, r in runs if r.get('train')]
        if not tr:
            continue
        rows.append([MODEL_NAME[model], name(m), str(len(tr)),
                     ms([t['best_dev_ndcg@10'] for t in tr if t.get('best_dev_ndcg@10') is not None]),
                     ', '.join(sorted({t.get('best_checkpoint', '') for t in tr})),
                     f"{statistics.mean(t.get('steps', 0) for t in tr):.0f}",
                     f"{statistics.mean(t.get('train_min', 0) for t in tr):.0f}"])
    return rows


all_conv = [r for model in MODELS for r in conv_rows(model)]
if all_conv:
    write_table('tab_training', 'Validation nDCG@10 of the selected checkpoint and training cost.',
                CONV_HEAD, all_conv, align='llrrlrr')

with open(f'{PAPER_DIR}/tables_06_10.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_parts) + '\n')
if _DOC is not None and table_log:
    _DOC.save(f'{PAPER_DIR}/tables_06_10.docx')

# %%
# =============================================================================
# 图
# =============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print('matplotlib / numpy not installed — figures skipped.')

fig_log = []
FIG_FONT = None
if HAVE_MPL:
    from matplotlib import font_manager
    _have = {f.name for f in font_manager.fontManager.ttflist}
    if FONT in _have:
        FIG_FONT = FONT
    else:
        FIG_FONT = next((n for n in ('Liberation Serif', 'TeX Gyre Termes', 'Nimbus Roman', 'DejaVu Serif')
                         if n in _have), 'serif')
        print(f'WARNING: "{FONT}" not found by matplotlib — figures use "{FIG_FONT}". '
              'Install Times New Roman, delete ~/.cache/matplotlib, then re-run.')
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,                 # 嵌入 TrueType 字体
        'font.family': 'serif', 'font.serif': [FIG_FONT, 'DejaVu Serif'],
        'mathtext.fontset': 'stix',                            # 数学符号与 Times 风格一致
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })


def save(fig, name_, desc):
    fig.savefig(f'{FIG_DIR}/{name_}.pdf')
    fig.savefig(f'{FIG_DIR}/{name_}.png', dpi=300)
    plt.close(fig)
    fig_log.append((name_, desc))


step('图 1：剂量曲线')
# ---- 图 1：剂量曲线 ----
if HAVE_MPL and dose:
    ms_ = [m for m in MODELS if m in dose]
    fig, axes = plt.subplots(1, len(ms_), figsize=(FIG_WIDTH_DOUBLE if len(ms_) > 1 else FIG_WIDTH_SINGLE, 2.6),
                             squeeze=False)
    for ax, model in zip(axes[0], ms_):
        pts, refs = dose[model]
        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                    marker='o', ms=3.5, capsize=2.5, lw=1.0, color=COLORS[0],
                    label='Ontology pool + injected same-disease')
        for k, (rm, ls) in enumerate((('random', '--'), ('mnrl', ':'))):
            if rm in refs:
                ax.axhline(refs[rm], ls=ls, lw=0.9, color=COLORS[k + 1], label=name(rm))
        ax.axvline(27, ls='-', lw=0.5, color='#BBBBBB')
        ax.text(27.5, 1.0, 'random ≈ 27%', transform=ax.get_xaxis_transform(), fontsize=8,
                color='#777777', va='top', ha='left')
        ax.set_xlabel('Injected same-disease negatives (nominal, %)')
        ax.set_ylabel('nDCG@10')
        ax.set_title(MODEL_NAME[model], loc='left')
        ax.set_xticks([p for _, p in DOSE_LEVELS])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='lower center', bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, 'fig_dose', 'nDCG@10 vs injected same-disease negatives')

step('图 2：相对 mnrl 的 Δ nDCG@10（逐题配对 bootstrap CI）')
# ---- 图 2：相对 mnrl 的 Δ nDCG@10（逐题配对 bootstrap CI）----
DELTA_METHODS = ['random', 'onto_pool_rand', 'onto_loose_rand', 'onto_pool_strat', 'onto_strat_loose',
                 'bm25', 'dense', 'dense_rank10', 'dense_rank100', 'margin', 'margin_onto']
if HAVE_MPL:
    data = {}
    for model in MODELS:
        base = per_query_mean('new', model, 'mnrl')
        if not base:
            continue
        pts = []
        for m in DELTA_METHODS:
            cur = per_query_mean('new', model, m)
            qs = sorted(set(base) & set(cur))
            if qs:
                d, lo, hi = paired_ci([cur[q] for q in qs], [base[q] for q in qs])
                pts.append((m, d * 100, lo * 100, hi * 100))
        if pts:
            data[model] = pts
    if data:
        ms_ = list(data)
        fig, axes = plt.subplots(1, len(ms_), figsize=(FIG_WIDTH_DOUBLE, 3.4), sharey=True, squeeze=False)
        for ax, model in zip(axes[0], ms_):
            pts = data[model]
            y = np.arange(len(pts))[::-1]
            for yi, (m, d, lo, hi) in zip(y, pts):
                col = COLORS[2] if m.startswith('onto') else COLORS[0] if m in ('random', 'bm25') else COLORS[5]
                ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt='o', ms=3.5, capsize=2, lw=1, color=col)
            ax.axvline(0, lw=0.7, color='#555555')
            ax.set_yticks(y)
            ax.set_yticklabels([name(m) for m, *_ in pts])
            ax.set_xlabel('Δ nDCG@10 vs in-batch only (points)')
            ax.set_title(MODEL_NAME[model], loc='left')
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], marker='o', ls='', color=COLORS[2], label='Ontology, no model'),
                            Line2D([], [], marker='o', ls='', color=COLORS[0], label='No model'),
                            Line2D([], [], marker='o', ls='', color=COLORS[5], label='Uses a retrieval model')],
                   frameon=False, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        save(fig, 'fig_delta', 'Δ nDCG@10 relative to in-batch negatives with 95% CI')

step('图 3：按数据来源的 nDCG@10')
# ---- 图 3：按数据来源的 nDCG@10 ----
SRC_METHODS = ['mnrl', 'random', 'onto_pool_rand', 'onto_loose_rand', 'margin']
SRC_NAME = {'cmdd': 'CMDD', 'huatuo26m': 'Huatuo-26M', 'medchatzh': 'MedChatZH'}
if HAVE_MPL:
    src_data = {}
    for model in MODELS:
        tab = {}
        for m in SRC_METHODS:
            runs = seed_results('new', model, m)
            if runs:
                srcs = sorted(runs[0][1].get('test_by_source', {}))
                tab[m] = {s: statistics.mean(r['test_by_source'][s]['ndcg@10'] * 100 for _, r in runs
                                             if s in r.get('test_by_source', {})) for s in srcs}
        if tab:
            src_data[model] = tab
    if src_data:
        ms_ = list(src_data)
        fig, axes = plt.subplots(1, len(ms_), figsize=(FIG_WIDTH_DOUBLE, 2.6), squeeze=False)
        for ax, model in zip(axes[0], ms_):
            tab = src_data[model]
            srcs = sorted({s for v in tab.values() for s in v})
            x = np.arange(len(srcs))
            w = 0.8 / len(tab)
            for k, (m, v) in enumerate(tab.items()):
                ax.bar(x + (k - (len(tab) - 1) / 2) * w, [v.get(s, 0) for s in srcs], w,
                       color=COLORS[k % len(COLORS)], label=name(m))
            ax.set_xticks(x)
            ax.set_xticklabels([SRC_NAME.get(s, s) for s in srcs])
            ax.set_ylabel('nDCG@10')
            ax.set_title(MODEL_NAME[model], loc='left')
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        save(fig, 'fig_by_source', 'nDCG@10 per source dataset')

step('图 4：负例疾病关系堆叠条形图')
# ---- 图 4：负例疾病关系堆叠条形图 ----
if HAVE_MPL and NEG_ORDER:
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE * 0.75, 0.24 * len(NEG_ORDER) + 1.0))
    y = np.arange(len(NEG_ORDER))[::-1]
    left = np.zeros(len(NEG_ORDER))
    for k in REL_KEYS:
        v = np.array([neg_stats[m].get(k, 0) * 100 for m in NEG_ORDER])
        ax.barh(y, v, left=left, color=REL_COLOR[k], label=k.capitalize(), height=0.7)
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([name(m) for m in NEG_ORDER])
    ax.set_xlabel('Share of hard negatives (%)')
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(0.40, 1.10), fontsize=8)
    fig.tight_layout()
    save(fig, 'fig_negatives', 'Disease relation of hard negatives per method')

step('图 5：重排效果（按模型分子图；正文只画本体池训练的模型，附录画全部）')
# ---- 图 5：重排效果（按模型分子图；正文只画本体池训练的模型，附录画全部）----
RR_METHOD_COLOR = {'mnrl': COLORS[4], 'random': COLORS[1], 'onto_pool_rand': COLORS[2],
                   'margin': COLORS[5], 'margin_onto': COLORS[3], 'onto_loose_rand': COLORS[0]}


def _rerank_fig(keys, fname, desc):
    ms_ = [m for m in MODELS if any(k.split('/', 1)[0] == m for k in keys)]
    if not ms_:
        return
    rules = [r for r in RR_RULES if r != 'none']
    fig, axes = plt.subplots(2, len(ms_), figsize=(FIG_WIDTH_DOUBLE, 4.8), squeeze=False, sharey='row')
    for col, model in enumerate(ms_):
        mk = [k for k in keys if k.split('/', 1)[0] == model]
        w = 0.8 / max(len(mk), 1)
        for j, key in enumerate(mk):
            method = key.split('/', 1)[1]
            col_c = RR_METHOD_COLOR.get(method, COLORS[j % len(COLORS)])
            comps = {c['rule']: c for c in r10[key].get('comparisons', [])}
            off = (j - (len(mk) - 1) / 2) * w
            d = [comps[r]['delta'] if r in comps else 0 for r in rules]
            err = [[comps[r]['delta'] - comps[r]['ci'][0] if r in comps else 0 for r in rules],
                   [comps[r]['ci'][1] - comps[r]['delta'] if r in comps else 0 for r in rules]]
            axes[0][col].bar(np.arange(len(rules)) + off, d, w, yerr=err, capsize=1.5, color=col_c,
                             label=name(method), error_kw={'lw': 0.6, 'ecolor': '#444444'})
            cf = [_rr_mean(r10[key].get(r), 'cross_family@10') or 0 for r in RR_RULES]
            axes[1][col].bar(np.arange(len(RR_RULES)) + off, cf, w, color=col_c)
        axes[0][col].axhline(0, lw=0.7, color='#555555')
        axes[0][col].set_xticks(np.arange(len(rules)))
        axes[0][col].set_xticklabels([RR_NAME[r] for r in rules], rotation=25, ha='right')
        axes[0][col].set_title(f'{MODEL_NAME[model]}: (a) Δ nDCG@10', loc='left')
        axes[1][col].set_xticks(np.arange(len(RR_RULES)))
        axes[1][col].set_xticklabels([RR_NAME[r] for r in RR_RULES], rotation=25, ha='right')
        axes[1][col].set_title(f'{MODEL_NAME[model]}: (b) cross-family errors', loc='left')
        if 'hard_down' in RR_RULES:      # 硬过滤后跨大类为 0，标出来以免看起来像缺数据
            axes[1][col].text(RR_RULES.index('hard_down'), 0, '0', ha='center', va='bottom', fontsize=8)
    axes[0][0].set_ylabel('Δ nDCG@10 vs no reranking')
    axes[1][0].set_ylabel('Cross-family in top-10 (%)')
    handles, labels = axes[0][0].get_legend_handles_labels()
    top = 1.0
    if len(labels) > 1:
        ncol = min(len(labels), 3)
        nrow = -(-len(labels) // ncol)
        top = 1.0 - 0.045 * (nrow + 1)          # 给顶部图例（含标题）留出空间
        fig.legend(handles, labels, frameon=False, ncol=ncol, loc='upper center',
                   bbox_to_anchor=(0.5, 1.0), fontsize=8, title='Retriever fine-tuned with', title_fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, fname, desc)


if HAVE_MPL and r10:
    _rerank_fig(main_keys, 'fig_rerank', 'Ontology reranking: Δ nDCG@10 and cross-family errors')
    if len(all_keys) > len(main_keys):
        _rerank_fig(all_keys, 'fig_rerank_appx', 'Ontology reranking for every fine-tuned retriever')

step('图 6：本体过滤修复检索式难负例 + 剂量曲线（机制图）')
# ---- 图 6：本体过滤修复检索式难负例 + 剂量曲线（机制图）----
if HAVE_MPL and filter_res:
    ncol = 2 if dose else 1
    fig, axes = plt.subplots(1, ncol, figsize=(FIG_WIDTH_DOUBLE if ncol == 2 else FIG_WIDTH_SINGLE, 3.0),
                             squeeze=False)
    ax = axes[0][0]
    bases = [b for b, _ in FILTER_PAIRS if any((m, b) in filter_res for m in MODELS)]
    ms_ = [m for m in MODELS if any((m, b) in filter_res for b in bases)]
    x = np.arange(len(bases))
    w = 0.8 / max(len(ms_), 1)
    for k, model in enumerate(ms_):
        d = [filter_res.get((model, b), {}).get('delta', 0) for b in bases]
        lo = [d[i] - filter_res.get((model, b), {}).get('lo', d[i]) for i, b in enumerate(bases)]
        hi = [filter_res.get((model, b), {}).get('hi', d[i]) - d[i] for i, b in enumerate(bases)]
        ax.bar(x + (k - (len(ms_) - 1) / 2) * w, d, w, yerr=[lo, hi], capsize=2,
               color=COLORS[k], label=MODEL_NAME[model], error_kw={'lw': 0.7})
    labels = []
    for b in bases:
        sh = rule_shares(dict(FILTER_PAIRS)[b], 'bge-base') or {}
        lab = name(b)
        labels.append(f"{lab}\n(filter applied {sh['applied']:.0%})" if sh.get('applied') is not None else lab)
    ax.axhline(0, lw=0.7, color='#555555')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Δ nDCG@10 from ontology filter')
    ax.set_title('(a) Filtering same-disease hard negatives', loc='left')
    handles, leg_labels = ax.get_legend_handles_labels()
    if dose:
        ax = axes[0][1]
        _dm = [m for m in MODELS if m in dose]
        for k, model in enumerate(_dm):
            # 误差线用 09 dose 报告里的逐题配对 95% CI（与表中的显著性检验一致），不用种子标准差
            pts = dose_delta(r09d, model)
            if pts:
                plot_dose_delta(ax, pts, COLORS[k], MODEL_NAME[model], dx=(k - (len(_dm) - 1) / 2) * 0.8)
        ax.axhline(0, lw=0.7, color='#555555')
        ax.set_xticks([p for _, p in DOSE_LEVELS])
        ax.set_xlabel('Random same-disease negatives injected (nominal, %)')
        ax.set_ylabel('Δ nDCG@10 vs 0% (paired 95% CI)')
        ax.set_title('(b) Injecting random same-disease negatives', loc='left')
    # 两个子图颜色一致（按模型），共用一个图例，放在图下方，避免压住误差线
    fig.legend(handles, leg_labels, frameon=False, ncol=len(leg_labels), loc='lower center',
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, 'fig_mechanism', 'Similar same-disease negatives hurt, random ones do not')

step('图 7：训练曲线（验证集 nDCG@10 随步数）')
# ---- 图 7：训练曲线（验证集 nDCG@10 随步数）----
if HAVE_MPL:
    curve_methods = ['mnrl', 'random', 'margin', 'onto_pool_rand']
    curves = {}
    for model in MODELS:
        for m in curve_methods:
            for s in SEEDS:
                p = f'{OUT_DIR}/results_new/{model}/{m}/seed{s}/train_log.json'
                if os.path.exists(p):
                    with open(p, encoding='utf-8') as f:
                        log = json.load(f)
                    pts = [(e['step'], e['eval_dev_cosine_ndcg@10'] * 100) for e in log
                           if 'eval_dev_cosine_ndcg@10' in e]
                    if pts:
                        curves.setdefault(model, {}).setdefault(m, []).append(pts)
    if curves:
        ms_ = list(curves)
        fig, axes = plt.subplots(1, len(ms_), figsize=(FIG_WIDTH_DOUBLE, 2.6), squeeze=False)
        for ax, model in zip(axes[0], ms_):
            for k, m in enumerate(curve_methods):
                runs = curves[model].get(m)
                if not runs:
                    continue
                # 只取所有种子都评测过的步数，避免某一步只平均了部分种子
                steps = sorted(set.intersection(*[{st for st, _ in r} for r in runs]))
                if not steps:
                    continue
                by_run = [dict(r) for r in runs]
                mean = [statistics.mean(br[x_] for br in by_run) for x_ in steps]
                ax.plot(steps, mean, lw=1.0, color=COLORS[k], label=f'{name(m)} (n={len(runs)})')
            ax.set_xlabel('Training step')
            ax.set_ylabel('Validation nDCG@10')
            ax.set_title(MODEL_NAME[model], loc='left')
        axes[0][0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        save(fig, 'fig_training', 'Validation nDCG@10 during training (mean over seeds)')

step('图 8：本体结构（各疾病大类下的概念数与训练问题数）')
# ---- 图 8：本体结构（各疾病大类下的概念数与训练问题数）----
if HAVE_MPL and os.path.exists(f'{OUT_DIR}/ontology_new/concepts.json'):
    with open(f'{OUT_DIR}/ontology_new/concepts.json', encoding='utf-8') as f:
        cons = json.load(f)['concepts']
    fam_n = defaultdict(int)
    for c in cons:
        fam_n[c.get('family', '-')] += 1
    qc = (r06 or {}).get('question_concept_counts', {})
    name2fam = {c['name']: c.get('family', '-') for c in cons}
    fam_q = defaultdict(int)
    for nm, k in qc.items():
        fam_q[name2fam.get(nm, '-')] += k
    FAMILY_NAME = {'viral': 'Viral hepatitis', 'cirrhosis': 'Fibrosis and cirrhosis', 'fatty': 'Fatty liver disease',
                   'cancer': 'Liver cancer', 'alcohol': 'Alcoholic liver disease', 'cyst': 'Liver cyst',
                   'failure': 'Liver failure', 'portal_vascular': 'Portal and vascular disorders',
                   'other_inflammatory': 'Other inflammatory diseases', 'benign_tumor': 'Benign liver tumor',
                   'toxic': 'Toxic liver disease', 'parasitic': 'Parasitic liver disease',
                   'metabolic': 'Metabolic liver disease'}
    fams = sorted(fam_n, key=lambda f_: -fam_q.get(f_, 0))
    fig, ax = plt.subplots(figsize=(FIG_WIDTH_SINGLE * 1.8, 0.27 * len(fams) + 0.9))
    y = np.arange(len(fams))[::-1]
    ax.barh(y, [fam_q.get(f_, 0) for f_ in fams], color=COLORS[0], height=0.7)
    for yi, f_ in zip(y, fams):
        k = fam_n[f_]
        ax.text(fam_q.get(f_, 0), yi, f'  {k} concept' + ('' if k == 1 else 's'), va='center', fontsize=8)
    ax.set_xlim(0, max(fam_q.values() or [1]) * 1.18)
    ax.set_yticks(y)
    ax.set_yticklabels([FAMILY_NAME.get(f_, f_.replace('_', ' ')) for f_ in fams])
    ax.set_xlabel('Training questions labeled with the family')
    fig.tight_layout()
    save(fig, 'fig_ontology', 'Disease families: concepts and labeled training questions')

step('图 9：方法框架图（一个本体、三种用法：过滤挖掘负例、本体负例池、推理重排）')
# ---- 图 9：方法框架图（两个领域通用，不含具体数字；数字在结果部分报告）----
if HAVE_MPL:
    from matplotlib.patches import FancyBboxPatch

    FW_H = 3.6                       # 图高（英寸）
    LH = 10.5 / (FW_H * 72)          # 8 pt 字的行距（坐标轴单位）

    def _box(ax, x, y, w, h, title, body=(), fc='#EAF2FB', fs=8):
        '''title 每行加粗；body 每项是字符串，或 (字符串, 文字参数)；空字符串占一行空白'''
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle='round,pad=0.004,rounding_size=0.010',
                                    fc=fc, ec='#333333', lw=0.7))
        lines = [(t, {'fontweight': 'bold'}) for t in title] + \
                [(b, {}) if isinstance(b, str) else b for b in body]
        top = y + (len(lines) - 1) * LH / 2
        for k, (t, kw) in enumerate(lines):
            if t:
                ax.text(x, top - k * LH, t, ha='center', va='center', fontsize=fs, **kw)

    def _arrow(ax, p, q, ls='-'):
        ax.annotate('', xy=q, xytext=p, arrowprops=dict(
            arrowstyle='-|>', lw=0.7, color='#333333', linestyle=ls, shrinkA=0, shrinkB=0,
            mutation_scale=7))

    def _route(ax, pts, ls=':'):
        '''折线：前几段画线，最后一段画箭头'''
        ax.plot(*zip(*pts[:-1]), color='#333333', lw=0.7, ls=ls)
        _arrow(ax, pts[-2], pts[-1], ls=ls)

    def _band(ax, x0, x1, label, fc, y0=0.02, y1=0.93):
        ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle='square,pad=0',
                                    fc=fc, ec='none', zorder=0))
        ax.text(x0 + 0.007, y1 - 0.012, label, fontsize=8.5, fontweight='bold', ha='left', va='top')

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, FW_H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    _band(ax, 0.005, 0.345, 'Disease knowledge', '#F7FAFD')
    _band(ax, 0.355, 0.745, 'Training: hard-negative selection', '#F6FBF8')
    _band(ax, 0.755, 0.995, 'Inference', '#FFFAF3')

    # 左：本体 → 标注（四种关系的颜色与负例关系图一致）
    YK, HK, WK = 0.455, 0.80, 0.15
    _box(ax, 0.088, YK, WK, HK, ['Disease ontology'],
         ['one per domain:', 'liver, neurology', '',
          'ICD-10 concepts', 'Chinese synonyms', 'disease families', '',
          'linked entities:', 'symptoms, drugs,', 'treatments, exams,', 'causes', '',
          'from Wikidata,', 'MSD Manual and', 'Chinese Wikipedia'])
    _box(ax, 0.262, YK, WK, HK, ['Disease labeling'],
         ['diseases named in', 'each question and', 'each answer', '',
          'question–answer', 'relation:',
          ('same disease', {'color': REL_COLOR['same'], 'fontweight': 'bold'}),
          ('same family', {'color': REL_COLOR['sibling'], 'fontweight': 'bold'}),
          ('different family', {'color': REL_COLOR['different'], 'fontweight': 'bold'}),
          ('unnamed', {'color': '#777777', 'fontweight': 'bold'})])
    _arrow(ax, (0.088 + WK / 2, YK), (0.262 - WK / 2, YK))

    # 中：训练阶段的两种用法（过滤挖掘负例是主要用法，放上面）→ 对比学习微调
    YA, YB, HT, WT = 0.665, 0.270, 0.32, 0.18
    _box(ax, 0.455, YA, WT, HT, ['Filter mined negatives'],
         ['BM25 / dense top-100', 'drop same-disease and', 'unnamed candidates',
          'keep the highest-ranked', 'remaining candidate'], fc='#E9F6EF')
    _box(ax, 0.455, YB, WT, HT, ['Ontology negative pool'],
         [('no retrieval model', {'style': 'italic'}), 'safe: other diseases',
          'loose: no same disease', 'negative drawn at random'], fc='#E9F6EF')
    WF = 0.16
    _box(ax, 0.655, 0.468, WF, 0.50, ['Contrastive', 'fine-tuning'],
         ['(question, positive,', 'hard negative)', '+ in-batch negatives', '', 'BGE / Qwen3'], fc='#F2F2F2')
    for y_ in (YA, YB):
        _arrow(ax, (0.262 + WK / 2, y_), (0.455 - WT / 2, y_), ls=':')    # 本体标签
        _arrow(ax, (0.455 + WT / 2, y_), (0.655 - WF / 2, y_))

    # 右：推理阶段的本体重排 → 评价
    _box(ax, 0.875, YA, 0.20, HT, ['Ontology reranking'],
         ['top-100 answers of the', 'fine-tuned retriever', 'same disease +λ', 'different family −λ',
          '(λ chosen on dev)'], fc='#FFF4E6')
    _box(ax, 0.875, YB, 0.20, HT, ['Ranked answers'],
         ['evaluated by nDCG@10', 'and cross-family', 'answers in the top-10'], fc='#FFFFFF')
    _arrow(ax, (0.655 + WF / 2, YA), (0.875 - 0.10, YA))
    _arrow(ax, (0.875, YA - HT / 2), (0.875, YB + HT / 2))

    # 本体标签送到推理阶段：从标注框顶部绕过训练区
    Y_BUS = 0.955
    _route(ax, [(0.262, YK + HK / 2), (0.262, Y_BUS), (0.875, Y_BUS), (0.875, YA + HT / 2)])
    ax.text(0.60, Y_BUS + 0.006, 'disease labels', fontsize=7, style='italic', ha='center', va='bottom')

    save(fig, 'fig_framework', 'Overview of the ontology-guided framework (domain-independent)')

# %%
# =============================================================================
step('跨领域合并表：肝病 + 神经科学（tables_domains.docx）')
# 跨领域合并表：肝病 + 神经科学（tables_domains.docx）

dom_log, dom_md = [], ['# Cross-domain tables (liver + neurology)', '',
                       'Generated by step 11 (liver). Word version: `tables_domains.docx`; CSV in `tables/`.', '']
DOM_DOC = _new_doc()
DOM = []   # [(领域名, 目录, 模型列表)]
for dname, droot in DOMAINS:
    if os.path.isdir(f'{droot}/reports'):
        dmodels = [m for m in MODELS if os.path.isdir(f'{droot}/results_new/{m}')]
        DOM.append((dname, droot, dmodels))
    else:
        SKIPPED.append(f'{droot}/reports (cross-domain tables)')


def _dload(root, fname):
    p = f'{root}/reports/{fname}'
    if not os.path.exists(p):
        SKIPPED.append(p)
        return None
    with open(p, encoding='utf-8') as f:
        return json.load(f)


def _dwrite(name_, caption, headers, rows, **kw):
    write_table(name_, caption, headers, rows, doc=DOM_DOC, md=dom_md, log=dom_log, **kw)


DOM_REP = {}
for dname, droot, dmodels in DOM:
    DOM_REP[dname] = {'r04': _dload(droot, '04_split.json'), 'r06': _dload(droot, '06_ontology_new.json'),
                      'r08': _dload(droot, '08_training_sets_bge-base_new.json'),
                      'r08c': _dload(droot, '08_training_sets_bge-base_new_core.json'),
                      'r09': _dload(droot, '09_results_new.json'),
                      'r09d': _dload(droot, '09_results_new_dose.json'), 'r10': _dload(droot, '10_rerank_new.json')}

if len(DOM) >= 2:
    dnames = [d for d, _, _ in DOM]
    cols = [(d, r, m) for d, r, ms_ in DOM for m in ms_]

    def _fmt(v, pct_of=None):
        if v is None:
            return '-'
        return f'{v:,} ({v / pct_of:.1%})' if pct_of else f'{v:,}'

    def _g(d, *ks):
        for k in ks:
            d = (d or {}).get(k)
        return d

    def _dm(d, m):
        return f'{d} · {MODEL_NAME[m]}'

    def _stack(build):
        '''按 领域 × 模型 上下拼接：build(d, root, model) 返回行列表；返回 (rows, groups)'''
        rows, groups = [], []
        for d, r, m in cols:
            rows_m = build(d, r, m)
            if rows_m:
                groups.append((len(rows), _dm(d, m)))
                rows += rows_m
        return rows, groups

    # 各领域测试集的本体标注（覆盖率 + 正确答案的疾病关系）；本领域已在前面算过
    for d, r, _ in DOM:
        if r == OUT_DIR:
            DOM_REP[d]['test'] = (TEST_COV, TEST_REL)
        else:
            try:
                DOM_REP[d]['test'] = test_ontology_stats(r, desc=f'{d} ')
            except Exception as _e:
                SKIPPED.append(f'{r}: test-set ontology coverage ({_e})')
                DOM_REP[d]['test'] = (None, None)

    # 各领域的概念与大类（精神科概念只用来标记"答案讲的是别的病"，不算本领域的疾病大类，与 fig_ontology 一致）
    DOM_CONCEPTS = {}
    for d, r, _ in DOM:
        p = f'{r}/ontology_new/concepts.json'
        if os.path.exists(p):
            with open(p, encoding='utf-8') as f:
                DOM_CONCEPTS[d] = json.load(f)['concepts']

    # ---- 表 D1：本体规模与标注覆盖（第四节）----
    rows = []

    def _row(label, fn):
        rows.append([label] + [fn(d, DOM_REP[d]) for d in dnames])

    _row('Disease concepts', lambda d, R: _fmt(_g(R['r06'], 'n_concepts')))
    _row('Disease families', lambda d, R: str(len({c.get('family', '-') for c in DOM_CONCEPTS[d]} - {'psych'}))
         if d in DOM_CONCEPTS else '-')
    _row('Surface terms (synonyms)', lambda d, R: _fmt(_g(R['r06'], 'n_terms')))
    _row('Entities (symptom / drug / treatment / exam / cause)',
         lambda d, R: _fmt(_g(R['r06'], 'relation_layer', 'entities')))
    _row('Disease–entity relations (used)', lambda d, R: _fmt(_g(R['r06'], 'relation_layer', 'relations_used')))
    for key, label in (('relations_by_type', '  relations of type {}'), ('relations_by_source', '  relations from {}')):
        tot = defaultdict(int)
        for d in dnames:
            for t, k in (_g(DOM_REP[d]['r06'], 'relation_layer', key) or {}).items():
                tot[t] += k
        for t in sorted(tot, key=lambda x: -tot[x]):
            _row(label.format(SRC_LABEL.get(t, t) if key == 'relations_by_source' else t),
                 lambda d, R, t=t: _fmt((_g(R['r06'], 'relation_layer', key) or {}).get(t, 0)))
    _row('Training questions', lambda d, R: _fmt(_g(R['r06'], 'questions')))
    _row('  with a disease label (question text)',
         lambda d, R: _fmt(_g(R['r06'], 'with_concept'), _g(R['r06'], 'questions')))
    _row('  including labels from the correct answer',
         lambda d, R: _fmt(_g(R['r08'], 'ontology', 'labelled_questions'), _g(R['r06'], 'questions')))
    _row('  with more than one disease', lambda d, R: _fmt(_g(R['r06'], 'multi_concept')))
    _row('Training answers', lambda d, R: _fmt(_g(R['r06'], 'answers')))
    _row('  naming a disease', lambda d, R: _fmt(_g(R['r06'], 'answers_specific'), _g(R['r06'], 'answers')))
    _row('  naming no disease (unnamed)', lambda d, R: _fmt(_g(R['r06'], 'answers_unnamed'), _g(R['r06'], 'answers')))
    def _prior(R):
        '''随机 (问题, 答案) 对的疾病关系分布：same / different / unnamed'''
        p = _g(R['r06'], 'random_pair_relation')
        return ' / '.join(f'{p.get(k, 0):.1%}' for k in ('same', 'different', 'unnamed')) if p else '-'

    def _rand_same(R):
        '''随机负例中同病的比例（全部训练问题为分母，与负例表、剂量图一致）'''
        v = _g(R['r08'], 'methods', 'random', 'same')
        return f'{v:.1%}' if v is not None else '-'

    _row('Random question–answer pairs (labeled questions): same / different / unnamed', lambda d, R: _prior(R))
    _row('Random negatives naming the question\'s disease', lambda d, R: _rand_same(R))
    _row('Test questions', lambda d, R: _fmt((R['test'][0] or {}).get('queries')))
    _row('  with a disease label',
         lambda d, R: _fmt((R['test'][0] or {}).get('queries_lab'), (R['test'][0] or {}).get('queries')))
    _row('Test retrieval corpus (answers)', lambda d, R: _fmt((R['test'][0] or {}).get('answers')))
    _row('  naming a disease',
         lambda d, R: _fmt((R['test'][0] or {}).get('answers_lab'), (R['test'][0] or {}).get('answers')))
    n_psych = {d: sum(c.get('family') == 'psych' for c in cs) for d, cs in DOM_CONCEPTS.items()}
    psych_note = ' '.join(f'The {d.lower()} concepts include {NUM_WORD.get(k, k)} psychiatric concepts that are kept '
                          f'only to mark answers about a different disease; they are not counted as a disease family.'
                          for d, k in n_psych.items() if k)
    _dwrite('tab_ontology_domains', 'Disease ontologies and annotation coverage of the two domains.',
            ['Item', *dnames], rows, align='l' + 'r' * len(dnames),
            note='Concepts follow WHO ICD-10 with Chinese synonyms; entity relations come from the sources listed '
                 'in the table (0: the source is not used in that domain).' + (' ' + psych_note if psych_note else '')
                 + ' Training '
                 'and test items are labeled with the same rules: a question from its own text and an answer '
                 'from its own text. During training, a question whose text names no disease takes the diseases '
                 'named in its correct answer (at most three); the remaining questions are "Unknown" in the '
                 'negative tables. Test questions are labeled from the question text only, since the correct '
                 'answer is not known at retrieval time. "Random negatives naming the question\'s disease": share '
                 'of training questions whose uniformly sampled negative names the question\'s disease, i.e., how '
                 'often a random negative is a likely false negative.')

    # ---- 表 D1b：训练配置（第六节）----
    rows = []
    for d, r, m in cols:
        cells = setup_cells(m, r)
        if cells:
            rows.append([d, MODEL_NAME[m], *cells])
    if rows:
        _dwrite('tab_setup_domains', 'Fine-tuning configuration in both disease domains.',
                ['Domain', 'Model', *SETUP_HEAD], rows, align='lll' + 'l' + 'r' * 8,
                note=SETUP_NOTE + ' The neurology domain is a control domain and uses BGE-base-zh only.')

    # ---- 表 D3：负例的疾病关系（同病 / 未命名）----
    D_NEG = ['random', 'bm25', 'dense', 'margin', 'bm25_onto', 'dense_onto', 'margin_onto',
             'onto_pool_rand', 'onto_loose_rand']
    rows = []
    for meth in D_NEG:
        row = [name(meth)]
        for d in dnames:
            st = {}
            for key in ('r08', 'r08c'):
                st = (DOM_REP[d][key] or {}).get('methods', {}).get(meth) or st
                if st:
                    break
            row += [f"{st['same']:.1%}" if 'same' in st else '-', f"{st['unnamed']:.1%}" if 'unnamed' in st else '-']
        if any(c != '-' for c in row[1:]):
            rows.append(row)
    _dwrite('tab_negatives_domains', 'Share of hard negatives that name the question\'s disease (same) or no disease (unnamed).',
            ['Method', *[f'{d}: {k}' for d in dnames for k in ('same', 'unnamed')]], rows,
            align='l' + 'r' * (2 * len(dnames)),
            note='Relations are assigned by the ontology on the BGE-base-zh training files. Same-disease '
                 'negatives are likely false negatives; questions without a disease label (neither from the '
                 'question text nor from the correct answer) are not counted as same or unnamed.')

    # ---- 表 D2：主结果精简版（nDCG@10，列 = 领域 × 模型）----
    D_MAIN = ['mnrl', 'random', 'bm25', 'dense', 'dense_rank100', 'margin', 'margin_onto',
              'onto_pool_rand', 'onto_loose_rand']
    rows = []
    for meth in D_MAIN:
        row = [name(meth)]
        for _, r, m in cols:
            runs = seed_results('new', m, meth, r)
            row.append(ms([x['test']['ndcg@10'] for _, x in runs]) if runs else '-')
        if any(c != '-' for c in row[1:]):
            rows.append(row)
    _short = {'bge-base': 'BGE', 'qwen3-0.6b': 'Qwen3'}
    _dwrite('tab_main_domains', 'nDCG@10 on the test set in both disease domains (mean ± std over three seeds).',
            ['Method', *[f'{d} · {_short.get(m, m)}' for d, _, m in cols]], rows, align='l' + 'r' * len(cols),
            note='Full results with MRR@10 and recall are in the per-domain tables. The neurology domain is a '
                 'control domain and uses BGE-base-zh only.' + SUITE_NOTE)

    # ---- 表 D2b：关键配对比较（各领域 × 各模型，取自各领域的 09 报告）----
    rows, groups = _stack(lambda d, r, m: compare_rows(DOM_REP[d]['r09'], m))
    if rows:
        _dwrite('tab_compare_domains', 'Paired comparisons on per-query nDCG@10 in both disease domains (test set).',
                COMPARE_HEAD, rows, align='lrrrrc', groups=groups,
                note=COMPARE_NOTE.replace('within each model', 'within each model and domain'))

    # ---- 表 D4：本体过滤修复检索式难负例----
    rows, groups, dom_filter = [], [], {}
    for d, r, ms_ in DOM:
        fr = filter_res if r == OUT_DIR else filter_repair(ms_, r)
        dom_filter[d] = fr
        for m in ms_:
            ks = [b for (mm, b) in fr if mm == m]
            if not ks:
                continue
            groups.append((len(rows), f'{d} · {MODEL_NAME[m]}'))
            for b in ks:
                rows.append([name(b)] + filter_repair_row(fr[(m, b)], b, m, r))
    if rows:
        _dwrite('tab_filter_repair_domains', 'Ontology filtering of retrieval-mined hard negatives in both '
                'disease domains (nDCG@10, test set).',
                ['Mining method', 'Filter applied', 'Fallback', 'Negative changed', 'Original',
                 '+ Ontology filter', 'Δ [95% CI]', 'p (Holm)', 'Seeds better', 'Gap to MNRL recovered'],
                rows, align='lrrrrrrrrr', groups=groups,
                note='Same filter and statistics as in the per-domain table: same-disease and unnamed candidates '
                     'are removed and the highest-ranked remaining candidate is used. Holm correction is applied '
                     'within each domain. "Gap to MNRL recovered": Δ divided by the distance from the original '
                     'method to in-batch-only training.' + SUITE_NOTE)

    # ---- 表 D4b：检索式挖负例规则的执行情况 ----
    rows, groups = _stack(lambda d, r, m: exec_rows(m, r))
    if rows:
        _dwrite('tab_rule_execution_domains', 'How often each retrieval-based mining rule could be applied, in both '
                'disease domains.', EXEC_HEAD, rows, align='lrrrrr', groups=groups,
                note=EXEC_NOTE.format('lower than for the other filtered methods'))

    # ---- 表 D5：剂量实验（注入随机同病负例）----
    D_DOSE_CMP = [('onto_dose_p10', 'onto_pool_rand'), ('onto_dose_p30', 'onto_pool_rand'),
                  ('onto_dose_p50', 'onto_pool_rand')]
    rows, groups, dom_dose = [], [], {}
    for d, r, ms_ in DOM:
        rep = DOM_REP[d]['r09d'] or {}
        dom_dose[d] = dose if r == OUT_DIR else dose_points(ms_, r)
        for m in ms_:
            comps = {(c['method'], c['baseline']): c for c in (rep.get(m) or {}).get('comparisons', [])}
            pts = dict((p[0], p) for p in (dom_dose[d].get(m) or ([], {}))[0])
            if not comps and not pts:
                continue
            groups.append((len(rows), f'{d} · {MODEL_NAME[m]}'))
            base = pts.get(0)
            if base:   # 0% 基准行：只给 nDCG@10，便于读出各注入比例的绝对分数
                rows.append([f'{name(DOSE_LEVELS[0][0])} (0%)', f'{base[1]:.2f} ± {base[2]:.2f}', '-', '-', '-', '-'])
            for a, b in D_DOSE_CMP:
                c = comps.get((a, b))
                if not c:
                    continue
                pct = dict(DOSE_LEVELS).get(a)
                lev = pts.get(pct)
                rows.append([f'{name(a)} vs {name(b)}',
                             f'{lev[1]:.2f} ± {lev[2]:.2f}' if lev else '-',
                             sgn(c['delta_ndcg@10']), f"[{sgn(c['ci95'][0])}, {sgn(c['ci95'][1])}]",
                             f"{c.get('p_holm', c['p']):.4f}", f"{c['seeds_better']}/{c['seeds_compared']}"])
    if rows:
        _dwrite('tab_dose_domains', 'Injecting random same-disease negatives into the ontology pool in both domains.',
                ['Comparison', 'nDCG@10', 'Δ nDCG@10', '95% CI', 'p (Holm)', 'Seeds better'], rows,
                align='lrrrrr', groups=groups,
                note='Each training pair draws a random same-disease answer with the given probability, otherwise '
                     'a safe-pool answer; questions without a disease label keep a random negative. Paired '
                     'bootstrap and permutation test on per-query nDCG@10 averaged over seeds; Holm over all '
                     'pre-declared comparisons of the dose experiment within each domain. Comparisons of the '
                     'ontology pool with Random and in-batch-only training are reported from the main experiment.'
                     + SUITE_NOTE)

    # ---- 表 D6：重排汇总 ----
    rows, groups = [], []
    for d, r, ms_ in DOM:
        r10_ = DOM_REP[d]['r10']
        if not r10_:
            continue
        keys_ = rr_keys(r10_, ms_)
        rr_rows_ = rerank_summary_rows(rerank_summary(r10_, keys_))
        if rr_rows_:
            groups.append((len(rows), f'{d} ({len(keys_)} fine-tuned retrievers)'))
            rows += rr_rows_
    if rows:
        _dwrite('tab_rerank_domains', 'Inference-time ontology reranking in both disease domains.',
                ['Rule', 'Δ nDCG@10 vs none (range)', 'Significant better / worse / n.s.',
                 'Relative change in cross-family answers'], rows, align='lrcr', groups=groups,
                note='Ranges are over all fine-tuned retrievers of a domain (each model × training method, three '
                     'seeds each). Significance: Holm-corrected p < 0.05 within each retriever. Relative change: '
                     'cross-family share in top-10 after reranking divided by the share before, minus one.')

    # ---- 表 D7：测试集正确答案与问题的疾病关系（重排诊断）----
    rel_of = {d: DOM_REP[d]['test'][1] for d in dnames}
    if all(rel_of.values()):
        lab_of = {d: sum(v for k, v in rel_of[d].items() if k != 'unknown') for d in dnames}
        rows = [[k.capitalize()] + [x for d in dnames for x in
                                    (f'{rel_of[d][k]:.1%}',
                                     f'{rel_of[d][k] / lab_of[d]:.1%}' if k != 'unknown' and lab_of[d] else '-')]
                for k in REL_KEYS]
        _dwrite('tab_rerank_diag_domains', 'Disease relation between test questions and their correct answers in '
                'both disease domains.',
                ['Relation', *[f'{d}: {h}' for d in dnames for h in ('all pairs', 'labeled questions')]], rows,
                align='l' + 'r' * (2 * len(dnames)),
                note='Share of all (question, correct answer) pairs, and share among pairs whose question has a '
                     'disease label. Labels use the same ontology and matching rules as negative mining and '
                     'reranking. A correct answer labeled "different" either names another disease (e.g., a '
                     'complication) or reflects a labeling miss; the hard-demotion rule pushes such answers out of '
                     'the top-10, which is consistent with its small nDCG loss. Pairs, not questions, are counted '
                     '(a question may have several correct answers).')
    else:
        SKIPPED.append('tab_rerank_diag_domains (test-set labels unavailable for a domain)')

    # ---- 附录：完整主结果、消融、被选负例排名、重排明细、训练收敛 ----
    rows, groups = _stack(lambda d, r, m: main_rows(m, r))
    if rows:
        _dwrite('tab_main_full_domains', 'Retrieval results on the test set in both disease domains (mean ± std '
                'over training seeds).', MAIN_HEAD, rows, align='lr' + 'r' * len(SHOW), groups=groups,
                note=MAIN_NOTE)

    rows, groups = _stack(lambda d, r, m: core_rows(m, r))
    if rows:
        _dwrite('tab_ablation_domains', 'Ablation in both disease domains: ontology-based difficulty scoring and '
                'ontology filtering of BM25 / dense candidates.', ['Method', 'Seeds', 'nDCG@10', 'R@10'], rows,
                align='lrrr', groups=groups, note=CORE_NOTE)

    rows, groups = [], []
    for d in dnames:
        rows_d = rank_rows((DOM_REP[d]['r08c'] or {}).get('methods', {}))
        if rows_d:
            groups.append((len(rows), f'{d} · BGE-base-zh'))
            rows += rows_d
    if rows:
        _dwrite('tab_negative_rank_domains', 'Rank of the selected hard negative in its candidate list, in both '
                'disease domains (BGE-base-zh).', RANK_HEAD, rows, align='lrrrrr', groups=groups,
                note='Rank 1 is the most similar candidate (BM25 or dense, top-100). Statistics come from the '
                     'BGE-base-zh training files and include questions where a fallback applies. With the BM25 '
                     'candidates the filter applies to most questions, and the selected negative still sits near '
                     'the top of the list, so the gain is not explained by switching to easy negatives. The dense '
                     'and TopK-PercPos rows are dominated by fallbacks (see the rule-execution table).')

    rows, groups = [], []
    for d, r, ms_ in DOM:
        r10_ = DOM_REP[d]['r10']
        if not r10_:
            continue
        rows_d, groups_d = _rr_rows(rr_keys(r10_, ms_), r10_, prefix=f'{d} · ')
        groups += [(i + len(rows), g) for i, g in groups_d]
        rows += rows_d
    if rows:
        _dwrite('tab_rerank_appx_domains', 'Ontology reranking for every fine-tuned retriever in both disease '
                'domains.', RR_HEAD, rows, align='llrrrrr', groups=groups, note=RR_NOTE)

    rows, groups = _stack(lambda d, r, m: conv_rows(m, r))
    if rows:
        _dwrite('tab_training_domains', 'Validation nDCG@10 of the selected checkpoint and training cost in both '
                'disease domains.', CONV_HEAD, rows, align='llrrlrr', groups=groups,
                note='Validation scores are computed on a 10,000-answer validation corpus and are used only for '
                     'model selection; they are not comparable with test scores.')

    with open(f'{PAPER_DIR}/tables_domains.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(dom_md) + '\n')
    if DOM_DOC is not None and dom_log:
        DOM_DOC.save(f'{PAPER_DIR}/tables_domains.docx')

    # ---- 图：跨领域机制图 (a) 本体过滤的提升 (b) 剂量曲线 ----
    if HAVE_MPL:
        series = [(d, m) for d, _, ms_ in DOM for m in ms_]
        DOM_COLOR = {s: COLORS[k % len(COLORS)] for k, s in enumerate(series)}
        DOM_MARK = {d: mk for d, mk in zip(dnames, ('o', 's', '^', 'D'))}

        def _lab(d, m):
            return f'{d} · {MODEL_NAME[m]}'

        fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, 3.1), squeeze=False)
        ax = axes[0][0]
        bases = [b for b, _ in FILTER_PAIRS]
        x = np.arange(len(bases))
        w = 0.8 / max(len(series), 1)
        for k, (d, m) in enumerate(series):
            fr = dom_filter.get(d) or {}
            dd = [fr.get((m, b), {}).get('delta', 0) for b in bases]
            lo = [dd[i] - fr.get((m, b), {}).get('lo', dd[i]) for i, b in enumerate(bases)]
            hi = [fr.get((m, b), {}).get('hi', dd[i]) - dd[i] for i, b in enumerate(bases)]
            ax.bar(x + (k - (len(series) - 1) / 2) * w, dd, w, yerr=[lo, hi], capsize=2,
                   color=DOM_COLOR[(d, m)], label=_lab(d, m), error_kw={'lw': 0.7})
        ax.axhline(0, lw=0.7, color='#555555')
        ax.set_xticks(x)
        ax.set_xticklabels([name(b) for b in bases])
        ax.set_ylabel('Δ nDCG@10 from ontology filter')
        ax.set_title('(a) Filtering same-disease hard negatives', loc='left')
        ax = axes[0][1]
        for k, (d, m) in enumerate(series):
            pts = dose_delta(DOM_REP[d]['r09d'], m)
            if pts:
                plot_dose_delta(ax, pts, DOM_COLOR[(d, m)], _lab(d, m), marker=DOM_MARK[d],
                                dx=(k - (len(series) - 1) / 2) * 0.8)
        ax.axhline(0, lw=0.7, color='#555555')
        ax.set_xticks([p for _, p in DOSE_LEVELS])
        ax.set_xlabel('Random same-disease negatives injected (nominal, %)')
        ax.set_ylabel('Δ nDCG@10 vs 0% (paired 95% CI)')
        ax.set_title('(b) Injecting random same-disease negatives', loc='left')
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=len(labels), loc='lower center', bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        save(fig, 'fig_mechanism_domains', 'Filtering mined negatives and injecting random same-disease '
             'negatives in both domains')
else:
    SKIPPED.append('cross-domain tables (need both domain directories: ' + ', '.join(r for _, r in DOMAINS) + ')')

# %%
# =============================================================================
# 清单
# =============================================================================

print('=' * 70)
print(f'Paper assets (steps 06–10) written to {PAPER_DIR}/')
print('=' * 70)
print('\nTables ({}; CSV in tables/; preview in tables_06_10.md):'.format(
    'Word: tables_06_10.docx' if _DOC is not None else 'Word skipped — install python-docx'))
for n_, cap in table_log:
    print(f'  {n_:<20s} {cap}')
if dom_log:
    print('\nCross-domain tables (Word: tables_domains.docx; preview in tables_domains.md):')
    for n_, cap in dom_log:
        print(f'  {n_:<26s} {cap}')
if fig_log:
    print('\nFigures (figures/*.pdf for the paper, *.png for preview; font: {}):'.format(FIG_FONT))
    for n_, d in fig_log:
        print(f'  {n_:<20s} {d}')
if SKIPPED:
    print('\nSkipped (input not found):')
    for p in sorted(set(SKIPPED)):
        print('  ' + p)

print('\nDone in {:.1f} min.'.format((time.time() - _T0) / 60))
