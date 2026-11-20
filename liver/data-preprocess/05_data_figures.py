
# =============================================================================
# 配置
# =============================================================================

OUT_DIR    = '/home/liver/data/liver_dataset'     # 和 00–04 一致
REPORT_DIR = f'{OUT_DIR}/reports'
SPLIT_DIR  = f'{OUT_DIR}/split'
PAPER_DIR  = f'{OUT_DIR}/paper'
NEURO_DIR = '/data/neuro/data/neuro_dataset'
DOMAINS   = [('Liver', OUT_DIR), ('Neurology', NEURO_DIR)]
SOURCE_META = {
    'cmdd':      {'name': 'CMDD',       'type': 'Online consultation',          'cite': 'cmdd'},
    'huatuo26m': {'name': 'Huatuo-26M', 'type': 'Encyclopedia, KG and consultation QA', 'cite': 'li2023huatuo'},
    'medchatzh': {'name': 'MedChatZH',  'type': 'Medical dialogue',             'cite': 'tan2024medchatzh'},
}

# 图宽（英寸）。IEEE 双栏 7.16 in、单栏 3.5 in；Elsevier 双栏 190 mm ≈ 7.48 in、单栏 90 mm ≈ 3.54 in。
FIG_WIDTH_DOUBLE = 7.16   # IEEE 通栏宽度（英寸）
FIG_WIDTH_SINGLE = 3.5    # IEEE 单栏宽度（英寸）

LONG_ANSWER_CHARS = 480       # 和 01/03 一致
N_EXAMPLES        = 5         # 每个来源给几条候选示例
SEED              = 42

# Okabe–Ito 色盲友好配色
COLORS = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00']

# %%
import csv, json, os, random, statistics
from collections import defaultdict
N_STEPS = 9
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


TAB_DIR = f'{PAPER_DIR}/tables'
FIG_DIR = f'{PAPER_DIR}/figures'
os.makedirs(TAB_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def load_report(name, root=OUT_DIR):
    path = f'{root}/reports/{name}.json'
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found — run the corresponding step first.')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def domain_data(root, desc='splits'):
    '''读一个领域的 00–04 报告和划分数据；05 的全部表格都只用到这些'''
    D = {'r00': load_report('00_concat', root), 'r02': load_report('02_merge', root),
         'r03': load_report('03_dedup', root), 'r04': load_report('04_split', root)}
    D['SOURCES'] = list(D['r02']['config']['SOURCES'])
    D['r01'] = {s: load_report(f'01_convert_{s}', root) for s in D['SOURCES']}
    D['SPLITS'] = list(D['r04']['config']['RATIOS'])
    D['split_rows'] = {sp: read_jsonl(f'{root}/split/{sp}.jsonl')
                       for sp in tqdm(D['SPLITS'], desc=f'  {desc}', leave=False)}
    D['files_of'] = {x['source']: x['n_files'] for x in D['r00']['sources']}
    return D


step('读取 00–04 的报告和划分数据（train / dev / test）')
D_MAIN = domain_data(OUT_DIR)
r00, r02, r03, r04 = D_MAIN['r00'], D_MAIN['r02'], D_MAIN['r03'], D_MAIN['r04']
SOURCES, r01, SPLITS = D_MAIN['SOURCES'], D_MAIN['r01'], D_MAIN['SPLITS']
split_rows, files_of = D_MAIN['split_rows'], D_MAIN['files_of']


def name_of(s):
    return SOURCE_META.get(s, {}).get('name', s)


def n(v):
    return f'{v:,}'


def pct(a, b):
    return f'{a / b:.1%}' if b else '—'


print(f'Sources: {SOURCES}')
print(f'Splits : {SPLITS}  ({", ".join(f"{sp}={len(split_rows[sp]):,}" for sp in SPLITS)})')

FONT = 'Times New Roman'
ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV']

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


def _word_table(caption, headers, rows, note=None, align=None, total_row=None, groups=None, doc=None, n=1):
    '''doc: 写入的 Word 文档；n: 表号（TABLE n，罗马数字）；groups: [(行号, 小标题)]，在该行之前插入分组标题行'''
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
    if total_row:
        body.append(('total', total_row))
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
            continue
        for j, v in enumerate(val):
            fill(cells[j], v, align[j], bold=kind == 'total')
            if kind == 'total':
                _line(cells[j], 'top', 6)
    for c in t.rows[-1].cells:
        _line(c, 'bottom', 12)
    if note:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        _font(p.add_run('Note: ' + note), 7)


md_parts  = ['# Tables for the paper', '',
             'Generated by step 05. Word version: `tables.docx`; CSV in `tables/`.', '']
table_log = []


def write_table(name, caption, headers, rows, align=None, note=None,
                total_row=None, tex_first_col=None, groups=None, doc=None, md=None, log=None):
    """
    写一张表：Word 三线表 + CSV + Markdown 预览。
    rows       : 每行是字符串列表
    total_row  : 合计行，前面加一条细线
    groups     : [(行号, 小标题)]，在该行之前插入分组标题行（CSV 里不写分组行）
    doc / md / log : 写入哪个 Word 文档、Markdown 和清单；默认是本领域的 tables.docx
    tex_first_col : 已不使用（原用于 LaTeX \\cite），保留参数以免改调用处
    """
    doc = _DOC if doc is None else doc
    md_out = md_parts if md is None else md
    log = table_log if log is None else log
    align = align or ('l' + 'r' * (len(headers) - 1))
    all_rows = rows + ([total_row] if total_row else [])

    # CSV：utf-8-sig 让 Excel 正确显示
    with open(f'{TAB_DIR}/{name}.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(all_rows)

    _word_table(caption, headers, rows, note=note, align=align, total_row=total_row,
                groups=groups, doc=doc, n=len(log) + 1)

    # Markdown
    group_at = dict(groups or [])
    md_align = [':---' if c == 'l' else '---:' if c == 'r' else ':---:' for c in align]
    md = [f'## {name}', '', f'*{caption}*', '',
          '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(md_align) + ' |']
    for i, r in enumerate(all_rows):
        if i in group_at:
            md.append('| **' + group_at[i] + '** |' + ' |' * (len(headers) - 1))
        md.append('| ' + ' | '.join(str(c) for c in r) + ' |')
    if note:
        md += ['', f'Note: {note}']
    md_out.extend(md + [''])
    log.append((name, caption))

# %%
# =============================================================================
step('表 1：数据来源')
# 表 1：数据来源
# =============================================================================

def sources_table(D):
    '''表 1 的行与合计行（D 为 domain_data 的返回值）'''
    S_, r01_, files_ = D['SOURCES'], D['r01'], D['files_of']
    final = D['r03']['per_source']
    total_final = sum(final.get(s, 0) for s in S_)
    rows = [[name_of(s), SOURCE_META.get(s, {}).get('type', ''),
             n(files_.get(s, 0)), n(r01_[s]['stats']['source_records']), n(r01_[s]['stats']['raw_questions']),
             n(final.get(s, 0)), pct(final.get(s, 0), total_final)] for s in S_]
    total = ['Total', '',
             n(sum(files_.get(s, 0) for s in S_)),
             n(sum(r01_[s]['stats']['source_records'] for s in S_)),
             n(sum(r01_[s]['stats']['raw_questions'] for s in S_)),
             n(total_final), '100.0%']
    return rows, total


SOURCES_HEAD = ['Dataset', 'Type', 'Files', 'Records', 'Questions', 'Final QA pairs', 'Share']
rows, total = sources_table(D_MAIN)
write_table(
    'tab_sources', 'Source datasets.', SOURCES_HEAD,
    rows, align='llrrrrr', total_row=total,
    note='Records and questions are counted after concatenating all files of a dataset; '
         'the official train/validation/test splits of Huatuo-26M are pooled and re-split. '
         'Final QA pairs are counted after cleaning, exact deduplication and '
         'near-duplicate deduplication.',
)

# %%
# =============================================================================
step('表 2：划分统计')
# 表 2：划分统计
# =============================================================================


def lengths(rows):
    """问题按不同文本、答案按不同 doc_id 统计长度，避免共享答案被重复计数"""
    q = {r['anchor'] for r in rows}
    a = {}
    for r in rows:
        a.setdefault(r['doc_id'], r['n_chars_a'])
    return [len(x) for x in q], list(a.values())


def mean_med(xs):
    return f'{statistics.mean(xs):.1f} ({statistics.median(xs):.0f})' if xs else '—'


def split_stats(D):
    '''每份数据（各 split 与 All）的规模、检索语料与长度：{列名: dict}'''
    S_, E_, SP = D['r04']['splits'], D['r04']['evaluation'], D['SPLITS']
    rows_of = {**D['split_rows'], 'All': [r for sp in SP for r in D['split_rows'][sp]]}
    out = {}
    for c in SP + ['All']:
        q_len, a_len = lengths(rows_of[c])
        keys = ('pairs', 'questions', 'answers')
        size = ({k: sum(S_[sp][k] for sp in SP) for k in keys} if c == 'All' else {k: S_[c][k] for k in keys})
        out[c] = {**size, 'q_len': q_len, 'a_len': a_len,
                  'corpus': E_[c]['corpus'] if c in E_ else None,
                  'multi': E_[c]['multi_positive'] if c in E_ else None,
                  'long': sum(x > LONG_ANSWER_CHARS for x in a_len)}
    return out


def split_label(c):
    return 'Validation' if c == 'dev' else c.capitalize()


cols = SPLITS + ['All']
ST   = split_stats(D_MAIN)
S, E = r04['splits'], r04['evaluation']
opt  = lambda v: n(v) if v is not None else '—'

rows = [
    ['QA pairs',              *[n(ST[c]['pairs']) for c in cols]],
    ['Unique questions',      *[n(ST[c]['questions']) for c in cols]],
    ['Unique answers',        *[n(ST[c]['answers']) for c in cols]],
    ['Retrieval corpus size', *[opt(ST[c]['corpus']) for c in cols]],
    ['Queries with >1 relevant answer', *[opt(ST[c]['multi']) for c in cols]],
    ['Question length, mean (median)', *[mean_med(ST[c]['q_len']) for c in cols]],
    ['Answer length, mean (median)',   *[mean_med(ST[c]['a_len']) for c in cols]],
    [f'Answers longer than {LONG_ANSWER_CHARS} chars',
     *[f"{ST[c]['long']:,} ({pct(ST[c]['long'], len(ST[c]['a_len']))})" for c in cols]],
]

ratios = r04['config']['RATIOS']
write_table(
    'tab_splits', 'Statistics of the training, validation and test sets.',
    ['', *[split_label(c) for c in cols]],
    rows,
    note=f'Splits are disjoint in questions, answers and near-duplicate answer clusters '
         f'(ratio {round(ratios["train"] * 100)}/{round(ratios["dev"] * 100)}/'
         f'{round(ratios["test"] * 100)}, seed {r04["config"]["SEED"]}, stratified by dataset). '
         f'Lengths are in Chinese characters, computed over unique questions and unique answers. '
         f'The test corpus contains all unique answers; the validation corpus contains the '
         f'validation answers plus sampled training answers and is used only for model selection.',
)

# %%
# =============================================================================
step('表 3：每个来源在三份数据中的条数')
# 表 3：每个来源在三份数据中的条数
# =============================================================================


def source_split_table(D):
    '''表 3 的行与合计行'''
    PS, S_, SP = D['r04']['per_source'], D['SOURCES'], D['SPLITS']
    rows = []
    for s in S_:
        tot = sum(PS[sp].get(s, 0) for sp in SP)
        rows.append([name_of(s), *[f'{PS[sp].get(s, 0):,} ({pct(PS[sp].get(s, 0), tot)})' for sp in SP], n(tot)])
    grand = sum(PS[sp].get(s, 0) for sp in SP for s in S_)
    total = ['Total', *[f'{sum(PS[sp].get(s, 0) for s in S_):,} '
                        f'({pct(sum(PS[sp].get(s, 0) for s in S_), grand)})' for sp in SP], n(grand)]
    return rows, total


rows, total = source_split_table(D_MAIN)
write_table(
    'tab_source_split', 'QA pairs per dataset in each split.',
    ['Dataset', *[split_label(sp) for sp in SPLITS], 'Total'],
    rows, total_row=total,
    note='Percentages are relative to the dataset (row).',
)

# %%
# =============================================================================
step('附录表：每一步剩多少 / 清洗 / 丢弃原因 / 去重参数')
# 附录表：每一步剩多少 / 清洗 / 丢弃原因 / 去重参数
# =============================================================================


def funnel_rows(D):
    '''每一步剩多少：[步骤, 各来源..., 合计]'''
    r01_, r02_, r03_, S_ = D['r01'], D['r02'], D['r03'], D['SOURCES']
    spec = [
        ('Records',                        lambda s: r01_[s]['stats']['source_records']),
        ('Question strings',               lambda s: r01_[s]['accounting']['in_input']),
        ('  Discarded',                    lambda s: r01_[s]['accounting']['dropped']),
        ('  Short answers set aside',      lambda s: r01_[s]['accounting']['ontology']),
        ('QA pairs after conversion',      lambda s: r01_[s]['accounting']['finetune']),
        ('After cross-source exact dedup', lambda s: r02_['per_source'][s]['kept']),
        ('After near-duplicate dedup',     lambda s: r03_['per_source'].get(s, 0)),
    ]
    return [[label, *[n(fn(s)) for s in S_], n(sum(fn(s) for s in S_))] for label, fn in spec]


CLEAN = [
    ('mod_html_entity',     'Decode HTML entities (e.g., &gt; to >)'),
    ('mod_greeting',        'Remove salutations and sign-offs'),
    ('mod_template_header', 'Remove website section headers'),
    ('mod_question_echo',   'Remove verbatim restatement of the question'),
]


def cleaning_rows(D):
    '''清洗规则修改的答案数：[规则, 各来源..., 合计]'''
    rows = []
    for key, desc in CLEAN:
        vals = [D['r01'][s]['stats'].get(key, 0) for s in D['SOURCES']]
        rows.append([desc, *map(n, vals), n(sum(vals))])
    return rows


def dropped_rows(D):
    '''丢弃原因：[原因, 各来源..., 合计]；原因按各来源报告中首次出现的顺序'''
    reasons, label_of = [], {}
    for s in D['SOURCES']:
        for d in D['r01'][s].get('dropped_by_reason', []):
            if d['reason'] not in label_of:
                reasons.append(d['reason'])
                label_of[d['reason']] = d['label']
    events = {s: {d['reason']: d['events'] for d in D['r01'][s].get('dropped_by_reason', [])}
              for s in D['SOURCES']}
    rows = []
    for rsn in reasons:
        vals = [events[s].get(rsn, 0) for s in D['SOURCES']]
        rows.append([label_of[rsn], *map(n, vals), n(sum(vals))])
    return rows


def dedup_rows(D):
    '''近重复检测的参数与结果：[项目, 值]'''
    c3, s3 = D['r03']['config'], D['r03']['stats']
    return [
        ['MinHash permutations',                      n(c3['NUM_PERM'])],
        ['Shingle size (characters)',                 n(c3['SHINGLE_K'])],
        ['Jaccard threshold',                         str(c3['THRESHOLD'])],
        ['Min. answer length for clustering (chars)', n(c3['MIN_SHINGLE_DOC_LEN'])],
        ['Max. cluster size collapsed',               n(c3['MAX_MERGE_SIZE'])],
        ['Unique answers before',                     n(s3['unique_answers'])],
        ['LSH candidate pairs',                       n(s3['lsh_candidate_pairs'])],
        ['Pairs verified above threshold',            f'{s3["verified_pairs_merged"]:,} '
                                                      f'({D["r03"]["verification_rate"]:.1%})'],
        ['Clusters',                                  n(s3['clusters'])],
        ['Unique answers after',                      n(s3['output_unique_answers'])],
    ]


FUNNEL_NOTE = ('Discarded and set-aside questions are subsets of the question strings. '
               'Every input question is accounted for in exactly one of: fine-tuning set, '
               'short-answer set, discarded. Questions whose answer is shorter than 30 '
               'characters after normalization are set aside and not used in any later step.')
CLEAN_NOTE = ('Rules only remove or decode formatting; clinical content is never rewritten. '
              'Every modification is logged with the original text.')
DROP_NOTE = ('Counts are events: a per-question filter counts one question, '
             'a per-group filter counts one question group, so the reasons can add up to '
             'slightly fewer than the discarded questions.')
DEDUP_NOTE = ('Each LSH candidate pair is verified against its MinHash Jaccard estimate before '
              'merging; clusters larger than the maximum size are kept for leakage control '
              'but not collapsed.')

write_table('tab_funnel', 'Number of items remaining after each processing step.',
            ['Step', *[name_of(s) for s in SOURCES], 'Total'], funnel_rows(D_MAIN), note=FUNNEL_NOTE)
write_table('tab_cleaning', 'Text-cleaning rules and the number of answers each rule modified.',
            ['Rule', *[name_of(s) for s in SOURCES], 'Total'], cleaning_rows(D_MAIN), note=CLEAN_NOTE)
write_table('tab_dropped', 'Items discarded during conversion, by reason.',
            ['Reason', *[name_of(s) for s in SOURCES], 'Total'], dropped_rows(D_MAIN), note=DROP_NOTE)
write_table('tab_dedup', 'Near-duplicate detection settings and results.',
            ['Item', 'Value'], dedup_rows(D_MAIN), align='lr', note=DEDUP_NOTE)

with open(f'{PAPER_DIR}/tables.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_parts) + '\n')
if _DOC is not None and table_log:
    _DOC.save(f'{PAPER_DIR}/tables.docx')

# %%
# =============================================================================
step('图 1：数据处理流程图')
# 图 1：数据处理流程图
# =============================================================================

try:
    import matplotlib
    matplotlib.use('Agg')                      # 服务器无显示器
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.ticker import FuncFormatter
    import numpy as np
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    print('matplotlib / numpy not installed — figures skipped. pip install matplotlib numpy')

FIG_FONT = None
if HAVE_MPL:
    _have = {f.name for f in font_manager.fontManager.ttflist}
    if FONT in _have:
        FIG_FONT = FONT
    else:
        FIG_FONT = next((n_ for n_ in ('Liberation Serif', 'TeX Gyre Termes', 'Nimbus Roman', 'DejaVu Serif')
                         if n_ in _have), 'serif')
        print(f'WARNING: "{FONT}" not found by matplotlib — figures use "{FIG_FONT}". '
              'Install Times New Roman, delete ~/.cache/matplotlib, then re-run.')
    plt.rcParams.update({
        'pdf.fonttype': 42, 'ps.fonttype': 42,   # 嵌入 TrueType 字体，IEEE 要求不能是 Type 3
        'font.family': 'serif', 'font.serif': [FIG_FONT, 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    })


def save(fig, name):
    fig.savefig(f'{FIG_DIR}/{name}.pdf')
    fig.savefig(f'{FIG_DIR}/{name}.png', dpi=300)
    plt.close(fig)


acc = lambda key: sum(r01[s]['accounting'][key] for s in SOURCES)
nums = {
    'records':          sum(r01[s]['stats']['source_records'] for s in SOURCES),
    'questions':        acc('in_input'),
    'pairs_converted':  acc('finetune'),
    'discarded':        acc('dropped'),
    'short_answers':    acc('ontology'),
    'pairs_exact':      r02['stats']['kept_pairs'],
    'pairs_neardup':    r03['stats']['output_pairs'],
    'answers_neardup':  r03['stats']['output_unique_answers'],
    **{f'{sp}_pairs': S[sp]['pairs'] for sp in SPLITS},
}
with open(f'{PAPER_DIR}/pipeline_numbers.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['item', 'value'])
    w.writerows(nums.items())

if HAVE_MPL:
    LH = 0.050   # 行高（坐标轴单位）

    def box(ax, x, y, w, h, title, body, fc):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle='round,pad=0.004,rounding_size=0.012',
                                    fc=fc, ec='#333333', lw=0.7))
        lines = [(t, 'bold') for t in title] + [(b, 'normal') for b in body]
        top = y + (len(lines) - 1) * LH / 2
        for k, (t, wt) in enumerate(lines):
            ax.text(x, top - k * LH, t, ha='center', va='center', fontsize=8, fontweight=wt)

    def arrow(ax, p, q):
        ax.annotate('', xy=q, xytext=p,
                    arrowprops=dict(arrowstyle='-|>', lw=0.7, color='#333333',
                                    shrinkA=0, shrinkB=0, mutation_scale=7))

    fig, ax = plt.subplots(figsize=(FIG_WIDTH_DOUBLE, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    MAIN_Y, MAIN_H, MAIN_W = 0.68, 0.56, 0.175
    xs = [0.095, 0.295, 0.5, 0.705, 0.905]
    main = [
        (['Source datasets'],
         [*[name_of(s) for s in SOURCES], f'{nums["records"]:,} records']),
        (['Conversion &', 'text cleaning'],
         [f'{nums["questions"]:,} questions in', f'{nums["pairs_converted"]:,} QA pairs out']),
        (['Cross-source', 'exact dedup'],
         [f'{nums["pairs_exact"]:,} QA pairs']),
        (['Near-duplicate', 'clustering'],
         ['(MinHash LSH)', f'{nums["pairs_neardup"]:,} QA pairs',
          f'{nums["answers_neardup"]:,} answers']),
        (['Leakage-free split'],
         [f'{"Validation" if sp == "dev" else sp.capitalize()} {S[sp]["pairs"]:,}'
          for sp in SPLITS]),
    ]
    for x, (title, body) in zip(xs, main):
        box(ax, x, MAIN_Y, MAIN_W, MAIN_H, title, body, '#EAF2FB')
    for x1, x2 in zip(xs, xs[1:]):
        arrow(ax, (x1 + MAIN_W / 2, MAIN_Y), (x2 - MAIN_W / 2, MAIN_Y))

    # 分支：从"转换与清洗"引出丢弃和短答案（短答案单独存放，后续步骤都不用）
    BR_Y, BR_H = 0.15, 0.26
    bottom = MAIN_Y - MAIN_H / 2
    box(ax, 0.13, BR_Y, 0.19, BR_H, ['Discarded'],
        [f'{nums["discarded"]:,} questions', 'duplicates, ads, noise'], '#F3F3F3')
    box(ax, 0.43, BR_Y, 0.23, BR_H, ['Short answers'],
        [f'{nums["short_answers"]:,} questions', 'answer < 30 characters', 'set aside, not used'],
        '#F3F3F3')
    arrow(ax, (xs[1] - 0.03, bottom), (0.13, BR_Y + BR_H / 2))
    arrow(ax, (xs[1] + 0.03, bottom), (0.43, BR_Y + BR_H / 2))

    save(fig, 'fig_pipeline')

# %%
# =============================================================================
step('图 2：问题与答案长度分布（按来源，对数横轴）')
# 图 2：问题与答案长度分布（按来源，对数横轴）
# =============================================================================

if HAVE_MPL:
    all_rows = [r for sp in SPLITS for r in split_rows[sp]]
    q_by, a_by = defaultdict(set), defaultdict(dict)
    for r in all_rows:
        q_by[r['source']].add(r['anchor'])
        a_by[r['source']].setdefault(r['doc_id'], r['n_chars_a'])
    q_len = {s: [len(x) for x in q_by[s]] for s in SOURCES}
    a_len = {s: list(a_by[s].values()) for s in SOURCES}

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_DOUBLE, 2.6))
    for ax, data, title in [(axes[0], q_len, '(a) Questions'), (axes[1], a_len, '(b) Answers')]:
        vals = [v for s in SOURCES for v in data[s]]
        bins = np.logspace(np.log10(max(1, min(vals))), np.log10(max(vals)), 45)
        for k, s in enumerate(SOURCES):
            ax.hist(data[s], bins=bins, histtype='step', lw=1.0,
                    color=COLORS[k % len(COLORS)], label=name_of(s))
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        ax.set_xlabel('Length (characters, log scale)')
        ax.set_ylabel('Count')
        ax.set_title(title, loc='left')
    # 阈值标注放进图例，不在曲线旁写字，避免压住虚线或分布峰
    axes[1].axvline(LONG_ANSWER_CHARS, ls='--', lw=0.8, color='#555555',
                    label=f'{LONG_ANSWER_CHARS} characters')
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, loc='upper right')
    fig.tight_layout()
    save(fig, 'fig_lengths')

# %%
# =============================================================================
step('示例候选：每个来源抽几条短小、干净的问答，供挑选和翻译')
# 示例候选：每个来源抽几条短小、干净的问答，供挑选和翻译
# =============================================================================
# 只从训练集里抽，避免在论文里展示测试题。

rng = random.Random(SEED)
cand_rows = []
for s in SOURCES:
    pool = [r for r in split_rows['train']
            if r['source'] == s
            and 8 <= len(r['anchor']) <= 40
            and 60 <= r['n_chars_a'] <= 200
            and '你好' not in r['positive'][:10] and '您好' not in r['positive'][:10]]
    for r in rng.sample(pool, min(N_EXAMPLES, len(pool))):
        cand_rows.append([name_of(s), r['anchor'], r['positive'], r['n_chars_a'], '', ''])

with open(f'{PAPER_DIR}/examples_candidates.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['dataset', 'question', 'answer', 'answer_chars',
                'question_en (fill in)', 'answer_en (fill in)'])
    w.writerows(cand_rows)

print('\nExample candidates (pick one per dataset and translate):')
for d, q, a, k, _, _ in cand_rows:
    print(f'  [{d}] Q: {q}')
    print(f'  {"":{len(d) + 2}s} A: {a[:80]}{"…" if len(a) > 80 else ""}')

# %%
# =============================================================================
step('跨领域合并表：肝病 + 神经科学（tables_data_domains.docx）')
# 跨领域合并表：肝病 + 神经科学（tables_data_domains.docx）
# =============================================================================
# 两个领域按分组行上下排列，各领域用与上面单领域表相同的函数计算。只读不写数据。

dom_log, dom_md = [], ['# Cross-domain data tables (liver + neurology)', '',
                       'Generated by step 05 (liver). Word version: `tables_data_domains.docx`; CSV in `tables/`.', '']
DOM_DOC = _new_doc()
DOM, DOM_SKIPPED = [], []
for dname, droot in DOMAINS:
    if droot == OUT_DIR:
        DOM.append((dname, D_MAIN))
    elif os.path.isdir(f'{droot}/reports') and os.path.isdir(f'{droot}/split'):
        try:
            DOM.append((dname, domain_data(droot, desc=f'{dname} splits')))
        except FileNotFoundError as e:
            DOM_SKIPPED.append(str(e))
    else:
        DOM_SKIPPED.append(f'{droot}/reports or {droot}/split not found')


def _dwrite(name_, caption, headers, rows, **kw):
    write_table(name_, caption, headers, rows, doc=DOM_DOC, md=dom_md, log=dom_log, **kw)


def _stack(build):
    '''把各领域的行上下拼接：build(D) 返回行列表，或 (行, 合计行)；返回 (rows, groups)'''
    rows, groups = [], []
    for dname, D in DOM:
        out = build(D)
        body, total = out if isinstance(out, tuple) else (out, None)
        groups.append((len(rows), dname))
        rows += body + ([total] if total else [])
    return rows, groups


if len(DOM) >= 2:
    same_sources = all(D['SOURCES'] == SOURCES for _, D in DOM)
    if not same_sources:
        DOM_SKIPPED.append('per-source cross-domain tables (the domains use different sources)')
    src_heads = [name_of(s) for s in SOURCES]

    # ---- 数据来源（正文）----
    rows, groups = _stack(sources_table)
    _dwrite('tab_sources_domains', 'Source datasets of the two disease domains.', SOURCES_HEAD, rows,
            align='llrrrrr', groups=groups,
            note='Records are counted in the domain subsets after keyword and language-model filtering, and '
                 'records and questions after concatenating all files of a dataset; the official '
                 'train/validation/test splits of Huatuo-26M are pooled and re-split. A record can yield several '
                 'question strings because some records contain several phrasings of the question. Final QA '
                 'pairs are counted after cleaning, exact deduplication and near-duplicate deduplication; '
                 '"Share" is the share of each source within its domain.')

    # ---- 划分统计（正文）----
    ST_OF = {id(D): (ST if D is D_MAIN else split_stats(D)) for _, D in DOM}

    def _split_rows(D):
        st = ST_OF[id(D)]
        return [[split_label(c), n(st[c]['pairs']), n(st[c]['questions']), n(st[c]['answers']),
                 opt(st[c]['corpus']), mean_med(st[c]['q_len']), mean_med(st[c]['a_len']),
                 f"{st[c]['long']:,} ({pct(st[c]['long'], len(st[c]['a_len']))})"]
                for c in D['SPLITS'] + ['All']]

    rows, groups = _stack(_split_rows)
    multi = '; '.join(f"{d.lower()}: {ST_OF[id(D)]['test']['multi']:,}" for d, D in DOM
                      if 'test' in ST_OF[id(D)] and ST_OF[id(D)]['test']['multi'] is not None)
    _dwrite('tab_splits_domains', 'Statistics of the training, validation and test sets in both disease domains.',
            ['Split', 'QA pairs', 'Questions', 'Answers', 'Retrieval corpus', 'Question length',
             'Answer length', f'Answers > {LONG_ANSWER_CHARS} chars'], rows, align='lrrrrrrr', groups=groups,
            note=f'Questions and answers are counted as unique texts. Lengths are in Chinese characters, given '
                 f'as mean (median). Splits are disjoint in questions, answers and near-duplicate answer clusters '
                 f'(ratio {round(ratios["train"] * 100)}/{round(ratios["dev"] * 100)}/{round(ratios["test"] * 100)}, '
                 f'stratified by dataset). The training answers form the candidate pool for hard-negative mining. '
                 f'The test corpus contains all unique answers of the domain; the validation corpus contains the '
                 f'validation answers plus sampled training answers and is used only for model selection.'
                 + (f' Test questions with more than one relevant answer: {multi}.' if multi else ''))

    if same_sources:
        # ---- 各来源在三份数据中的条数（附录）----
        rows, groups = _stack(source_split_table)
        _dwrite('tab_source_split_domains', 'QA pairs per dataset in each split, in both disease domains.',
                ['Dataset', *[split_label(sp) for sp in SPLITS], 'Total'], rows, groups=groups,
                note='Percentages are relative to the dataset (row).')

        # ---- 每一步剩多少 / 清洗 / 丢弃原因（附录）----
        rows, groups = _stack(funnel_rows)
        _dwrite('tab_funnel_domains', 'Number of items remaining after each processing step, in both disease domains.',
                ['Step', *src_heads, 'Total'], rows, groups=groups, note=FUNNEL_NOTE)
        rows, groups = _stack(cleaning_rows)
        _dwrite('tab_cleaning_domains', 'Text-cleaning rules and the number of answers each rule modified, in '
                'both disease domains.', ['Rule', *src_heads, 'Total'], rows, groups=groups, note=CLEAN_NOTE)
        rows, groups = _stack(dropped_rows)
        _dwrite('tab_dropped_domains', 'Items discarded during conversion, by reason, in both disease domains.',
                ['Reason', *src_heads, 'Total'], rows, groups=groups, note=DROP_NOTE)

    # ---- 近重复检测（附录；两个领域并排）----
    dd = [dedup_rows(D) for _, D in DOM]
    rows = [[dd[0][i][0], *[x[i][1] for x in dd]] for i in range(len(dd[0]))]
    _dwrite('tab_dedup_domains', 'Near-duplicate detection settings and results in both disease domains.',
            ['Item', *[d for d, _ in DOM]], rows, align='l' + 'r' * len(DOM), note=DEDUP_NOTE)

    with open(f'{PAPER_DIR}/tables_data_domains.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(dom_md) + '\n')
    if DOM_DOC is not None and dom_log:
        DOM_DOC.save(f'{PAPER_DIR}/tables_data_domains.docx')
else:
    DOM_SKIPPED.append('cross-domain tables (need both domain directories: '
                       + ', '.join(r for _, r in DOMAINS) + ')')

# %%
# =============================================================================
# 清单
# =============================================================================

print('\n' + '=' * 70)
print(f'Paper assets written to {PAPER_DIR}/')
print('=' * 70)
print('\nTables ({}; CSV in tables/; preview in tables.md):'.format(
    'Word: tables.docx' if _DOC is not None else 'Word skipped — install python-docx'))
for name, cap in table_log:
    print(f'  {name:<18s} {cap}')
if HAVE_MPL:
    print('\nFigures (figures/*.pdf for the paper, *.png for preview; font: {}):'.format(FIG_FONT))
    print('  fig_pipeline       Data-processing pipeline with counts')
    print('  fig_lengths        Question and answer length distributions')
print('\nOther:')
print('  examples_candidates.csv   pick one example per dataset and translate')
print('  pipeline_numbers.csv      numbers in the flowchart, for redrawing in draw.io / PowerPoint')
if dom_log:
    print('\nCross-domain tables (Word: tables_data_domains.docx; preview in tables_data_domains.md):')
    for name, cap in dom_log:
        print(f'  {name:<26s} {cap}')
if DOM_SKIPPED:
    print('\nSkipped:')
    for p in DOM_SKIPPED:
        print('  ' + p)

print('\nDone in {:.1f} min.'.format((time.time() - _T0) / 60))
