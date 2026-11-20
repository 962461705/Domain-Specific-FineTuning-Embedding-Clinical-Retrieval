# %%
# =============================================================================
# 配置
# =============================================================================

OUT_DIR    = '/data/neuro/data/neuro_dataset'     # 和 00–04 一致
REPORT_DIR = f'{OUT_DIR}/reports'
SPLIT_DIR  = f'{OUT_DIR}/split'
PAPER_DIR  = f'{OUT_DIR}/paper'
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


def load_report(name):
    path = f'{REPORT_DIR}/{name}.json'
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found — run the corresponding step first.')
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


step('读取 00–04 的报告')
r00 = load_report('00_concat')
r02 = load_report('02_merge')
r03 = load_report('03_dedup')
r04 = load_report('04_split')
SOURCES = list(r02['config']['SOURCES'])
r01 = {s: load_report(f'01_convert_{s}') for s in SOURCES}
SPLITS = list(r04['config']['RATIOS'])

step('读取划分数据（train / dev / test）')
split_rows = {sp: read_jsonl(f'{SPLIT_DIR}/{sp}.jsonl') for sp in tqdm(SPLITS, desc='  splits', leave=False)}
files_of   = {x['source']: x['n_files'] for x in r00['sources']}


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
    _DOC = Document()
    _sec = _DOC.sections[0]
    _sec.orientation = WD_ORIENT.LANDSCAPE
    _sec.page_width, _sec.page_height = _sec.page_height, _sec.page_width
    _sec.left_margin = _sec.right_margin = Inches(0.7)
    _DOC.styles['Normal'].font.name = FONT
    _DOC.styles['Normal'].font.size = Pt(8)
except ImportError:
    _DOC = None
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


def _word_table(caption, headers, rows, note=None, align=None, total_row=None):
    if _DOC is None:
        return
    n = len(table_log) + 1
    num = ROMAN[n - 1] if n <= len(ROMAN) else str(n)
    A = {'l': WD_ALIGN_PARAGRAPH.LEFT, 'r': WD_ALIGN_PARAGRAPH.RIGHT, 'c': WD_ALIGN_PARAGRAPH.CENTER}
    ncol = len(headers)
    align = (align or ('l' + 'r' * (ncol - 1))).ljust(ncol, 'l')

    for txt, caps, before in ((f'TABLE {num}', False, 12), (caption.rstrip('.'), True, 0)):
        p = _DOC.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(before), Pt(2)
        _font(p.add_run(txt), 8, caps=caps)

    body = list(rows) + ([total_row] if total_row else [])
    t = _DOC.add_table(rows=1 + len(body), cols=ncol)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    def fill(cell, text, a, bold=False):
        cell.text = ''
        para = cell.paragraphs[0]
        para.alignment = A.get(a, WD_ALIGN_PARAGRAPH.LEFT)
        para.paragraph_format.space_after = Pt(0)
        _font(para.add_run(str(text)), 8, bold=bold)

    for j, h in enumerate(headers):
        c = t.rows[0].cells[j]
        fill(c, h, align[j], bold=True)
        _line(c, 'top', 12)
        _line(c, 'bottom', 6)
    for i, r in enumerate(body, 1):
        is_total = total_row is not None and i == len(body)
        for j, v in enumerate(r):
            fill(t.rows[i].cells[j], v, align[j], bold=is_total)
            if is_total:
                _line(t.rows[i].cells[j], 'top', 6)
    for c in t.rows[-1].cells:
        _line(c, 'bottom', 12)
    if note:
        p = _DOC.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        _font(p.add_run('Note: ' + note), 7)


md_parts  = ['# Tables for the paper', '',
             'Generated by step 05. Word version: `tables.docx`; CSV in `tables/`.', '']
table_log = []


def write_table(name, caption, headers, rows, align=None, note=None,
                total_row=None, tex_first_col=None):
    """
    写一张表：Word 三线表 + CSV + Markdown 预览。
    rows       : 每行是字符串列表
    total_row  : 合计行，前面加一条细线
    tex_first_col : 已不使用（原用于 LaTeX \\cite），保留参数以免改调用处
    """
    align = align or ('l' + 'r' * (len(headers) - 1))
    all_rows = rows + ([total_row] if total_row else [])

    # CSV：utf-8-sig 让 Excel 正确显示
    with open(f'{TAB_DIR}/{name}.csv', 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(all_rows)

    _word_table(caption, headers, rows, note=note, align=align, total_row=total_row)

    # Markdown
    md_align = [':---' if c == 'l' else '---:' if c == 'r' else ':---:' for c in align]
    md = [f'## {name}', '', f'*{caption}*', '',
          '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(md_align) + ' |']
    md += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in all_rows]
    if note:
        md += ['', f'Note: {note}']
    md_parts.extend(md + [''])
    table_log.append((name, caption))

# %%
# =============================================================================
step('表 1：数据来源')
# 表 1：数据来源
# =============================================================================

final_by_src = r03['per_source']
total_final  = sum(final_by_src.get(s, 0) for s in SOURCES)

rows = []
for s in SOURCES:
    st = r01[s]['stats']
    rows.append([name_of(s), SOURCE_META.get(s, {}).get('type', ''),
                 n(files_of.get(s, 0)), n(st['source_records']), n(st['raw_questions']),
                 n(final_by_src.get(s, 0)), pct(final_by_src.get(s, 0), total_final)])

write_table(
    'tab_sources', 'Source datasets.',
    ['Dataset', 'Type', 'Files', 'Records', 'Questions', 'Final QA pairs', 'Share'],
    rows, align='llrrrrr',
    total_row=['Total', '',
               n(sum(files_of.get(s, 0) for s in SOURCES)),
               n(sum(r01[s]['stats']['source_records'] for s in SOURCES)),
               n(sum(r01[s]['stats']['raw_questions'] for s in SOURCES)),
               n(total_final), '100.0%'],
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


cols   = SPLITS + ['All']
rows_of = {**split_rows, 'All': [r for sp in SPLITS for r in split_rows[sp]]}
L      = {c: lengths(rows_of[c]) for c in cols}
S, E   = r04['splits'], r04['evaluation']


def per_split(fn, all_value):
    return [fn(sp) for sp in SPLITS] + [all_value]


long_cells = []
for c in cols:
    a = L[c][1]
    k = sum(x > LONG_ANSWER_CHARS for x in a)
    long_cells.append(f'{k:,} ({pct(k, len(a))})')

rows = [
    ['QA pairs',          *per_split(lambda sp: n(S[sp]['pairs']),
                                     n(sum(S[sp]['pairs'] for sp in SPLITS)))],
    ['Unique questions',  *per_split(lambda sp: n(S[sp]['questions']),
                                     n(sum(S[sp]['questions'] for sp in SPLITS)))],
    ['Unique answers',    *per_split(lambda sp: n(S[sp]['answers']),
                                     n(sum(S[sp]['answers'] for sp in SPLITS)))],
    ['Retrieval corpus size', *per_split(lambda sp: n(E[sp]['corpus']) if sp in E else '—', '—')],
    ['Queries with >1 relevant answer',
                          *per_split(lambda sp: n(E[sp]['multi_positive']) if sp in E else '—', '—')],
    ['Question length, mean (median)', *[mean_med(L[c][0]) for c in cols]],
    ['Answer length, mean (median)',   *[mean_med(L[c][1]) for c in cols]],
    [f'Answers longer than {LONG_ANSWER_CHARS} chars', *long_cells],
]

ratios = r04['config']['RATIOS']
write_table(
    'tab_splits', 'Statistics of the training, validation and test sets.',
    ['', *[c.capitalize() if c != 'dev' else 'Validation' for c in cols]],
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

PS = r04['per_source']
rows = []
for s in SOURCES:
    tot = sum(PS[sp].get(s, 0) for sp in SPLITS)
    rows.append([name_of(s),
                 *[f'{PS[sp].get(s, 0):,} ({pct(PS[sp].get(s, 0), tot)})' for sp in SPLITS],
                 n(tot)])
grand = sum(PS[sp].get(s, 0) for sp in SPLITS for s in SOURCES)
write_table(
    'tab_source_split', 'QA pairs per dataset in each split.',
    ['Dataset', 'Train', 'Validation', 'Test', 'Total'],
    rows,
    total_row=['Total', *[f'{sum(PS[sp].get(s, 0) for s in SOURCES):,} '
                          f'({pct(sum(PS[sp].get(s, 0) for s in SOURCES), grand)})'
                          for sp in SPLITS], n(grand)],
    note='Percentages are relative to the dataset (row).',
)

# %%
# =============================================================================
step('附录表：每一步剩多少 / 清洗 / 丢弃原因 / 去重参数')
# 附录表：每一步剩多少 / 清洗 / 丢弃原因 / 去重参数
# =============================================================================

# ---- 每一步剩多少 ----
FUNNEL = [
    ('Records',                              lambda s: r01[s]['stats']['source_records']),
    ('Question strings',                     lambda s: r01[s]['accounting']['in_input']),
    ('  Discarded',                          lambda s: r01[s]['accounting']['dropped']),
    ('  Short answers set aside',            lambda s: r01[s]['accounting']['ontology']),
    ('QA pairs after conversion',            lambda s: r01[s]['accounting']['finetune']),
    ('After cross-source exact dedup',       lambda s: r02['per_source'][s]['kept']),
    ('After near-duplicate dedup',           lambda s: r03['per_source'].get(s, 0)),
]
write_table(
    'tab_funnel', 'Number of items remaining after each processing step.',
    ['Step', *[name_of(s) for s in SOURCES], 'Total'],
    [[label, *[n(fn(s)) for s in SOURCES], n(sum(fn(s) for s in SOURCES))]
     for label, fn in FUNNEL],
    note='Discarded and set-aside questions are subsets of the question strings. '
         'Every input question is accounted for in exactly one of: fine-tuning set, '
         'short-answer set, discarded. Questions whose answer is shorter than 30 '
         'characters after normalization are set aside and not used in any later step.',
)

# ---- 清洗 ----
CLEAN = [
    ('mod_html_entity',     'Decode HTML entities (e.g., &gt; to >)'),
    ('mod_greeting',        'Remove salutations and sign-offs'),
    ('mod_template_header', 'Remove website section headers'),
    ('mod_question_echo',   'Remove verbatim restatement of the question'),
]
rows = []
for key, desc in CLEAN:
    vals = [r01[s]['stats'].get(key, 0) for s in SOURCES]
    rows.append([desc, *map(n, vals), n(sum(vals))])
write_table(
    'tab_cleaning', 'Text-cleaning rules and the number of answers each rule modified.',
    ['Rule', *[name_of(s) for s in SOURCES], 'Total'], rows,
    note='Rules only remove or decode formatting; clinical content is never rewritten. '
         'Every modification is logged with the original text.',
)

# ---- 丢弃原因 ----
reasons, label_of = [], {}
for s in SOURCES:
    for d in r01[s].get('dropped_by_reason', []):
        if d['reason'] not in label_of:
            reasons.append(d['reason'])
            label_of[d['reason']] = d['label']
events = {s: {d['reason']: d['events'] for d in r01[s].get('dropped_by_reason', [])}
          for s in SOURCES}
rows = []
for rsn in reasons:
    vals = [events[s].get(rsn, 0) for s in SOURCES]
    rows.append([label_of[rsn], *map(n, vals), n(sum(vals))])
write_table(
    'tab_dropped', 'Items discarded during conversion, by reason.',
    ['Reason', *[name_of(s) for s in SOURCES], 'Total'], rows,
    note='Counts are events: a per-question filter counts one question, '
         'a per-group filter counts one question group.',
)

# ---- 去重参数与统计 ----
c3, s3 = r03['config'], r03['stats']
write_table(
    'tab_dedup', 'Near-duplicate detection settings and results.',
    ['Item', 'Value'],
    [
        ['MinHash permutations',                         n(c3['NUM_PERM'])],
        ['Shingle size (characters)',                    n(c3['SHINGLE_K'])],
        ['Jaccard threshold',                            str(c3['THRESHOLD'])],
        ['Min. answer length for clustering (chars)',    n(c3['MIN_SHINGLE_DOC_LEN'])],
        ['Max. cluster size collapsed',                  n(c3['MAX_MERGE_SIZE'])],
        ['Unique answers before',                        n(s3['unique_answers'])],
        ['LSH candidate pairs',                          n(s3['lsh_candidate_pairs'])],
        ['Pairs verified above threshold',               f'{s3["verified_pairs_merged"]:,} '
                                                         f'({r03["verification_rate"]:.1%})'],
        ['Clusters',                                     n(s3['clusters'])],
        ['Unique answers after',                         n(s3['output_unique_answers'])],
    ],
    align='lr',
    note='Each LSH candidate pair is verified against its MinHash Jaccard estimate before '
         'merging; clusters larger than the maximum size are kept for leakage control '
         'but not collapsed.',
)

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

print('\nDone in {:.1f} min.'.format((time.time() - _T0) / 60))
