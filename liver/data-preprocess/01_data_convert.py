
SOURCE_NAME = 'medchatzh'

OUT_DIR = '/home/liver/data/liver_dataset'     # 00/01/02/03 四个脚本必须一致

# 以下路径全部自动派生，不要手改
INPUT_PATH    = f'{OUT_DIR}/raw/{SOURCE_NAME}_all.jsonl'     # 00 的输出
OUTPUT_PATH   = f'{OUT_DIR}/converted/{SOURCE_NAME}.jsonl'
ONTO_PATH     = f'{OUT_DIR}/ontology_src/{SOURCE_NAME}.jsonl'
DROPPED_PATH  = f'{OUT_DIR}/dropped/{SOURCE_NAME}.jsonl'
MODIFIED_PATH = f'{OUT_DIR}/modified/{SOURCE_NAME}.jsonl'
REPORT_DIR    = f'{OUT_DIR}/reports'

MIN_QUERY_LEN     = 5     # 归一化后的字符数
ONTOLOGY_MAX_LEN  = 30    # 归一化后短于此值 → 不进微调，进本体池
ATOMIC_MAX_LEN    = 12    # 本体池内再标一层：像"实体值"的极短答案（"超声检查"）
LONG_ANSWER_CHARS = 480   # 超过此长度 → is_long=True。【只打标记，不丢弃】03 里的同名参数必须一致
                          # 480 是 512 token 上限的粗略字数代理：
                          # bge-*-zh-v1.5 的 max_seq_length=512，中文约 1 字 1 token，
                          # 留 32 token 给 [CLS]/[SEP]、标点、数字和英文。

# ---- 文本清洗（只删除或还原格式性文字，不动医学内容）----
UNESCAPE_HTML       = True   # 还原 HTML 转义：&gt; → >，&amp; → &，&nbsp; → 空格
STRIP_GREETING      = True   # 剥掉答案首尾的寒暄套话（含寒暄后面的冒号）
STRIP_TEMPLATE      = True   # 去掉"问题分析：""意见建议："这类模板标题
STRIP_QUESTION_ECHO = True   # 去掉答案开头对问题的逐字复述

# ---- 过滤 ----
DROP_PREFIX_DUP = True    # 丢弃"截断标题"型问法（是另一问法的前缀）
DROP_JUNK       = True    # 丢弃含联系方式 / 纯导流 / 无信息的答案

# %%
import datetime, hashlib, html, json, os, random, re, sys, unicodedata
from collections import Counter, defaultdict
from tqdm.auto import tqdm

if not os.path.isfile(INPUT_PATH):
    raise FileNotFoundError(
        f'{INPUT_PATH} not found. Run step 00 first, and check that SOURCE_NAME '
        f'is one of the keys of SOURCES in step 00.'
    )

for p in (OUTPUT_PATH, ONTO_PATH, DROPPED_PATH, MODIFIED_PATH):
    os.makedirs(os.path.dirname(p), exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# %%
# =============================================================================
# 清洗与过滤规则
# =============================================================================

# 含这些就整条丢：导流、联系方式，没有医学信息量
JUNK_RE = re.compile(
    r'https?://|www\.'
    r'|1[3-9]\d{9}'                        # 手机号
    r'|\d{3,4}-\d{7,8}'                    # 座机
    # 微信号/QQ 号只由字母数字组成。不能写 \w：Python 3 的 \w 也匹配汉字，
    # "我没有微信或者微博" 这种正常句子会被当成联系方式。
    r'|(微信|qq|weixin)\s*[:：]?\s*[A-Za-z0-9_-]{5,}'
    r'|(加|添加|拨打).{0,4}(微信|好友|热线)',
    re.I,
)


GREETING_HEAD_RE = re.compile(
    r'^[\s\ufeff\u200b\u200c\u200d\u2060]*'
    r'(?:(?:这位|那位|亲爱的|尊敬的)?'
    r'(?:患者朋友|朋友|患者|网友|病友|家长|先生|女士|用户)(?:您|你)?[,，、\s]*)?'
    r'(您好|你好|hi(?![a-z])|哈喽)'
    r'[!！,，。.:：~～\s]*',
    re.I,
)
GREETING_TAIL_RE = re.compile(
    r'[\s,，。]*(祝(您|你).{0,6}(健康|康复|好运)|希望(我的)?回答(对(您|你))?有(所)?帮助'
    r'|如有疑问.{0,10}|欢迎(继续)?(咨询|提问))[。!！~\s]*$'
)

# 问诊网站的段落模板标题。只删标题本身，后面的内容原样保留。
# 只收明确是模板的多字标题；"建议：" 这种短词可能是正文的一部分，不收。
# 标题必须出现在开头或句末标点之后，避免误删正文中间恰好出现的同样字样。
TEMPLATE_HEADER_RE = re.compile(
    r'(?:^|(?<=[。！？!?；;.\s]))'
    r'(问题分析|病情分析|意见建议|指导意见|医生建议|治疗建议|处理意见|温馨提示)'
    r'\s*[:：]\s*'
)

# 只有"去医院看看"这类话、没有实质内容的回答
NON_ANSWER_RE = re.compile(
    r'^(建议)?(您|你)?(最好)?(及时|尽快)?(到|去)?(正规)?(医院|门诊|我院|专科)'
    r'.{0,12}(检查|就诊|看看|面诊|诊治)[。!！\s]*$'
    r'|^具体(情况|问题).{0,10}(面诊|就诊|医生)'
    r'|^(不好说|说不准|因人而异)[。!！\s]*$'
)

ECHO_MIN_LEN = 8              # 问题归一化后至少这么长才检查复述，太短容易误伤
SENT_END     = '.。?？!！'

# 清洗规则：(changes 里的键, 报告里的名字)。质检二和汇总报告共用这一份。
CLEANING_RULES = [
    ('html_entity',     'HTML entities unescaped'),
    ('greeting',        'Greetings stripped'),
    ('template_header', 'Template headers removed'),
    ('question_echo',   'Question echo removed'),
]


def disp_width(s):
    """终端显示宽度。CJK 字符占两列，而 f-string 的 :<N 是按字符数补齐的，
    中英混排的表格会歪。"""
    return sum(2 if unicodedata.east_asian_width(c) in 'WF' else 1 for c in s)


def pad(s, width):
    """按显示宽度左对齐补齐"""
    return s + ' ' * max(0, width - disp_width(s))


def normalize(text):
    """去重专用归一化：全角转半角、小写、只保留中英文数字"""
    text = unicodedata.normalize('NFKC', text or '').lower()
    return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text)


def strip_greeting(t):
    """剥掉首尾寒暄。返回 (文本, 是否改动)"""
    stripped = GREETING_TAIL_RE.sub('', GREETING_HEAD_RE.sub('', t)).strip()
    return stripped, stripped != t


def clean_answer(text):
    """
    还原 HTML 转义、剥掉首尾寒暄和模板标题，压掉多余空白。
    返回 (清洗后文本, 做过的修改列表)。每一步都只删除或还原格式性文字。

    HTML 转义放在最前面：寒暄和模板标题的规则匹配的是正常字符，
    转义没还原的话，"您好&#65306;" 这种写法就匹配不上。

    寒暄和模板标题可能互相挡住："问题分析：您好，…" 要先删标题才露出寒暄，
    "您好，问题分析：…" 要先删寒暄才露出标题。所以两条规则反复做，直到不再变化。
    """
    t = (text or '').strip()
    changes = []

    def mark(c):
        if c not in changes:
            changes.append(c)

    if UNESCAPE_HTML:
        unescaped = html.unescape(t)
        if unescaped != t:
            mark('html_entity')
            t = unescaped.strip()

    for _ in range(3):
        changed = False
        if STRIP_GREETING:
            t, g = strip_greeting(t)
            if g:
                mark('greeting')
                changed = True
        if STRIP_TEMPLATE:
            t, n = TEMPLATE_HEADER_RE.subn('', t)
            t = t.strip()
            if n:
                mark('template_header')
                changed = True
        if not changed:
            break

    # \s 在 Python 3 里也匹配 &nbsp; 还原出来的 \xa0，这里一并压成普通空格
    return re.sub(r'\s+', ' ', t).strip(), changes


def strip_question_echo(answer, questions):
    """
    答案开头逐字复述问题时，去掉复述部分。返回 (文本, 是否去掉了)。

    只有复述后面紧跟句末标点才去掉——那说明是把问题整句抄了一遍。
    "乙肝疫苗多久打一次要看年龄" 是答案顺着话题往下说，去掉就只剩半句话，不能动。

    这一步针对的是检索训练里的字面捷径：答案里原样包含问题时，模型和 BM25
    靠字面重合就能配对，不需要学语义。
    """
    for q in sorted(questions, key=len, reverse=True):
        nq = normalize(q)
        if len(nq) < ECHO_MIN_LEN:
            continue
        i = j = 0          # i：答案原文的位置；j：已匹配的问题归一化字符数
        while i < len(answer) and j < len(nq):
            c = normalize(answer[i])
            if c:
                if not nq.startswith(c, j):
                    break
                j += len(c)
            i += 1
        if j < len(nq):
            continue
        rest = answer[i:].lstrip()
        if not rest.strip(SENT_END + ' '):
            return '', True               # 答案只是把问题抄了一遍，后面会作为无内容丢弃
        if rest[0] in SENT_END:
            return rest.lstrip(SENT_END + ' '), True
    return answer, False


def flatten_group(g):
    """单个问题组拍平成 list[str]"""
    if isinstance(g, list):
        return [x.strip() for x in g if isinstance(x, str) and x.strip()]
    if isinstance(g, str) and g.strip():
        return [g.strip()]
    return []


def pair_qa_groups(item):
    """
    Pair question groups with answers.

    Returns `(pairs, unpaired_groups, unpaired_answers)`:
    - `pairs`            list of `(group_index, [questions], answer)`
    - `unpaired_groups`  list of `(group_index, [questions])` with no answer
    - `unpaired_answers` list of answers with no question group

    Nothing is discarded silently: whatever cannot be paired is returned so the
    caller can log it.

    """
    groups  = [g for g in (flatten_group(q) for q in (item.get('questions') or [])) if g]
    answers = [a for a in (item.get('answers') or []) if isinstance(a, str) and a.strip()]

    if len(answers) == 1 and groups:             # 多组问法共享一个答案（含一组一答）
        return [(i, g, answers[0]) for i, g in enumerate(groups)], [], []

    n = min(len(groups), len(answers))           # 一一对应；对不齐时只配前 n 个
    return ([(i, groups[i], answers[i]) for i in range(n)],
            [(i, groups[i]) for i in range(n, len(groups))],
            answers[n:])


def dedup_questions(qs):
    """
    Deduplicate the questions within one group.

    Returns `(kept, dropped)`, where `dropped` is a list of
    `(question, reason, kept_as)`. Every input question appears in exactly one
    of the two lists.

    去掉组内的无效问法：
    - 归一化后为空（只有标点之类）
    - 归一化后与组内另一条完全相同（比如只差一个全角问号）
    - 截断型问法：是另一更长条的前缀。CMDD 这类数据常见 "标题(被截断) + 完整提问"
      的结构，它不是同义改写，留着只会制造冗余和 in-batch 假负例。

    顺序是确定性的：先按输入顺序去重，再按长度稳定排序。
    """
    first_by_norm, uniq, dropped = {}, [], []
    for q in qs:
        nq = normalize(q)
        if not nq:
            dropped.append((q, 'drop_empty_question', None))
        elif nq in first_by_norm:
            dropped.append((q, 'drop_identical_question', first_by_norm[nq]))
        else:
            first_by_norm[nq] = q
            uniq.append(q)

    kept = []
    for q in sorted(uniq, key=len, reverse=True):
        nq = normalize(q)
        longer = None
        if DROP_PREFIX_DUP:
            longer = next((k for k in kept if normalize(k).startswith(nq)), None)
        if longer is not None:
            dropped.append((q, 'drop_prefix_question', longer))
        else:
            kept.append(q)
    return kept, dropped

# %%
# =============================================================================
# 转换
# =============================================================================

records      = []     # 进微调
onto_recs    = []     # 进本体池
dropped      = []     # 被丢弃的，连同原因
modified     = []     # 被清洗过的保留答案，连同原文
doc_ids      = {}     # 归一化答案 -> doc_id
seen_pairs   = set()  # (归一化query, doc_id)
stats        = Counter()
samples      = defaultdict(list)   # reason -> 打印用样本
long_samples = []

SAMPLE_CAP = {'skip_short_query': 15}   # 其余原因默认 5 条


def drop(reason, src, questions=(), answer=None, **extra):
    """
    记录一次丢弃：计数 + 写入 dropped + 留样本给质检一打印。

    stats[reason] 计的是"事件数"；对账用的是 dropped 里每条记录的
    questions 长度，口径是"问题条数"。
    """
    questions = list(questions)
    stats[reason] += 1
    rec = {'reason': reason, 'source': SOURCE_NAME, 'src': src,
           'questions': questions, 'answer': answer}
    rec.update({k: v for k, v in extra.items() if v is not None})
    dropped.append(rec)

    if len(samples[reason]) < SAMPLE_CAP.get(reason, 5):
        q = questions[0][:50] if questions else ''
        a = str(answer if answer is not None else extra.get('raw', ''))[:70]
        line = f'Q: {q!r}  |  A: {a}'
        if extra.get('kept_as'):
            line += f'\n      kept instead: {extra["kept_as"][:50]!r}'
        samples[reason].append(line)


def log_changes(changes, src, group_id, before, after, kept_in):
    """
    记录一次文本清洗。只记最终保留下来的答案（进了微调集或本体池的），
    被丢弃的答案改没改都无关紧要。计数单位是"答案"（问答组）。
    """
    if not changes:
        return
    for c in changes:
        stats[f'mod_{c}'] += 1
    modified.append({'source': SOURCE_NAME, 'src': src, 'group_id': group_id,
                     'kept_in': kept_in, 'changes': list(changes),
                     'before': before, 'after': after})


with open(INPUT_PATH, encoding='utf-8') as f:
    for line_no, line in enumerate(tqdm(f, desc=f'Converting {SOURCE_NAME}')):
        if not line.strip():
            continue
        src = f'{SOURCE_NAME}:{line_no}'

        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            drop('skip_json_error', src, raw=line.rstrip('\n'))
            continue

        stats['source_records'] += 1
        pairs, unpaired_groups, unpaired_answers = pair_qa_groups(item)
        stats['raw_questions'] += (sum(len(qs) for _, qs, _ in pairs)
                                   + sum(len(qs) for _, qs in unpaired_groups))

        if not (pairs or unpaired_groups or unpaired_answers):
            drop('skip_missing_field', src, raw=item)
            continue
        for gi, qs in unpaired_groups:
            drop('drop_unpaired_group', src, qs, group_id=f'{src}:{gi}')
        for a in unpaired_answers:
            drop('drop_unpaired_answer', src, answer=a)

        for gi, qs_raw, answer_raw in pairs:
            stats['qa_groups'] += 1
            group_id = f'{src}:{gi}'

            # ---------- 文本清洗：只删除或还原格式性文字 ----------
            answer, changes = clean_answer(answer_raw)
            if STRIP_QUESTION_ECHO:
                answer, echoed = strip_question_echo(answer, qs_raw)
                if echoed:
                    changes.append('question_echo')
                    # 复述去掉后，紧跟在后面的寒暄（"问题原文。你好，…"）会露到开头，再清一次
                    if STRIP_GREETING:
                        answer, g = strip_greeting(answer)
                        if g and 'greeting' not in changes:
                            changes.append('greeting')
                            
            na = normalize(answer)
            if not na:
                reason = 'drop_echo_only' if 'question_echo' in changes else 'skip_empty_answer'
                drop(reason, src, qs_raw, answer_raw, group_id=group_id)
                continue

            if DROP_JUNK and JUNK_RE.search(answer):
                drop('drop_junk_contact', src, qs_raw, answer_raw, group_id=group_id)
                continue

            qs, q_dropped = dedup_questions(qs_raw)
            for q, reason, kept_as in q_dropped:
                drop(reason, src, [q], answer_raw, group_id=group_id, kept_as=kept_as)
            if not qs:
                # 问题已经在上面逐条记过了，这里只记这个答案，questions 留空以免对账重复计数
                drop('skip_no_valid_question', src, [], answer_raw, group_id=group_id)
                continue

            # ---------- 短答案 → 本体池，不进微调 ----------
            if len(na) < ONTOLOGY_MAX_LEN:
                informative = not NON_ANSWER_RE.search(answer)
                for q in qs:
                    onto_recs.append({
                        'source':      SOURCE_NAME,
                        'src':         src,
                        'group_id':    group_id,
                        'question':    q,
                        'answer':      answer,
                        'n_chars_a':   len(answer),
                        'is_atomic':   len(na) < ATOMIC_MAX_LEN,  # 像实体值，可直接入三元组
                        'informative': informative,               # False 的多半是"去医院看看"
                    })
                stats['route_ontology'] += len(qs)
                log_changes(changes, src, group_id, answer_raw, answer, 'ontology')
                continue

            # ---------- 无信息的长答案，丢 ----------
            if DROP_JUNK and NON_ANSWER_RE.search(answer):
                drop('drop_non_answer', src, qs, answer_raw, group_id=group_id)
                continue

            # ---------- 进微调 ----------
            # doc_id 用内容哈希：同一段答案不论出现在哪个来源都得到同一个 id，
            # 所以多个来源合并时不会冲突，也不需要重新编号。
            if na not in doc_ids:
                doc_ids[na] = hashlib.md5(na.encode()).hexdigest()[:16]
            did = doc_ids[na]
            is_long = len(answer) > LONG_ANSWER_CHARS

            n_before = len(records)
            for q in qs:
                nq = normalize(q)
                if len(nq) < MIN_QUERY_LEN:
                    drop('skip_short_query', src, [q], answer_raw, group_id=group_id)
                    continue
                if (nq, did) in seen_pairs:
                    drop('skip_dup_pair', src, [q], answer_raw, group_id=group_id, doc_id=did)
                    continue
                seen_pairs.add((nq, did))

                # anchor、positive 必须是前两个键：ST 按列的位置把数据交给损失函数
                records.append({
                    'anchor':    q,
                    'positive':  answer,
                    'doc_id':    did,
                    'source':    SOURCE_NAME,
                    'src':       src,
                    'group_id':  group_id,
                    'n_chars_a': len(answer),
                    'is_long':   is_long,
                })
                stats['output_pairs'] += 1
            if len(records) > n_before:
                stats['groups_to_finetune'] += 1
                log_changes(changes, src, group_id, answer_raw, answer, 'finetune')
                if is_long:
                    stats['mark_long_answer'] += 1
                    if len(long_samples) < 5:
                        long_samples.append(f'[{len(answer)} chars] {answer[:80]}')

# %%
# =============================================================================
# 写出 + 报告
# =============================================================================

for path, rows in [(OUTPUT_PATH, records), (ONTO_PATH, onto_recs),
                   (DROPPED_PATH, dropped), (MODIFIED_PATH, modified)]:
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

# 报告用的标签表。按流水线顺序列，不用 sorted(stats)——
# 字母序会把"丢弃"和"输出"打散，看不出数据是在哪一步掉的。
# 这张表直接就是数据清洗统计表的内容。
# 键名与 dropped 文件里的 reason 字段一一对应。
REPORT_SECTIONS = [
    ('Input', [
        ('source_records',          'Source records read'),
        ('skip_json_error',         'Malformed JSON lines'),
        ('skip_missing_field',      'Records with no questions or answers'),
        ('qa_groups',               'QA groups paired'),
        ('raw_questions',           'Question strings in input'),
    ]),
    ('Pairing', [
        ('drop_unpaired_group',     'Question groups without an answer'),
        ('drop_unpaired_answer',    'Answers without a question group'),
    ]),
    ('Filtered out, per group', [
        ('skip_empty_answer',       'Empty answer after cleaning'),
        ('drop_echo_only',          'Answer only repeats the question'),
        ('drop_junk_contact',       'Ads / contact information'),
        ('skip_no_valid_question',  'No valid question left'),
        ('drop_non_answer',         'Non-informative answers'),
    ]),
    ('Filtered out, per question', [
        ('drop_empty_question',     'Empty after normalization'),
        ('drop_identical_question', 'Identical to another in its group'),
        ('drop_prefix_question',    'Truncated question variants'),
        ('skip_short_query',        'Query too short'),
        ('skip_dup_pair',           'Duplicate (query, answer) pairs'),
    ]),
    ('Modified, record kept (counted per answer)',
        [(f'mod_{key}', label) for key, label in CLEANING_RULES]),
    ('Routed', [
        ('route_ontology',          'Questions moved to ontology pool'),
    ]),
    ('Output', [
        ('groups_to_finetune',      'QA groups kept for fine-tuning'),
        ('mark_long_answer',        'Groups flagged is_long (kept)'),
        ('output_pairs',            'Training pairs written'),
    ]),
]
LABELS = {k: label for _, rows in REPORT_SECTIONS for k, label in rows}
W = 40   # 标签列宽

print('=' * 70)
print(f'Step 01 · Conversion finished: {SOURCE_NAME}')
print('=' * 70)

for section, rows in REPORT_SECTIONS:
    print(f'\n{section}')
    for key, label in rows:
        print(f'  {label:<{W}s} {stats[key]:>9,}')

# 兜底：以后新增计数项但忘了登记到 REPORT_SECTIONS，也不会被静默吞掉
for key in sorted(set(stats) - set(LABELS)):
    print(f'  {key:<{W}s} {stats[key]:>9,}   <- add to REPORT_SECTIONS')

# ---------- 对账 ----------
# 输入里的每一个问题必须恰好落在三处之一。对不上说明代码里有一条路径在静默丢数据。
n_dropped_q = sum(len(d['questions']) for d in dropped)
accounted   = stats['output_pairs'] + stats['route_ontology'] + n_dropped_q
diff        = accounted - stats['raw_questions']

print('\nAccounting (question strings)')
print(f'  {"In input":<{W}s} {stats["raw_questions"]:>9,}')
print(f'  {"  written to fine-tuning set":<{W}s} {stats["output_pairs"]:>9,}')
print(f'  {"  routed to ontology pool":<{W}s} {stats["route_ontology"]:>9,}')
print(f'  {"  written to dropped file":<{W}s} {n_dropped_q:>9,}')
print(f'  {"Check":<{W}s} {"OK" if diff == 0 else f"MISMATCH {diff:+,}":>9s}')

# ---------- 派生指标 ----------
n_groups_kept = max(stats['groups_to_finetune'], 1)
n_unique      = len({r['doc_id'] for r in records})
print('\nDerived')
print(f'  {"Unique answers (doc_id)":<{W}s} {n_unique:>9,}')
print(f'  {"Query variants per group (mean)":<{W}s} '
      f'{stats["output_pairs"] / n_groups_kept:>9.2f}')
# 分母用"问答组"而不是"输出样本"：一组里的多个同义问法共享同一个答案，
# 拿问题行数当分母的话，光是问法展开就能刷出 ~47% 的假"复用率"。
print(f'  {"Answer duplication rate":<{W}s} {1 - n_unique / n_groups_kept:>9.1%}')
print(f'  {"":<{W}s}   ({n_groups_kept - n_unique:,} groups have an answer '
      f'identical to another group)')

print(f'\nFine-tuning set : {OUTPUT_PATH}  ({len(records):,} pairs)')
print(f'Ontology pool   : {ONTO_PATH}  ({len(onto_recs):,} entries)')
print(f'Dropped items   : {DROPPED_PATH}  ({len(dropped):,} events)')
print(f'Modified answers: {MODIFIED_PATH}  ({len(modified):,} answers)')

if records:
    print('\nSample record (anchor and positive must come first):')
    print(json.dumps(records[0], ensure_ascii=False, indent=2))

# %%
# =============================================================================
# 质检一：被过滤掉的东西长什么样
# =============================================================================
# 这里只打印每类几条样本，完整列表在 dropped 文件里，按 reason 筛就行。

print('\n' + '=' * 70)
print('QC 1 · What the filters removed')
print(f'       (full list: {DROPPED_PATH})')
print('=' * 70)

for _, rows in REPORT_SECTIONS:
    for reason, label in rows:
        if not samples.get(reason):
            continue
        print(f'\n[DROPPED · {label}] {stats[reason]:,} events, samples:')
        for s in samples[reason]:
            print(f'  - {s}')
        if reason == 'skip_short_query':
            print('  If most of these are genuine questions, lower MIN_QUERY_LEN and re-run.')

# 超长答案【没有被丢弃】，只是打了 is_long=True 等微调时切块，单独列出来避免误会
if long_samples:
    print('\n[KEPT · long answers, only flagged is_long=True] samples:')
    for s in long_samples:
        print(f'  - {s}')

# %%
# =============================================================================
# 质检二：清洗前后对比
# =============================================================================
# 这一步不需要医学知识：只看被删掉的是不是标题、寒暄、照抄的问题，
# 以及 &gt; 这类转义是不是还原成了正常字符。
# 人工抽查用文件末尾的代码从 modified 文件随机抽 200 条。

cleaning_examples = {}      # change -> 抽到的样本，也会写进汇总报告

if modified:
    print('\n' + '=' * 70)
    print('QC 2 · Text cleaning, before vs. after')
    print(f'       (full list: {MODIFIED_PATH})')
    print('=' * 70)

    rng = random.Random(0)
    for change, label in CLEANING_RULES:
        pool = [m for m in modified if change in m['changes']]
        if not pool:
            continue
        print(f'\n[{label}] {len(pool):,} answers, samples:')
        cleaning_examples[change] = rng.sample(pool, min(3, len(pool)))
        for m in cleaning_examples[change]:
            print(f'  before: {m["before"][:110]}')
            print(f'  after : {m["after"][:110]}')
            print()

# %%
# =============================================================================
# 质检三：长度分布
# =============================================================================

if records:
    ql = sorted(len(r['anchor'])  for r in records)
    al = sorted(r['n_chars_a']    for r in records)
    pct = lambda xs, p: xs[min(int(len(xs) * p), len(xs) - 1)]

    print('\n' + '=' * 70)
    print('QC 3 · Length distribution (characters)')
    print('=' * 70)
    print(f'\n{"":8s} {"min":>7s} {"p25":>7s} {"p50":>7s} {"p75":>7s} '
          f'{"p95":>7s} {"p99":>7s} {"max":>7s}')
    for name, xs in [('Query', ql), ('Answer', al)]:
        print(f'{name:8s} {xs[0]:>7,} {pct(xs,.25):>7,} {pct(xs,.5):>7,} '
              f'{pct(xs,.75):>7,} {pct(xs,.95):>7,} {pct(xs,.99):>7,} {xs[-1]:>7,}')

    n_long = sum(r['is_long'] for r in records)
    print(f'\nLonger than {LONG_ANSWER_CHARS} chars, to be chunked before training: '
          f'{n_long:,} pairs ({n_long/len(records):.1%})')

    print('\n3 shortest answers (check they are not boilerplate or ads):')
    for r in sorted(records, key=lambda x: x['n_chars_a'])[:3]:
        print(f'  [{r["n_chars_a"]:>4} chars] {r["positive"][:80]}')

# %%
# =============================================================================
# 质检四：本体池盘点
# =============================================================================

if onto_recs:
    usable = [r for r in onto_recs if r['is_atomic'] and r['informative']]
    print('\n' + '=' * 70)
    print('QC 4 · Ontology pool')
    print('=' * 70)
    print(f'\n{len(onto_recs):,} entries, of which {len(usable):,} are atomic '
          f'and informative')
    print('\nSamples:')
    for r in usable[:8]:
        print(f'  Q: {pad(r["question"][:40], 44)} A: {r["answer"]}')

# %%
# =============================================================================
# 质检五：用真实分词器标定 LONG_ANSWER_CHARS（可选)
# =============================================================================


CALIBRATE_WITH_TOKENIZER = False        # 要跑就改成 True
MODEL_NAME  = 'BAAI/bge-large-zh-v1.5'  # 换成你实际要微调的模型
MAX_TOKENS  = 512                       # bge-*-zh 是 512；bge-m3 是 8192

if CALIBRATE_WITH_TOKENIZER and records:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    sub = records[:20000]               # 抽样即可，全量没必要
    n_tok = [len(tok(r['positive'], add_special_tokens=True)['input_ids'])
             for r in tqdm(sub, desc='Tokenizing')]

    ratio = sum(n_tok) / max(sum(r['n_chars_a'] for r in sub), 1)
    over  = sum(t > MAX_TOKENS for t in n_tok)
    st    = sorted(n_tok)
    pct   = lambda xs, p: xs[min(int(len(xs) * p), len(xs) - 1)]

    print('\n' + '=' * 70)
    print('QC 5 · Tokenizer calibration')
    print('=' * 70)
    print(f'\nTokenizer: {MODEL_NAME}   limit: {MAX_TOKENS} tokens')
    print(f'  Measured tokens per character : {ratio:.3f}')
    print(f'  Token count p50/p95/p99/max   : '
          f'{pct(st,.5)} / {pct(st,.95)} / {pct(st,.99)} / {st[-1]}')
    print(f'  Actually above {MAX_TOKENS} tokens     : '
          f'{over:,}/{len(sub):,} ({over/len(sub):.1%})')
    print(f'\n  -> Suggested LONG_ANSWER_CHARS: {int((MAX_TOKENS - 32) / ratio)} '
          f'(currently {LONG_ANSWER_CHARS})')

# %%
# =============================================================================
# 汇总报告
# =============================================================================
# 数据处理部分的数字都从这里取，不用再翻控制台输出。


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


def percentiles(values):
    xs = sorted(values)
    if not xs:
        return {}
    at = lambda p: xs[min(int(len(xs) * p), len(xs) - 1)]
    return {'min': xs[0], 'p25': at(.25), 'p50': at(.5), 'p75': at(.75),
            'p95': at(.95), 'p99': at(.99), 'max': xs[-1]}


drop_events, drop_questions = Counter(), Counter()
for d in dropped:
    drop_events[d['reason']] += 1
    drop_questions[d['reason']] += len(d['questions'])
reason_order  = [k for _, rows in REPORT_SECTIONS for k, _ in rows if k in drop_events]
reason_order += sorted(set(drop_events) - set(reason_order))

report = {
    'step':         '01_convert',
    'source':       SOURCE_NAME,
    'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    'python':       sys.version.split()[0],
    'config': {
        'INPUT_PATH':          INPUT_PATH,
        'MIN_QUERY_LEN':       MIN_QUERY_LEN,
        'ONTOLOGY_MAX_LEN':    ONTOLOGY_MAX_LEN,
        'ATOMIC_MAX_LEN':      ATOMIC_MAX_LEN,
        'LONG_ANSWER_CHARS':   LONG_ANSWER_CHARS,
        'UNESCAPE_HTML':       UNESCAPE_HTML,
        'STRIP_GREETING':      STRIP_GREETING,
        'STRIP_TEMPLATE':      STRIP_TEMPLATE,
        'STRIP_QUESTION_ECHO': STRIP_QUESTION_ECHO,
        'ECHO_MIN_LEN':        ECHO_MIN_LEN,
        'DROP_PREFIX_DUP':     DROP_PREFIX_DUP,
        'DROP_JUNK':           DROP_JUNK,
    },
    'stats': {**{k: stats[k] for k in LABELS}, **stats},
    'accounting': {
        'in_input': stats['raw_questions'],
        'finetune': stats['output_pairs'],
        'ontology': stats['route_ontology'],
        'dropped':  n_dropped_q,
        'ok':       diff == 0,
    },
    'derived': {
        'unique_answers':          n_unique,
        'variants_per_group':      round(stats['output_pairs'] / n_groups_kept, 4),
        'answer_duplication_rate': round(1 - n_unique / n_groups_kept, 4),
    },
    'lengths': {
        'query':      percentiles(len(r['anchor']) for r in records),
        'answer':     percentiles(r['n_chars_a'] for r in records),
        'long_pairs': sum(r['is_long'] for r in records),
    },
    'ontology': {
        'entries':            len(onto_recs),
        'atomic_informative': sum(bool(r['is_atomic'] and r['informative']) for r in onto_recs),
    },
    'dropped_by_reason': [
        {'reason': r, 'label': LABELS.get(r, r),
         'events': drop_events[r], 'questions': drop_questions[r]}
        for r in reason_order
    ],
    'outputs': {name: file_info(path) for name, path in [
        ('finetune', OUTPUT_PATH), ('ontology', ONTO_PATH),
        ('dropped', DROPPED_PATH), ('modified', MODIFIED_PATH)]},
}

md = [
    f'# Step 01 · Conversion report — `{SOURCE_NAME}`',
    '',
    f'Generated {report["generated_at"]} · Python {report["python"]}',
    '',
    '## Configuration',
    '',
    md_table(['Parameter', 'Value'],
             [[k, v] for k, v in report['config'].items()], numeric=False),
    '',
    '## Processing statistics',
    '',
    md_table(['Stage', 'Item', 'Count'],
             [[section, label, fmt(stats[key])]
              for section, rows in REPORT_SECTIONS for key, label in rows]),
    '',
    '> 数据清洗流程表。"Filtered out" 是过滤掉的，"Modified" 是清洗过但保留的，'
    '"Routed" 进了本体池，"Output" 是最终进入微调的。',
    '',
    '## Accounting (question strings)',
    '',
    md_table(['Destination', 'Count'], [
        ['In input',        fmt(stats['raw_questions'])],
        ['Fine-tuning set', fmt(stats['output_pairs'])],
        ['Ontology pool',   fmt(stats['route_ontology'])],
        ['Dropped file',    fmt(n_dropped_q)],
        ['Check',           'OK' if diff == 0 else f'MISMATCH {diff:+,}'],
    ]),
    '',
    '> 对账通过说明每一条输入都有去处，没有被静默丢弃。',
    '',
    '## Derived metrics',
    '',
    md_table(['Metric', 'Value'], [
        ['Unique answers (doc_id)',         fmt(n_unique)],
        ['Query variants per group (mean)', f'{report["derived"]["variants_per_group"]:.2f}'],
        ['Answer duplication rate',         f'{report["derived"]["answer_duplication_rate"]:.1%}'],
    ]),
    '',
]

if records:
    L = report['lengths']
    md += ['## Length distribution (characters)', '',
           md_table(['', *L['query']],
                    [['Query',  *map(fmt, L['query'].values())],
                     ['Answer', *map(fmt, L['answer'].values())]]),
           '',
           f'Answers longer than {LONG_ANSWER_CHARS} characters are kept and flagged '
           f'`is_long`: {L["long_pairs"]:,} pairs ({L["long_pairs"] / len(records):.1%}).',
           '']

md += ['## Text cleaning', '',
       f'Every modified answer is listed in `{MODIFIED_PATH}`. '
       'The examples below are a random sample (seed 0).', '']
for change, label in CLEANING_RULES:
    ex = cleaning_examples.get(change)
    if not ex:
        continue
    md += [f'### {label} — {stats["mod_" + change]:,} answers', '',
           md_table(['Before', 'After'],
                    [[m['before'][:150], m['after'][:150]] for m in ex], numeric=False),
           '']
md += ['> 方法部分介绍清洗规则时各举一例；人工抽查用本文件末尾的代码导出 200 条。', '']

md += ['## Dropped items', '',
       f'Every dropped item is listed in `{DROPPED_PATH}` with its reason.', '',
       md_table(['Reason', 'Description', 'Events', 'Questions'],
                [[d['reason'], d['label'], fmt(d['events']), fmt(d['questions'])]
                 for d in report['dropped_by_reason']]),
       '']
for r in reason_order:
    if samples.get(r):
        md += [f'**{LABELS.get(r, r)}** — samples', '']
        md += ['- ' + ' '.join(s.split()) for s in samples[r]]
        md += ['']

md += ['## Ontology pool', '',
       md_table(['Item', 'Count'], [
           ['Entries',                fmt(report['ontology']['entries'])],
           ['Atomic and informative', fmt(report['ontology']['atomic_informative'])],
       ]),
       '',
       '## Output files', '',
       md_table(['File', 'Lines', 'md5'],
                [[i['path'], fmt(i['lines']), i['md5']] for i in report['outputs'].values()]),
       '',
       '> md5 用来核对：同样的输入和参数重跑一遍，md5 应该完全一样。']

md_path   = f'{REPORT_DIR}/01_convert_{SOURCE_NAME}.md'
json_path = f'{REPORT_DIR}/01_convert_{SOURCE_NAME}.json'
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
with open(md_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(md) + '\n')

print(f'\nReport: {md_path}')
print(f'        {json_path}')