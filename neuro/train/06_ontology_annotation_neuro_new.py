# %%
# =============================================================================
# 配置
# =============================================================================
import argparse, csv, datetime, hashlib, json, math, os, random, re, sys, unicodedata
from collections import Counter, defaultdict

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=os.environ.get('NEURO_OUT_DIR', '/data/neuro/data/neuro_dataset'))
    ap.add_argument('--tag', default='new', help='输出目录后缀，默认 new')
    ap.add_argument('--seed', type=int, default=42)
    a, _unknown = ap.parse_known_args()    # Jupyter 会塞进 -f kernel.json 之类的参数，忽略即可
    return a


ARGS = _args()
OUT_DIR, TAG, SEED = ARGS.out_dir, ARGS.tag, ARGS.seed

SPLIT_DIR = f'{OUT_DIR}/split'
ONTO_DIR = f'{OUT_DIR}/ontology_{TAG}'
REPORT_DIR = f'{OUT_DIR}/reports'
TRAIN_PATH = f'{SPLIT_DIR}/train.jsonl'
CORPUS_PATH = f'{SPLIT_DIR}/train_corpus.jsonl'

SHORT_QUESTION_CHARS = 40     # 没有提问小句时，短于这个长度才整句判断方面
TAG_ASPECT = True             # 方面标签只作为问题的附加信息输出，08 不使用
N_CHECK = 200                 # 每张人工抽检表的条数
N_PRIOR_PAIRS = 20000         # 估计随机配对先验比例用的对数
TOP_UNLABELLED_TERMS = 80     # 漏标诊断里列出的高频词数
USE_FAERS = True              # 只由 FAERS 支持的药物关系（来源 wd_faers）是否使用
ENTITY_WEIGHT = 0.5           # onto_sim 里 entity_score 的权重，其余给疾病层关系
MAX_CANDIDATES = 5000         # 难负例抽检时每个问题最多打分的候选答案数
HARD_TOPK = 3                 # 难负例抽检表里每个问题列出的候选数

os.makedirs(ONTO_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# %%
# =============================================================================
# 基础函数（normalize / qid_of 必须与 04 及后续步骤完全一致）
# =============================================================================


def normalize(text):
    '''与 01–04 完全一致的归一化。qid 依赖它，不能改。这里不做繁简转换。'''
    text = unicodedata.normalize('NFKC', text or '').lower()
    return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text)


def qid_of(nq):
    '''与 04 及后续步骤一致：归一化问题的 md5 前 16 位'''
    return hashlib.md5(nq.encode()).hexdigest()[:16]


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


# 繁体转简体：训练集里可能还有繁体（00–04 没有转换）。只在匹配词典时转换。
try:
    from opencc import OpenCC
    _t2s = OpenCC('t2s').convert
    T2S_BACKEND = 'opencc'
except ImportError:
    _FALLBACK = str.maketrans({
        '腦': '脑', '頭': '头', '癲': '癫', '癇': '痫', '顫': '颤', '經': '经', '痺': '痹', '痙': '痉',
        '攣': '挛', '綜': '综', '體': '体', '狀': '状', '變': '变', '壓': '压', '動': '动', '脈': '脉',
        '靜': '静', '腫': '肿', '膠': '胶', '質': '质', '徵': '征', '礙': '碍', '縮': '缩', '無': '无',
        '藥': '药', '療': '疗', '診': '诊', '斷': '断', '檢': '检', '術': '术', '醫': '医', '遺': '遗',
        '傳': '传', '種': '种', '類': '类', '發': '发', '熱': '热', '燒': '烧', '暈': '晕', '覺': '觉',
        '視': '视', '語': '语', '認': '认', '記': '记', '憶': '忆', '減': '减', '憂': '忧', '鬱': '郁',
        '慮': '虑', '頓': '顿', '聯': '联', '陣': '阵', '勢': '势', '嚥': '咽', '顏': '颜', '瞼': '睑',
        '側': '侧', '險': '险', '驚': '惊', '懷': '怀', '輸': '输', '導': '导', '暫': '暂', '盤': '盘',
        '積': '积', '態': '态', '額': '额', '顳': '颞', '葉': '叶', '進': '进', '邊': '边', '創': '创',
        '傷': '伤', '損': '损', '營': '营', '電': '电', '層': '层', '複': '复', '細': '细', '預': '预',
        '護': '护', '復': '复', '針': '针', '劑': '剂', '臨': '临', '飲': '饮', '確': '确', '頸': '颈',
        '關': '关', '節': '节', '網': '网', '轉': '转', '運': '运', '遲': '迟', '緩': '缓', '協': '协',
        '調': '调', '盪': '荡',
    })
    _t2s = lambda s: s.translate(_FALLBACK)
    T2S_BACKEND = 'built-in table (partial)'
    print('opencc not installed: using a small built-in Traditional->Simplified table. '
          'pip install opencc-python-reimplemented for full coverage.')


def prep(text):
    '''匹配词典前的统一处理：全角转半角、繁转简、英文转大写。'''
    return _t2s(unicodedata.normalize('NFKC', text or '')).upper()

# %%
# =============================================================================
# 概念层：神经科学词典（神经内科 + 神经外科；精神科只留三个用来区分“不同病”的概念）
# =============================================================================
# 每个概念：id / name / parent / icd10 / icd10_title / wiki / synonyms / extra
# 同义词来源：中文维基条目标题和重定向、MSD 页面标题括号里的别名、WHO ICD-10 中文名、
# 精筛神经科学.py 里逐词核验过的关键词。extra 放指向该病的非病名词（别名、简称），每条写明依据。
# icd10_title 为空，表示这次没取到该编码的 WHO 标题，请对照 WHO 页面补。
# 兄弟概念 = 同一个 parent 下的概念。

W = 'https://zh.wikipedia.org/wiki/'
FAMILY = {}


def C(cid, name, parent, fam, icd10, title, wiki, syn, extra=None):
    '''fam 是大类，写进 FAMILY，供 08 做“不同病”的保守判断'''
    FAMILY[cid] = fam
    return {'id': cid, 'name': name, 'parent': parent, 'icd10': icd10, 'icd10_title': title,
            'wiki': W + wiki if wiki else None, 'synonyms': syn, 'extra': extra or {}}


CONCEPTS = [
    # ---- 脑血管病 [ICD-10 I60-I69, G45] ----
    C('cerebrovascular', '脑血管病', None, 'cerebrovascular', ['I60-I69'], 'Cerebrovascular diseases', None,
      ['脑血管病', '脑血管疾病']),
    C('stroke', '脑卒中', 'cerebrovascular', 'cerebrovascular', ['I60-I64'],
      'Subarachnoid haemorrhage; Intracerebral haemorrhage; Cerebral infarction; Stroke, not specified as haemorrhage or infarction',
      '中風', ['脑卒中', '中风', '卒中', '脑血管意外', '脑血管事件', '脑中风'],
      {'脑血管病后遗症': 'WHO ICD-10 I69 Sequelae of cerebrovascular disease；精筛神经科学.py 归在脑血管病'}),
    C('cerebral_infarction', '脑梗死', 'stroke', 'cerebrovascular', ['I63'], 'Cerebral infarction', '脑梗死',
      ['脑梗死', '脑梗塞', '脑梗', '缺血性脑卒中', '缺血性卒中', '缺血性中风', '缺血性脑中风', '脑血栓', '脑栓塞',
       '腔隙性脑梗死', '腔梗']),
    C('cerebral_hemorrhage', '脑出血', 'stroke', 'cerebrovascular', ['I61'], 'Intracerebral haemorrhage', '腦出血',
      ['脑出血', '脑溢血', '出血性脑卒中', '出血性卒中', '出血性中风', '颅内出血']),
    C('sah', '蛛网膜下腔出血', 'stroke', 'cerebrovascular', ['I60'], 'Subarachnoid haemorrhage', '蛛网膜下腔出血',
      ['蛛网膜下腔出血', '蜘蛛膜下腔出血']),
    C('tia', '短暂性脑缺血发作', 'cerebrovascular', 'cerebrovascular', ['G45'],
      'Transient cerebral ischaemic attacks and related syndromes', '短暂性脑缺血发作',
      ['短暂性脑缺血发作', '短暂性脑缺血', '短暂性大脑缺血发作', '暂时性脑缺血', '小中风', 'TIA']),
    C('cerebral_aneurysm', '脑动脉瘤', 'cerebrovascular', 'cerebrovascular', ['I67.1'],
      'Cerebral aneurysm, nonruptured', None, ['脑动脉瘤', '颅内动脉瘤', '脑内的动脉瘤']),
    C('moyamoya', '烟雾病', 'cerebrovascular', 'cerebrovascular', ['I67.5'], 'Moyamoya disease', '烟雾病', ['烟雾病']),
    C('avm', '脑动静脉畸形', 'cerebrovascular', 'cerebrovascular', ['Q28.2'],
      'Arteriovenous malformation of cerebral vessels', '動靜脈畸形', ['脑动静脉畸形', '动静脉畸形']),
    C('cvst', '脑静脉窦血栓', 'cerebrovascular', 'cerebrovascular', ['I67.6', 'I63.6'],
      'Nonpyogenic thrombosis of intracranial venous system; Cerebral infarction due to cerebral venous thrombosis, nonpyogenic',
      None, ['脑静脉窦血栓', '静脉窦血栓', '颅内静脉窦血栓']),

    # ---- 发作性疾病 [ICD-10 G40-G41, R56.0] ----
    C('epilepsy', '癫痫', None, 'epilepsy', ['G40'], 'Epilepsy', '癫痫',
      ['癫痫', '癫痫病', '癫痫发作', '羊癫疯', '羊角风', '全面性发作', '局灶性发作', '失神发作', '强直阵挛发作',
       '婴儿痉挛症', 'Dravet综合征', 'Lennox-Gastaut综合征']),
    C('status_epilepticus', '癫痫持续状态', 'epilepsy', 'epilepsy', ['G41'], 'Status epilepticus', '癲癇重積狀態',
      ['癫痫持续状态', '癫痫状态', '癫痫重积状态']),
    C('febrile_seizure', '热性惊厥', None, 'epilepsy', ['R56.0'], 'Febrile convulsions', '熱性痙攣',
      ['热性惊厥', '热性痉挛', '热性抽搐', '高热惊厥', '发烧性抽搐', '热性全身痉挛']),

    # ---- 头痛与颅面痛 [ICD-10 G43-G44, G50-G51] ----
    C('migraine', '偏头痛', None, 'headache', ['G43'], 'Migraine', '偏頭痛', ['偏头痛', '偏头疼']),
    C('tension_headache', '紧张型头痛', None, 'headache', ['G44.2'], 'Tension-type headache', None,
      ['紧张型头痛', '紧张性头痛']),
    C('cluster_headache', '丛集性头痛', None, 'headache', ['G44.0'], 'Cluster headache syndrome', '丛集性头痛',
      ['丛集性头痛']),
    C('trigeminal_neuralgia', '三叉神经痛', None, 'cranial_nerve', ['G50.0'], 'Trigeminal neuralgia', '三叉神經痛',
      ['三叉神经痛', '非典型三叉神经痛']),
    C('bell_palsy', '面神经麻痹', None, 'cranial_nerve', ['G51.0'], 'Bell palsy', '顏面神經麻痺',
      ['面神经麻痹', '面瘫', '贝尔麻痹', '周围性面瘫', '颜面神经麻痹']),
    C('hemifacial_spasm', '面肌痉挛', None, 'cranial_nerve', [], '', '半面痉挛',
      ['面肌痉挛', '偏侧面肌痉挛', '半面痉挛']),
]

CONCEPTS += [
    # ---- 运动障碍与变性病 [ICD-10 G10-G26] ----
    C('parkinson', '帕金森病', None, 'movement', ['G20'], 'Parkinson disease', '帕金森氏症',
      ['帕金森病', '帕金森氏症', '帕金森', '帕金森氏病', '巴金森氏症', '巴金森病', '柏金逊症', '柏金逊病']),
    C('parkinsonism', '帕金森综合征', 'parkinson', 'movement', ['G21', 'G22'],
      'Secondary parkinsonism; Parkinsonism in diseases classified elsewhere', '帕金森症候群',
      ['帕金森综合征', '帕金森综合症', '帕金森氏综合征', '帕金森症候群', '继发性帕金森综合征']),
    C('msa', '多系统萎缩', 'parkinsonism', 'movement', ['G23.2', 'G23.3'],
      'Multiple system atrophy, parkinsonian type [MSA-P]; Multiple system atrophy, cerebellar type [MSA-C]', None,
      ['多系统萎缩']),
    C('psp', '进行性核上性麻痹', 'parkinsonism', 'movement', ['G23.1'],
      'Progressive supranuclear ophthalmoplegia [Steele-Richardson-Olszewski]', '進行性上眼神經核麻痺症',
      ['进行性核上性麻痹', '进行性上眼神经核麻痹症']),
    C('essential_tremor', '特发性震颤', None, 'movement', ['G25.0'], 'Essential tremor', None,
      ['特发性震颤', '原发性震颤']),
    C('dystonia', '肌张力障碍', None, 'movement', ['G24'], 'Dystonia', '肌张力障碍',
      ['肌张力障碍', '颈部肌张力障碍']),
    C('torticollis', '痉挛性斜颈', 'dystonia', 'movement', ['G24.3'], 'Spasmodic torticollis', '斜頸',
      ['痉挛性斜颈', '斜颈']),
    C('blepharospasm', '眼睑痉挛', 'dystonia', 'movement', ['G24.5'], 'Blepharospasm', '眼瞼痙攣',
      ['眼睑痉挛', '睑痉挛']),
    C('huntington', '亨廷顿病', None, 'movement', ['G10'], 'Huntington disease', '亨廷頓病',
      ['亨廷顿病', '亨廷顿舞蹈病', '亨廷顿舞蹈症', '亨廷顿氏症', '亨廷顿氏舞蹈病', '亨丁顿舞蹈症',
       '杭丁顿舞蹈症', '杭廷顿舞蹈病', '遗传性舞蹈病', '慢性进行性舞蹈病']),
    C('tic_disorder', '抽动障碍', None, 'movement', ['F95'], 'Tic disorders', '抽动障碍',
      ['抽动障碍', '抽动症']),
    C('tourette', '抽动秽语综合征', 'tic_disorder', 'movement', ['F95.2'],
      'Combined vocal and multiple motor tic disorder [de la Tourette]', '妥瑞氏症',
      ['抽动秽语综合征', '妥瑞氏症', '妥瑞症', '图雷特综合症', 'Tourette综合征', '发声与多种运动联合抽动障碍']),
    C('rls', '不宁腿综合征', None, 'movement', ['G25.8'], 'Other specified extrapyramidal and movement disorders',
      '不寧腿綜合症', ['不宁腿综合征', '不宁腿综合症', '不安腿综合征', '多动腿综合征', '不宁腿症']),
    C('ataxia', '遗传性共济失调', None, 'movement', ['G11'], 'Hereditary ataxia', None,
      ['遗传性共济失调', '小脑萎缩', '小脑萎缩症', '脊髓小脑性共济失调']),
    C('wilson', '肝豆状核变性', None, 'metabolic', ['E83.0'], 'Disorders of copper metabolism', '肝豆狀核變性',
      ['肝豆状核变性', '威尔森氏症', '威尔逊病', '威尔逊氏病']),

    # ---- 痴呆 [ICD-10 G30-G31, F00-F03] ----
    C('dementia', '痴呆', None, 'dementia', ['F00-F03'], 'Dementia in Alzheimer disease; Vascular dementia; '
      'Dementia in other diseases classified elsewhere; Unspecified dementia', '失智症',
      ['痴呆', '痴呆症', '失智症', '老年痴呆', '老年痴呆症', '老人痴呆', '脑退化症', '认知障碍症']),
    C('alzheimer', '阿尔茨海默病', 'dementia', 'dementia', ['G30', 'F00'], 'Alzheimer disease', '阿茲海默症',
      ['阿尔茨海默病', '阿尔茨海默症', '阿兹海默症', '阿兹海默病', '阿尔兹海默病', '阿尔兹海默症']),
    C('vascular_dementia', '血管性痴呆', 'dementia', 'dementia', ['F01'], 'Vascular dementia', '血管性失智症',
      ['血管性痴呆', '血管性痴呆症', '血管性失智症', '多发梗塞性痴呆']),
    C('ftd', '额颞叶痴呆', 'dementia', 'dementia', ['G31.0'], 'Circumscribed brain atrophy', '額顳葉失智症',
      ['额颞叶痴呆', '额颞叶失智症', '额颞叶型失智症']),
    C('dlb', '路易体痴呆', 'dementia', 'dementia', ['G31.8'], 'Other specified degenerative diseases of nervous system',
      None, ['路易体痴呆', '路易体失智症']),

    # ---- 中枢神经系统脱髓鞘疾病 [ICD-10 G35-G37, G04.0] ----
    C('ms', '多发性硬化', None, 'demyelinating', ['G35'], 'Multiple sclerosis', '多发性硬化症',
      ['多发性硬化', '多发性硬化症', '多发性硬化病']),
    C('nmo', '视神经脊髓炎', None, 'demyelinating', ['G36.0'], 'Neuromyelitis optica [Devic]', None,
      ['视神经脊髓炎', '视神经脊髓炎谱系疾病', 'NMOSD']),
    C('adem', '急性播散性脑脊髓炎', None, 'demyelinating', ['G04.0'], 'Acute disseminated encephalitis', None,
      ['急性播散性脑脊髓炎']),
    C('myelitis', '脊髓炎', None, 'demyelinating', ['G04', 'G37.3'],
      'Encephalitis, myelitis and encephalomyelitis; Acute transverse myelitis in demyelinating disease of CNS',
      '脊髓炎', ['脊髓炎', '横贯性脊髓炎', '急性横贯性脊髓炎']),

    # ---- 中枢神经系统感染 [ICD-10 G00-G09, A17, A80-A89, B02, B69] ----
    C('meningitis', '脑膜炎', None, 'infection', ['G00-G03'], 'Bacterial meningitis, NEC; Meningitis due to other '
      'and unspecified causes', '脑膜炎', ['脑膜炎']),
    C('bacterial_meningitis', '细菌性脑膜炎', 'meningitis', 'infection', ['G00'],
      'Bacterial meningitis, not elsewhere classified', None,
      ['细菌性脑膜炎', '化脓性脑膜炎', '急性细菌性脑膜炎']),
    C('viral_meningitis', '病毒性脑膜炎', 'meningitis', 'infection', ['A87'], 'Viral meningitis', None,
      ['病毒性脑膜炎']),
    C('tb_meningitis', '结核性脑膜炎', 'meningitis', 'infection', ['A17.0'], 'Tuberculous meningitis', None,
      ['结核性脑膜炎']),
    C('encephalitis', '脑炎', None, 'infection', ['G04', 'A86'], 'Encephalitis, myelitis and encephalomyelitis',
      '脑炎', ['脑炎', '病毒性脑炎', '脑膜脑炎']),
    C('je', '流行性乙型脑炎', 'encephalitis', 'infection', ['A83.0'], 'Japanese encephalitis', '流行性乙型脑炎',
      ['流行性乙型脑炎', '乙型脑炎', '乙脑', '日本脑炎', '日本乙型脑炎']),
    C('autoimmune_encephalitis', '自身免疫性脑炎', 'encephalitis', 'infection', [], '', None,
      ['自身免疫性脑炎']),
    C('brain_abscess', '脑脓肿', None, 'infection', ['G06'], 'Intracranial and intraspinal abscess and granuloma',
      None, ['脑脓肿', '颅内脓肿', '硬膜下积脓']),
    C('polio', '脊髓灰质炎', None, 'infection', ['A80', 'G14'], 'Acute poliomyelitis; Postpolio syndrome',
      '脊髓灰質炎', ['脊髓灰质炎', '小儿麻痹', '小儿麻痹症', '急性脊髓灰质炎', '小儿麻痹后遗症', '脊灰']),
    C('cjd', '克雅病', None, 'infection', ['A81.0'], 'Creutzfeldt-Jakob disease', '克雅二氏病',
      ['克雅病', '克雅二氏病', '克罗伊茨费尔特-雅各布病', '朊病毒病']),
    C('neurocysticercosis', '脑囊虫病', None, 'infection', ['B69.0'], 'Cysticercosis of central nervous system',
      '囊虫病', ['脑囊虫病', '脑囊尾蚴病', '囊虫病', '囊尾蚴病', '猪囊尾蚴病']),
    C('shingles', '带状疱疹', None, 'infection', ['B02'], 'Zoster [herpes zoster]', '带状疱疹',
      ['带状疱疹']),
    C('phn', '带状疱疹后神经痛', 'shingles', 'infection', ['B02.2'],
      'Zoster with other nervous system involvement', None, ['带状疱疹后神经痛', '带状疱疹后遗神经痛']),

    # ---- 周围神经与神经肌肉疾病 [ICD-10 G12, G60-G73] ----
    C('peripheral_neuropathy', '周围神经病', None, 'neuromuscular', ['G60-G64'],
      'Polyneuropathies and other disorders of the peripheral nervous system', '周邊神經病變',
      ['周围神经病', '周围神经病变', '外周神经病变', '多发性神经病', '多发性周围神经病', '多发性神经炎',
       '周边神经病变']),
    C('diabetic_neuropathy', '糖尿病周围神经病变', 'peripheral_neuropathy', 'neuromuscular', ['G63'],
      'Polyneuropathy in diseases classified elsewhere', None,
      ['糖尿病周围神经病变', '糖尿病神经病变', '糖尿病性周围神经病变']),
    C('gbs', '吉兰-巴雷综合征', 'peripheral_neuropathy', 'neuromuscular', ['G61.0'], 'Guillain-Barré syndrome',
      '格林-巴利综合征',
      ['吉兰-巴雷综合征', '格林-巴利综合征', '格林巴利综合征', '格林-巴利综合症', '吉兰巴雷综合征',
       '古兰-巴雷综合征', '急性炎性脱髓鞘多发性神经根病', '急性特发性多发性神经炎', 'GBS']),
    C('cidp', '慢性炎性脱髓鞘性多发性神经病', 'peripheral_neuropathy', 'neuromuscular', ['G61.8'],
      'Other inflammatory polyneuropathies', None, ['慢性炎性脱髓鞘性多发性神经病', 'CIDP']),
    C('cmt', '腓骨肌萎缩症', 'peripheral_neuropathy', 'neuromuscular', ['G60.0'],
      'Hereditary motor and sensory neuropathy', '进行性神经性腓骨肌萎缩症',
      ['腓骨肌萎缩症', '进行性神经性腓骨肌萎缩症', '遗传性运动感觉神经病']),
    C('mg', '重症肌无力', None, 'neuromuscular', ['G70.0'], 'Myasthenia gravis', '重症肌無力',
      ['重症肌无力', '重症肌无力症', '重肌无力症']),
    C('lems', '肌无力综合征', None, 'neuromuscular', ['G73.1'], 'Lambert-Eaton syndrome', None,
      ['肌无力综合征', 'Lambert-Eaton综合征']),
    C('als', '肌萎缩侧索硬化', None, 'neuromuscular', ['G12.2'], 'Motor neuron disease', '肌萎缩侧索硬化',
      ['肌萎缩侧索硬化', '肌萎缩侧索硬化症', '肌萎缩性侧索硬化', '肌萎缩性侧索硬化症', '肌肉萎缩性侧索硬化症',
       '运动神经元病', '运动神经元疾病', '渐冻人', '渐冻症', '渐冻人症', '卢伽雷氏症', '葛雷克氏症', 'ALS']),
    C('sma', '脊髓性肌萎缩症', None, 'neuromuscular', ['G12.0', 'G12.1'],
      'Infantile spinal muscular atrophy, type I [Werdnig-Hoffman]; Other inherited spinal muscular atrophy',
      '脊髓性肌肉萎縮症', ['脊髓性肌萎缩症', '脊髓性肌肉萎缩症', '脊肌萎缩症', 'SMA']),
    C('muscular_dystrophy', '肌营养不良', None, 'neuromuscular', ['G71.0'], 'Muscular dystrophy', None,
      ['肌营养不良', '肌营养不良症', '进行性肌营养不良', '进行性肌营养不良症']),
    C('periodic_paralysis', '周期性麻痹', None, 'neuromuscular', ['G72.3'], 'Periodic paralysis', None,
      ['周期性麻痹', '周期性瘫痪', '低钾性周期性麻痹']),

    # ---- 神经、神经根和脊柱相关 [ICD-10 G56-G58, M47, M50-M51, M54.3] ----
    C('carpal_tunnel', '腕管综合征', None, 'peripheral_nerve', ['G56.0'], 'Carpal tunnel syndrome',
      '腕隧道症候群', ['腕管综合征', '腕管综合症', '腕隧道症候群', '腕道症候群', '鼠标手']),
    C('sciatica', '坐骨神经痛', None, 'peripheral_nerve', ['M54.3', 'G57.0'], 'Sciatica; Lesion of sciatic nerve',
      '坐骨神經痛', ['坐骨神经痛']),
    C('disc_herniation', '椎间盘突出症', None, 'peripheral_nerve', ['M50', 'M51'],
      'Cervical disc disorders; Other intervertebral disc disorders', '椎間盤突出症',
      ['椎间盘突出症', '椎间盘突出', '椎间盘脱出', '腰椎间盘突出', '腰椎间盘突出症', '腰间盘突出',
       '颈椎间盘突出症', '腰突']),
    C('cervical_spondylosis', '颈椎病', None, 'peripheral_nerve', ['M47'], 'Spondylosis', '椎關節病',
      ['颈椎病', '脊髓型颈椎病', '神经根型颈椎病', '椎动脉型颈椎病', '退化性脊椎炎']),
    C('intercostal_neuralgia', '肋间神经痛', None, 'peripheral_nerve', ['G58.0'], 'Intercostal neuropathy',
      '肋间神经痛', ['肋间神经痛']),

    # ---- 睡眠障碍 [ICD-10 G47] ----
    C('insomnia', '失眠', None, 'sleep', ['G47.0', 'F51.0'],
      'Disorders of initiating and maintaining sleep [insomnias]; Nonorganic insomnia', '失眠',
      ['失眠', '失眠症']),
    C('narcolepsy', '发作性睡病', None, 'sleep', ['G47.4'], 'Narcolepsy and cataplexy', '發作性睡病',
      ['发作性睡病', '发作性嗜睡病', '嗜睡症', '猝睡症', '突发性睡眠症', '嗜眠发作']),
    C('sleep_apnea', '睡眠呼吸暂停', None, 'sleep', ['G47.3'], 'Sleep apnoea', '睡眠呼吸暂停',
      ['睡眠呼吸暂停', '睡眠呼吸暂停综合征', '睡眠呼吸暂停症', '睡眠窒息症', '睡眠呼吸中止症',
       '阻塞性睡眠呼吸暂停']),

    # ---- 神经系统肿瘤 [ICD-10 C70-C72, C79.3, D32-D33, D35.2] ----
    C('brain_tumor', '脑肿瘤', None, 'tumor', ['C71', 'D33'],
      'Malignant neoplasm of brain; Benign neoplasm of brain and other parts of CNS', '腦瘤',
      ['脑肿瘤', '脑瘤', '脑癌', '脑部肿瘤', '颅内肿瘤', '颅内占位']),
    C('glioma', '胶质瘤', 'brain_tumor', 'tumor', ['C71'], 'Malignant neoplasm of brain', '神經膠質瘤',
      ['胶质瘤', '神经胶质瘤', '脑神经胶质瘤', '星形细胞瘤', '少突胶质细胞瘤', '室管膜瘤']),
    C('glioblastoma', '胶质母细胞瘤', 'glioma', 'tumor', ['C71'], 'Malignant neoplasm of brain', '胶质母细胞瘤',
      ['胶质母细胞瘤']),
    C('meningioma', '脑膜瘤', 'brain_tumor', 'tumor', ['D32'], 'Benign neoplasm of meninges', '腦膜瘤', ['脑膜瘤']),
    C('pituitary_adenoma', '垂体瘤', 'brain_tumor', 'tumor', ['D35.2'], 'Benign neoplasm: Pituitary gland', '垂体瘤',
      ['垂体瘤', '垂体腺瘤', '脑垂体瘤', '泌乳素瘤']),
    C('acoustic_neuroma', '听神经瘤', 'brain_tumor', 'tumor', ['D33.3'], 'Benign neoplasm: Cranial nerves', None,
      ['听神经瘤', '前庭神经鞘瘤']),
    C('brain_metastasis', '脑转移瘤', 'brain_tumor', 'tumor', ['C79.3'],
      'Secondary malignant neoplasm of brain and cerebral meninges', None, ['脑转移瘤', '脑转移']),

    # ---- 颅脑及脊髓创伤 [ICD-10 S06, T09.3] ----
    C('tbi', '颅脑损伤', None, 'trauma', ['S06'], 'Intracranial injury', '创伤性脑损伤',
      ['颅脑损伤', '颅脑外伤', '脑外伤', '创伤性脑损伤', '外伤性脑损伤', '脑挫伤', '脑挫裂伤', '弥漫性轴索损伤',
       '颅内损伤']),
    C('concussion', '脑震荡', 'tbi', 'trauma', ['S06.0'], 'Concussion', '腦震盪', ['脑震荡']),
    C('subdural_hematoma', '硬膜下血肿', 'tbi', 'trauma', ['S06.5'], 'Traumatic subdural haemorrhage', None,
      ['硬膜下血肿', '慢性硬膜下血肿', '硬膜外血肿', '颅内血肿']),
    C('sci', '脊髓损伤', None, 'trauma', ['T09.3'], 'Injury of spinal cord, level unspecified', '脊髓損傷',
      ['脊髓损伤', '脊椎损伤', '脊髓受伤']),

    # ---- 脑脊液 / 颅内压 [ICD-10 G91, G93.2, G93.6] ----
    C('hydrocephalus', '脑积水', None, 'csf', ['G91'], 'Hydrocephalus', '脑积水', ['脑积水']),
    C('nph', '正常压力脑积水', 'hydrocephalus', 'csf', ['G91.2'], 'Normal-pressure hydrocephalus', None,
      ['正常压力脑积水', '正常颅压脑积水']),
    C('cerebral_edema', '脑水肿', None, 'csf', ['G93.6'], 'Cerebral oedema', '腦水腫', ['脑水肿']),
    C('iih', '特发性颅内压增高', None, 'csf', ['G93.2'], 'Benign intracranial hypertension', None,
      ['特发性颅内压增高', '良性颅内压增高']),

    # ---- 脊髓疾病 [ICD-10 G95, G83.4, Q05, Q07.0] ----
    C('syringomyelia', '脊髓空洞症', None, 'spinal_cord', ['G95.0'], 'Syringomyelia and syringobulbia', '脊髓空洞症',
      ['脊髓空洞症', '脊髓空洞']),
    C('chiari', '小脑扁桃体下疝畸形', None, 'spinal_cord', ['Q07.0'], 'Arnold-Chiari syndrome', None,
      ['小脑扁桃体下疝畸形', '小脑扁桃体下疝', 'Chiari畸形', '下疝畸形']),
    C('cord_compression', '脊髓压迫症', None, 'spinal_cord', ['G95.2'], 'Cord compression, unspecified', None,
      ['脊髓压迫症', '脊髓压迫', '脊髓栓系']),
    C('cauda_equina', '马尾综合征', None, 'spinal_cord', ['G83.4'], 'Cauda equina syndrome', None, ['马尾综合征']),
    C('spina_bifida', '脊柱裂', None, 'spinal_cord', ['Q05'], 'Spina bifida', '脊柱裂',
      ['脊柱裂', '二分脊椎症', '脊髓脊膜膨出']),

    # ---- 眩晕（前庭）[ICD-10 H81] ----
    C('bppv', '良性阵发性位置性眩晕', None, 'vestibular', ['H81.1'], 'Benign paroxysmal vertigo',
      '良性陣發性姿勢性眩暈', ['良性阵发性位置性眩晕', '良性位置性眩晕', '耳石症']),
    C('meniere', '梅尼埃病', None, 'vestibular', ['H81.0'], 'Ménière disease', '梅尼埃病',
      ['梅尼埃病', '美尼尔氏症', '美尼尔氏综合征', '美尼尔综合征', '梅尼尔氏症', '美尼尔', '耳水不平衡']),
    C('vestibular_neuritis', '前庭神经炎', None, 'vestibular', ['H81.2'], 'Vestibular neuronitis', '迷路炎',
      ['前庭神经炎', '前庭神经元炎', '迷路炎']),

    # ---- 发育、先天和围生期 [ICD-10 G80, Q02, P91.6, Q85] ----
    C('cerebral_palsy', '脑性瘫痪', None, 'developmental', ['G80'], 'Cerebral palsy', '腦性麻痺',
      ['脑性瘫痪', '脑瘫', '脑性麻痹', '大脑性麻痹', '脑麻痹']),
    C('microcephaly', '小头畸形', None, 'developmental', ['Q02'], 'Microcephaly', '小頭畸形', ['小头畸形', '小头症']),
    C('hie', '缺氧缺血性脑病', None, 'developmental', ['P91.6'], 'Hypoxic ischaemic encephalopathy of newborn', None,
      ['缺氧缺血性脑病', '新生儿缺氧缺血性脑病']),
    C('neurofibromatosis', '神经纤维瘤病', None, 'developmental', ['Q85'], 'Phakomatoses, not elsewhere classified', None,
      ['神经纤维瘤病', '神经纤维瘤', '结节性硬化症']),

    # ---- 自主神经 [ICD-10 G90] ----
    C('autonomic', '自主神经紊乱', None, 'autonomic', ['G90'], 'Disorders of autonomic nervous system', '自主神经紊乱',
      ['自主神经紊乱', '植物神经紊乱', '自主神经功能紊乱', '植物神经功能紊乱', '自律神经失调', '自主神经病']),

    # ---- 精神科（只用于识别“不同病”）[ICD-10 F32, F41, F48.0] ----
    C('depression', '抑郁症', None, 'psych', ['F32'], 'Depressive episode', '抑郁症',
      ['抑郁症', '忧郁症', '抑郁障碍', '抑郁发作']),
    C('anxiety', '焦虑症', None, 'psych', ['F41'], 'Other anxiety disorders', '焦虑症',
      ['焦虑症', '焦虑障碍', '焦虑性障碍', '广泛性焦虑', '惊恐障碍']),
    C('neurasthenia', '神经衰弱', None, 'psych', ['F48.0'], 'Neurasthenia', '神經衰弱', ['神经衰弱', '神经衰弱症']),
]

# 这些词只说明“和神经系统有关”，不足以确定具体疾病；匹配到它们时不贴概念标签。
GENERIC_TERMS = ['神经系统疾病', '神经病', '脑病', '脑部疾病', '神经内科', '神经外科', '神经科',
                 '神经损伤', '神经受损', '神经痛', '头痛', '头晕', '头疼']

# %%
# =============================================================================
# 关系层的来源
# =============================================================================


WD_URL = 'https://www.wikidata.org/wiki/'
MSD_URL = 'https://www.msdmanuals.cn/home/'
MSD_NEURO = 'brain-spinal-cord-and-nerve-disorders/'

CONCEPT_WD = {
    'stroke': ['Q12202'], 'cerebral_infarction': ['Q1209150'], 'cerebral_hemorrhage': ['Q1368943'],
    'sah': ['Q693442'], 'tia': ['Q593958'], 'moyamoya': ['Q1951267'], 'avm': ['Q1584110'],
    'epilepsy': ['Q41571'], 'status_epilepticus': ['Q980709'], 'febrile_seizure': ['Q127664'],
    'migraine': ['Q133823'], 'cluster_headache': ['Q166907'], 'trigeminal_neuralgia': ['Q1132120'],
    'bell_palsy': ['Q7562539'], 'hemifacial_spasm': ['Q1568480'],
    'parkinson': ['Q11085'], 'parkinsonism': ['Q1531991'], 'psp': ['Q945930'], 'dystonia': ['Q906492'],
    'huntington': ['Q190564'], 'tic_disorder': ['Q1077139'], 'tourette': ['Q191779'], 'rls': ['Q916280'],
    'wilson': ['Q117121'],
    'dementia': ['Q83030'], 'alzheimer': ['Q11081'], 'vascular_dementia': ['Q583908'], 'ftd': ['Q18592'],
    'ms': ['Q8277'], 'myelitis': ['Q551085'],
    'meningitis': ['Q48143'], 'encephalitis': ['Q199615'], 'je': ['Q738292'], 'polio': ['Q12195'],
    'cjd': ['Q49989'], 'neurocysticercosis': ['Q246068'], 'shingles': ['Q182155'],
    'peripheral_neuropathy': ['Q945238'], 'gbs': ['Q205214'], 'cmt': ['Q1052687'], 'mg': ['Q8285'],
    'als': ['Q206901'], 'sma': ['Q580290'], 'muscular_dystrophy': ['Q1137767'],
    'carpal_tunnel': ['Q332293'], 'sciatica': ['Q565276'], 'disc_herniation': ['Q697117'],
    'cervical_spondylosis': ['Q1097028'], 'intercostal_neuralgia': ['Q18658089'],
    'insomnia': ['Q1869874'], 'narcolepsy': ['Q189561'], 'sleep_apnea': ['Q213600'],
    'brain_tumor': ['Q233309'], 'glioma': ['Q1365309'], 'glioblastoma': ['Q282142'], 'meningioma': ['Q369157'],
    'pituitary_adenoma': ['Q864296'],
    'tbi': ['Q1995526'], 'concussion': ['Q326921'], 'sci': ['Q1415275'],
    'hydrocephalus': ['Q193003'], 'cerebral_edema': ['Q979168'], 'syringomyelia': ['Q1112977'],
    'spina_bifida': ['Q844717'], 'bppv': ['Q817310'], 'meniere': ['Q460167'], 'cerebral_palsy': ['Q210427'],
    'autonomic': ['Q620639'], 'depression': ['Q211022'], 'anxiety': ['Q544006'], 'neurasthenia': ['Q510236'],
}

MSD_PAGE = {
    'stroke': [MSD_NEURO + 'stroke/overview-of-stroke'],
    'epilepsy': [MSD_NEURO + 'seizure-disorders/seizure-disorders'],
    'migraine': [MSD_NEURO + 'headaches/migraines'],
    'trigeminal_neuralgia': [MSD_NEURO + 'cranial-nerve-disorders/trigeminal-neuralgia'],
    'parkinson': [MSD_NEURO + 'movement-disorders/parkinson-disease-pd'],
    'huntington': [MSD_NEURO + 'movement-disorders/huntington-disease'],
    'alzheimer': [MSD_NEURO + 'delirium-and-dementia/alzheimer-disease'],
    'ms': [MSD_NEURO + 'multiple-sclerosis-ms-and-related-disorders/multiple-sclerosis-ms'],
    'bacterial_meningitis': [MSD_NEURO + 'meningitis/acute-bacterial-meningitis'],
    'gbs': [MSD_NEURO + 'peripheral-nerve-and-related-disorders/guillain-barr%C3%A9-syndrome-gbs'],
    'mg': [MSD_NEURO + 'peripheral-nerve-and-related-disorders/myasthenia-gravis'],
    'als': [MSD_NEURO + 'peripheral-nerve-and-related-disorders/'
            'amyotrophic-lateral-sclerosis-als-and-other-motor-neuron-diseases-mnds'],
}

ZHWIKI_PAGE = {
    'stroke': '中風', 'epilepsy': '癫痫', 'migraine': '偏頭痛', 'bell_palsy': '顏面神經麻痺',
    'parkinson': '帕金森氏症', 'huntington': '亨廷頓病', 'tourette': '妥瑞氏症', 'alzheimer': '阿茲海默症',
    'ms': '多发性硬化症', 'meningitis': '脑膜炎', 'polio': '脊髓灰質炎', 'cjd': '克雅二氏病',
    'shingles': '带状疱疹', 'gbs': '格林-巴利综合征', 'mg': '重症肌無力', 'sma': '脊髓性肌肉萎縮症',
    'carpal_tunnel': '腕隧道症候群', 'sciatica': '坐骨神經痛', 'disc_herniation': '椎間盤突出症',
    'cervical_spondylosis': '椎關節病', 'insomnia': '失眠', 'narcolepsy': '發作性睡病',
    'sleep_apnea': '睡眠呼吸暂停', 'brain_tumor': '腦瘤', 'concussion': '腦震盪', 'hydrocephalus': '脑积水',
    'bppv': '良性陣發性姿勢性眩暈', 'anxiety': '焦虑症',
}


def source_urls(cid, tokens):
    '''把来源键展开成网址，写进 entities.json，方便逐条核对'''
    out = []
    for t in tokens:
        if t in ('wd', 'wd_faers'):
            out += [WD_URL + q for q in CONCEPT_WD[cid]]
        elif t == 'msd':
            out += [MSD_URL + p for p in MSD_PAGE[cid]]
        elif t == 'zhwiki':
            out.append(W + ZHWIKI_PAGE[cid])
        elif t.startswith('wd:'):
            out.append(WD_URL + t[3:])
        elif t.startswith('msd:'):
            out.append(MSD_URL + t[4:])
        elif t.startswith('zhwiki:'):
            out.append(W + t[7:])
        else:
            raise ValueError(f'unknown source token {t!r} for {cid}')
    return sorted(set(out))

# %%
# =============================================================================
# 实体层（症状 / 药物 / 治疗 / 检查 / 病因）
# =============================================================================
# 每个实体：id -> (type, 名称, Wikidata 条目或 None, 匹配用的中文词)
# 匹配词只取 Wikidata 中文标签、MSD 和中文维基原文里出现过的写法（繁体已转简体），
# 以及精筛神经科学.py 里逐词核验过的词。实体的 Wikidata 条目这次没有逐个反查，统一写 None。
# 刻意不用的写法：单字（“晕”“抖”“锂”）、商品名（美多芭、德巴金、安理申）、“安定”（歧义）、“麻痹”（会命中病名）。

ENTITY_TABLE = {
    # ---------------- 症状 / 体征 ----------------
    'headache':        ('symptom', '头痛', None, ['头痛', '头疼', '头部疼痛']),
    'dizziness':       ('symptom', '头晕/眩晕', None, ['头晕', '眩晕', '头昏', '晕眩']),
    'nausea':          ('symptom', '恶心', None, ['恶心', '反胃']),
    'vomiting':        ('symptom', '呕吐', None, ['呕吐', '喷射性呕吐']),
    'fever':           ('symptom', '发热', None, ['发热', '发烧', '高热', '低热']),
    'neck_stiffness':  ('symptom', '颈项强直', None, ['颈部僵硬', '颈项强直', '颈强直', '颈项僵直', '颈部僵直']),
    'photophobia':     ('symptom', '畏光', None, ['畏光', '怕光', '恐光']),
    'unconsciousness': ('symptom', '意识障碍', None, ['意识丧失', '失去意识', '意识障碍', '昏迷', '神志不清', '意识不清']),
    'confusion':       ('symptom', '精神错乱', None, ['精神错乱', '神智错乱', '意识模糊', '神智混乱', '思维混乱']),
    'convulsion':      ('symptom', '抽搐', None, ['抽搐', '惊厥']),
    'tremor':          ('symptom', '震颤', None, ['震颤', '手抖', '手震', '发抖']),
    'rigidity':        ('symptom', '肌强直', None, ['僵直', '肌强直', '肌肉僵硬', '肢体僵硬', '肌张力增高']),
    'bradykinesia':    ('symptom', '运动迟缓', None, ['运动迟缓', '动作缓慢', '行动迟缓', '运动减少']),
    'postural':        ('symptom', '姿势不稳', None, ['姿势不稳', '平衡障碍', '失去平衡', '平衡困难']),
    'gait':            ('symptom', '步态异常', None, ['步态异常', '走路不稳', '行走不稳', '步态不稳']),
    'weakness':        ('symptom', '肌无力', None, ['肌无力', '肌肉无力', '肢体无力', '四肢无力']),
    'muscle_atrophy':  ('symptom', '肌肉萎缩', None, ['肌肉萎缩', '肌萎缩']),
    'numbness':        ('symptom', '麻木', None, ['麻木', '发麻']),
    'paresthesia':     ('symptom', '感觉异常', None, ['感觉异常', '针刺感', '刺痛感', '感觉减退', '感觉丧失', '蚁走感']),
    'neuropathic_pain': ('symptom', '神经痛', None, ['神经痛', '放电样', '电击样', '闪电样', '刀割样']),
    'visual':          ('symptom', '视力障碍', None, ['视力障碍', '视力下降', '视物模糊', '视力模糊', '复视', '重影',
                                                  '视野缺损', '偏盲']),
    'ptosis':          ('symptom', '眼睑下垂', None, ['眼睑下垂', '眼皮下垂', '上睑下垂']),
    'dysphagia':       ('symptom', '吞咽困难', None, ['吞咽困难', '饮水呛咳', '吞咽障碍']),
    'dysarthria':      ('symptom', '言语不清', None, ['言语不清', '口齿不清', '说话不清', '构音障碍', '言语障碍', '吐字不清']),
    'aphasia':         ('symptom', '失语', None, ['失语', '语言障碍']),
    'memory_loss':     ('symptom', '记忆减退', None, ['记忆减退', '记忆力减退', '记忆力下降', '记忆障碍', '健忘', '忘事']),
    'cognitive':       ('symptom', '认知下降', None, ['认知障碍', '认知功能下降', '智力下降', '反应迟钝']),
    'hallucination':   ('symptom', '幻觉', None, ['幻觉', '幻视', '幻听']),
    'hypersomnia':     ('symptom', '嗜睡', None, ['嗜睡', '白天过度嗜睡', '日间嗜睡', '困倦']),
    'cataplexy':       ('symptom', '猝倒', None, ['猝倒']),
    'paralysis':       ('symptom', '瘫痪', None, ['瘫痪', '偏瘫', '截瘫', '四肢瘫']),
    'facial_droop':    ('symptom', '口角歪斜', None, ['口角歪斜', '嘴角歪斜', '嘴歪', '面部不对称', '嘴角下垂']),
    'chorea':          ('symptom', '不自主运动', None, ['舞蹈样动作', '不自主运动', '不自主动作']),
    'tic':             ('symptom', '抽动', None, ['抽动', '挤眼', '清嗓子']),
    'myoclonus':       ('symptom', '肌阵挛', None, ['肌阵挛']),
    'spasm':           ('symptom', '痉挛', None, ['痉挛', '抽筋']),
    'dyspnea':         ('symptom', '呼吸困难', None, ['呼吸困难', '呼吸衰竭', '呼吸无力']),
    'fatigue':         ('symptom', '疲劳', None, ['疲劳', '疲倦', '乏力', '易疲劳']),
    'incontinence':    ('symptom', '失禁', None, ['尿失禁', '小便失禁', '大小便失禁', '大小便失控']),
    'back_pain':       ('symptom', '腰痛', None, ['腰痛', '腰疼', '下背痛', '腰腿痛']),
    'radiating_pain':  ('symptom', '放射痛', None, ['放射痛', '放射性疼痛']),
    'rash':            ('symptom', '皮疹/水疱', None, ['皮疹', '水疱', '水泡', '红疹']),
    'snoring':         ('symptom', '打鼾', None, ['打鼾', '打呼噜', '鼾声']),
    'tinnitus':        ('symptom', '耳鸣', None, ['耳鸣']),
    'hearing_loss':    ('symptom', '听力下降', None, ['听力下降', '听力减退', '耳聋', '听觉障碍']),
    'papilledema':     ('symptom', '视乳头水肿', None, ['视神经乳头水肿', '视乳头水肿']),
    'icp':             ('symptom', '颅内压增高', None, ['颅内压增高', '颅压高', '颅内高压']),
    'weight_loss':     ('symptom', '体重减轻', None, ['体重减轻', '体重下降', '消瘦']),
    'kf_ring':         ('symptom', '角膜色素环', None, ['角膜色素环']),

    # ---------------- 药物 ----------------
    'levodopa':        ('drug', '左旋多巴', None, ['左旋多巴', 'L-多巴']),
    'carbidopa':       ('drug', '卡比多巴', None, ['卡比多巴']),
    'pramipexole':     ('drug', '普拉克索', None, ['普拉克索']),
    'ropinirole':      ('drug', '罗匹尼罗', None, ['罗匹尼罗']),
    'rotigotine':      ('drug', '罗替戈汀', None, ['罗替戈汀']),
    'selegiline':      ('drug', '司来吉兰', None, ['司来吉兰']),
    'rasagiline':      ('drug', '雷沙吉兰', None, ['雷沙吉兰']),
    'entacapone':      ('drug', '恩他卡朋', None, ['恩他卡朋']),
    'tolcapone':       ('drug', '托卡朋', None, ['托卡朋']),
    'amantadine':      ('drug', '金刚烷胺', None, ['金刚烷胺']),
    'trihexyphenidyl': ('drug', '苯海索', None, ['苯海索']),
    'bromocriptine':   ('drug', '溴隐亭', None, ['溴隐亭']),
    'apomorphine':     ('drug', '阿扑吗啡', None, ['阿扑吗啡']),
    'dopamine_agonist': ('drug', '多巴胺激动剂', None, ['多巴胺激动剂', '多巴胺受体激动剂']),
    'carbamazepine':   ('drug', '卡马西平', None, ['卡马西平']),
    'oxcarbazepine':   ('drug', '奥卡西平', None, ['奥卡西平']),
    'valproate':       ('drug', '丙戊酸', None, ['丙戊酸', '丙戊酸钠']),
    'phenytoin':       ('drug', '苯妥英', None, ['苯妥英', '苯妥英钠']),
    'fosphenytoin':    ('drug', '磷苯妥英', None, ['磷苯妥英']),
    'levetiracetam':   ('drug', '左乙拉西坦', None, ['左乙拉西坦']),
    'lamotrigine':     ('drug', '拉莫三嗪', None, ['拉莫三嗪']),
    'topiramate':      ('drug', '托吡酯', None, ['托吡酯']),
    'phenobarbital':   ('drug', '苯巴比妥', None, ['苯巴比妥']),
    'lacosamide':      ('drug', '拉科酰胺', None, ['拉科酰胺']),
    'perampanel':      ('drug', '吡仑帕奈', None, ['吡仑帕奈']),
    'zonisamide':      ('drug', '唑尼沙胺', None, ['唑尼沙胺']),
    'clobazam':        ('drug', '氯巴占', None, ['氯巴占']),
    'ethosuximide':    ('drug', '乙琥胺', None, ['乙琥胺']),
    'gabapentin':      ('drug', '加巴喷丁', None, ['加巴喷丁']),
    'pregabalin':      ('drug', '普瑞巴林', None, ['普瑞巴林']),
    'clonazepam':      ('drug', '氯硝西泮', None, ['氯硝西泮']),
    'lorazepam':       ('drug', '劳拉西泮', None, ['劳拉西泮']),
    'diazepam':        ('drug', '地西泮', None, ['地西泮']),
    'midazolam':       ('drug', '咪达唑仑', None, ['咪达唑仑']),
    'benzodiazepine':  ('drug', '苯二氮卓类', None, ['苯二氮卓', '苯二氮䓬']),
    'triptan':         ('drug', '曲坦类', None, ['曲坦', '翠普登']),
    'sumatriptan':     ('drug', '舒马曲坦', None, ['舒马曲坦', '舒马普坦']),
    'zolmitriptan':    ('drug', '佐米曲普坦', None, ['佐米曲普坦']),
    'rizatriptan':     ('drug', '利扎曲坦', None, ['利扎曲坦']),
    'ergotamine':      ('drug', '麦角胺', None, ['麦角胺']),
    'ibuprofen':       ('drug', '布洛芬', None, ['布洛芬']),
    'acetaminophen':   ('drug', '对乙酰氨基酚', None, ['对乙酰氨基酚', '扑热息痛']),
    'naproxen':        ('drug', '萘普生', None, ['萘普生']),
    'nsaid':           ('drug', '非甾体抗炎药', None, ['非甾体抗炎药', '非甾体类抗炎药']),
    'aspirin':         ('drug', '阿司匹林', None, ['阿司匹林', '阿斯匹林', '阿斯匹灵']),
    'propranolol':     ('drug', '普萘洛尔', None, ['普萘洛尔']),
    'amitriptyline':   ('drug', '阿米替林', None, ['阿米替林']),
    'botulinum':       ('drug', '肉毒毒素', None, ['肉毒毒素', '肉毒杆菌毒素', '肉毒素']),
    'verapamil':       ('drug', '维拉帕米', None, ['维拉帕米']),
    'lithium':         ('drug', '锂盐', None, ['碳酸锂', '锂盐']),
    'clopidogrel':     ('drug', '氯吡格雷', None, ['氯吡格雷']),
    'cilostazol':      ('drug', '西洛他唑', None, ['西洛他唑']),
    'antiplatelet':    ('drug', '抗血小板药', None, ['抗血小板']),
    'warfarin':        ('drug', '华法林', None, ['华法林']),
    'rivaroxaban':     ('drug', '利伐沙班', None, ['利伐沙班']),
    'dabigatran':      ('drug', '达比加群', None, ['达比加群']),
    'anticoagulant':   ('drug', '抗凝药', None, ['抗凝药', '抗凝血药', '抗凝治疗']),
    'statin':          ('drug', '他汀类', None, ['他汀']),
    'alteplase':       ('drug', '阿替普酶', None, ['阿替普酶', 'RT-PA']),
    'tenecteplase':    ('drug', '替奈普酶', None, ['替奈普酶']),
    'edaravone':       ('drug', '依达拉奉', None, ['依达拉奉']),
    'nimodipine':      ('drug', '尼莫地平', None, ['尼莫地平']),
    'tranexamic_acid': ('drug', '氨甲环酸', None, ['氨甲环酸', '传明酸']),
    'mannitol':        ('drug', '甘露醇', None, ['甘露醇']),
    'furosemide':      ('drug', '呋塞米', None, ['呋塞米']),
    'vinpocetine':     ('drug', '长春西汀', None, ['长春西汀']),
    'donepezil':       ('drug', '多奈哌齐', None, ['多奈哌齐']),
    'galantamine':     ('drug', '加兰他敏', None, ['加兰他敏']),
    'memantine':       ('drug', '美金刚', None, ['美金刚']),
    'rivastigmine':    ('drug', 'rivastigmine', None, ['RIVASTIGMINE']),
    'lecanemab':       ('drug', '仑卡奈单抗', None, ['仑卡奈单抗']),
    'tetrabenazine':   ('drug', '丁苯那嗪', None, ['丁苯那嗪']),
    'deutetrabenazine': ('drug', '氘丁苯那嗪', None, ['氘丁苯那嗪']),
    'pimozide':        ('drug', '匹莫齐特', None, ['匹莫齐特']),
    'perphenazine':    ('drug', '奋乃静', None, ['奋乃静']),
    'haloperidol':     ('drug', '氟哌啶醇', None, ['氟哌啶醇']),
    'aripiprazole':    ('drug', '阿立哌唑', None, ['阿立哌唑']),
    'risperidone':     ('drug', '利培酮', None, ['利培酮']),
    'quetiapine':      ('drug', '喹硫平', None, ['喹硫平']),
    'clozapine':       ('drug', '氯氮平', None, ['氯氮平']),
    'clonidine':       ('drug', '可乐定', None, ['可乐定']),
    'antipsychotic':   ('drug', '抗精神病药', None, ['抗精神病药']),
    'antidepressant':  ('drug', '抗抑郁药', None, ['抗抑郁药', '选择性5-羟色胺再吸收抑制剂']),
    'interferon':      ('drug', '干扰素', None, ['干扰素']),
    'fingolimod':      ('drug', '芬戈莫德', None, ['芬戈莫德']),
    'natalizumab':     ('drug', '那他珠单抗', None, ['那他珠单抗']),
    'teriflunomide':   ('drug', '特立氟胺', None, ['特立氟胺']),
    'cladribine':      ('drug', '克拉屈滨', None, ['克拉屈滨']),
    'dimethyl_fumarate': ('drug', '富马酸二甲酯', None, ['富马酸二甲酯']),
    'ocrelizumab':     ('drug', '奥瑞珠单抗', None, ['奥瑞珠单抗', '奥美珠单抗']),
    'ofatumumab':      ('drug', '奥法妥木单抗', None, ['奥法妥木单抗']),
    'alemtuzumab':     ('drug', '阿仑单抗', None, ['阿仑单抗']),
    'mitoxantrone':    ('drug', '米托蒽醌', None, ['米托蒽醌']),
    'dalfampridine':   ('drug', '4-氨基吡啶', None, ['4-氨基吡啶']),
    'methylprednisolone': ('drug', '甲泼尼龙', None, ['甲泼尼龙', '甲基强的松龙']),
    'prednisone':      ('drug', '泼尼松', None, ['泼尼松', '强的松']),
    'corticosteroid':  ('drug', '糖皮质激素', None, ['糖皮质激素', '皮质类固醇', '类固醇']),
    'dexamethasone':   ('drug', '地塞米松', None, ['地塞米松']),
    'baclofen':        ('drug', '巴氯芬', None, ['巴氯芬']),
    'dantrolene':      ('drug', '丹曲林', None, ['丹曲林']),
    'tizanidine':      ('drug', '替扎尼定', None, ['替扎尼定']),
    'modafinil':       ('drug', '莫达非尼', None, ['莫达非尼']),
    'methylphenidate': ('drug', '哌甲酯', None, ['哌甲酯']),
    'sodium_oxybate':  ('drug', 'γ-羟基丁酸钠', None, ['羟基丁酸钠', '羟丁酸钠']),
    'pyridostigmine':  ('drug', '溴吡斯的明', None, ['溴吡斯的明', '吡斯的明']),
    'neostigmine':     ('drug', '新斯的明', None, ['新斯的明']),
    'azathioprine':    ('drug', '硫唑嘌呤', None, ['硫唑嘌呤']),
    'cyclosporine':    ('drug', '环孢素', None, ['环孢素']),
    'tacrolimus':      ('drug', '他克莫司', None, ['他克莫司']),
    'mycophenolate':   ('drug', '吗替麦考酚酯', None, ['吗替麦考酚酯', '霉酚酸酯']),
    'eculizumab':      ('drug', '依库珠单抗', None, ['依库珠单抗']),
    'rituximab':       ('drug', '利妥昔单抗', None, ['利妥昔单抗']),
    'ivig':            ('drug', '免疫球蛋白', None, ['免疫球蛋白']),
    'riluzole':        ('drug', '利鲁唑', None, ['利鲁唑']),
    'nusinersen':      ('drug', '诺西那生', None, ['诺西那生']),
    'risdiplam':       ('drug', '利司扑兰', None, ['利司扑兰']),
    'ceftriaxone':     ('drug', '头孢曲松', None, ['头孢曲松']),
    'cefotaxime':      ('drug', '头孢噻肟', None, ['头孢噻肟']),
    'vancomycin':      ('drug', '万古霉素', None, ['万古霉素']),
    'ampicillin':      ('drug', '氨苄西林', None, ['氨苄西林', '氨苄青霉素']),
    'antibiotic':      ('drug', '抗生素', None, ['抗生素']),
    'acyclovir':       ('drug', '阿昔洛韦', None, ['阿昔洛韦']),
    'valacyclovir':    ('drug', '伐昔洛韦', None, ['伐昔洛韦']),
    'famciclovir':     ('drug', '泛昔洛韦', None, ['泛昔洛韦']),
    'capsaicin':       ('drug', '辣椒素', None, ['辣椒素']),
    'duloxetine':      ('drug', '度洛西汀', None, ['度洛西汀']),
    'albendazole':     ('drug', '阿苯达唑', None, ['阿苯达唑']),
    'praziquantel':    ('drug', '吡喹酮', None, ['吡喹酮']),
    'temozolomide':    ('drug', '替莫唑胺', None, ['替莫唑胺']),
    'zolpidem':        ('drug', '唑吡坦', None, ['唑吡坦']),
    'eszopiclone':     ('drug', '艾司佐匹克隆', None, ['艾司佐匹克隆', '右佐匹克隆']),
    'alprazolam':      ('drug', '阿普唑仑', None, ['阿普唑仑']),
    'estazolam':       ('drug', '艾司唑仑', None, ['艾司唑仑']),
    'trazodone':       ('drug', '曲唑酮', None, ['曲唑酮']),
    'mirtazapine':     ('drug', '米氮平', None, ['米氮平']),
    'suvorexant':      ('drug', '苏沃雷生', None, ['苏沃雷生']),
    'sertraline':      ('drug', '舍曲林', None, ['舍曲林']),
    'paroxetine':      ('drug', '帕罗西汀', None, ['帕罗西汀']),
    'escitalopram':    ('drug', '艾司西酞普兰', None, ['艾司西酞普兰']),
    'venlafaxine':     ('drug', '文拉法辛', None, ['文拉法辛']),
    'buspirone':       ('drug', '丁螺环酮', None, ['丁螺环酮']),
    'penicillamine':   ('drug', '青霉胺', None, ['青霉胺']),
    'trientine':       ('drug', '曲恩汀', None, ['曲恩汀', '三乙烯四胺']),
    'vitamin_b':       ('drug', '维生素B', None, ['维生素B', '维生素B12', '氰钴胺']),

    # ---------------- 治疗手段 ----------------
    'thrombolysis':    ('treatment', '溶栓', None, ['溶栓', '静脉溶栓', '血栓溶解']),
    'thrombectomy':    ('treatment', '取栓', None, ['取栓', '机械取栓', '血栓切除', '血栓移除']),
    'stent':           ('treatment', '支架/血管成形术', None, ['支架', '血管成形术']),
    'endarterectomy':  ('treatment', '颈动脉内膜切除术', None, ['颈动脉内膜切除']),
    'clipping':        ('treatment', '动脉瘤夹闭', None, ['夹闭']),
    'coiling':         ('treatment', '弹簧圈栓塞', None, ['弹簧圈']),
    'dbs':             ('treatment', '脑深部电刺激', None, ['脑深部电刺激', '深部脑刺激', 'DBS']),
    'plasmapheresis':  ('treatment', '血浆置换', None, ['血浆置换']),
    'thymectomy':      ('treatment', '胸腺切除', None, ['胸腺切除']),
    'cpap':            ('treatment', '持续正压通气', None, ['CPAP', '持续正压通气', '持续正压呼吸机']),
    'ventilator':      ('treatment', '机械通气', None, ['机械通气', '呼吸机', '呼吸支持']),
    'oral_appliance':  ('treatment', '止鼾器', None, ['止鼾器', '口腔矫治器']),
    'rehab':           ('treatment', '康复治疗', None, ['康复训练', '康复治疗', '康复锻炼', '物理治疗', '理疗', '复健']),
    'cbt':             ('treatment', '认知行为疗法', None, ['认知行为疗法', '认知行为治疗']),
    'psychotherapy':   ('treatment', '心理治疗', None, ['心理治疗']),
    'nerve_procedure': ('treatment', '射频/微血管减压', None, ['射频', '微血管减压', '神经阻滞']),
    'shunt':           ('treatment', '分流术', None, ['分流术', '脑室腹腔分流', '分流']),
    'radiotherapy':    ('treatment', '放疗', None, ['放疗', '放射治疗', '伽马刀']),
    'chemotherapy':    ('treatment', '化疗', None, ['化疗', '化学治疗']),
    'surgery':         ('treatment', '手术治疗', None, ['手术治疗', '手术切除', '开颅']),
    'traction_massage': ('treatment', '牵引/推拿', None, ['牵引', '推拿', '按摩']),
    'acupuncture':     ('treatment', '针灸', None, ['针灸', '针刺']),
    'repositioning':   ('treatment', '耳石复位', None, ['耳石复位', '手法复位', 'EPLEY']),
    'vaccine':         ('treatment', '疫苗接种', None, ['疫苗', '接种']),
    'weight_reduction': ('treatment', '减重', None, ['减肥', '减重', '减轻体重']),
    'oxygen':          ('treatment', '吸氧', None, ['吸氧', '氧疗', '氧气疗法']),

    # ---------------- 检查 ----------------
    'mri':             ('exam', '磁共振', None, ['MRI', '磁共振', '核磁共振', '核磁']),
    'ct':              ('exam', 'CT', None, ['CT', '计算机断层', '电脑断层']),
    'eeg':             ('exam', '脑电图', None, ['脑电图', 'EEG']),
    'emg':             ('exam', '肌电图', None, ['肌电图', 'EMG']),
    'ncv':             ('exam', '神经传导检查', None, ['神经传导']),
    'lumbar_puncture': ('exam', '腰椎穿刺', None, ['腰椎穿刺', '腰穿', '脊椎穿刺', '脑脊液检查', '脑脊液']),
    'blood_test':      ('exam', '血液检查', None, ['血液检查', '验血', '抽血', '血常规']),
    'gene_test':       ('exam', '基因检测', None, ['基因检测', '基因测试', '基因检查']),
    'angiography':     ('exam', '血管造影', None, ['血管造影', 'DSA', 'CTA', 'MRA']),
    'ultrasound':      ('exam', '超声', None, ['超声', 'B超', '彩超', '多普勒']),
    'pet':             ('exam', 'PET', None, ['PET', '正电子']),
    'spect':           ('exam', 'SPECT', None, ['SPECT']),
    'neuropsych':      ('exam', '心理/认知测验', None, ['心理测验', '量表', '认知评估', '神经心理']),
    'antibody_test':   ('exam', '抗体检测', None, ['抗体检测', '抗体']),
    'ice_test':        ('exam', '冰敷/疲劳试验', None, ['冰敷试验', '新斯的明试验', '疲劳试验']),
    'xray':            ('exam', 'X 线', None, ['X线', 'X光', 'X射线', '拍片']),
    'dix_hallpike':    ('exam', 'Dix-Hallpike 试验', None, ['DIX-HALLPIKE', '变位试验']),
    'audiometry':      ('exam', '听力检查', None, ['听力检查', '听力测试', '电测听']),
    'pcr':             ('exam', 'PCR', None, ['PCR', '聚合酶链式反应']),
    'elisa':           ('exam', 'ELISA', None, ['ELISA', '酶联免疫']),
    'biopsy':          ('exam', '活检', None, ['活检', '病理检查']),

    # ---------------- 病因 / 危险因素 ----------------
    'hypertension':    ('cause', '高血压', None, ['高血压']),
    'diabetes':        ('cause', '糖尿病', None, ['糖尿病']),
    'hyperlipidemia':  ('cause', '血脂异常', None, ['高脂血症', '高血脂', '血脂异常', '高胆固醇']),
    'smoking':         ('cause', '吸烟', None, ['吸烟', '抽烟']),
    'atherosclerosis': ('cause', '动脉粥样硬化', None, ['动脉粥样硬化', '动脉硬化']),
    'afib':            ('cause', '心房颤动', None, ['心房颤动', '房颤']),
    'genetic':         ('cause', '遗传', None, ['遗传', '基因突变', '家族史']),
    'autoimmune':      ('cause', '自身免疫', None, ['自身免疫', '免疫系统攻击']),
    'bacteria':        ('cause', '细菌感染', None, ['细菌感染', '肺炎链球菌', '脑膜炎球菌', '脑膜炎双球菌',
                                                   '流感嗜血杆菌', '李斯特菌']),
    'virus':           ('cause', '病毒感染', None, ['病毒感染', 'EB病毒']),
    'vzv':             ('cause', '水痘-带状疱疹病毒', None, ['水痘-带状疱疹病毒', '带状疱疹病毒', '水痘']),
    'je_virus':        ('cause', '乙脑病毒', None, ['乙脑病毒', '日本脑炎病毒']),
    'poliovirus':      ('cause', '脊髓灰质炎病毒', None, ['脊髓灰质炎病毒']),
    'prion':           ('cause', '朊病毒', None, ['朊病毒', '朊毒体']),
    'head_trauma':     ('cause', '头部外伤', None, ['头部外伤', '头部损伤', '头部受伤', '头颅外伤']),
    'accident':        ('cause', '车祸/跌倒', None, ['交通事故', '车祸', '跌倒', '坠落']),
    'stress':          ('cause', '精神压力', None, ['压力', '精神紧张', '情绪激动']),
    'sleep_lack':      ('cause', '睡眠不足', None, ['睡眠不足', '熬夜', '睡眠缺乏']),
    'alcohol':         ('cause', '饮酒', None, ['饮酒', '喝酒', '酗酒']),
    'obesity':         ('cause', '肥胖', None, ['肥胖']),
    'thymoma':         ('cause', '胸腺瘤', None, ['胸腺瘤', '胸腺增生']),
    'hypoxia':         ('cause', '缺氧', None, ['缺氧']),
    'campylobacter':   ('cause', '弯曲杆菌感染', None, ['弯曲杆菌']),
}

ENTITIES = [{'id': k, 'type': t, 'name': n, 'wikidata': q, 'terms': terms}
            for k, (t, n, q, terms) in ENTITY_TABLE.items()]

# %%
# =============================================================================
# 疾病 -> 实体 关系（每条带来源键）
# =============================================================================
# 只写到最具体的概念上；父概念（脑卒中、痴呆…）通过上下位关系自动带上子概念的资料。
# msd / zhwiki：该词出现在对应页面的症状、诊断、治疗、病因各节原文里（临时/msd_extract.txt、wiki_extract*.txt）。
# 没有写进来的概念是这次没找到可靠来源，不是没有症状。

REL = {
    'stroke': {
        'headache': 'msd', 'dizziness': 'msd', 'nausea': 'msd', 'vomiting': 'msd', 'unconsciousness': 'msd',
        'confusion': 'msd', 'postural': 'msd zhwiki', 'weakness': 'msd zhwiki', 'numbness': 'zhwiki',
        'paresthesia': 'msd', 'visual': 'msd zhwiki', 'dysphagia': 'msd', 'aphasia': 'msd zhwiki',
        'paralysis': 'msd zhwiki', 'facial_droop': 'zhwiki',
        'aspirin': 'msd zhwiki', 'clopidogrel': 'msd zhwiki', 'warfarin': 'msd zhwiki', 'statin': 'zhwiki',
        'alteplase': 'zhwiki', 'tenecteplase': 'zhwiki', 'edaravone': 'zhwiki', 'antidepressant': 'msd',
        'thrombolysis': 'msd zhwiki', 'thrombectomy': 'msd zhwiki', 'stent': 'msd zhwiki',
        'endarterectomy': 'zhwiki', 'rehab': 'msd zhwiki', 'surgery': 'msd zhwiki',
        'ct': 'msd zhwiki', 'mri': 'msd zhwiki', 'blood_test': 'msd zhwiki', 'angiography': 'msd zhwiki',
        'ultrasound': 'msd zhwiki',
        'hypertension': 'msd zhwiki', 'diabetes': 'msd zhwiki', 'hyperlipidemia': 'msd', 'smoking': 'msd zhwiki',
        'atherosclerosis': 'msd zhwiki', 'afib': 'msd zhwiki', 'obesity': 'msd zhwiki', 'alcohol': 'msd zhwiki',
    },
    'cerebral_infarction': {
        'cilostazol': 'wd_faers', 'rivaroxaban': 'wd_faers', 'statin': 'wd_faers', 'clopidogrel': 'wd_faers',
        'alteplase': 'wd_faers',
    },
    'cerebral_hemorrhage': {'statin': 'wd'},
    'sah': {
        'vomiting': 'wd', 'photophobia': 'wd', 'nimodipine': 'wd', 'tranexamic_acid': 'wd',
        'clipping': 'wd', 'coiling': 'wd', 'ct': 'wd',
    },
    'tia': {
        'rivaroxaban': 'wd_faers', 'warfarin': 'wd_faers', 'statin': 'wd_faers', 'clopidogrel': 'wd',
        'dabigatran': 'wd_faers',
    },
    'vascular_dementia': {
        'donepezil': 'wd_faers', 'memantine': 'wd_faers', 'vinpocetine': 'wd', 'anticoagulant': 'wd',
        'statin': 'wd', 'mri': 'wd', 'neuropsych': 'wd', 'hypertension': 'wd', 'diabetes': 'wd',
    },
    'epilepsy': {
        'convulsion': 'msd zhwiki', 'unconsciousness': 'msd zhwiki', 'confusion': 'msd',
        'incontinence': 'msd zhwiki', 'paresthesia': 'msd', 'numbness': 'msd',
        'carbamazepine': 'wd msd zhwiki', 'oxcarbazepine': 'wd_faers msd zhwiki', 'valproate': 'msd zhwiki',
        'phenytoin': 'wd_faers msd zhwiki', 'levetiracetam': 'wd_faers msd', 'lamotrigine': 'wd_faers msd',
        'topiramate': 'wd msd', 'phenobarbital': 'wd_faers msd', 'lacosamide': 'wd_faers',
        'perampanel': 'wd_faers', 'zonisamide': 'wd_faers', 'clobazam': 'wd_faers', 'ethosuximide': 'wd_faers',
        'gabapentin': 'wd_faers msd', 'pregabalin': 'wd_faers msd', 'clonazepam': 'wd_faers msd',
        'lorazepam': 'msd', 'diazepam': 'wd_faers', 'benzodiazepine': 'msd', 'ivig': 'msd', 'vitamin_b': 'msd',
        'surgery': 'msd',
        'eeg': 'msd zhwiki', 'mri': 'msd', 'ct': 'msd', 'blood_test': 'msd zhwiki',
        'genetic': 'msd zhwiki', 'head_trauma': 'wd msd', 'sleep_lack': 'msd zhwiki', 'alcohol': 'msd zhwiki',
        'stress': 'msd zhwiki', 'hypoxia': 'msd',
    },
    'status_epilepticus': {
        'fosphenytoin': 'wd', 'lorazepam': 'wd', 'phenobarbital': 'wd', 'midazolam': 'wd', 'diazepam': 'wd',
    },
    'febrile_seizure': {'phenobarbital': 'wd', 'diazepam': 'wd', 'phenytoin': 'wd'},
    'migraine': {
        'headache': 'wd msd zhwiki', 'nausea': 'wd msd zhwiki', 'vomiting': 'wd msd zhwiki',
        'photophobia': 'wd msd zhwiki', 'dizziness': 'msd zhwiki', 'numbness': 'zhwiki', 'fatigue': 'zhwiki',
        'triptan': 'wd msd zhwiki', 'sumatriptan': 'wd_faers zhwiki', 'zolmitriptan': 'wd_faers',
        'rizatriptan': 'wd_faers', 'ergotamine': 'wd_faers', 'ibuprofen': 'wd_faers msd zhwiki',
        'acetaminophen': 'wd msd zhwiki', 'naproxen': 'wd_faers', 'aspirin': 'msd zhwiki',
        'propranolol': 'wd_faers msd zhwiki', 'amitriptyline': 'wd_faers', 'topiramate': 'wd_faers msd zhwiki',
        'valproate': 'msd zhwiki', 'gabapentin': 'wd_faers zhwiki', 'botulinum': 'wd_faers msd zhwiki',
        'antidepressant': 'wd msd', 'duloxetine': 'wd_faers', 'venlafaxine': 'wd_faers',
        'lamotrigine': 'wd_faers', 'pregabalin': 'wd_faers', 'zonisamide': 'wd_faers',
        'dexamethasone': 'zhwiki', 'vitamin_b': 'zhwiki',
        'traction_massage': 'zhwiki', 'surgery': 'zhwiki',
        'ct': 'msd', 'mri': 'msd', 'lumbar_puncture': 'msd',
        'stress': 'wd msd zhwiki', 'sleep_lack': 'msd',
    },
    'cluster_headache': {
        'headache': 'wd', 'ptosis': 'wd', 'photophobia': 'wd',
        'ergotamine': 'wd', 'triptan': 'wd', 'lithium': 'wd', 'verapamil': 'wd', 'corticosteroid': 'wd',
        'oxygen': 'wd',
    },
    'trigeminal_neuralgia': {
        'neuropathic_pain': 'msd',
        'carbamazepine': 'wd msd', 'oxcarbazepine': 'wd_faers msd', 'baclofen': 'wd msd',
        'gabapentin': 'wd_faers msd', 'pregabalin': 'wd_faers', 'lamotrigine': 'wd_faers msd',
        'phenytoin': 'msd', 'tizanidine': 'wd', 'antidepressant': 'msd', 'botulinum': 'msd',
        'nerve_procedure': 'msd', 'surgery': 'msd',
        'mri': 'msd',
    },
    'bell_palsy': {'rehab': 'zhwiki', 'mri': 'zhwiki', 'ct': 'zhwiki', 'emg': 'zhwiki', 'ncv': 'zhwiki'},
    'parkinson': {
        'tremor': 'wd msd zhwiki', 'rigidity': 'msd zhwiki', 'bradykinesia': 'wd msd zhwiki', 'postural': 'msd',
        'gait': 'wd msd zhwiki', 'hallucination': 'msd zhwiki', 'hypersomnia': 'msd zhwiki',
        'dysphagia': 'msd zhwiki', 'incontinence': 'msd zhwiki', 'numbness': 'zhwiki', 'paresthesia': 'zhwiki',
        'dizziness': 'msd',
        'levodopa': 'wd msd', 'carbidopa': 'wd msd', 'pramipexole': 'wd msd zhwiki', 'ropinirole': 'wd msd',
        'rotigotine': 'wd', 'selegiline': 'wd msd zhwiki', 'rasagiline': 'wd_faers msd zhwiki',
        'entacapone': 'wd msd zhwiki', 'tolcapone': 'wd', 'amantadine': 'wd msd zhwiki',
        'trihexyphenidyl': 'wd msd', 'bromocriptine': 'wd', 'apomorphine': 'wd_faers',
        'dopamine_agonist': 'zhwiki', 'clozapine': 'wd_faers', 'quetiapine': 'msd zhwiki', 'rivastigmine': 'wd',
        'zonisamide': 'wd_faers', 'antipsychotic': 'msd zhwiki', 'antidepressant': 'msd',
        'dbs': 'wd msd', 'surgery': 'wd msd zhwiki', 'rehab': 'msd zhwiki', 'nerve_procedure': 'msd',
        'mri': 'wd msd zhwiki', 'ct': 'msd zhwiki', 'pet': 'wd', 'spect': 'wd',
        'genetic': 'msd zhwiki', 'head_trauma': 'msd',
    },
    'parkinsonism': {'levodopa': 'wd_faers', 'carbidopa': 'wd_faers', 'ropinirole': 'wd_faers'},
    'dystonia': {'botulinum': 'wd'},
    'huntington': {
        'chorea': 'wd msd zhwiki', 'gait': 'msd zhwiki', 'dysphagia': 'zhwiki', 'weight_loss': 'wd',
        'tetrabenazine': 'wd msd zhwiki', 'deutetrabenazine': 'msd', 'pimozide': 'wd', 'perphenazine': 'wd',
        'baclofen': 'wd', 'antipsychotic': 'msd', 'clonazepam': 'msd',
        'gene_test': 'wd msd', 'mri': 'msd', 'ct': 'msd',
        'genetic': 'wd msd zhwiki',
    },
    'tic_disorder': {'tic': 'wd'},
    'tourette': {
        'tic': 'wd zhwiki',
        'tetrabenazine': 'wd_faers', 'pimozide': 'wd', 'haloperidol': 'wd', 'aripiprazole': 'wd_faers',
        'risperidone': 'wd', 'clonidine': 'wd', 'perphenazine': 'wd', 'trihexyphenidyl': 'zhwiki',
        'antipsychotic': 'zhwiki', 'cbt': 'zhwiki', 'surgery': 'zhwiki',
    },
    'rls': {
        'rotigotine': 'wd_faers', 'clonazepam': 'wd', 'gabapentin': 'wd_faers', 'pregabalin': 'wd_faers',
        'pramipexole': 'wd_faers', 'ropinirole': 'wd_faers', 'carbamazepine': 'wd',
    },
    'wilson': {'kf_ring': 'wd', 'penicillamine': 'wd_faers', 'trientine': 'wd', 'genetic': 'wd'},
    'dementia': {
        'donepezil': 'wd', 'galantamine': 'wd', 'memantine': 'wd_faers', 'rivastigmine': 'wd',
        'quetiapine': 'wd_faers', 'risperidone': 'wd', 'aripiprazole': 'wd_faers',
    },
    'alzheimer': {
        'memory_loss': 'wd msd', 'confusion': 'msd', 'hallucination': 'msd zhwiki', 'incontinence': 'msd zhwiki',
        'donepezil': 'wd_faers msd', 'galantamine': 'wd msd zhwiki', 'memantine': 'wd msd zhwiki',
        'rivastigmine': 'wd', 'lecanemab': 'msd zhwiki', 'antipsychotic': 'zhwiki', 'risperidone': 'wd_faers',
        'quetiapine': 'wd_faers', 'aripiprazole': 'wd_faers',
        'mri': 'wd msd zhwiki', 'ct': 'msd zhwiki', 'gene_test': 'wd msd', 'neuropsych': 'wd msd zhwiki',
        'blood_test': 'msd zhwiki', 'lumbar_puncture': 'msd zhwiki',
        'genetic': 'msd zhwiki', 'hypertension': 'msd zhwiki', 'diabetes': 'msd zhwiki', 'hyperlipidemia': 'msd',
        'smoking': 'msd',
    },
    'ftd': {'aphasia': 'wd', 'memory_loss': 'wd', 'antipsychotic': 'wd', 'antidepressant': 'wd', 'mri': 'wd'},
    'ms': {
        'visual': 'msd zhwiki', 'numbness': 'msd zhwiki', 'paresthesia': 'zhwiki', 'weakness': 'msd zhwiki',
        'fatigue': 'msd zhwiki', 'tremor': 'msd zhwiki', 'dizziness': 'msd', 'incontinence': 'msd',
        'neuropathic_pain': 'wd msd zhwiki', 'memory_loss': 'msd', 'rigidity': 'msd', 'postural': 'msd',
        'interferon': 'wd msd zhwiki', 'fingolimod': 'wd msd zhwiki', 'natalizumab': 'wd_faers msd',
        'teriflunomide': 'wd', 'cladribine': 'wd', 'dimethyl_fumarate': 'wd_faers', 'ocrelizumab': 'wd',
        'ofatumumab': 'wd', 'alemtuzumab': 'wd_faers', 'mitoxantrone': 'wd', 'dalfampridine': 'wd',
        'rituximab': 'wd_faers', 'azathioprine': 'wd', 'methylprednisolone': 'wd_faers msd',
        'corticosteroid': 'msd', 'baclofen': 'wd msd', 'dantrolene': 'wd', 'gabapentin': 'wd_faers msd',
        'pregabalin': 'wd_faers msd', 'carbamazepine': 'msd', 'modafinil': 'wd_faers', 'clonazepam': 'msd',
        'antidepressant': 'msd',
        'plasmapheresis': 'msd zhwiki', 'rehab': 'msd zhwiki',
        'mri': 'wd msd zhwiki', 'lumbar_puncture': 'wd msd zhwiki', 'pet': 'wd',
        'virus': 'msd zhwiki', 'smoking': 'msd zhwiki', 'genetic': 'msd zhwiki', 'autoimmune': 'msd zhwiki',
    },
    'meningitis': {
        'headache': 'wd zhwiki', 'photophobia': 'wd zhwiki', 'fever': 'wd zhwiki', 'nausea': 'wd',
        'vomiting': 'wd zhwiki', 'neck_stiffness': 'zhwiki', 'unconsciousness': 'zhwiki', 'rash': 'zhwiki',
        'ceftriaxone': 'wd zhwiki', 'cefotaxime': 'zhwiki', 'vancomycin': 'zhwiki', 'ampicillin': 'zhwiki',
        'antibiotic': 'wd zhwiki', 'dexamethasone': 'zhwiki', 'mannitol': 'zhwiki', 'acyclovir': 'zhwiki',
        'ivig': 'zhwiki', 'vaccine': 'zhwiki', 'ventilator': 'zhwiki', 'shunt': 'zhwiki',
        'lumbar_puncture': 'wd zhwiki', 'pcr': 'wd', 'ct': 'zhwiki', 'mri': 'zhwiki',
        'bacteria': 'wd zhwiki', 'virus': 'wd zhwiki',
    },
    'bacterial_meningitis': {
        'fever': 'msd', 'headache': 'msd', 'neck_stiffness': 'msd', 'photophobia': 'msd', 'vomiting': 'msd',
        'unconsciousness': 'msd', 'confusion': 'msd', 'convulsion': 'msd', 'hypersomnia': 'msd', 'rash': 'msd',
        'paralysis': 'msd', 'icp': 'msd',
        'ceftriaxone': 'msd', 'dexamethasone': 'msd', 'antibiotic': 'msd', 'acetaminophen': 'msd',
        'vaccine': 'msd',
        'lumbar_puncture': 'msd', 'ct': 'msd', 'mri': 'msd',
        'bacteria': 'msd',
    },
    'encephalitis': {'autoimmune': 'wd'},
    'je': {
        'confusion': 'wd', 'unconsciousness': 'wd', 'fever': 'wd', 'headache': 'wd', 'fatigue': 'wd',
        'vomiting': 'wd',
        'mannitol': 'wd', 'furosemide': 'wd', 'neostigmine': 'wd',
        'elisa': 'wd', 'ct': 'wd', 'lumbar_puncture': 'wd', 'eeg': 'wd', 'pcr': 'wd', 'mri': 'wd',
        'je_virus': 'wd',
    },
    'polio': {
        'paralysis': 'wd zhwiki', 'vomiting': 'wd', 'fever': 'wd', 'weakness': 'wd zhwiki',
        'muscle_atrophy': 'zhwiki', 'fatigue': 'zhwiki', 'dysphagia': 'zhwiki',
        'rehab': 'wd zhwiki', 'traction_massage': 'wd zhwiki', 'ventilator': 'zhwiki', 'vaccine': 'zhwiki',
        'poliovirus': 'wd',
    },
    'cjd': {
        'myoclonus': 'wd', 'visual': 'wd zhwiki', 'paralysis': 'wd', 'weight_loss': 'wd', 'tremor': 'wd zhwiki',
        'dizziness': 'zhwiki', 'paresthesia': 'wd zhwiki',
        'amantadine': 'zhwiki', 'acyclovir': 'zhwiki', 'interferon': 'zhwiki',
        'eeg': 'wd zhwiki', 'mri': 'wd',
        'prion': 'wd',
    },
    'neurocysticercosis': {'albendazole': 'wd', 'praziquantel': 'wd'},
    'shingles': {
        'paresthesia': 'wd zhwiki', 'headache': 'wd', 'rash': 'wd zhwiki', 'fever': 'wd zhwiki',
        'neuropathic_pain': 'wd zhwiki', 'fatigue': 'zhwiki',
        'acyclovir': 'wd zhwiki', 'valacyclovir': 'wd', 'famciclovir': 'wd', 'capsaicin': 'wd',
        'gabapentin': 'wd_faers', 'pregabalin': 'wd_faers', 'acetaminophen': 'zhwiki', 'vaccine': 'zhwiki',
        'vzv': 'wd zhwiki',
    },
    'peripheral_neuropathy': {
        'duloxetine': 'wd_faers', 'capsaicin': 'wd_faers', 'gabapentin': 'wd_faers', 'pregabalin': 'wd_faers',
    },
    'gbs': {
        'weakness': 'msd zhwiki', 'paresthesia': 'msd zhwiki', 'numbness': 'zhwiki', 'paralysis': 'wd zhwiki',
        'neuropathic_pain': 'zhwiki', 'gait': 'msd',
        'ivig': 'msd zhwiki', 'acetaminophen': 'msd',
        'plasmapheresis': 'msd zhwiki', 'ventilator': 'msd zhwiki', 'rehab': 'msd zhwiki',
        'emg': 'msd zhwiki', 'ncv': 'msd zhwiki', 'lumbar_puncture': 'msd zhwiki', 'blood_test': 'msd',
        'autoimmune': 'msd zhwiki', 'campylobacter': 'msd', 'virus': 'msd',
    },
    'cmt': {
        'weakness': 'wd', 'paresthesia': 'wd', 'muscle_atrophy': 'wd',
        'rehab': 'wd', 'emg': 'wd', 'gene_test': 'wd', 'genetic': 'wd',
    },
    'mg': {
        'ptosis': 'msd zhwiki', 'visual': 'msd zhwiki', 'weakness': 'msd zhwiki', 'fatigue': 'msd zhwiki',
        'dysphagia': 'msd zhwiki', 'dyspnea': 'msd', 'dysarthria': 'zhwiki',
        'pyridostigmine': 'wd msd zhwiki', 'neostigmine': 'wd zhwiki', 'prednisone': 'wd_faers zhwiki',
        'azathioprine': 'wd_faers zhwiki', 'cyclosporine': 'wd zhwiki', 'tacrolimus': 'wd_faers',
        'mycophenolate': 'wd_faers zhwiki', 'eculizumab': 'wd_faers msd', 'rituximab': 'wd_faers',
        'ivig': 'msd zhwiki', 'corticosteroid': 'msd',
        'thymectomy': 'msd zhwiki', 'plasmapheresis': 'msd zhwiki', 'ventilator': 'msd zhwiki',
        'emg': 'msd zhwiki', 'blood_test': 'msd zhwiki', 'antibody_test': 'msd zhwiki', 'ice_test': 'msd zhwiki',
        'ct': 'msd zhwiki', 'mri': 'msd zhwiki',
        'autoimmune': 'msd zhwiki', 'thymoma': 'msd zhwiki', 'genetic': 'msd',
    },
    'als': {
        'weakness': 'msd', 'muscle_atrophy': 'msd', 'rigidity': 'msd', 'dysphagia': 'msd', 'dyspnea': 'msd',
        'riluzole': 'wd msd', 'edaravone': 'wd msd', 'baclofen': 'msd', 'benzodiazepine': 'msd',
        'antidepressant': 'msd',
        'ventilator': 'msd', 'rehab': 'msd',
        'emg': 'msd', 'ncv': 'msd', 'mri': 'msd', 'blood_test': 'msd', 'gene_test': 'msd',
        'genetic': 'msd',
    },
    'sma': {
        'weakness': 'zhwiki', 'muscle_atrophy': 'zhwiki', 'dysphagia': 'zhwiki', 'dyspnea': 'zhwiki',
        'nusinersen': 'wd zhwiki', 'risdiplam': 'zhwiki', 'gene_test': 'zhwiki', 'genetic': 'zhwiki',
    },
    'carpal_tunnel': {
        'numbness': 'zhwiki', 'paresthesia': 'zhwiki', 'weakness': 'zhwiki', 'muscle_atrophy': 'zhwiki',
        'neuropathic_pain': 'zhwiki',
        'gabapentin': 'zhwiki', 'corticosteroid': 'zhwiki', 'vitamin_b': 'zhwiki',
        'surgery': 'zhwiki', 'rehab': 'zhwiki',
        'emg': 'zhwiki', 'ncv': 'zhwiki', 'ultrasound': 'zhwiki', 'mri': 'zhwiki',
        'diabetes': 'zhwiki', 'obesity': 'zhwiki', 'genetic': 'zhwiki',
    },
    'sciatica': {
        'back_pain': 'zhwiki', 'weakness': 'zhwiki', 'numbness': 'zhwiki',
        'gabapentin': 'zhwiki', 'surgery': 'zhwiki',
    },
    'disc_herniation': {
        'surgery': 'zhwiki', 'rehab': 'zhwiki', 'traction_massage': 'zhwiki',
        'mri': 'zhwiki', 'ct': 'zhwiki', 'xray': 'zhwiki',
    },
    'cervical_spondylosis': {
        'numbness': 'zhwiki', 'muscle_atrophy': 'zhwiki', 'paralysis': 'zhwiki',
        'traction_massage': 'zhwiki', 'acupuncture': 'zhwiki', 'rehab': 'zhwiki',
        'smoking': 'zhwiki',
    },
    'insomnia': {
        'zolpidem': 'wd_faers', 'eszopiclone': 'wd_faers', 'alprazolam': 'wd_faers', 'clonazepam': 'wd_faers',
        'lorazepam': 'wd', 'diazepam': 'wd_faers', 'estazolam': 'wd', 'trazodone': 'wd',
        'mirtazapine': 'wd_faers', 'suvorexant': 'wd_faers', 'quetiapine': 'wd_faers',
        'amitriptyline': 'wd_faers', 'gabapentin': 'wd_faers',
        'cbt': 'wd zhwiki', 'psychotherapy': 'wd',
    },
    'narcolepsy': {
        'hypersomnia': 'zhwiki', 'cataplexy': 'zhwiki', 'hallucination': 'zhwiki',
        'modafinil': 'wd', 'methylphenidate': 'wd', 'sodium_oxybate': 'wd:Q944473',
    },
    'sleep_apnea': {
        'snoring': 'zhwiki', 'headache': 'zhwiki', 'fatigue': 'zhwiki', 'hypersomnia': 'zhwiki',
        'cpap': 'wd zhwiki', 'oral_appliance': 'wd zhwiki', 'weight_reduction': 'wd', 'surgery': 'zhwiki',
        'obesity': 'zhwiki', 'alcohol': 'zhwiki', 'hypertension': 'zhwiki',
    },
    'brain_tumor': {
        'headache': 'zhwiki', 'vomiting': 'zhwiki', 'unconsciousness': 'zhwiki', 'hypersomnia': 'zhwiki',
        'paralysis': 'zhwiki', 'icp': 'zhwiki',
        'dexamethasone': 'zhwiki', 'radiotherapy': 'zhwiki', 'chemotherapy': 'zhwiki', 'surgery': 'zhwiki',
        'mri': 'zhwiki', 'ct': 'zhwiki', 'biopsy': 'zhwiki', 'lumbar_puncture': 'zhwiki',
    },
    'glioblastoma': {'temozolomide': 'wd'},
    'pituitary_adenoma': {'bromocriptine': 'wd:Q954831'},
    'concussion': {
        'headache': 'wd zhwiki', 'visual': 'wd zhwiki', 'nausea': 'wd zhwiki', 'vomiting': 'zhwiki',
        'dizziness': 'zhwiki', 'unconsciousness': 'zhwiki',
        'acetaminophen': 'wd', 'nsaid': 'wd',
        'accident': 'wd',
    },
    'sci': {
        'paresthesia': 'wd', 'paralysis': 'wd',
        'baclofen': 'wd', 'dantrolene': 'wd',
        'xray': 'wd', 'mri': 'wd', 'ct': 'wd',
    },
    'hydrocephalus': {
        'headache': 'zhwiki', 'vomiting': 'zhwiki', 'nausea': 'zhwiki', 'papilledema': 'zhwiki',
        'hypersomnia': 'zhwiki', 'unconsciousness': 'zhwiki', 'gait': 'zhwiki', 'incontinence': 'zhwiki',
        'icp': 'zhwiki',
        'shunt': 'zhwiki',
    },
    'cerebral_edema': {'mannitol': 'wd', 'dexamethasone': 'wd'},
    'bppv': {
        'dizziness': 'wd zhwiki', 'nausea': 'zhwiki', 'vomiting': 'zhwiki',
        'repositioning': 'wd', 'dix_hallpike': 'wd',
    },
    'meniere': {'dizziness': 'wd', 'tinnitus': 'wd', 'hearing_loss': 'wd', 'audiometry': 'wd'},
    'cerebral_palsy': {
        'spasm': 'wd', 'rigidity': 'wd',
        'baclofen': 'wd_faers', 'botulinum': 'wd_faers',
        'mri': 'wd', 'gene_test': 'wd', 'ultrasound': 'wd',
        'hypoxia': 'wd',
    },
    'anxiety': {
        'sertraline': 'wd', 'paroxetine': 'wd_faers', 'escitalopram': 'wd_faers', 'venlafaxine': 'wd',
        'duloxetine': 'wd_faers', 'buspirone': 'wd', 'alprazolam': 'wd', 'lorazepam': 'wd', 'diazepam': 'wd',
        'clonazepam': 'wd_faers', 'pregabalin': 'wd_faers', 'propranolol': 'wd', 'mirtazapine': 'wd_faers',
        'quetiapine': 'wd_faers', 'antidepressant': 'zhwiki', 'benzodiazepine': 'zhwiki',
        'cbt': 'zhwiki', 'psychotherapy': 'wd',
    },
}

# %%
# =============================================================================
# 方面层（沿用肝病版 06；只作为问题的附加输出，不进入答案标签）
# =============================================================================
# 提问句式的线索词，不是医学事实。与肝病版相比：去掉乙肝专用词（母婴阻断、分餐、献血、癌变、转成），
# 增加神经科问答里常见的问法（后遗症、复发、康复训练、针灸、片子）。

ASPECTS = {
    '医疗费用': {
        'strong': ['多少钱', '费用', '价格', '贵不贵', '贵吗', '花费', '医保', '报销', '要花多少'],
        'weak':   [],
    },
    '就医建议': {
        'strong': ['哪家医院', '哪个医院', '什么医院', '挂什么科', '挂哪个科', '看什么科',
                   '哪个科', '去哪里看', '去哪看', '哪里治', '哪里看', '推荐医院', '哪位医生'],
        'weak':   [],
    },
    '传播途径': {
        'strong': ['传染', '传播', '传给', '遗传', '被感染', '感染给', '隔离'],
        'weak':   [],
    },
    '指标解读': {
        'strong': ['指标', '化验单', '报告单', '片子', '检查结果', '化验结果', '偏高', '偏低', '升高',
                   '超标', '正常值', '参考值', '阳性', '阴性', '数值', '是什么意思',
                   '什么意思', '代表什么', '说明什么', '正常吗', '高不高'],
        'weak':   [],
    },
    '病情诊断': {
        'strong': ['是否得了', '是不是得了', '怎么确诊', '如何确诊', '确诊', '如何诊断',
                   '怎么诊断', '怎么判断', '如何判断', '怎么知道', '怎么查', '做什么检查',
                   '做哪些检查', '查什么', '检查方法', '辅助检查', '检查有', '检查项目',
                   '能检查出', '能查出', '检测出', '查得出', '鉴别', '准确率', '准不准'],
        'weak':   ['是不是', '会不会是', '算不算'],
    },
    '病因分析': {
        'strong': ['原因', '为什么', '为何', '怎么引起', '什么引起', '引起的', '怎么得的',
                   '怎么会', '诱因', '病因', '有关系', '有关吗', '有关联'],
        'weak':   [],
    },
    '后果表述': {
        'strong': ['严重吗', '严重不', '严不严重', '会怎样', '后果', '危害', '能活多久',
                   '活多久', '寿命', '生命危险', '后遗症', '复发', '会发展成', '会变成',
                   '恶化', '预后', '会不会死'],
        'weak':   [],
    },
    '注意事项': {
        'strong': ['注意什么', '注意事项', '要注意', '注意些什么', '注意哪些', '能不能吃',
                   '能吃', '可以吃', '不能吃', '吃什么', '饮食', '忌口', '食物', '能喝',
                   '喝酒', '护理', '预防', '保养', '疫苗', '接种', '怀孕', '怀上', '要孩子',
                   '生孩子', '生育', '喂奶', '母乳', '哺乳', '多久复查',
                   '多久检查', '多久一次', '摄入', '运动', '熬夜'],
        'weak':   ['能不能', '可以吗', '可不可以', '能否', '可否', '行吗'],
    },
    '功效作用': {
        'strong': ['作用', '功效', '有什么用', '有用吗', '管什么用', '有效吗', '好用吗',
                   '用法用量', '用法', '副作用'],
        'weak':   [],
    },
    '治疗方案': {
        'strong': ['怎么治', '如何治', '怎样治', '咋治', '怎么医治', '如何医治', '治疗',
                   '救治', '医治', '吃什么药', '用什么药', '什么药', '吃药', '中药', '西药',
                   '能治好', '治得好', '能治愈', '治愈', '根治', '治好', '好治', '难治',
                   '康复', '康复训练', '怎么调理', '调理', '偏方', '手术', '针灸', '停药', '用药',
                   '服药', '缓解'],
        'weak':   ['怎么办', '怎么处理', '咋办'],
    },
    '疾病表述': {
        'strong': ['症状', '表现', '什么感觉', '早期', '潜伏期', '分期', '分型', '区别',
                   '有哪些症状'],
        'weak':   ['是什么', '什么是', '什么病', '什么叫', '什么样'],
    },
}

# 问题的方面决定“问题这种病的资料里，哪类实体和它相关”。这是检索设计上的取舍，不是医学事实；
# 没列出的方面（或问题没有方面）使用全部类型。
ASPECT_ENTITY_TYPES = {
    '治疗方案': {'drug', 'treatment'},
    '功效作用': {'drug', 'treatment'},
    '疾病表述': {'symptom'},
    '后果表述': {'symptom'},
    '病情诊断': {'exam', 'symptom'},
    '指标解读': {'exam'},
    '病因分析': {'cause'},
    '注意事项': {'cause', 'treatment'},
}

FORM_FIELDS = ['曾经的治疗情况和效果', '曾经治疗情况和效果', '曾经的治疗情况及效果',
               '想得到怎样的帮助', '在乎怎样的帮助', '需要怎样的帮助', '病情描述',
               '详细病情及咨询目的', '患者性别', '患者年龄', '问题补充', '化验检查结果',
               '主要症状', '发病时间', '病情叙述']

INTERROGATIVE = ['？', '?', '吗', '呢', '什么', '怎', '如何', '哪', '多少', '多久', '几',
                 '是否', '能否', '可否', '是不是', '会不会', '能不能', '可不可', '请问',
                 '想知道', '想问', '咋']

# %%
# =============================================================================
# 构建词典：同义词 -> 概念；实体词典 + 疾病资料（profile）
# =============================================================================

concept_by_id = {c['id']: c for c in CONCEPTS}
assert len(concept_by_id) == len(CONCEPTS), 'duplicate concept id'
for c in CONCEPTS:
    assert c['parent'] is None or c['parent'] in concept_by_id, 'bad parent: ' + c['id']
    assert c['id'] in FAMILY, 'concept without family: ' + c['id']
cname = {c['id']: c['name'] for c in CONCEPTS}

term2concepts = defaultdict(set)
for c in CONCEPTS:
    for t in c['synonyms']:
        term2concepts[prep(t)].add(c['id'])
    for t in c['extra']:
        term2concepts[prep(t)].add(c['id'])

# 同一个词指向多个概念时，说明词典本身有歧义
ambiguous = {t: sorted(cs) for t, cs in term2concepts.items() if len(cs) > 1}

# 纯英文缩写只在两侧不是英文字母时才算匹配
ascii_terms = {t for t in term2concepts if re.fullmatch(r'[A-Z0-9\-]+', t)}
cjk_terms = sorted((t for t in term2concepts if t not in ascii_terms), key=len, reverse=True)
CJK_RE = re.compile('|'.join(map(re.escape, cjk_terms)))
ASCII_RE = (re.compile(r'(?<![A-Z])('
                       + '|'.join(map(re.escape, sorted(ascii_terms, key=len, reverse=True)))
                       + r')(?![A-Z])') if ascii_terms else None)
GENERIC_RE = re.compile('|'.join(map(re.escape, sorted(map(prep, GENERIC_TERMS), key=len, reverse=True))))

_anc_cache = {}


def ancestors(cid):
    if cid not in _anc_cache:
        out, c = [], cid
        while concept_by_id[c]['parent']:
            c = concept_by_id[c]['parent']
            out.append(c)
        _anc_cache[cid] = out
    return _anc_cache[cid]


def lineage(c):
    return {c, *ancestors(c)}


def related(a, b):
    '''相同，或互为上下位（脑卒中 / 脑梗死）'''
    return a in lineage(b) or b in lineage(a)


def count_hits(text):
    '''文本里每个概念的命中次数（最长匹配）'''
    q = prep(text)
    cnt = Counter()
    for m in CJK_RE.finditer(q):
        for c in term2concepts[m.group(0)]:
            cnt[c] += 1
    if ASCII_RE:
        for m in ASCII_RE.finditer(q):
            for c in term2concepts[m.group(1)]:
                cnt[c] += 1
    return cnt


def specific_counts(text):
    '''命中次数，且同时命中父、子概念时只保留子概念（脑卒中，脑梗死 -> 脑梗死）'''
    cnt = count_hits(text)
    redundant = {a for c in cnt for a in ancestors(c)}
    return {c: n for c, n in cnt.items() if c not in redundant}


def tag_concepts(question):
    return set(specific_counts(question))


entity_by_id = {e['id']: e for e in ENTITIES}
ename = {e['id']: e['name'] for e in ENTITIES}
for cid, rels in REL.items():
    assert cid in concept_by_id, 'REL: unknown concept ' + cid
    for eid, src in rels.items():
        assert eid in entity_by_id, f'REL[{cid}]: unknown entity {eid}'
        source_urls(cid, src.split())          # 来源键写错会在这里报错

term2entities = defaultdict(set)
for e in ENTITIES:
    for t in e['terms']:
        term2entities[prep(t)].add(e['id'])
ambiguous_entity_terms = {t: sorted(es) for t, es in term2entities.items() if len(es) > 1}

e_ascii = {t for t in term2entities if re.fullmatch(r'[A-Z0-9\-]+', t)}
E_CJK_RE = re.compile('|'.join(map(re.escape, sorted((t for t in term2entities if t not in e_ascii),
                                                     key=len, reverse=True))))
E_ASCII_RE = (re.compile(r'(?<![A-Z])('
                         + '|'.join(map(re.escape, sorted(e_ascii, key=len, reverse=True)))
                         + r')(?![A-Z])') if e_ascii else None)


def entity_counts(text):
    '''文本里每个实体的命中次数（最长匹配，与概念层同一规则）'''
    q = prep(text)
    cnt = Counter()
    for m in E_CJK_RE.finditer(q):
        for e in term2entities[m.group(0)]:
            cnt[e] += 1
    if E_ASCII_RE:
        for m in E_ASCII_RE.finditer(q):
            for e in term2entities[m.group(1)]:
                cnt[e] += 1
    return dict(cnt)


def _usable(src):
    toks = src.split()
    return USE_FAERS or any(t != 'wd_faers' for t in toks)


# profile[c]：概念 c 自己的实体；owners[e]：哪些概念的资料里有 e
profile = {cid: {e for e, src in rels.items() if _usable(src)} for cid, rels in REL.items()}
owners = defaultdict(set)
for cid, es in profile.items():
    for e in es:
        owners[e].add(cid)

# 与概念 c 相同或互为上下位的概念（问脑梗死时，脑卒中的资料也算；问脑卒中时，各子型的资料都算）
related_concepts = {c: {x for x in concept_by_id if related(x, c)} for c in concept_by_id}

# 方面判断（沿用肝病版 06：最后一个提问小句里，最靠近句末的关键词）
KW = {tier: [(prep(k), name) for name, d in ASPECTS.items() for k in d[tier]]
      for tier in ('strong', 'weak')}
FORM_RE = re.compile('|'.join(map(re.escape, sorted(map(prep, FORM_FIELDS), key=len, reverse=True))))
CLAUSE_RE = re.compile(r'[^，,。！!？?；;\n]+[？?]?')
INTERROG = [prep(x) for x in INTERROGATIVE]


def focus_clause(question):
    q = FORM_RE.sub('，', prep(question))
    clauses = [m.group(0).strip() for m in CLAUSE_RE.finditer(q) if m.group(0).strip()]
    asking = [c for c in clauses if any(w in c for w in INTERROG)]
    if asking:
        return 'clause', asking[-1]
    if len(q) <= SHORT_QUESTION_CHARS:
        return 'short', q
    return None, ''


def rightmost(text, keywords):
    best = None
    for kw, name in keywords:
        i = text.rfind(kw)
        if i < 0:
            continue
        key = (i + len(kw), len(kw))
        if best is None or key > best[0]:
            best = (key, name, kw)
    return (best[1], best[2]) if best else None


def tag_aspect(question):
    how, text = focus_clause(question)
    if how is None:
        return None, None, None, ''
    for tier in ('strong', 'weak'):
        hit = rightmost(text, KW[tier])
        if hit:
            return hit[0], hit[1], tier, text
    return None, None, None, text

# %%
# =============================================================================
# 读入训练数据，给问题贴标签（一个 qid 一条）
# =============================================================================

train = read_jsonl(TRAIN_PATH)
corpus = read_jsonl(CORPUS_PATH)
doc_text = {d['doc_id']: d['text'] for d in corpus}
row_qid = [qid_of(normalize(r['anchor'])) for r in train]
print(f'{len(train):,} training pairs, {len(set(row_qid)):,} unique questions, '
      f'{len(corpus):,} training answers   (t2s backend: {T2S_BACKEND})')

q_info = {}
for r, qid in tqdm(list(zip(train, row_qid)), desc='Tagging questions'):
    if qid in q_info:
        continue
    concepts = sorted(tag_concepts(r['anchor']))
    rec = {'qid': qid, 'anchor': r['anchor'], 'source': r['source'],
           'concepts': concepts,
           'families': sorted({FAMILY[c] for c in concepts}),
           'generic': bool(GENERIC_RE.search(prep(r['anchor']))) and not concepts,
           'entities': entity_counts(r['anchor'])}
    if TAG_ASPECT:
        aspect, kw, tier, focus = tag_aspect(r['anchor'])
        rec.update({'aspect': aspect, 'aspect_kw': kw, 'aspect_tier': tier, 'focus': focus})
    q_info[qid] = rec

# 答案继承：与它配对的训练问题的概念（一个答案可能被多个问题配对）
inherited = defaultdict(Counter)
q_pos = defaultdict(set)
for r, qid in zip(train, row_qid):
    q_pos[qid].add(r['doc_id'])
    for c in q_info[qid]['concepts']:
        inherited[r['doc_id']][c] += 1

# 给答案贴标签：正文点名的疾病（mentioned）+ 继承的疾病（inherited）+ 实体（entities）
# specific：正文点名了至少一种病。
# unnamed ：正文没点名任何病（“注意休息、规律作息”之类），什么病的问题都可能用得上，
#           08 不会把它当作“不同病”的负例。

doc_rec = {}
for d in tqdm(corpus, desc='Tagging answers'):
    did = d['doc_id']
    ment = specific_counts(d['text'])
    doc_rec[did] = {'doc_id': did,
                    'mentioned': ment,
                    'inherited': dict(inherited.get(did, {})),
                    'status': 'specific' if ment else 'unnamed',
                    'generic_term': bool(GENERIC_RE.search(prep(d['text']))),
                    'entities': entity_counts(d['text']),
                    'n_chars': len(d['text'])}

missing_docs = set(inherited) - set(doc_rec)
if missing_docs:
    print(f'warning: {len(missing_docs):,} answers in train.jsonl are not in train_corpus.jsonl')


def relation(q_concepts, d_mentioned, d_inherited):
    '''
    问题与答案在疾病层面的关系（08 使用同一条规则）：
      unknown   问题没有疾病标签，无法判断
      same      答案正文点名了、或它配对的问题涉及与问题相同/上下位的疾病
      different 答案正文点名了疾病，且正文和配对问题都与问题的疾病无关
      unnamed   答案正文没点名任何病，也没有关联到问题的病（泛泛的回答，不能当作不同病）
    '''
    if not q_concepts:
        return 'unknown'
    d_all = set(d_mentioned) | set(d_inherited)
    if any(related(c, x) for c in d_all for x in q_concepts):
        return 'same'
    return 'different' if d_mentioned else 'unnamed'


# 随机配对的先验：问题和答案随机配时，各关系占多少。这是“不同病”命中率的基线
rng = random.Random(SEED)
labelled_q = [q for q in q_info.values() if q['concepts']]
doc_ids_all = sorted(doc_rec)
prior = Counter()
if labelled_q and doc_ids_all:
    for _ in range(N_PRIOR_PAIRS):
        q = rng.choice(labelled_q)
        d = doc_rec[rng.choice(doc_ids_all)]
        prior[relation(set(q['concepts']), d['mentioned'], d['inherited'])] += 1

# 继承标签与正文点名的一致程度：低说明“只继承”不可靠
both = [x for x in doc_rec.values() if x['mentioned'] and x['inherited']]
agree = sum(any(related(c, m) for c in x['inherited'] for m in x['mentioned']) for x in both)

# %%
# =============================================================================
# 本体相似度（不依赖模型）
# =============================================================================
# 实体权重 = 在训练答案里的 IDF：越多答案提到（头痛、磁共振），越不能说明两段话像。

n_docs = max(len(doc_rec), 1)
entity_df = Counter(e for x in doc_rec.values() for e in x['entities'])
idf = {e: math.log((n_docs + 1) / (entity_df.get(e, 0) + 1)) + 1.0 for e in entity_by_id}

# 答案方面：继承它配对的那些训练问题的方面（多数派）
doc_aspect = {}
for r, qid in zip(train, row_qid):
    asp = q_info[qid].get('aspect')
    if asp:
        doc_aspect.setdefault(r['doc_id'], Counter())[asp] += 1
doc_aspect = {d: c.most_common(1)[0][0] for d, c in doc_aspect.items()}


def expected_entities(q):
    '''问题这种病（含上下位）的资料里、和问题方面相关的实体，再加上问题自己点名的实体'''
    out = set()
    for c in q['concepts']:
        for x in related_concepts[c]:
            out |= profile.get(x, set())
    types = ASPECT_ENTITY_TYPES.get(q.get('aspect'))
    if types:
        out = {e for e in out if entity_by_id[e]['type'] in types} or out
    return out | set(q['entities'])


def onto_cos_score(q_info_row, doc_id):
    '''
    归一化 IDF 余弦相似度：
      q_vec[e] = idf[e]  若 e 在 expected_entities(q) 中，否则 0
      d_vec[e] = idf[e]  若 e 在答案实体中，否则 0
      score = dot(q_vec, d_vec) / (||q_vec|| * ||d_vec|| + 1e-9)
    再乘以方面匹配系数（同方面 ASPECT_BOOST，无方面信息 1.0，不同方面 1/ASPECT_BOOST）。
    '''
    ASPECT_BOOST = 2.0
    exp = expected_entities(q_info_row)
    d_ent = set(doc_rec[doc_id]['entities']) if doc_id in doc_rec else set()
    q_vec = {e: idf[e] for e in exp}
    d_vec = {e: idf[e] for e in d_ent}
    if not q_vec or not d_vec:
        return 0.0
    dot = sum(q_vec[e] * d_vec[e] for e in q_vec if e in d_vec)
    norm_q = math.sqrt(sum(v * v for v in q_vec.values()))
    norm_d = math.sqrt(sum(v * v for v in d_vec.values()))
    cos = dot / (norm_q * norm_d + 1e-9)
    q_asp = q_info_row.get('aspect')
    d_asp = doc_aspect.get(doc_id)
    if q_asp and d_asp:
        cos = min(1.0, cos * ASPECT_BOOST) if q_asp == d_asp else cos / ASPECT_BOOST
    return cos


def entity_score(q, d):
    '''support / conflict / missing 供抽检表解释；score 用 onto_cos_score'''
    q_ent, d_ent = set(q['entities']), set(d['entities'])
    exp = expected_entities(q)
    allowed = set().union(*(related_concepts[c] for c in q['concepts'])) if q['concepts'] else set()
    support = d_ent & exp
    conflict = ({e for e in d_ent - exp if owners.get(e) and not (owners[e] & allowed)}
                if allowed else set())
    missing = q_ent - d_ent
    return onto_cos_score(q, d['doc_id']), support, conflict, missing


RELATION_BASE = {'same': 1.0, 'different_same_family': 0.5, 'unnamed': 0.3, 'unknown': 0.3,
                 'different': 0.0}


def doc_families(d):
    return {FAMILY[c] for c in set(d['mentioned']) | set(d['inherited'])}


def onto_sim(q, d):
    '''疾病层关系 + 归一化余弦相似度。08 直接 import 或复制 onto_cos_score。'''
    rel = relation(set(q['concepts']), d['mentioned'], d['inherited'])
    if rel == 'different' and set(q['families']) & doc_families(d):
        rel = 'different_same_family'
    es, support, conflict, missing = entity_score(q, d)
    return {'relation': rel, 'entity_score': round(es, 4),
            'score': round((1 - ENTITY_WEIGHT) * RELATION_BASE[rel] + ENTITY_WEIGHT * es, 4),
            'support': sorted(support), 'conflict': sorted(conflict), 'missing': sorted(missing)}

# %%
# =============================================================================
# 写出
# =============================================================================

with open(f'{ONTO_DIR}/concepts.json', 'w', encoding='utf-8') as f:
    json.dump({'built_at': datetime.datetime.now().isoformat(timespec='seconds'),
               'note': 'neuroscience dictionary with family and Wikidata ids; '
                       'relations in entities.json were collected online on 2026-11-14',
               'sources': {'icd10': 'https://icd.who.int/browse10/2019/en',
                           'wikipedia': 'https://zh.wikipedia.org/wiki/',
                           'wikidata': 'https://www.wikidata.org/',
                           'msd_manual_zh': 'https://www.msdmanuals.cn/home'},
               'concepts': [{**c, 'family': FAMILY[c['id']],
                             'wikidata': CONCEPT_WD.get(c['id'], [])} for c in CONCEPTS],
               'generic_terms': GENERIC_TERMS,
               'aspects': ASPECTS,
               'aspect_entity_types': {k: sorted(v) for k, v in ASPECT_ENTITY_TYPES.items()},
               'form_fields': FORM_FIELDS},
              f, ensure_ascii=False, indent=1)

with open(f'{ONTO_DIR}/entities.json', 'w', encoding='utf-8') as f:
    json.dump({'use_faers': USE_FAERS,
               'source_keys': {'wd': 'Wikidata claim on the concept item (non-FAERS reference or none)',
                               'wd_faers': 'Wikidata P2176 claim whose only reference is FAERS (Q56863002)',
                               'msd': 'MSD Manual (Chinese consumer edition) page of the concept',
                               'zhwiki': 'Chinese Wikipedia page of the concept'},
               'entities': [{**e, 'idf': round(idf[e['id']], 4),
                             'answer_df': entity_df.get(e['id'], 0)} for e in ENTITIES],
               'relations': [{'concept': cid, 'entity': eid, 'type': entity_by_id[eid]['type'],
                              'sources': src.split(), 'urls': source_urls(cid, src.split()),
                              'used': _usable(src)}
                             for cid, rels in REL.items() for eid, src in rels.items()],
               'ambiguous_terms': ambiguous_entity_terms},
              f, ensure_ascii=False, indent=1)

write_jsonl(f'{ONTO_DIR}/question_labels.jsonl', list(q_info.values()))
write_jsonl(f'{ONTO_DIR}/doc_labels.jsonl', [doc_rec[d] for d in doc_ids_all])

# %%
# =============================================================================
# 人工抽检表 + 词典漏标诊断
# =============================================================================


def names(ids):
    return '、'.join(sorted(cname[c] for c in ids)) or '-'


def enames(ids):
    return '、'.join(sorted(ename[e] for e in ids)) or '-'


rng = random.Random(SEED)

# 1. 问题标签准确率：随机抽 N_CHECK 条有概念标签的问题
with open(f'{ONTO_DIR}/check_questions_labelled.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['question', 'concepts', 'entities', 'concept_correct (y/n)'])
    for q in rng.sample(labelled_q, min(N_CHECK, len(labelled_q))):
        w.writerow([q['anchor'], names(q['concepts']), enames(q['entities']), ''])

# 2. 词典漏标：没有概念、也不是泛指的问题。看看缺的是哪些词
no_signal = [q for q in q_info.values() if not q['concepts'] and not q['generic']]
with open(f'{ONTO_DIR}/check_questions_unlabelled.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['question', 'which_disease_if_any (name or n)'])
    for q in rng.sample(no_signal, min(N_CHECK, len(no_signal))):
        w.writerow([q['anchor'], ''])

# 漏标问题里“脑 / 神经 / 颅 / 脊 / 髓 / 癫 / 瘫 / 痴 / 眩”前后的词，按频次列出，用来补词典
TERM_RE = re.compile(r'[\u4e00-\u9fa5]{0,3}(?:脑|神经|颅|脊|髓|癫|瘫|痴|眩|晕|抽)[\u4e00-\u9fa5]{0,3}')
term_cnt = Counter()
for q in no_signal:
    for t in TERM_RE.findall(prep(q['anchor'])):
        term_cnt[t] += 1
with open(f'{ONTO_DIR}/unlabelled_neuro_terms.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['term_around_neuro_char', 'count_in_unlabelled_questions'])
    for t, k in term_cnt.most_common(TOP_UNLABELLED_TERMS):
        w.writerow([t, k])

# 3. 答案正文点名是否可靠：随机抽 N_CHECK 条点名了疾病的答案
specific_docs = [d for d in doc_ids_all if doc_rec[d]['mentioned']]
with open(f'{ONTO_DIR}/check_answers_specific.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['answer_head', 'mentioned', 'inherited', 'entities', 'mentioned_is_main_topic (y/n)'])
    for did in rng.sample(specific_docs, min(N_CHECK, len(specific_docs))):
        x = doc_rec[did]
        w.writerow([doc_text[did][:200], names(x['mentioned']), names(x['inherited']),
                    enames(x['entities']), ''])

# 4. 本体相似度挑出的难负例。每个问题列出 onto_sim 最高、但不是它配对答案的 HARD_TOPK 条
by_concept, by_entity = defaultdict(set), defaultdict(set)
for did, x in doc_rec.items():
    for c in set(x['mentioned']) | set(x['inherited']):
        by_concept[c].add(did)
    for e in x['entities']:
        by_entity[e].add(did)

N_HARD_Q = 150
POOL_NAMES = ('all', 'not_same')
REL_ORDER = ('same', 'different_same_family', 'unnamed', 'unknown', 'different')
LABEL_COL = 'candidate_also_answers_question (y/n/?)'

doc_comp, q_comp = {}, {}
for r, qid in zip(train, row_qid):
    doc_comp.setdefault(r['doc_id'], r.get('component_id'))
    q_comp.setdefault(qid, r.get('component_id'))

family_concepts = defaultdict(set)
for c in CONCEPTS:
    family_concepts[FAMILY[c['id']]].add(c['id'])


def candidate_docs(q):
    '''问题 q 的候选答案：共享资料实体的（按实体 IDF 之和排序在前）+ 同大类疾病的，最多 MAX_CANDIDATES 个'''
    hits = {}
    for e in expected_entities(q):
        for did in by_entity.get(e, ()):
            hits[did] = hits.get(did, 0.0) + idf[e]
    for fam in q['families']:
        for c in family_concepts[fam]:
            for did in by_concept.get(c, ()):
                hits.setdefault(did, 0.0)
    skip = q_pos[q['qid']]
    qc = q_comp.get(q['qid'])
    out = [d for d in hits if d not in skip and (qc is None or doc_comp.get(d) != qc)]
    out.sort(key=lambda d: (-hits[d], d))
    return out[:MAX_CANDIDATES]


hard_rel = Counter()
rng = random.Random(SEED + 7)
sample_q = rng.sample(labelled_q, min(N_HARD_Q, len(labelled_q)))
picked = {}                # (qid, doc_id) -> 这条候选在哪些组、第几名、onto_sim 结果
n_no_cand = 0
n_cand = []
for q in tqdm(sample_q, desc='Scoring candidates'):
    scored = []
    for did in candidate_docs(q):
        s = onto_sim(q, doc_rec[did])
        scored.append((s['score'], did, s))
    if not scored:
        n_no_cand += 1
        continue
    n_cand.append(len(scored))
    scored.sort(key=lambda x: (-x[0], x[1]))
    groups = {'all': scored[:HARD_TOPK],
              'not_same': [x for x in scored if x[2]['relation'] != 'same'][:HARD_TOPK]}
    for pool_name, rows in groups.items():
        for rank, (sc, did, s) in enumerate(rows, 1):
            item = picked.setdefault((q['qid'], did), {'q': q, 'doc_id': did, 'sim': s, 'pools': [], 'ranks': []})
            item['pools'].append(pool_name)
            item['ranks'].append(rank)
            hard_rel[(pool_name, s['relation'])] += 1

items = list(picked.values())
random.Random(SEED + 8).shuffle(items)

sheet = ONTO_DIR + '/onto_hard_check.csv'
key = ONTO_DIR + '/onto_hard_check_key.csv'
with open(sheet, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'question', 'reference_answer', 'candidate_answer', LABEL_COL])
    for i, it in enumerate(items, 1):
        q = it['q']
        ref = sorted(q_pos[q['qid']])[0]
        w.writerow([i, q['anchor'], doc_text.get(ref, '')[:300], doc_text[it['doc_id']][:300], ''])
with open(key, 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'qid', 'doc_id', 'pools', 'ranks', 'relation', 'onto_sim', 'entity_score',
                'support', 'conflict', 'missing', 'question_concepts', 'aspect'])
    for i, it in enumerate(items, 1):
        s = it['sim']
        w.writerow([i, it['q']['qid'], it['doc_id'], '|'.join(it['pools']), '|'.join(map(str, it['ranks'])),
                    s['relation'], s['score'], s['entity_score'], enames(s['support']), enames(s['conflict']),
                    enames(s['missing']), names(it['q']['concepts']), it['q'].get('aspect') or ''])

hard_rows = [{'qid': it['q']['qid'], 'doc_id': it['doc_id'], 'pools': it['pools'], 'ranks': it['ranks'],
              'relation': it['sim']['relation'], 'onto_sim': it['sim']['score'],
              'entity_score': it['sim']['entity_score'], 'support': it['sim']['support'],
              'conflict': it['sim']['conflict'], 'missing': it['sim']['missing']} for it in items]
write_jsonl(ONTO_DIR + '/onto_hard_candidates.jsonl', hard_rows)

hard_stats = {}
for pool_name in POOL_NAMES:
    rows = [it for it in items if pool_name in it['pools']]
    n = max(len(rows), 1)
    hard_stats[pool_name] = {
        'rows': len(rows),
        'relation': {k: round(hard_rel[(pool_name, k)] / n, 4) for k in REL_ORDER},
        'mean_onto_sim': round(sum(it['sim']['score'] for it in rows) / n, 4),
        'mean_entity_score': round(sum(it['sim']['entity_score'] for it in rows) / n, 4),
        'with_support': round(sum(bool(it['sim']['support']) for it in rows) / n, 4),
        'with_conflict': round(sum(bool(it['sim']['conflict']) for it in rows) / n, 4)}
cand_sorted = sorted(n_cand)
cand_median = cand_sorted[len(cand_sorted) // 2] if cand_sorted else 0

# %%
# =============================================================================
# 报告
# =============================================================================

nq = len(q_info)
n_c = len(labelled_q)
n_gen = sum(1 for q in q_info.values() if q['generic'])
n_multi = sum(1 for q in q_info.values() if len(q['concepts']) > 1)
n_doc = len(doc_rec)
n_spec = sum(1 for x in doc_rec.values() if x['status'] == 'specific')
n_inh = sum(1 for x in doc_rec.values() if x['inherited'])
n_unnamed_inh = sum(1 for x in doc_rec.values() if x['status'] == 'unnamed' and x['inherited'])
q_cnt = Counter(c for q in q_info.values() for c in q['concepts'])
d_cnt = Counter(c for x in doc_rec.values() for c in x['mentioned'])
n_prior = max(sum(prior.values()), 1)

q_with_ent = sum(1 for q in q_info.values() if q['entities'])
q_lab_with_ent = sum(1 for q in labelled_q if q['entities'])
d_with_ent = sum(1 for x in doc_rec.values() if x['entities'])
n_rel_all = sum(len(r) for r in REL.values())
n_rel_used = sum(1 for r in REL.values() for s in r.values() if _usable(s))
n_faers_only = sum(1 for r in REL.values() for s in r.values() if all(t == 'wd_faers' for t in s.split()))
rel_by_type = Counter(entity_by_id[e]['type'] for r in REL.values() for e in r)
rel_by_src = Counter(t.split(':')[0] for r in REL.values() for s in r.values() for t in s.split())
no_profile = [c['id'] for c in CONCEPTS if not profile.get(c['id'])]
no_related_profile = [c for c in concept_by_id if not any(profile.get(x) for x in related_concepts[c])]
dead_entities = [e for e in entity_by_id if entity_df.get(e, 0) == 0]
top_entities = entity_df.most_common(15)
fam_cnt = Counter(f for q in q_info.values() for f in q['families'])

print('=' * 70)
print('Step 06 (neuro) · Ontology labelling + relation layer')
print('=' * 70)
print('')
print('  Concepts / synonym terms      : {} / {}   families: {}   ambiguous terms: {} {}'.format(
    len(CONCEPTS), len(term2concepts), len(set(FAMILY.values())), len(ambiguous), ambiguous if ambiguous else ''))
print('')
print('  Unique training questions     : {:,}'.format(nq))
print('  with a concept                : {:,} ({:.1%})'.format(n_c, n_c / max(nq, 1)))
print('  generic only                  : {:,} ({:.1%})'.format(n_gen, n_gen / max(nq, 1)))
print('  no signal (vocabulary gap?)   : {:,} ({:.1%})   -> see unlabelled_neuro_terms.csv'.format(
    len(no_signal), len(no_signal) / max(nq, 1)))
print('  with more than one concept    : {:,}'.format(n_multi))
print('')
print('  Training answers              : {:,}'.format(n_doc))
print('  name a disease in the body    : {:,} ({:.1%})  [specific]'.format(n_spec, n_spec / max(n_doc, 1)))
print('  name no disease               : {:,} ({:.1%})  [unnamed]'.format(n_doc - n_spec, (n_doc - n_spec) / max(n_doc, 1)))
print('  with an inherited label       : {:,} ({:.1%})'.format(n_inh, n_inh / max(n_doc, 1)))
print('  inherited but body names none : {:,}'.format(n_unnamed_inh))
if both:
    print('  inherited vs. body agree      : {:,} / {:,} ({:.1%})'.format(agree, len(both), agree / len(both)))
print('')
print('  Random question-answer pairs (prior of each relation, labelled questions only)')
for k in ('same', 'different', 'unnamed'):
    print('    {:<10s} {:>7.1%}'.format(k, prior[k] / n_prior))

print('')
print('Question concepts (top 15)')
for c, k in q_cnt.most_common(15):
    print('  {:<14s} questions {:>7,}   answers naming it {:>7,}'.format(cname[c], k, d_cnt.get(c, 0)))
print('Question families : ' + ', '.join('{} {:,}'.format(f, k) for f, k in fam_cnt.most_common()))

print('')
print('Relation layer (symptom / drug / treatment / exam / cause)')
print('  Entities {} ; relations {} (used {}; only FAERS-supported {}; USE_FAERS={})'.format(
    len(ENTITIES), n_rel_all, n_rel_used, n_faers_only, USE_FAERS))
print('  relations by entity type : ' + ', '.join('{} {}'.format(t, k) for t, k in sorted(rel_by_type.items())))
print('  relations by source      : ' + ', '.join('{} {}'.format(t, k) for t, k in sorted(rel_by_src.items())))
print('  questions mentioning an entity : {:,} ({:.1%}); among labelled {:,} ({:.1%})'.format(
    q_with_ent, q_with_ent / max(nq, 1), q_lab_with_ent, q_lab_with_ent / max(n_c, 1)))
print('  answers mentioning an entity   : {:,} ({:.1%})'.format(d_with_ent, d_with_ent / max(n_doc, 1)))
print('  concepts with no profile of their own : {}'.format(names(no_profile)))
print('  concepts with no profile even through parent/child : {}'.format(names(no_related_profile)))
print('  entities never found in any answer ({}): {}'.format(len(dead_entities), enames(dead_entities[:20])))
print('  most frequent entities in answers : ' + ', '.join('{} {:,}'.format(ename[e], k) for e, k in top_entities))
if ambiguous_entity_terms:
    print('  ambiguous entity terms: {}'.format(ambiguous_entity_terms))

print('')
print('Hard-negative check by onto_sim ({} sampled questions, {} without any candidate, median candidates {})'.format(
    len(sample_q), n_no_cand, cand_median))
print('  {:<10s}{:>7s}{:>10s}{:>10s}{:>12s}{:>10s}{:>10s}{:>10s}'.format(
    'group', 'rows', 'same', 'diff_fam', 'mean_sim', 'support', 'conflict', 'unnamed'))
for pool_name in POOL_NAMES:
    h = hard_stats[pool_name]
    print('  {:<10s}{:>7d}{:>10.1%}{:>10.1%}{:>12.3f}{:>10.1%}{:>10.1%}{:>10.1%}'.format(
        pool_name, h['rows'], h['relation']['same'], h['relation']['different_same_family'],
        h['mean_onto_sim'], h['with_support'], h['with_conflict'], h['relation']['unnamed']))
print('  blind sheet : {}   ({} rows)'.format(sheet, len(items)))
print('  answer key  : {}   (open after judging)'.format(key))

report = {
    'step': '06_ontology_neuro_new', 'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    't2s_backend': T2S_BACKEND, 'n_concepts': len(CONCEPTS), 'n_terms': len(term2concepts),
    'n_families': len(set(FAMILY.values())), 'ambiguous_terms': ambiguous,
    'questions': nq, 'with_concept': n_c, 'generic_only': n_gen, 'no_signal': len(no_signal),
    'multi_concept': n_multi,
    'answers': n_doc, 'answers_specific': n_spec, 'answers_unnamed': n_doc - n_spec,
    'answers_with_inherited': n_inh, 'answers_inherited_but_unnamed': n_unnamed_inh,
    'inherited_vs_mentioned': {'answers_with_both': len(both), 'agree': agree},
    'random_pair_relation': {k: round(prior[k] / n_prior, 4) for k in ('same', 'different', 'unnamed')},
    'question_concept_counts': {cname[c]: k for c, k in q_cnt.most_common()},
    'answer_concept_counts': {cname[c]: k for c, k in d_cnt.most_common()},
    'question_family_counts': dict(fam_cnt),
    'config': {'ENTITY_WEIGHT': ENTITY_WEIGHT, 'USE_FAERS': USE_FAERS, 'MAX_CANDIDATES': MAX_CANDIDATES,
               'HARD_TOPK': HARD_TOPK, 'N_HARD_Q': N_HARD_Q},
    'relation_layer': {
        'entities': len(ENTITIES), 'relations': n_rel_all, 'relations_used': n_rel_used,
        'relations_faers_only': n_faers_only, 'relations_by_type': dict(rel_by_type),
        'relations_by_source': dict(rel_by_src),
        'questions_with_entity': q_with_ent, 'labelled_questions_with_entity': q_lab_with_ent,
        'answers_with_entity': d_with_ent,
        'concepts_without_own_profile': no_profile, 'concepts_without_any_profile': no_related_profile,
        'entities_not_in_any_answer': dead_entities,
        'top_entities': {e: k for e, k in top_entities}},
    'onto_hard_check': {'sampled_questions': len(sample_q), 'questions_without_candidate': n_no_cand,
                        'median_candidates': cand_median, 'rows': len(items), 'groups': hard_stats,
                        'sheet': sheet, 'key': key},
}
if TAG_ASPECT:
    report['aspect_counts'] = dict(Counter(q.get('aspect') or '(none)' for q in q_info.values()))
with open(REPORT_DIR + '/06_ontology_' + TAG + '.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)


# ---- Markdown 报告 ----
def _md_table(headers, rows):
    esc = lambda v: str(v).replace('|', '/').replace('\n', ' ')
    align = [':---'] + ['---:'] * (len(headers) - 1)
    lines = ['| ' + ' | '.join(map(esc, headers)) + ' |',
             '| ' + ' | '.join(align) + ' |']
    lines += ['| ' + ' | '.join(map(esc, r)) + ' |' for r in rows]
    return '\n'.join(lines)


_md = [
    '# Step 06 (neuro) · 神经科学本体标注', '',
    'Generated {}. t2s backend: `{}`. Output dir: `{}/`.'.format(
        report['generated_at'], T2S_BACKEND, ONTO_DIR), '',
    '## 概念词典', '',
    _md_table(['Item', 'Value'], [
        ['Concepts', len(CONCEPTS)],
        ['Families', len(set(FAMILY.values()))],
        ['Synonym terms', len(term2concepts)],
        ['Ambiguous terms', len(ambiguous)],
        ['Generic terms (not labelled)', len(GENERIC_TERMS)],
    ]), '',
    '## 问题标注', '',
    _md_table(['Item', 'Value'], [
        ['Unique questions', '{:,}'.format(nq)],
        ['With a concept label', '{:,} ({:.1%})'.format(n_c, n_c / max(nq, 1))],
        ['Generic only (no specific label)', '{:,} ({:.1%})'.format(n_gen, n_gen / max(nq, 1))],
        ['No signal', '{:,} ({:.1%})'.format(len(no_signal), len(no_signal) / max(nq, 1))],
        ['Multiple concepts', '{:,}'.format(n_multi)],
    ]), '',
    '## 答案标注', '',
    _md_table(['Item', 'Value'], [
        ['Training answers', '{:,}'.format(n_doc)],
        ['Specific (body names a disease)', '{:,} ({:.1%})'.format(n_spec, n_spec / max(n_doc, 1))],
        ['Unnamed (no disease in body)', '{:,} ({:.1%})'.format(n_doc - n_spec, (n_doc - n_spec) / max(n_doc, 1))],
        ['With inherited label', '{:,} ({:.1%})'.format(n_inh, n_inh / max(n_doc, 1))],
        ['Inherited but body unnamed', '{:,}'.format(n_unnamed_inh)],
        ['Inherited ↔ body agree (of answers with both)', '{} / {} ({:.1%})'.format(
            agree, len(both), agree / len(both)) if both else '-'],
    ]), '',
    '## 随机配对先验（有标签问题）', '',
    _md_table(['Relation', 'Share'], [
        [k, '{:.1%}'.format(prior[k] / n_prior)] for k in ('same', 'different', 'unnamed')
    ]), '',
    '## 高频概念（前 15）', '',
    _md_table(['Concept', 'Questions', 'Answers naming it'],
              [[cname[c], '{:,}'.format(k), '{:,}'.format(d_cnt.get(c, 0))]
               for c, k in q_cnt.most_common(15)]), '',
    '## 问题大类分布', '',
    _md_table(['Family', 'Questions'], [[f, '{:,}'.format(k)] for f, k in fam_cnt.most_common()]), '',
    '## 关系层（症状 / 药物 / 治疗 / 检查 / 病因）', '',
    _md_table(['Item', 'Value'], [
        ['Entities', len(ENTITIES)],
        ['Relations total', n_rel_all],
        ['Relations used (after USE_FAERS filter)', n_rel_used],
        ['FAERS-only relations', n_faers_only],
        ['USE_FAERS', str(USE_FAERS)],
        ['Questions mentioning an entity', '{:,} ({:.1%})'.format(q_with_ent, q_with_ent / max(nq, 1))],
        ['Labelled questions with entity', '{:,} ({:.1%})'.format(q_lab_with_ent, q_lab_with_ent / max(n_c, 1))],
        ['Answers mentioning an entity', '{:,} ({:.1%})'.format(d_with_ent, d_with_ent / max(n_doc, 1))],
        ['Entities never in any answer', '{}'.format(enames(dead_entities[:20]))],
        ['Concepts without own profile', names(no_profile)],
        ['Concepts without any profile', names(no_related_profile) or '-'],
    ]), '',
    _md_table(['Entity type', 'Relations'], [[t, k] for t, k in sorted(rel_by_type.items())]), '',
    _md_table(['Source', 'Relations'], [[t, k] for t, k in sorted(rel_by_src.items())]), '',
    '## 难负例抽检（onto_sim，不依赖模型）', '',
    'Sampled {} questions, {} without candidate, median candidates {}.'.format(
        len(sample_q), n_no_cand, cand_median), '',
    _md_table(['Group', 'Rows', 'same', 'diff_family', 'unnamed', 'mean_sim', 'support%', 'conflict%'],
              [[pool_name,
                hard_stats[pool_name]['rows'],
                '{:.1%}'.format(hard_stats[pool_name]['relation']['same']),
                '{:.1%}'.format(hard_stats[pool_name]['relation']['different_same_family']),
                '{:.1%}'.format(hard_stats[pool_name]['relation']['unnamed']),
                '{:.3f}'.format(hard_stats[pool_name]['mean_onto_sim']),
                '{:.1%}'.format(hard_stats[pool_name]['with_support']),
                '{:.1%}'.format(hard_stats[pool_name]['with_conflict'])]
               for pool_name in POOL_NAMES]), '',
    'Blind sheet: `{}` ({} rows). Key: `{}`.'.format(sheet, len(items), key), '',
    '## 输出文件', '',
    '`{}/` — concepts.json, entities.json, question_labels.jsonl, doc_labels.jsonl, '
    'check_*.csv, unlabelled_neuro_terms.csv, onto_hard_check*.csv, onto_hard_candidates.jsonl'.format(ONTO_DIR),
]

with open(REPORT_DIR + '/06_ontology_' + TAG + '.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(_md) + '\n')

print('')
print('Outputs: {}/  (concepts.json, entities.json, question_labels.jsonl, doc_labels.jsonl,'.format(ONTO_DIR))
print('         check_*.csv, unlabelled_neuro_terms.csv, onto_hard_check*.csv, onto_hard_candidates.jsonl)')
print('Reports: {}/06_ontology_{}.json / .md'.format(REPORT_DIR, TAG))
print('Next   : fill onto_hard_check.csv first; if not_same is clearly safer than all, add onto_sim to 08. Dont forget run 07')
