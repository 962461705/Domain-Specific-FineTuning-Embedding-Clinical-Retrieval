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
    ap.add_argument('--out-dir', default=os.environ.get('LIVER_OUT_DIR', '/home/liver/data/liver_dataset'))
    ap.add_argument('--tag', default='new', help='输出目录后缀，默认 new')
    ap.add_argument('--seed', type=int, default=42)
    a, _unknown = ap.parse_known_args()    
    return a


ARGS = _args()
OUT_DIR = ARGS.out_dir
TAG = ARGS.tag
SEED = ARGS.seed

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
# 基础函数（normalize / qid_of 必须与 04、07、08 完全一致）
# =============================================================================


def normalize(text):
    '''与 01–04 完全一致的归一化。qid 依赖它，不能改。这里不做繁简转换。'''
    text = unicodedata.normalize('NFKC', text or '').lower()
    return re.sub(r'[^\u4e00-\u9fa5a-z0-9]', '', text)


def qid_of(nq):
    '''与 04、07、08 一致：归一化问题的 md5 前 16 位'''
    return hashlib.md5(nq.encode()).hexdigest()[:16]


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


# 繁体转简体：00–04 没有做这一步，所以训练集里可能还有繁体。只在匹配词典时转换。
try:
    from opencc import OpenCC
    _t2s = OpenCC('t2s').convert
    T2S_BACKEND = 'opencc'
except ImportError:
    _FALLBACK = str.maketrans({
        '膽': '胆', '脈': '脉', '靜': '静', '壓': '压', '腦': '脑', '變': '变', '狀': '状',
        '腫': '肿', '蟲': '虫', '體': '体', '臟': '脏', '藥': '药', '療': '疗', '護': '护',
        '診': '诊', '斷': '断', '異': '异', '傳': '传', '會': '会', '區': '区', '學': '学',
        '證': '证', '瘍': '疡', '膿': '脓', '腸': '肠', '點': '点', '關': '关', '係': '系',
        '歲': '岁', '嚴': '严', '復': '复', '發': '发', '熱': '热', '質': '质', '應': '应',
        '該': '该', '與': '与', '無': '无', '從': '从', '這': '这', '個': '个', '後': '后',
        '時': '时', '間': '间', '飲': '饮', '纖': '纤', '維': '维', '轉': '转', '檢': '检',
        '術': '术', '醫': '医', '遺': '遗', '種': '种', '類': '类', '長': '长', '為': '为',
        '請': '请', '問': '问', '過': '过', '說': '说', '對': '对', '產': '产', '婦': '妇',
        '黃': '黄', '膚': '肤', '紅': '红', '腎': '肾', '雞': '鸡', '魚': '鱼', '糞': '粪',
        '隱': '隐', '輸': '输', '麼': '么', '嗎': '吗', '還': '还', '讓': '让', '機': '机',
        '塊': '块', '親': '亲', '兒': '儿',
    })
    _t2s = lambda s: s.translate(_FALLBACK)
    T2S_BACKEND = 'built-in table (partial)'
    print('opencc not installed: using a small built-in Traditional->Simplified table. '
          'pip install opencc-python-reimplemented for full coverage.')


def prep(text):
    '''匹配词典前的统一处理：全角转半角、繁转简、英文转大写。与原 06 相同。'''
    return _t2s(unicodedata.normalize('NFKC', text or '')).upper()

# %%
# =============================================================================
# 概念层：肝病词典
# =============================================================================
# 每个概念：id / name / parent / icd10 / icd10_title / wiki / synonyms / extra
# 兄弟概念 = 同一个 parent 下的概念。

W = 'https://zh.wikipedia.org/wiki/'
VIRAL_FULLNAME = ('与维基重定向“乙型病毒性肝炎”同一构词方式；'
                  '对应 WHO ICD-10 B15–B17 标题')

CONCEPTS = [
    {'id': 'viral_hepatitis', 'name': '病毒性肝炎', 'parent': None,
     'icd10': ['B15-B19'], 'icd10_title': 'Viral hepatitis',
     'wiki': W + '病毒性肝炎',
     'synonyms': ['病毒性肝炎', '病毒型肝炎'],
     'extra': {}},

    {'id': 'hep_a', 'name': '甲型肝炎', 'parent': 'viral_hepatitis',
     'icd10': ['B15'], 'icd10_title': 'Acute hepatitis A',
     'wiki': W + '甲型肝炎',
     'synonyms': ['甲型肝炎', 'A型肝炎', '甲肝', '甲肝病毒', '甲型肝炎病毒', 'A型肝炎病毒'],
     'extra': {'甲型病毒性肝炎': VIRAL_FULLNAME}},

    {'id': 'hep_b', 'name': '乙型肝炎', 'parent': 'viral_hepatitis',
     'icd10': ['B16', 'B18.1'],
     'icd10_title': 'Acute hepatitis B; Chronic viral hepatitis B without delta-agent',
     'wiki': W + '乙型肝炎',
     # 大三阳、小三阳在中文维基里是“乙型肝炎”的重定向，所以归入乙肝本身
     'synonyms': ['乙型肝炎', '乙型病毒性肝炎', 'B肝', 'B型肝炎', '乙肝', '大三阳', '小三阳',
                  '血清性肝炎', '大三阳小三阳'],
     'extra': {
         '乙型肝炎病毒': W + '乙型肝炎病毒', '乙肝病毒': W + '乙型肝炎病毒',
         'B型肝炎病毒': W + '乙型肝炎病毒', 'HBV': W + '乙型肝炎病毒',
         '乙型肝炎表面抗原': W + '乙型肝炎表面抗原', '乙肝表面抗原': W + '乙型肝炎表面抗原',
         '澳抗': W + '乙型肝炎表面抗原', 'HBsAg': W + '乙型肝炎表面抗原',
         '乙型肝炎疫苗': W + '乙型肝炎疫苗', '乙肝疫苗': W + '乙型肝炎疫苗',
         '两对半': W + '乙型肝炎（正文）',
     }},

    {'id': 'hep_c', 'name': '丙型肝炎', 'parent': 'viral_hepatitis',
     'icd10': ['B17.1', 'B18.2'],
     'icd10_title': 'Acute hepatitis C; Chronic viral hepatitis C',
     'wiki': W + '丙型肝炎',
     'synonyms': ['丙型肝炎', 'C型肝炎', '丙肝', 'C肝'],
     'extra': {
         '丙型病毒性肝炎': VIRAL_FULLNAME,
         '丙型肝炎病毒': W + '丙型肝炎病毒', '丙肝病毒': W + '丙型肝炎病毒',
         'C型肝炎病毒': W + '丙型肝炎病毒',
     }},

    {'id': 'hep_d', 'name': '丁型肝炎', 'parent': 'viral_hepatitis',
     'icd10': ['B17.0', 'B18.0'],
     'icd10_title': 'Acute delta-(super)infection in chronic hepatitis B; '
                    'Chronic viral hepatitis B with delta-agent',
     'wiki': W + '丁型肝炎',
     'synonyms': ['丁型肝炎', 'D型肝炎', '丁肝'],
     'extra': {'丁型病毒性肝炎': VIRAL_FULLNAME}},

    {'id': 'hep_e', 'name': '戊型肝炎', 'parent': 'viral_hepatitis',
     'icd10': ['B17.2'], 'icd10_title': 'Acute hepatitis E',
     'wiki': W + '戊型肝炎',
     'synonyms': ['戊型肝炎', 'E型肝炎', '戊肝'],
     'extra': {'戊型病毒性肝炎': VIRAL_FULLNAME}},

    {'id': 'alcoholic_liver', 'name': '酒精性肝病', 'parent': None,
     'icd10': ['K70'], 'icd10_title': 'Alcoholic liver disease',
     'wiki': W + '酒精性肝病',
     'synonyms': ['酒精性肝病', '酒精相关性肝病', '酒精性肝病变', '酒精性肝疾病',
                  '酒精性脂肪肝', 'ARLD'],
     'extra': {
         '酒精肝': 'WHO ICD-10 K70 “Alcoholic liver disease” 的常用简称；见 ' + W + '酒精性肝病',
     }},

    {'id': 'alcoholic_hepatitis', 'name': '酒精性肝炎', 'parent': 'alcoholic_liver',
     'icd10': ['K70.1'], 'icd10_title': 'Alcoholic hepatitis',
     'wiki': W + '酒精性肝炎',
     'synonyms': ['酒精性肝炎'],
     'extra': {}},

    {'id': 'toxic_liver', 'name': '中毒性肝病', 'parent': None,
     'icd10': ['K71'], 'icd10_title': 'Toxic liver disease',
     'wiki': W + '肝毒性',
     'synonyms': ['中毒性肝病', '肝毒性', '肝脏毒性'],
     'extra': {
         '药物性肝损伤': 'WHO ICD-10 K71 Incl. “drug-induced liver disease”；中文维基无条目',
         '药物性肝炎':   'WHO ICD-10 K73 Excl. “hepatitis (chronic): drug-induced (K71.-)”；中文维基无条目',
         '药物性肝损害': 'WHO ICD-10 K71 Incl. “drug-induced liver disease”；中文维基无条目',
     }},

    {'id': 'liver_failure', 'name': '肝衰竭', 'parent': None,
     'icd10': ['K72'], 'icd10_title': 'Hepatic failure, not elsewhere classified',
     'wiki': W + '肝衰竭',
     'synonyms': ['肝衰竭', '肝功能衰竭', '肝脏衰竭', 'ACLF'],
     'extra': {}},

    {'id': 'hepatic_encephalopathy', 'name': '肝性脑病', 'parent': 'liver_failure',
     'icd10': ['K72', 'G94.3'],
     'icd10_title': 'Hepatic failure, NEC (incl. hepatic coma / encephalopathy NOS)',
     'wiki': W + '肝性脑病',
     'synonyms': ['肝性脑病', '肝性脑病变', '肝因性脑病变'],
     'extra': {
         '肝昏迷': 'WHO ICD-10 K72 Incl. “hepatic coma NOS”；中文维基无单独条目',
     }},

    {'id': 'chronic_hepatitis', 'name': '慢性肝炎', 'parent': None,
     'icd10': ['K73'], 'icd10_title': 'Chronic hepatitis, not elsewhere classified',
     'wiki': None,
     'synonyms': ['慢性肝炎'],
     'extra': {}},

    {'id': 'fibrosis_cirrhosis', 'name': '肝纤维化和肝硬化', 'parent': None,
     'icd10': ['K74'], 'icd10_title': 'Fibrosis and cirrhosis of liver',
     'wiki': None,
     'synonyms': [],
     'extra': {}},

    {'id': 'liver_fibrosis', 'name': '肝纤维化', 'parent': 'fibrosis_cirrhosis',
     'icd10': ['K74.0'], 'icd10_title': 'Hepatic fibrosis',
     'wiki': None,
     'synonyms': ['肝纤维化'],
     'extra': {}},

    {'id': 'cirrhosis', 'name': '肝硬化', 'parent': 'fibrosis_cirrhosis',
     'icd10': ['K74.6'], 'icd10_title': 'Other and unspecified cirrhosis of liver',
     'wiki': W + '肝硬化',
     'synonyms': ['肝硬化', '肝硬变'],
     'extra': {}},

    {'id': 'pbc', 'name': '原发性胆汁性胆管炎', 'parent': 'fibrosis_cirrhosis',
     'icd10': ['K74.3'], 'icd10_title': 'Primary biliary cirrhosis',
     'wiki': W + '原发性胆汁性胆管炎',
     'synonyms': ['原发性胆汁性胆管炎', '原发性胆汁性肝硬化', '原发性胆汁性肝硬变', 'PBC'],
     'extra': {}},

    {'id': 'liver_abscess', 'name': '肝脓肿', 'parent': None,
     'icd10': ['K75.0', 'A06.4'], 'icd10_title': 'Abscess of liver; Amoebic liver abscess',
     'wiki': W + '肝脓肿',
     'synonyms': ['肝脓肿'],
     'extra': {
         '阿米巴性肝脓肿': '中文维基 ICD-10 第一章 A06.4 “阿米巴性肝脓疡”',
         '肝脓疡': '中文维基 ICD-10 第一章 A06.4 “阿米巴性肝脓疡”',
     }},

    {'id': 'autoimmune_hepatitis', 'name': '自身免疫性肝炎', 'parent': None,
     'icd10': ['K75.4'], 'icd10_title': 'Autoimmune hepatitis',
     'wiki': W + '自身免疫性肝炎',
     'synonyms': ['自身免疫性肝炎', '自体免疫性肝炎', '自体免疫肝炎'],
     'extra': {
         '自免肝':     '“自身免疫性肝炎”的通用简称；中文维基无重定向',
         '自免性肝炎': '“自身免疫性肝炎”的通用简称；中文维基无重定向',
     }},

    {'id': 'fatty_liver', 'name': '脂肪肝', 'parent': None,
     'icd10': ['K76.0'], 'icd10_title': 'Fatty (change of) liver, not elsewhere classified',
     'wiki': W + '脂肪肝',
     'synonyms': ['脂肪肝', 'FLD'],
     'extra': {
         '脂肪性肝病': '维基“代谢功能障碍相关脂肪性肝病”条目标题',
     }},

    {'id': 'nafld', 'name': '代谢功能障碍相关脂肪性肝病', 'parent': 'fatty_liver',
     'icd10': ['K76.0', 'K75.8'],
     'icd10_title': 'Fatty (change of) liver; Other specified inflammatory liver diseases (NASH)',
     'wiki': W + '代谢功能障碍相关脂肪性肝病',
     'synonyms': ['代谢功能障碍相关脂肪性肝病', '非酒精性脂肪肝', '非酒精性脂肪性肝炎',
                  '非酒精性脂肪肝病', '非酒精性脂肪性肝病', 'MASLD', 'NAFLD', 'MAFLD'],
     'extra': {
         '代谢功能障碍相关脂肪性肝炎': '中文维基条目名（页面暂缺）',
     }},

    {'id': 'portal_hypertension', 'name': '门静脉高压', 'parent': None,
     'icd10': ['K76.6'], 'icd10_title': 'Portal hypertension',
     'wiki': W + '门静脉高压',
     'synonyms': ['门静脉高压', '门静脉高压症', '门脉高压', '肝门静脉高压'],
     'extra': {}},

    {'id': 'hepatorenal', 'name': '肝肾综合征', 'parent': None,
     'icd10': ['K76.7'], 'icd10_title': 'Hepatorenal syndrome',
     'wiki': W + '肝肾综合征',
     'synonyms': ['肝肾综合征', '肝肾症候群', '肝肾综合症'],
     'extra': {}},

    {'id': 'budd_chiari', 'name': '布加综合征', 'parent': None,
     'icd10': ['I82.0'], 'icd10_title': 'Budd-Chiari syndrome',
     'wiki': W + '布加综合征',
     'synonyms': ['布加综合征', '布加氏综合征', '布-加氏综合征', '布-加综合征',
                  '巴德-吉亚利综合征', '布加症候群'],
     'extra': {}},

    {'id': 'liver_cancer', 'name': '肝癌', 'parent': None,
     'icd10': ['C22'], 'icd10_title': 'Malignant neoplasm of liver and intrahepatic bile ducts',
     'wiki': W + '肝癌',
     'synonyms': ['肝癌'],
     'extra': {
         '肝脏恶性肿瘤': '中文维基 ICD-10 第二章 C22 “肝和肝内胆管的恶性肿瘤”',
         '原发性肝癌': '中文维基“肝细胞癌”重定向“原发性肝细胞癌”',
     }},

    {'id': 'hcc', 'name': '肝细胞癌', 'parent': 'liver_cancer',
     'icd10': ['C22.0'], 'icd10_title': 'Liver cell carcinoma',
     'wiki': W + '肝細胞癌',
     'synonyms': ['肝细胞癌', '原发性肝细胞癌', '肝细胞癌症', '肝细胞性肝癌', 'HCC'],
     'extra': {}},

    {'id': 'cholangiocarcinoma', 'name': '肝内胆管癌', 'parent': 'liver_cancer',
     'icd10': ['C22.1'], 'icd10_title': 'Intrahepatic bile duct carcinoma',
     'wiki': W + '膽管癌',
     'synonyms': ['肝内胆管癌', '胆管细胞癌'],
     'extra': {}},

    {'id': 'hepatoblastoma', 'name': '肝母细胞瘤', 'parent': 'liver_cancer',
     'icd10': ['C22.2'], 'icd10_title': 'Hepatoblastoma',
     'wiki': None,
     'synonyms': ['肝母细胞瘤'],
     'extra': {}},

    {'id': 'benign_liver_tumor', 'name': '肝良性肿瘤', 'parent': None,
     'icd10': ['D13.4'], 'icd10_title': 'Benign neoplasm: Liver',
     'wiki': None,
     'synonyms': ['肝良性肿瘤', '肝脏良性肿瘤'],
     'extra': {}},

    {'id': 'liver_hemangioma', 'name': '肝血管瘤', 'parent': 'benign_liver_tumor',
     'icd10': ['D18.0'], 'icd10_title': 'Haemangioma, any site',
     'wiki': None,
     'synonyms': ['肝血管瘤', '肝海绵状血管瘤'],
     'extra': {}},

    {'id': 'liver_cyst', 'name': '肝囊肿', 'parent': None,
     'icd10': ['K76.8'], 'icd10_title': 'Other specified diseases of liver (incl. simple cyst of liver)',
     'wiki': None,
     'synonyms': ['肝囊肿'],
     'extra': {}},

    {'id': 'polycystic_liver', 'name': '多囊肝', 'parent': None,
     'icd10': ['Q44.6'], 'icd10_title': 'Cystic disease of liver',
     'wiki': None,
     'synonyms': ['多囊肝'],
     'extra': {}},

    {'id': 'hydatid_liver', 'name': '肝包虫病', 'parent': None,
     'icd10': ['B67.0', 'B67.5', 'B67.8'],
     'icd10_title': 'Echinococcus infection of liver',
     'wiki': W + '棘球蚴病',
     'synonyms': ['肝包虫病', '肝包虫', '包虫病', '棘球蚴病', '包囊虫病'],
     'extra': {}},

    {'id': 'clonorchiasis', 'name': '华支睾吸虫病', 'parent': None,
     'icd10': ['B66.1'], 'icd10_title': 'Clonorchiasis',
     'wiki': W + '中華肝吸蟲',
     'synonyms': ['华支睾吸虫病', '华支睾吸虫', '中华肝吸虫', '中华枝睾吸虫', '华枝睾吸虫'],
     'extra': {
         '肝吸虫':   '维基重定向“中华肝吸虫”的通用简称；中文维基无单独页面',
         '肝吸虫病': '维基重定向“中华肝吸虫”的通用简称；中文维基无单独页面',
     }},

    {'id': 'wilson', 'name': '肝豆状核变性', 'parent': None,
     'icd10': ['E83.0'], 'icd10_title': 'Disorders of copper metabolism',
     'wiki': W + '肝豆状核变性',
     'synonyms': ['肝豆状核变性', '威尔森氏症', '威尔逊病', '威尔逊氏病', '威尔逊氏症'],
     'extra': {}},

    {'id': 'hemochromatosis', 'name': '血色病', 'parent': None,
     'icd10': ['E83.1'], 'icd10_title': 'Disorders of iron metabolism',
     'wiki': W + '血色病',
     'synonyms': ['血色病', '血色沉着病', '血色素沉着症', '血铁沉积症', '铁质沉积症',
                  '血色素沉着病'],
     'extra': {}},
]

# %%
# =============================================================================
# 大类（family）。按 ICD-10 章节区间归并，供 08 做更保守的“不同病”判断
# =============================================================================

FAMILY = {
    'viral_hepatitis': 'viral', 'hep_a': 'viral', 'hep_b': 'viral', 'hep_c': 'viral',
    'hep_d': 'viral', 'hep_e': 'viral', 'chronic_hepatitis': 'viral',
    'alcoholic_liver': 'alcohol', 'alcoholic_hepatitis': 'alcohol',
    'toxic_liver': 'toxic',
    'liver_failure': 'failure', 'hepatic_encephalopathy': 'failure',
    'fibrosis_cirrhosis': 'cirrhosis', 'liver_fibrosis': 'cirrhosis', 'cirrhosis': 'cirrhosis',
    'pbc': 'cirrhosis',
    'liver_abscess': 'other_inflammatory', 'autoimmune_hepatitis': 'other_inflammatory',
    'fatty_liver': 'fatty', 'nafld': 'fatty',
    'portal_hypertension': 'portal_vascular', 'hepatorenal': 'portal_vascular',
    'budd_chiari': 'portal_vascular',
    'liver_cancer': 'cancer', 'hcc': 'cancer', 'cholangiocarcinoma': 'cancer',
    'hepatoblastoma': 'cancer',
    'benign_liver_tumor': 'benign_tumor', 'liver_hemangioma': 'benign_tumor',
    'liver_cyst': 'cyst', 'polycystic_liver': 'cyst',
    'hydatid_liver': 'parasitic', 'clonorchiasis': 'parasitic',
    'wilson': 'metabolic', 'hemochromatosis': 'metabolic',
}

# 这些词只说明“和肝有关”，不足以确定具体疾病；匹配到它们时不贴概念标签。
GENERIC_TERMS = ['肝炎', '肝病', '肝脏疾病', '肝功能异常', '肝损伤', '转氨酶', '肝腹水', '腹水']

# %%
# =============================================================================
# 新增：关系层的来源
# =============================================================================

WD_URL = 'https://www.wikidata.org/wiki/'
MSD_URL = 'https://www.msdmanuals.cn/home/'
MEDLINEPLUS_URL = 'https://wsearch.nlm.nih.gov/ws/query?db=healthTopics&term='

CONCEPT_WD = {
    'viral_hepatitis': ['Q1983841'], 'hep_a': ['Q157661'], 'hep_b': ['Q6853'],
    'hep_c': ['Q154869'], 'hep_d': ['Q327281'], 'hep_e': ['Q326643'],
    'alcoholic_liver': ['Q558404'], 'alcoholic_hepatitis': ['Q3144986'],
    'toxic_liver': ['Q1349821'], 'liver_failure': ['Q970208', 'Q1192530'],
    'hepatic_encephalopathy': ['Q642548'], 'cirrhosis': ['Q147778'], 'pbc': ['Q1072420'],
    'liver_abscess': ['Q1811092', 'Q2820891', 'Q7263138'], 'autoimmune_hepatitis': ['Q786844'],
    'fatty_liver': ['Q6058862'], 'nafld': ['Q1546498'], 'portal_hypertension': ['Q124604'],
    'hepatorenal': ['Q247131'], 'budd_chiari': ['Q183098'], 'liver_cancer': ['Q623031'],
    'hcc': ['Q1148337'], 'cholangiocarcinoma': ['Q124292'], 'hepatoblastoma': ['Q1607658'],
    'liver_hemangioma': ['Q2596039'], 'liver_cyst': ['Q1811141'], 'polycystic_liver': ['Q246002'],
    'hydatid_liver': ['Q777087'], 'clonorchiasis': ['Q1102300'], 'wilson': ['Q117121'],
    'hemochromatosis': ['Q2025687', 'Q3144934'],
}

MSD_PAGE = {
    'hep_a': ['liver-and-gallbladder-disorders/hepatitis/hepatitis-a'],
    'hep_b': ['liver-and-gallbladder-disorders/hepatitis/hepatitis-b-acute', 'liver-and-gallbladder-disorders/hepatitis/hepatitis-b-chronic'],
    'hep_c': ['liver-and-gallbladder-disorders/hepatitis/hepatitis-c-acute', 'liver-and-gallbladder-disorders/hepatitis/hepatitis-c-chronic'],
    'hep_d': ['liver-and-gallbladder-disorders/hepatitis/hepatitis-d'],
    'hep_e': ['liver-and-gallbladder-disorders/hepatitis/hepatitis-e'],
    'alcoholic_liver': ['liver-and-gallbladder-disorders/alcohol-related-liver-disease/alcohol-related-liver-disease'],
    'toxic_liver': ['liver-and-gallbladder-disorders/drugs-and-the-liver/liver-injury-caused-by-drugs'],
    'liver_failure': ['liver-and-gallbladder-disorders/manifestations-of-liver-disease/liver-failure'],
    'hepatic_encephalopathy': ['liver-and-gallbladder-disorders/manifestations-of-liver-disease/hepatic-encephalopathy'],
    'chronic_hepatitis': ['liver-and-gallbladder-disorders/hepatitis/overview-of-chronic-hepatitis'],
    'liver_fibrosis': ['liver-and-gallbladder-disorders/fibrosis-and-cirrhosis-of-the-liver/fibrosis-of-the-liver'],
    'cirrhosis': ['liver-and-gallbladder-disorders/fibrosis-and-cirrhosis-of-the-liver/cirrhosis-of-the-liver'],
    'pbc': ['liver-and-gallbladder-disorders/fibrosis-and-cirrhosis-of-the-liver/primary-biliary-cholangitis-pbc'],
    'autoimmune_hepatitis': ['liver-and-gallbladder-disorders/hepatitis/overview-of-chronic-hepatitis'],
    'fatty_liver': ['liver-and-gallbladder-disorders/manifestations-of-liver-disease/steatotic-liver-disease'],
    'nafld': ['liver-and-gallbladder-disorders/manifestations-of-liver-disease/steatotic-liver-disease'],
    'portal_hypertension': ['liver-and-gallbladder-disorders/manifestations-of-liver-disease/portal-hypertension'],
    'budd_chiari': ['liver-and-gallbladder-disorders/blood-vessel-disorders-of-the-liver/budd-chiari-syndrome'],
    'hcc': ['cancer/liver-cancers/hepatocellular-carcinoma'],
    'hepatoblastoma': ['cancer/liver-cancers/other-primary-liver-cancers'],
    'liver_hemangioma': ['liver-and-gallbladder-disorders/tumors-of-the-liver/hemangiomas-of-the-liver'],
    'liver_cyst': ['liver-and-gallbladder-disorders/tumors-of-the-liver/overview-of-liver-tumors'],
    'polycystic_liver': ['liver-and-gallbladder-disorders/tumors-of-the-liver/overview-of-liver-tumors'],
    'hydatid_liver': ['infections/parasitic-infections-cestodes-tapeworms/echinococcosis-dog-tapeworm-infection'],
    'clonorchiasis': ['infections/parasitic-infections-trematodes-flukes/fluke-infections-of-the-liver'],
    'hemochromatosis': ['blood-disorders/iron-overload/hemochromatosis'],
}

ZHWIKI_PAGE = {
    'alcoholic_hepatitis': '酒精性肝炎', 'toxic_liver': '肝毒性', 'liver_failure': '肝衰竭',
    'liver_abscess': '肝脓肿', 'fatty_liver': '脂肪肝', 'nafld': '代谢功能障碍相关脂肪性肝病',
    'budd_chiari': '布加综合征', 'clonorchiasis': '中華肝吸蟲', 'hemochromatosis': '血色沉著病',
}

# %%
# =============================================================================
# 新增：实体层（症状 / 药物 / 治疗 / 检查 / 病因）
# =============================================================================
# 每个实体：id -> (type, 名称, Wikidata 条目或 None, 匹配用的中文词)
# 匹配词只取 Wikidata 中文标签/别名，或 MSD、中文维基原文里出现过的写法。
# 刻意不用的 Wikidata 写法：weight loss 的中文标签“减肥”（意思不对，改用 MSD 的“体重减轻”）；
# 单字别名（如“吐”“咳”）；值本身是疾病的“症状”声明（肝炎、癌症、肝硬化等，交给概念层处理）。
# 消胆胺、考来烯胺分别取自 MSD 不同页面，这里没有把它们合并成一个实体。

ENTITY_TABLE = {
    # ---------------- 症状 / 体征 ----------------
    'jaundice':          ('symptom', '黄疸', 'Q133244', ['黄疸', '黄胆', '黄染', '皮肤发黄', '眼白发黄']),
    'nausea':            ('symptom', '恶心', 'Q186889', ['恶心', '反胃']),
    'vomiting':          ('symptom', '呕吐', 'Q127076', ['呕吐', '作呕']),
    'anorexia':          ('symptom', '食欲不振', 'Q254327', ['食欲不振', '食欲缺乏', '食欲减退', '胃口不佳', '食欲欠佳']),
    'fatigue':           ('symptom', '疲劳', 'Q9690', ['疲倦', '疲劳', '疲乏', '乏力']),
    'malaise':           ('symptom', '全身不适', 'Q2673323', ['身体不适', '全身不适', '周身不适', '虚弱', '全身乏力']),
    'fever':             ('symptom', '发热', 'Q38933', ['发热', '发烧', '低热', '低烧']),
    'abdominal_pain':    ('symptom', '腹痛', 'Q183425', ['腹痛', '肚痛', '右上腹痛', '上腹部疼痛', '腹部疼痛',
                                                       '肝区隐痛', '上腹不适', '上腹部不适', '腹部不适']),
    'hepatomegaly':      ('symptom', '肝肿大', 'Q1362864', ['肝肿大', '肝脏肿大', '肝脾肿大']),
    'splenomegaly':      ('symptom', '脾肿大', 'Q1129121', ['脾肿大', '脾大', '脾脏肿大']),
    'ascites':           ('symptom', '腹水', 'Q193889', ['腹水', '腹腔积液', '腹腔内积液']),
    'edema':             ('symptom', '水肿', 'Q152234', ['水肿', '浮肿']),
    'arthralgia':        ('symptom', '关节痛', 'Q683498', ['关节痛', '关节疼痛']),
    'pruritus':          ('symptom', '瘙痒', None, ['瘙痒']),
    'dark_urine':        ('symptom', '尿色变深', None, ['尿色变深', '深色的尿液']),
    'pale_stool':        ('symptom', '大便颜色变浅', None, ['大便颜色变浅', '粪便颜色变浅']),
    'spider_nevi':       ('symptom', '蜘蛛痣', None, ['蜘蛛痣']),
    'palmar_erythema':   ('symptom', '肝掌', None, ['肝掌', '手掌发红']),
    'varices':           ('symptom', '静脉曲张', None, ['静脉曲张']),
    'hematemesis':       ('symptom', '呕血', None, ['呕血']),
    'bloody_stool':      ('symptom', '便血', None, ['便血', '柏油样']),
    'bleeding_tendency': ('symptom', '出血倾向', None, ['出血倾向', '凝血障碍', '凝血病', '瘀伤']),
    'confusion':         ('symptom', '意识障碍', None, ['精神错乱', '意识障碍', '定向障碍']),
    'drowsiness':        ('symptom', '昏睡', None, ['昏睡', '困倦']),
    'coma':              ('symptom', '昏迷', 'Q159557', ['昏迷']),
    'tremor':            ('symptom', '扑翼样震颤', 'Q223907', ['手震', '扑翼样震颤']),
    'fetor_hepaticus':   ('symptom', '肝病性口臭', 'Q3280552', ['肝病性口臭', '发霉的甜味']),
    'weight_loss':       ('symptom', '体重减轻', 'Q718113', ['体重减轻', '体重下降']),
    'diarrhea':          ('symptom', '腹泻', 'Q40878', ['腹泻', '拉肚子']),
    'cough':             ('symptom', '咳嗽', 'Q35805', ['咳嗽']),
    'hemoptysis':        ('symptom', '咳血', 'Q647099', ['咳血', '咯血']),
    'heartburn':         ('symptom', '胃灼热', 'Q537297', ['胃灼热']),
    'urticaria':         ('symptom', '荨麻疹', None, ['荨麻疹']),
    'abdominal_distension': ('symptom', '腹胀', None, ['腹胀', '腹部膨胀']),
    'telangiectasia':    ('symptom', '毛细血管扩张', 'Q1634800', ['毛细血管扩张', '毛细管扩张']),
    'caput_medusae':     ('symptom', '血管怒张', 'Q1035226', ['血管怒张']),
    'kf_ring':           ('symptom', '角膜色素环', 'Q1423745', ['角膜色素环']),
    'dry_mouth_eyes':    ('symptom', '口眼干燥', None, ['口眼干燥', '口干', '眼干']),
    'xanthoma':          ('symptom', '黄瘤', None, ['黄瘤', '黄斑瘤']),
    'steatorrhea':       ('symptom', '脂肪泻', None, ['脂肪泻']),
    'bronze_skin':       ('symptom', '青铜色皮肤', None, ['青铜色皮肤']),
    'dupuytren':         ('symptom', '掌腱膜挛缩', None, ['掌腱膜挛缩']),
    'gynecomastia':      ('symptom', '男性乳房发育', None, ['乳房增大', '男性乳房发育']),
    'clubbing':          ('symptom', '杵状指', None, ['杵状指']),
    'amenorrhea':        ('symptom', '停经', None, ['停经']),
    'chills':            ('symptom', '寒战', None, ['寒战']),

    # ---------------- 药物 ----------------
    'entecavir':       ('drug', '恩替卡韦', 'Q27126952', ['恩替卡韦']),
    'tenofovir':       ('drug', '替诺福韦', 'Q27132753', ['替诺福韦', '替诺福韦酯', '泰诺福韦']),
    'taf':             ('drug', '丙酚替诺福韦', 'Q22075912', ['丙酚替诺福韦']),
    'peginterferon':   ('drug', '聚乙二醇干扰素', 'Q420056', ['聚乙二醇干扰素', '聚乙二醇化干扰素']),
    'interferon':      ('drug', '干扰素', 'Q188269', ['干扰素']),
    'adefovir':        ('drug', '阿德福韦', 'Q353551', ['阿德福韦']),
    'lamivudine':      ('drug', '拉米夫定', 'Q422631', ['拉米夫定', '贺普丁', '干安能']),
    'telbivudine':     ('drug', '替比夫定', None, ['替比夫定']),
    'bicyclol':        ('drug', '双环醇', 'Q27271958', ['双环醇']),
    'hbig':            ('drug', '乙型肝炎免疫球蛋白', None, ['乙型肝炎免疫球蛋白', '肝炎免疫球蛋白']),
    'sofosbuvir':      ('drug', '索磷布韦', 'Q2502747', ['索磷布韦', '索非布韦']),
    'ledipasvir':      ('drug', '雷迪帕韦', 'Q15409409', ['雷迪帕韦', '来迪派韦']),
    'sof_led':         ('drug', '复方索非布韦雷迪帕韦片', 'Q19597688', ['复方索非布韦雷迪帕韦片']),
    'velpatasvir':     ('drug', '维帕他韦', None, ['维帕他韦']),
    'daclatasvir':     ('drug', '达拉他韦', 'Q5207712', ['达拉他韦', '达卡他韦']),
    'simeprevir':      ('drug', '司美匹韦', 'Q7517689', ['司美匹韦']),
    'elbasvir':        ('drug', '艾尔巴韦', None, ['艾尔巴韦']),
    'grazoprevir':     ('drug', '格拉瑞韦', 'Q19786991', ['格拉瑞韦']),
    'glecaprevir':     ('drug', '格卡瑞韦', None, ['格卡瑞韦']),
    'pibrentasvir':    ('drug', '哌仑他韦', None, ['哌仑他韦']),
    'voxilaprevir':    ('drug', '伏西瑞韦', None, ['伏西瑞韦']),
    'ribavirin':       ('drug', '利巴韦林', 'Q421862', ['利巴韦林', '病毒唑', '雷巴威林']),
    'ritonavir':       ('drug', '利托那韦', 'Q422618', ['利托那韦']),
    'antiviral':       ('drug', '抗病毒药物', 'Q846227', ['抗病毒药']),
    'cholestyramine_a': ('drug', '消胆胺', None, ['消胆胺']),
    'cholestyramine_b': ('drug', '考来烯胺', None, ['考来烯胺']),
    'lactulose':       ('drug', '乳果糖', 'Q422689', ['乳果糖']),
    'rifaximin':       ('drug', '利福昔明', 'Q416073', ['利福昔明']),
    'terlipressin':    ('drug', '特利加压素', 'Q324147', ['特利加压素', '三甘氨酰赖氨酸加压素']),
    'propranolol':     ('drug', '普萘洛尔', 'Q423364', ['普萘洛尔']),
    'beta_blocker':    ('drug', 'β-受体阻滞剂', None, ['β-受体阻滞剂']),
    'furosemide':      ('drug', '呋塞米', 'Q388801', ['呋塞米', '速尿']),
    'torasemide':      ('drug', '托拉塞米', 'Q419948', ['托拉塞米']),
    'bumetanide':      ('drug', '布美他尼', 'Q275926', ['布美他尼']),
    'ethacrynic_acid': ('drug', '依他尼酸', 'Q418571', ['依他尼酸']),
    'tolvaptan':       ('drug', '托伐普坦', 'Q426132', ['托伐普坦']),
    'diuretic':        ('drug', '利尿剂', None, ['利尿剂']),
    'anticoagulant':   ('drug', '抗凝药物', None, ['抗凝药物', '抗凝血剂']),
    'heparin':         ('drug', '肝素', None, ['肝素']),
    'warfarin':        ('drug', '华法林', None, ['华法林']),
    'thrombolytic':    ('drug', '溶栓药物', None, ['溶栓药物']),
    'udca':            ('drug', '熊去氧胆酸', 'Q241374', ['熊去氧胆酸']),
    'obeticholic_acid': ('drug', '奥贝胆酸', 'Q15708271', ['奥贝胆酸']),
    'elafibranor':     ('drug', 'elafibranor', 'Q15409440', ['ELAFIBRANOR']),
    'seladelpar':      ('drug', 'seladelpar', None, ['SELADELPAR']),
    'bezafibrate':     ('drug', '苯扎贝特', None, ['苯扎贝特']),
    'fenofibrate':     ('drug', '非诺贝特', None, ['非诺贝特']),
    'rifampicin':      ('drug', '利福平', None, ['利福平']),
    'naltrexone':      ('drug', '纳曲酮', None, ['纳曲酮']),
    'sertraline':      ('drug', '舍曲林', None, ['舍曲林']),
    'penicillamine':   ('drug', '青霉胺', 'Q421239', ['青霉胺']),
    'trientine':       ('drug', '曲恩汀', 'Q418386', ['三乙烯四胺', '曲恩汀']),
    'deferoxamine':    ('drug', '去铁胺', 'Q419618', ['去铁胺']),
    'deferasirox':     ('drug', '地拉罗司', 'Q5251502', ['地拉罗司']),
    'deferiprone':     ('drug', '去铁酮', 'Q749664', ['去铁酮']),
    'corticosteroid':  ('drug', '皮质类固醇', None, ['皮质类固醇', '糖皮质激素', '类固醇']),
    'prednisolone':    ('drug', '泼尼松龙', 'Q11426176', ['泼尼松龙']),
    'azathioprine':    ('drug', '硫唑嘌呤', 'Q18939', ['硫唑嘌呤']),
    'tacrolimus':      ('drug', '他克莫司', 'Q411648', ['他克莫司']),
    'pentoxifylline':  ('drug', '己酮可可碱', None, ['己酮可可碱']),
    'acetylcysteine':  ('drug', '乙酰半胱氨酸', None, ['乙酰半胱氨酸']),
    'benzodiazepine':  ('drug', '苯二氮卓类', None, ['苯二氮卓类']),
    'ademetionine':    ('drug', 'S-腺苷-L-甲硫氨酸', None, ['S-腺苷-L-甲硫氨酸']),
    'metadoxine':      ('drug', '美他多辛', None, ['美他多辛']),
    'vitamin_b':       ('drug', 'B族维生素', None, ['B族维生素']),
    'sorafenib':       ('drug', '索拉非尼', 'Q421136', ['索拉非尼', '蕾莎瓦']),
    'lenvatinib':      ('drug', '仑伐替尼', 'Q6523413', ['乐伐替尼', '仑伐替尼']),
    'regorafenib':     ('drug', '瑞戈非尼', 'Q3891664', ['瑞戈非尼']),
    'atezolizumab':    ('drug', '阿替利珠单抗', 'Q20707748', ['阿替利珠单抗', '泰圣奇']),
    'bevacizumab':     ('drug', '贝伐珠单抗', 'Q413299', ['贝伐珠单抗']),
    'durvalumab':      ('drug', '度伐利尤单抗', None, ['度伐利尤单抗']),
    'tremelimumab':    ('drug', 'tremelimumab', None, ['TREMELIMUMAB']),
    'cabozantinib':    ('drug', '卡博替尼', 'Q795057', ['卡博替尼']),
    'ramucirumab':     ('drug', '雷莫芦单抗', 'Q7290345', ['雷莫芦单抗', '雷莫西尤单抗']),
    'everolimus':      ('drug', '依维莫司', 'Q421052', ['依维莫司']),
    'cisplatin':       ('drug', '顺铂', 'Q412415', ['顺铂']),
    'doxorubicin':     ('drug', '多柔比星', 'Q18936', ['阿霉素', '多柔比星']),
    'albendazole':     ('drug', '阿苯达唑', 'Q411629', ['阿苯达唑']),
    'praziquantel':    ('drug', '吡喹酮', 'Q424145', ['吡喹酮']),
    'resmetirom':      ('drug', 'resmetirom', None, ['RESMETIROM']),
    'semaglutide':     ('drug', '司美格鲁肽', None, ['司美格鲁肽']),
    'vitamin_e':       ('drug', '维生素E', None, ['维生素E']),
    'thiazolidinedione': ('drug', '噻唑烷二酮类', None, ['罗格列酮', '匹格列酮']),

    # ---------------- 治疗手段 ----------------
    'liver_transplant': ('treatment', '肝移植', 'Q1368191', ['肝移植', '肝脏移植']),
    'abstinence':       ('treatment', '戒酒', None, ['戒酒', '停止饮酒', '不饮酒']),
    'tips':             ('treatment', '经颈静脉肝内门体分流术', None, ['TIPS', '经颈静脉肝内门体分流术', '门体分流术']),
    'band_ligation':    ('treatment', '套扎', None, ['套扎']),
    'low_salt':         ('treatment', '低盐饮食', None, ['低盐饮食', '限制钠']),
    'angioplasty':      ('treatment', '血管成形术', None, ['血管成形术']),
    'phlebotomy':       ('treatment', '放血', None, ['放血', '静脉切开术']),
    'rfa':              ('treatment', '射频消融', None, ['射频消融']),
    'tace':             ('treatment', '化疗栓塞', None, ['化疗栓塞']),
    'radioembolization': ('treatment', '内放疗', None, ['内放疗']),
    'chemotherapy':     ('treatment', '化疗', None, ['化疗']),
    'resection':        ('treatment', '手术切除', None, ['手术切除']),
    'drainage':         ('treatment', '引流', None, ['引流']),
    'embolization':     ('treatment', '动脉栓塞', None, ['动脉栓塞']),
    'weight_reduction': ('treatment', '减轻体重', None, ['减轻体重', '减重', '减肥']),
    'diet_control':     ('treatment', '饮食控制', None, ['饮食控制', '饮食调控']),
    'exercise':         ('treatment', '运动锻炼', None, ['有氧运动', '运动锻炼']),
    'mediterranean_diet': ('treatment', '地中海饮食', None, ['地中海饮食']),

    # ---------------- 检查 ----------------
    'ultrasound':       ('exam', '超声检查', 'Q234904', ['超声', 'B超', '彩超', '超音波']),
    'ct':               ('exam', 'CT', 'Q32566', ['CT', '计算机断层扫描', '电脑断层']),
    'mri':              ('exam', '磁共振', 'Q161238', ['MRI', '磁共振', '核磁共振']),
    'liver_biopsy':     ('exam', '肝活检', 'Q360841', ['肝活检', '肝脏活检', '肝脏穿刺', '肝脏切片', '肝脏活体组织']),
    'pcr':              ('exam', 'PCR', 'Q176996', ['PCR', '聚合酶链式反应', '聚合酶链反应']),
    'elisa':            ('exam', 'ELISA', 'Q332126', ['ELISA', '酶联免疫吸附']),
    'blood_test':       ('exam', '血液检查', 'Q886837', ['验血', '血液检查', '血液检验']),
    'liver_function_test': ('exam', '肝功能检查', 'Q1359801', ['肝功能测试', '肝功能检查', '肝功能化验']),
    'cbc':              ('exam', '血常规', 'Q886518', ['血常规', '全血细胞计数', '血象']),
    'chest_xray':       ('exam', '胸片', 'Q1283318', ['胸片', '胸部X光']),

    # ---------------- 病因 / 危险因素 ----------------
    'alcohol':          ('cause', '饮酒', 'Q2647488', ['饮酒', '喝酒', '酗酒']),
    'aflatoxin':        ('cause', '黄曲霉毒素', 'Q212936', ['黄曲霉毒素', '黄曲霉素', '黄曲毒素']),
    'acetaminophen':    ('cause', '对乙酰氨基酚', None, ['对乙酰氨基酚', '扑热息痛']),
    'isoniazid':        ('cause', '异烟肼', None, ['异烟肼']),
    'nsaid':            ('cause', '非类固醇抗发炎药', None, ['非类固醇抗发炎药']),
    'heshouwu':         ('cause', '何首乌', None, ['何首乌']),
    'ketoconazole':     ('cause', '酮康唑', None, ['酮康唑']),
    'obesity':          ('cause', '肥胖', None, ['肥胖']),
    'diabetes':         ('cause', '糖尿病', None, ['糖尿病']),
    'dyslipidemia':     ('cause', '血脂异常', None, ['高脂血症', '血脂异常', '血脂障碍']),
    'insulin_resistance': ('cause', '胰岛素抵抗', None, ['胰岛素抵抗']),
    'psc':              ('cause', '原发性硬化性胆管炎', 'Q1058608', ['原发性硬化性胆管炎']),
    'portal_vein_thrombosis': ('cause', '门静脉血栓', 'Q1704044', ['门静脉血栓']),
    'restrictive_cardiomyopathy': ('cause', '限制型心肌病', 'Q2151267', ['限制型心肌病']),
    'schistosomiasis':  ('cause', '血吸虫病', 'Q221159', ['血吸虫病']),
    'pericarditis':     ('cause', '心包炎', 'Q501561', ['心包炎']),
    'polycythemia_vera': ('cause', '真性红细胞增多症', None, ['真性红细胞增多症']),
    'pregnancy':        ('cause', '妊娠', None, ['妊娠']),
    'oral_contraceptive': ('cause', '口服避孕药', None, ['口服避孕药']),
    'pnh':              ('cause', '阵发性睡眠性血红蛋白尿', None, ['阵发性睡眠性血红蛋白尿']),
    'raw_fish':         ('cause', '生鱼', None, ['生鱼', '鱼生']),
    'appendicitis':     ('cause', '阑尾炎', None, ['阑尾炎']),
    'diverticulitis':   ('cause', '憩室炎', None, ['憩室炎']),
    'transfusion':      ('cause', '输血', None, ['输血']),
    'polycystic_kidney': ('cause', '多囊性肾病', None, ['多囊性肾病']),
    'echinococcus':     ('cause', '棘球绦虫', 'Q630200', ['细粒棘球蚴', '棘球绦虫']),
}

ENTITIES = [{'id': k, 'type': t, 'name': n, 'wikidata': q, 'terms': terms}
            for k, (t, n, q, terms) in ENTITY_TABLE.items()]

# %%
# =============================================================================
# 疾病 -> 实体 关系（每条带来源键）
# =============================================================================


MSD_CHRONIC = 'msd:liver-and-gallbladder-disorders/hepatitis/overview-of-chronic-hepatitis'
MSD_CIRRHOSIS = 'msd:liver-and-gallbladder-disorders/fibrosis-and-cirrhosis-of-the-liver/cirrhosis-of-the-liver'
MSD_FIBROSIS = 'msd:liver-and-gallbladder-disorders/fibrosis-and-cirrhosis-of-the-liver/fibrosis-of-the-liver'
MSD_HE = 'msd:liver-and-gallbladder-disorders/manifestations-of-liver-disease/hepatic-encephalopathy'
MSD_HCC = 'msd:cancer/liver-cancers/hepatocellular-carcinoma'

REL = {
    'hep_a': {
        'jaundice': 'wd msd', 'vomiting': 'wd msd', 'nausea': 'wd', 'anorexia': 'wd msd',
        'malaise': 'wd msd', 'fever': 'wd', 'abdominal_pain': 'msd', 'dark_urine': 'msd',
        'pruritus': 'msd', 'pale_stool': 'msd',
        'cholestyramine_a': 'msd',
        'pcr': 'wd', 'elisa': 'wd',
    },
    'hep_b': {
        'jaundice': 'wd msd', 'vomiting': 'wd msd', 'nausea': 'wd msd', 'anorexia': 'wd msd',
        'arthralgia': 'wd msd', 'hepatomegaly': 'wd', 'splenomegaly': 'wd msd', 'fever': 'msd',
        'malaise': 'msd', 'fatigue': 'msd', 'abdominal_pain': 'msd', 'urticaria': 'msd',
        'spider_nevi': 'msd', 'palmar_erythema': 'msd', 'ascites': 'msd',
        'bleeding_tendency': 'msd', 'varices': 'msd', 'coma': 'msd',
        'entecavir': 'wd_faers msd', 'tenofovir': 'wd msd', 'taf': 'wd_faers',
        'peginterferon': 'wd_faers msd', 'adefovir': 'wd msd', 'lamivudine': 'wd_faers msd',
        'telbivudine': 'msd', 'bicyclol': 'wd', 'hbig': 'msd', 'antiviral': 'msd',
        'liver_transplant': 'msd',
        'pcr': 'wd', 'elisa': 'wd', 'liver_biopsy': 'wd', 'blood_test': 'wd',
    },
    'hep_c': {
        'jaundice': 'wd msd', 'vomiting': 'wd msd', 'nausea': 'wd msd', 'anorexia': 'wd msd',
        'arthralgia': 'wd', 'fatigue': 'wd msd', 'abdominal_pain': 'wd msd', 'ascites': 'wd msd',
        'caput_medusae': 'wd', 'fever': 'msd', 'malaise': 'msd', 'splenomegaly': 'msd',
        'spider_nevi': 'msd', 'palmar_erythema': 'msd', 'bleeding_tendency': 'msd', 'varices': 'msd',
        'sofosbuvir': 'wd_faers msd', 'ledipasvir': 'wd msd', 'sof_led': 'wd_faers',
        'velpatasvir': 'wd_faers msd', 'daclatasvir': 'wd', 'simeprevir': 'wd',
        'elbasvir': 'msd', 'grazoprevir': 'wd msd', 'glecaprevir': 'wd_faers msd',
        'pibrentasvir': 'wd_faers msd', 'voxilaprevir': 'msd', 'ribavirin': 'wd msd',
        'ritonavir': 'wd_faers', 'peginterferon': 'wd_faers', 'bicyclol': 'wd', 'antiviral': 'wd msd',
        'liver_transplant': 'wd msd',
        'pcr': 'wd', 'elisa': 'wd', 'liver_biopsy': 'wd', 'blood_test': 'wd',
    },
    'hep_d': {
        'nausea': 'wd', 'coma': 'msd',
        'interferon': 'wd msd', 'liver_transplant': 'wd msd',
        'pcr': 'wd', 'elisa': 'wd', 'blood_test': 'wd',
    },
    'hep_e': {
        'jaundice': 'wd msd', 'vomiting': 'wd msd', 'nausea': 'wd msd', 'anorexia': 'wd msd',
        'fever': 'wd msd', 'hepatomegaly': 'wd', 'abdominal_pain': 'msd', 'pruritus': 'msd',
        'malaise': 'msd',
        'ribavirin': 'wd_faers msd', 'cholestyramine_b': 'msd', 'antiviral': 'wd',
        'pcr': 'wd', 'elisa': 'wd',
    },
    'alcoholic_liver': {
        'jaundice': 'wd msd', 'nausea': 'wd', 'fever': 'msd', 'hepatomegaly': 'msd',
        'fatigue': 'msd', 'palmar_erythema': 'msd', 'spider_nevi': 'msd', 'dupuytren': 'msd',
        'gynecomastia': 'msd',
        'alcohol': 'wd msd',
        'corticosteroid': 'msd', 'benzodiazepine': 'msd', 'ademetionine': 'msd',
        'metadoxine': 'msd', 'vitamin_b': 'msd',
        'abstinence': 'msd', 'liver_transplant': 'wd msd',
        'liver_function_test': 'wd', 'cbc': 'wd',
    },
    'alcoholic_hepatitis': {
        'jaundice': 'zhwiki', 'ascites': 'zhwiki', 'fatigue': 'zhwiki', 'hepatomegaly': 'zhwiki',
        'tremor': 'zhwiki', 'coma': 'zhwiki',
        'alcohol': 'zhwiki',
        'corticosteroid': 'zhwiki', 'prednisolone': 'zhwiki', 'pentoxifylline': 'zhwiki',
        'acetylcysteine': 'zhwiki',
        'abstinence': 'zhwiki', 'liver_transplant': 'zhwiki',
        'liver_biopsy': 'zhwiki', 'liver_function_test': 'zhwiki',
    },
    'toxic_liver': {
        'fatigue': 'msd', 'malaise': 'msd', 'nausea': 'msd', 'pruritus': 'msd', 'anorexia': 'msd',
        'jaundice': 'msd', 'hepatomegaly': 'msd', 'abdominal_pain': 'msd', 'confusion': 'msd',
        'acetaminophen': 'msd zhwiki', 'isoniazid': 'zhwiki', 'nsaid': 'zhwiki',
        'heshouwu': 'zhwiki', 'ketoconazole': 'zhwiki',
        'acetylcysteine': 'msd zhwiki', 'corticosteroid': 'msd',
        'liver_transplant': 'msd zhwiki',
        'liver_function_test': 'zhwiki',
    },
    'liver_failure': {
        'jaundice': 'msd', 'ascites': 'msd', 'confusion': 'msd', 'drowsiness': 'msd',
        'fatigue': 'msd', 'malaise': 'msd', 'nausea': 'msd', 'anorexia': 'msd',
        'fetor_hepaticus': 'msd', 'bleeding_tendency': 'msd zhwiki', 'hematemesis': 'msd',
        'bloody_stool': 'msd',
        'liver_transplant': 'msd zhwiki',
    },
    'hepatic_encephalopathy': {
        'jaundice': 'wd', 'coma': 'wd msd', 'tremor': 'wd msd', 'fetor_hepaticus': 'wd msd',
        'confusion': 'msd', 'drowsiness': 'msd',
        'lactulose': 'wd msd', 'rifaximin': 'msd',
    },
    'chronic_hepatitis': {
        'malaise': 'msd', 'anorexia': 'msd', 'fatigue': 'msd', 'fever': 'msd',
        'abdominal_pain': 'msd', 'splenomegaly': 'msd', 'spider_nevi': 'msd',
        'palmar_erythema': 'msd', 'ascites': 'msd', 'bleeding_tendency': 'msd', 'varices': 'msd',
        'jaundice': 'msd', 'pruritus': 'msd', 'pale_stool': 'msd',
        'antiviral': 'msd',
    },
    'liver_fibrosis': {
        'antiviral': 'msd', 'resmetirom': 'msd', 'semaglutide': 'msd', 'abstinence': 'msd',
    },
    'cirrhosis': {
        'fatigue': 'msd', 'malaise': 'msd', 'anorexia': 'msd', 'weight_loss': 'msd',
        'clubbing': 'msd', 'jaundice': 'msd', 'dark_urine': 'msd', 'pruritus': 'msd',
        'palmar_erythema': 'msd', 'dupuytren': 'msd', 'spider_nevi': 'msd',
        'gynecomastia': 'msd', 'ascites': 'msd', 'varices': 'msd',
        'alcohol': 'wd msd', 'psc': 'wd',
        'furosemide': 'wd', 'torasemide': 'wd', 'bumetanide': 'wd', 'ethacrynic_acid': 'wd',
        'tolvaptan': 'wd_faers', 'obeticholic_acid': 'wd_faers', 'sof_led': 'wd_faers',
        'sofosbuvir': 'wd_faers', 'entecavir': 'wd_faers', 'adefovir': 'wd_faers',
        'rifaximin': 'wd_faers', 'peginterferon': 'wd_faers', 'ribavirin': 'wd_faers',
        'daclatasvir': 'wd_faers', 'simeprevir': 'wd_faers',
        'beta_blocker': 'msd', 'antiviral': 'msd',
        'liver_transplant': 'wd msd', 'tips': 'msd', 'band_ligation': 'msd', 'low_salt': 'msd',
        'abstinence': 'msd',
        'ultrasound': 'wd', 'liver_biopsy': 'wd', 'blood_test': 'wd', 'ct': 'wd',
    },
    'pbc': {
        'pruritus': 'msd', 'fatigue': 'msd', 'dry_mouth_eyes': 'msd', 'xanthoma': 'msd',
        'jaundice': 'msd', 'ascites': 'msd', 'edema': 'msd', 'steatorrhea': 'msd',
        'udca': 'wd msd', 'obeticholic_acid': 'wd_faers', 'elafibranor': 'wd msd',
        'seladelpar': 'msd', 'bezafibrate': 'msd', 'fenofibrate': 'msd', 'penicillamine': 'wd',
        'cholestyramine_b': 'msd', 'rifampicin': 'msd', 'naltrexone': 'msd', 'sertraline': 'msd',
        'liver_transplant': 'msd', 'abstinence': 'msd',
    },
    'liver_abscess': {
        'appendicitis': 'zhwiki', 'diverticulitis': 'zhwiki',
    },
    'autoimmune_hepatitis': {
        'amenorrhea': 'msd', 'arthralgia': 'msd', 'anorexia': 'msd', 'nausea': 'msd',
        'corticosteroid': MSD_CIRRHOSIS,
        'prednisolone': 'wd_faers', 'azathioprine': 'wd_faers', 'tacrolimus': 'wd_faers',
    },
    'fatty_liver': {
        'anorexia': 'zhwiki', 'fatigue': 'zhwiki msd', 'nausea': 'zhwiki', 'vomiting': 'zhwiki',
        'abdominal_distension': 'zhwiki', 'abdominal_pain': 'zhwiki', 'hepatomegaly': 'zhwiki',
        'alcohol': 'zhwiki', 'obesity': 'zhwiki', 'diabetes': 'zhwiki', 'dyslipidemia': 'zhwiki',
        'weight_reduction': 'zhwiki msd', 'diet_control': 'zhwiki', 'exercise': 'zhwiki',
        'mediterranean_diet': 'msd', 'abstinence': 'msd',
        'vitamin_e': 'msd', 'thiazolidinedione': 'msd', 'resmetirom': 'msd',
        'ultrasound': 'zhwiki', 'ct': 'zhwiki', 'liver_function_test': 'zhwiki',
        'liver_biopsy': 'zhwiki',
    },
    'nafld': {
        'obesity': 'zhwiki', 'diabetes': 'zhwiki', 'insulin_resistance': 'zhwiki',
        'dyslipidemia': 'zhwiki',
        'weight_reduction': 'msd', 'diet_control': 'zhwiki', 'exercise': 'zhwiki',
        'resmetirom': 'msd ' + MSD_FIBROSIS, 'semaglutide': MSD_FIBROSIS,
        'vitamin_e': 'msd', 'thiazolidinedione': 'msd',
    },
    'portal_hypertension': {
        'splenomegaly': 'wd msd', 'jaundice': 'wd', 'telangiectasia': 'wd', 'ascites': 'wd msd',
        'abdominal_distension': 'msd', 'varices': 'msd', 'hematemesis': 'msd',
        'bloody_stool': 'msd', 'confusion': 'msd', 'drowsiness': 'msd', 'bleeding_tendency': 'msd',
        'portal_vein_thrombosis': 'wd', 'restrictive_cardiomyopathy': 'wd',
        'schistosomiasis': 'wd', 'pericarditis': 'wd',
        'propranolol': 'wd',
        'tips': 'msd', 'liver_transplant': 'msd',
    },
    'hepatorenal': {
        'terlipressin': 'wd',
    },
    'budd_chiari': {
        'abdominal_pain': 'zhwiki msd', 'hepatomegaly': 'zhwiki msd', 'ascites': 'zhwiki msd',
        'edema': 'zhwiki msd', 'jaundice': 'zhwiki msd', 'fatigue': 'msd', 'varices': 'msd',
        'hematemesis': 'msd', 'vomiting': 'msd',
        'polycythemia_vera': 'zhwiki', 'pregnancy': 'zhwiki', 'oral_contraceptive': 'zhwiki',
        'pnh': 'zhwiki',
        'anticoagulant': 'zhwiki msd', 'heparin': 'zhwiki', 'warfarin': 'zhwiki msd',
        'thrombolytic': 'msd', 'diuretic': 'zhwiki msd',
        'tips': 'zhwiki msd', 'angioplasty': 'zhwiki msd', 'liver_transplant': 'msd',
        'low_salt': 'zhwiki msd',
        'ultrasound': 'zhwiki msd', 'ct': 'zhwiki', 'mri': 'zhwiki', 'liver_biopsy': 'zhwiki',
    },
    'liver_cancer': {
        'jaundice': 'wd', 'hepatomegaly': 'wd', 'abdominal_pain': 'wd', 'weight_loss': 'wd',
        'aflatoxin': 'wd',
        'atezolizumab': 'wd', 'bevacizumab': 'wd', 'sorafenib': 'wd', 'lenvatinib': 'wd',
        'ramucirumab': 'wd', 'cabozantinib': 'wd',
        'mri': 'wd', 'ct': 'wd', 'ultrasound': 'wd', 'liver_biopsy': 'wd', 'blood_test': 'wd',
    },
    'hcc': {
        'jaundice': 'wd', 'hepatomegaly': 'wd', 'abdominal_pain': 'wd msd', 'weight_loss': 'wd msd',
        'fever': 'msd',
        'sorafenib': 'wd msd', 'lenvatinib': 'msd', 'regorafenib': 'msd', 'atezolizumab': 'msd',
        'bevacizumab': 'msd', 'durvalumab': 'msd', 'tremelimumab': 'msd',
        'resection': 'msd', 'liver_transplant': 'msd', 'rfa': 'msd', 'tace': 'msd',
        'radioembolization': 'msd', 'chemotherapy': 'msd',
    },
    'cholangiocarcinoma': {
        'weight_loss': 'zhwiki:中華肝吸蟲', 'jaundice': 'zhwiki:中華肝吸蟲', 'ascites': 'zhwiki:中華肝吸蟲',
        'everolimus': 'wd', 'regorafenib': 'wd',
    },
    'hepatoblastoma': {
        'cisplatin': 'wd_faers', 'doxorubicin': 'wd_faers',
        'resection': 'msd', 'chemotherapy': 'msd',
    },
    'liver_hemangioma': {
        'abdominal_pain': 'msd', 'abdominal_distension': 'msd', 'anorexia': 'msd', 'nausea': 'msd',
        'corticosteroid': 'msd', 'embolization': 'msd', 'liver_transplant': 'msd',
        'ultrasound': 'msd', 'ct': 'msd', 'mri': 'msd',
    },
    'liver_cyst': {
        'drainage': 'msd', 'liver_transplant': 'msd',
    },
    'polycystic_liver': {
        'polycystic_kidney': 'msd',
    },
    'hydatid_liver': {
        'vomiting': 'wd', 'hepatomegaly': 'wd', 'abdominal_pain': 'wd msd', 'nausea': 'wd',
        'heartburn': 'wd', 'hemoptysis': 'wd msd', 'weight_loss': 'wd', 'cough': 'wd',
        'jaundice': 'msd', 'urticaria': 'msd',
        'echinococcus': 'wd msd',
        'albendazole': 'wd msd',
        'resection': 'msd', 'drainage': 'msd', 'liver_transplant': 'msd',
        'ct': 'wd', 'mri': 'wd', 'ultrasound': 'wd', 'elisa': 'wd', 'chest_xray': 'wd',
    },
    'clonorchiasis': {
        'jaundice': 'wd zhwiki msd', 'abdominal_pain': 'wd zhwiki msd', 'nausea': 'wd msd',
        'fever': 'wd zhwiki msd', 'diarrhea': 'wd zhwiki msd', 'anorexia': 'zhwiki',
        'arthralgia': 'zhwiki', 'edema': 'zhwiki', 'hepatomegaly': 'zhwiki msd',
        'chills': 'msd', 'vomiting': 'msd', 'urticaria': 'msd', 'pruritus': 'msd',
        'weight_loss': 'msd',
        'raw_fish': 'zhwiki msd',
        'praziquantel': 'wd zhwiki msd', 'albendazole': 'wd zhwiki msd',
    },
    'wilson': {
        'jaundice': 'wd', 'edema': 'wd', 'kf_ring': 'wd',
        'penicillamine': 'wd_faers', 'trientine': 'wd',
    },
    'hemochromatosis': {
        'fatigue': 'msd medlineplus:hemochromatosis', 'malaise': 'msd medlineplus:hemochromatosis',
        'bronze_skin': 'msd', 'arthralgia': 'msd medlineplus:hemochromatosis',
        'weight_loss': 'medlineplus:hemochromatosis', 'abdominal_pain': 'medlineplus:hemochromatosis',
        'transfusion': 'zhwiki',
        'deferoxamine': 'wd', 'deferasirox': 'wd_faers', 'deferiprone': 'wd_faers',
        'phlebotomy': 'msd',
    },
}


def source_urls(cid, tokens):
    '''把来源键展开成网址，写进 entities.json，方便逐条核对'''
    out = []
    for t in tokens:
        if t in ('wd', 'wd_faers'):
            out += [WD_URL + q for q in CONCEPT_WD.get(cid, [])]
        elif t == 'msd':
            out += [MSD_URL + p for p in MSD_PAGE[cid]]
        elif t == 'zhwiki':
            out.append(W + ZHWIKI_PAGE[cid])
        elif t.startswith('msd:'):
            out.append(MSD_URL + t[4:])
        elif t.startswith('zhwiki:'):
            out.append(W + t[7:])
        elif t.startswith('medlineplus:'):
            out.append(MEDLINEPLUS_URL + t[12:])
        else:
            raise ValueError(f'unknown source token {t!r} for {cid}')
    return sorted(set(out))

# %%
# =============================================================================
# 方面层
# =============================================================================

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
        'strong': ['传染', '传播', '传给', '传人', '母婴阻断', '分餐', '遗传', '被感染',
                   '感染给', '隔离'],
        'weak':   [],
    },
    '指标解读': {
        'strong': ['指标', '化验单', '报告单', '检查结果', '化验结果', '偏高', '偏低', '升高',
                   '超标', '正常值', '参考值', '阳性', '阴性', '定量', '数值', '是什么意思',
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
                   '活多久', '寿命', '生命危险', '癌变', '会发展成', '会变成', '转成',
                   '恶化', '预后', '会不会死'],
        'weak':   [],
    },
    '注意事项': {
        'strong': ['注意什么', '注意事项', '要注意', '注意些什么', '注意哪些', '能不能吃',
                   '能吃', '可以吃', '不能吃', '吃什么', '饮食', '忌口', '食物', '能喝',
                   '喝酒', '护理', '预防', '保养', '疫苗', '接种', '怀孕', '怀上', '要孩子',
                   '生孩子', '生育', '喂奶', '母乳', '哺乳', '献血', '多久查', '多久复查',
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
                   '康复', '怎么调理', '调理', '偏方', '手术', '抗病毒', '停药', '用药',
                   '缓解'],
        'weak':   ['怎么办', '怎么处理', '咋办'],
    },
    '疾病表述': {
        'strong': ['症状', '表现', '什么感觉', '早期', '潜伏期', '分期', '分型', '区别',
                   '有哪些症状'],
        'weak':   ['是什么', '什么是', '什么病', '什么叫', '什么样'],
    },
}


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
# 构建词典：同义词 -> 概念
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
    '''相同，或互为上下位（肝癌 / 肝细胞癌）'''
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
    '''命中次数，且同时命中父、子概念时只保留子概念（肝癌，肝细胞癌 -> 肝细胞癌）'''
    cnt = count_hits(text)
    redundant = {a for c in cnt for a in ancestors(c)}
    return {c: n for c, n in cnt.items() if c not in redundant}


def tag_concepts(question):
    return set(specific_counts(question))

# %%
# =============================================================================
# 构建实体词典 + 疾病资料（profile）
# =============================================================================

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

# 与概念 c 相同或互为上下位的概念（问乙肝时，病毒性肝炎的资料也算；问病毒性肝炎时，各型的资料都算）
related_concepts = {c: {x for x in concept_by_id if related(x, c)} for c in concept_by_id}

# %%
# =============================================================================
# 方面判断
# =============================================================================

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

# %%
# =============================================================================
# 给答案贴标签：正文点名的疾病（mentioned）+ 继承的疾病（inherited）+ 实体（entities）
# =============================================================================
# specific：正文点名了至少一种病。
# unnamed ：正文没点名任何病（“清淡饮食、多休息”之类），什么病的问题都可能用得上，
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
# 实体权重 = 在训练答案里的 IDF：越是多答案提到（黄疸、超声），越是不能说明两段话像。

n_docs = max(len(doc_rec), 1)
entity_df = Counter(e for x in doc_rec.values() for e in x['entities'])
idf = {e: math.log((n_docs + 1) / (entity_df.get(e, 0) + 1)) + 1.0 for e in entity_by_id}

# 答案方面：继承它配对的那些训练问题的方面（多数派）
doc_aspect = {}
for r, qid in zip(train, row_qid):
    asp = q_info[qid].get('aspect')
    if asp:
        did = r['doc_id']
        doc_aspect.setdefault(did, Counter())[asp] += 1
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
    # 方面系数
    q_asp = q_info_row.get('aspect')
    d_asp = doc_aspect.get(doc_id)
    if q_asp and d_asp:
        if q_asp == d_asp:
            cos = min(1.0, cos * ASPECT_BOOST)
        else:
            cos = cos / ASPECT_BOOST
    return cos


def entity_score(q, d):
    '''
    旧接口保留（供 onto_sim / 抽检用）：
      support / conflict / missing 同旧版；score 改用 onto_cos_score。
    '''
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
               'note': 'same dictionary as step 06, plus family and Wikidata ids; '
                       'relations in entities.json were collected online on 2026-10-31',
               'sources': {'icd10': 'https://icd.who.int/browse10/2019/en',
                           'wikipedia': 'https://zh.wikipedia.org/wiki/',
                           'wikidata': 'https://www.wikidata.org/',
                           'msd_manual_zh': 'https://www.msdmanuals.cn/home',
                           'cblue': 'https://github.com/CBLUEbenchmark/CBLUE'},
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

TERM_RE = re.compile(r'[\u4e00-\u9fa5]{0,3}肝[\u4e00-\u9fa5]{0,3}')
term_cnt = Counter()
for q in no_signal:
    for t in TERM_RE.findall(q['anchor']):
        term_cnt[t] += 1
with open(f'{ONTO_DIR}/unlabelled_liver_terms.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.writer(f)
    w.writerow(['term_around_肝', 'count_in_unlabelled_questions'])
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

# 4. 新增：本体相似度挑出的难负例。每个问题列出 onto_sim 最高、但不是它配对答案的 HARD_TOPK 条
by_concept, by_entity = defaultdict(set), defaultdict(set)
for did, x in doc_rec.items():
    for c in set(x['mentioned']) | set(x['inherited']):
        by_concept[c].add(did)
    for e in x['entities']:
        by_entity[e].add(did)

hard_rows, hard_rel = [], Counter()

# %%
# =============================================================================
# 新增：难负例抽检（本体相似度，不依赖任何模型）
# =============================================================================

N_HARD_Q = 150                       # 抽多少个问题做难负例抽检
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
        w.writerow([i, q['anchor'], doc_text[ref][:300], doc_text[it['doc_id']][:300], ''])
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

print('=' * 70)
print('Step 06 (new) · Ontology labelling + relation layer')
print('=' * 70)
print('')
print('  Concepts / synonym terms      : {} / {}   ambiguous terms: {} {}'.format(
    len(CONCEPTS), len(term2concepts), len(ambiguous), ambiguous if ambiguous else ''))
print('')
print('  Unique training questions     : {:,}'.format(nq))
print('  with a concept                : {:,} ({:.1%})'.format(n_c, n_c / max(nq, 1)))
print('  generic only                  : {:,} ({:.1%})'.format(n_gen, n_gen / max(nq, 1)))
print('  no signal (vocabulary gap?)   : {:,} ({:.1%})   -> see unlabelled_liver_terms.csv'.format(len(no_signal), len(no_signal) / max(nq, 1)))
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
    'step': '06_ontology_new', 'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
    't2s_backend': T2S_BACKEND, 'n_concepts': len(CONCEPTS), 'n_terms': len(term2concepts),
    'ambiguous_terms': ambiguous,
    'questions': nq, 'with_concept': n_c, 'generic_only': n_gen, 'no_signal': len(no_signal),
    'multi_concept': n_multi,
    'answers': n_doc, 'answers_specific': n_spec, 'answers_unnamed': n_doc - n_spec,
    'answers_with_inherited': n_inh, 'answers_inherited_but_unnamed': n_unnamed_inh,
    'inherited_vs_mentioned': {'answers_with_both': len(both), 'agree': agree},
    'random_pair_relation': {k: round(prior[k] / n_prior, 4) for k in ('same', 'different', 'unnamed')},
    'question_concept_counts': {cname[c]: k for c, k in q_cnt.most_common()},
    'answer_concept_counts': {cname[c]: k for c, k in d_cnt.most_common()},
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
    '# Step 06 (new) · 肝病本体标注', '',
    'Generated {}. t2s backend: `{}`. Output dir: `{}/`.'.format(
        report['generated_at'], T2S_BACKEND, ONTO_DIR), '',
    '## 概念词典', '',
    _md_table(['Item', 'Value'], [
        ['Concepts', len(CONCEPTS)],
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
    'check_*.csv, unlabelled_liver_terms.csv, onto_hard_check*.csv, onto_hard_candidates.jsonl'.format(ONTO_DIR),
]

with open(REPORT_DIR + '/06_ontology_' + TAG + '.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(_md) + '\n')

print('')
print('Outputs: {}/  (concepts.json, entities.json, question_labels.jsonl, doc_labels.jsonl,'.format(ONTO_DIR))
print('         check_*.csv, unlabelled_liver_terms.csv, onto_hard_check*.csv, onto_hard_candidates.jsonl)')
print('Reports: {}/06_ontology_{}.json / .md'.format(REPORT_DIR, TAG))
print('Next   : fill onto_hard_check.csv first; if not_same is clearly safer than all, add onto_sim to 08. Dont forget run 07')
