import json
import torch
import vllm
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import warnings
import re
import os
warnings.filterwarnings('ignore')

# =============================================================================
# 配置区域
# =============================================================================

MODEL_PATH = "/data/DeepSeek-R1-Distill-Qwen-32B"

INPUT_FILE = '/home/liver/data/Chinese Medical Dialogue Dataset/liver_CMDD.jsonl'
OUTPUT_FILE = '/home/liver/data/Chinese Medical Dialogue Dataset/liver_CMDD_clean.jsonl'
FILTERED_FILE = '/home/liver/data/Chinese Medical Dialogue Dataset/liver_CMDD_filtered.jsonl'

BATCH_SIZE = 512
MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 512
GPU_MEMORY_UTILIZATION = 0.92
TENSOR_PARALLEL_SIZE = 4

ENABLE_DOUBLE_CHECK = True

DEBUG_MODE = False
DEBUG_SAMPLES = 5

# =============================================================================
# 关键词预过滤规则
#
# 核心逻辑：
#   1. 在【问题】中检查非肝病关键词
#   2. 在【问题+答案】中检查肝脏保护关键词
#   3. 只有"问题命中非肝病词 且 问题+答案都没有肝脏保护词"时才预过滤
#
# 这样设计的原因：
#   - 问题决定了核心主题（"丙肝肝硬化有什么症状" → 核心是肝病）
#   - 答案中经常附带提及其他系统疾病（"肝硬化可能伴有贫血"），不应触发预过滤
#   - 肝病问题中答案提到"贫血""凝血障碍"等是正常的并发症讨论
# =============================================================================

# 肝脏相关保护关键词：问题或答案中出现任意一个即跳过预过滤，交给模型判断
LIVER_PROTECT_KEYWORDS = [
    # ---- 直接含"肝"的词 ----
    '肝', '护肝', '保肝', '养肝',

    # ---- 病毒性肝炎的各种别名 ----
    '乙肝', '丙肝', '甲肝', '戊肝', '丁肝',
    'HBV', 'HCV', 'HAV', 'HEV', 'HDV',
    '大三阳', '小三阳', '两对半', '乙肝五项',
    'A型肝炎', 'B型肝炎', 'C型肝炎', 'D型肝炎', 'E型肝炎',
    '甲型肝炎', '乙型肝炎', '丙型肝炎', '丁型肝炎', '戊型肝炎',
    '慢性肝炎', '急性肝炎', '病毒性肝炎', '重型肝炎', '暴发性肝炎',
    'HBsAg', 'HBeAg', 'HBcAb', 'HBsAb', 'HBeAb',
    'HBV DNA', 'HCV RNA',

    # ---- 肝脏疾病 ----
    '肝硬化', '肝癌', '肝炎', '脂肪肝', '肝纤维化',
    '肝衰竭', '肝移植', '肝脓肿', '肝包虫', '肝吸虫',
    '肝损伤', '肝损害', '肝功能', '肝大', '肝肿大',
    '肝囊肿', '肝血管瘤', '肝腺瘤', '肝母细胞瘤',
    '肝豆状核', 'Wilson病', '血色病', '铁过载',
    '药物性肝', 'DILI', '中毒性肝',
    'NAFLD', 'NASH', 'MAFLD', 'MASLD',
    '酒精性肝', '酒精肝',

    # ---- 肝功能指标 ----
    '转氨酶', 'ALT', 'AST', 'GGT', 'ALP',
    '谷丙转氨酶', '谷草转氨酶', '碱性磷酸酶', '谷氨酰转肽酶',
    '胆红素', '黄疸', '白蛋白', '球蛋白', '白球比',
    '凝血酶原时间', 'PT', 'INR', '胆碱酯酶',
    'AFP', '甲胎蛋白', 'AFP-L3', 'PIVKA',

    # ---- 肝病体征/并发症 ----
    '门脉高压', '门静脉', '腹水', '肝性脑病', '肝昏迷',
    '蜘蛛痣', '肝掌', '肝病面容', '肝臭',
    '食管胃底静脉曲张', '脾功能亢进',
    '肝肾综合征', 'HRS', '肝肺综合征', 'HPS',

    # ---- 肝病治疗 ----
    '恩替卡韦', '替诺福韦', 'TAF', 'TDF', 'ETV',
    '拉米夫定', '阿德福韦', '替比夫定',
    '聚乙二醇干扰素', 'PEG-IFN',
    '索磷布韦', '达卡他韦', '维帕他韦', '丙通沙', '夏帆宁',
    'DAAs', 'SVR',
    '索拉非尼', '仑伐替尼', '多纳非尼', 'TACE', 'HAIC',
    '甘草酸', '异甘草酸镁', '双环醇', '水飞蓟',
    '多烯磷脂酰胆碱', '易善复', '熊去氧胆酸', 'UDCA',
    '腺苷蛋氨酸', '思美泰',
    'TIPS', '人工肝', '血浆置换',

    # ---- 评分系统 ----
    'Child-Pugh', 'MELD', 'FibroScan', 'APRI', 'FIB-4',
    'BCLC', 'CNLC', 'Metavir', '米兰标准',

    # ---- 自免肝标志物 ----
    'AMA', 'AMA-M2', 'LKM', 'SLA', 'PBC', 'PSC', 'AIH',
    '抗线粒体抗体', '抗肝肾微粒体',

    # ---- 其他 ----
    '肝活检', '肝穿刺', '肝弹性',
    '乳果糖', '利福昔明', '门冬氨酸鸟氨酸',
    'ICG', '铜蓝蛋白', 'K-F环',
]

# 明确的非肝病疾病关键词（只在问题中检查）
NON_LIVER_KEYWORDS = [
    # ---- 肿瘤/癌症（非肝脏）----
    '肺癌', '胃癌', '结肠癌', '直肠癌', '乳腺癌', '宫颈癌', '卵巢癌',
    '前列腺癌', '膀胱癌', '肾癌', '食管癌', '鼻咽癌', '甲状腺癌',
    '子宫内膜癌', '淋巴瘤', '白血病', '骨肉瘤', '脑瘤', '脑癌',
    '喉癌', '舌癌', '皮肤癌', '黑色素瘤', '睾丸癌', '阴茎癌',
    '霍奇金', '非霍奇金', '多发性骨髓瘤', '胸腺瘤', '纵隔肿瘤',

    # ---- 妇科 ----
    '宫颈糜烂', '宫颈炎', '阴道炎', '盆腔炎', '子宫肌瘤', '卵巢囊肿',
    '子宫内膜异位', '多囊卵巢', '月经不调', '痛经', '闭经', '功血',
    '外阴瘙痒', '宫外孕', '葡萄胎', '子宫脱垂', '阴道出血',
    '乳腺增生', '乳腺炎', '乳房胀痛', '乳头溢液',

    # ---- 骨科/关节/口腔颌面 ----
    '骨折', '腰椎间盘', '颈椎病', '肩周炎', '关节炎', '类风湿',
    '股骨头坏死', '脊柱侧弯', '腱鞘炎', '滑膜炎', '痛风',
    '骨质疏松', '半月板', '韧带损伤', '骨刺', '强直性脊柱炎',
    '颞下颌', '颌面', '颧骨', '颧面', '下颌关节', '咬合',
    '正颌', '颌骨', '面部塌陷',

    # ---- 皮肤科 ----
    '湿疹', '荨麻疹', '银屑病', '牛皮癣', '白癜风', '痤疮', '青春痘',
    '脂溢性皮炎', '带状疱疹', '疥疮', '脚气', '灰指甲', '毛囊炎',
    '扁平疣', '鸡眼', '神经性皮炎', '玫瑰糠疹',
    '乳房瘙痒', '乳头瘙痒', '乳房外侧',
    '皮疹', '皮肤瘙痒', '皮肤过敏',

    # ---- 呼吸科 ----
    '哮喘', '肺炎', '肺结核', '慢阻肺', 'COPD', '支气管炎',
    '肺气肿', '气胸', '肺栓塞', '间质性肺病', '肺纤维化',
    '鼻炎', '鼻窦炎', '咽炎', '扁桃体炎', '喉炎',
    '打鼾', '睡眠呼吸暂停',

    # ---- 心血管 ----
    '冠心病', '心肌梗死', '心绞痛', '心律失常', '房颤', '心衰',
    '高血压', '动脉硬化', '主动脉夹层', '心肌炎', '心包炎',
    '瓣膜病', '先天性心脏病', '心肌病', '静脉曲张',

    # ---- 肾脏/泌尿/男科 ----
    '肾结石', '肾炎', '肾病综合征', '尿毒症', '肾透析',
    '肾小球肾炎', 'IgA肾病', '膀胱炎', '尿道炎', '前列腺炎',
    '前列腺增生', '肾积水', '多囊肾', '肾囊肿',
    '早泄', '阳痿', '勃起功能障碍', '精索静脉曲张', '包皮',
    '睾丸炎', '附睾炎', '精瘀', '睾丸积液', '鞘膜积液',
    '尿路感染', '尿频', '尿急', '尿痛', '血尿',
    '输尿管结石', '肾盂肾炎',

    # ---- 神经科/精神科 ----
    '癫痫', '帕金森', '阿尔茨海默', '脑梗', '脑出血', '脑中风',
    '偏头痛', '三叉神经痛', '面瘫', '脑膜炎', '多发性硬化',
    '抑郁症', '焦虑症', '精神分裂', '躁郁症', '失眠症',
    '强迫症', '恐惧症', '自闭症', '头痛', '头晕', '眩晕',

    # ---- 消化科（非肝脏）----
    '胃溃疡', '十二指肠溃疡', '胃炎', '肠炎', '结肠炎',
    '克罗恩', '溃疡性结肠炎', '阑尾炎', '肠梗阻', '肠套叠',
    '胃息肉', '肠息肉', '胃下垂', '胃食管反流', '贲门失弛缓',
    '胃痛', '胃疼', '腹泻', '便秘', '痔疮', '肛裂', '肛瘘',
    '直肠脱垂', '肛周脓肿',

    # ---- 胰腺 ----
    '胰腺炎', '胰腺癌', '胰岛素瘤', '胰腺囊肿',

    # ---- 单纯胆道（不涉及肝脏）----
    '胆囊结石', '胆囊炎', '胆囊息肉', '胆石症', '胆绞痛',

    # ---- 内分泌 ----
    '甲亢', '甲减', '甲状腺结节', '甲状腺炎', '糖尿病足',
    '库欣综合征', '嗜铬细胞瘤', '垂体瘤', '尿崩症',
    '甲状腺肿', '桥本氏',

    # ---- 眼科/耳鼻喉 ----
    '白内障', '青光眼', '近视', '结膜炎', '角膜炎',
    '中耳炎', '耳鸣', '突发性耳聋', '鼻出血',
    '飞蚊症', '散光', '远视', '干眼症',

    # ---- 儿科特有 ----
    '手足口病', '川崎病', '百日咳', '佝偻病',

    # ---- 血液科（非肝脏相关）----
    '地中海贫血', '再生障碍性贫血', '血友病',
    '血小板减少性紫癜', '单核细胞增多症', '传染性单核细胞',
    '缺铁性贫血', '溶血性贫血',

    # ---- 风湿免疫（非肝脏相关）----
    '系统性红斑狼疮', '干燥综合征', '硬皮病', '皮肌炎',
    '血管炎', '结缔组织病', '未分化结缔组织',
    '风湿热', '抗磷脂综合征',

    # ---- 纵隔 ----
    '纵隔',

    # ---- 传染病（非肝炎）----
    '艾滋病', 'HIV', '梅毒', '淋病', '结核', '疟疾',
    '登革热', '流感', '新冠', 'COVID',
    '水痘', '麻疹', '风疹', '腮腺炎',
    '狂犬病', '破伤风', '炭疽',

    # ---- 其他 ----
    '骨头响', '骨头咯噔', '关节响', '弹响',
    '减肥', '增肥', '美白', '祛斑', '脱发', '白发',
    '口臭', '口腔溃疡', '牙痛', '龋齿', '智齿',
    '痱子', '冻疮', '烫伤', '烧伤',
    '中暑', '晕车', '晕船',
    '打嗝', '嗳气',
]


# =============================================================================
# 提示词模板（strength + few-shot）
# =============================================================================

SYSTEM_PROMPT = """你是一位专业的肝脏病学领域专家。你的任务是严格判断医疗问答是否属于肝病范畴。

关键判断原则：
1. "肝病"指肝脏本身的疾病，必须以肝脏为核心讨论对象
2. 问答的核心主题必须是肝脏疾病，而非其他疾病附带提到肝脏
3. 其他疾病的问答中偶尔提及"肝功能""转氨酶"等词，不代表属于肝病范畴
4. 必须严格判断，宁可错过也不要误收

请只回答"是"或"否"。"""

USER_PROMPT_TEMPLATE = """请严格判断以下医疗问答的核心主题是否属于肝病范畴。

判断标准：
- 问答的核心主题必须是肝脏本身的疾病（如肝炎、肝硬化、肝癌、脂肪肝、肝衰竭等）
- 仅仅在其他疾病的讨论中提及肝脏相关指标（如转氨酶、肝功能），不算肝病
- 全身性疾病（如淋巴瘤、结缔组织病、传染性单核细胞增多症）即使累及肝脏，只要核心主题不是肝病，就判"否"
- 其他器官/系统的疾病（如胃病、肾病、皮肤病、妇科病、骨科病等）判"否"

以下是几个判断示例：

示例1:
问题：乙肝大三阳能治好吗？
回答：乙肝大三阳是指...抗病毒治疗...
判断：是
原因：核心主题是乙肝（肝脏疾病）

示例2:
问题：纵隔霍奇金淋巴瘤的症状是什么？
回答：霍奇金淋巴瘤...纵隔占位...淋巴结肿大...
判断：否
原因：核心主题是淋巴瘤（血液系统肿瘤），不是肝脏疾病

示例3:
问题：传染性单核细胞增多症的注意事项
回答：发热...肝大、转氨酶高时可口服维生素C...脾大...
判断：否
原因：核心主题是传染性单核细胞增多症（感染性疾病），仅附带提及肝脏

示例4:
问题：未分化结缔组织病的诊断是什么？
回答：UCTD患者...部分患者出现转氨酶升高...自身免疫性肝损害...
判断：否
原因：核心主题是结缔组织病（风湿免疫病），肝损害仅是其并发症之一

示例5:
问题：肝硬化腹水怎么治疗？
回答：肝硬化...利尿剂...白蛋白输注...
判断：是
原因：核心主题是肝硬化及其并发症（肝脏疾病）

示例6:
问题：宫颈糜烂什么样算重度
回答：宫颈糜烂...柱状上皮外翻...
判断：否
原因：核心主题是宫颈糜烂（妇科疾病）

示例7:
问题：药物性肝损伤怎么办？
回答：药物性肝损伤...停用可疑药物...保肝治疗...
判断：是
原因：核心主题是药物性肝损伤（肝脏疾病）

示例8:
问题：大半夜胃痛怎么缓解
回答：胃溃疡...胃病...
判断：否
原因：核心主题是胃病（消化系统但非肝脏）

示例9:
问题：慢性肝炎怎么治疗？
回答：慢性肝炎...抗病毒...保肝降酶...
判断：是
原因：核心主题是慢性肝炎（肝脏疾病）

示例10:
问题：服用某药须注意的事项
回答：...严重肝功能损伤...贫血或凝血功能障碍...慎用
判断：是
原因：核心讨论涉及肝功能损伤相关的用药注意事项

现在请判断以下问答：

问题：{question}
回答：{answer}

这个问答的核心主题是否是肝脏疾病？请只回答"是"或"否"。
判断："""

# =============================================================================
# 二次验证提示词（更严格）
# =============================================================================

DOUBLE_CHECK_SYSTEM_PROMPT = """你是一位严格的肝病学审核专家。请再次确认以下问答是否真正属于肝病范畴。
你必须非常严格：只有核心主题确实是肝脏本身的疾病时才判"是"。
请只回答"是"或"否"。"""

DOUBLE_CHECK_USER_TEMPLATE = """请再次严格确认：以下问答的核心主题是否是肝脏本身的疾病？

严格标准：
- 核心讨论的必须是肝脏疾病（肝炎、肝硬化、肝癌、脂肪肝、肝衰竭、药物性肝损伤等）
- 以下情况判"否"：
  * 其他系统疾病（淋巴瘤、结缔组织病、单核细胞增多症等）即使提及肝功能异常
  * 全身性感染/免疫疾病中肝脏仅是受累器官之一
  * 其他器官的疾病（胃病、肾病、皮肤病、妇科、骨科等）
  * 单纯的胆囊/胰腺疾病

问题：{question}
回答：{answer}

判断："""

# 需要触发二次验证的边界关键词（出现在问题或答案中）
BORDERLINE_KEYWORDS = [
    '淋巴', '结缔组织', '单核细胞', '红斑狼疮', '血管炎',
    '类风湿', '干燥综合', '硬皮病', '皮肌炎',
    '贫血', '血小板', '白血病', '骨髓',
    '心肌', '心包', '胸腔积液',
    '肾功能', '肾损', '肾衰',
    '糖尿病', '甲状腺', '甲亢', '甲减',
    '妊娠', '怀孕',
    '感染', '脓毒症', '败血症',
    '肿瘤', '化疗', '放疗',
]


# =============================================================================
# 工具函数
# =============================================================================

def get_question_text(item):
    """
    从数据项中提取问题文本（处理各种嵌套格式）
    支持格式:
      - {"questions": ["问题"]}
      - {"questions": [["问题1", "问题2"]]}
      - {"questions": [["问题"]]}
    """
    questions = item.get('questions', [])
    if not questions:
        return ''
    question = questions[0]
    if isinstance(question, list):
        return question[0] if question else ''
    return question


def get_answer_text(item):
    """从数据项中提取答案文本"""
    answers = item.get('answers', [])
    if not answers:
        return ''
    answer = answers[0]
    if isinstance(answer, list):
        return answer[0] if answer else ''
    return answer


def keyword_prefilter(question_text, answer_text):
    """
    关键词预过滤：快速排除明显非肝病数据

    核心逻辑：
    1. 在【问题+答案】中检查肝脏保护关键词 → 有则跳过预过滤
    2. 在【问题】中检查非肝病关键词 → 命中且无保护词则直接过滤

    返回值:
        'reject'  - 明确非肝病，直接过滤
        'unknown' - 无法确定，交给模型判断
    """
    combined = question_text + ' ' + answer_text

    # 第一步：检查肝脏保护关键词（在问题+答案中检查）
    # 只要出现任一肝脏关键词，就不进行预过滤，交给模型判断
    has_liver_keyword = any(kw in combined for kw in LIVER_PROTECT_KEYWORDS)
    if has_liver_keyword:
        return 'unknown'

    # 第二步：只在【问题】中检查非肝病关键词
    has_non_liver_keyword = any(kw in question_text for kw in NON_LIVER_KEYWORDS)
    if has_non_liver_keyword:
        return 'reject'

    # 无法确定，交给模型
    return 'unknown'


def needs_double_check(question_text, answer_text):
    """判断是否需要二次验证"""
    combined = question_text + ' ' + answer_text
    return any(kw in combined for kw in BORDERLINE_KEYWORDS)


def extract_answer_from_thinking(response):
    """从 DeepSeek-R1 输出中提取最终"是"/"否"，去除思考链"""
    if not response:
        return ""

    # 情况1: </think> 正常闭合
    cleaned = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    cleaned = cleaned.strip()
    if cleaned:
        return cleaned

    # 情况2: </think> 未闭合（被截断）
    if '<think>' in response and '</think>' not in response:
        return "[TRUNCATED]"

    # 情况3: </think> 闭合但后面没有内容
    think_match = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        last_part = think_content[-80:] if len(think_content) > 80 else think_content
        if re.search(r'不属于|不是肝病|应该判否|判断为否|回答.*否|答案.*否|判"否"|判断：否', last_part):
            return "否"
        elif re.search(r'属于肝病|是肝病|应该判是|判断为是|回答.*是|答案.*是|判"是"|判断：是', last_part):
            return "是"
        elif "否" in last_part[-20:]:
            return "否"
        elif "是" in last_part[-20:] and "不是" not in last_part[-20:]:
            return "是"

    return response.strip()


def judge_is_liver(response_text):
    """根据提取后的文本判断是否属于肝病"""
    if response_text == "[TRUNCATED]":
        return False
    if not response_text:
        return False

    response_head = response_text[:5]

    if "否" in response_head:
        return False
    elif "是" in response_head and "否" not in response_head:
        return True
    elif response_text.startswith("否"):
        return False
    elif response_text.startswith("是"):
        return True
    else:
        if "不属于" in response_text or "不是" in response_text:
            return False
        elif "属于" in response_text:
            return True
        else:
            return False


def build_messages(question, answer, system_prompt=None, user_template=None):
    """构建聊天消息格式"""
    if system_prompt is None:
        system_prompt = SYSTEM_PROMPT
    if user_template is None:
        user_template = USER_PROMPT_TEMPLATE
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(
            question=question, answer=answer
        )}
    ]


# =============================================================================
# 主分类器
# =============================================================================

class LiverDiseaseClassifier:
    """肝病分类器 - vLLM版本 (v2.1)"""

    def __init__(self, model_path):
        self.debug_count = 0
        self.truncated_count = 0
        self.prefiltered_count = 0
        self.double_check_filtered_count = 0

        print("=" * 70)
        print("🚀 初始化肝病分类器 (vLLM + DeepSeek-R1) v2.1")
        print("=" * 70)
        print(f"模型路径: {model_path}")
        print(f"张量并行: {TENSOR_PARALLEL_SIZE} GPU")
        print(f"显存利用率: {GPU_MEMORY_UTILIZATION * 100:.0f}%")
        print(f"最大上下文: {MAX_MODEL_LEN}")
        print(f"最大生成token: {MAX_NEW_TOKENS}")
        print(f"批处理大小: {BATCH_SIZE}")
        print(f"二次验证: {'开启' if ENABLE_DOUBLE_CHECK else '关闭'}")
        print(f"非肝病关键词数: {len(NON_LIVER_KEYWORDS)}")
        print(f"肝脏保护关键词数: {len(LIVER_PROTECT_KEYWORDS)}")

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"\n检测到 {gpu_count} 个GPU设备:")
            for i in range(min(gpu_count, TENSOR_PARALLEL_SIZE)):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"  GPU {i}: {gpu_name}, 显存: {gpu_memory:.2f}GB")

        print("\n🤖 初始化vLLM引擎（首次加载需要几分钟）...")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            max_model_len=MAX_MODEL_LEN,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = self.llm.get_tokenizer()

        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=MAX_NEW_TOKENS,
            stop=["<|endoftext|>", "<｜end▁of▁sentence｜>", "\n\n"],
        )

        print("✅ vLLM引擎初始化完成")
        print("=" * 70)
        print()

    def _build_prompt(self, question, answer, system_prompt=None, user_template=None):
        """将消息转换为模型输入prompt"""
        messages = build_messages(question, answer, system_prompt, user_template)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            sp = system_prompt or SYSTEM_PROMPT
            ut = user_template or USER_PROMPT_TEMPLATE
            prompt = (
                f"<|begin_of_sentence|>"
                f"<|User|>{sp}\n\n"
                f"{ut.format(question=question, answer=answer)}"
                f"<|Assistant|>"
            )
        return prompt

    def prepare_prompts(self, items):
        """将数据项转换为模型输入prompt"""
        prompts = []
        for item in items:
            question = get_question_text(item)
            answer = get_answer_text(item)
            prompt = self._build_prompt(question, answer)
            prompts.append(prompt)
        return prompts

    def prepare_double_check_prompts(self, items):
        """构建二次验证的prompt"""
        prompts = []
        for item in items:
            question = get_question_text(item)
            answer = get_answer_text(item)
            prompt = self._build_prompt(
                question, answer,
                system_prompt=DOUBLE_CHECK_SYSTEM_PROMPT,
                user_template=DOUBLE_CHECK_USER_TEMPLATE
            )
            prompts.append(prompt)
        return prompts

    def classify_batch(self, items):
        """使用vLLM批量分类（含预过滤和二次验证）"""
        if not items:
            return []

        # =====================================================================
        # 第一步：关键词预过滤
        # =====================================================================
        need_model_items = []
        need_model_indices = []
        results = [None] * len(items)

        for idx, item in enumerate(items):
            question = get_question_text(item)
            answer = get_answer_text(item)
            prefilter_result = keyword_prefilter(question, answer)

            if prefilter_result == 'reject':
                results[idx] = False
                self.prefiltered_count += 1
                if DEBUG_MODE and self.debug_count < DEBUG_SAMPLES:
                    print(f"\n🔑 预过滤: [{question[:60]}...] → ❌ 直接过滤（关键词匹配）")
            else:
                need_model_items.append(item)
                need_model_indices.append(idx)

        # =====================================================================
        # 第二步：模型推理（首轮）
        # =====================================================================
        if need_model_items:
            prompts = self.prepare_prompts(need_model_items)
            outputs = self.llm.generate(prompts, self.sampling_params)

            first_round_positive_items = []
            first_round_positive_indices = []

            for i, output in enumerate(outputs):
                raw_response = output.outputs[0].text.strip()
                response = extract_answer_from_thinking(raw_response)
                orig_idx = need_model_indices[i]

                if response == "[TRUNCATED]":
                    self.truncated_count += 1

                if DEBUG_MODE and self.debug_count < DEBUG_SAMPLES:
                    item = need_model_items[i]
                    question = get_question_text(item)
                    print("\n" + "=" * 70)
                    print(f"🔍 调试样本 #{self.debug_count + 1}")
                    print("-" * 70)
                    print(f"问题: {question[:100]}...")
                    if raw_response != response:
                        print(f"原始输出(含思考链): [{raw_response[:300]}...]")
                    print(f"提取结果: [{response}]")
                    if response == "[TRUNCATED]":
                        print("⚠️  思考链被截断，无法获取最终答案")
                    self.debug_count += 1

                is_liver = judge_is_liver(response)

                if is_liver:
                    first_round_positive_items.append(need_model_items[i])
                    first_round_positive_indices.append(orig_idx)
                else:
                    results[orig_idx] = False

                if DEBUG_MODE and self.debug_count <= DEBUG_SAMPLES:
                    print(f"首轮判断: {'✅ 保留（肝病）' if is_liver else '❌ 过滤（非肝病）'}")
                    print("=" * 70)

            # =================================================================
            # 第三步：二次验证（对首轮判"是"的边界数据）
            # =================================================================
            if ENABLE_DOUBLE_CHECK and first_round_positive_items:
                need_recheck_items = []
                need_recheck_orig_indices = []
                direct_accept_orig_indices = []

                for i, item in enumerate(first_round_positive_items):
                    question = get_question_text(item)
                    answer = get_answer_text(item)
                    orig_idx = first_round_positive_indices[i]

                    if needs_double_check(question, answer):
                        need_recheck_items.append(item)
                        need_recheck_orig_indices.append(orig_idx)
                    else:
                        direct_accept_orig_indices.append(orig_idx)

                # 直接接受的
                for orig_idx in direct_accept_orig_indices:
                    results[orig_idx] = True

                # 需要二次验证的
                if need_recheck_items:
                    recheck_prompts = self.prepare_double_check_prompts(need_recheck_items)
                    recheck_outputs = self.llm.generate(recheck_prompts, self.sampling_params)

                    for i, output in enumerate(recheck_outputs):
                        raw_response = output.outputs[0].text.strip()
                        response = extract_answer_from_thinking(raw_response)
                        orig_idx = need_recheck_orig_indices[i]

                        is_liver_recheck = judge_is_liver(response)
                        results[orig_idx] = is_liver_recheck

                        if not is_liver_recheck:
                            self.double_check_filtered_count += 1

                        if DEBUG_MODE:
                            item = need_recheck_items[i]
                            question = get_question_text(item)
                            print(f"\n🔄 二次验证: [{question[:60]}...] → "
                                  f"{'✅ 确认保留' if is_liver_recheck else '❌ 改判过滤'}")
            else:
                for orig_idx in first_round_positive_indices:
                    results[orig_idx] = True

        return results

    def process_file(self, input_file, output_file, filtered_file):
        """处理文件"""
        print(f"📂 开始处理文件: {input_file}\n")

        print("📖 加载数据到内存...")
        data_list = []
        errors = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="读取数据"):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line.strip())
                    questions = item.get('questions', [])
                    answers = item.get('answers', [])
                    if isinstance(questions, list) and isinstance(answers, list) and questions and answers:
                        data_list.append(item)
                    else:
                        errors += 1
                except json.JSONDecodeError:
                    errors += 1

        total_count = len(data_list)
        print(f"✅ 数据加载完成: {total_count} 条有效数据")
        if errors > 0:
            print(f"⚠️  跳过 {errors} 条错误数据")
        print()

        if not data_list:
            print("⚠️ 没有有效数据可处理")
            return None

        if DEBUG_MODE:
            print(f"🔍 调试模式已启用，将显示前 {DEBUG_SAMPLES} 个样本\n")

        kept_count = 0
        filtered_count = 0

        print("🔄 开始批量分类（关键词预过滤 + vLLM推理 + 二次验证）...")
        print()

        with open(output_file, 'w', encoding='utf-8') as f_out, \
             open(filtered_file, 'w', encoding='utf-8') as f_filtered:

            progress_bar = tqdm(
                range(0, total_count, BATCH_SIZE),
                desc="推理进度",
                unit="batch",
                ncols=100
            )

            for i in progress_bar:
                batch_items = data_list[i:i + BATCH_SIZE]
                results = self.classify_batch(batch_items)

                for item, is_liver in zip(batch_items, results):
                    if is_liver:
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                        kept_count += 1
                    else:
                        f_filtered.write(json.dumps(item, ensure_ascii=False) + '\n')
                        filtered_count += 1

                total_processed = kept_count + filtered_count
                keep_ratio = kept_count / total_processed * 100 if total_processed > 0 else 0
                progress_bar.set_postfix({
                    '保留': kept_count,
                    '过滤': filtered_count,
                    '比例': f'{keep_ratio:.1f}%'
                })

                if (i // BATCH_SIZE) % 10 == 0:
                    f_out.flush()
                    f_filtered.flush()

        print("\n" + "=" * 70)
        print("✅ 处理完成!")
        print("=" * 70)
        print(f"\n📊 统计信息:")
        print(f"  总数据量:        {total_count:>8} 条")
        print(f"  保留数据:        {kept_count:>8} 条 ({kept_count / total_count * 100:>6.2f}%)")
        print(f"  过滤数据:        {filtered_count:>8} 条 ({filtered_count / total_count * 100:>6.2f}%)")
        if errors > 0:
            print(f"  错误数据:        {errors:>8} 条")
        print(f"\n📊 过滤明细:")
        print(f"  关键词预过滤:    {self.prefiltered_count:>8} 条")
        model_filtered = filtered_count - self.prefiltered_count - self.double_check_filtered_count
        print(f"  模型首轮过滤:    {model_filtered:>8} 条")
        if ENABLE_DOUBLE_CHECK:
            print(f"  二次验证过滤:    {self.double_check_filtered_count:>8} 条")
        if self.truncated_count > 0:
            print(f"  截断保守过滤:    {self.truncated_count:>8} 条")
        print(f"\n📁 输出文件:")
        print(f"  保留: {output_file}")
        print(f"  过滤: {filtered_file}")
        print("=" * 70)
        print()

        return {
            'total': total_count,
            'kept': kept_count,
            'filtered': filtered_count,
            'errors': errors,
            'truncated': self.truncated_count,
            'prefiltered': self.prefiltered_count,
            'double_check_filtered': self.double_check_filtered_count,
            'kept_ratio': kept_count / total_count * 100 if total_count > 0 else 0
        }


# =============================================================================
# 主入口
# =============================================================================

def run_filter():
    """主函数"""
    print("\n" + "=" * 70)
    print("🎯 肝病数据过滤器")
    print("   (关键词预过滤 + vLLM + DeepSeek-R1 + 二次验证)")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        print("❌ 未检测到GPU，程序退出")
        return None

    if not Path(INPUT_FILE).exists():
        print(f"❌ 输入文件不存在: {INPUT_FILE}")
        return None

    if not Path(MODEL_PATH).exists():
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        return None

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(FILTERED_FILE).parent.mkdir(parents=True, exist_ok=True)

    classifier = LiverDiseaseClassifier(MODEL_PATH)

    start_time = datetime.now()
    print(f"⏰ 开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    stats = classifier.process_file(INPUT_FILE, OUTPUT_FILE, FILTERED_FILE)

    end_time = datetime.now()
    duration = end_time - start_time

    print(f"⏰ 结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️  总耗时: {duration}")

    if stats and stats['total'] > 0:
        speed = stats['total'] / duration.total_seconds()
        print(f"🚀 处理速度: {speed:.2f} 条/秒")

    print()
    return stats


if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    stats = run_filter()