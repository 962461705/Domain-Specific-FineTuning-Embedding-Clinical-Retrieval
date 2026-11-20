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
import math
import time
warnings.filterwarnings('ignore')

# =============================================================================
# 配置区域
# =============================================================================

MODEL_PATH = "/data/DeepSeek-R1-Distill-Qwen-32B"

INPUT_FILE = '/home/neuro/MedChatZH/neuro_MedChatZH.jsonl'
OUTPUT_FILE = '/home/neuro/MedChatZH/neuro_MedChatZH_clean.jsonl'
FILTERED_FILE = '/home/neuro/MedChatZH/neuro_MedChatZH_filtered.jsonl'

PROGRESS_FILE = OUTPUT_FILE + '.progress.json'   # 断点续跑进度（全部完成后自动删除）
RESUME = True                     # 中断后重跑时从上次完成的批次继续

# 批越大，思维链回退的样本越多，解码阶段越能填满 max_num_seqs，长尾越少；
# 代价是进度条更新更稀疏、中断时丢失的进度更多
BATCH_SIZE = 4096
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.92
TENSOR_PARALLEL_SIZE = int(os.environ.get('TP_SIZE', '4'))   # 4 张 L20 做张量并行
QUANTIZATION = os.environ.get('QUANT', 'fp8') or None        # fp8 在线量化（L20 的 FP8 算力是 BF16 的 2 倍）；QUANT= 设为空字符串则回到 bf16
LIMIT = int(os.environ.get('LIMIT', '0'))                    # 只处理均匀抽样的 N 条用于测速，结果写到 .test 文件；0 表示全量
MAX_NUM_BATCHED_TOKENS = 32768    # 直答以预填充为主，加大单步token预算

# ---- 方案一：跳过思维链直接回答 + "是/否" logprob 打分 ----
# 直答: 在 assistant 开头预填空思考段 "<think>\n\n</think>\n\n"，只生成1个token，
#       取 "是"/"否" 的概率 p_yes = P(是) / (P(是) + P(否))
# p_yes >= CONF_HIGH → 判是；p_yes <= CONF_LOW → 判否；介于两者之间 → 不确定
# 不确定的样本（及"是/否"概率总和过低的样本）回退到原来的思维链推理
CONF_HIGH = 0.8
CONF_LOW = 0.2
MIN_YES_NO_MASS = 0.5             # 首token中"是/否"类token概率之和低于此值视为不确定
LOGPROBS_TOPK = 20                # vLLM 默认 max_logprobs=20，不要超过
ENABLE_THINK_FALLBACK = False      # 不确定样本是否回退思维链；False 则按 p_yes>=0.5 直接判
THINK_MAX_TOKENS = 512            # 回退思维链的最大生成长度
FORCE_ANSWER_ON_TRUNCATE = True   # 思维链被截断时补上 "</think>" 再取1个token的是/否概率，而不是直接判否
MAX_QUESTION_CHARS = 300          # 问题超过此字符数时截断；0 表示不截断
DOUBLE_CHECK_SKIP_P = 0.97        # 首轮 p_yes 不低于此值的样本不再二次验证
MAX_ANSWER_CHARS = int(os.environ.get('MAX_ANS', '500'))           # 回答超过此字符数时截断（判断主题无需完整长回答，减少预填充）；0 表示不截断

ENABLE_DOUBLE_CHECK = True

DEBUG_MODE = False
DEBUG_SAMPLES = 5

# =============================================================================
# 关键词来源（均为联网获取并逐词核验命中的原文）
#   [ICD-10] 维基百科中文 ICD-10 章节条目：
#            第六章 神经系统疾病 G00-G99；第九章 I60-I69 脑血管病；
#            第二章 C70-C72/C75.1/C75.3；第一章 A17/A80-A89；第十七章 Q00-Q07；
#            第十八章 R25-R56；第十九章 S04/S06/S14/S24/S34/T90；
#            第五章 F00-F99 精神和行为障碍（仅用于非神经科学关键词）
#            https://zh.wikipedia.org/wiki/ICD-10_第六章：神经系统疾病
#   [维基]   维基百科中文 疾病/药物/手术条目
#            https://zh.wikipedia.org/wiki/神經外科 、 https://zh.wikipedia.org/wiki/神經內科
#   [MSD]    默沙东诊疗手册专业版（中文）神经系统疾病 / 中枢神经系统肿瘤 / 外伤性脑损伤 / 脊柱外伤
#            https://www.msdmanuals.cn/professional/neurologic-disorders
#            https://www.msdmanuals.cn/professional/oncology/central-nervous-system-tumors
#
# 范围：神经内科 + 神经外科；不纳入精神科（ICD-10 F00-F99）
# =============================================================================

# =============================================================================
# 关键词预过滤规则
#
# 核心逻辑：
#   1. 在【问题】中检查非神经科学关键词
#   2. 在【问题+答案】中检查神经科学保护关键词
#   3. 只有"问题命中非神经科学词 且 问题+答案都没有神经科学保护词"时才预过滤
#
# 这样设计的原因：
#   - 问题决定了核心主题（"脑梗后遗症怎么康复" → 核心是神经科学）
#   - 答案中经常附带提及其他系统疾病（"脑梗患者常合并高血压、糖尿病"），不应触发预过滤
#   - 神经科学问题中答案提到"肺炎""骨折"等是正常的并发症讨论
# =============================================================================

# 神经科学保护关键词：问题或答案中出现任意一个即跳过预过滤，交给模型判断
NEURO_PROTECT_KEYWORDS = [

    # ---- 基础科室：科室 ----
    "神经内科", "神经外科", "神经科", "神经病学", "脑外科", "脑血管病",

    # ---- 神经系统疾病：脑血管病 [ICD-10 I60-I69, G45-G46] ----
    "脑卒中", "中风", "缺血性脑卒中", "缺血性卒中", "脑梗死", "脑梗", "脑梗塞",
    "脑血栓", "脑栓塞", "腔隙性脑梗死", "腔梗", "短暂性脑缺血发作", "TIA", "脑出血",
    "颅内出血", "出血性卒中", "蛛网膜下腔出血", "脑动脉瘤", "颅内动脉瘤", "动脉瘤破裂", "脑动静脉畸形",
    "动静脉畸形", "烟雾病", "颈动脉狭窄", "脑静脉窦血栓", "静脉窦血栓", "椎基底动脉供血不足", "脑小血管病",
    "脑白质病变", "血管性痴呆", "脑血管病后遗症",

    # ---- 神经系统疾病：发作性疾病 [ICD-10 G40-G41] ----
    "癫痫", "癫痫发作", "癫痫持续状态", "羊癫疯", "羊角风", "全面性发作", "局灶性发作",
    "失神发作", "强直阵挛发作", "热性惊厥", "婴儿痉挛症", "Dravet综合征", "Lennox-Gastaut综合征",

    # ---- 神经系统疾病：头痛与颅面痛 [ICD-10 G43-G44, G50] ----
    "偏头痛", "紧张型头痛", "丛集性头痛", "药物滥用性头痛", "低颅压", "特发性颅内压增高", "三叉神经痛",
    "舌咽神经痛", "枕神经痛",

    # ---- 神经系统疾病：运动障碍与变性病 [ICD-10 G10-G26] ----
    "帕金森病", "帕金森", "帕金森综合征", "特发性震颤", "原发性震颤", "肌张力障碍", "痉挛性斜颈",
    "颈部肌张力障碍", "舞蹈病", "舞蹈症", "亨廷顿病", "亨廷顿舞蹈病", "多系统萎缩", "进行性核上性麻痹",
    "小脑萎缩", "脊髓小脑性共济失调", "肝豆状核变性", "肌阵挛", "面肌痉挛", "偏侧面肌痉挛", "不宁腿综合征",

    # ---- 神经系统疾病：痴呆与认知障碍 [ICD-10 G30-G31] ----
    "阿尔茨海默病", "老年痴呆", "路易体痴呆", "额颞叶痴呆", "轻度认知障碍", "正常压力脑积水", "克雅病",
    "朊病毒病",

    # ---- 神经系统疾病：中枢神经系统脱髓鞘疾病 [ICD-10 G35-G37] ----
    "多发性硬化", "视神经脊髓炎", "视神经脊髓炎谱系疾病", "NMOSD", "急性播散性脑脊髓炎", "横贯性脊髓炎",

    # ---- 神经系统疾病：中枢神经系统感染 [ICD-10 G00-G09, A17, A80-A89] ----
    "脑膜炎", "细菌性脑膜炎", "化脓性脑膜炎", "病毒性脑膜炎", "结核性脑膜炎", "隐球菌脑膜炎", "脑炎",
    "病毒性脑炎", "乙型脑炎", "自身免疫性脑炎", "脑脓肿", "硬膜下积脓", "脑囊虫病", "脊髓灰质炎",
    "小儿麻痹", "进行性多灶性白质脑病",

    # ---- 神经系统疾病：周围神经与神经肌肉疾病 [ICD-10 G50-G73] ----
    "周围神经病", "周围神经病变", "多发性神经病", "多发性周围神经病", "糖尿病周围神经病变", "格林-巴利综合征", "吉兰-巴雷综合征",
    "格林巴利综合征", "慢性炎性脱髓鞘性多发性神经病", "CIDP", "面神经麻痹", "面瘫", "贝尔麻痹", "周围性面瘫",
    "坐骨神经痛", "腕管综合征", "神经根病", "胸廓出口综合征", "带状疱疹后神经痛", "神经病理性疼痛", "复杂区域疼痛综合征",
    "重症肌无力", "肌无力综合征", "肌萎缩侧索硬化", "运动神经元病", "脊髓性肌萎缩症", "脊肌萎缩症", "肌营养不良",
    "进行性肌营养不良", "周期性瘫痪", "周期性麻痹", "腓骨肌萎缩症", "僵人综合征", "自主神经病",

    # ---- 神经系统疾病：脊髓疾病 [ICD-10 G95, MSD 脊髓疾病] ----
    "脊髓病", "脊髓压迫症", "脊髓空洞症", "脊髓损伤", "脊髓炎", "脊髓梗死", "脊髓血管畸形",
    "脊髓型颈椎病", "马尾综合征", "遗传性痉挛性截瘫", "亚急性联合变性", "小脑扁桃体下疝", "Chiari畸形", "脊髓栓系",

    # ---- 神经系统疾病：神经系统肿瘤 [ICD-10 C70-C72, C75.1, C75.3] ----
    "脑肿瘤", "脑瘤", "颅内肿瘤", "颅内占位", "脑癌", "胶质瘤", "神经胶质瘤",
    "胶质母细胞瘤", "星形细胞瘤", "少突胶质细胞瘤", "室管膜瘤", "髓母细胞瘤", "脑膜瘤", "垂体瘤",
    "垂体腺瘤", "听神经瘤", "前庭神经鞘瘤", "神经鞘瘤", "颅咽管瘤", "松果体区肿瘤", "生殖细胞瘤",
    "原发性脑淋巴瘤", "脑转移瘤", "脑转移", "脊髓肿瘤", "椎管内肿瘤", "血管母细胞瘤", "神经纤维瘤病",
    "神经纤维瘤", "结节性硬化症", "脑膜癌",

    # ---- 神经系统疾病：颅脑及脊柱脊髓创伤 [ICD-10 S04, S06, S14, S24, S34, T90] ----
    "颅脑损伤", "颅脑外伤", "脑外伤", "创伤性脑损伤", "外伤性脑损伤", "脑震荡", "脑挫伤",
    "脑挫裂伤", "弥漫性轴索损伤", "硬膜外血肿", "硬膜下血肿", "慢性硬膜下血肿", "颅内血肿", "颅骨骨折",
    "颅底骨折", "颅内损伤",

    # ---- 神经系统疾病：颅内压与脑疝 [ICD-10 G93] ----
    "颅内压增高", "颅内高压", "颅内压", "脑水肿", "脑疝", "小脑幕切迹疝", "枕骨大孔疝",

    # ---- 神经系统疾病：先天性与儿童神经疾病 [ICD-10 Q00-Q07, G80, G91] ----
    "脑积水", "先天性脑积水", "脊柱裂", "脑膨出", "无脑畸形", "小头畸形", "脑性瘫痪",
    "脑瘫", "孔洞脑",

    # ---- 神经系统疾病：意识障碍与脑病 [ICD-10 G92-G93, R40] ----
    "昏迷", "植物状态", "植物人", "最低意识状态", "闭锁综合征", "脑死亡", "中毒性脑病",
    "韦尼克脑病",

    # ---- 神经系统疾病：睡眠相关神经疾病 [ICD-10 G47] ----
    "发作性睡病", "猝倒",

    # ---- 神经症状：头痛眩晕 [ICD-10 R42, R51] ----
    "头痛", "头晕", "眩晕", "晕厥",

    # ---- 神经症状：发作性症状 [ICD-10 R56] ----
    "抽搐", "惊厥",

    # ---- 神经症状：意识与认知 [ICD-10 R40-R41] ----
    "意识障碍", "失语", "构音障碍",

    # ---- 神经症状：运动障碍 [ICD-10 G81-G83, R25-R27] ----
    "偏瘫", "截瘫", "四肢瘫", "瘫痪", "肢体无力", "震颤",

    # ---- 神经症状：感觉障碍 [ICD-10 R20] ----
    "麻木", "肢体麻木",

    # ---- 神经症状：脑神经及眼部神经体征 ----
    "复视", "上睑下垂", "视野缺损", "偏盲", "视乳头水肿", "Horner综合征",

    # ---- 神经症状：神经系统体征 ----
    "颈项强直", "脑膜刺激征", "病理反射", "巴宾斯基征",

    # ---- 诊断检查：神经影像 [MSD 神经系统检查和操作] ----
    "头颅CT", "头部CT", "颅脑CT", "头颅MRI", "颅脑磁共振", "脑CT", "脑MRI",
    "磁共振血管成像", "MRA", "脑血管造影", "弥散加权成像", "DWI", "脊髓造影", "经颅多普勒",
    "TCD",

    # ---- 诊断检查：神经电生理 ----
    "脑电图", "EEG", "视频脑电图", "动态脑电图", "肌电图", "EMG", "神经传导",
    "诱发电位",

    # ---- 诊断检查：脑脊液检查 ----
    "腰椎穿刺", "腰穿", "脑脊液检查", "脑脊液压力", "寡克隆带",

    # ---- 诊断检查：量表与评分 ----
    "格拉斯哥昏迷评分", "格拉斯哥昏迷指数", "GCS", "NIHSS", "MMSE", "MoCA",

    # ---- 诊断检查：实验室与特殊检查 ----
    "乙酰胆碱受体抗体", "AQP4", "神经活检", "肌肉活检",

    # ---- 治疗方法：脑血管病治疗 ----
    "静脉溶栓", "阿替普酶", "rt-PA", "替奈普酶", "机械取栓", "取栓", "颈动脉内膜剥脱术",
    "颈动脉支架", "依达拉奉", "丁苯酞", "尼莫地平",

    # ---- 治疗方法：降颅压与神经外科手术 ----
    "甘露醇", "开颅手术", "开颅", "去骨瓣减压", "血肿清除", "脑室引流", "脑室腹腔分流术",
    "弹簧圈", "伽玛刀", "立体定向放射外科", "微血管减压术", "椎板切除",

    # ---- 治疗方法：肿瘤放化疗 ----
    "替莫唑胺",

    # ---- 治疗方法：神经调控 ----
    "脑深部电刺激", "深部脑刺激", "DBS", "迷走神经刺激",

    # ---- 治疗方法：抗癫痫药 ----
    "抗癫痫药", "卡马西平", "奥卡西平", "丙戊酸", "丙戊酸钠", "左乙拉西坦", "拉莫三嗪",
    "托吡酯", "苯妥英", "苯巴比妥",

    # ---- 治疗方法：神经病理性疼痛 ----
    "加巴喷丁", "普瑞巴林",

    # ---- 治疗方法：帕金森病 ----
    "左旋多巴", "多巴丝肼", "卡比多巴", "普拉克索", "罗匹尼罗", "司来吉兰", "雷沙吉兰",
    "恩他卡朋", "金刚烷胺", "苯海索",

    # ---- 治疗方法：痴呆 ----
    "多奈哌齐", "美金刚", "卡巴拉汀", "加兰他敏",

    # ---- 治疗方法：神经免疫/神经肌肉 ----
    "溴吡斯的明", "吡斯的明", "依库珠单抗", "诺西那生钠", "利鲁唑",

    # ---- 治疗方法：头痛 ----
    "曲普坦", "舒马曲普坦", "佐米曲普坦", "氟桂利嗪",

]

# 明确的非神经科学疾病关键词（只在问题中检查）
# 说明: 精神科（ICD-10 F00-F99）按要求不纳入，列于此处
NON_NEURO_KEYWORDS = [

    # ---- 肿瘤/癌症（非神经系统） ----
    "肺癌", "胃癌", "结肠癌", "乳腺癌", "宫颈癌", "卵巢癌", "前列腺癌",
    "膀胱癌", "肾癌", "鼻咽癌", "甲状腺癌", "白血病", "骨肉瘤", "喉癌",
    "皮肤癌", "黑色素瘤", "睾丸癌", "阴茎癌", "霍奇金", "多发性骨髓瘤", "胸腺瘤",
    "肝癌", "胰腺癌",

    # ---- 精神科（用户要求不纳入） ----
    "精神分裂", "抑郁症", "焦虑症", "躁郁症", "双相情感障碍", "强迫症", "恐惧症",
    "自闭症", "孤独症", "多动症", "神经衰弱", "癔症", "疑病症", "人格障碍",
    "创伤后应激障碍", "神经性厌食", "心理咨询",

    # ---- 妇科 ----
    "宫颈糜烂", "宫颈炎", "阴道炎", "盆腔炎", "卵巢囊肿", "子宫内膜异位", "多囊卵巢",
    "痛经", "闭经", "葡萄胎", "乳腺炎",

    # ---- 骨科/关节（单纯骨关节病） ----
    "骨折", "关节炎", "类风湿", "脊柱侧弯", "腱鞘炎", "滑膜炎", "痛风",
    "骨质疏松", "韧带损伤",

    # ---- 皮肤科 ----
    "湿疹", "荨麻疹", "银屑病", "牛皮癣", "白癜风", "痤疮", "脂溢性皮炎",
    "疥疮", "毛囊炎", "鸡眼", "玫瑰糠疹",

    # ---- 呼吸科/耳鼻喉 ----
    "哮喘", "肺炎", "肺结核", "COPD", "支气管炎", "肺气肿", "气胸",
    "肺栓塞", "鼻炎", "鼻窦炎", "咽炎", "扁桃体炎", "喉炎", "中耳炎",
    "梅尼埃病", "鼻出血",

    # ---- 心血管（非脑血管） ----
    "冠心病", "心肌梗死", "心绞痛", "心律失常", "房颤", "心衰", "心肌炎",
    "心包炎", "先天性心脏病", "心肌病", "下肢静脉曲张",

    # ---- 肝胆胰 ----
    "肝炎", "肝硬化", "胆囊炎", "胰腺炎",

    # ---- 肾脏/泌尿/男科 ----
    "肾结石", "肾炎", "尿毒症", "膀胱炎", "尿道炎", "前列腺炎", "前列腺增生",
    "肾积水", "多囊肾", "早泄", "勃起功能障碍", "精索静脉曲张", "包皮", "睾丸炎",
    "附睾炎", "鞘膜积液", "尿路感染", "输尿管结石", "肾盂肾炎",

    # ---- 消化科 ----
    "胃溃疡", "十二指肠溃疡", "胃炎", "肠炎", "结肠炎", "克罗恩", "溃疡性结肠炎",
    "阑尾炎", "肠梗阻", "肠息肉", "胃食管反流", "腹泻", "便秘", "痔疮",

    # ---- 内分泌 ----
    "甲亢", "甲减", "甲状腺结节", "甲状腺炎", "糖尿病足", "库欣综合征", "嗜铬细胞瘤",
    "甲状腺肿",

    # ---- 眼科 ----
    "白内障", "青光眼", "近视", "结膜炎", "角膜炎", "散光", "远视",
    "干眼症",

    # ---- 儿科传染/感染 ----
    "手足口病", "百日咳", "佝偻病",

    # ---- 血液科 ----
    "地中海贫血", "再生障碍性贫血", "血友病", "缺铁性贫血", "溶血性贫血",

    # ---- 风湿免疫 ----
    "系统性红斑狼疮", "干燥综合征", "硬皮病", "皮肌炎", "风湿热",

    # ---- 传染病（非中枢神经系统感染） ----
    "艾滋病", "梅毒", "淋病", "疟疾", "登革热", "流感", "水痘",
    "麻疹", "风疹", "腮腺炎",

    # ---- 其他 ----
    "牙痛", "烫伤", "烧伤", "中暑", "打嗝", "脱发",
]


# =============================================================================
# 提示词模板（strength + few-shot）
# =============================================================================

SYSTEM_PROMPT = """你是一位专业的神经病学与神经外科学领域专家。你的任务是严格判断医疗问答是否属于神经科学（神经内科、神经外科）范畴。

关键判断原则：
1. "神经科学疾病"指脑、脊髓、脑神经、周围神经及神经肌肉接头/肌肉本身的疾病，以及颅脑、脊髓的外伤和肿瘤
2. 问答的核心主题必须是神经系统疾病，而非其他疾病附带提到神经系统症状
3. 精神科疾病（抑郁症、焦虑症、精神分裂症、强迫症、失眠等心理/精神问题）不属于本范畴
4. 其他疾病的问答中偶尔提及"头痛""头晕""麻木"等症状，不代表属于神经科学范畴
5. 必须严格判断，宁可错过也不要误收

请只回答"是"或"否"。"""

USER_PROMPT_TEMPLATE = """请严格判断以下医疗问答的核心主题是否属于神经科学（神经内科、神经外科）范畴。

判断标准：
- 核心主题必须是神经系统本身的疾病，例如：脑梗死、脑出血、蛛网膜下腔出血、癫痫、帕金森病、阿尔茨海默病、偏头痛、三叉神经痛、面神经麻痹、多发性硬化、重症肌无力、格林-巴利综合征、周围神经病、脑膜炎、脑炎、脑肿瘤（胶质瘤、脑膜瘤、垂体瘤、听神经瘤）、颅脑损伤、硬膜下/硬膜外血肿、脑积水、脊髓损伤、脊髓空洞症等
- 精神科疾病（抑郁症、焦虑症、精神分裂症、双相情感障碍、强迫症、神经衰弱、心理问题）判"否"
- 仅仅在其他疾病的讨论中提及头痛、头晕、麻木、乏力等症状，不算神经科学
- 感冒发热引起的头痛、颈腰部肌肉劳损、单纯骨折、耳源性眩晕（如梅尼埃病）等，核心不是神经系统疾病，判"否"
- 其他器官/系统的疾病（如心脏病、胃病、肝病、肾病、皮肤病、妇科病、单纯骨关节病等）判"否"

以下是几个判断示例：

示例1:
问题：脑梗死后半身不遂怎么康复？
回答：脑梗死...偏瘫...康复训练...抗血小板治疗...
判断：是
原因：核心主题是脑梗死（脑血管病）

示例2:
问题：最近心情低落，总是想哭是抑郁症吗？
回答：抑郁症...情绪低落...兴趣减退...心理治疗...抗抑郁药...
判断：否
原因：核心主题是抑郁症（精神科疾病），不属于神经内外科

示例3:
问题：感冒发烧头痛怎么办？
回答：上呼吸道感染...发热...多喝水...退烧药...
判断：否
原因：核心主题是感冒（呼吸道感染），头痛只是伴随症状

示例4:
问题：脑膜瘤需要做手术吗？
回答：脑膜瘤...良性肿瘤...开颅手术切除...伽玛刀...
判断：是
原因：核心主题是脑膜瘤（神经外科颅内肿瘤）

示例5:
问题：癫痫病人可以停药吗？
回答：癫痫...抗癫痫药...脑电图复查...逐渐减量...
判断：是
原因：核心主题是癫痫（神经内科疾病）

示例6:
问题：糖尿病人血糖控制不好怎么办？
回答：糖尿病...胰岛素...饮食控制...长期高血糖可能出现周围神经病变...
判断：否
原因：核心主题是糖尿病的血糖控制（内分泌疾病），神经病变仅附带提及

示例7:
问题：车祸后头部受伤，CT显示硬膜下血肿怎么办？
回答：硬膜下血肿...颅内压...血肿清除术...观察意识...
判断：是
原因：核心主题是颅脑损伤后硬膜下血肿（神经外科疾病）

示例8:
问题：经常失眠睡不着怎么办？
回答：失眠...睡眠卫生...放松...心理因素...
判断：否
原因：核心主题是失眠（睡眠/心理问题），不是神经系统器质性疾病

示例9:
问题：面瘫了嘴歪眼斜怎么治疗？
回答：面神经麻痹...激素...抗病毒...针灸...
判断：是
原因：核心主题是面神经麻痹（脑神经疾病）

示例10:
问题：手指发麻是什么原因？
回答：腕管综合征...正中神经受压...肌电图检查...
判断：是
原因：核心讨论是周围神经卡压（周围神经疾病）

现在请判断以下问答：

问题：{question}
回答：{answer}

这个问答的核心主题是否是神经系统疾病（神经内科或神经外科）？请只回答"是"或"否"。
判断："""

# =============================================================================
# 二次验证提示词（更严格）
# =============================================================================

DOUBLE_CHECK_SYSTEM_PROMPT = """你是一位严格的神经病学与神经外科学审核专家。请再次确认以下问答是否真正属于神经科学（神经内科、神经外科）范畴。
你必须非常严格：只有核心主题确实是神经系统本身的疾病时才判"是"。精神科疾病一律判"否"。
请只回答"是"或"否"。"""

DOUBLE_CHECK_USER_TEMPLATE = """请再次严格确认：以下问答的核心主题是否是神经系统本身的疾病（神经内科或神经外科）？

严格标准：
- 核心讨论的必须是神经系统疾病（脑血管病、癫痫、帕金森病、痴呆、头痛类疾病、脑神经与周围神经疾病、神经肌肉疾病、中枢神经系统感染、脑及脊髓肿瘤、颅脑及脊髓损伤、脑积水等）
- 以下情况判"否"：
  * 精神科疾病（抑郁症、焦虑症、精神分裂症、强迫症、失眠、心理问题等）
  * 其他系统疾病（糖尿病、高血压、心脏病、肿瘤化疗等）即使提及头痛、头晕、麻木
  * 全身性疾病或感染中神经系统症状仅为伴随表现
  * 单纯骨关节、颈腰部肌肉问题，或耳源性眩晕、眼科疾病

问题：{question}
回答：{answer}

判断："""

# 需要触发二次验证的边界关键词（出现在问题或答案中）
BORDERLINE_KEYWORDS = [
    '抑郁', '焦虑', '失眠', '心理', '精神', '情绪', '睡眠',
    '耳鸣', '耳聋', '眼科', '视力',
    '颈椎', '腰椎', '椎间盘', '骨折', '关节',
    '糖尿病', '高血压', '心脏', '甲状腺',
    '感冒', '发热', '发烧', '感染',
    '肿瘤', '化疗', '放疗',
    '头晕', '头痛', '麻木', '乏力',
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
    关键词预过滤：快速排除明显非神经科学数据

    核心逻辑：
    1. 在【问题+答案】中检查神经科学保护关键词 → 有则跳过预过滤
    2. 在【问题】中检查非神经科学关键词 → 命中且无保护词则直接过滤

    返回值:
        'reject'  - 明确非神经科学，直接过滤
        'unknown' - 无法确定，交给模型判断
    """
    combined = question_text + ' ' + answer_text

    # 第一步：检查神经科学保护关键词（在问题+答案中检查）
    # 只要出现任一神经科学关键词，就不进行预过滤，交给模型判断
    has_neuro_keyword = any(kw in combined for kw in NEURO_PROTECT_KEYWORDS)
    if has_neuro_keyword:
        return 'unknown'

    # 第二步：只在【问题】中检查非神经科学关键词
    has_non_neuro_keyword = any(kw in question_text for kw in NON_NEURO_KEYWORDS)
    if has_non_neuro_keyword:
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
        if re.search(r'不属于|不是神经科学疾病|应该判否|判断为否|回答.*否|答案.*否|判"否"|判断：否', last_part):
            return "否"
        elif re.search(r'属于神经科学|是神经科学疾病|应该判是|判断为是|回答.*是|答案.*是|判"是"|判断：是', last_part):
            return "是"
        elif "否" in last_part[-20:]:
            return "否"
        elif "是" in last_part[-20:] and "不是" not in last_part[-20:]:
            return "是"

    return response.strip()


def judge_is_neuro(response_text):
    """根据提取后的文本判断是否属于神经科学"""
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


def extract_final_answer(response):
    """
    从思维链输出中提取最终答案。
    兼容两种情况：聊天模板已在 assistant 开头加了 "<think>\\n"（输出里只有 </think>），
    或输出自带完整的 <think>...</think>。
    """
    if not response:
        return ""
    if '</think>' in response:
        after = response.split('</think>')[-1].strip()
        if after:
            return after
        # </think> 后没有内容：用原有启发式从思考内容末尾推断
        think_content = response.split('</think>')[0].replace('<think>', '')
        return extract_answer_from_thinking('<think>' + think_content + '</think>')
    # 没有 </think>：很短的输出视为直接作答，否则视为思考被截断
    stripped = response.replace('<think>', '').strip()
    if len(stripped) <= 10:
        return stripped
    return "[TRUNCATED]"


def get_yes_no_token_ids(tokenizer):
    """收集首token为"是"/"否"/"不"的 token id（含带前导空格的变体）"""
    yes_ids, no_ids = set(), set()
    for text, target in [("是", yes_ids), (" 是", yes_ids),
                         ("否", no_ids), (" 否", no_ids),
                         ("不", no_ids), (" 不", no_ids)]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        for tid in ids:
            piece = tokenizer.decode([tid]).strip()
            if piece in ("是", "否", "不"):
                target.add(tid)
                break
    if not yes_ids or not no_ids:
        raise RuntimeError(f"无法在分词器中找到'是'/'否'对应的token: yes={yes_ids}, no={no_ids}")
    return yes_ids, no_ids


def score_yes_no(output, yes_ids, no_ids):
    """
    根据首个生成token的top-k logprobs计算 p_yes 和 "是/否"概率总和。
    返回 (p_yes, mass)；拿不到logprobs时返回 (0.5, 0.0)，即视为不确定。
    """
    comp = output.outputs[0]
    if not comp.logprobs:
        return 0.5, 0.0
    top = comp.logprobs[0]  # dict: token_id -> Logprob
    p_yes = sum(math.exp(lp.logprob) for tid, lp in top.items() if tid in yes_ids)
    p_no = sum(math.exp(lp.logprob) for tid, lp in top.items() if tid in no_ids)
    mass = p_yes + p_no
    if mass <= 0:
        return 0.5, 0.0
    return p_yes / mass, mass


def fit_to_context(tokenizer, question, answer, system_prompt=None, user_template=None):
    '''按token数裁剪问题/回答，保证 prompt + 生成长度不超过 MAX_MODEL_LEN（先裁回答，再裁问题）'''
    _m = build_messages('', '', system_prompt, user_template)
    _tpl_chars = len(_m[0]['content']) + len(_m[1]['content'])
    budget = MAX_MODEL_LEN - (THINK_MAX_TOKENS if ENABLE_THINK_FALLBACK else 1) - 64  # 64 token 留给对话模板、预填思考段等额外开销

    if _tpl_chars + len(question) + len(answer) <= budget:
        return question, answer  # 字符数不小于 token 数，短文本直接跳过整段分词

    def count(q, a):
        msgs = build_messages(q, a, system_prompt, user_template)
        text = msgs[0]['content'] + msgs[1]['content']
        return len(tokenizer.encode(text, add_special_tokens=False))

    n = count(question, answer)
    if n <= budget:
        return question, answer

    a_ids = tokenizer.encode(answer, add_special_tokens=False)
    keep_a = max(len(a_ids) - (n - budget), 256)
    if keep_a < len(a_ids):
        answer = tokenizer.decode(a_ids[:keep_a], skip_special_tokens=True)

    n = count(question, answer)
    if n > budget:
        q_ids = tokenizer.encode(question, add_special_tokens=False)
        keep_q = max(len(q_ids) - (n - budget), 128)
        question = tokenizer.decode(q_ids[:keep_q], skip_special_tokens=True)
    return question, answer


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

class NeuroDiseaseClassifier:
    """神经科学疾病分类器 - vLLM版本 (v3.0：直答 + logprob 打分，不确定样本回退思维链)"""

    def __init__(self, model_path):
        self.debug_count = 0
        self.truncated_count = 0
        self.prefiltered_count = 0
        self.double_check_filtered_count = 0
        self.direct_decided_count = 0     # 直答阶段高置信直接定论的次数
        self.fallback_count = 0           # 回退思维链（或低置信按0.5判）的次数

        print("=" * 70)
        print("🚀 初始化神经科学疾病分类器 (vLLM + DeepSeek-R1) v3.0 直答+logprob")
        print("=" * 70)
        print(f"模型路径: {model_path}")
        print(f"张量并行: {TENSOR_PARALLEL_SIZE} GPU")
        print(f"显存利用率: {GPU_MEMORY_UTILIZATION * 100:.0f}%")
        print(f"最大上下文: {MAX_MODEL_LEN}")
        print(f"批处理大小: {BATCH_SIZE}")
        print(f"量化: {QUANTIZATION}  LIMIT: {LIMIT}  回答截断: {MAX_ANSWER_CHARS}字符  二次验证跳过阈值: {DOUBLE_CHECK_SKIP_P}")
        print(f"置信阈值: 是>={CONF_HIGH} / 否<={CONF_LOW}，是否概率和下限 {MIN_YES_NO_MASS}")
        print(f"思维链回退: {'开启' if ENABLE_THINK_FALLBACK else '关闭'} (最大生成 {THINK_MAX_TOKENS} token)")
        print(f"二次验证: {'开启' if ENABLE_DOUBLE_CHECK else '关闭'}")
        print(f"非神经科学关键词数: {len(NON_NEURO_KEYWORDS)}")
        print(f"神经科学保护关键词数: {len(NEURO_PROTECT_KEYWORDS)}")

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
            max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
            max_num_seqs=512,
            quantization=QUANTIZATION,
            enable_prefix_caching=True,   # system prompt + few-shot 是共享前缀
            trust_remote_code=True,
            dtype="bfloat16",
        )
        self.tokenizer = self.llm.get_tokenizer()

        # 直答：只生成1个token，返回top-k logprobs
        self.direct_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=LOGPROBS_TOPK,
            detokenize=False,
        )
        # 思维链回退：不能用 "\n\n" 作停止符（思考过程本身含空行，会被提前截断）
        self.think_params = SamplingParams(
            temperature=0,
            max_tokens=THINK_MAX_TOKENS,
        )

        self.yes_ids, self.no_ids = get_yes_no_token_ids(self.tokenizer)
        print(f"'是' token ids: {sorted(self.yes_ids)}  '否/不' token ids: {sorted(self.no_ids)}")

        print("✅ vLLM引擎初始化完成")
        print("=" * 70)
        print()

    # -------------------------------------------------------------------------
    # Prompt 构造
    # -------------------------------------------------------------------------

    def _build_prompt(self, question, answer, system_prompt=None, user_template=None, direct=False):
        """
        将消息转换为模型输入prompt。
        direct=True 时在 assistant 开头预填空思考段，让模型跳过思维链直接作答。
        """
        question, answer = fit_to_context(self.tokenizer, question, answer, system_prompt, user_template)
        messages = build_messages(question, answer, system_prompt, user_template)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            sp = system_prompt or SYSTEM_PROMPT
            ut = user_template or USER_PROMPT_TEMPLATE
            prompt = (
                f"<｜begin▁of▁sentence｜>{sp}"
                f"<｜User｜>{ut.format(question=question, answer=answer)}"
                f"<｜Assistant｜>"
            )

        if direct:
            # 新版 R1 模板会以 "<think>\n" 结尾，旧版不会，两种都处理
            if prompt.rstrip().endswith("<think>"):
                prompt = prompt.rstrip() + "\n\n</think>\n\n"
            else:
                prompt = prompt + "<think>\n\n</think>\n\n"
        return prompt

    def _build_prompts(self, items, system_prompt=None, user_template=None, direct=False):
        prompts = []
        for item in items:
            question = get_question_text(item)
            answer = get_answer_text(item)
            answer = answer[:MAX_ANSWER_CHARS] if MAX_ANSWER_CHARS > 0 else answer
            question = question[:MAX_QUESTION_CHARS] if MAX_QUESTION_CHARS > 0 else question
            prompts.append(self._build_prompt(question, answer, system_prompt, user_template, direct))
        return prompts

    # -------------------------------------------------------------------------
    # 判定：直答打分 → 不确定样本回退思维链
    # -------------------------------------------------------------------------

    def _decide(self, items, system_prompt=None, user_template=None, stage="首轮"):
        """对 items 逐条给出 True/False（是否神经科学）"""
        if not items:
            return []

        decisions = [None] * len(items)
        self.last_p_yes = [0.5] * len(items)

        # ---- 直答 + logprob ----
        direct_prompts = self._build_prompts(items, system_prompt, user_template, direct=True)
        direct_outputs = self.llm.generate(direct_prompts, self.direct_params, use_tqdm=False)

        uncertain_idx = []
        for i, output in enumerate(direct_outputs):
            p_yes, mass = score_yes_no(output, self.yes_ids, self.no_ids)
            self.last_p_yes[i] = p_yes if mass >= MIN_YES_NO_MASS else 0.5

            if mass >= MIN_YES_NO_MASS and p_yes >= CONF_HIGH:
                decisions[i] = True
                self.direct_decided_count += 1
            elif mass >= MIN_YES_NO_MASS and p_yes <= CONF_LOW:
                decisions[i] = False
                self.direct_decided_count += 1
            else:
                uncertain_idx.append(i)
                self.fallback_count += 1
                if not ENABLE_THINK_FALLBACK:
                    decisions[i] = p_yes >= 0.5

            if DEBUG_MODE and self.debug_count < DEBUG_SAMPLES:
                question = get_question_text(items[i])
                first_tok = output.outputs[0].text
                state = "不确定→回退" if decisions[i] is None else ("是" if decisions[i] else "否")
                print(f"\n🔍 [{stage}直答] {question[:60]}... | 首token=[{first_tok}] "
                      f"p_yes={p_yes:.3f} mass={mass:.3f} → {state}")
                self.debug_count += 1

        # ---- 不确定样本：思维链推理 ----
        if ENABLE_THINK_FALLBACK and uncertain_idx:
            think_items = [items[i] for i in uncertain_idx]
            think_prompts = self._build_prompts(think_items, system_prompt, user_template, direct=False)
            think_outputs = self.llm.generate(think_prompts, self.think_params, use_tqdm=False)

            for i, output in zip(uncertain_idx, think_outputs):
                raw_response = output.outputs[0].text.strip()
                response = extract_final_answer(raw_response)
                if response == "[TRUNCATED]":
                    self.truncated_count += 1
                decisions[i] = judge_is_neuro(response)

                if DEBUG_MODE:
                    question = get_question_text(items[i])
                    print(f"\n🧠 [{stage}思维链] {question[:60]}... | 提取=[{response[:20]}] "
                          f"→ {'是' if decisions[i] else '否'}")

        return decisions

    def classify_batch(self, items):
        """批量分类：关键词预过滤 → 直答打分(+思维链回退) → 边界样本二次验证"""
        if not items:
            return []

        results = [None] * len(items)

        # ---- 第一步：关键词预过滤 ----
        need_model_items = []
        need_model_indices = []
        for idx, item in enumerate(items):
            question = get_question_text(item)
            answer = get_answer_text(item)
            if keyword_prefilter(question, answer) == 'reject':
                results[idx] = False
                self.prefiltered_count += 1
            else:
                need_model_items.append(item)
                need_model_indices.append(idx)

        if not need_model_items:
            return results

        # ---- 第二步：首轮判定 ----
        first_round = self._decide(need_model_items, stage="首轮")
        first_p = list(self.last_p_yes)

        positive_items = []
        positive_indices = []
        positive_p = []
        for item, orig_idx, is_neuro, p in zip(need_model_items, need_model_indices, first_round, first_p):
            if is_neuro:
                positive_items.append(item)
                positive_p.append(p)
                positive_indices.append(orig_idx)
            else:
                results[orig_idx] = False

        # ---- 第三步：对首轮判"是"且含边界关键词的样本做二次验证 ----
        if ENABLE_DOUBLE_CHECK and positive_items:
            recheck_items = []
            recheck_indices = []
            for item, orig_idx, p in zip(positive_items, positive_indices, positive_p):
                if p < DOUBLE_CHECK_SKIP_P and needs_double_check(get_question_text(item), get_answer_text(item)):
                    recheck_items.append(item)
                    recheck_indices.append(orig_idx)
                else:
                    results[orig_idx] = True

            if recheck_items:
                recheck = self._decide(
                    recheck_items,
                    system_prompt=DOUBLE_CHECK_SYSTEM_PROMPT,
                    user_template=DOUBLE_CHECK_USER_TEMPLATE,
                    stage="二次验证",
                )
                for orig_idx, is_neuro in zip(recheck_indices, recheck):
                    results[orig_idx] = is_neuro
                    if not is_neuro:
                        self.double_check_filtered_count += 1
        else:
            for orig_idx in positive_indices:
                results[orig_idx] = True

        return results

    def process_file(self, input_file, output_file, filtered_file):
        """处理文件"""
        if LIMIT > 0:
            output_file += '.test'  # 测速结果写到 .test 文件，不覆盖正式输出
            filtered_file += '.test'
        progress_file = output_file + '.progress.json'
        print(f"📂 开始处理文件: {input_file}\n")

        print("📖 加载数据到内存...")
        data_list = []
        errors = 0

        with open(input_file, 'rb') as f:
            line_count = sum(1 for _ in f)

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=line_count, desc="读取数据", unit="行", ncols=100):
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

        if LIMIT > 0:
            data_list = data_list[::max(1, len(data_list) // LIMIT)][:LIMIT]
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
        file_mode = 'w'
        start_offset = 0
        resume_kept = 0
        resume_filtered = 0
        if RESUME and os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as pf:
                    prog = json.load(pf)
                if prog.get('total') == total_count:
                    os.truncate(output_file, int(prog['out_size']))
                    os.truncate(filtered_file, int(prog['filt_size']))
                    start_offset = int(prog['done'])
                    resume_kept = int(prog['kept'])
                    resume_filtered = int(prog['filtered'])
                    file_mode = 'a'
                    print(f'⏩ 检测到断点，从第 {start_offset} 条继续')
            except Exception as e:
                print(f'⚠️ 读取断点失败，从头开始: {e}')
                start_offset = 0
                resume_kept = 0
                resume_filtered = 0
                file_mode = 'w'
        filtered_count = 0

        print("🔄 开始批量分类（关键词预过滤 + 直答logprob(+思维链回退) + 二次验证）...")
        print()

        with open(output_file, file_mode, encoding='utf-8') as f_out, \
             open(filtered_file, file_mode, encoding='utf-8') as f_filtered:

            kept_count = resume_kept
            filtered_count = resume_filtered
            progress_bar = tqdm(
                total=total_count,
                initial=start_offset,
                desc="推理进度",
                unit="条",
                ncols=100
            )

            for i in range(start_offset, total_count, BATCH_SIZE):
                batch_items = data_list[i:i + BATCH_SIZE]
                results = self.classify_batch(batch_items)
                progress_bar.update(len(batch_items))

                for item, is_neuro in zip(batch_items, results):
                    if is_neuro:
                        f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                        kept_count += 1
                    else:
                        f_filtered.write(json.dumps(item, ensure_ascii=False) + '\n')
                        filtered_count += 1

                total_processed = kept_count + filtered_count
                keep_ratio = kept_count / total_processed * 100 if total_processed > 0 else 0
                judged = self.direct_decided_count + self.fallback_count
                fallback_ratio = self.fallback_count / judged * 100 if judged > 0 else 0
                progress_bar.set_postfix({
                    '保留': kept_count,
                    '过滤': filtered_count,
                    '比例': f'{keep_ratio:.1f}%',
                    '回退': f'{fallback_ratio:.1f}%'
                })

                f_out.flush()
                f_filtered.flush()
                with open(progress_file + '.tmp', 'w', encoding='utf-8') as pf:
                    json.dump({'total': total_count, 'done': min(i + BATCH_SIZE, total_count),
                               'kept': kept_count, 'filtered': filtered_count,
                               'out_size': os.path.getsize(output_file),
                               'filt_size': os.path.getsize(filtered_file)}, pf)
                os.replace(progress_file + '.tmp', progress_file)

            progress_bar.close()
            if os.path.exists(progress_file):
                os.remove(progress_file)

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
            print(f"  思维链截断(判否): {self.truncated_count:>8} 条")
        judged = self.direct_decided_count + self.fallback_count
        print(f"\n📊 模型判定方式（含二次验证，按调用次数计）:")
        print(f"  直答高置信定论:  {self.direct_decided_count:>8} 次")
        print(f"  低置信回退:      {self.fallback_count:>8} 次 "
              f"({self.fallback_count / judged * 100 if judged else 0:.2f}%)")
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
            'direct_decided': self.direct_decided_count,
            'fallback': self.fallback_count,
            'kept_ratio': kept_count / total_count * 100 if total_count > 0 else 0
        }


# =============================================================================
# 主入口
# =============================================================================

def run_filter():
    """主函数"""
    print("\n" + "=" * 70)
    print("🎯 神经科学数据过滤器")
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

    classifier = NeuroDiseaseClassifier(MODEL_PATH)

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