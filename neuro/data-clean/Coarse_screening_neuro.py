import json
import os
import re
from collections import defaultdict
from tqdm import tqdm

# ============================================================================
# 神经科学（神经内科 + 神经外科）关键词体系（附来源标注）
# 说明: 不纳入精神科（ICD-10 F00-F99），精神科疾病放入排除词
# ============================================================================
# 来源缩写说明:
#   [ICD-10]     = WHO ICD-10 分类（维基百科中文 ICD-10 章节条目）
#                  第六章 神经系统疾病 G00-G99
#                  https://zh.wikipedia.org/wiki/ICD-10_第六章：神经系统疾病
#                  第九章 循环系统疾病 I60-I69 脑血管病
#                  https://zh.wikipedia.org/wiki/ICD-10_第九章：循环系统疾病
#                  第二章 肿瘤 C70-C72 / C75.1 / C75.3
#                  https://zh.wikipedia.org/wiki/ICD-10_第二章：肿瘤
#                  第一章 A17 神经系统结核 / A80-A89 中枢神经系统病毒感染
#                  https://zh.wikipedia.org/wiki/ICD-10_第一章：某些传染病和寄生虫病
#                  第十七章 Q00-Q07 神经系统先天性畸形
#                  https://zh.wikipedia.org/wiki/ICD-10_第十七章：先天畸形、变形和染色体异常
#                  第十八章 R25-R29 / R40-R56 神经相关症状体征
#                  https://zh.wikipedia.org/wiki/ICD-10_第十八章：症状、体征和临床与实验室异常所见
#                  第十九章 S04 / S06 / S14 / S24 / S34 / T90 颅脑与脊髓损伤
#                  https://zh.wikipedia.org/wiki/ICD-10_第十九章：损伤、中毒和外因的某些其他后果
#                  第五章 F00-F99 精神和行为障碍（仅用于排除词）
#                  https://zh.wikipedia.org/wiki/ICD-10_第五章：精神和行为障碍
#   [维基]       = 维基百科中文 疾病/药物/手术条目（简体转换）
#                  如 神经外科、神经内科、脑瘤、神经胶质瘤、脑膜瘤、癫痫、帕金森氏症、
#                  中风、蛛网膜下腔出血、多发性硬化症、重症肌无力、脑积水、脑震荡 等
#                  https://zh.wikipedia.org/wiki/神經外科
#                  https://zh.wikipedia.org/wiki/神經內科
#   [MSD]        = 默沙东诊疗手册专业版（中文）
#                  神经系统疾病   https://www.msdmanuals.cn/professional/neurologic-disorders
#                  中枢神经系统肿瘤 https://www.msdmanuals.cn/professional/oncology/central-nervous-system-tumors
#                  外伤性脑损伤   https://www.msdmanuals.cn/professional/injuries-poisoning/traumatic-brain-injury-tbi
#                  脊柱外伤       https://www.msdmanuals.cn/professional/injuries-poisoning/spinal-trauma
#                  先天性神经系统异常 https://www.msdmanuals.cn/professional/pediatrics/congenital-neurologic-anomalies
# ============================================================================

NEURO_KEYWORDS = {
    # =========================================================================
    # 1.1 基础科室与学科
    # =========================================================================
    "基础科室": {

        # 科室
        "神经内科",         # [维基] 临床科室名称
        "神经外科",         # [维基][MSD] 临床科室名称
        "神经科",           # [维基][MSD] 临床科室名称
        "神经病学",         # [维基][MSD] 学科名称(Neurology)
        "脑外科",           # [维基] 神经外科俗称
        "神经系统疾病",     # [ICD-10][维基][MSD] 神经系统疾病总称
        "脑血管病",         # [ICD-10 I60-I69][维基][MSD] ICD-10 I60-I69 类目名
    },

    # =========================================================================
    # 1.2 神经系统解剖结构
    # =========================================================================
    "解剖结构": {

        # 中枢神经系统大体结构
        "中枢神经系统",     # [ICD-10][维基][MSD] CNS
        "大脑",             # [ICD-10][维基][MSD] 端脑/大脑半球
        "小脑",             # [ICD-10][维基][MSD] 运动协调中枢
        "脑干",             # [维基][MSD] 中脑+脑桥+延髓
        "中脑",             # [维基][MSD] 脑干上部
        "脑桥",             # [维基][MSD] 脑干中部
        "延髓",             # [维基][MSD] 脑干下部
        "丘脑",             # [维基][MSD] 间脑主要结构
        "下丘脑",           # [维基][MSD] 间脑
        "基底节",           # [ICD-10][维基][MSD] 锥体外系核团
        "基底神经节",       # [维基][MSD] 同基底节
        "黑质",             # [维基][MSD] 帕金森病病变部位
        "纹状体",           # [维基][MSD] 基底节组成
        "额叶",             # [维基][MSD] 大脑叶
        "顶叶",             # [维基][MSD] 大脑叶
        "颞叶",             # [ICD-10][维基][MSD] 大脑叶
        "枕叶",             # [维基][MSD] 大脑叶
        "岛叶",             # [MSD] 大脑叶
        "边缘系统",         # [维基][MSD] 边缘叶
        "胼胝体",           # [维基][MSD] 连接两侧半球
        "内囊",             # [维基][MSD] 皮质下白质纤维
        "脑白质",           # [维基][MSD] 白质
        "大脑皮层",         # [维基][MSD] 皮质

        # 脑室与脑脊液系统
        "脑室",             # [维基][MSD] 脑室系统
        "侧脑室",           # [维基][MSD] 脑室
        "第三脑室",         # [维基][MSD] 脑室
        "第四脑室",         # [MSD] 脑室
        "脑脊液",           # [ICD-10][维基][MSD] CSF
        "脉络丛",           # [维基][MSD] 分泌脑脊液

        # 脑膜与颅骨
        "脑膜",             # [ICD-10][维基][MSD] 硬脑膜/蛛网膜/软脑膜
        "硬脑膜",           # [维基][MSD] 脑膜外层
        "蛛网膜",           # [ICD-10][维基][MSD] 脑膜中层
        "蛛网膜下腔",       # [ICD-10][维基][MSD] 脑脊液循环腔隙
        "软脑膜",           # [维基][MSD] 脑膜内层
        "颅骨",             # [ICD-10][维基][MSD] 颅腔骨性结构
        "颅底",             # [维基][MSD] 颅腔底部
        "枕骨大孔",         # [维基][MSD] 颅颈交界
        "颅颈交界",         # [MSD] 颅颈交界区

        # 脊髓与周围神经
        "脊髓",             # [ICD-10][维基][MSD] 中枢神经系统组成
        "椎管",             # [维基][MSD] 脊髓所在腔
        "神经根",           # [ICD-10][维基][MSD] 脊神经根
        "脊神经",           # [维基][MSD] 周围神经
        "臂丛",             # [MSD] 上肢神经丛
        "腰骶丛",           # [MSD] 下肢神经丛
        "周围神经",         # [ICD-10][维基][MSD] 周围神经系统
        "周围神经系统",     # [ICD-10][维基][MSD] PNS
        "自主神经系统",     # [ICD-10][维基][MSD] 内脏神经系统
        "交感神经",         # [维基][MSD] 自主神经
        "副交感神经",       # [维基][MSD] 自主神经

        # 脑神经
        "脑神经",           # [ICD-10][维基][MSD] 12对脑神经
        "颅神经",           # [ICD-10][维基][MSD] 同脑神经
        "嗅神经",           # [维基][MSD] 第I对
        "视神经",           # [ICD-10][维基][MSD] 第II对
        "动眼神经",         # [ICD-10][维基][MSD] 第III对
        "滑车神经",         # [维基][MSD] 第IV对
        "三叉神经",         # [ICD-10][维基][MSD] 第V对
        "外展神经",         # [维基][MSD] 第VI对
        "展神经",           # [维基][MSD] 第VI对
        "面神经",           # [ICD-10][维基][MSD] 第VII对
        "前庭蜗神经",       # [维基] 第VIII对
        "听神经",           # [维基][MSD] 第VIII对俗称
        "舌咽神经",         # [MSD] 第IX对
        "迷走神经",         # [维基][MSD] 第X对
        "副神经",           # [ICD-10][维基][MSD] 第XI对
        "舌下神经",         # [MSD] 第XII对

        # 常见周围神经干
        "坐骨神经",         # [维基][MSD] 下肢神经干
        "正中神经",         # [维基][MSD] 上肢神经
        "尺神经",           # [MSD] 上肢神经
        "桡神经",           # [MSD] 上肢神经
        "腓总神经",         # [MSD] 下肢神经

        # 神经细胞学结构
        "神经元",           # [ICD-10][维基][MSD] 神经细胞
        "神经胶质细胞",     # [维基][MSD] 胶质细胞
        "星形胶质细胞",     # [维基][MSD] 胶质细胞
        "少突胶质细胞",     # [维基][MSD] 髓鞘形成细胞
        "小胶质细胞",       # [维基][MSD] 中枢免疫细胞
        "髓鞘",             # [ICD-10][维基][MSD] 轴突外包膜
        "轴突",             # [维基][MSD] 神经纤维
        "突触",             # [维基][MSD] 神经元连接
        "神经递质",         # [维基][MSD] 化学信使
        "运动神经元",       # [维基][MSD] 上/下运动神经元
        "锥体束",           # [维基][MSD] 皮质脊髓束
        "皮质脊髓束",       # [维基][MSD] 运动传导通路
        "神经肌肉接头",     # [维基][MSD] 运动终板
        "血脑屏障",         # [维基][MSD] BBB

        # 脑血管
        "颈内动脉",         # [维基][MSD] 脑供血动脉
        "椎动脉",           # [维基][MSD] 后循环供血
        "基底动脉",         # [维基][MSD] 后循环
        "大脑中动脉",       # [维基][MSD] 前循环
        "大脑前动脉",       # [维基][MSD] 前循环
        "大脑后动脉",       # [维基][MSD] 后循环
        "Willis环",         # [维基][MSD] 大脑动脉环
        "静脉窦",           # [维基][MSD] 硬脑膜静脉窦
    },

    # =========================================================================
    # 1.3 神经系统疾病（神经内科 + 神经外科）
    # =========================================================================
    "神经系统疾病": {

        # ===== 脑血管病 [ICD-10 I60-I69, G45-G46] =====
        "脑卒中",           # [ICD-10 I64][维基][MSD] 卒中总称
        "卒中",             # [维基][MSD] 脑卒中简称
        "中风",             # [ICD-10 I64][维基][MSD] 卒中俗称
        "缺血性脑卒中",     # [维基][MSD] 缺血性卒中
        "缺血性卒中",       # [维基][MSD] 同上
        "脑梗死",           # [ICD-10 I63][维基][MSD] ICD-10 I63
        "脑梗",             # [ICD-10][维基][MSD] 脑梗死简称
        "脑梗塞",           # [维基][MSD] 脑梗死别称
        "脑血栓",           # [MSD] 脑血栓形成
        "脑栓塞",           # [MSD] 心源性栓塞等
        "腔隙性脑梗死",     # [维基][MSD] 腔隙性梗死
        "腔梗",             # [MSD] 腔隙性脑梗死简称
        "短暂性脑缺血发作", # [ICD-10 G45][维基][MSD] TIA,ICD-10 G45
        "TIA",              # [ICD-10][维基][MSD] 短暂性脑缺血发作缩写
        "脑出血",           # [维基][MSD] 脑实质出血
        "颅内出血",         # [ICD-10 I61][维基][MSD] ICD-10 I61/I62
        "出血性卒中",       # [维基][MSD] 出血性脑卒中
        "蛛网膜下腔出血",   # [ICD-10 I60][维基][MSD] ICD-10 I60
        "脑动脉瘤",         # [维基][MSD] 颅内动脉瘤
        "颅内动脉瘤",       # [维基][MSD] 同上
        "动脉瘤破裂",       # [维基][MSD] SAH主要病因
        "脑动静脉畸形",     # [维基][MSD] 脑AVM
        "动静脉畸形",       # [维基][MSD] AVM
        "烟雾病",           # [维基][MSD] Moyamoya病
        "颈动脉狭窄",       # [维基][MSD] 缺血性卒中病因
        "脑静脉窦血栓",     # [维基][MSD] 脑静脉血栓形成
        "静脉窦血栓",       # [维基][MSD] 同上
        "椎基底动脉供血不足", # [维基] 后循环缺血
        "脑小血管病",       # [MSD] 小血管病
        "脑白质病变",       # [MSD] 白质病变
        "血管性痴呆",       # [ICD-10][维基][MSD] 血管性认知障碍
        "脑血管病后遗症",   # [ICD-10 I69] ICD-10 I69

        # ===== 发作性疾病 [ICD-10 G40-G41] =====
        "癫痫",             # [ICD-10 G40][维基][MSD] ICD-10 G40
        "癫痫发作",         # [ICD-10][维基][MSD] 发作
        "癫痫持续状态",     # [ICD-10 G41][维基][MSD] ICD-10 G41 癫痫状态
        "羊癫疯",           # [维基] 癫痫俗称
        "羊角风",           # [维基] 癫痫俗称
        "全面性发作",       # [MSD] 癫痫发作分类
        "局灶性发作",       # [维基][MSD] 癫痫发作分类
        "失神发作",         # [MSD] 全面性发作类型
        "强直阵挛发作",     # [维基][MSD] 全面性发作类型
        "热性惊厥",         # [维基][MSD] 儿童发热惊厥
        "婴儿痉挛症",       # [维基] West综合征
        "Dravet综合征",     # [维基][MSD] 婴儿严重肌阵挛癫痫
        "Lennox-Gastaut综合征", # [MSD] 癫痫性脑病

        # ===== 头痛与颅面痛 [ICD-10 G43-G44, G50] =====
        "偏头痛",           # [ICD-10 G43][维基][MSD] ICD-10 G43
        "紧张型头痛",       # [维基][MSD] ICD-10 G44 其他头痛综合征
        "丛集性头痛",       # [维基][MSD] 三叉自主神经性头痛
        "药物滥用性头痛",   # [MSD] 药物过度使用性头痛
        "低颅压",           # [MSD] 低颅压性头痛
        "特发性颅内压增高", # [MSD] 假性脑瘤
        "三叉神经痛",       # [维基][MSD] ICD-10 G50 三叉神经疾患
        "舌咽神经痛",       # [MSD] 脑神经痛
        "枕神经痛",         # [维基][MSD] 枕部神经痛

        # ===== 运动障碍与变性病 [ICD-10 G10-G26] =====
        "帕金森病",         # [维基][MSD] ICD-10 G20
        "帕金森",           # [ICD-10 G20][维基][MSD] 帕金森病简称
        "帕金森综合征",     # [ICD-10 G21][维基][MSD] ICD-10 G21/G22
        "特发性震颤",       # [维基][MSD] 原发性震颤
        "原发性震颤",       # [维基][MSD] 同上
        "肌张力障碍",       # [ICD-10 G24][维基][MSD] ICD-10 G24
        "痉挛性斜颈",       # [维基][MSD] 颈部肌张力障碍
        "颈部肌张力障碍",   # [MSD] 同上
        "舞蹈病",           # [ICD-10][维基][MSD] 舞蹈症
        "舞蹈症",           # [ICD-10][维基][MSD] 同上
        "亨廷顿病",         # [ICD-10 G10][维基][MSD] ICD-10 G10
        "亨廷顿舞蹈病",     # [维基][MSD] 同上
        "多系统萎缩",       # [维基][MSD] MSA
        "进行性核上性麻痹", # [维基][MSD] PSP
        "共济失调",         # [ICD-10 G11][维基][MSD] ICD-10 G11 遗传性共济失调
        "小脑萎缩",         # [ICD-10][维基][MSD] 小脑萎缩症
        "脊髓小脑性共济失调", # [维基][MSD] SCA
        "肝豆状核变性",     # [维基][MSD] Wilson病(神经型)
        "肌阵挛",           # [维基][MSD] 多动障碍
        "面肌痉挛",         # [维基][MSD] 偏侧面肌痉挛
        "偏侧面肌痉挛",     # [MSD] 同上
        "不宁腿综合征",     # [维基][MSD] RLS

        # ===== 痴呆与认知障碍 [ICD-10 G30-G31] =====
        "阿尔茨海默病",     # [ICD-10 G30][维基][MSD] ICD-10 G30
        "老年痴呆",         # [维基][MSD] 阿尔茨海默病俗称
        "痴呆",             # [ICD-10][维基][MSD] 痴呆综合征
        "路易体痴呆",       # [维基][MSD] DLB
        "额颞叶痴呆",       # [维基][MSD] FTD
        "轻度认知障碍",     # [ICD-10][维基][MSD] MCI
        "认知障碍",         # [ICD-10][维基][MSD] 认知功能障碍
        "正常压力脑积水",   # [维基][MSD] NPH
        "克雅病",           # [MSD] CJD
        "朊病毒病",         # [维基][MSD] 朊蛋白病

        # ===== 中枢神经系统脱髓鞘疾病 [ICD-10 G35-G37] =====
        "多发性硬化",       # [ICD-10 G35][维基][MSD] ICD-10 G35
        "视神经脊髓炎",     # [维基][MSD] NMO
        "视神经脊髓炎谱系疾病", # [MSD] NMOSD
        "NMOSD",            # [MSD] 英文缩写
        "急性播散性脑脊髓炎", # [维基][MSD] ADEM
        "脱髓鞘",           # [ICD-10][维基][MSD] 脱髓鞘病变
        "横贯性脊髓炎",     # [维基][MSD] 急性横贯性脊髓炎

        # ===== 中枢神经系统感染 [ICD-10 G00-G09, A17, A80-A89] =====
        "脑膜炎",           # [ICD-10 G03][维基][MSD] ICD-10 G00-G03
        "细菌性脑膜炎",     # [ICD-10 G00][维基][MSD] ICD-10 G00
        "化脓性脑膜炎",     # [MSD] 细菌性脑膜炎
        "病毒性脑膜炎",     # [ICD-10 A87][维基][MSD] ICD-10 A87
        "结核性脑膜炎",     # [ICD-10 A17][维基][MSD] ICD-10 A17.0
        "隐球菌脑膜炎",     # [维基][MSD] 真菌性脑膜炎
        "脑炎",             # [ICD-10 G04][维基][MSD] ICD-10 G04
        "病毒性脑炎",       # [ICD-10 A86][维基][MSD] ICD-10 A83-A86
        "乙型脑炎",         # [维基] 流行性乙型脑炎
        "自身免疫性脑炎",   # [维基][MSD] 抗体介导脑炎
        "脑脓肿",           # [维基][MSD] ICD-10 G06 颅内脓肿
        "硬膜下积脓",       # [MSD] 颅内感染
        "脑囊虫病",         # [MSD] 脑囊尾蚴病
        "脊髓灰质炎",       # [ICD-10 A80][维基][MSD] ICD-10 A80
        "小儿麻痹",         # [维基] 脊髓灰质炎俗称
        "进行性多灶性白质脑病", # [ICD-10 A81][维基][MSD] PML,ICD-10 A81.2

        # ===== 周围神经与神经肌肉疾病 [ICD-10 G50-G73] =====
        "周围神经病",       # [维基][MSD] 周围神经病变
        "周围神经病变",     # [维基][MSD] 同上
        "多发性神经病",     # [维基][MSD] ICD-10 G60-G64
        "多发性周围神经病", # [MSD] 同上
        "糖尿病周围神经病变", # [维基] 代谢性神经病
        "格林-巴利综合征",  # [维基][MSD] GBS
        "吉兰-巴雷综合征",  # [维基][MSD] GBS规范译名
        "格林巴利综合征",   # [MSD] GBS
        "慢性炎性脱髓鞘性多发性神经病", # [MSD] CIDP
        "CIDP",             # [维基][MSD] 英文缩写
        "面神经麻痹",       # [维基][MSD] ICD-10 G51 面神经疾患
        "面瘫",             # [维基][MSD] 面神经麻痹俗称
        "贝尔麻痹",         # [MSD] Bell麻痹
        "周围性面瘫",       # [MSD] 面神经麻痹
        "坐骨神经痛",       # [维基][MSD] 坐骨神经受压
        "腕管综合征",       # [维基][MSD] 正中神经卡压
        "神经根病",         # [维基][MSD] 神经根疾病
        "胸廓出口综合征",   # [维基][MSD] TOS
        "带状疱疹后神经痛", # [维基][MSD] 神经病理性疼痛
        "神经痛",           # [维基][MSD] 神经性疼痛
        "神经病理性疼痛",   # [维基][MSD] 神经性疼痛
        "复杂区域疼痛综合征", # [MSD] CRPS
        "重症肌无力",       # [ICD-10 G70][维基][MSD] ICD-10 G70
        "肌无力综合征",     # [维基][MSD] Lambert-Eaton
        "肌萎缩侧索硬化",   # [维基][MSD] ALS
        "运动神经元病",     # [MSD] MND
        "脊髓性肌萎缩症",   # [维基][MSD] SMA
        "脊肌萎缩症",       # [维基][MSD] SMA
        "肌营养不良",       # [维基][MSD] ICD-10 G71
        "进行性肌营养不良", # [维基] DMD等
        "周期性瘫痪",       # [MSD] 周期性麻痹
        "周期性麻痹",       # [维基][MSD] 同上
        "腓骨肌萎缩症",     # [维基][MSD] 遗传性运动感觉神经病
        "僵人综合征",       # [维基][MSD] SPS
        "自主神经病",       # [MSD] 自主神经功能障碍

        # ===== 脊髓疾病 [ICD-10 G95, MSD 脊髓疾病] =====
        "脊髓病",           # [维基][MSD] 脊髓病变
        "脊髓压迫症",       # [维基][MSD] 脊髓压迫
        "脊髓空洞症",       # [维基][MSD] ICD-10 G95.0
        "脊髓损伤",         # [ICD-10][维基][MSD] 脊髓外伤
        "脊髓炎",           # [ICD-10 G04][维基][MSD] ICD-10 G04
        "脊髓梗死",         # [MSD] 脊髓缺血
        "脊髓血管畸形",     # [MSD] 脊髓AVM
        "脊髓型颈椎病",     # [维基][MSD] 颈椎病性脊髓病
        "马尾综合征",       # [MSD] 马尾神经受压
        "遗传性痉挛性截瘫", # [维基][MSD] HSP
        "亚急性联合变性",   # [MSD] 维生素B12缺乏
        "小脑扁桃体下疝",   # [维基][MSD] Chiari畸形
        "Chiari畸形",       # [维基][MSD] 小脑扁桃体下疝畸形
        "脊髓栓系",         # [维基] 脊髓栓系综合征

        # ===== 神经系统肿瘤 [ICD-10 C70-C72, C75.1, C75.3] =====
        "脑肿瘤",           # [ICD-10 C71][维基][MSD] ICD-10 C71
        "脑瘤",             # [维基][MSD] 脑肿瘤俗称
        "颅内肿瘤",         # [维基][MSD] 颅内占位
        "颅内占位",         # [维基][MSD] 影像学描述
        "脑癌",             # [ICD-10 C71][维基] ICD-10 C71
        "胶质瘤",           # [维基][MSD] 神经胶质瘤
        "神经胶质瘤",       # [维基][MSD] 同上
        "胶质母细胞瘤",     # [维基][MSD] GBM,WHO 4级
        "星形细胞瘤",       # [维基][MSD] 胶质瘤亚型
        "少突胶质细胞瘤",   # [MSD] 胶质瘤亚型
        "室管膜瘤",         # [维基][MSD] 胶质瘤亚型
        "髓母细胞瘤",       # [维基][MSD] 儿童后颅窝肿瘤
        "脑膜瘤",           # [维基][MSD] 最常见颅内良性肿瘤
        "垂体瘤",           # [维基][MSD] ICD-10 C75.1/垂体腺瘤
        "垂体腺瘤",         # [维基][MSD] 同上
        "听神经瘤",         # [维基][MSD] 前庭神经鞘瘤
        "前庭神经鞘瘤",     # [维基][MSD] 同上
        "神经鞘瘤",         # [维基][MSD] 施万细胞瘤
        "颅咽管瘤",         # [维基][MSD] 鞍区肿瘤
        "松果体区肿瘤",     # [MSD] ICD-10 C75.3
        "生殖细胞瘤",       # [MSD] 松果体区常见肿瘤
        "原发性脑淋巴瘤",   # [MSD] PCNSL
        "脑转移瘤",         # [维基][MSD] 继发性脑肿瘤
        "脑转移",           # [维基][MSD] 转移性脑肿瘤
        "脊髓肿瘤",         # [维基][MSD] 椎管内肿瘤
        "椎管内肿瘤",       # [维基][MSD] 同上
        "血管母细胞瘤",     # [MSD] 小脑血管母细胞瘤
        "神经纤维瘤病",     # [维基][MSD] NF1/NF2
        "神经纤维瘤",       # [维基][MSD] NF相关肿瘤
        "结节性硬化症",     # [维基][MSD] TSC
        "脑膜癌",           # [ICD-10][MSD] ICD-10 C70

        # ===== 颅脑及脊柱脊髓创伤 [ICD-10 S04, S06, S14, S24, S34, T90] =====
        "颅脑损伤",         # [维基][MSD] 颅脑外伤
        "颅脑外伤",         # [维基][MSD] 同上
        "脑外伤",           # [维基][MSD] 颅脑损伤俗称
        "创伤性脑损伤",     # [维基][MSD] TBI
        "外伤性脑损伤",     # [MSD] TBI
        "脑震荡",           # [ICD-10][维基][MSD] 轻型颅脑损伤
        "脑挫伤",           # [MSD] 脑挫裂伤
        "脑挫裂伤",         # [MSD] 同上
        "弥漫性轴索损伤",   # [维基][MSD] DAI
        "硬膜外血肿",       # [维基][MSD] 颅内血肿
        "硬膜下血肿",       # [维基][MSD] 颅内血肿
        "慢性硬膜下血肿",   # [维基][MSD] 老年人常见
        "颅内血肿",         # [维基][MSD] ICD-10 S06 颅内损伤
        "颅骨骨折",         # [维基][MSD] 头部外伤
        "颅底骨折",         # [维基][MSD] 头部外伤
        "颅内损伤",         # [ICD-10 S06][维基] ICD-10 S06

        # ===== 颅内压与脑疝 [ICD-10 G93] =====
        "颅内压增高",       # [维基][MSD] 颅高压
        "颅内高压",         # [维基][MSD] 同上
        "颅内压",           # [维基][MSD] ICP
        "脑水肿",           # [维基][MSD] ICD-10 G93.6
        "脑疝",             # [维基][MSD] 颅内压增高终末表现
        "小脑幕切迹疝",     # [MSD] 脑疝类型
        "枕骨大孔疝",       # [MSD] 小脑扁桃体疝

        # ===== 先天性与儿童神经疾病 [ICD-10 Q00-Q07, G80, G91] =====
        "脑积水",           # [ICD-10 G91][维基][MSD] ICD-10 G91
        "先天性脑积水",     # [ICD-10 Q03][维基][MSD] ICD-10 Q03
        "脊柱裂",           # [ICD-10 Q05][维基][MSD] ICD-10 Q05
        "脑膨出",           # [ICD-10 Q01][MSD] ICD-10 Q01
        "无脑畸形",         # [ICD-10 Q00][MSD] ICD-10 Q00
        "小头畸形",         # [MSD] ICD-10 Q02
        "脑性瘫痪",         # [维基] ICD-10 G80
        "脑瘫",             # [维基][MSD] 脑性瘫痪简称
        "孔洞脑",           # [MSD] 脑穿通畸形

        # ===== 意识障碍与脑病 [ICD-10 G92-G93, R40] =====
        "昏迷",             # [ICD-10 R40][维基][MSD] ICD-10 R40
        "植物状态",         # [维基][MSD] 持续性植物状态
        "植物人",           # [维基][MSD] 植物状态俗称
        "最低意识状态",     # [MSD] MCS
        "闭锁综合征",       # [维基][MSD] 脑桥腹侧病变
        "脑死亡",           # [维基][MSD] 全脑功能不可逆丧失
        "中毒性脑病",       # [ICD-10 G92][维基][MSD] ICD-10 G92
        "韦尼克脑病",       # [MSD] 维生素B1缺乏
        "脑病",             # [ICD-10][维基][MSD] 弥漫性脑功能障碍

        # ===== 睡眠相关神经疾病 [ICD-10 G47] =====
        "发作性睡病",       # [维基][MSD] 猝倒
        "猝倒",             # [维基][MSD] 发作性睡病症状
    },

    # =========================================================================
    # 1.4 神经系统症状与体征
    # =========================================================================
    "神经症状": {

        # --- 头痛眩晕 [ICD-10 R42, R51] ---
        "头痛",             # [ICD-10 R51][维基][MSD] ICD-10 R51
        "头疼",             # [维基] 头痛口语
        "头晕",             # [ICD-10 R42][维基][MSD] ICD-10 R42
        "眩晕",             # [ICD-10 R42][维基][MSD] ICD-10 R42
        "晕厥",             # [ICD-10 R55][维基][MSD] ICD-10 R55

        # --- 发作性症状 [ICD-10 R56] ---
        "抽搐",             # [维基][MSD] 发作性症状
        "惊厥",             # [ICD-10 R56][维基][MSD] ICD-10 R56

        # --- 意识与认知 [ICD-10 R40-R41] ---
        "意识障碍",         # [维基][MSD] 意识水平下降
        "意识丧失",         # [维基][MSD] 发作症状
        "嗜睡",             # [ICD-10][维基][MSD] 意识障碍轻型
        "记忆力下降",       # [MSD] 认知症状
        "记忆障碍",         # [维基][MSD] 认知症状
        "遗忘",             # [ICD-10][维基][MSD] 记忆障碍
        "失语",             # [ICD-10][维基][MSD] 语言障碍
        "构音障碍",         # [ICD-10][维基][MSD] 言语障碍
        "言语障碍",         # [ICD-10 R47][MSD] ICD-10 R47

        # --- 运动障碍 [ICD-10 G81-G83, R25-R27] ---
        "偏瘫",             # [ICD-10 G81][维基][MSD] ICD-10 G81
        "截瘫",             # [ICD-10 G82][维基][MSD] ICD-10 G82
        "四肢瘫",           # [ICD-10 G82][MSD] ICD-10 G82
        "瘫痪",             # [ICD-10][维基][MSD] 运动功能丧失
        "肢体无力",         # [维基][MSD] 运动症状
        "肌无力",           # [ICD-10][维基][MSD] 运动症状
        "肌肉萎缩",         # [ICD-10][维基][MSD] 下运动神经元损害
        "肌萎缩",           # [维基][MSD] 同上
        "共济失调",         # [ICD-10][维基][MSD] 协调障碍
        "步态不稳",         # [维基][MSD] 步态障碍
        "步态异常",         # [维基][MSD] ICD-10 R26
        "震颤",             # [ICD-10][维基][MSD] 不自主运动
        "不自主运动",       # [ICD-10 R25][维基][MSD] ICD-10 R25
        "肌张力增高",       # [MSD] 锥体外系/锥体束征
        "肌强直",           # [维基][MSD] 帕金森病主征
        "肌肉痉挛",         # [维基][MSD] 痉挛
        "运动迟缓",         # [维基][MSD] 帕金森病主征

        # --- 感觉障碍 [ICD-10 R20] ---
        "麻木",             # [维基][MSD] 感觉症状
        "肢体麻木",         # [MSD] 感觉症状
        "感觉异常",         # [维基][MSD] 感觉症状
        "感觉障碍",         # [维基][MSD] 感觉症状

        # --- 脑神经及眼部神经体征 ---
        "复视",             # [ICD-10][维基][MSD] 眼外肌麻痹
        "上睑下垂",         # [ICD-10][维基][MSD] 动眼神经/重症肌无力
        "眼睑下垂",         # [维基][MSD] 同上
        "视野缺损",         # [维基][MSD] 视路病变
        "偏盲",             # [ICD-10][维基][MSD] 视路病变
        "视乳头水肿",       # [维基][MSD] 颅内压增高体征
        "Horner综合征",     # [MSD] 交感神经通路损害

        # --- 神经系统体征 ---
        "颈项强直",         # [MSD] 脑膜刺激征
        "脑膜刺激征",       # [MSD] 脑膜炎体征
        "病理反射",         # [MSD] 锥体束征
        "巴宾斯基征",       # [MSD] 病理反射
        "腱反射",           # [MSD] 深反射
        "大小便失禁",       # [MSD] 马尾/脊髓损害
    },

    # =========================================================================
    # 1.5 诊断与检查
    # =========================================================================
    "诊断检查": {

        # ===== 神经影像 [MSD 神经系统检查和操作] =====
        "头颅CT",           # [维基][MSD] 颅脑CT
        "头部CT",           # [MSD] 同上
        "颅脑CT",           # [MSD] 同上
        "头颅MRI",          # [MSD] 头颅磁共振
        "颅脑磁共振",       # [MSD] 同上
        "脑CT",             # [MSD] 同上
        "脑MRI",            # [MSD] 同上
        "CTA",              # [ICD-10][维基][MSD] CT血管成像
        "CT血管造影",       # [MSD] CTA
        "磁共振血管成像",   # [MSD] MRA
        "MRA",              # [维基][MSD] 英文缩写
        "DSA",              # [维基][MSD] 数字减影血管造影
        "数字减影血管造影", # [MSD] DSA
        "脑血管造影",       # [维基][MSD] 脑导管血管造影
        "弥散加权成像",     # [MSD] DWI
        "DWI",              # [维基][MSD] 英文缩写
        "灌注成像",         # [维基][MSD] CT/MR灌注
        "脊髓造影",         # [维基][MSD] 椎管造影
        "经颅多普勒",       # [维基][MSD] TCD
        "TCD",              # [维基] 英文缩写
        "颈动脉超声",       # [MSD] 颈部血管超声

        # ===== 神经电生理 =====
        "脑电图",           # [维基][MSD] EEG
        "EEG",              # [维基][MSD] 英文缩写
        "视频脑电图",       # [MSD] 长程监测
        "动态脑电图",       # [MSD] 长程监测
        "肌电图",           # [维基][MSD] EMG
        "神经传导",         # [维基][MSD] 神经传导检查
        "诱发电位",         # [维基][MSD] 诱发反应

        # ===== 脑脊液检查 =====
        "腰椎穿刺",         # [维基][MSD] 腰穿
        "腰穿",             # [MSD] 腰椎穿刺简称
        "脑脊液检查",       # [维基][MSD] CSF分析
        "脑脊液压力",       # [维基][MSD] 颅内压测定
        "寡克隆带",         # [MSD] 多发性硬化诊断

        # ===== 量表与评分 =====
        "格拉斯哥昏迷评分", # [MSD] GCS
        "格拉斯哥昏迷指数", # [维基] GCS
        "GCS",              # [维基][MSD] 英文缩写
        "NIHSS",            # [维基][MSD] 卒中量表
        "MMSE",             # [维基] 简易精神状态检查
        "MoCA",             # [MSD] 蒙特利尔认知评估

        # ===== 实验室与特殊检查 =====
        "乙酰胆碱受体抗体", # [MSD] 重症肌无力诊断
        "AQP4",             # [维基] NMOSD抗体
        "神经活检",         # [MSD] 神经和肌肉活检
        "肌肉活检",         # [维基][MSD] 同上
    },

    # =========================================================================
    # 1.6 治疗方法
    # =========================================================================
    "治疗方法": {

        # ===== 脑血管病治疗 =====
        "溶栓",             # [维基][MSD] 静脉溶栓
        "静脉溶栓",         # [MSD] 急性缺血性卒中
        "溶栓治疗",         # [维基][MSD] 同上
        "阿替普酶",         # [维基][MSD] rt-PA
        "rt-PA",            # [维基] 阿替普酶缩写
        "替奈普酶",         # [维基][MSD] 溶栓药
        "尿激酶",           # [维基] 溶栓药
        "机械取栓",         # [维基][MSD] 血管内取栓
        "取栓",             # [维基][MSD] 血管内取栓
        "血管内治疗",       # [维基][MSD] 介入治疗
        "颈动脉内膜剥脱术", # [MSD] CEA
        "颈动脉支架",       # [维基][MSD] CAS
        "依达拉奉",         # [维基][MSD] 神经保护剂
        "丁苯酞",           # [维基] 脑卒中治疗药
        "尼莫地平",         # [维基][MSD] SAH后血管痉挛

        # ===== 降颅压与神经外科手术 =====
        "甘露醇",           # [维基][MSD] 脱水降颅压
        "开颅手术",         # [维基][MSD] 开颅术
        "开颅",             # [维基][MSD] 开颅手术简称
        "去骨瓣减压",       # [MSD] 减压颅骨切除术
        "血肿清除",         # [MSD] 颅内血肿清除
        "脑室引流",         # [维基][MSD] 脑室外引流
        "脑室腹腔分流术",   # [维基][MSD] V-P分流
        "分流术",           # [维基][MSD] 脑积水分流
        "弹簧圈",           # [MSD] 血管内栓塞
        "立体定向",         # [维基][MSD] 立体定向手术
        "伽玛刀",           # [维基][MSD] 立体定向放射外科
        "立体定向放射外科", # [维基][MSD] SRS
        "微血管减压术",     # [维基][MSD] 三叉神经痛/面肌痉挛
        "椎板切除",         # [MSD] 脊髓减压

        # ===== 肿瘤放化疗 =====
        "替莫唑胺",         # [维基][MSD] 胶质瘤化疗

        # ===== 神经调控 =====
        "脑深部电刺激",     # [维基][MSD] DBS
        "深部脑刺激",       # [维基][MSD] DBS
        "DBS",              # [维基][MSD] 英文缩写
        "迷走神经刺激",     # [MSD] VNS
        "神经调控",         # [维基][MSD] 神经调节

        # ===== 抗癫痫药 =====
        "抗癫痫药",         # [维基][MSD] 抗发作药
        "卡马西平",         # [维基][MSD] 抗癫痫药
        "奥卡西平",         # [维基][MSD] 抗癫痫药
        "丙戊酸",           # [维基][MSD] 抗癫痫药
        "丙戊酸钠",         # [维基][MSD] 抗癫痫药
        "左乙拉西坦",       # [维基][MSD] 抗癫痫药
        "拉莫三嗪",         # [维基][MSD] 抗癫痫药
        "托吡酯",           # [维基][MSD] 抗癫痫药
        "苯妥英",           # [维基][MSD] 抗癫痫药
        "苯巴比妥",         # [维基][MSD] 抗癫痫药

        # ===== 神经病理性疼痛 =====
        "加巴喷丁",         # [维基][MSD] 神经痛用药
        "普瑞巴林",         # [维基][MSD] 神经痛用药

        # ===== 帕金森病 =====
        "左旋多巴",         # [维基][MSD] 帕金森病一线药
        "多巴丝肼",         # [维基] 左旋多巴复方
        "卡比多巴",         # [维基][MSD] 左旋多巴复方
        "普拉克索",         # [维基][MSD] 多巴胺受体激动剂
        "罗匹尼罗",         # [维基][MSD] 多巴胺受体激动剂
        "司来吉兰",         # [维基][MSD] MAO-B抑制剂
        "雷沙吉兰",         # [维基][MSD] MAO-B抑制剂
        "恩他卡朋",         # [维基][MSD] COMT抑制剂
        "金刚烷胺",         # [维基][MSD] 抗帕金森药
        "苯海索",           # [维基][MSD] 抗胆碱能药

        # ===== 痴呆 =====
        "多奈哌齐",         # [维基][MSD] 胆碱酯酶抑制剂
        "美金刚",           # [维基][MSD] NMDA受体拮抗剂
        "卡巴拉汀",         # [MSD] 胆碱酯酶抑制剂
        "加兰他敏",         # [维基][MSD] 胆碱酯酶抑制剂

        # ===== 神经免疫/神经肌肉 =====
        "溴吡斯的明",       # [维基] 重症肌无力对症
        "吡斯的明",         # [维基] 同上
        "甲泼尼龙",         # [维基][MSD] 激素冲击
        "静脉注射免疫球蛋白", # [维基][MSD] IVIG
        "IVIG",             # [MSD] 英文缩写
        "利妥昔单抗",       # [维基][MSD] NMOSD/MG
        "依库珠单抗",       # [MSD] 补体抑制剂
        "诺西那生钠",       # [MSD] SMA治疗
        "利鲁唑",           # [维基][MSD] ALS治疗

        # ===== 头痛 =====
        "曲普坦",           # [维基][MSD] 偏头痛急性期
        "舒马曲普坦",       # [维基] 曲坦类
        "佐米曲普坦",       # [维基] 曲坦类
        "氟桂利嗪",         # [维基][MSD] 偏头痛预防
        "肉毒杆菌毒素",     # [维基][MSD] 肌张力障碍/偏头痛
        "肉毒毒素",         # [维基][MSD] 同上

        # ===== 其他 =====
        "巴氯芬",           # [维基][MSD] 痉挛治疗
        "高压氧",           # [维基] 脑损伤辅助
    },
}

# ============================================================================
# 排除词列表（附来源标注）
# 说明: 精神科疾病按要求不纳入；仅当文本中不含任何神经科学关键词时才排除
# ============================================================================
EXCLUDE_KEYWORDS = {

    # --- 精神科疾病（用户要求不纳入精神科）[ICD-10 F00-F99] ---
    "精神分裂症",       # [ICD-10 F20][维基][MSD] ICD-10 F20
    "抑郁症",           # [ICD-10 F32][维基][MSD] 抑郁发作
    "焦虑症",           # [ICD-10][维基][MSD] 焦虑障碍
    "焦虑障碍",         # [ICD-10 F41][MSD] ICD-10 F41
    "双相情感障碍",     # [ICD-10 F31] ICD-10 F31
    "躁狂",             # [ICD-10 F30][维基][MSD] ICD-10 F30
    "强迫症",           # [ICD-10 F42][维基][MSD] ICD-10 F42
    "强迫性障碍",       # [ICD-10 F42] ICD-10 F42
    "恐惧症",           # [ICD-10 F40][维基] ICD-10 F40
    "创伤后应激障碍",   # [ICD-10 F43][维基][MSD] ICD-10 F43.1
    "神经衰弱",         # [ICD-10 F48][维基] ICD-10 F48.0
    "癔症",             # [MSD] 分离转换性障碍
    "疑病症",           # [ICD-10][维基] ICD-10 F45.2
    "自闭症",           # [ICD-10 F84][维基][MSD] ICD-10 F84.0
    "孤独症",           # [ICD-10 F84][MSD] ICD-10 F84
    "多动症",           # [ICD-10 F90][维基] ICD-10 F90
    "注意缺陷多动障碍", # [维基] ADHD
    "抽动秽语综合征",   # [ICD-10 F95][MSD] ICD-10 F95.2
    "神经性厌食",       # [ICD-10 F50][维基] ICD-10 F50.0
    "人格障碍",         # [ICD-10 F60][维基][MSD] ICD-10 F60
    "精神发育迟滞",     # [ICD-10][维基] ICD-10 F70-F79
    "酒精依赖",         # [ICD-10 F10] ICD-10 F10.2
    "精神障碍",         # [ICD-10][维基][MSD] 精神科范畴
    "精神病",           # [ICD-10][维基][MSD] 精神科范畴
    "精神科",           # [维基][MSD] 科室名称
    "心理咨询",         # [MSD] 心理科

    # --- 皮肤科同名词 [ICD-10 L28] ---

    # --- 耳科(内耳性眩晕/耳聋) [ICD-10 H81, H90-H91] ---
    "良性阵发性位置性眩晕", # [ICD-10 H81][MSD] ICD-10 H81.1
    "梅尼埃病",         # [ICD-10 H81][维基][MSD] ICD-10 H81.0

    # --- 骨科单纯骨关节病 [ICD-10 M] ---
    "骨质疏松",         # [ICD-10 M81][维基][MSD] ICD-10 M80-M81
}


# ============================================================================
# 以下为数据处理函数
# ============================================================================

def preprocess_text(text):
    """
    文本预处理：清理和标准化文本
    1. 转换为小写
    2. 规范化空白字符
    3. 去除特殊字符（保留医学常用符号如"-"和希腊字母等）
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    # 保留中文、英文、数字、连字符、希腊字母
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\-αβγδ]', '', text)
    text = text.strip()
    return text

def calculate_relevance_score(text, keywords_dict, exclude_keywords):
    """
    计算文本与神经科学（神经内科/神经外科）的相关性得分

    Args:
        text (str): 待分析文本
        keywords_dict (dict): 关键词字典
        exclude_keywords (set): 排除词集合

    Returns:
        tuple: (得分, 匹配的关键词列表, 命中的类别数)
    """
    text = preprocess_text(text)
    score = 0
    matched_keywords = set()
    matched_categories = set()

    # 首先检查排除词
    for keyword in exclude_keywords:
        if keyword.lower() in text:
            # 如果同时含有神经科学关键词，不直接排除
            has_neuro_keyword = False
            for category, keywords in keywords_dict.items():
                for kw in keywords:
                    if kw.lower() in text and len(kw) >= 2:
                        has_neuro_keyword = True
                        break
                if has_neuro_keyword:
                    break
            if not has_neuro_keyword:
                return -1, [], 0

    # 检查每个类别的关键词（文本已转小写，关键词同样转小写比较，保证TIA/EEG/DBS等英文缩写可匹配）
    for category, keywords in keywords_dict.items():
        category_match = False
        for keyword in keywords:
            if keyword.lower() in text:
                matched_keywords.add(keyword)
                category_match = True
                score += 1
        if category_match:
            matched_categories.add(category)

    category_bonus = len(matched_categories) * 2
    score += category_bonus

    return score, list(matched_keywords), len(matched_categories)

def extract_neuro_data(input_file_path, output_file_path, keywords_dict, exclude_keywords, min_score=3):
    """
    从JSONL文件中筛选神经科学（神经内科/神经外科）相关的问答对

    Args:
        input_file_path (str): 输入文件路径
        output_file_path (str): 输出文件路径
        keywords_dict (dict): 关键词字典
        exclude_keywords (set): 排除词集合
        min_score (int): 最小相关性得分阈值
    """
    extracted_count = 0
    total_lines = 0
    score_distribution = defaultdict(int)

    print(f"开始筛选文件: {input_file_path}")

    try:
        output_dir = os.path.dirname(output_file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 先统计总行数，用于显示进度百分比和剩余时间
        with open(input_file_path, 'rb') as f:
            line_count = sum(1 for _ in f)
        print(f"文件总行数: {line_count}")

        with open(input_file_path, 'r', encoding='utf-8') as infile, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            progress_bar = tqdm(infile, total=line_count, desc="筛选进度", unit="行", ncols=100)
            for line in progress_bar:
                total_lines += 1
                if total_lines % 1000 == 0:
                    progress_bar.set_postfix({
                        '提取': extracted_count,
                        '比例': f'{extracted_count / total_lines * 100:.2f}%'
                    })

                try:
                    data = json.loads(line.strip())

                    # 提取问题
                    questions_list = data.get("questions", [])
                    all_questions = []
                    if questions_list and isinstance(questions_list, list):
                        for inner_list in questions_list:
                            if isinstance(inner_list, list):
                                all_questions.extend([q for q in inner_list if isinstance(q, str)])
                            elif isinstance(inner_list, str):
                                all_questions.append(inner_list)
                    questions_text = " ".join(all_questions)

                    # 提取答案
                    answers_data = data.get("answers", "")
                    if isinstance(answers_data, list) and answers_data:
                        answers_text = answers_data[0]
                    else:
                        answers_text = str(answers_data)

                    # 计算相关性得分
                    text_to_check = questions_text + " " + answers_text
                    score, matched_keywords, num_categories = calculate_relevance_score(
                        text_to_check, keywords_dict, exclude_keywords
                    )

                    score_distribution[score] += 1

                    if score >= min_score and num_categories >= 1:
                        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                        extracted_count += 1

                except json.JSONDecodeError:
                    tqdm.write(f"警告: 第 {total_lines} 行JSON解析失败")
                except Exception as e:
                    tqdm.write(f"警告: 处理第 {total_lines} 行时发生错误: {str(e)}")

            progress_bar.set_postfix({
                '提取': extracted_count,
                '比例': f'{extracted_count / max(total_lines, 1) * 100:.2f}%'
            })
            progress_bar.close()

        print("\n=== 数据筛选完成 ===")
        print(f"总处理行数: {total_lines}")
        print(f"提取的神经科学数据条数: {extracted_count}")
        print(f"提取率: {(extracted_count/total_lines)*100:.2f}%")

        print("\n得分分布:")
        for score in sorted(score_distribution.keys()):
            count = score_distribution[score]
            percentage = (count / total_lines) * 100
            print(f"得分 {score}: {count} 条 ({percentage:.2f}%)")

        print(f"\n结果已保存至: {output_file_path}")

    except FileNotFoundError:
        print(f"错误: 输入文件 {input_file_path} 不存在")
    except Exception as e:
        print(f"错误: 发生意外错误: {str(e)}")


# --- 运行脚本 ---
if __name__ == "__main__":
    input_filename = "/data/HUA-TUO-26M/在线医疗百科/test_datasets.jsonl"
    output_filename = "/data/HUA-TUO-26M/在线医疗百科/Neuro/neuro_test_datasets.jsonl"

    if os.path.exists(input_filename):
        extract_neuro_data(
            input_filename,
            output_filename,
            NEURO_KEYWORDS,
            EXCLUDE_KEYWORDS,
            min_score=3
        )
    else:
        print(f"错误: 文件 '{input_filename}' 不存在")
