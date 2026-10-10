# -*- coding: utf-8 -*-
"""用户画像维度抽取器。

两种后端：
    heuristic —— 基于关键词词典的规则抽取（离线可用，用于自测与基线）。
    llm       —— 调用团队统一大模型做结构化抽取（更高准确率，需模型就绪）。

输出统一格式：
    {
        "<dim_id>": {"value": "<选项值>", "confidence": 0.0~1.0},
        ...
    }
所有枚举维度在信息不足时一律回落到 "未知"，避免模型强行猜测（防"瞎编"）。
"""
from __future__ import annotations

import json
import re

# ---------------------------------------------------------------------------
# 启发式关键词词典：dim_id -> [(value, [关键词...]), ...]
# 命中任一关键词即判定为该值；优先级按列表顺序（先匹配先得）。
# 该词典仅用于离线基线与自测，模型就绪后应切换到 llm 后端。
# ---------------------------------------------------------------------------
KEYWORD_RULES: dict[str, list[tuple[str, list[str]]]] = {
    "gender": [
        ("男", ["我是男", "我作为男生", "我是个男", "男朋友", "我是男生"]),
        ("女", ["我是女", "我作为女生", "我是个女", "女朋友", "我是女生"]),
    ],
    "age_group": [
        ("未成年", ["初中", "高中", "我未成年", "我今年16", "我今年17"]),
        ("18-24", ["大学", "上大学", "大一", "大二", "大三", "大四", "刚毕业", "大学生"]),
        ("25-34", ["工作", "上班", "职场", "程序员", "加班", "打工"]),
        ("35-44", ["孩子", "结婚", "房贷", "中年"]),
        ("45-59", ["退休", "孩子上大学", "不惑"]),
        ("60以上", ["退休", "孙子", "孙女", "养老"]),
    ],
    "occupation": [
        ("学生", ["学生", "上学", "上课", "考试", "写作业", "读研", "读博", "考研", "大学", "高中", "初中"]),
        # 注意：不可用裸"运营"（"新媒体运营""电商运营"不是 IT 互联网岗），见 logs Iter 21
        ("IT互联网", ["代码", "程序员", "开发", "产品经理", "算法", "写代码", "互联网", "前端", "后端"]),
        ("教育", ["老师", "教师", "讲课", "学生", "备课"]),
        ("医疗", ["医生", "护士", "医院", "患者"]),
        ("金融", ["银行", "证券", "投资", "基金", "保险"]),
        ("制造业", ["工厂", "车间", "产线", "制造业"]),
        ("自由职业", ["自由职业", "接单", "个体", "自媒体", "主播", "创业"]),
    ],
    "location": [
        ("一线城市", ["北京", "上海", "深圳", "广州", "一线"]),
        ("二线城市", ["杭州", "成都", "武汉", "南京", "西安", "重庆", "二线"]),
        ("海外", ["国外", "美国", "英国", "日本", "澳洲", "留学", "海外"]),
    ],
    # 注意（Iter 20）：**不再抽取"当前情绪"**。
    # 逐轮情绪由成员1 的情感识别模块负责（经 emotion_label 传入并注入提示词）；
    # 本模块若再抽一份，同一 prompt 会出现两个情绪标签、可能互相矛盾。
    # 只保留下面 emotion_tone（跨轮次的长期基调），二者粒度不同。
    "emotion_tone": [
        ("积极", ["开心", "高兴", "幸福", "期待", "满意", "顺利", "真好"]),
        ("消极", ["难过", "焦虑", "压力", "烦", "累", "崩溃", "痛苦", "迷茫", "不开心", "低落"]),
        ("中性", ["还好", "一般", "正常", "没什么"]),
    ],
    "stress_level": [
        ("高", ["压力大", "压力山大", "喘不过气", "扛不住", "崩溃", "焦虑", "deadline", "deadline"]),
        ("中", ["有点压力", "有压力", "忙", "加班"]),
        ("低", ["轻松", "没压力", "没有压力", "没什么压力", "悠闲", "清闲"]),
    ],
    "loneliness_level": [
        ("高", ["一个人", "孤独", "孤单", "寂寞", "没人陪", "没人说话", "一个人吃饭"]),
        ("中", ["偶尔一个人", "有点孤独", "朋友不多"]),
        ("低", ["和朋友一起", "有人陪", "不孤单", "朋友多"]),
    ],
    "hobbies": [
        ("阅读", ["看书", "读书", "阅读", "小说"]),
        ("影视", ["看电影", "电影", "追剧", "电视剧", "动漫", "刷剧"]),
        ("游戏", ["打游戏", "游戏", "开黑", "手游", "switch", "王者"]),
        ("音乐", ["听歌", "音乐", "唱歌", "弹琴", "吉他"]),
        ("运动", ["跑步", "健身", "打球", "运动", "瑜伽", "游泳", "篮球", "足球"]),
        # 注意：不可用裸"吃"（"一个人吃饭"不是美食爱好），见 logs Iter 3
        ("美食", ["美食", "做饭", "探店", "火锅", "奶茶", "吃货", "喜欢吃"]),
        ("旅行", ["旅游", "旅行", "出去玩", "爬山", "露营"]),
        ("科技", ["数码", "科技", "手机", "电脑", "AI", "人工智能"]),
    ],
    "topic_preference": [
        # 注意：不可用裸"喜欢"（"喜欢看科幻电影"不是情感话题），见 logs Iter 3
        ("情感关系", ["感情", "恋爱", "分手", "对象", "男朋友", "女朋友", "暗恋", "异地"]),
        ("工作学习", ["工作", "上班", "学习", "考试", "作业", "老板", "同事", "读研"]),
        ("家庭", ["爸妈", "父母", "家里", "妈妈", "爸爸", "家人", "吵架"]),
        ("健康", ["身体", "健康", "失眠", "生病", "睡眠"]),
        ("兴趣分享", ["游戏", "音乐", "电影", "动漫", "看剧", "爱好"]),
        ("日常闲聊", ["今天", "天气", "吃饭", "日常"]),
    ],
    "comfort_style": [
        ("倾听型", ["听我说", "听我讲", "倾诉", "不用给建议", "陪我说说"]),
        ("建议型", ["给我建议", "怎么办", "支支招", "帮我出主意", "你觉得"]),
        ("鼓励型", ["鼓励", "加油", "给我打气", "夸夸"]),
        ("陪伴型", ["陪陪我", "陪我", "在身边", "陪着我"]),
        ("幽默型", ["逗我笑", "讲个笑话", "开心点", "搞笑"]),
    ],
    "relationship_status": [
        # 注意：不可用裸"一个人"（独处≠单身），见 logs Iter 3
        ("单身", ["单身", "没对象", "单身狗", "没人要", "没有对象"]),
        ("恋爱中", ["男朋友", "女朋友", "对象", "恋爱", "异地恋", "谈恋爱"]),
        ("已婚", ["老公", "老婆", "结婚", "我先生", "我太太", "婚姻"]),
        ("复杂", ["暧昧", "分手", "离婚", "感情复杂"]),
    ],
    "family_relation": [
        ("融洽", ["爸妈很好", "家里和睦", "和爸妈关系好", "家人支持"]),
        ("紧张", ["吵架", "爸妈不理解", "和家里闹", "关系紧张", "父母逼", "催婚"]),
        ("一般", ["和家里一般", "跟爸妈一般"]),
    ],
    "social_circle": [
        ("朋友较多", ["朋友多", "很多朋友", "社交圈大", "聚会"]),
        ("少数密友", ["朋友不多", "几个朋友", "密友", "知心朋友", "能交心"]),
        ("较孤单", ["没朋友", "没人", "没有朋友", "没人说话", "连说话的人都没",
                   "社交恐惧", "社恐", "不社交", "一个人"]),
    ],
    "sleep_quality": [
        ("差", ["失眠", "睡不好", "睡不着", "熬夜", "睡眠差", "半夜醒"]),
        ("良好", ["睡得香", "睡眠好", "睡得不错", "睡眠质量好"]),
        ("一般", ["睡眠一般", "睡得一般", "偶尔失眠"]),
    ],
    "work_study_pressure": [
        ("高", ["考试", "期末", "考研", "deadline", "加班", "工作压力", "学习压力", "项目紧张", "赶工"]),
        ("中", ["有点忙", "作业多", "工作忙"]),
        ("低", ["工作轻松", "学习轻松", "不忙"]),
    ],
    "personality": [
        ("内向", ["内向", "社恐", "社交恐惧", "慢热", "不爱说话", "宅"]),
        ("外向", ["外向", "自来熟", "爱社交", "活泼", "话多", "喜欢交朋友"]),
    ],
    "attachment_need": [
        ("高", ["想找人陪", "需要陪伴", "想找人说话", "一个人好累", "想要人陪", "陪我"]),
        ("中", ["偶尔想找人聊", "有时想聊"]),
        ("低", ["喜欢一个人", "不想被打扰", "不需要陪伴"]),
    ],
}


# 否定修饰识别（调优记录见 logs/experiment_log.md）：
#   第一版用单字否定表 "不没别无未莫"，误伤了 "特别开心"/"特别想要人陪"
#   （"别" 属于 "特别" 而非否定），故改为"否定词 + 可选程度副词"的正则，
#   只匹配紧邻关键词左侧的否定结构，避免跨词误判。
# "别" 仅在非"特别/分别/个别/差别/级别/识别/告别/离别/派别/区别/类别"等复合词中才算否定。
NEG_PATTERN = re.compile(
    r"(?:[不没无未](?:太|很|怎么|那么|这么|是|会|能|多|再|有|要|想|什么)?"
    r"|(?<![特分个差级识告离派区类])别(?:太|这么|那么|再)?)$"
)

# 后置否定：否定词出现在关键词右侧，如"我一点压力都没有"、"焦虑没了"。
NEG_POST_PATTERN = re.compile(r"^(?:都|也|并|一点都|全)?(?:没有|没了|不存在|不怎么|不太)")


def _contains_keyword(text: str, keyword: str) -> bool:
    """在 text 中查找 keyword，且该次出现未被紧邻的否定结构修饰。

    例：「最近总是不开心」中 "开心" 被 "不" 否定 -> 不命中；
        「今天特别开心」中 "开心" 未被否定 -> 命中；
        「我一点压力都没有」中 "压力" 被后置的 "都没有" 否定 -> 不命中。
    """
    start = 0
    while True:
        i = text.find(keyword, start)
        if i == -1:
            return False
        prefix = text[max(0, i - 6):i]
        suffix = text[i + len(keyword): i + len(keyword) + 4]
        if not NEG_PATTERN.search(prefix) and not NEG_POST_PATTERN.search(suffix):
            return True
        start = i + 1


class ProfileExtractor:
    def __init__(self, dimensions: list[dict], config: dict | None = None, llm_client=None):
        """
        dimensions: 来自 profiles/profile_dimensions.yaml 的维度定义列表。
        config:     profile 节配置（含 backend / min_confidence）。
        llm_client: 可选，backend='llm' 时需要。
        """
        self.dimensions = dimensions
        self.dim_by_id = {d["id"]: d for d in dimensions}
        self.backend = (config or {}).get("backend", "heuristic")
        self.min_confidence = float((config or {}).get("min_confidence", 0.5))
        self.llm = llm_client

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def extract(self, conversation: str) -> dict:
        """从对话文本抽取全部维度，返回 {dim_id: {value, confidence}}。"""
        if self.backend == "llm" and self.llm is not None:
            return self._extract_llm(conversation)
        return self._extract_heuristic(conversation)

    def to_flat(self, result: dict) -> dict:
        """把抽取结果拍平成 {dim_id: value}，便于评测/存储。"""
        return {k: (v["value"] if isinstance(v, dict) else v) for k, v in result.items()}

    # ------------------------------------------------------------------
    # 启发式抽取
    # ------------------------------------------------------------------
    def _extract_heuristic(self, conversation: str) -> dict:
        result = {}
        for dim in self.dimensions:
            dim_id = dim["id"]
            value, conf = self._match_dimension(dim_id, conversation)
            result[dim_id] = {"value": value, "confidence": conf}
        return result

    def _match_dimension(self, dim_id: str, conversation: str):
        rules = KEYWORD_RULES.get(dim_id, [])
        for value, keywords in rules:
            for kw in keywords:
                if _contains_keyword(conversation, kw):
                    return value, 1.0
        return "未知", 0.0

    # ------------------------------------------------------------------
    # LLM 抽取（结构化 JSON）
    # ------------------------------------------------------------------
    def _extract_llm(self, conversation: str) -> dict:
        prompt = self._build_llm_prompt(conversation)
        raw = self.llm.chat([
            {"role": "system", "content": "你是一个用户画像抽取助手，只输出 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ], temperature=0.0)
        parsed = self._parse_json(raw)
        # 归一化：只保留合法维度与合法选项，其余回落到"未知"
        result = {}
        for dim in self.dimensions:
            dim_id = dim["id"]
            value = parsed.get(dim_id, "未知")
            conf = 0.9
            if isinstance(value, dict):
                conf = float(value.get("confidence", 0.9))
                value = value.get("value", "未知")
            options = dim.get("options", [])
            if options and value not in options:
                value = "未知"
                conf = 0.0
            result[dim_id] = {"value": value, "confidence": conf}
        return result

    def _build_llm_prompt(self, conversation: str) -> str:
        dims_desc = []
        for d in self.dimensions:
            opts = "、".join(str(o) for o in d.get("options", []))
            dims_desc.append(
                f"- {d['id']}({d['name']})：{d['description']} 候选值：[{opts}]"
            )
        return (
            "请根据下面的用户对话，抽取用户画像维度，输出 JSON 对象。\n\n"
            "维度清单：\n" + "\n".join(dims_desc) + "\n\n"
            "要求：\n"
            "1. 每个维度只能从候选值中选择一个；信息不足时填\"未知\"，不要臆测。\n"
            "2. 仅输出 JSON，格式如 {\"gender\": {\"value\": \"男\", \"confidence\": 0.9}, ...}。\n\n"
            f"对话内容：\n{conversation}\n"
        )

    @staticmethod
    def _parse_json(text: str):
        """从模型输出中稳健地提取 JSON 对象。"""
        text = text.strip()
        # 去掉可能的 markdown 代码块围栏
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # 退而求其次：截取第一个 { 到最后一个 }
            start, end = text.find("{"), text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except json.JSONDecodeError:
                    pass
        return {}
