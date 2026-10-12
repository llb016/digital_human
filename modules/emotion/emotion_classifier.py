# -*- coding:utf-8 -*-

"""
情感识别模块（成员1）

输入:
    用户文本

输出:
    {
        "emotion":"焦虑",
        "confidence":0.85
    }

对外接口（保持稳定，队长在 dev-captain 分支按此集成）:
    from modules.emotion import EmotionClassifier

    classifier = EmotionClassifier()
    classifier.predict("最近论文压力特别大，我感觉很焦虑")
    # -> {"emotion": "焦虑", "confidence": 0.7}

配置（Iter 6：按组长评审意见完成配置化改造）:
    置信度换算系数、情绪判定阈值、并列判定容差不再硬编码在本文件，
    统一从仓库根目录 config.yaml 读取；读不到文件或缺键时，
    回退到本文件内置默认值 DEFAULT_CONFIG（与改造前行为完全一致），
    保证模块可脱离公共配置独立运行、可单元测试。

    根目录 config.yaml 中的扁平键（与团队扁平配置格式一致）:
        emotion_threshold           主程序采纳阈值（原有键；本模块只读取，不改语义、不重复判阈值）
        emotion_confidence_base     置信度 = base + 最高分 * coef
        emotion_confidence_coef
        emotion_confidence_max      置信度上限
        emotion_unknown_confidence  没有任何情绪命中时的置信度
        emotion_min_score           情绪判定阈值：最高分低于它 -> 未知
        emotion_score_epsilon       并列判定容差（浮点差值比较）
        emotion_weights             关键词权重覆盖，如 {愤怒: 2, 悲伤: 2}
        emotion_labels              情绪标签集（声明给全队，成员2 用它做标签漂移观测）
"""

import os

try:  # PyYAML 是团队公共依赖（requirements.txt）；缺失时退回下面的极简扁平解析器
    import yaml as _yaml
except ImportError:  # pragma: no cover - 仅在未安装 PyYAML 的环境触发
    _yaml = None


# ---------------------------------------------------------------------------
# 情绪词典与判定策略（算法本体，保持 Iter 1~5 的逻辑不变）
# 只把「可调参数」外提到 config.yaml：权重、阈值、置信度换算系数、容差
# ---------------------------------------------------------------------------
EMOTION_WORDS = {

    "开心": [
        "开心",
        "高兴",
        "快乐",
        "幸福",
        "激动",
        "成功",
        "完成",
        "奖励",
        "第一名",
        "录取",
        "进步",
        "轻松",
        "礼物",
        "惊喜",
        "表扬",
        "赞美",
        "鼓励",
        "喜欢",
        "爱",
        "温暖",
        "感动",
        "感激",
        "感谢",
        "满足",
    ],

    "焦虑": [
        "焦虑",
        "压力",
        "担心",
        "害怕",
        "睡不着",
        "不知道怎么办",
        "不知道从哪里开始",
        "忙不过来",
        "不知所措",
        "来不及",
        "面试",
        "考试",
        "紧张",
    ],

    "悲伤": [
        "难过",
        "伤心",
        "失去",
        "失败",
        "遗憾",
        "无法接受",
        "痛苦",
    ],

    "愤怒": [
        "生气",
        "愤怒",
        "责怪",
        "委屈",
        "不公平",
        "讨厌",
    ],

    "孤独": [
        "孤独",
        "没人",
        "没有朋友",
        "没人理解",
        "没有人真正理解",
        "没人聊天",
        "找不到人",
        "不知道找谁",
        "朋友分开",
        "越来越远",
        "一个人",
    ],

    "疲惫": [
        "累",
        "疲惫",
        "加班",
        "没有力气",
        "坚持不住",
        "身心俱疲",
        "筋疲力尽",
        "精疲力尽",
        "忙了一天",
        "身体被掏空",
        "心力交瘁",
        "心累",
    ],

    "平静": [
        "放松",
        "平和",
        "安心",
    ],
}

# 转折/负面语义修正词
NEGATIVE_WORDS = [
    "但是",
    "可是",
    "却",
    "落后",
    "失败",
    "不足",
    "不够",
    "担心",
    "害怕",
]

POSITIVE_WORDS = [
    "开心",
    "高兴",
    "成功",
    "考得好",
    "第一名",
    "获得",
    "录取",
]

# 情绪优先级：分数并列时按此顺序裁决
PRIORITY = [
    "愤怒",
    "焦虑",
    "悲伤",
    "孤独",
    "疲惫",
    "开心",
    "平静",
    "未知",
]

UNKNOWN_LABEL = "未知"

CONFIG_FILENAME = "config.yaml"
# 从模块目录向上最多查找几级（本仓库布局：modules/emotion -> digital_human/config.yaml，共 2 级）
CONFIG_SEARCH_LEVELS = 4

# 内置默认值：与改造前硬编码的取值一致，保证读不到配置时行为不变
DEFAULT_CONFIG = {
    "confidence_base": 0.5,
    "confidence_coef": 0.1,
    "confidence_max": 0.95,
    "unknown_confidence": 0.3,
    "min_score": 1,
    "score_epsilon": 1e-6,
    "accept_threshold": 0.6,
    "weights": {"愤怒": 2, "悲伤": 2},
    "labels": [
        "开心",
        "平静",
        "焦虑",
        "悲伤",
        "愤怒",
        "孤独",
        "疲惫",
        UNKNOWN_LABEL,
    ],
}

# 根目录 config.yaml 的扁平键 -> 模块内部配置键
FLAT_KEY_MAP = {
    "emotion_confidence_base": "confidence_base",
    "emotion_confidence_coef": "confidence_coef",
    "emotion_confidence_max": "confidence_max",
    "emotion_unknown_confidence": "unknown_confidence",
    "emotion_min_score": "min_score",
    "emotion_score_epsilon": "score_epsilon",
    "emotion_threshold": "accept_threshold",
    "emotion_weights": "weights",
    "emotion_labels": "labels",
}


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------
def find_public_config(start_dir=None, max_levels=CONFIG_SEARCH_LEVELS):
    """从 start_dir 开始向上逐级查找公共 config.yaml，返回绝对路径或 None。

    为什么要向上找：模块可能被放在不同位置（modules/emotion、顶层模块目录等），
    写死 "../config.yaml" 在另一种布局下会静默退回默认配置。
    """
    cur = os.path.abspath(start_dir or os.path.dirname(os.path.abspath(__file__)))
    for _ in range(max_levels):
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            break
        cur = parent
        candidate = os.path.join(cur, CONFIG_FILENAME)
        if os.path.isfile(candidate):
            return candidate
    return None


def parse_scalar(value):
    """把扁平 YAML 的标量文本转成 int/float/bool/str/list/dict。"""
    value = value.strip()
    if value == "":
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [parse_scalar(part) for part in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        out = {}
        if inner:
            for part in inner.split(","):
                key, _, val = part.partition(":")
                out[key.strip()] = parse_scalar(val)
        return out
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_flat_yaml(text):
    """极简扁平 YAML 解析器（PyYAML 缺失时的兜底）。

    只支持团队 config.yaml 的实际用法：`key: value` 扁平键 + `#` 注释，
    值可以是标量、内联列表 [a, b]、内联映射 {k: v}。
    注意：不支持嵌套缩进结构、也不处理出现在引号内部的 `#`。
    """
    data = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        if not key:
            continue
        data[key] = parse_scalar(value)
    return data


def load_yaml_file(path):
    """读取 YAML 文件：优先 PyYAML，缺失或语法错误时退回极简扁平解析器。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if _yaml is not None:
        try:
            data = _yaml.safe_load(text)
            if isinstance(data, dict):
                return data
        except Exception:  # noqa: BLE001 - 语法错误时尽量保住能读到的扁平键
            pass
    return _parse_flat_yaml(text)


def _number(value, default, minimum=None, maximum=None, notes=None, name=""):
    """数值校验：非法值回退默认值，越界值截断到边界，并记录原因。"""
    if isinstance(value, bool) or value is None:
        parsed = None
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = None

    if parsed is None or parsed != parsed:  # None / nan
        if notes is not None:
            notes.append(f"{name}={value!r} 不是有效数值，已回退默认值 {default!r}")
        return float(default)

    if minimum is not None and parsed < minimum:
        if notes is not None:
            notes.append(f"{name}={parsed} 小于下限 {minimum}，已截断")
        parsed = float(minimum)
    if maximum is not None and parsed > maximum:
        if notes is not None:
            notes.append(f"{name}={parsed} 大于上限 {maximum}，已截断")
        parsed = float(maximum)
    return parsed


def normalize_config(raw=None):
    """扁平原始配置 -> 模块内部配置（含类型校验与默认值回退）。

    raw 里既可以用 config.yaml 的扁平键（emotion_*），
    也可以直接用内部键（confidence_max / weights / ...），后者便于单测与对比实验直接覆盖。

    返回的 dict 额外带一个 `_notes` 字段，记录本次校验中的回退/截断原因，
    便于自测与排查「为什么配置没生效」。
    """
    notes = []
    raw = raw or {}

    merged = dict(DEFAULT_CONFIG)
    merged["weights"] = dict(DEFAULT_CONFIG["weights"])
    merged["labels"] = list(DEFAULT_CONFIG["labels"])

    for flat_key, inner_key in FLAT_KEY_MAP.items():
        if flat_key in raw:
            merged[inner_key] = raw[flat_key]
    for inner_key in DEFAULT_CONFIG:
        if inner_key in raw:
            merged[inner_key] = raw[inner_key]

    cfg = {
        "confidence_base": _number(
            merged["confidence_base"], DEFAULT_CONFIG["confidence_base"], 0.0, 1.0, notes, "confidence_base"),
        "confidence_coef": _number(
            merged["confidence_coef"], DEFAULT_CONFIG["confidence_coef"], 0.0, None, notes, "confidence_coef"),
        "confidence_max": _number(
            merged["confidence_max"], DEFAULT_CONFIG["confidence_max"], 0.0, 1.0, notes, "confidence_max"),
        "unknown_confidence": _number(
            merged["unknown_confidence"], DEFAULT_CONFIG["unknown_confidence"], 0.0, 1.0, notes, "unknown_confidence"),
        "min_score": _number(
            merged["min_score"], DEFAULT_CONFIG["min_score"], 0.0, None, notes, "min_score"),
        "score_epsilon": _number(
            merged["score_epsilon"], DEFAULT_CONFIG["score_epsilon"], 1e-12, None, notes, "score_epsilon"),
        "accept_threshold": _number(
            merged["accept_threshold"], DEFAULT_CONFIG["accept_threshold"], 0.0, 1.0, notes, "accept_threshold"),
    }

    # 置信度上限不能小于起始值，否则 min() 会让下限失效
    if cfg["confidence_max"] < cfg["confidence_base"]:
        notes.append(
            f"confidence_max={cfg['confidence_max']} 小于 confidence_base={cfg['confidence_base']}，"
            "已抬升到 confidence_base"
        )
        cfg["confidence_max"] = cfg["confidence_base"]

    # 权重：仅覆盖词典里真实存在的情绪，非法值忽略并记录
    weights = dict(DEFAULT_CONFIG["weights"])
    raw_weights = merged["weights"]
    if isinstance(raw_weights, dict):
        for name, weight in raw_weights.items():
            name = str(name)
            if name not in EMOTION_WORDS:
                notes.append(f"emotion_weights 中的 {name!r} 不在情绪词典内，已忽略")
                continue
            if isinstance(weight, bool):
                notes.append(f"emotion_weights[{name}]={weight!r} 不是有效数值，已忽略")
                continue
            try:
                weights[name] = float(weight)
            except (TypeError, ValueError):
                notes.append(f"emotion_weights[{name}]={weight!r} 不是有效数值，已忽略")
    else:
        notes.append(f"emotion_weights={raw_weights!r} 不是映射，已回退默认权重 {DEFAULT_CONFIG['weights']}")
    cfg["weights"] = weights

    # 标签集：list 才接受；必须含「未知」
    raw_labels = merged["labels"]
    if isinstance(raw_labels, (list, tuple)) and raw_labels:
        labels = [str(item) for item in raw_labels]
    else:
        notes.append(f"emotion_labels={raw_labels!r} 不是非空列表，已回退默认标签集")
        labels = list(DEFAULT_CONFIG["labels"])
    if UNKNOWN_LABEL not in labels:
        labels.append(UNKNOWN_LABEL)
    cfg["labels"] = labels
    cfg["_notes"] = notes
    return cfg


# ---------------------------------------------------------------------------
# 分类器
# ---------------------------------------------------------------------------
class EmotionClassifier:
    """基于关键词计分 + 优先级裁决的情绪分类器。

    参数:
        config:      可选 dict，直接覆盖配置（实验/单测用，键名与内部配置键一致）
        config_path: 可选 str，指定 config.yaml 路径；默认自动向上查找公共配置
    """

    def __init__(self, config=None, config_path=None):

        raw, source = {}, None

        if config_path:
            if os.path.isfile(config_path):
                raw, source = load_yaml_file(config_path), os.path.abspath(config_path)
            else:
                raw = {}
        else:
            found = find_public_config(os.path.dirname(os.path.abspath(__file__)))
            if found:
                raw, source = load_yaml_file(found), found

        if config:
            raw = dict(raw)
            raw.update(config)

        cfg = normalize_config(raw)
        if config_path and source is None:
            cfg["_notes"].append(f"指定的配置文件不存在：{config_path}，已使用内置默认值")
        elif source is None:
            cfg["_notes"].append("未找到公共 config.yaml，已使用内置默认值")

        self.config = cfg
        self.config_path = source

        # 情绪关键词库（权重可由 config.yaml 的 emotion_weights 覆盖）
        self.emotion_words = {name: list(words) for name, words in EMOTION_WORDS.items()}
        self.weights = {name: cfg["weights"].get(name, 1.0) for name in EMOTION_WORDS}

        self.labels = list(cfg["labels"])
        self.priority = [name for name in PRIORITY if name in self.labels]
        self.priority += [name for name in self.labels if name not in self.priority]

        # 置信度换算系数与判定阈值（全部来自 config.yaml）
        self.confidence_base = cfg["confidence_base"]
        self.confidence_coef = cfg["confidence_coef"]
        self.confidence_max = cfg["confidence_max"]
        self.unknown_confidence = cfg["unknown_confidence"]
        self.min_score = cfg["min_score"]
        self.score_epsilon = cfg["score_epsilon"]
        # 主程序采纳阈值（config.yaml 的 emotion_threshold）：本模块只读取并对外暴露，
        # 不在 predict 里二次过滤，避免「两处各判一次阈值」造成行为不一致。
        self.confidence_threshold = cfg["accept_threshold"]

    # -- 配置自检 ---------------------------------------------------------
    def config_notes(self):
        """返回配置校验过程中的回退/截断说明（空列表 = 配置全部生效）。"""
        return list(self.config.get("_notes", []))

    def is_confident(self, result):
        """按 emotion_threshold 判断识别结果是否达到主程序采纳标准。"""
        return float(result.get("confidence", 0.0)) >= self.confidence_threshold

    # -- 主接口 -----------------------------------------------------------
    def predict(self, text):
        """识别用户文本的情绪。

        参数:
            text: 用户输入文本

        返回:
            {"emotion": <情绪标签>, "confidence": <置信度 0~1>}

        注意：返回结构保持稳定，队长在 dev-captain 分支按此对接；
        如需改动入参或返回结构，需先同步全队。
        """
        text = "" if text is None else str(text)

        scores = {label: 0.0 for label in self.labels}

        # 1) 关键词计分（权重来自 config.yaml，未配置的情绪权重为 1）
        for emotion, words in self.emotion_words.items():
            if emotion not in scores:
                continue
            weight = self.weights.get(emotion, 1.0)
            for word in words:
                if word in text:
                    scores[emotion] += weight

        # 2) 转折/负面语义修正
        if "焦虑" in scores:
            for word in NEGATIVE_WORDS:
                if word in text:
                    scores["焦虑"] += 1

        # 3) 没有任何情绪命中（或最高分低于判定阈值）-> 未知
        #    max_score <= 0 是「完全没有信号」的结构性判断，与阈值解耦：
        #    即使把 emotion_min_score 调到 0，无信号文本仍返回未知，不会误判成第一个优先级情绪。
        max_score = max(scores.values())
        if max_score <= 0 or max_score < self.min_score:
            return {
                "emotion": UNKNOWN_LABEL,
                "confidence": round(self.unknown_confidence, 2),
            }

        # 4) 正向语义修正
        if "开心" in scores:
            for word in POSITIVE_WORDS:
                if word in text:
                    scores["开心"] += 1
                    break

        # 5) 情绪优先级裁决
        #    浮点分数并列改用差值判断：abs(a - b) < epsilon，
        #    避免 == 直接比较时「差 1e-16 就当不并列」的精度风险。
        max_score = max(scores.values())
        emotion = None
        for candidate in self.priority:
            if candidate in scores and abs(scores[candidate] - max_score) < self.score_epsilon:
                emotion = candidate
                break
        if emotion is None:
            # 兜底：配置标签集与优先级列表不一致时，取分数最高的标签
            emotion = max(scores, key=lambda name: scores[name])

        # 6) 置信度换算（系数来自 config.yaml）
        confidence = min(
            self.confidence_base + max_score * self.confidence_coef,
            self.confidence_max,
        )

        return {
            "emotion": emotion,
            "confidence": round(confidence, 2),
        }
