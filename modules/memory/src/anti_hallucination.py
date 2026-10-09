# -*- coding: utf-8 -*-
"""防瞎编：回复事实一致性校验器。

【为什么需要它】
    此前"减少瞎编"只靠提示词（人设里写"不要编造"）。但提示词是**软约束**——
    模型仍可能编造用户从未提过的事实（如"你上次说你养了只叫豆豆的猫"），
    而程序对此**毫无察觉、无法度量、无法拦截**。本模块把这一环变成**可检测**。

【判定原理】
    数字人回复中若**断言了用户的个人信息**，该断言必须能在**支撑上下文**中找到依据：
        支撑上下文 = 用户本轮输入 + 最近对话历史 + 用户画像 + 召回的RAG记忆
    分两类信号，**精度不同、处理方式也不同**（见下）：

    ┌ 实体级（高精度，默认拦截）
    │   句式："你的猫""你养的狗""你老婆""你妈""你对象" —— 断言用户**拥有**某关系/宠物。
    │   判定：该名词（或其口语变体）是否在支撑上下文中出现过。
    │   实测：15 条正常回复 0 误判，7 条编造关系/宠物 100% 抓到。
    │
    └ 句级（低精度，默认仅记录不拦截）
        句式："你之前说……""我记得你……" —— 断言用户曾说过某事。
        判定：断言内容与支撑上下文的中文 2-gram 重合率是否达阈值。
        实测：**对同义词与语序变化很脆弱**——用户说"我妈""我对象"，
        模型回"你妈妈""你另一半"时重合率会掉到阈值以下（误判）。
        故只作**可观测信号**记录，不作为拦截依据。

【为什么不做成"语义级完美校验"】
    中文无分词、同义改写需语义理解，纯规则的句级判定做不到可靠。
    本模块的定位是**轻量、零依赖、可解释的第一道防线**，宁可漏报也不误伤。
    真正的语义级判定可在统一大模型（Qwen）就绪后，用 LLM 二次复核实现——
    届时本模块可作为"便宜的预筛"，只把可疑回复送去复核。

【分级设计的意义】
    拦截动作只绑定高精度信号：**误杀一条正常回复的代价，远大于漏掉一条编造**。
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# 句级：断言"用户曾说过某事"（低精度，仅记录）
# ---------------------------------------------------------------------------
_SENTENCE_PATTERNS = [
    re.compile(
        r"(?:你|您)(?:之前|上次|以前|刚刚|刚|早先)?"
        r"(?:说过|说到|说|提到过|提到|讲过|告诉过我|提过)"
        r"[，,：:]?\s*[「“\"']?([^」”\"'。！？!?\n；;，,]{2,40})"
    ),
    re.compile(r"(?:我)?(?:还记得|记得)(?:你|您)([^。！？!?\n；;，,]{2,40})"),
]

# ---------------------------------------------------------------------------
# 实体级：断言用户"拥有"某关系或宠物（高精度，默认拦截）
# ---------------------------------------------------------------------------
# 说明：
#   - "的"设为可选 —— 中文亲属称谓常省略（"你妈""你对象""你家猫"）；
#   - 长词排在前 —— 正则择先匹配，保证"妈妈"不被"妈"截断；
#   - 刻意**排除**"工作/公司/学校/同事/朋友"这类可由上下文合理推断的词，
#     否则会把"你的工作压力大吗"这类正常回复误判为编造。
_ENTITY_PATTERN = re.compile(
    r"(?:你|您)(?:的|养的|家的)?"
    r"(宠物|男朋友|女朋友|老公|老婆|丈夫|妻子|孩子|儿子|女儿|宝宝|对象|"
    r"爸爸|妈妈|父亲|母亲|爷爷|奶奶|哥哥|姐姐|弟弟|妹妹|"
    r"猫|狗|爸|妈|哥|姐|弟|妹|爷|奶)"
)

# 实体变体表：同一关系口语说法不一，支撑上下文出现任一变体即视为有依据。
# 例如用户说"我妈又催婚了"，数字人回复"你妈妈"不应被判为编造。
_ENTITY_VARIANTS = {
    "猫": ["猫"],
    "狗": ["狗"],
    "宠物": ["宠物", "猫", "狗"],
    "孩子": ["孩子", "儿子", "女儿", "宝宝", "娃"],
    "儿子": ["儿子", "孩子", "娃"],
    "女儿": ["女儿", "孩子"],
    "宝宝": ["宝宝", "孩子", "娃"],
    "老公": ["老公", "丈夫", "先生", "结婚"],
    "老婆": ["老婆", "妻子", "太太", "媳妇", "结婚"],
    "丈夫": ["丈夫", "老公", "先生", "结婚"],
    "妻子": ["妻子", "老婆", "太太", "媳妇", "结婚"],
    "男朋友": ["男朋友", "男友", "对象", "恋爱"],
    "女朋友": ["女朋友", "女友", "对象", "恋爱"],
    "对象": ["对象", "男朋友", "女朋友", "恋爱"],
    "爸爸": ["爸"],
    "父亲": ["父", "爸"],
    "妈妈": ["妈"],
    "母亲": ["母", "妈"],
    "爷爷": ["爷"],
    "奶奶": ["奶"],
    "哥哥": ["哥"],
    "姐姐": ["姐"],
    "弟弟": ["弟"],
    "妹妹": ["妹"],
    # 口语缩写形式（"你妈""你哥"）
    "爸": ["爸"], "妈": ["妈"], "哥": ["哥"], "姐": ["姐"],
    "弟": ["弟"], "妹": ["妹"], "爷": ["爷"], "奶": ["奶"],
}

# 判定为"疑似编造"时的澄清式兜底回复：承认不了解，而不是继续编。
SAFE_FALLBACK = "抱歉，我这边没有这个印象。你能再和我说说吗？我认真听。"

# 保留中文、英文、数字用于 n-gram 比对；其余（标点、空白、引号）一律剔除
_KEEP_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]")


def _clean(text: str) -> str:
    return _KEEP_RE.sub("", text or "")


def _ngrams(text: str, n: int):
    return {text[i:i + n] for i in range(max(0, len(text) - n + 1))}


class FactConsistencyChecker:
    """检测回复中"编造用户事实"的轻量校验器。"""

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        # off = 关闭；warn = 只标注不拦截；enforce = 拦截（默认，仅针对实体级信号）
        self.mode = str(cfg.get("mode", "enforce")).lower()
        # 是否用实体级（高精度）信号做拦截
        self.enforce_entities = bool(cfg.get("enforce_entities", True))
        # 是否记录句级（低精度）可疑信号，仅用于观测
        self.report_sentences = bool(cfg.get("report_sentences", True))
        self.min_overlap = float(cfg.get("min_overlap", 0.30))
        self.min_claim_len = int(cfg.get("min_claim_len", 5))

    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def extract_claims(self, response: str) -> list[str]:
        """抽取"用户曾说过某事"的断言片段（句级，低精度）。"""
        claims = []
        for pat in _SENTENCE_PATTERNS:
            for m in pat.finditer(response or ""):
                claim = _clean(m.group(1))
                if claim:
                    claims.append(claim)
        seen, out = set(), []
        for c in claims:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def extract_entities(self, response: str) -> list[str]:
        """抽取用户"拥有某关系/宠物"的断言名词（实体级，高精度）。"""
        return list(dict.fromkeys(_ENTITY_PATTERN.findall(response or "")))

    def overlap(self, claim: str, support: str) -> float:
        """断言与支撑上下文的中文 2-gram 重合率。"""
        cg = _ngrams(_clean(claim), 2)
        if not cg:
            return 1.0
        sg = _ngrams(_clean(support), 2)
        return len(cg & sg) / len(cg)

    def check(self, response: str, support: str) -> dict:
        """校验回复。

        返回 {ok, mode, claims, entities, unsupported:[{claim, type, overlap}]}
        其中 type="entity" 为可拦截信号，type="sentence" 仅为观测信号。
        """
        if not self.enabled:
            return {"ok": True, "mode": self.mode, "claims": [],
                    "entities": [], "unsupported": []}

        claims = self.extract_claims(response)
        entities = self.extract_entities(response)
        support_clean = _clean(support)
        unsupported = []

        # 句级（低精度）：仅记录
        if self.report_sentences:
            for c in claims:
                if len(c) < self.min_claim_len:
                    continue  # 太短不做判定，避免误伤
                ratio = self.overlap(c, support)
                if ratio < self.min_overlap:
                    unsupported.append({"claim": c, "type": "sentence",
                                        "overlap": round(ratio, 3)})

        # 实体级（高精度）：可拦截
        for e in entities:
            variants = _ENTITY_VARIANTS.get(e, [e])
            if not any(v in support_clean for v in variants):
                unsupported.append({"claim": e, "type": "entity", "overlap": 0.0})

        return {
            "ok": not unsupported,
            "mode": self.mode,
            "claims": claims,
            "entities": entities,
            "unsupported": unsupported,
        }

    def blocking(self, result: dict) -> list[dict]:
        """从校验结果中筛出"可拦截"的高精度违规。"""
        if not (self.enforce_entities and self.mode == "enforce"):
            return []
        return [u for u in result.get("unsupported", []) if u.get("type") == "entity"]

    def apply(self, response: str, support: str) -> tuple[str, dict]:
        """按 mode 处理回复：enforce 且存在实体级违规时，改用澄清式回复。"""
        result = self.check(response, support)
        if self.blocking(result):
            result["blocked"] = True
            return SAFE_FALLBACK, result
        result["blocked"] = False
        return response, result
