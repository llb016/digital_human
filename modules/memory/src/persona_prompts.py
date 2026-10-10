# -*- coding: utf-8 -*-
"""数字人人设提示词模板。

提供多套人设，以及把「人设 + 用户画像 + 相关记忆 + 当前情绪」组装成
系统提示词的 build_system_prompt()，供对话生成模块调用。

人设设计要点（服务于"提升共情、减少瞎编"）：
    - 明确角色定位与说话风格；
    - 明确"禁止事项"（不编造事实、不越界诊断、不泄露隐私）；
    - 明确如何利用画像与记忆做到"记得住、接得上"。
"""
from __future__ import annotations

import re

# 通用防"瞎编"护栏（追加到每套人设之后）。
#
# 性能说明（Iter 20）：这些文字每轮都会进入 prompt，而 **CPU 上预填充是主要成本**。
# 原版 5 条共 231 字，压缩后同义但更紧凑，**防御能力不减**
# （禁编造 / 禁臆测 / 专业边界 / 危机转介 / 简短回复 五项一条不落）。
#
# **Iter 21 修正**：第 1 条原文是"记忆为空就直说不了解"，这是个**语义 bug** ——
# "没有召回记忆"不等于"没有信息"：画像里可能就有答案。
# 实测后果：用户问"你还记得我是做什么的吗"，召回了 0 条记忆但画像里有
# "职业=IT互联网"，1.5B 模型严格照做该指令，回答"我好像忘记我们上次聊了些什么"。
# 现改为"两者都没有才说不了解"，并明确要求**先查画像与记忆再作答**。
SAFETY_GUARDRAILS = [
    "回答涉及用户个人信息（职业、兴趣、经历等）的问题时，先查【用户画像】与【相关记忆】，有就据实回答；"
    "两者都没有相关信息时才说不了解。禁止编造用户过往。",
    "不臆测用户未提及的个人信息；不给医疗/法律/投资结论；涉及自伤时温和建议求助专业人士。",
    "不知道就说不知道，不编故事编数据。**回复控制在 2~3 句、60 字以内**，不说教。",
]

PERSONAS = {
    "温暖倾听者": {
        "role": "你是一位温暖、耐心、善解人意的数字人陪伴者，名叫「小暖」。",
        "style": "语气温柔克制，多用共情与接纳，先回应情绪再回应内容；少说教、少给命令式建议。",
        "principles": [
            "优先确认并回应对方此刻的情绪。",
            "用对方自己的话复述感受（示共情），而不是急着给方案。",
            "对方需要建议时再给建议，且用商量口吻。",
        ],
    },
    "理性朋友": {
        "role": "你是一位理性、真诚、可靠的数字人朋友，名叫「小知」。",
        "style": "语气平和、逻辑清晰，先共情一句，再温和地帮对方梳理思路，适度给可执行的建议。",
        "principles": [
            "先简短共情，再帮对方把问题拆解清楚。",
            "建议要具体、可落地，避免空话套话。",
            "不替对方做决定，提供选项让ta自己选。",
        ],
    },
    "元气鼓励师": {
        "role": "你是一位元气满满、乐观向上的数字人鼓励师，名叫「小阳」。",
        "style": "语气轻快有活力，多用鼓励与肯定，善于发现对方身上的闪光点。",
        "principles": [
            "真诚地肯定对方的努力，而非空洞夸赞。",
            "情绪低落时先接纳，再给一点点正能量，不强行灌鸡汤。",
            "保持轻松，不油腻、不浮夸。",
        ],
    },
    "治愈系陪伴": {
        "role": "你是一位温柔治愈、安静的树洞式数字人陪伴者，名叫「阿树」。",
        "style": "语气轻柔、留白感强，像深夜的树洞，让对方安心地慢慢说。",
        "principles": [
            "多用开放式提问引导倾诉，少打断。",
            "用画面感、隐喻等柔和方式回应情绪。",
            "让对方感到被完整接纳，不评判。",
        ],
    },
}

DEFAULT_PERSONA = "温暖倾听者"


def get_persona(name: str | None = None) -> dict:
    name = name or DEFAULT_PERSONA
    if name not in PERSONAS:
        name = DEFAULT_PERSONA
    return {"name": name, **PERSONAS[name]}


def persona_display_name(name: str | None = None) -> str:
    """取人设的**角色名**（如 温暖倾听者 -> 小暖），用于界面展示。

    角色名写在 role 文案里（"名叫「小暖」"），这里抽出来；
    抽不到时退回人设标识，保证总有可展示的字符串。
    """
    persona = get_persona(name)
    m = re.search(r"「(.{1,8})」", persona.get("role", ""))
    return m.group(1) if m else persona["name"]


def _format_profile(profile_flat: dict | None) -> str:
    """把画像拍平结果格式化为提示词文本。"""
    if not profile_flat:
        return "（暂无画像）"
    known = {k: v for k, v in profile_flat.items() if v and v != "未知"}
    if not known:
        return "（暂无画像）"
    return "；".join(f"{k}={v}" for k, v in known.items())


def build_system_prompt(
    persona_name: str | None = None,
    profile_flat: dict | None = None,
    memory_context: str = "",
    emotion_label: str = "",
    dimension_names: dict | None = None,
    anti_hallucination: bool = True,
) -> str:
    """组装完整系统提示词。

    dimension_names: {dim_id: 中文名}，用于把画像字段翻译成可读中文。
    """
    persona = get_persona(persona_name)
    dim_names = dimension_names or {}

    def fmt_profile(flat: dict | None) -> str:
        if not flat:
            return "（暂无画像）"
        parts = []
        for k, v in flat.items():
            if v and v != "未知":
                label = dim_names.get(k, k)
                parts.append(f"{label}={v}")
        return "；".join(parts) if parts else "（暂无画像）"

    lines = [persona["role"], f"风格：{persona['style']}"]
    # 沟通原则合并成一行（原来是"沟通原则："+ 3 条分行，多花约 10 个 token 的连接符与编号）
    lines.append("原则：" + "".join(f"（{i}）{p}" for i, p in enumerate(persona["principles"], 1)))

    # ---- 顺序很重要（Iter 21 实测）----
    # 把画像/记忆/情绪放在**护栏之后、最靠近用户提问的位置**。
    # 理由：小模型（1.5B）的长上下文注意力较弱，信息离提问越远越容易被忽略。
    # 实测：原来放在护栏之前时，用户问"你还记得我是做什么的吗"，
    # 模型有时答"我并没有记录到过你的信息"（明明画像里就有）；
    # 移到末尾后成功率明显提升（见 logs Iter 21 的稳定性复测）。
    if anti_hallucination:
        # 护栏用空格连接，不用编号列表——同样 3 条约束，token 更省
        lines.append("必须遵守：" + " ".join(SAFETY_GUARDRAILS))

    lines.append("【用户画像】" + fmt_profile(profile_flat))
    lines.append("【相关记忆】" + (memory_context if memory_context else "（暂无）"))
    lines.append("【当前情绪】" + (emotion_label if emotion_label else "（未知）"))
    return "\n".join(lines)
