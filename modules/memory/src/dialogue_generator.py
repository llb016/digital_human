# -*- coding: utf-8 -*-
"""对话生成模块 —— 把人设、画像、记忆、情绪整合为一次回复。

流程：
    1. 用当前用户输入检索 RAG 记忆库，拿到相关历史记忆；
    2. 读取该用户的画像（若已有则使用，否则实时抽取并累积）；
    3. 组装系统提示词（人设 + 画像 + 记忆 + 情绪 + 防瞎编护栏）；
    4. 调用统一大模型生成回复；
    5. 后处理（去重/长度/护栏检查）；
    6. 记住本轮对话、更新用户画像，返回结构化结果。

成员1 的情感识别模块只需把 emotion_label 通过接口传入即可联动。
"""
from __future__ import annotations

import json
import os
import re

from .persona_prompts import build_system_prompt
from .anti_hallucination import FactConsistencyChecker

# ChatML / Qwen 特殊标记（如 <|im_start|>、<|im_end|>、<|endoftext|>）。
# 推理服务端模板应用不当时会漏进回复正文，必须清洗掉。
_SPECIAL_TOKEN_RE = re.compile(r"<\|[A-Za-z_]+\|>")


class DialogueGenerator:
    def __init__(self, config, llm_client, memory_store, profile_extractor, dimension_names=None):
        """
        config:           Config 实例（含 memory/profile/persona 节）。
        llm_client:       LLMClient 实例。
        memory_store:     VectorMemoryStore 实例。
        profile_extractor:ProfileExtractor 实例。
        dimension_names:  {dim_id: 中文名}，用于提示词可读性。
        """
        self.config = config
        self.llm = llm_client
        self.memory = memory_store
        self.extractor = profile_extractor
        self.dimension_names = dimension_names or {}
        self.persona_cfg = config.section("persona")
        self.profile_cfg = config.section("profile")
        self.memory_cfg = config.section("memory")

        self.profiles: dict[str, dict] = {}          # user_id -> {dim_id: {value, confidence}}
        self.history: dict[str, list[dict]] = {}     # user_id -> [{"role":..., "content":...}]
        self.profiles_path = config.resolve("data/profiles.json")

        # 单轮回复的生成长度上限（Iter 22）。
        # 为什么单独设：实测（logs Iter 22）**生成长度是延迟的主导因素**——
        # 生成 4.3~5.5 tok/s，而预填充仅约 2 秒；一次 99 token 的回复要 18 秒。
        # 陪伴对话本就要求"1~3 句"，故默认收紧到 80，同时**不超过团队 config.yaml
        # 里 max_new_tokens 给的模型级上限**（不越权改团队配置，只在调用处收紧）。
        self.model_cfg = config.section("model")
        model_limit = int(self.model_cfg.get("max_new_tokens")
                          or self.model_cfg.get("max_tokens") or 200)
        reply_cap = int(self.persona_cfg.get("max_reply_tokens") or 80)
        self.max_reply_tokens = min(reply_cap, model_limit)
        # 防瞎编校验器（实体级拦截 + 句级观测），配置见 persona.hallucination_check
        self.checker = FactConsistencyChecker(self.persona_cfg.get("hallucination_check"))

        # 成员1 情感识别模块的标签集（仅用于**观测**，绝不拦截）。
        # 逐轮情绪一律以成员1 传入的 emotion_label 为准；这里只记录
        # "配置声明的标签集" 与 "实际收到的标签" 是否一致，便于发现口径漂移。
        self.emotion_cfg = config.section("emotion")
        self.known_emotions = set(self.emotion_cfg.get("labels") or [])
        self.unseen_emotions: set[str] = set()   # 收到过、但不在声明标签集里的值

    def emotion_report(self) -> dict:
        """情绪标签对接情况（供 check_model.py 与成员3 核对口径）。"""
        return {
            "declared_labels": sorted(self.known_emotions),
            "unseen_labels": sorted(self.unseen_emotions),
        }

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def generate(
        self,
        user_input: str,
        user_id: str = "default",
        emotion_label: str = "",
        persona_name: str | None = None,
    ) -> dict:
        """生成一次回复，并更新记忆与画像。返回结构化结果。"""
        user_input = (user_input or "").strip()
        if not user_input:
            return {"response": "", "error": "empty input"}

        persona_name = persona_name or self.persona_cfg.get("default_persona", "温暖倾听者")

        # 情绪标签：一律以成员1 传入的为准。这里只做**观测**——
        # 若收到的标签不在配置声明的标签集里，记录下来供核对口径，但绝不放行/拦截。
        emotion_known = True
        if emotion_label and self.known_emotions:
            emotion_known = emotion_label in self.known_emotions
            if not emotion_known:
                self.unseen_emotions.add(emotion_label)

        # 1) RAG 检索相关记忆
        retrieved = self.memory.search(user_input, top_k=self.memory_cfg.get("top_k"))
        memory_context = self.memory.build_context(user_input, top_k=self.memory_cfg.get("top_k"))

        # 2) 用户画像（累积式：把实时抽取结果合并进历史画像）
        profile_flat = self._update_profile(user_id, user_input)

        # 3) 组装系统提示词
        system = build_system_prompt(
            persona_name=persona_name,
            profile_flat=profile_flat,
            memory_context=memory_context,
            emotion_label=emotion_label,
            dimension_names=self.dimension_names,
            anti_hallucination=bool(self.persona_cfg.get("anti_hallucination", True)),
        )

        # 4) 历史上下文（最近 N 轮）
        hist = self.history.get(user_id, [])[-int(self.persona_cfg.get("max_history_turns", 8)) * 2:]
        messages = [{"role": "system", "content": system}] + hist + [
            {"role": "user", "content": user_input}
        ]

        # 5) 生成（显式传入本模块的回复长度上限，见 __init__ 的 max_reply_tokens）
        raw = self.llm.chat(messages, max_tokens=self.max_reply_tokens)
        response = self._postprocess(raw)

        # 5.5) 事实一致性校验（防瞎编第一道防线，见 src/anti_hallucination.py）
        #      支撑上下文 = 本轮输入 + 召回记忆 + 画像 + 最近历史
        support = self._build_support(
            user_input, memory_context, profile_flat,
            self.history.get(user_id, [])[-int(self.persona_cfg.get("max_history_turns", 8)) * 2:],
        )
        response, ah = self.checker.apply(response, support)

        # 6) 记住本轮 & 更新历史
        self.memory.remember_turn(user_input, response, emotion=emotion_label)
        self._append_history(user_id, user_input, response)

        return {
            "response": response,
            "persona": persona_name,
            "emotion_label": emotion_label,
            "emotion_label_known": emotion_known,   # 是否在声明标签集内（仅观测）
            "retrieved_memories": [{"text": it["text"], "score": round(s, 3)} for it, s in retrieved],
            "profile": profile_flat,
            "hallucination_check": ah,   # 可观测：命中了哪些断言、是否被拦截
            "system_prompt": system,     # 便于自测检查上下文注入是否生效
        }

    @staticmethod
    def _build_support(user_input, memory_context, profile_flat, history) -> str:
        """拼装"支撑上下文"：回复中关于用户的事实必须能在这里找到依据。"""
        parts = [user_input or "", memory_context or ""]
        parts.extend(str(v) for v in (profile_flat or {}).values() if v and v != "未知")
        parts.extend(h.get("content", "") for h in (history or []))
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 画像累积
    # ------------------------------------------------------------------
    def _update_profile(self, user_id: str, conversation: str) -> dict:
        """实时抽取 + 累积合并，返回拍平后的画像 {dim_id: value}。"""
        extracted = self.extractor.extract(conversation)
        cur = self.profiles.get(user_id, {})

        for dim_id, item in extracted.items():
            value = item.get("value", "未知")
            conf = float(item.get("confidence", 0.0))
            prev = cur.get(dim_id, {})
            prev_conf = float(prev.get("confidence", 0.0)) if isinstance(prev, dict) else 0.0
            # 高置信度覆盖低置信度；同为高置信度时保留先前的稳定值
            if value != "未知" and conf >= prev_conf:
                cur[dim_id] = {"value": value, "confidence": conf}
            elif dim_id not in cur:
                cur[dim_id] = {"value": "未知", "confidence": 0.0}

        self.profiles[user_id] = cur
        return {k: (v["value"] if isinstance(v, dict) else v) for k, v in cur.items()}

    def get_profile(self, user_id: str = "default") -> dict:
        cur = self.profiles.get(user_id, {})
        return {k: (v["value"] if isinstance(v, dict) else v) for k, v in cur.items()}

    # ------------------------------------------------------------------
    # 历史管理
    # ------------------------------------------------------------------
    def _append_history(self, user_id: str, user_input: str, response: str):
        self.history.setdefault(user_id, []).append({"role": "user", "content": user_input})
        self.history.setdefault(user_id, []).append({"role": "assistant", "content": response})

    # ------------------------------------------------------------------
    # 后处理：长度与基本护栏
    # ------------------------------------------------------------------
    def _postprocess(self, response: str) -> str:
        resp = (response or "").strip()
        # 1) 剥离 ChatML/Qwen 特殊标记。
        #    Qwen 使用 <|im_start|>role ... <|im_end|> 模板；若服务端模板应用不当
        #    （模板未生效、或被 max_tokens 截断），这些标记会漏进回复正文。
        resp = _SPECIAL_TOKEN_RE.sub("", resp)
        # 2) 剥离模型回显的角色前缀，如 "assistant\n..."
        resp = re.sub(r"^(assistant|system|user)\s*[:：]?\s*\n", "", resp, flags=re.IGNORECASE)
        # 3) 去掉角色名前缀重复，如 "小暖：小暖：..."
        resp = re.sub(r"^([^\s：:]{1,6}[：:])\s*\1", r"\1", resp)
        # 4) 去掉首尾多余的空白与引用符号
        resp = resp.strip().strip("\"' ")
        if not resp:
            resp = "我在听，你可以慢慢说。"
        return resp

    # ------------------------------------------------------------------
    # 持久化（画像/历史，可选）
    # ------------------------------------------------------------------
    def save_profiles(self, path: str | None = None):
        path = path or self.profiles_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"profiles": self.profiles, "history": self.history}, f, ensure_ascii=False, indent=2)

    def load_profiles(self, path: str | None = None):
        path = path or self.profiles_path
        if not os.path.isfile(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.profiles = data.get("profiles", {})
        self.history = data.get("history", {})
        return True
