# -*- coding: utf-8 -*-
"""RAG 向量记忆库 —— 解决长对话"聊天失忆"问题。

核心能力：
    1) 把每轮对话（以及抽取出的画像快照/摘要）编码为向量并持久化；
    2) 给定当前输入，检索最相关的历史记忆（top_k + 相似度阈值）；
    3) 把检索到的记忆组织成可直接注入提示词的文本；
    4) 分层压缩，避免向量库无限膨胀。

分层压缩设计（经实验迭代得出，见 logs/experiment_log.md Iter 6）：
    第一层：原始对话轮(turn)   —— 最近 N 轮，保留完整上下文
    第二层：事实记忆(fact)     —— 超限时，**丢弃助手回声**，把用户的陈述
                                 逐条转成独立记忆。关键：逐条而非拼成一大段，
                                 否则长文档会稀释向量、导致短查询检索不到。
    第三层：摘要记忆(summary)  —— 事实条目也过多时，才把最老的一批合并成一条。

存储：内存中维护 (id, text, metadata, embedding)，并以 JSON 持久化到磁盘，
     零外部依赖、可复现；生产环境可无缝替换为 chromadb/faiss。
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Optional

import numpy as np

from .embedding import cosine_similarity

# 记忆类型：raw=原始对话轮, fact=关键事实, summary=摘要, profile_snapshot=画像快照
KIND_TURN = "turn"
KIND_SUMMARY = "summary"
KIND_FACT = "fact"
KIND_PROFILE = "profile_snapshot"


class VectorMemoryStore:
    def __init__(self, embedder, memory_cfg: dict | None = None):
        self.embedder = embedder
        cfg = memory_cfg or {}
        self.top_k = int(cfg.get("top_k", 5))
        self.threshold = float(cfg.get("similarity_threshold", 0.15))
        # 阈值必须匹配**实际生效**的嵌入后端：不同后端相似度分布差异极大
        # （实测 hashing 命中约 0.25，ST 通常 0.6+）。而 embedder 存在"首选后端
        # 缺依赖时自动降级"的行为，若只有一个全局阈值就会失配，故支持按后端覆盖。
        self.threshold_by_backend = dict(cfg.get("similarity_threshold_by_backend") or {})
        actual_backend = getattr(embedder, "backend_name", None)
        if actual_backend and actual_backend in self.threshold_by_backend:
            self.threshold = float(self.threshold_by_backend[actual_backend])
        # 相对下限：只保留与最佳命中同一量级的记忆（0 = 关闭该机制）
        self.relative_ratio = float(cfg.get("relative_ratio", 0.6))
        self.persist_path = cfg.get("persist_path", "data/memory_store.json")
        self.max_turns = int(cfg.get("max_dialogue_turns", 200))
        self.summarize_every = int(cfg.get("summarize_every", 20))
        self.max_facts = int(cfg.get("max_facts", 60))
        self.items: list[dict] = []  # {id, text, metadata, embedding(list)}

    # ------------------------------------------------------------------
    # 基本写入 / 检索
    # ------------------------------------------------------------------
    def _append(self, text: str, metadata: dict, kind: str) -> str:
        """仅写入，不触发容量压缩（供内部压缩流程使用，避免递归）。"""
        item_id = uuid.uuid4().hex
        emb = np.asarray(self.embedder.embed(text), dtype=np.float32)
        meta = dict(metadata)
        meta.setdefault("kind", kind)
        meta.setdefault("ts", time.time())
        self.items.append({
            "id": item_id,
            "text": text,
            "metadata": meta,
            "embedding": emb.tolist(),
        })
        return item_id

    def add(self, text: str, metadata: Optional[dict] = None, kind: str = KIND_TURN) -> str:
        """写入一条记忆，返回其 id。"""
        item_id = self._append(text, metadata or {}, kind)
        self._enforce_limit()
        return item_id

    def search(self, query: str, top_k: int | None = None, threshold: float | None = None):
        """检索与 query 最相关的记忆，返回 [(item_dict, score), ...]，按分数降序。

        过滤采用**绝对下限 + 相对下限**双重条件：
          - 绝对下限 threshold：过滤"完全无关"（避免无关记忆污染提示词）；
          - 相对下限 threshold = top_score * relative_ratio：只保留与最佳命中
            同一量级的记忆。

        为什么需要相对下限（实测驱动）：同一目标记忆的相似度会随**提问措辞**剧烈波动
        （实测同一只猫：'猫'=0.218、'猫怎么样'=0.082、'我养的那只猫最近怎么样'=0.238，
        相差 3 倍），但**排序始终稳定正确**。因此在一个会漂移的量上画固定绝对线必然
        时松时紧；相对下限则能自适应措辞差异。
        """
        top_k = int(top_k or self.top_k)
        threshold = float(self.threshold if threshold is None else threshold)
        if not self.items:
            return []
        q = np.asarray(self.embedder.embed(query), dtype=np.float32)
        scored = []
        for it in self.items:
            v = np.asarray(it["embedding"], dtype=np.float32)
            scored.append((it, cosine_similarity(q, v)))
        scored.sort(key=lambda x: x[1], reverse=True)

        floor = threshold
        if self.relative_ratio > 0 and scored:
            floor = max(threshold, scored[0][1] * self.relative_ratio)
        return [(it, s) for it, s in scored if s >= floor][:top_k]

    def build_context(self, query: str, top_k: int | None = None) -> str:
        """把检索结果组织为可注入提示词的中文文本块。"""
        hits = self.search(query, top_k=top_k)
        if not hits:
            return "（无相关记忆）"
        lines = []
        for it, score in hits:
            lines.append(f"- [{it['metadata'].get('kind', '')}] {it['text']} (相关度{score:.2f})")
        return "\n".join(lines)

    def remember_turn(self, user_text: str, assistant_text: str, emotion: str = "") -> tuple[str, str]:
        """记录一轮完整对话（用户 + 助手），并返回两条记忆 id。"""
        uid = self.add(user_text, metadata={"speaker": "user", "emotion": emotion}, kind=KIND_TURN)
        aid = self.add(assistant_text, metadata={"speaker": "assistant"}, kind=KIND_TURN)
        return uid, aid

    def add_summary(self, text: str, source_turn_ids: list[str] | None = None):
        """写入一条摘要记忆（用于压缩历史）。"""
        return self.add(text, metadata={"source_turns": source_turn_ids or []}, kind=KIND_SUMMARY)

    def add_profile_snapshot(self, text: str):
        """写入画像快照记忆。"""
        return self.add(text, metadata={}, kind=KIND_PROFILE)

    # ------------------------------------------------------------------
    # 分层容量管理
    # ------------------------------------------------------------------
    def _by_kind(self, kind: str) -> list[dict]:
        return [it for it in self.items if it["metadata"].get("kind") == kind]

    def _enforce_limit(self):
        """分层压缩，控制向量库规模（见模块 docstring 的三层设计）。"""
        # ---- 第一层：原始轮超限 -> 抽取用户陈述为独立事实记忆 ----
        turns = self._by_kind(KIND_TURN)
        if len(turns) > self.max_turns:
            overflow = len(turns) - self.max_turns
            batch = turns[: max(overflow, self.summarize_every)]
            # 只保留用户自己的陈述；助手回复多为回声/模板，丢弃不损信息量
            fact_texts = [
                it["text"] for it in batch
                if it["metadata"].get("speaker") == "user" and it["text"].strip()
            ]
            drop = {it["id"] for it in batch}
            self.items = [it for it in self.items if it["id"] not in drop]
            for text in fact_texts:
                # 用 _append 而非 add，避免压缩过程中再次递归触发压缩
                self._append(text, {"speaker": "user", "compressed": True}, KIND_FACT)

        # ---- 第二层：事实过多 -> 合并最老的一批为摘要 ----
        facts = self._by_kind(KIND_FACT)
        if len(facts) > self.max_facts:
            overflow = len(facts) - self.max_facts
            batch = facts[: max(overflow, self.summarize_every)]
            summary_text = self._heuristic_summarize(batch)
            drop = {it["id"] for it in batch}
            self.items = [it for it in self.items if it["id"] not in drop]
            if summary_text:
                self._append(summary_text, {"source_turn_ids": list(drop)}, KIND_SUMMARY)

    @staticmethod
    def _heuristic_summarize(items: list[dict]) -> str:
        """离线摘要：拼接关键句（生产环境可替换为 LLM 摘要）。"""
        texts = [it["text"] for it in items if it["text"].strip()]
        if not texts:
            return ""
        joined = "；".join(texts)
        return "近期聊到：" + (joined if len(joined) <= 300 else joined[:300] + "…")

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self, path: str | None = None):
        path = path or self.persist_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"items": self.items}, f, ensure_ascii=False, indent=2)

    def load(self, path: str | None = None):
        path = path or self.persist_path
        if not os.path.isfile(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = data.get("items", [])
        return True

    def clear(self):
        self.items = []

    def __len__(self):
        return len(self.items)
