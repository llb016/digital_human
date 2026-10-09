# -*- coding: utf-8 -*-
"""文本向量化（embedding）——为 RAG 记忆库提供向量检索能力。

三种后端，按 config.yaml 的 embedding.backend 选择：
    hashing               —— 默认。中文字符 n-gram + 特征哈希 + L2 归一化，
                            仅依赖 numpy，离线可用，检索质量对中文短文本够用。
    sentence_transformers —— 本地高质量向量（需 pip install sentence-transformers）。
    api                   —— 远程 OpenAI 兼容 /embeddings 接口。

所有后端统一接口：embed(text)->np.ndarray, embed_batch(texts)->np.ndarray。
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request

import numpy as np


def _normalize(text: str) -> str:
    """归一化：折叠空白、统一为小写（对英文）、保留中文与常用符号。"""
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# 中文高频虚词/功能字。由**纯虚词**构成的 n-gram（如"你我""的什""什么""吗"）
# 几乎不携带检索信息，却会在任意两句中文里大量共现，造成虚假相似度。
# 实测（logs Iter 11）：查询"你还记得我养的猫叫什么吗"与无关的日常回声因共享
# "你/我/的/什/么/吗"6 个单字，相似度(0.311)竟高于真正相关的"我养了一只叫豆豆的橘猫"(0.175)。
FUNCTION_CHARS = set(
    "的了着过地得我你他她它您咱们"
    "是不没无未别很太更最挺蛮好"
    "在有无会要能可就都也还又再才"
    "和与及或而且但只则如若因由于"
    "这那哪什么怎怎样么呢吗吧啊呀哦嗯"
    "个些位种次点上下中里外前后"
    "来说去到给被把让从对向为以之其此该等"
    "一二三四五六七八九十零两"
)


class HashingEmbedder:
    """中文字符 n-gram 特征哈希嵌入（零外部依赖）。

    在 signed hashing 基础上增加**按信息量加权**：只由虚词构成的 n-gram 会被降权，
    抑制"你我/的什/什么/吗"这类噪声，显著改善自然口语句子的检索排序。
    """

    backend_name = "hashing"

    def __init__(self, dim: int = 512, ngram_min: int = 1, ngram_max: int = 2,
                 function_word_weight: float = 0.15):
        self.dim = int(dim)
        self.ngram_min = int(ngram_min)
        self.ngram_max = int(ngram_max)
        # 纯虚词 n-gram 的权重（1.0 = 不降权；0 = 完全忽略）
        self.function_word_weight = float(function_word_weight)

    def _ngrams(self, text: str):
        t = _normalize(text)
        for n in range(self.ngram_min, self.ngram_max + 1):
            for i in range(len(t) - n + 1):
                yield t[i:i + n]

    def _weight(self, gram: str) -> float:
        """含任一实词的 n-gram 记满权重；纯虚词的降权。"""
        for ch in gram:
            if ch not in FUNCTION_CHARS:
                return 1.0
        return self.function_word_weight

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        count = 0
        for g in self._ngrams(text):
            digest = hashlib.md5(g.encode("utf-8")).digest()
            # 用 8 字节计算桶下标，再用 1 bit 决定正负号（signed hashing）
            idx = int.from_bytes(digest[:8], "little") % self.dim
            sign = 1.0 if digest[8] & 1 else -1.0
            v[idx] += sign * self._weight(g)
            count += 1
        if count == 0:
            return v
        norm = float(np.linalg.norm(v))
        if norm > 0:
            v = v / norm
        return v

    def embed_batch(self, texts) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)


class SentenceTransformerEmbedder:
    """本地 sentence-transformers 后端（可选依赖）。"""

    backend_name = "sentence_transformers"

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "未安装 sentence-transformers，请执行: pip install sentence-transformers"
            ) from e
        self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode([text])[0], dtype=np.float32)

    def embed_batch(self, texts) -> np.ndarray:
        return np.asarray(self.model.encode(list(texts)), dtype=np.float32)


class ApiEmbedder:
    """远程 OpenAI 兼容 /embeddings 后端。"""

    backend_name = "api"

    def __init__(self, base_url: str, model_name: str, api_key_env: str = "OPENAI_API_KEY", timeout: int = 60):
        import os
        self.url = (base_url or "").rstrip("/") + "/embeddings"
        self.model_name = model_name
        self.api_key = os.environ.get(api_key_env, "")
        self.timeout = int(timeout)

    def _request(self, texts):
        import os
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        payload = {"model": self.model_name, "input": list(texts)}
        req = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(f"无法连接嵌入接口 {self.url}: {e.reason}") from e
        items = body.get("data", [])
        items.sort(key=lambda x: x.get("index", 0))
        return [np.asarray(it["embedding"], dtype=np.float32) for it in items]

    def embed(self, text: str) -> np.ndarray:
        return self._request([text])[0]

    def embed_batch(self, texts) -> np.ndarray:
        return np.stack(self._request(texts))


# 已提示过降级的后端，避免同一进程内重复刷屏
_WARNED: set[str] = set()


def _make(backend: str, embedding_cfg: dict, model_cfg: dict):
    if backend == "hashing":
        return HashingEmbedder(
            dim=embedding_cfg.get("dim", 512),
            ngram_min=embedding_cfg.get("ngram_min", 1),
            ngram_max=embedding_cfg.get("ngram_max", 2),
            function_word_weight=embedding_cfg.get("function_word_weight", 0.15),
        )
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(embedding_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"))
    if backend == "api":
        return ApiEmbedder(
            base_url=model_cfg.get("base_url", ""),
            model_name=embedding_cfg.get("model_name", "BAAI/bge-small-zh-v1.5"),
            api_key_env=model_cfg.get("api_key_env", "OPENAI_API_KEY"),
            timeout=model_cfg.get("timeout", 60),
        )
    raise ValueError(f"未知的 embedding.backend: {backend!r}")


def build_embedder(embedding_cfg: dict, model_cfg: dict | None = None):
    """根据配置构建 embedder。

    团队首选 `sentence_transformers`（中文检索质量更高）。若目标环境尚未安装该依赖
    （或权重下载失败），本函数自动降级到 `fallback_backend`（默认 `hashing`）并打印
    警告，保证**任何环境下程序都能跑起来**，不会因缺依赖而崩溃。

    实际生效的后端可通过 `embedder.backend_name` 读取。
    """
    embedding_cfg = embedding_cfg or {}
    model_cfg = model_cfg or {}
    backend = embedding_cfg.get("backend", "hashing")
    fallback = embedding_cfg.get("fallback_backend", "hashing")

    try:
        return _make(backend, embedding_cfg, model_cfg)
    except Exception as e:  # ImportError / 权重下载失败 / 接口不可达 等
        if fallback and fallback != backend:
            if backend not in _WARNED:
                _WARNED.add(backend)  # 每进程只提示一次，避免刷屏
                print(
                    f"[WARN] 嵌入后端 '{backend}' 不可用（{type(e).__name__}: {e}）；"
                    f"已自动降级为 '{fallback}'。"
                    f"要启用高质量嵌入请执行: pip install sentence-transformers"
                )
            return _make(fallback, embedding_cfg, model_cfg)
        raise


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """两个已归一化向量的余弦相似度（等价于点积）。"""
    return float(np.dot(a, b))
