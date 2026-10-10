# -*- coding: utf-8 -*-
"""模块统一对外接口（Facade）。

给成员1（集成）、成员3（自动评测）提供**唯一入口**，避免直接耦合内部实现。

典型用法：
    from src.pipeline import CompanionPipeline
    pipe = CompanionPipeline.from_config()
    out = pipe.chat("今天好累啊", user_id="u1", emotion_label="疲惫")
    print(out["response"])
"""
from __future__ import annotations

import os

from .config_loader import load_config, Config
from .llm_client import LLMClient
from .embedding import build_embedder
from .memory_store import VectorMemoryStore
from .profile_extractor import ProfileExtractor
from .dialogue_generator import DialogueGenerator
from . import yaml_light

_DEFAULT_MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CompanionPipeline:
    def __init__(self, config: Config, llm, embedder, memory, extractor, generator, dimension_names):
        self.config = config
        self.llm = llm
        self.embedder = embedder
        self.memory = memory
        self.extractor = extractor
        self.generator = generator
        self.dimension_names = dimension_names

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, module_dir: str | None = None, config_path: str | None = None,
                    force_mock: bool = False):
        """从配置构建整条流水线。

        force_mock=True 时强制使用离线 mock 后端。**自测与离线演示必须用它**：
        本模块放进团队仓库 `modules/memory/` 后会读到团队 config.yaml（含 `model_path`）
        并自动切到 `transformers_local`；若不强制 mock，任何未装 torch / 未下权重的
        环境都跑不了自测——而自测正是"没有模型也能验证逻辑"的保障。
        真实模型链路由 `check_model.py` 单独验证。
        """
        module_dir = module_dir or _DEFAULT_MODULE_DIR
        cfg = Config(load_config(module_dir), module_dir)
        if force_mock:
            cfg.data["model"]["backend"] = "mock"

        llm = LLMClient(cfg.section("model"))
        embedder = build_embedder(cfg.section("embedding"), cfg.section("model"))

        memory_cfg = dict(cfg.section("memory"))
        memory_cfg["persist_path"] = cfg.resolve(memory_cfg.get("persist_path", "data/memory_store.json"))
        memory = VectorMemoryStore(embedder, memory_cfg)

        dims_path = cfg.resolve(cfg["profile"]["dimensions_path"])
        with open(dims_path, "r", encoding="utf-8") as f:
            dimensions = yaml_light.loads(f.read())["dimensions"]

        extractor = ProfileExtractor(dimensions, cfg.section("profile"), llm_client=llm)
        dim_names = {d["id"]: d["name"] for d in dimensions}
        generator = DialogueGenerator(cfg, llm, memory, extractor, dimension_names=dim_names)

        return cls(cfg, llm, embedder, memory, extractor, generator, dim_names)

    # ------------------------------------------------------------------
    # 成员1 / 成员3 调用入口
    # ------------------------------------------------------------------
    def chat(self, user_input: str, user_id: str = "default", emotion_label: str = "",
             persona_name: str | None = None) -> dict:
        """一次完整对话：返回 response 及检索/画像信息。

        成员1 情感识别模块只需把情绪标签通过 emotion_label 传入。
        """
        return self.generator.generate(
            user_input, user_id=user_id, emotion_label=emotion_label, persona_name=persona_name
        )

    def extract_profile(self, text: str, flat: bool = True) -> dict:
        """抽取用户画像。flat=True 返回 {dim_id: value}，否则含 confidence。"""
        result = self.extractor.extract(text)
        return self.extractor.to_flat(result) if flat else result

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        """检索相关记忆，返回 [{"text":..., "score":..., "kind":...}]。"""
        return [
            {"text": it["text"], "score": round(s, 4), "kind": it["metadata"].get("kind")}
            for it, s in self.memory.search(query, top_k=top_k)
        ]

    def get_profile(self, user_id: str = "default") -> dict:
        return self.generator.get_profile(user_id)

    def reset(self, user_id: str | None = None):
        """清空指定用户的画像与历史（不传则清空全部）。"""
        if user_id is None:
            self.generator.profiles.clear()
            self.generator.history.clear()
            self.memory.clear()
        else:
            self.generator.profiles.pop(user_id, None)
            self.generator.history.pop(user_id, None)

    def save(self):
        """持久化记忆库与画像。"""
        self.memory.save()
        self.generator.save_profiles()

    def load(self):
        """加载已持久化的记忆库与画像。"""
        ok_mem = self.memory.load()
        ok_prof = self.generator.load_profiles()
        return {"memory": ok_mem, "profiles": ok_prof}
