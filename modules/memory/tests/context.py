# -*- coding: utf-8 -*-
"""自测公共上下文：构建模块各组件，供各 test_*.py 与 run_self_test.py 复用。"""
import json
import os
import sys

# 保证可 import src 包（模块根目录加入 sys.path）
_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)


def module_root() -> str:
    return _MODULE_ROOT


def load_dimensions():
    from src.config_loader import load_config, Config
    from src import yaml_light
    cfg = load_config(_MODULE_ROOT)
    dims_path = Config(cfg, _MODULE_ROOT).resolve(cfg["profile"]["dimensions_path"])
    with open(dims_path, "r", encoding="utf-8") as f:
        data = yaml_light.loads(f.read())
    return data["dimensions"]


def build(force_mock: bool = True):
    """构建并返回自测所需组件 dict。

    force_mock=True（默认）：**自测是离线用例，强制使用 mock 后端**。
    为什么必须这样：本模块放进团队仓库 `modules/memory/` 后，会读到团队的
    `config.yaml`（含 `model_path`）从而自动切到 `transformers_local`；
    若不自测时强制 mock，任何未装 torch/未下权重的环境都会跑不了自测——
    而自测恰恰是"没模型也能验证逻辑"的保障。
    真实模型链路由 `check_model.py` 单独验证。
    """
    from src.config_loader import load_config, Config
    from src.llm_client import LLMClient
    from src.embedding import build_embedder
    from src.memory_store import VectorMemoryStore
    from src.profile_extractor import ProfileExtractor

    cfg = Config(load_config(_MODULE_ROOT), _MODULE_ROOT)
    if force_mock and cfg.section("model").get("backend") != "mock":
        cfg.data["model"]["backend"] = "mock"

    llm = LLMClient(cfg.section("model"))
    embedder = build_embedder(cfg.section("embedding"), cfg.section("model"))

    memory_cfg = dict(cfg.section("memory"))
    memory_cfg["persist_path"] = os.path.join(_MODULE_ROOT, "data", "test_memory_store.json")
    memory = VectorMemoryStore(embedder, memory_cfg)

    dimensions = load_dimensions()
    extractor = ProfileExtractor(dimensions, cfg.section("profile"), llm_client=llm)

    dim_names = {d["id"]: d["name"] for d in dimensions}

    from src.dialogue_generator import DialogueGenerator
    generator = DialogueGenerator(cfg, llm, memory, extractor, dimension_names=dim_names)

    return {
        "cfg": cfg,
        "llm": llm,
        "embedder": embedder,
        "memory": memory,
        "extractor": extractor,
        "generator": generator,
        "dimensions": dimensions,
        "module_root": _MODULE_ROOT,
    }


def load_profile_samples():
    path = os.path.join(_MODULE_ROOT, "data", "profile_samples.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
