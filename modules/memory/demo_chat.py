# -*- coding: utf-8 -*-
"""成员2模块端到端演示：多轮情感陪伴对话（含 RAG 长对话记忆）。

用法：
    python demo_chat.py

演示重点：
    1) 多轮对话中画像如何逐步累积；
    2) 第 1 轮说过的信息，在第 5 轮能被检索出来并影响回复（解决"聊天失忆"）；
    3) 情感识别标签（模拟成员1的输出）如何注入对话生成。
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from src.config_loader import load_config, Config          # noqa: E402
from src.llm_client import LLMClient                        # noqa: E402
from src.embedding import build_embedder                    # noqa: E402
from src.memory_store import VectorMemoryStore              # noqa: E402
from src.profile_extractor import ProfileExtractor          # noqa: E402
from src.dialogue_generator import DialogueGenerator        # noqa: E402
from src import yaml_light                                  # noqa: E402


def build():
    cfg = Config(load_config(_HERE), _HERE)
    llm = LLMClient(cfg.section("model"))
    embedder = build_embedder(cfg.section("embedding"), cfg.section("model"))

    memory_cfg = dict(cfg.section("memory"))
    memory_cfg["persist_path"] = os.path.join(_HERE, "data", "demo_memory_store.json")
    memory = VectorMemoryStore(embedder, memory_cfg)

    with open(cfg.resolve(cfg["profile"]["dimensions_path"]), "r", encoding="utf-8") as f:
        dimensions = yaml_light.loads(f.read())["dimensions"]
    extractor = ProfileExtractor(dimensions, cfg.section("profile"), llm_client=llm)
    dim_names = {d["id"]: d["name"] for d in dimensions}

    gen = DialogueGenerator(cfg, llm, memory, extractor, dimension_names=dim_names)
    return cfg, gen, dim_names


# (用户输入, 模拟成员1情感识别模块输出的情绪标签)
SCRIPT = [
    ("你好呀，我是男生，最近在北京做程序员，压力有点大。", "焦虑"),
    ("我平时喜欢看科幻电影，周末常去电影院。", "平静"),
    ("项目下周要上线，天天加班，昨晚还失眠了。", "疲惫"),
    ("有时候觉得挺孤单的，一个人吃饭一个人回家。", "孤独"),
    ("你还记得我喜欢什么吗？", "平静"),
]


def main():
    cfg, gen, dim_names = build()
    print("=" * 70)
    print("成员2 模块演示：用户画像 + RAG记忆 + 对话生成")
    print(f"模型后端={cfg.get('model.backend')}  向量后端={cfg.get('embedding.backend')}")
    print("=" * 70)

    for i, (user, emotion) in enumerate(SCRIPT, 1):
        out = gen.generate(user, user_id="demo", emotion_label=emotion)
        print(f"\n--- 第 {i} 轮 ---")
        print(f"[用户] {user}")
        print(f"[情感识别] {emotion}")
        hits = out["retrieved_memories"]
        print(f"[RAG检索] 命中 {len(hits)} 条" + (f"，top1: {hits[0]['text'][:24]}…" if hits else ""))
        print(f"[回复] {out['response']}")

    print("\n" + "=" * 70)
    prof = gen.get_profile("demo")
    known = {dim_names.get(k, k): v for k, v in prof.items() if v and v != "未知"}
    print("最终累积用户画像（仅显示已知项）：")
    for k, v in known.items():
        print(f"  - {k}: {v}")
    print(f"\n记忆库规模：{len(gen.memory)} 条")


if __name__ == "__main__":
    main()
