# -*- coding: utf-8 -*-
"""自测：RAG 向量记忆库的写入、检索、持久化与容量压缩。"""
import os

from tests import context


def run():
    ctx = context.build()
    memory = ctx["memory"]
    root = ctx["module_root"]
    memory.clear()

    results = []

    # 1. 写入
    memory.add("我特别喜欢打篮球，每周都去球场", metadata={"speaker": "user"})
    memory.add("我最近养了一只橘猫叫豆豆", metadata={"speaker": "user"})
    memory.add("工作上的项目下周就要上线了", metadata={"speaker": "user"})
    results.append({"name": "写入3条记忆", "ok": len(memory) == 3, "info": f"len={len(memory)}"})

    # 2. 检索排序：query="篮球" 应命中"打篮球"记忆
    hits = memory.search("篮球")
    top_text = hits[0][0]["text"] if hits else ""
    ok = bool(hits) and "篮球" in top_text
    results.append({"name": "检索排序(篮球命中)", "ok": ok,
                    "info": f"top1={top_text[:20]} score={hits[0][1]:.3f}" if hits else "无命中"})

    # 3. 相似度阈值过滤：无关 query 不应返回低相关记忆
    unrelated = memory.search("zzzz完全不相关的查询词", threshold=0.3)
    results.append({"name": "相似度阈值过滤无关查询", "ok": len(unrelated) == 0,
                    "info": f"命中数={len(unrelated)}"})

    # 4. build_context 输出可用于提示词
    ctx_text = memory.build_context("篮球", top_k=2)
    ok = "篮球" in ctx_text and "相关度" in ctx_text
    results.append({"name": "检索结果可格式化注入提示词", "ok": ok, "info": ctx_text[:40]})

    # 5. remember_turn 一轮返回两个 id
    u, a = memory.remember_turn("我今天心情不好", "我陪着你", emotion="悲伤")
    ok = bool(u) and bool(a)
    results.append({"name": "记录一轮对话返回2条记忆", "ok": ok, "info": f"{u[:8]}…/{a[:8]}…"})

    # 6. 持久化 round-trip
    path = os.path.join(root, "data", "test_memory_persist.json")
    memory.save(path)
    from src.memory_store import VectorMemoryStore
    m2 = VectorMemoryStore(ctx["embedder"], ctx["cfg"].section("memory"))
    loaded = m2.load(path)
    ok = loaded and len(m2) == len(memory)
    results.append({"name": "记忆持久化与加载(round-trip)", "ok": ok,
                    "info": f"原{len(memory)}条, 载入{len(m2)}条"})
    if os.path.isfile(path):
        os.remove(path)

    # 7. 分层压缩 tier-1：原始轮超限 -> 抽取用户陈述为独立"事实记忆"
    memory.clear()
    memory.max_turns = 5
    memory.max_facts = 6
    memory.summarize_every = 3
    for i in range(30):
        memory.remember_turn(f"第{i}轮用户说了很多话", f"第{i}轮我回应了", emotion="平静")
    n_turn = sum(1 for it in memory.items if it["metadata"].get("kind") == "turn")
    n_fact = sum(1 for it in memory.items if it["metadata"].get("kind") == "fact")
    n_summary = sum(1 for it in memory.items if it["metadata"].get("kind") == "summary")
    ok = n_turn <= 5 and n_fact > 0
    results.append({"name": "分层压缩tier1(原始轮→事实记忆)", "ok": ok,
                    "info": f"原始轮={n_turn}<=上限5, 事实={n_fact}, 摘要={n_summary}, 总={len(memory)}"})

    # 8. 分层压缩 tier-2：事实过多 -> 合并为摘要（max_facts 调小使其触发）
    ok2 = n_summary > 0 and n_fact <= memory.max_facts
    results.append({"name": "分层压缩tier2(事实→摘要)", "ok": ok2,
                    "info": f"事实={n_fact}<=上限{memory.max_facts}, 摘要={n_summary}"})

    # 9. 虚词降权回归（Iter 11）：自然口语句子查询下，相关记忆必须排在无关长文本之前。
    #    背景：未降权时，"你还记得我养的猫叫什么吗" 与无关的日常回声共享 6 个虚词单字，
    #    相似度(0.311)竟高于真正相关的"我养了一只叫豆豆的橘猫"(0.175)，导致排序错误。
    memory.clear()
    memory.add("我养了一只叫豆豆的橘猫", metadata={"mid": "cat", "speaker": "user"})
    memory.add("小暖：我在听，先陪着你。你说到「今天没什么特别的」，可以多和我讲讲吗？我不着急，你慢慢说。",
               metadata={"mid": "filler", "speaker": "assistant"})
    hits = memory.search("你还记得我养的猫叫什么吗？", top_k=2, threshold=0.0)
    top = hits[0][0]["metadata"]["mid"] if hits else None
    results.append({
        "name": "虚词降权：自然句查询排序正确",
        "ok": top == "cat",
        "info": f"top1={top}, 分数={hits[0][1]:.3f}" if hits else "无命中",
    })

    memory.clear()
    return {"suite": "RAG向量记忆库", "results": results}
