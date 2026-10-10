# -*- coding: utf-8 -*-
"""自测：持久化（记忆库 / 画像 / 历史 的保存与恢复）。

这几条代码路径此前**零测试覆盖**（`memory.save/load`、`generator.save_profiles/
load_profiles`、`pipeline.save/load`）。若它们坏了——比如路径解析错、字段丢、
恢复后检索失效——平时不会有人发现，等到部署重启才暴雷。

覆盖点：
    1. 记忆库 round-trip 后条目数、文本、向量检索能力均保持；
    2. 画像与对话历史 round-trip 后内容一致；
    3. pipeline 整体 save → 新建实例 load → 跨进程语义上可继续对话；
    4. 文件不存在时 load 返回 False 而不抛异常。
"""
import json
import os

from tests import context


def run():
    ctx = context.build()
    root = ctx["module_root"]
    tmp_dir = os.path.join(root, "data")
    os.makedirs(tmp_dir, exist_ok=True)

    mem_path = os.path.join(tmp_dir, "test_persist_memory.json")
    prof_path = os.path.join(tmp_dir, "test_persist_profiles.json")
    results = []

    # ---------- 1. 记忆库 round-trip ----------
    mem = ctx["memory"]
    mem.clear()
    mem.add("我特别喜欢打篮球，每周都去球场", metadata={"speaker": "user"})
    mem.add("我最近养了一只橘猫叫豆豆", metadata={"speaker": "user"})
    before_hits = [it["text"] for it, _ in mem.search("篮球")]

    mem.save(mem_path)
    from src.memory_store import VectorMemoryStore
    mem2 = VectorMemoryStore(ctx["embedder"], ctx["cfg"].section("memory"))
    ok_load = mem2.load(mem_path)
    after_hits = [it["text"] for it, _ in mem2.search("篮球")]

    ok = ok_load and len(mem2) == len(mem) and before_hits == after_hits and after_hits
    results.append({
        "name": "记忆库 round-trip（含检索能力保持）",
        "ok": bool(ok),
        "info": f"原{len(mem)}条→载入{len(mem2)}条, 检索 top1 一致={before_hits == after_hits}",
    })

    # ---------- 2. 画像与历史 round-trip ----------
    gen = ctx["generator"]
    gen.memory.clear()
    gen.profiles.clear()
    gen.history.clear()
    gen.generate("我是男生，在北京做程序员，最近压力有点大", user_id="p1", emotion_label="焦虑")
    prof_before = gen.get_profile("p1")
    hist_before = list(gen.history.get("p1", []))

    gen.profiles_path = prof_path
    gen.save_profiles()
    gen.profiles.clear()
    gen.history.clear()
    ok_load2 = gen.load_profiles()
    prof_after = gen.get_profile("p1")
    hist_after = gen.history.get("p1", [])

    ok2 = ok_load2 and prof_before == prof_after and len(hist_before) == len(hist_after)
    results.append({
        "name": "画像与历史 round-trip",
        "ok": ok2,
        "info": f"画像{len(prof_before)}项一致={prof_before == prof_after}, "
                f"历史{len(hist_before)}→{len(hist_after)}条",
    })

    # ---------- 3. pipeline 整体 save → 新实例 load ----------
    from src.pipeline import CompanionPipeline
    pipe = CompanionPipeline.from_config(module_dir=root, force_mock=True)
    pipe.reset()
    pipe.memory.persist_path = mem_path
    pipe.generator.profiles_path = prof_path
    pipe.chat("我养了一只叫团子的布偶猫", user_id="pw", emotion_label="平静")
    prof_pw = pipe.get_profile("pw")
    pipe.save()

    pipe2 = CompanionPipeline.from_config(module_dir=root, force_mock=True)
    pipe2.memory.persist_path = mem_path
    pipe2.generator.profiles_path = prof_path
    loaded = pipe2.load()

    hits2 = [h["text"] for h in pipe2.retrieve("猫", top_k=5)]
    ok3 = (loaded["memory"] and loaded["profiles"]
           and pipe2.get_profile("pw") == prof_pw
           and any("团子" in h or "布偶" in h for h in hits2))
    results.append({
        "name": "pipeline.save → 新实例 load 后可继续检索",
        "ok": ok3,
        "info": f"memory={loaded['memory']}, profiles={loaded['profiles']}, "
                f"画像一致={pipe2.get_profile('pw') == prof_pw}",
    })

    # ---------- 4. 文件不存在时 load 应返回 False，不抛异常 ----------
    missing = os.path.join(tmp_dir, "test_persist_not_exist.json")
    if os.path.isfile(missing):
        os.remove(missing)
    try:
        r1 = mem2.load(missing)
        r2 = gen.load_profiles(missing)
        ok4 = (r1 is False) and (r2 is False)
        info4 = f"memory.load={r1}, profiles.load={r2}"
    except Exception as e:  # noqa: BLE001
        ok4, info4 = False, f"抛异常：{type(e).__name__}: {e}"
    results.append({"name": "文件不存在时 load 返回 False 不抛异常", "ok": ok4, "info": info4})

    # ---------- 5. 落盘文件结构可读（便于人工排查/成员3审阅） ----------
    with open(mem_path, "r", encoding="utf-8") as f:
        dumped = json.load(f)
    ok5 = isinstance(dumped.get("items"), list) and "embedding" in dumped["items"][0]
    results.append({"name": "落盘 JSON 结构完整", "ok": ok5,
                    "info": f"items={len(dumped.get('items', []))}, 字段={sorted(dumped['items'][0].keys())}"})

    # 清理临时文件
    for p in (mem_path, prof_path):
        if os.path.isfile(p):
            os.remove(p)
    pipe.reset()
    pipe2.reset()
    return {"suite": "持久化", "results": results}
