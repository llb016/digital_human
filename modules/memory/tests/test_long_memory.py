# -*- coding: utf-8 -*-
"""自测：长对话记忆（成员2 任务2 的核心验收证据）。

目标：证明「聊天失忆」被真正解决，而不只是 2 轮内的巧合。
场景：前 3 轮植入用户事实 -> 中间 24 轮日常闲聊（触发记忆容量压缩）
      -> 最后追问早期事实，检查能否召回。
"""
from tests import context

FACTS = {
    "cat": "我养了一只叫豆豆的橘猫",
    "job": "我是做前端开发的",
    "singer": "我最喜欢周杰伦的歌",
}


def run():
    ctx = context.build()
    gen = ctx["generator"]
    gen.memory.clear()
    gen.profiles.clear()
    gen.history.clear()

    mem = gen.memory
    # 故意把容量调小，强制触发"压缩为摘要"逻辑，验证压缩后仍不失忆
    mem.max_turns = 8
    mem.summarize_every = 4

    results = []

    # 1) 前 3 轮植入事实
    gen.generate(FACTS["cat"], user_id="L")
    gen.generate(FACTS["job"], user_id="L")
    gen.generate(FACTS["singer"], user_id="L")

    # 2) 中间 24 轮无关闲聊
    for i in range(24):
        gen.generate(f"今天没什么特别的，就是普通的第{i + 4}天。", user_id="L")

    n_turn = sum(1 for it in mem.items if it["metadata"].get("kind") == "turn")
    n_fact = sum(1 for it in mem.items if it["metadata"].get("kind") == "fact")
    results.append({
        "name": "长对话已触发分层压缩(原始轮→事实记忆)",
        "ok": n_fact > 0 and n_turn <= mem.max_turns,
        "info": f"原始轮={n_turn}(<=上限{mem.max_turns}), 事实={n_fact}, 总计={len(mem)}",
    })

    # 3) 追问早期事实：应能召回
    checks = [
        ("你还记得我养的猫叫什么吗？", ["豆豆", "橘猫"], "猫名"),
        ("我是做什么工作的来着？", ["前端"], "职业"),
        ("我最喜欢哪个歌手的歌？", ["周杰伦"], "歌手"),
    ]
    for query, needles, label in checks:
        out = gen.generate(query, user_id="L")
        texts = " ".join(m["text"] for m in out["retrieved_memories"])
        hit = any(n in texts for n in needles)
        results.append({
            "name": f"长对话后召回早期事实({label})",
            "ok": hit,
            "info": f"命中={hit}, top1={(texts[:38] + '…') if texts else '无召回'}",
        })

    # 4) 记忆容量受控（不会随轮数无限膨胀）
    n_turn2 = sum(1 for it in mem.items if it["metadata"].get("kind") == "turn")
    results.append({
        "name": "记忆库规模受控(不无限膨胀)",
        "ok": n_turn2 <= mem.max_turns,
        "info": f"原始轮={n_turn2} <= 上限{mem.max_turns}",
    })

    gen.memory.clear()
    return {"suite": "长对话记忆", "results": results}
