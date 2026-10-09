# -*- coding: utf-8 -*-
"""自测：对话生成（人设注入、RAG记忆联动、画像累积、防瞎编护栏）。"""
from tests import context


def run():
    ctx = context.build()
    gen = ctx["generator"]
    gen.memory.clear()
    gen.profiles.clear()
    gen.history.clear()

    results = []

    # 1. 基本生成
    out = gen.generate("最近工作好累啊，感觉快撑不住了", user_id="u1", emotion_label="疲惫")
    ok = bool(out.get("response", "").strip())
    results.append({"name": "对话生成返回非空回复", "ok": ok,
                    "info": out.get("response", "")[:40]})

    # 2. 人设注入
    sysp = out.get("system_prompt", "")
    ok = "小暖" in sysp and "温暖" in sysp
    results.append({"name": "系统提示词注入人设角色", "ok": ok, "info": "含「小暖」与温暖人设"})

    # 3. 上下文结构（画像/记忆/情绪三段齐全）
    ok = all(tag in sysp for tag in ("【用户画像】", "【相关记忆】", "【当前情绪】"))
    results.append({"name": "提示词含画像/记忆/情绪三段上下文", "ok": ok,
                    "info": "【用户画像】【相关记忆】【当前情绪】"})

    # 4. 防瞎编护栏（校验"每条护栏都在"，不绑定具体措辞，便于 Iter 20 精简后仍有效）
    from src.persona_prompts import SAFETY_GUARDRAILS
    missing = [g[:10] for g in SAFETY_GUARDRAILS if g not in sysp]
    results.append({"name": f"防瞎编护栏已注入({len(SAFETY_GUARDRAILS) - len(missing)}/{len(SAFETY_GUARDRAILS)})",
                    "ok": not missing,
                    "info": ("缺失：" + "；".join(missing)) if missing else "全部护栏条款均已注入"})

    # 4b. 提示词紧凑度回归（Iter 20 精简后不应再膨胀 —— CPU 预填充是主要成本）
    ok = len(sysp) <= 420
    results.append({"name": "提示词长度受控(<=420字)", "ok": ok,
                    "info": f"当前 {len(sysp)} 字（精简前 440 字）"})

    # 5. 跨轮记忆：先说偏好，再问"记得吗"。
    #    注意（Iter 24）：这里断言的是**结果**（该信息在本轮上下文中可见），
    #    而不是**机制**（必须由 RAG 召回）。因为最近几轮的原文本来就在历史消息里，
    #    此时 RAG 再去召回同一句会导致同一内容在提示词里出现两次（已去重）。
    #    历史窗口外的旧事才必须靠 RAG —— 那一条由下面的用例单独覆盖。
    gen.generate("我特别喜欢看科幻电影，周末常去电影院", user_id="u2")
    out2 = gen.generate("你还记得我喜欢什么吗？", user_id="u2")
    mem_texts = " ".join(m["text"] for m in out2.get("retrieved_memories", []))
    hist_texts = " ".join(h["content"] for h in gen.history.get("u2", []))
    visible = (mem_texts + hist_texts + out2.get("system_prompt", ""))
    ok = "科幻电影" in visible
    results.append({"name": "跨轮记忆（信息在本轮上下文中可见）", "ok": ok,
                    "info": f"RAG召回 {len(out2.get('retrieved_memories', []))} 条；"
                            f"信息经{'记忆' if '科幻电影' in mem_texts else '历史消息'}可见"})

    # 5b. 记忆去重（Iter 24）：同一内容不得在【相关记忆】与历史消息中同时出现
    overlaps = {m["text"].strip() for m in out2.get("retrieved_memories", [])} & \
               {h["content"].strip() for h in gen.history.get("u2", [])}
    results.append({"name": "记忆去重（不与历史消息重复注入）", "ok": not overlaps,
                    "info": f"重复 {len(overlaps)} 条" if overlaps else "无重复注入"})

    # 5c. 长程召回仍有效：把关键信息挤出历史窗口后，必须靠 RAG 想起来
    keep = gen.persona_cfg.get("max_history_turns")
    gen.persona_cfg["max_history_turns"] = 1
    gen.generate("我养了一只叫豆豆的橘猫", user_id="u4")
    for i in range(4):
        gen.generate(f"随便聊聊第{i}句", user_id="u4")
    out4 = gen.generate("你还记得我养的猫叫什么吗", user_id="u4")
    mem4 = " ".join(m["text"] for m in out4.get("retrieved_memories", []))
    ok = ("豆豆" in mem4) or ("橘猫" in mem4)
    gen.persona_cfg["max_history_turns"] = keep
    results.append({"name": "长程召回（历史窗口外仍能想起）", "ok": ok,
                    "info": f"RAG召回 {len(out4.get('retrieved_memories', []))} 条, "
                            f"命中={'豆豆' in mem4 or '橘猫' in mem4}"})

    # 6. 画像累积：明确透露性别后应写入用户画像
    gen.generate("我是男生，平时压力挺大的", user_id="u3")
    prof = gen.get_profile("u3")
    ok = prof.get("gender") == "男"
    results.append({"name": "画像跨轮累积(gender=男)", "ok": ok,
                    "info": f"gender={prof.get('gender')}"})

    # 7. 多轮历史维护
    ok = len(gen.history.get("u1", [])) >= 2
    results.append({"name": "多轮对话历史维护", "ok": ok,
                    "info": f"u1历史条数={len(gen.history.get('u1', []))}"})

    # 8. 四种人设均可构造提示词
    names = ["温暖倾听者", "理性朋友", "元气鼓励师", "治愈系陪伴"]
    ok = True
    for n in names:
        o = gen.generate("今天有点烦", user_id=f"p_{n}", persona_name=n)
        ok = ok and bool(o.get("response"))
        gen.memory.clear()
    results.append({"name": "四套人设均可正常生成", "ok": ok, "info": "、".join(names)})

    # 8b. 与成员1 的接口契约（见 docs/interface_for_member1.md）——
    #     成员1 的情感识别输出经 emotion_label 传入，本模块**不校验、不拦截**。
    gen.memory.clear()
    gen.profiles.clear()
    gen.history.clear()

    def emotion_line(out):
        return next((l for l in out["system_prompt"].splitlines() if "当前情绪" in l), "")

    # a) 成员1 未就绪：不传标签 -> 显示"（未知）"，功能不受影响
    oa = gen.generate("今天好累", user_id="ct_a")
    ok_a = emotion_line(oa) == "【当前情绪】（未知）" and bool(oa["response"])

    # b) 成员1 就绪：传已知标签 -> 原样注入
    ob = gen.generate("今天好累", user_id="ct_b", emotion_label="疲惫")
    ok_b = emotion_line(ob) == "【当前情绪】疲惫" and ob["emotion_label_known"] is True

    # c) 成员1 用了配置外的标签 -> 照常注入、不拦截，但记入漂移清单
    oc = gen.generate("好烦", user_id="ct_c", emotion_label="烦躁")
    ok_c = (emotion_line(oc) == "【当前情绪】烦躁"
            and bool(oc["response"])
            and oc["emotion_label_known"] is False
            and "烦躁" in gen.emotion_report()["unseen_labels"])

    for name, okk, info in [
        ("契约A: 不传情绪标签仍可正常对话", ok_a, emotion_line(oa)),
        ("契约B: 传入情绪标签原样注入提示词", ok_b, emotion_line(ob)),
        ("契约C: 未知标签不拦截且记入漂移清单", ok_c, emotion_line(oc)),
    ]:
        results.append({"name": name, "ok": okk, "info": info})

    # 9. 输出清洗：Qwen/ChatML 特殊标记与角色回显必须被剥离
    dirty = "<|im_start|>assistant\n小暖：小暖：我在这里陪着你。<|im_end|>"
    clean = gen._postprocess(dirty)
    ok = ("<|" not in clean and "im_end" not in clean
          and "im_start" not in clean and "assistant" not in clean.lower())
    results.append({"name": "输出清洗(剥离Qwen特殊标记/角色回显)", "ok": ok, "info": clean})

    # 10. 空回复兜底
    fallback = gen._postprocess("   ")
    ok = bool(fallback.strip())
    results.append({"name": "空回复兜底", "ok": ok, "info": fallback})

    gen.memory.clear()
    return {"suite": "对话生成与人设", "results": results}
