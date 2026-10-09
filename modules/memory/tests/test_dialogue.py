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

    # 5. RAG 长对话记忆：先说偏好，再问"记得吗"，应检索到先前记忆
    gen.generate("我特别喜欢看科幻电影，周末常去电影院", user_id="u2")
    out2 = gen.generate("你还记得我喜欢什么吗？", user_id="u2")
    mem_texts = " ".join(m["text"] for m in out2.get("retrieved_memories", []))
    ok = "科幻电影" in mem_texts
    results.append({"name": "RAG长对话记忆(跨轮检索命中)", "ok": ok,
                    "info": f"检索到 {len(out2.get('retrieved_memories', []))} 条, 命中={'科幻电影' in mem_texts}"})

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
